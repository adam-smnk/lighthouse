import torch
from torch.profiler import profile, ProfilerActivity, record_function

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import bufferization
from mlir.dialects.transform import gpu
from mlir.dialects.transform import interpreter
from mlir.dialects.transform import memref
from mlir.dialects.transform import vector
from mlir.dialects.transform import loop
from mlir.execution_engine import ExecutionEngine

from lighthouse import utils as lh_utils


def create_kernel(ctx: ir.Context) -> ir.Module:
    """
    Create an MLIR module containing a function to execute.

    Args:
        ctx: MLIR context.
    """
    with ctx:
        module = ir.Module.parse(
            r"""
    // Compute matrix multiplication in form of: C = A * B
    func.func @matmul(%a: memref<1024x1024xf32>, %b: memref<1024x1024xf32>, %out: memref<1024x1024xf32>) {
        %tA = bufferization.to_tensor %a restrict : memref<1024x1024xf32> to tensor<1024x1024xf32>
        %tB = bufferization.to_tensor %b restrict : memref<1024x1024xf32> to tensor<1024x1024xf32>
        %tOut = bufferization.to_tensor %out restrict writable : memref<1024x1024xf32> to tensor<1024x1024xf32>

        %c0 = arith.constant 0.0 : f32
        %c = linalg.fill ins(%c0 : f32) outs(%tOut : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
        %res = linalg.matmul ins(%tA, %tB : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                             outs(%c : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

        bufferization.materialize_in_destination %res in restrict writable %out
        : (tensor<1024x1024xf32>, memref<1024x1024xf32>) -> ()
        return
    }
"""
        )
    return module


def create_schedule(ctx: ir.Context) -> ir.Module:
    """
    Create an MLIR module containing transformation schedule.

    Args:
        ctx: MLIR context.
    """
    with ctx, ir.Location.unknown(context=ctx):
        # Create transform module.
        schedule = ir.Module.create()
        schedule.operation.attributes["transform.with_named_sequence"] = (
            ir.UnitAttr.get()
        )

        # Create entry point transformation sequence.
        with ir.InsertionPoint(schedule.body):
            named_seq = transform.NamedSequenceOp(
                "__transform_main",
                [transform.any_op_t()],
                [],
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
            )

        # Create the schedule.
        with ir.InsertionPoint(named_seq.body):
            anytype = transform.any_op_t()

            # Find the kernel's function op.
            # Use C interface wrappers - required to make function executable after jitting.
            func = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["func.func"]
            )
            func = transform.apply_registered_pass(
                anytype, func, "llvm-request-c-wrappers"
            )

            # Tile matmul C matrix for better cache accesses and potential parallelization.
            # Fuse matmul with its zero initializer to have a small temporary buffer for
            # accumulation.
            mm = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["linalg.matmul"]
            ).result
            tiled_mm = structured.FuseOp(
                mm, tile_sizes=[64, 64], apply_cleanup=True
            ).results[0]

            # Sub-tile zero initialization to allow for simple vectorization.
            # Ensure that tiles are contiguous in memory.
            tiled_fill = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["linalg.fill"]
            ).result
            reg_fill = structured.TileUsingForOp(tiled_fill, sizes=[1, 64]).results[0]

            # Sub-tile C matrix tile into chunks that can fit into registers.
            # Unit-step through the reduction dimension.
            #
            # Register sub-tiling ensures that whole accumulation (the hot loop)
            # can happen without extra memory accesses.
            # Shapes must be chosen such that computation fits into (multiple) vector
            # operations and each iteration's data can remain within registers
            # without spills.
            reg_mm = structured.TileUsingForOp(tiled_mm, sizes=[8, 32, 1]).results[0]

            # Vectorize operations.
            structured.structured_vectorize(reg_mm, [], create_named_contraction=True)
            structured.structured_vectorize(reg_fill, [])

            # Loop hoisting.
            # Primarily needed to hoist C matrix handling and avoid extra memory
            # accesses within the innermost accumulation loop.
            all_loops = structured.MatchOp(
                anytype,
                named_seq.bodyTarget,
                interface=structured.MatchInterfaceEnum.LoopLikeInterface,
            ).results
            transform.apply_licm(all_loops)
            loop.loop_hoist_loop_invariant_subsets(all_loops)

            # Find the kernel's module op.
            mod = transform.get_parent_op(
                anytype, func, op_name="builtin.module", deduplicate=True
            )
            with ir.InsertionPoint(transform.ApplyPatternsOp(mod).patterns):
                # Unroll GEMM to create small contraction like outer products.
                #
                # A simple F32 GEMM can be computed using FMA operations.
                # This a stepping stone toward that goal.
                gpu.apply_patterns_gpu_unroll_vectors_subgroup_mma(m=1, n=32, k=1)
                # Flatten vector transfer to allow for direct lowering to LLVM
                # loads and stores without need for extra loops when vector
                # transfers are multidimensional.
                vector.apply_patterns_vector_cast_away_vector_leading_one_dim()
                transform.apply_patterns_canonicalization()

            # Lower the small contractions into actual outer product ops which
            # can be then lowered directly to broadcast+FMA instructions.
            with ir.InsertionPoint(transform.ApplyPatternsOp(mod).patterns):
                vector.apply_patterns_vector_lower_contraction(
                    lowering_strategy=vector.VectorContractLowering.OuterProduct
                )
                vector.apply_patterns_vector_lower_outerproduct()

            # Cleanup.
            transform.apply_cse(mod)
            with ir.InsertionPoint(transform.ApplyPatternsOp(mod).patterns):
                transform.apply_patterns_canonicalization()

            # Bufferize
            bufferization.bufferization_eliminate_empty_tensors(mod)
            mod = bufferization.bufferization_one_shot_bufferize(anytype, mod)
            mod = transform.apply_registered_pass(
                anytype, mod, "buffer-deallocation-pipeline"
            )

            # Cleanup.
            transform.apply_cse(mod)
            with ir.InsertionPoint(transform.ApplyPatternsOp(mod).patterns):
                transform.apply_patterns_canonicalization()
                memref.apply_patterns_memref_expand_strided_metadata()

            # Lower to LLVM.
            mod = transform.apply_registered_pass(anytype, mod, "convert-vector-to-scf")
            mod = transform.apply_registered_pass(anytype, mod, "lower-affine")
            mod = transform.apply_registered_pass(anytype, mod, "convert-scf-to-cf")
            mod = transform.apply_registered_pass(
                anytype, mod, "convert-vector-to-llvm"
            )
            mod = transform.apply_registered_pass(anytype, mod, "convert-to-llvm")
            mod = transform.apply_registered_pass(
                anytype, mod, "reconcile-unrealized-casts"
            )

            # Cleanup.
            transform.apply_cse(mod)
            with ir.InsertionPoint(transform.ApplyPatternsOp(mod).patterns):
                transform.apply_patterns_canonicalization()

            # Terminate the schedule.
            transform.YieldOp()
    return schedule


def time_cpu_execution(fn, args, n_warmup=2, n_repeat=5):
    for _ in range(n_warmup):
        fn(*args)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(n_repeat):
            with record_function("profiled_fn"):
                fn(*args)

    events = [e for e in prof.events() if e.name.startswith("profiled_fn")]

    times = torch.tensor([e.cpu_time for e in events], dtype=torch.float)

    # Trim extremes if there are enough measurements.
    if len(times) > 10:
        times = torch.sort(times).values[1:-1]

    return torch.mean(times).item()


def main():
    torch.set_num_threads(1)

    # Baseline computation.
    a = torch.randn(1024, 1024, dtype=torch.float32)
    b = torch.randn(1024, 1024, dtype=torch.float32)
    out_ref = torch.matmul(a, b)

    # MLIR kernel.
    ctx = ir.Context()
    kernel = create_kernel(ctx)

    # Optimize and lower to LLVM.
    schedule = create_schedule(ctx)
    interpreter.apply_named_sequence(
        payload_root=kernel,
        transform_root=schedule.body.operations[0],
        transform_module=schedule,
    )

    # JIT and execute.
    eng = ExecutionEngine(kernel, opt_level=3)
    eng.initialize()
    func = eng.lookup("matmul")

    out = torch.empty_like(out_ref)
    args = lh_utils.torch_to_packed_args([a, b, out])
    func(args)

    # Check numerical correctness.
    torch.testing.assert_close(out, out_ref, rtol=0.01, atol=0.01)

    # Measure execution time.
    n_warmup = 20
    n_repeat = 300
    meas = time_cpu_execution(
        torch.matmul, [a, b], n_warmup=n_warmup, n_repeat=n_repeat
    )
    print(f"torch: {meas}us")
    meas = time_cpu_execution(func, [args], n_warmup=n_warmup, n_repeat=n_repeat)
    print(f"mlir: {meas}us")


if __name__ == "__main__":
    main()
