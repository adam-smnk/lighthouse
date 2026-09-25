# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate


def run(name: str, payload_str: str, cores: int = 2):
    print(f"Test: {name}", flush=True)
    with TargetInfo.override(arch="x86_64", features=["avx512f"], core_count=cores):
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            payload = ir.Module.parse(payload_str)
            with schedule_boilerplate() as (sched, named_seq):
                ops = lh_transform.match_op(
                    named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
                )
                gemms = transform_ext.filter_contraction_ops(ops)
                transform_ext.assign_panel_tile_sizes(gemms)
                transform.yield_()
            sched.body.operations[0].apply(payload.operation)
            print(payload)


# Packed 64x64x32 GEMM (32x32 blocks) -> packed row max -> packed exp -> unpack
# -> plain row sum -> plain division, i.e. a GEMM + softmax epilogue.
GEMM_SOFTMAX = """
#a = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#b = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#c = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#rows4 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#rows2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%A: tensor<2x1x32x32xf32>, %B: tensor<2x1x32x32xf32>,
      %C: tensor<2x2x32x32xf32>, %m: tensor<2x32xf32>, %s: tensor<64xf32>,
      %o: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %mm = linalg.generic {indexing_maps = [#a, #b, #c], iterator_types =
        ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
        ins(%A, %B : tensor<2x1x32x32xf32>, tensor<2x1x32x32xf32>)
        outs(%C : tensor<2x2x32x32xf32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %p = arith.mulf %x, %y : f32
      %r = arith.addf %acc, %p : f32
      linalg.yield %r : f32
    } -> tensor<2x2x32x32xf32>
    %max = linalg.generic {indexing_maps = [#id4, #rows4],
        iterator_types = ["parallel", "reduction", "parallel", "reduction"]}
        ins(%mm : tensor<2x2x32x32xf32>) outs(%m : tensor<2x32xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.maximumf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<2x32xf32>
    %exp = linalg.generic {indexing_maps = [#id4, #rows4, #id4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%mm, %max : tensor<2x2x32x32xf32>, tensor<2x32xf32>)
        outs(%C : tensor<2x2x32x32xf32>) {
    ^bb0(%x: f32, %mx: f32, %out: f32):
      %d = arith.subf %x, %mx : f32
      %r = math.exp %d : f32
      linalg.yield %r : f32
    } -> tensor<2x2x32x32xf32>
    %u = linalg.unpack %exp inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %o
        : tensor<2x2x32x32xf32> -> tensor<64x64xf32>
    %sum = linalg.generic {indexing_maps = [#id2, #rows2],
        iterator_types = ["parallel", "reduction"]}
        ins(%u : tensor<64x64xf32>) outs(%s : tensor<64xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.addf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<64xf32>
    %div = linalg.generic {indexing_maps = [#id2, #rows2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%u, %sum : tensor<64x64xf32>, tensor<64xf32>)
        outs(%o : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %r = arith.divf %x, %y : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    return %div : tensor<64x64xf32>
  }
}
"""

# One 32-row block per panel: the terminal division is the only annotated root
# (full columns), every other op of the chain is fused into it.
# CHECK-LABEL: Test: gemm_softmax_panel
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: arith.mulf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: arith.maximumf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: math.exp
# CHECK: linalg.unpack
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: arith.addf
# CHECK: linalg.generic
# CHECK-NOT: transform_ext.fusion_panel
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 0>
# CHECK: arith.divf
run("gemm_softmax_panel", GEMM_SOFTMAX)

# Two panels cannot keep 16 cores busy: no panel is formed.
# CHECK-LABEL: Test: gemm_softmax_low_parallelism
# CHECK-NOT: transform_ext.fusion_panel
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
run("gemm_softmax_low_parallelism", GEMM_SOFTMAX, cores=16)
