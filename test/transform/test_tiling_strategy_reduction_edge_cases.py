# RUN: %PYTHON %s | FileCheck %s

# Edge cases of the reduction tiling heuristics: element types, SIMD widths,
# dynamic and non-divisible extents and non-splittable bodies.

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform.transform_ext.utils.tiling.strategy_cache import (
    ReductionCacheTiling,
)
from lighthouse.dialects.transform.transform_ext.utils.tiling.strategy_register_reduction import (
    ReductionRegisterTiling,
)
from lighthouse.execution.target import TargetInfo

ROW_REDUCE = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%a: tensor<ROWSxCOLSxTYPE>, %o: tensor<ROWSxTYPE>) -> tensor<ROWSxTYPE> {
    %r = linalg.generic {indexing_maps = [#id, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%a : tensor<ROWSxCOLSxTYPE>) outs(%o : tensor<ROWSxTYPE>) {
    ^bb0(%in: TYPE, %out: TYPE):
      BODY
    } -> tensor<ROWSxTYPE>
    return %r : tensor<ROWSxTYPE>
  }
}
"""

ADD = "%s = arith.addf %in, %out : TYPE\n      linalg.yield %s : TYPE"


def run(
    name: str,
    rows: str = "64",
    cols: str = "4096",
    elem: str = "f32",
    body: str = ADD,
    features: tuple[str, ...] = ("avx512f",),
    cores: int = 16,
):
    payload = (
        ROW_REDUCE.replace("BODY", body)
        .replace("ROWS", rows)
        .replace("COLS", cols)
        .replace("TYPE", elem)
    )
    with TargetInfo.override(
        arch="x86_64", features=list(features), core_count=cores, l2_cache_bytes=1 << 20
    ):
        target = TargetInfo.host()
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            module = ir.Module.parse(payload)
            func = module.body.operations[0]
            op = func.regions[0].blocks[0].operations[0]
            rt = ReductionRegisterTiling
            print(
                f"{name}: split={rt.split_factor(op, target)}"
                f" parallel={rt.parallel_tiles(op, target)}"
                f" reduction={rt.reduction_tiles(op, target)}"
                f" unroll={rt.unroll_tiles(op, target)}"
                f" cache={ReductionCacheTiling.choose_tile_shape(op, target)}"
            )


# Split factor = lanes x 8 chains of the compute width: bf16 is computed as f32.
# The 64 rows are split into single-row cache panels to feed 16 cores 4 times.
# CHECK: f32: split=128 parallel=[1, 0] reduction=[0, 128] unroll=[1, 0] cache=[1, 0]
run("f32")
# CHECK: bf16: split=128
run("bf16", elem="bf16")
# CHECK: f64: split=64 parallel=[1, 0] reduction=[0, 64]
run("f64", elem="f64")
# CHECK: i8: split=512
run("i8", elem="i8", body=ADD.replace("addf", "addi"))
# CHECK: sse: split=32 parallel=[1, 0] reduction=[0, 32]
run("sse", features=("sse4_2",))

# Short rows need no split and are reduced whole; more rows share the chains.
# CHECK: short_row: split=None parallel=[2, 0] reduction=None unroll=[1, 0]
run("short_row", cols="64")

# Fewer cores: 8-row panels (8 x 4096 x 4 B x 3 live tensors = 384 KiB) fit
# half of the 1 MiB L2.
# CHECK: few_cores: split=128 {{.*}} cache=[8, 0]
run("few_cores", rows="256", cores=4)

# No multiple of the vector width divides 4095: neither split nor tiled.
# CHECK: non_divisible: split=None parallel=[1, 0] reduction=None unroll=[1, 0]
run("non_divisible", cols="4095")

# Dynamic reduced extent: only the unit unroll shape is known.
# CHECK: dynamic: split=None parallel=None reduction=None unroll=[1, 0] cache=None
run("dynamic", cols="?")

# Non-associative combiner (acc - x) and index-dependent bodies cannot be split.
# CHECK: sub_combiner: split=None
run("sub_combiner", body=ADD.replace("%in, %out", "%out, %in").replace("add", "sub"))
# CHECK: index_body: split=None
run(
    "index_body",
    body="""%i = linalg.index 1 : index
      %c = arith.index_cast %i : index to i32
      %f = arith.sitofp %c : i32 to f32
      %v = arith.addf %in, %f : f32
      %s = arith.addf %v, %out : f32
      linalg.yield %s : f32""",
)

# All split factors the split schedule enumerates.
with TargetInfo.override(arch="x86_64", features=["avx512f"]):
    print(f"split_factors={ReductionRegisterTiling.split_factors(TargetInfo.host())}")
# CHECK: split_factors=[64, 128, 256, 512]
