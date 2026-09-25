# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform.transform_ext import assign_tile_sizes
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate


# Pinned target: 16 cores and a 1 MiB L2, i.e. a 512 KiB panel budget.
def run(name: str, payload_str: str):
    print(f"Test: {name}", flush=True)
    with TargetInfo.override(
        arch="x86_64", features=["avx512f"], core_count=16, l2_cache_bytes=1 << 20
    ):
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            payload = ir.Module.parse(payload_str)
            with schedule_boilerplate() as (sched, named_seq):
                ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.generic")
                assign_tile_sizes(ops, strategy="cache")
                transform.yield_()
            sched.body.operations[0].apply(payload.operation)
            print(payload)


REDUCE_TEMPLATE = """
#in = affine_map<IN_MAP>
#out = affine_map<OUT_MAP>
module {
  func.func @main(%a: tensor<IN_TYPE>, %o: tensor<OUT_TYPE>) -> tensor<OUT_TYPE> {
    %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = [ITERS]}
        ins(%a : tensor<IN_TYPE>) outs(%o : tensor<OUT_TYPE>) {
    ^bb0(%in: f32, %out: f32):
      %s = arith.addf %in, %out : f32
      linalg.yield %s : f32
    } -> tensor<OUT_TYPE>
    return %r : tensor<OUT_TYPE>
  }
}
"""


def reduce_payload(in_map, out_map, iters, in_type, out_type):
    return (
        REDUCE_TEMPLATE.replace("IN_MAP", in_map)
        .replace("OUT_MAP", out_map)
        .replace("ITERS", iters)
        .replace("IN_TYPE", in_type)
        .replace("OUT_TYPE", out_type)
    )


def row_reduce(rows: int, cols: int):
    return reduce_payload(
        "(d0, d1) -> (d0, d1)",
        "(d0, d1) -> (d0)",
        '"parallel", "reduction"',
        f"{rows}x{cols}xf32",
        f"{rows}xf32",
    )


# A 8192-wide row panel costs 96 KiB (3 live tensors): 5 rows fit the budget,
# rounded down to a divisor of 1024.
# CHECK-LABEL: Test: cache_reduction_row_panel
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 4, 0>
run("cache_reduction_row_panel", row_reduce(1024, 8192))


# 8 rows would fit, but only 64 rows exist: rows shrink to expose 4 tiles/core.
# CHECK-LABEL: Test: cache_reduction_row_parallelism
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 0>
run("cache_reduction_row_parallelism", row_reduce(64, 4096))


# RMSNorm-like channel reduction: the contiguous dim is taken whole first.
# CHECK-LABEL: Test: cache_reduction_column
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 0, 2, 256>
run(
    "cache_reduction_column",
    reduce_payload(
        "(d0, d1, d2, d3) -> (d0, d1, d2, d3)",
        "(d0, d1, d2, d3) -> (d0, d2, d3)",
        '"parallel", "reduction", "parallel", "parallel"',
        "8x64x128x256xf32",
        "8x128x256xf32",
    ),
)


# LayerNorm-like all-reduce larger than the budget: one row per tile.
# CHECK-LABEL: Test: cache_reduction_all_reduce
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 0, 0, 0>
run(
    "cache_reduction_all_reduce",
    reduce_payload(
        "(d0, d1, d2, d3) -> (d0, d1, d2, d3)",
        "(d0, d1, d2, d3) -> (d0)",
        '"parallel", "reduction", "reduction", "reduction"',
        "16x64x256x256xf32",
        "16xf32",
    ),
)
