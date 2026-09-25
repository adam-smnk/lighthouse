# RUN: %PYTHON %s | FileCheck %s

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.linalg import split_inner_reductions


def run(name: str, payload_str: str):
    print(f"Test: {name}", flush=True)
    with TargetInfo.override(arch="x86_64", features=["avx512f"], core_count=16):
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            payload = ir.Module.parse(payload_str)
            sched = split_inner_reductions()
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
      %s = arith.maximumf %in, %out : f32
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


def row_reduce(cols: int):
    return reduce_payload(
        "(d0, d1) -> (d0, d1)",
        "(d0, d1) -> (d0)",
        '"parallel", "reduction"',
        f"64x{cols}xf32",
        "64xf32",
    )


# The split factor is lanes (16) x accumulator chains (8) on AVX-512 f32.
# CHECK-LABEL: Test: split_row
# CHECK: %[[NEUTRAL:.+]] = arith.constant 0xFF800000 : f32
# CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2]] output_shape [64, 32, 128]
# CHECK: %[[FILL:.+]] = linalg.fill ins(%[[NEUTRAL]] : f32) outs(%{{.*}} : tensor<64x128xf32>)
# CHECK: %[[PARTIAL:.+]] = linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
# CHECK-SAME: ins(%{{.*}} : tensor<64x32x128xf32>) outs(%[[FILL]] : tensor<64x128xf32>)
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "reduction"]
# CHECK-SAME: ins(%[[PARTIAL]] : tensor<64x128xf32>) outs(%{{.*}} : tensor<64xf32>)
run("split_row", row_reduce(4096))


# Rank-0 output (e.g. a row slice inside a cache-level loop).
# CHECK-LABEL: Test: split_rank1
# CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [8, 128]
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["reduction", "parallel"]
# CHECK: } -> tensor<128xf32>
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["reduction"]
# CHECK: } -> tensor<f32>
run(
    "split_rank1",
    reduce_payload("(d0) -> (d0)", "(d0) -> ()", '"reduction"', "1024xf32", "f32"),
)


# Not splittable: already short enough, non-divisible, column and multi-dim
# reductions are left untouched.
# CHECK-LABEL: Test: no_split_short_row
# CHECK-NOT: tensor.expand_shape
# CHECK: return
run("no_split_short_row", row_reduce(128))

# CHECK-LABEL: Test: no_split_non_divisible
# CHECK-NOT: tensor.expand_shape
# CHECK: return
run("no_split_non_divisible", row_reduce(4095))

# CHECK-LABEL: Test: no_split_column
# CHECK-NOT: tensor.expand_shape
# CHECK: return
run(
    "no_split_column",
    reduce_payload(
        "(d0, d1) -> (d0, d1)",
        "(d0, d1) -> (d1)",
        '"reduction", "parallel"',
        "4096x64xf32",
        "64xf32",
    ),
)

# CHECK-LABEL: Test: no_split_multi_dim
# CHECK-NOT: tensor.expand_shape
# CHECK: return
run(
    "no_split_multi_dim",
    reduce_payload(
        "(d0, d1, d2) -> (d0, d1, d2)",
        "(d0, d1, d2) -> (d0)",
        '"parallel", "reduction", "reduction"',
        "16x64x256xf32",
        "16xf32",
    ),
)


# Matmul followed by an elementwise epilogue and a row max: only the row
# reduction is split, the matmul and its epilogue are untouched.
MATMUL_SOFTMAX = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%a: tensor<64x256xf32>, %b: tensor<256x1024xf32>,
      %c: tensor<64x1024xf32>, %o: tensor<64xf32>) -> tensor<64xf32> {
    %mm = linalg.matmul ins(%a, %b : tensor<64x256xf32>, tensor<256x1024xf32>)
        outs(%c : tensor<64x1024xf32>) -> tensor<64x1024xf32>
    %t = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm : tensor<64x1024xf32>) outs(%c : tensor<64x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %e = math.tanh %in : f32
      linalg.yield %e : f32
    } -> tensor<64x1024xf32>
    %r = linalg.generic {indexing_maps = [#id, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%t : tensor<64x1024xf32>) outs(%o : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %s = arith.maximumf %in, %out : f32
      linalg.yield %s : f32
    } -> tensor<64xf32>
    return %r : tensor<64xf32>
  }
}
"""

# CHECK-LABEL: Test: split_after_matmul
# CHECK: %[[MM:.+]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<64x256xf32>, tensor<256x1024xf32>)
# CHECK: %[[EPI:.+]] = linalg.generic
# CHECK-SAME: ins(%[[MM]] : tensor<64x1024xf32>)
# CHECK: math.tanh
# CHECK: tensor.expand_shape %[[EPI]] {{\[\[}}0], [1, 2]] output_shape [64, 8, 128]
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "reduction"]
# CHECK-NOT: linalg.generic
# CHECK: return
run("split_after_matmul", MATMUL_SOFTMAX)
