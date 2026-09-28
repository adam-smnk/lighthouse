# RUN: %PYTHON %s | FileCheck %s

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule import linalg as linalg_sched
from lighthouse.schedule import tile_and_fuse as tf_sched


# Cache-level tile-and-fuse as in the x86 default pipeline.
def cache_stage():
    return [
        linalg_sched.linalg_contract_fold_unit_dims(),
        tf_sched.assign_and_propagate_tile_sizes(strategy="cache"),
        tf_sched.assign_reduction_tile_sizes(strategy="cache", propagate=True),
        tf_sched.assign_elementwise_tile_sizes(strategy="cache"),
        tf_sched.tile_and_fuse_annotated(use_forall=True),
    ]


def run(name: str, payload_str: str):
    print(f"Test: {name}", flush=True)
    with TargetInfo.override(
        arch="x86_64", features=["avx512f"], core_count=16, l2_cache_bytes=1 << 20
    ):
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            payload = ir.Module.parse(payload_str)
            for sched in cache_stage():
                sched.body.operations[0].apply(payload.operation)
            print(payload)


# Softmax with keepdim (d0, 0) reductions, as imported from PyTorch.
SOFTMAX = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @main(%x: tensor<1024x8192xf32>) -> tensor<1024x8192xf32> {
    %ninf = arith.constant 0xFF800000 : f32
    %zero = arith.constant 0.0 : f32
    %e1 = tensor.empty() : tensor<1024x1xf32>
    %m0 = linalg.fill ins(%ninf : f32) outs(%e1 : tensor<1024x1xf32>) -> tensor<1024x1xf32>
    %max = linalg.generic {indexing_maps = [#id, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%x : tensor<1024x8192xf32>) outs(%m0 : tensor<1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %r = arith.maximumf %in, %out : f32
      linalg.yield %r : f32
    } -> tensor<1024x1xf32>
    %e2 = tensor.empty() : tensor<1024x8192xf32>
    %exp = linalg.generic {indexing_maps = [#id, #row, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%x, %max : tensor<1024x8192xf32>, tensor<1024x1xf32>)
        outs(%e2 : tensor<1024x8192xf32>) {
    ^bb0(%in: f32, %m: f32, %out: f32):
      %s = arith.subf %in, %m : f32
      %r = math.exp %s : f32
      linalg.yield %r : f32
    } -> tensor<1024x8192xf32>
    %s0 = linalg.fill ins(%zero : f32) outs(%e1 : tensor<1024x1xf32>) -> tensor<1024x1xf32>
    %sum = linalg.generic {indexing_maps = [#id, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%exp : tensor<1024x8192xf32>) outs(%s0 : tensor<1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %r = arith.addf %in, %out : f32
      linalg.yield %r : f32
    } -> tensor<1024x1xf32>
    %div = linalg.generic {indexing_maps = [#id, #row, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%exp, %sum : tensor<1024x8192xf32>, tensor<1024x1xf32>)
        outs(%e2 : tensor<1024x8192xf32>) {
    ^bb0(%in: f32, %s: f32, %out: f32):
      %r = arith.divf %in, %s : f32
      linalg.yield %r : f32
    } -> tensor<1024x8192xf32>
    return %div : tensor<1024x8192xf32>
  }
}
"""

# The whole softmax is one row-panel loop (4 full rows per tile): the max, the
# exp (computed once), the sum and the division; nothing is left outside.
# CHECK-LABEL: Test: softmax_row_panel
# CHECK-NOT: linalg.generic
# CHECK: scf.forall (%{{.*}}) = (0) to (1024) step (4)
# CHECK: arith.maximumf
# CHECK-SAME: f32
# CHECK: math.exp
# CHECK-NOT: math.exp
# CHECK: arith.addf
# CHECK: linalg.elementwise <div>
# CHECK: scf.forall.in_parallel
# CHECK-NOT: linalg.generic
# CHECK: return
run("softmax_row_panel", SOFTMAX)
