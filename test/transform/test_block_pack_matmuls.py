# RUN: %PYTHON %s | FileCheck %s

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.schedule.packing import block_pack_matmuls


def run(name: str, payload_str: str):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = block_pack_matmuls(block_factors=[32, 32, 32])
        sched.body.operations[0].apply(payload.operation)
        print(payload)


# Matmul -> exp -> keepdim row max over a unit-dim expand_shape -> exp - max,
# as imported from PyTorch for a softmax epilogue.
MATMUL_ROW_MAX = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#rows3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
#col = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %bias: tensor<64xf32>,
      %c: tensor<64x64xf32>, %m: tensor<64xf32>) -> tensor<64x64xf32> {
    %ninf = arith.constant 0xFF800000 : f32
    %mm = linalg.matmul ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
    %exp = linalg.generic {indexing_maps = [#id, #col, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm, %bias : tensor<64x64xf32>, tensor<64xf32>)
        outs(%c : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.addf %x, %y : f32
      %r = math.exp %s : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    %m1 = tensor.expand_shape %m [[0, 1]] output_shape [64, 1]
        : tensor<64xf32> into tensor<64x1xf32>
    %init = linalg.fill ins(%ninf : f32) outs(%m1 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %e = tensor.expand_shape %exp [[0, 1], [2]] output_shape [64, 1, 64]
        : tensor<64x64xf32> into tensor<64x1x64xf32>
    %max = linalg.generic {indexing_maps = [#id3, #rows3],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%e : tensor<64x1x64xf32>) outs(%init : tensor<64x1xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.maximumf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<64x1xf32>
    %sub = linalg.generic {indexing_maps = [#id, #row, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%exp, %max : tensor<64x64xf32>, tensor<64x1xf32>)
        outs(%c : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %r = arith.subf %x, %y : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    return %sub : tensor<64x64xf32>
  }
}
"""

# The row max reads the packed exp result directly instead of an unpack and a
# repack into row blocks.
# CHECK-LABEL: Test: matmul_row_max
# CHECK: math.exp
# CHECK: } -> tensor<2x2x32x32xf32>
# CHECK-NOT: linalg.unpack
# CHECK: linalg.generic
# CHECK-SAME: ins(%{{.+}} : tensor<2x2x32x32xf32>)
# CHECK: arith.maximumf
run("matmul_row_max", MATMUL_ROW_MAX)
