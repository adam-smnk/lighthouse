# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def apply_filter(payload: str, name: str):
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(payload)
        with schedule_boilerplate() as (sched, named_seq):
            candidates = lh_transform.match_op(
                named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
            )
            filtered = transform_ext.filter_not_feeding_reductions(candidates)
            transform.print_(target=filtered, name=name)
            transform.yield_()
        sched.body.operations[0].apply(module.operation)


# @feeds: the matmul reaches the row max through an elementwise op and a
# rank-reducing full slice, the elementwise op through the slice.
# @no_feed: the matmul only feeds an elementwise op, and the expand_shape is
# not a fusable relayout, so the second matmul does not feed the row max.
PAYLOAD = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @feeds(%a: tensor<1x8x16xf32>, %b: tensor<1x16x8xf32>,
      %c: tensor<1x8x8xf32>, %r: tensor<8xf32>) -> tensor<8xf32> {
    %mm = linalg.batch_matmul
        ins(%a, %b : tensor<1x8x16xf32>, tensor<1x16x8xf32>)
        outs(%c : tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
    %ew = linalg.elementwise <exp>
        ins(%mm : tensor<1x8x8xf32>) outs(%c : tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
    %s = tensor.extract_slice %ew[0, 0, 0] [1, 8, 8] [1, 1, 1]
        : tensor<1x8x8xf32> to tensor<8x8xf32>
    %max = linalg.generic {indexing_maps = [#id, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%s : tensor<8x8xf32>) outs(%r : tensor<8xf32>) {
    ^bb0(%x: f32, %o: f32):
      %m = arith.maximumf %x, %o : f32
      linalg.yield %m : f32
    } -> tensor<8xf32>
    return %max : tensor<8xf32>
  }
  func.func @no_feed(%a: tensor<8x16xf32>, %b: tensor<16x8xf32>,
      %c: tensor<8x8xf32>, %r: tensor<2x4xf32>)
      -> (tensor<8x8xf32>, tensor<2x4xf32>) {
    %mm = linalg.matmul ins(%a, %b : tensor<8x16xf32>, tensor<16x8xf32>)
        outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
    %ew = linalg.elementwise <exp>
        ins(%mm : tensor<8x8xf32>) outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
    %mm2 = linalg.matmul ins(%a, %b : tensor<8x16xf32>, tensor<16x8xf32>)
        outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
    %e = tensor.expand_shape %mm2 [[0, 1], [2]] output_shape [2, 4, 8]
        : tensor<8x8xf32> into tensor<2x4x8xf32>
    %max = linalg.reduce ins(%e : tensor<2x4x8xf32>) outs(%r : tensor<2x4xf32>)
        dimensions = [2]
      (%x: f32, %o: f32) {
        %m = arith.maximumf %x, %o : f32
        linalg.yield %m : f32
      }
    return %ew, %max : tensor<8x8xf32>, tensor<2x4xf32>
  }
}
"""

# CHECK-LABEL: IR printer: PAYLOAD
# CHECK-NOT: linalg.batch_matmul
# CHECK-NOT: tensor<1x8x8xf32>
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "reduction"]
# CHECK-NEXT: ^bb0
# CHECK: linalg.matmul
# CHECK: linalg.elementwise
# CHECK: linalg.matmul
# CHECK: linalg.reduce
apply_filter(PAYLOAD, name="PAYLOAD")
