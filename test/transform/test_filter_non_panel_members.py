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
            filtered = transform_ext.filter_non_panel_members(candidates)
            transform.print_(target=filtered, name=name)
            transform.yield_()
        sched.body.operations[0].apply(module.operation)


# The panel member matmul is dropped; the (annotated) panel root is kept.
PAYLOAD = """
module {
  func.func @main(%a: tensor<8x16xf32>, %b: tensor<16x8xf32>,
      %c: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %mm = linalg.matmul {transform_ext.fusion_panel}
        ins(%a, %b : tensor<8x16xf32>, tensor<16x8xf32>)
        outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
    %ew = linalg.elementwise <exp> {transform_ext.tile_sizes = array<i64: 8, 0>}
        ins(%mm : tensor<8x8xf32>) outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %ew : tensor<8x8xf32>
  }
}
"""

# CHECK-LABEL: IR printer: PAYLOAD
# CHECK-NOT: linalg.matmul
# CHECK: linalg.elementwise
# CHECK-NOT: linalg.matmul
apply_filter(PAYLOAD, name="PAYLOAD")
