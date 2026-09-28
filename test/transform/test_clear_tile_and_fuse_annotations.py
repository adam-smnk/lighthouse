# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate

# Tile sizes, fusion boundary and panel member annotations, one nested in a loop.
PAYLOAD = """
module {
  func.func @main(%a: tensor<8x16xf32>, %b: tensor<16x8xf32>,
      %c: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %mm = linalg.matmul {transform_ext.fusion_panel}
        ins(%a, %b : tensor<8x16xf32>, tensor<16x8xf32>)
        outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
    %ew = linalg.elementwise <exp>
        {transform_ext.tile_sizes = array<i64: 8, 0>, transform_ext.fusion_boundary}
        ins(%mm : tensor<8x8xf32>) outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %l = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %ew) -> tensor<8x8xf32> {
      %n = linalg.elementwise <exp> {transform_ext.fusion_panel}
          ins(%acc : tensor<8x8xf32>) outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
      scf.yield %n : tensor<8x8xf32>
    }
    return %l : tensor<8x8xf32>
  }
}
"""

# CHECK-LABEL: func.func @main
# CHECK-NOT: transform_ext.
# CHECK: return
with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    module = ir.Module.parse(PAYLOAD)
    with schedule_boilerplate() as (sched, named_seq):
        transform_ext.clear_tile_and_fuse_annotations(named_seq.bodyTarget)
        transform.yield_()
    sched.body.operations[0].apply(module.operation)
    print(module)
