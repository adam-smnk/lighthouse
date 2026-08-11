# RUN: %PYTHON %s | FileCheck %s

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule import tile_and_fuse as tf


def run(name: str, payload_str: str, *schedules):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        modules = []
        for make_schedule in schedules:
            sched = make_schedule()
            modules.append(sched)
            sched.body.operations[0].apply(payload.operation)
        print(payload)


def run_with_target_override(
    name: str,
    payload_str: str,
    *,
    features: list[str] | None = None,
    arch: str | None = None,
    schedules,
):
    with TargetInfo.override(features=features, arch=arch):
        run(name, payload_str, *schedules)


def assign_register_parallel():
    return tf.assign_and_propagate_tile_sizes(
        tile_size=32,
        strategy="register_parallel",
    )


def assign_register_parallel_auto_amx():
    return tf.assign_and_propagate_tile_sizes(
        tile_size=32,
        strategy="register_parallel",
    )


def tile_and_fuse_keep():
    return tf.tile_and_fuse_annotated(clear_annotations=False)


MLP = """
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<32x64xf32>, %w: tensor<64x128xf32>, %b: tensor<128xf32>)
        -> tensor<32x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %3 = linalg.matmul ins(%arg0, %w : tensor<32x64xf32>, tensor<64x128xf32>)
        outs(%2 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%3, %b : tensor<32x128xf32>, tensor<128xf32>)
        outs(%1 : tensor<32x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<32x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%4 : tensor<32x128xf32>)
        outs(%1 : tensor<32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.cmpf ugt, %in, %cst : f32
      %7 = arith.select %6, %in, %cst : f32
      linalg.yield %7 : f32
    } -> tensor<32x128xf32>
    return %5 : tensor<32x128xf32>
  }
}
"""


# CHECK-LABEL: Test: register_parallel_tile_and_fuse
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 8, 32, 0>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 8, 32>
# CHECK: scf.forall
run(
    "register_parallel_tile_and_fuse", MLP, assign_register_parallel, tile_and_fuse_keep
)


# Register-parallel strategy can infer target defaults from features.
# CHECK-LABEL: Test: register_parallel_target_defaults_amx
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 8, 32, 0>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 8, 32>
run_with_target_override(
    "register_parallel_target_defaults_amx",
    MLP,
    features=["amx_tile"],
    schedules=(assign_register_parallel_auto_amx, tile_and_fuse_keep),
)
