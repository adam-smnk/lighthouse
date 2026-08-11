# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform.transform_ext import assign_tile_sizes
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate


def run(name: str, payload_str: str, build_schedule):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = build_schedule()
        sched.body.operations[0].apply(payload.operation)
        print(payload)


F32_MATMUL = """
module {
  func.func @main(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<128x128xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<128x64xf32>, tensor<64x128xf32>)
        outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %mm : tensor<128x128xf32>
  }
}
"""


BF16_MATMUL = """
module {
  func.func @main(%a: tensor<128x64xbf16>, %b: tensor<64x128xbf16>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<128x128xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<128x64xbf16>, tensor<64x128xbf16>)
        outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %mm : tensor<128x128xf32>
  }
}
"""


def build_register_parallel():
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.matmul")
        assign_tile_sizes(
            ops,
            strategy="register_parallel",
        )
        transform.yield_()
    return sched


def build_register_reduction():
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.matmul")
        assign_tile_sizes(
            ops,
            strategy="register_reduction",
        )
        transform.yield_()
    return sched


# CHECK-LABEL: Test: f32_register_parallel_default
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 8, 32, 0>
run("f32_register_parallel_default", F32_MATMUL, lambda: build_register_parallel())


# CHECK-LABEL: Test: bf16_amx_register_parallel_default
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32, 0>
with TargetInfo.override(features=["amx_tile"]):
    run(
        "bf16_amx_register_parallel_default",
        BF16_MATMUL,
        lambda: build_register_parallel(),
    )


# CHECK-LABEL: Test: f32_register_reduction_default
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 0, 2>
run("f32_register_reduction_default", F32_MATMUL, lambda: build_register_reduction())


# CHECK-LABEL: Test: bf16_amx_register_reduction_default
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 0, 32>
with TargetInfo.override(features=["amx_tile"]):
    run(
        "bf16_amx_register_reduction_default",
        BF16_MATMUL,
        lambda: build_register_reduction(),
    )
