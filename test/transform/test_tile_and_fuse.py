# RUN: %PYTHON %s | FileCheck %s

"""Tests for the generic tile-and-fuse transform ops and schedules.

Exercises the three-step strategy on linalg payloads:
    1. assign tile sizes to anchor ops (GEMMs / elementwise)
    2. propagate them to neighbouring ops
    3. tile and fuse using the annotations
"""

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.schedule import tile_and_fuse as tf


def run(name: str, payload_str: str, *schedules):
    """Parse a payload, apply the given schedules in order and print it."""
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load(reload=True)
        payload = ir.Module.parse(payload_str)
        # Keep schedule modules alive while applying them.
        modules = []
        for make_schedule in schedules:
            sched = make_schedule()
            modules.append(sched)
            sched.body.operations[0].apply(payload.operation)
        print(payload)


def assign_gemm():
    return tf.assign_tile_sizes(tile_size=32)


def assign_elementwise():
    return tf.assign_elementwise_tile_sizes(tile_size=32)


def tile_and_fuse():
    return tf.tile_and_fuse_annotated()


MLP = """
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<32x64xf32>, %w: tensor<64x128xf32>, %b: tensor<128xf32>) -> tensor<32x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %3 = linalg.matmul ins(%arg0, %w : tensor<32x64xf32>, tensor<64x128xf32>) outs(%2 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %b : tensor<32x128xf32>, tensor<128xf32>) outs(%1 : tensor<32x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<32x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<32x128xf32>) outs(%1 : tensor<32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.cmpf ugt, %in, %cst : f32
      %7 = arith.select %6, %in, %cst : f32
      linalg.yield %7 : f32
    } -> tensor<32x128xf32>
    return %5 : tensor<32x128xf32>
  }
}
"""

ELTWISE = """
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<64x256xf32>
    %1 = linalg.generic {indexing_maps=[#map,#map], iterator_types=["parallel","parallel"]} ins(%a: tensor<64x256xf32>) outs(%0: tensor<64x256xf32>) {
    ^bb0(%in: f32, %o: f32):
      %e = math.exp %in : f32
      linalg.yield %e : f32
    } -> tensor<64x256xf32>
    %2 = linalg.generic {indexing_maps=[#map,#map], iterator_types=["parallel","parallel"]} ins(%1: tensor<64x256xf32>) outs(%0: tensor<64x256xf32>) {
    ^bb0(%in: f32, %o: f32):
      %c = arith.cmpf ugt, %in, %cst : f32
      %s = arith.select %c, %in, %cst : f32
      linalg.yield %s : f32
    } -> tensor<64x256xf32>
    return %2 : tensor<64x256xf32>
  }
}
"""

BMM = """
module {
  func.func @main(%a: tensor<8x64x96xf32>, %b: tensor<8x96x128xf32>) -> tensor<8x64x128xf32> {
    %cst = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<8x64x128xf32>
    %1 = linalg.fill ins(%cst: f32) outs(%0: tensor<8x64x128xf32>) -> tensor<8x64x128xf32>
    %2 = linalg.batch_matmul ins(%a, %b: tensor<8x64x96xf32>, tensor<8x96x128xf32>) outs(%1: tensor<8x64x128xf32>) -> tensor<8x64x128xf32>
    return %2 : tensor<8x64x128xf32>
  }
}
"""

REDUCE = """
#in = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%a: tensor<64x256xf32>) -> tensor<64xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<64xf32>
    %f = linalg.fill ins(%cst: f32) outs(%e: tensor<64xf32>) -> tensor<64xf32>
    %r = linalg.generic {indexing_maps=[#in,#out], iterator_types=["parallel","reduction"]} ins(%a: tensor<64x256xf32>) outs(%f: tensor<64xf32>) {
    ^bb0(%in: f32, %o: f32):
      %s = arith.addf %in, %o : f32
      linalg.yield %s : f32
    } -> tensor<64xf32>
    return %r : tensor<64xf32>
  }
}
"""

# Two GEMMs chained through a relu. The relu is the epilogue of the first matmul
# and the "prologue" of the second: it must be tiled to match its producer GEMM.
TWO_GEMM = """
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<64x64xf32>, %w0: tensor<64x64xf32>, %w1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %cst = arith.constant 0.0 : f32
    %e0 = tensor.empty() : tensor<64x64xf32>
    %f0 = linalg.fill ins(%cst: f32) outs(%e0: tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm0 = linalg.matmul ins(%a, %w0: tensor<64x64xf32>, tensor<64x64xf32>) outs(%f0: tensor<64x64xf32>) -> tensor<64x64xf32>
    %relu0 = linalg.generic {indexing_maps=[#map,#map], iterator_types=["parallel","parallel"]} ins(%mm0: tensor<64x64xf32>) outs(%e0: tensor<64x64xf32>) {
    ^bb0(%in: f32, %o: f32):
      %c = arith.cmpf ugt, %in, %cst : f32
      %s = arith.select %c, %in, %cst : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    %f1 = linalg.fill ins(%cst: f32) outs(%e0: tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm1 = linalg.matmul ins(%relu0, %w1: tensor<64x64xf32>, tensor<64x64xf32>) outs(%f1: tensor<64x64xf32>) -> tensor<64x64xf32>
    %relu1 = linalg.generic {indexing_maps=[#map,#map], iterator_types=["parallel","parallel"]} ins(%mm1: tensor<64x64xf32>) outs(%e0: tensor<64x64xf32>) {
    ^bb0(%in: f32, %o: f32):
      %c = arith.cmpf ugt, %in, %cst : f32
      %s = arith.select %c, %in, %cst : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %relu1 : tensor<64x64xf32>
  }
}
"""

# A 3-layer MLP: each layer is matmul + bias-add + relu (last layer bias only).
MLP3 = """
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%x: tensor<64x64xf32>, %w0: tensor<64x64xf32>, %b0: tensor<64xf32>, %w1: tensor<64x64xf32>, %b1: tensor<64xf32>, %w2: tensor<64x64xf32>, %b2: tensor<64xf32>) -> tensor<64x64xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<64x64xf32>
    %f0 = linalg.fill ins(%cst: f32) outs(%e: tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm0 = linalg.matmul ins(%x, %w0: tensor<64x64xf32>, tensor<64x64xf32>) outs(%f0: tensor<64x64xf32>) -> tensor<64x64xf32>
    %a0 = linalg.generic {indexing_maps=[#map,#map1,#map], iterator_types=["parallel","parallel"]} ins(%mm0, %b0: tensor<64x64xf32>, tensor<64xf32>) outs(%e: tensor<64x64xf32>) {
    ^bb0(%in: f32, %ib: f32, %o: f32):
      %s = arith.addf %in, %ib : f32
      %c = arith.cmpf ugt, %s, %cst : f32
      %r = arith.select %c, %s, %cst : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    %mm1 = linalg.matmul ins(%a0, %w1: tensor<64x64xf32>, tensor<64x64xf32>) outs(%f0: tensor<64x64xf32>) -> tensor<64x64xf32>
    %a1 = linalg.generic {indexing_maps=[#map,#map1,#map], iterator_types=["parallel","parallel"]} ins(%mm1, %b1: tensor<64x64xf32>, tensor<64xf32>) outs(%e: tensor<64x64xf32>) {
    ^bb0(%in: f32, %ib: f32, %o: f32):
      %s = arith.addf %in, %ib : f32
      %c = arith.cmpf ugt, %s, %cst : f32
      %r = arith.select %c, %s, %cst : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    %mm2 = linalg.matmul ins(%a1, %w2: tensor<64x64xf32>, tensor<64x64xf32>) outs(%f0: tensor<64x64xf32>) -> tensor<64x64xf32>
    %a2 = linalg.generic {indexing_maps=[#map,#map1,#map], iterator_types=["parallel","parallel"]} ins(%mm2, %b2: tensor<64x64xf32>, tensor<64xf32>) outs(%e: tensor<64x64xf32>) {
    ^bb0(%in: f32, %ib: f32, %o: f32):
      %s = arith.addf %in, %ib : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %a2 : tensor<64x64xf32>
  }
}
"""


# A GEMM anchors tiling; its tile sizes are propagated to the fill producer and
# the elementwise (bias, relu) consumers, then the whole group is fused.
# CHECK-LABEL: Test: mlp_assign_propagate
# CHECK: linalg.fill {transform_ext.tile_sizes = array<i64: 32, 32>}
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
run("mlp_assign_propagate", MLP, assign_gemm)


# Tiling the fusion root (relu) and fusing producers pulls the fill, matmul and
# both elementwise ops into a single tiled scf.forall loop.
# CHECK-LABEL: Test: mlp_tile_and_fuse
# CHECK: scf.forall
# CHECK: linalg.fill
# CHECK: linalg.matmul
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
run("mlp_tile_and_fuse", MLP, assign_gemm, tile_and_fuse)


# No GEMM: an elementwise op is anchored and its sizes propagated; the chain is
# tiled into 2D tiles and fused.
# CHECK-LABEL: Test: elementwise_tile_and_fuse
# CHECK: scf.forall ({{.*}}) = (0, 0) to (64, 256) step (32, 32)
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
run("elementwise_tile_and_fuse", ELTWISE, assign_elementwise, tile_and_fuse)


# Batch matmul: the batch dim is tiled by 1, M/N by the tile size, K untiled.
# CHECK-LABEL: Test: batch_matmul
# CHECK: linalg.batch_matmul {transform_ext.tile_sizes = array<i64: 1, 32, 32, 0>}
run("batch_matmul", BMM, assign_gemm)


# Reduction: the parallel dim is tiled, the reduction dim is left untiled (0).
# CHECK-LABEL: Test: reduction
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 0>
run("reduction", REDUCE, assign_elementwise)


# Two consecutive GEMMs: the shared relu is the epilogue of the first matmul, so
# it must be tiled to match that matmul's M/N tiles (2D, not the second matmul's
# M/K prologue tiles). The first matmul is followed by a fully-tiled relu.
# CHECK-LABEL: Test: two_consecutive_gemms_assign
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
run("two_consecutive_gemms_assign", TWO_GEMM, assign_gemm)


# GEMMs act as fusion barriers: two consecutive GEMMs are NOT fused together.
# Each GEMM + its epilogue forms its own scf.forall; the second reads the first
# loop's result through a slice (proving they are separate).
# CHECK-LABEL: Test: two_consecutive_gemms_fuse
# CHECK: %[[F0:.+]] = scf.forall
# CHECK: linalg.matmul
# CHECK: scf.forall.in_parallel
# CHECK: scf.forall
# CHECK: tensor.extract_slice %[[F0]]
# CHECK: linalg.matmul
# CHECK: scf.forall.in_parallel
run("two_consecutive_gemms_fuse", TWO_GEMM, assign_gemm, tile_and_fuse)


# A 3-layer MLP: each layer (GEMM + epilogue) becomes its own fused scf.forall,
# chained through the loop results. Each forall fuses its own (shared) fill.
# CHECK-LABEL: Test: mlp_three_layers
# CHECK: %[[L0:.+]] = scf.forall
# CHECK: linalg.fill
# CHECK: linalg.matmul
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
# CHECK: %[[L1:.+]] = scf.forall
# CHECK: tensor.extract_slice %[[L0]]
# CHECK: linalg.fill
# CHECK: linalg.matmul
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
# CHECK: scf.forall
# CHECK: tensor.extract_slice %[[L1]]
# CHECK: linalg.fill
# CHECK: linalg.matmul
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
run("mlp_three_layers", MLP3, assign_gemm, tile_and_fuse)
