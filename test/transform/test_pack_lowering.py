# RUN: %PYTHON %s | FileCheck %s

"""Tests for the x86 pack/unpack lowering schedule.

`lower_packs_unpacks` derives the initial pack/unpack tiling from the
tile_size_analysis framework and then vectorizes the resulting transpose / copy
ops. These tests check that it lowers pack/unpack ops of varying dimensionality
(1-D bias packs, 2-D matrix packs, higher-rank packs) to 1-D vector transfers
with nothing left un-lowered, and that the pack/unpack tile sizes are selected
from the pack structure.
"""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
from lighthouse.schedule.x86 import pack_lowering as pl


def run(name, payload_str, make_schedule):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load(reload=True)
        payload = ir.Module.parse(payload_str)
        # Keep the schedule module alive while it is applied.
        sched = make_schedule()
        sched.body.operations[0].apply(payload.operation)
        print(payload)


# A 2-D matrix pack, a copy on the packed tensor and a matching unpack.
PACK_UNPACK_2D = """
module {
  func.func @main(%a: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %d = tensor.empty() : tensor<4x8x32x32xf32>
    %packed = linalg.pack %a inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %d
        : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
    %e = tensor.empty() : tensor<4x8x32x32xf32>
    %r = linalg.copy ins(%packed : tensor<4x8x32x32xf32>)
        outs(%e : tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32>
    %o = tensor.empty() : tensor<128x256xf32>
    %u = linalg.unpack %r inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %o
        : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
    return %u : tensor<128x256xf32>
  }
}
"""

# A higher-rank (batched) pack / unpack: the leading batch dim is handled by the
# per-op tiling without a dedicated batch parameter.
PACK_UNPACK_3D = """
module {
  func.func @main(%a: tensor<2x128x256xf32>) -> tensor<2x128x256xf32> {
    %d = tensor.empty() : tensor<2x4x8x32x32xf32>
    %packed = linalg.pack %a inner_dims_pos = [1, 2] inner_tiles = [32, 32] into %d
        : tensor<2x128x256xf32> -> tensor<2x4x8x32x32xf32>
    %e = tensor.empty() : tensor<2x4x8x32x32xf32>
    %r = linalg.copy ins(%packed : tensor<2x4x8x32x32xf32>)
        outs(%e : tensor<2x4x8x32x32xf32>) -> tensor<2x4x8x32x32xf32>
    %o = tensor.empty() : tensor<2x128x256xf32>
    %u = linalg.unpack %r inner_dims_pos = [1, 2] inner_tiles = [32, 32] into %o
        : tensor<2x4x8x32x32xf32> -> tensor<2x128x256xf32>
    return %u : tensor<2x128x256xf32>
  }
}
"""

# Mixed dimensionality in one payload: a 1-D bias pack (1-D -> 2-D) alongside a
# 2-D matrix pack / unpack (2-D -> 4-D -> 2-D). Exercises the generalization to
# heterogeneous pack ranks in a single schedule application.
MIXED = """
module {
  func.func @main(%bias: tensor<512xf32>, %mat: tensor<128x1024xf32>)
        -> (tensor<16x32xf32>, tensor<128x1024xf32>) {
    %db = tensor.empty() : tensor<16x32xf32>
    %pbias = linalg.pack %bias inner_dims_pos = [0] inner_tiles = [32] into %db
        : tensor<512xf32> -> tensor<16x32xf32>
    %dm = tensor.empty() : tensor<4x32x32x32xf32>
    %pmat = linalg.pack %mat inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %dm
        : tensor<128x1024xf32> -> tensor<4x32x32x32xf32>
    %e = tensor.empty() : tensor<4x32x32x32xf32>
    %r = linalg.copy ins(%pmat : tensor<4x32x32x32xf32>)
        outs(%e : tensor<4x32x32x32xf32>) -> tensor<4x32x32x32xf32>
    %o = tensor.empty() : tensor<128x1024xf32>
    %u = linalg.unpack %r inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %o
        : tensor<4x32x32x32xf32> -> tensor<128x1024xf32>
    return %pbias, %u : tensor<16x32xf32>, tensor<128x1024xf32>
  }
}
"""

# Pre-annotated payload to check pack/unpack tile-size selection in isolation.
PACK_ASSIGN = """
module {
  func.func @main(%a: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %d = tensor.empty() : tensor<4x8x32x32xf32>
    %p = linalg.pack %a inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %d
        : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
    %o = tensor.empty() : tensor<128x256xf32>
    %u = linalg.unpack %p inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %o
        : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
    return %u : tensor<128x256xf32>
  }
}
"""


def lower():
    return pl.lower_packs_unpacks(tile_size=32)


def assign_pack_tiles():
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(
            named_seq.bodyTarget, ["linalg.pack", "linalg.unpack"]
        )
        transform_ext.assign_tile_sizes(ops)
        transform.yield_()
    return sched


# A 2-D pack / unpack lowers to 1-D vector transfers, with no pack/unpack/
# transpose/copy left behind.
# CHECK-LABEL: Test: pack_unpack_2d
# CHECK: vector<32xf32>
# CHECK-NOT: linalg.pack
# CHECK-NOT: linalg.unpack
# CHECK-NOT: linalg.transpose
# CHECK-NOT: linalg.copy
run("pack_unpack_2d", PACK_UNPACK_2D, lower)


# A higher-rank pack / unpack lowers the same way, the batch dim needing no
# special handling.
# CHECK-LABEL: Test: pack_unpack_3d
# CHECK: vector<32xf32>
# CHECK-NOT: linalg.pack
# CHECK-NOT: linalg.unpack
# CHECK-NOT: linalg.transpose
# CHECK-NOT: linalg.copy
run("pack_unpack_3d", PACK_UNPACK_3D, lower)


# 1-D and 2-D packs in the same payload both lower to 1-D vector transfers.
# CHECK-LABEL: Test: mixed
# CHECK: vector<32xf32>
# CHECK-NOT: linalg.pack
# CHECK-NOT: linalg.unpack
# CHECK-NOT: linalg.transpose
# CHECK-NOT: linalg.copy
run("mixed", MIXED, lower)


# Tile-size selection: a pack tiles every source dim by 1; an unpack tiles its
# packed output dims by the inner tile size.
# CHECK-LABEL: Test: pack_unpack_tile_sizes
# CHECK: linalg.pack
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 1>
# CHECK: linalg.unpack
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
run("pack_unpack_tile_sizes", PACK_ASSIGN, assign_pack_tiles)
