# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def run(name: str, payload_str: str):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        with schedule_boilerplate() as (sched, named_seq):
            transform_ext.fold_relayout_into_reduction(named_seq.bodyTarget)
            lh_transform.cleanup(named_seq.bodyTarget)
            transform.yield_()
        sched.body.operations[0].apply(payload.operation)
        print(payload)


# A packed 64x64 tensor (32x32 blocks) is unpacked and repacked with row blocks
# only to feed a row max; the unit dim comes from a keepdim reduction.
RELAYOUT_TEMPLATE = """
#in = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#out = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
module {
  func.func @main(%x: tensor<2x2x32x32xf32>, %o: tensor<2x1x32xf32>) -> tensor<2x1x32xf32> {
    %e = tensor.expand_shape %x [[0, 1], [2], [3], [4]] output_shape [2, 1, 2, 32, 32]
        : tensor<2x2x32x32xf32> into tensor<2x1x2x32x32xf32>
    %u0 = tensor.empty() : tensor<64x1x64xf32>
    %u = linalg.unpack %e inner_dims_pos = [0, 2] inner_tiles = [32, 32] into %u0
        : tensor<2x1x2x32x32xf32> -> tensor<64x1x64xf32>
    %p0 = tensor.empty() : tensor<2x1x64xPACK_TILExf32>
    %p = linalg.pack %u inner_dims_pos = [0] inner_tiles = [PACK_TILE] into %p0
        : tensor<64x1x64xf32> -> tensor<2x1x64xPACK_TILExf32>
    %r = linalg.generic {indexing_maps = [#in, #out],
        iterator_types = ["parallel", "parallel", "reduction", "parallel"]}
        ins(%p : tensor<2x1x64xPACK_TILExf32>) outs(%o : tensor<2x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %m = arith.maximumf %in, %out : f32
      linalg.yield %m : f32
    } -> tensor<2x1x32xf32>
    return %r : tensor<2x1x32xf32>
  }
}
"""

# The max reads the packed source directly: the reduced columns split into the
# column block (d2) and in-block column (d4) loops; the relayout ops are gone.
# CHECK-LABEL: Test: fold_relayout
# CHECK-DAG: #[[IN:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
# CHECK-DAG: #[[OUT:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
# CHECK: func.func @main(%[[X:.+]]: tensor<2x2x32x32xf32>
# CHECK-NOT: linalg.unpack
# CHECK-NOT: linalg.pack
# CHECK: linalg.generic {indexing_maps = [#[[IN]], #[[OUT]]]
# CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel", "reduction"]
# CHECK-SAME: ins(%[[X]] : tensor<2x2x32x32xf32>)
run("fold_relayout", RELAYOUT_TEMPLATE.replace("PACK_TILE", "32"))

# Row blocks of 16 differ from the source blocks of 32: left untouched.
# CHECK-LABEL: Test: no_fold_different_blocks
# CHECK: linalg.unpack
# CHECK: linalg.pack
# CHECK: linalg.generic
run(
    "no_fold_different_blocks",
    RELAYOUT_TEMPLATE.replace("PACK_TILE", "16")
    .replace("2x1x64x16", "4x1x64x16")
    .replace("tensor<2x1x32xf32>", "tensor<4x1x16xf32>"),
)


def plain_relayout(
    unpack_attrs: str = "",
    plain: str = "64x64",
    pack_dim: int = 0,
    packed: str = "2x64x32",
    out_map: str = "(d0, d2)",
    out: str = "2x32",
    iterators: str = '"parallel", "reduction", "parallel"',
    body: str = "",
    val: str = "%in",
) -> str:
    """Plain row max over pack(unpack(x)) of a 2x2x32x32 (32x32 blocks) tensor."""
    return f"""
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out = affine_map<(d0, d1, d2) -> {out_map}>
module {{
  func.func @main(%x: tensor<2x2x32x32xf32>, %o: tensor<{out}xf32>) -> tensor<{out}xf32> {{
    %u0 = tensor.empty() : tensor<{plain}xf32>
    %u = linalg.unpack %x {unpack_attrs} inner_dims_pos = [0, 1] inner_tiles = [32, 32]
        into %u0 : tensor<2x2x32x32xf32> -> tensor<{plain}xf32>
    %p0 = tensor.empty() : tensor<{packed}xf32>
    %p = linalg.pack %u inner_dims_pos = [{pack_dim}] inner_tiles = [32] into %p0
        : tensor<{plain}xf32> -> tensor<{packed}xf32>
    %r = linalg.generic {{indexing_maps = [#id, #out], iterator_types = [{iterators}]}}
        ins(%p : tensor<{packed}xf32>) outs(%o : tensor<{out}xf32>) {{
    ^bb0(%in: f32, %acc: f32):
      {body}
      %m = arith.maximumf {val}, %acc : f32
      linalg.yield %m : f32
    }} -> tensor<{out}xf32>
    return %r : tensor<{out}xf32>
  }}
}}
"""


# Without an expand_shape: the reduced columns split into the column block (d1)
# and in-block column (d3) loops of the unpack source.
# CHECK-LABEL: Test: fold_relayout_plain
# CHECK-DAG: #[[IN:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-DAG: #[[OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
# CHECK: func.func @main(%[[X:.+]]: tensor<2x2x32x32xf32>
# CHECK-NOT: linalg.unpack
# CHECK-NOT: linalg.pack
# CHECK: linalg.generic {indexing_maps = [#[[IN]], #[[OUT]]]
# CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel", "reduction"]
# CHECK-SAME: ins(%[[X]] : tensor<2x2x32x32xf32>)
run("fold_relayout_plain", plain_relayout())

# The unpack drops padding (60 columns out of 64): the packed source holds
# padding elements that must not be reduced.
# CHECK-LABEL: Test: no_fold_unpack_padding
# CHECK: linalg.unpack
# CHECK: linalg.pack
# CHECK: linalg.generic
run("no_fold_unpack_padding", plain_relayout(plain="64x60", packed="2x60x32"))

# Rows are blocked by the unpack only but not reduced: the row loop cannot be
# split into the source block dims without changing the output.
# CHECK-LABEL: Test: no_fold_unblocked_parallel_dim
# CHECK: linalg.unpack
# CHECK: linalg.pack
# CHECK: linalg.generic
run(
    "no_fold_unblocked_parallel_dim",
    plain_relayout(
        pack_dim=1,
        packed="64x2x32",
        out_map="(d0)",
        out="64",
        iterators='"parallel", "reduction", "reduction"',
    ),
)

# Permuted outer dims of the unpack are not supported.
# CHECK-LABEL: Test: no_fold_outer_perm
# CHECK: linalg.unpack
# CHECK: linalg.pack
# CHECK: linalg.generic
run("no_fold_outer_perm", plain_relayout(unpack_attrs="outer_dims_perm = [1, 0]"))

# The body reads the iteration indices, which change with the loop structure.
# CHECK-LABEL: Test: no_fold_index_op
# CHECK: linalg.unpack
# CHECK: linalg.pack
# CHECK: linalg.index
run(
    "no_fold_index_op",
    plain_relayout(
        body="""%i = linalg.index 1 : index
      %c = arith.index_cast %i : index to i32
      %f = arith.sitofp %c : i32 to f32
      %v = arith.addf %in, %f : f32""",
        val="%v",
    ),
)

# The pack source is not an unpack: nothing to fold.
# CHECK-LABEL: Test: no_fold_without_unpack
# CHECK: linalg.pack
# CHECK: linalg.generic
# CHECK-SAME: ins(%{{.+}} : tensor<2x64x32xf32>)
run(
    "no_fold_without_unpack",
    """
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  func.func @main(%x: tensor<64x64xf32>, %o: tensor<2x32xf32>) -> tensor<2x32xf32> {
    %p0 = tensor.empty() : tensor<2x64x32xf32>
    %p = linalg.pack %x inner_dims_pos = [0] inner_tiles = [32] into %p0
        : tensor<64x64xf32> -> tensor<2x64x32xf32>
    %r = linalg.generic {indexing_maps = [#id, #out],
        iterator_types = ["parallel", "reduction", "parallel"]}
        ins(%p : tensor<2x64x32xf32>) outs(%o : tensor<2x32xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %m = arith.maximumf %in, %acc : f32
      linalg.yield %m : f32
    } -> tensor<2x32xf32>
    return %r : tensor<2x32xf32>
  }
}
""",
)
