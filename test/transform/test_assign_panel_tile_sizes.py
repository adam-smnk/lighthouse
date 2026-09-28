# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate


def run(name: str, payload_str: str, cores: int = 2, l2_cache_bytes: int = 1 << 20):
    print(f"Test: {name}", flush=True)
    with TargetInfo.override(
        arch="x86_64",
        features=["avx512f"],
        core_count=cores,
        l2_cache_bytes=l2_cache_bytes,
    ):
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            payload = ir.Module.parse(payload_str)
            with schedule_boilerplate() as (sched, named_seq):
                ops = lh_transform.match_op(
                    named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
                )
                gemms = transform_ext.filter_contraction_ops(ops)
                transform_ext.assign_panel_tile_sizes(gemms)
                transform.yield_()
            sched.body.operations[0].apply(payload.operation)
            print(payload)


# Packed 64x64x32 GEMM (32x32 blocks) -> packed row max -> packed exp -> unpack
# -> plain row sum -> plain division, i.e. a GEMM + softmax epilogue.
GEMM_SOFTMAX = """
#a = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#b = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#c = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#rows4 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#rows2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%A: tensor<2x1x32x32xf32>, %B: tensor<2x1x32x32xf32>,
      %C: tensor<2x2x32x32xf32>, %m: tensor<2x32xf32>, %s: tensor<64xf32>,
      %o: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %mm = linalg.generic {indexing_maps = [#a, #b, #c], iterator_types =
        ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
        ins(%A, %B : tensor<2x1x32x32xf32>, tensor<2x1x32x32xf32>)
        outs(%C : tensor<2x2x32x32xf32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %p = arith.mulf %x, %y : f32
      %r = arith.addf %acc, %p : f32
      linalg.yield %r : f32
    } -> tensor<2x2x32x32xf32>
    %max = linalg.generic {indexing_maps = [#id4, #rows4],
        iterator_types = ["parallel", "reduction", "parallel", "reduction"]}
        ins(%mm : tensor<2x2x32x32xf32>) outs(%m : tensor<2x32xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.maximumf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<2x32xf32>
    %exp = linalg.generic {indexing_maps = [#id4, #rows4, #id4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%mm, %max : tensor<2x2x32x32xf32>, tensor<2x32xf32>)
        outs(%C : tensor<2x2x32x32xf32>) {
    ^bb0(%x: f32, %mx: f32, %out: f32):
      %d = arith.subf %x, %mx : f32
      %r = math.exp %d : f32
      linalg.yield %r : f32
    } -> tensor<2x2x32x32xf32>
    %u = linalg.unpack %exp inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %o
        : tensor<2x2x32x32xf32> -> tensor<64x64xf32>
    %sum = linalg.generic {indexing_maps = [#id2, #rows2],
        iterator_types = ["parallel", "reduction"]}
        ins(%u : tensor<64x64xf32>) outs(%s : tensor<64xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.addf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<64xf32>
    %div = linalg.generic {indexing_maps = [#id2, #rows2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%u, %sum : tensor<64x64xf32>, tensor<64xf32>)
        outs(%o : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %r = arith.divf %x, %y : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    return %div : tensor<64x64xf32>
  }
}
"""

# One 32-row block per panel: the terminal division is the only annotated root
# (full columns), every other op of the chain is fused into it.
# CHECK-LABEL: Test: gemm_softmax_panel
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: arith.mulf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: arith.maximumf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: math.exp
# CHECK: linalg.unpack
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: arith.addf
# CHECK: linalg.generic
# CHECK-NOT: transform_ext.fusion_panel
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 0>
# CHECK: arith.divf
run("gemm_softmax_panel", GEMM_SOFTMAX)

# Two panels cannot keep 16 cores busy and sub-block panels would split the
# unpack's blocks: the GEMM result is materialized once instead. Its consumers
# start new fusion groups so that they cannot pull the GEMM in again.
# CHECK-LABEL: Test: gemm_softmax_split
# CHECK-NOT: transform_ext.fusion_panel
# CHECK: linalg.generic
# CHECK-NOT: transform_ext.fusion_boundary
# CHECK: arith.mulf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_boundary
# CHECK: arith.maximumf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_boundary
# CHECK: math.exp
# CHECK-NOT: transform_ext.fusion_panel
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
run("gemm_softmax_split", GEMM_SOFTMAX, cores=16)


# The exp result is also returned: fusing it into panels would compute it (and
# the GEMM) once more to materialize it in full, so the chain is left alone.
# CHECK-LABEL: Test: gemm_softmax_escaping_intermediate
# CHECK-NOT: transform_ext.fusion_panel
# CHECK-NOT: transform_ext.fusion_boundary
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
run(
    "gemm_softmax_escaping_intermediate",
    GEMM_SOFTMAX.replace(
        "-> tensor<64x64xf32> {",
        "-> (tensor<64x64xf32>, tensor<2x2x32x32xf32>) {",
    ).replace(
        "return %div : tensor<64x64xf32>",
        "return %div, %exp : tensor<64x64xf32>, tensor<2x2x32x32xf32>",
    ),
)


# The GEMM writes column blocks outermost ([NB][MB][mb][nb]) and the unpack
# permutes them back: the row block is still found through the permutation.
GEMM_SOFTMAX_PERMUTED = """
#a = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#b = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#c = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#rows4 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#rows2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%A: tensor<2x1x32x32xf32>, %B: tensor<2x1x32x32xf32>,
      %C: tensor<2x2x32x32xf32>, %m: tensor<2x32xf32>, %s: tensor<64xf32>,
      %o: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %mm = linalg.generic {indexing_maps = [#a, #b, #c], iterator_types =
        ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
        ins(%A, %B : tensor<2x1x32x32xf32>, tensor<2x1x32x32xf32>)
        outs(%C : tensor<2x2x32x32xf32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %p = arith.mulf %x, %y : f32
      %r = arith.addf %acc, %p : f32
      linalg.yield %r : f32
    } -> tensor<2x2x32x32xf32>
    %max = linalg.generic {indexing_maps = [#id4, #rows4],
        iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
        ins(%mm : tensor<2x2x32x32xf32>) outs(%m : tensor<2x32xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.maximumf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<2x32xf32>
    %exp = linalg.generic {indexing_maps = [#id4, #rows4, #id4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%mm, %max : tensor<2x2x32x32xf32>, tensor<2x32xf32>)
        outs(%C : tensor<2x2x32x32xf32>) {
    ^bb0(%x: f32, %mx: f32, %out: f32):
      %d = arith.subf %x, %mx : f32
      %r = math.exp %d : f32
      linalg.yield %r : f32
    } -> tensor<2x2x32x32xf32>
    %u = linalg.unpack %exp outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
        inner_tiles = [32, 32] into %o : tensor<2x2x32x32xf32> -> tensor<64x64xf32>
    %sum = linalg.generic {indexing_maps = [#id2, #rows2],
        iterator_types = ["parallel", "reduction"]}
        ins(%u : tensor<64x64xf32>) outs(%s : tensor<64xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.addf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<64xf32>
    %div = linalg.generic {indexing_maps = [#id2, #rows2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%u, %sum : tensor<64x64xf32>, tensor<64xf32>)
        outs(%o : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %r = arith.divf %x, %y : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    return %div : tensor<64x64xf32>
  }
}
"""

# CHECK-LABEL: Test: gemm_softmax_permuted_outer_dims
# CHECK: linalg.unpack
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: arith.addf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 0>
# CHECK: arith.divf
run("gemm_softmax_permuted_outer_dims", GEMM_SOFTMAX_PERMUTED)


# Softmax fully in the packed layout with only a trailing unpack: the unpack
# stays outside the panel and the packed division is the root, so panels can
# be split into sub-blocks along the in-block rows.
GEMM_SOFTMAX_PACKED = """
#a = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#b = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#c = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#rows4 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
module {
  func.func @main(%A: tensor<2x1x32x32xf32>, %B: tensor<2x1x32x32xf32>,
      %C: tensor<2x2x32x32xf32>, %m: tensor<2x32xf32>, %s: tensor<2x32xf32>,
      %o: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %mm = linalg.generic {indexing_maps = [#a, #b, #c], iterator_types =
        ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
        ins(%A, %B : tensor<2x1x32x32xf32>, tensor<2x1x32x32xf32>)
        outs(%C : tensor<2x2x32x32xf32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %p = arith.mulf %x, %y : f32
      %r = arith.addf %acc, %p : f32
      linalg.yield %r : f32
    } -> tensor<2x2x32x32xf32>
    %max = linalg.generic {indexing_maps = [#id4, #rows4],
        iterator_types = ["parallel", "reduction", "parallel", "reduction"]}
        ins(%mm : tensor<2x2x32x32xf32>) outs(%m : tensor<2x32xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.maximumf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<2x32xf32>
    %exp = linalg.generic {indexing_maps = [#id4, #rows4, #id4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%mm, %max : tensor<2x2x32x32xf32>, tensor<2x32xf32>)
        outs(%C : tensor<2x2x32x32xf32>) {
    ^bb0(%x: f32, %mx: f32, %out: f32):
      %d = arith.subf %x, %mx : f32
      %r = math.exp %d : f32
      linalg.yield %r : f32
    } -> tensor<2x2x32x32xf32>
    %sum = linalg.generic {indexing_maps = [#id4, #rows4],
        iterator_types = ["parallel", "reduction", "parallel", "reduction"]}
        ins(%exp : tensor<2x2x32x32xf32>) outs(%s : tensor<2x32xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.addf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<2x32xf32>
    %div = linalg.generic {indexing_maps = [#id4, #rows4, #id4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%exp, %sum : tensor<2x2x32x32xf32>, tensor<2x32xf32>)
        outs(%C : tensor<2x2x32x32xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %r = arith.divf %x, %y : f32
      linalg.yield %r : f32
    } -> tensor<2x2x32x32xf32>
    %u = linalg.unpack %div inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %o
        : tensor<2x2x32x32xf32> -> tensor<64x64xf32>
    return %u : tensor<64x64xf32>
  }
}
"""

# Two 32-row panels cannot feed 4 cores: 16-row sub-block panels are used.
# CHECK-LABEL: Test: gemm_softmax_sub_block
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: arith.mulf
# CHECK: arith.maximumf
# CHECK: math.exp
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: arith.addf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 0, 16, 0>
# CHECK: arith.divf
# CHECK: linalg.unpack
# CHECK-NOT: transform_ext.fusion_panel
# CHECK: return
run("gemm_softmax_sub_block", GEMM_SOFTMAX_PACKED, cores=4)

# A 32-row panel (24 KiB) and a 16-row one (12 KiB) exceed the 8 KiB budget of a
# 16 KiB L2: 8-row sub-block panels are used.
# CHECK-LABEL: Test: gemm_softmax_l2_budget
# CHECK: transform_ext.tile_sizes = array<i64: 1, 0, 8, 0>
# CHECK-NOT: linalg.generic
# CHECK: arith.divf
run("gemm_softmax_l2_budget", GEMM_SOFTMAX_PACKED, l2_cache_bytes=16 * 1024)

# Not even an 8-row panel (6 KiB) fits a 4 KiB L2: split after the GEMM.
# CHECK-LABEL: Test: gemm_softmax_l2_split
# CHECK-NOT: transform_ext.fusion_panel
# CHECK: arith.mulf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_boundary
# CHECK: arith.maximumf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_boundary
# CHECK: math.exp
# CHECK-NOT: transform_ext.fusion_panel
# CHECK: return
run("gemm_softmax_l2_split", GEMM_SOFTMAX_PACKED, l2_cache_bytes=4 * 1024)


# Plain (unpacked) matmul with a plain row softmax.
PLAIN_MATMUL_SOFTMAX = """
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#rows2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%A: tensor<64x32xf32>, %B: tensor<32x64xf32>,
      %C: tensor<64x64xf32>, %m: tensor<64xf32>, %s: tensor<64xf32>)
      -> tensor<64x64xf32> {
    %mm = linalg.matmul ins(%A, %B : tensor<64x32xf32>, tensor<32x64xf32>)
        outs(%C : tensor<64x64xf32>) -> tensor<64x64xf32>
    %max = linalg.generic {indexing_maps = [#id2, #rows2],
        iterator_types = ["parallel", "reduction"]}
        ins(%mm : tensor<64x64xf32>) outs(%m : tensor<64xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.maximumf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<64xf32>
    %exp = linalg.generic {indexing_maps = [#id2, #rows2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm, %max : tensor<64x64xf32>, tensor<64xf32>)
        outs(%C : tensor<64x64xf32>) {
    ^bb0(%x: f32, %mx: f32, %out: f32):
      %d = arith.subf %x, %mx : f32
      %r = math.exp %d : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    %sum = linalg.generic {indexing_maps = [#id2, #rows2],
        iterator_types = ["parallel", "reduction"]}
        ins(%exp : tensor<64x64xf32>) outs(%s : tensor<64xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.addf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<64xf32>
    %div = linalg.generic {indexing_maps = [#id2, #rows2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%exp, %sum : tensor<64x64xf32>, tensor<64xf32>)
        outs(%C : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %r = arith.divf %x, %y : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    return %div : tensor<64x64xf32>
  }
}
"""

# Plain M is tiled by the default 32 rows.
# CHECK-LABEL: Test: plain_matmul_softmax
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.fusion_panel
# CHECK-COUNT-3: transform_ext.fusion_panel
# CHECK: linalg.generic
# CHECK-NOT: transform_ext.fusion_panel
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 0>
# CHECK: arith.divf
run("plain_matmul_softmax", PLAIN_MATMUL_SOFTMAX)

# Two 32-row panels cannot feed 4 cores: 16-row panels are used.
# CHECK-LABEL: Test: plain_matmul_softmax_half_rows
# CHECK: transform_ext.tile_sizes = array<i64: 16, 0>
# CHECK: arith.divf
run("plain_matmul_softmax_half_rows", PLAIN_MATMUL_SOFTMAX, cores=4)

# A column max reduces the panel rows: every panel would need all rows, so the
# GEMM is split from its consumers instead.
# CHECK-LABEL: Test: plain_matmul_column_max
# CHECK-NOT: transform_ext.fusion_panel
# CHECK: linalg.matmul
# CHECK-NOT: transform_ext.fusion_boundary
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_boundary
# CHECK: arith.maximumf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_boundary
# CHECK: math.exp
# CHECK-NOT: transform_ext.fusion_panel
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
run(
    "plain_matmul_column_max",
    """
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#cols2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%A: tensor<64x32xf32>, %B: tensor<32x64xf32>,
      %C: tensor<64x64xf32>, %m: tensor<64xf32>) -> tensor<64x64xf32> {
    %mm = linalg.matmul ins(%A, %B : tensor<64x32xf32>, tensor<32x64xf32>)
        outs(%C : tensor<64x64xf32>) -> tensor<64x64xf32>
    %max = linalg.generic {indexing_maps = [#id2, #cols2],
        iterator_types = ["reduction", "parallel"]}
        ins(%mm : tensor<64x64xf32>) outs(%m : tensor<64xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.maximumf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<64xf32>
    %exp = linalg.generic {indexing_maps = [#id2, #cols2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm, %max : tensor<64x64xf32>, tensor<64xf32>)
        outs(%C : tensor<64x64xf32>) {
    ^bb0(%x: f32, %mx: f32, %out: f32):
      %d = arith.subf %x, %mx : f32
      %r = math.exp %d : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    return %exp : tensor<64x64xf32>
  }
}
""",
)

# Reshapes cannot be fused through: the chain ends at the expand_shape and has
# no reduction, so nothing is annotated.
# CHECK-LABEL: Test: plain_matmul_reshape
# CHECK-NOT: transform_ext.fusion_panel
# CHECK-NOT: transform_ext.fusion_boundary
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
run(
    "plain_matmul_reshape",
    """
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#rows3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%A: tensor<64x32xf32>, %B: tensor<32x64xf32>,
      %C: tensor<64x64xf32>, %m: tensor<2x32xf32>) -> tensor<2x32xf32> {
    %mm = linalg.matmul ins(%A, %B : tensor<64x32xf32>, tensor<32x64xf32>)
        outs(%C : tensor<64x64xf32>) -> tensor<64x64xf32>
    %e = tensor.expand_shape %mm [[0, 1], [2]] output_shape [2, 32, 64]
        : tensor<64x64xf32> into tensor<2x32x64xf32>
    %max = linalg.generic {indexing_maps = [#id3, #rows3],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%e : tensor<2x32x64xf32>) outs(%m : tensor<2x32xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.maximumf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<2x32xf32>
    return %max : tensor<2x32xf32>
  }
}
""",
)

# Two GEMMs feed one softmax: the chain of the first GEMM claims the panel and
# the overlapping chain of the second one is left alone (a single root).
# CHECK-LABEL: Test: two_gemms_one_chain
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: linalg.matmul
# CHECK-NOT: transform_ext.fusion_panel
# CHECK-SAME: -> tensor<64x64xf32>
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: arith.addf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.fusion_panel
# CHECK: arith.maximumf
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 0>
# CHECK: arith.subf
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
run(
    "two_gemms_one_chain",
    """
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#rows2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%A: tensor<64x32xf32>, %B: tensor<32x64xf32>,
      %C: tensor<64x64xf32>, %D: tensor<64x64xf32>, %m: tensor<64xf32>)
      -> tensor<64x64xf32> {
    %mm1 = linalg.matmul ins(%A, %B : tensor<64x32xf32>, tensor<32x64xf32>)
        outs(%C : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm2 = linalg.matmul ins(%A, %B : tensor<64x32xf32>, tensor<32x64xf32>)
        outs(%D : tensor<64x64xf32>) -> tensor<64x64xf32>
    %add = linalg.generic {indexing_maps = [#id2, #id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm1, %mm2 : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%C : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %r = arith.addf %x, %y : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    %max = linalg.generic {indexing_maps = [#id2, #rows2],
        iterator_types = ["parallel", "reduction"]}
        ins(%add : tensor<64x64xf32>) outs(%m : tensor<64xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.maximumf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<64xf32>
    %sub = linalg.generic {indexing_maps = [#id2, #rows2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%add, %max : tensor<64x64xf32>, tensor<64xf32>)
        outs(%C : tensor<64x64xf32>) {
    ^bb0(%x: f32, %mx: f32, %out: f32):
      %r = arith.subf %x, %mx : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    return %sub : tensor<64x64xf32>
  }
}
""",
)
