"""Generic tile-and-fuse schedules.

These schedules generalise the matmul-specific cache tiling to arbitrary
KernelBench workloads. They implement the three-step strategy:

    1. assign target tile sizes to anchor ops (GEMMs by default)
    2. propagate the sizes to neighbouring elementwise ops
    3. tile and fuse using the assigned sizes (sizes act as fusion hints)

The sizes are recorded as ``transform_ext.tile_sizes`` attributes between the
steps, so the assignment / propagation policy is decoupled from the tiling.

Tiling decisions are dominated by GEMMs: their tiles take precedence and define
the tiling of elementwise consumers. Kernels without a GEMM fall back to
anchoring an elementwise op and propagating from there.
"""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform

# Op names treated as GEMM anchors. Their tiles take precedence and drive the
# tiling of surrounding elementwise ops via propagation.
GEMM_OPS = [
    "linalg.matmul",
    "linalg.matmul_transpose_a",
    "linalg.matmul_transpose_b",
    "linalg.batch_matmul",
    "linalg.batch_reduce_matmul",
    "linalg.contract",
    "linalg.matvec",
    "linalg.vecmat",
    "linalg.batch_matvec",
]


def assign_tile_sizes(
    anchor_op: str | list[str] | None = None,
    tile_size: int = 32,
) -> ir.Module:
    """
    Assign and propagate target tile sizes (steps 1 and 2).

    Anchor ops matched by `anchor_op` are annotated with target tile sizes,
    which are then propagated to neighbouring elementwise / fill ops so that a
    fusable group shares a consistent tiling.

    This can be applied multiple times with different anchors (e.g. GEMMs first,
    then leftover elementwise ops). Ops that are already annotated keep their
    sizes, so earlier assignments take precedence.

    Args:
        anchor_op: Op(s) to anchor tiling on. Defaults to the GEMM op family.
        tile_size: Size used for tiled dimensions. User hint, default 32.
    Returns:
        Schedule module.
    """
    if anchor_op is None:
        anchor_op = GEMM_OPS

    with schedule_boilerplate() as (sched, named_seq):
        anchors = lh_transform.match_op(named_seq.bodyTarget, anchor_op)
        annotated = transform_ext.assign_tile_sizes(anchors, tile_size=tile_size)
        transform_ext.propagate_tile_sizes(annotated)
        transform.yield_()
    return sched


def assign_elementwise_tile_sizes(tile_size: int = 32) -> ir.Module:
    """
    Anchor tiling on elementwise ops (step 1 and 2 fallback).

    Intended for kernels without a GEMM: an elementwise op's tile size is chosen
    and propagated to its neighbours so they can be fused. Ops already annotated
    (e.g. by a preceding GEMM assignment) are left untouched.

    Args:
        tile_size: Size used for tiled dimensions. User hint, default 32.
    Returns:
        Schedule module.
    """
    return assign_tile_sizes(anchor_op="linalg.generic", tile_size=tile_size)


def tile_and_fuse_annotated(
    target_op: str | list[str] | None = None,
    use_forall: bool = True,
) -> ir.Module:
    """
    Tile and fuse annotated groups using their tile-size annotations (step 3).

    Among the matched candidate ops, the fusion roots are selected (annotated
    ops with no annotated consumer, e.g. the last elementwise op after a GEMM).
    Each root is tiled with its annotated `transform_ext.tile_sizes` and its
    producers are greedily fused into the tiled loop. This pulls a GEMM and its
    elementwise consumers into a single tiled loop. The annotations act as the
    fusion hints: producers sharing a consistent tiling fuse cleanly.

    Args:
        target_op: Candidate op(s) to consider. Defaults to all linalg ops.
        use_forall: Generate `scf.forall` loops (parallel) when tiling.
    Returns:
        Schedule module.
    """
    if target_op is None:
        target_op = structured.MatchInterfaceEnum.LinalgOp

    with schedule_boilerplate() as (sched, named_seq):
        candidates = lh_transform.match_op(named_seq.bodyTarget, target_op)
        roots = transform_ext.get_fusion_roots(candidates)
        with lh_transform.foreach(roots) as op:
            tiles = transform_ext.get_tile_sizes(op)
            structured.FuseOp(
                op,
                tile_sizes=tiles,
                apply_cleanup=True,
                use_forall=use_forall,
            )
            transform.yield_()
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return sched
