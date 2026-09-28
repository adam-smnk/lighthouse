from collections.abc import Sequence

from mlir import ir
from mlir.dialects import linalg, tensor

from lighthouse.utils.mlir import (
    dim_position,
    indexing_map_for_value,
    indexing_maps,
    map_dims,
    opview,
    pack_inner_blocks,
    pack_outer_perm,
)
from lighthouse.dialects.transform.transform_ext.utils import fusion_analysis as fa


def is_propagatable(op: ir.Operation | ir.OpView) -> bool:
    """Check whether tile sizes may be propagated onto this op.

    True for any structured linalg op that is not a fusion barrier; non-linalg
    ops have no indexing maps to translate tiles through and are excluded.

    Args:
        op: Candidate op to annotate.
    Returns:
        True if `op` can receive propagated tile sizes.
    """
    return indexing_maps(op) is not None and not fa.is_fusion_barrier(op)


def tiles_on_value(
    op: ir.Operation | ir.OpView,
    sizes: Sequence[int],
    value: ir.Value,
) -> list[int] | None:
    """Per-dimension tiles that `op`, tiled by `sizes`, induces on `value`.

    Projecting both a producer's and a consumer's sizes onto the shared tensor
    this way makes tile comparisons robust to transposition: differently ordered
    iteration spaces still agree when they tile the shared tensor identically.

    Args:
        op: Structured linalg op the tiles originate from.
        sizes: `op`'s tile sizes, in its loop order.
        value: Operand or result of `op` to project the tiles onto.
    Returns:
        One entry per dimension of `value` (tensor-dim order), where 0 means the
        dimension is left untiled / not constrained by `op`; None when `op` is not
        a structured linalg op or does not touch `value`.
    """
    ov = opview(op)
    value_map = indexing_map_for_value(ov, value)
    if value_map is None:
        return None
    tiles = [0] * ir.ShapedType(value.type).rank
    for tensor_dim, expr in enumerate(value_map.results):
        pos = dim_position(expr)
        if pos is not None and pos < len(sizes):
            tiles[tensor_dim] = sizes[pos]
    return tiles


def compatible_on_value(
    src_op: ir.Operation | ir.OpView,
    src_sizes: Sequence[int],
    dst_op: ir.Operation | ir.OpView,
    dst_sizes: Sequence[int],
    shared: ir.Value,
) -> bool:
    """Check whether two ops tile a shared tensor compatibly.

    The tiles each op induces on `shared` are compared per tensor dimension. A
    dimension only conflicts when both ops tile it with *different* non-zero
    sizes; a zero (untiled / broadcast / unconstrained) side is a wildcard and
    never conflicts.

    Args:
        src_op: First op sharing the tensor.
        src_sizes: `src_op`'s tile sizes, in its loop order.
        dst_op: Second op sharing the tensor.
        dst_sizes: `dst_op`'s tile sizes, in its loop order.
        shared: Tensor value both ops use.
    Returns:
        True if the two tilings agree on `shared`. Also True when the tiles
        cannot be determined, so grouping errs toward fusion rather than
        over-splitting.
    """
    return compatible_on_values(
        src_op,
        src_sizes,
        shared,
        dst_op,
        dst_sizes,
        shared,
    )


def compatible_on_values(
    src_op: ir.Operation | ir.OpView,
    src_sizes: Sequence[int],
    src_shared: ir.Value,
    dst_op: ir.Operation | ir.OpView,
    dst_sizes: Sequence[int],
    dst_shared: ir.Value,
) -> bool:
    """Check compatibility when source and destination see aliased values.

    This is used when the shared tensor crosses through lightweight wrapper ops
    (for example scf.yield), so the producer and consumer do not use the exact
    same SSA value.

    Args:
        src_op: First op sharing the tensor.
        src_sizes: `src_op`'s tile sizes, in its loop order.
        src_shared: Shared tensor as seen by `src_op`.
        dst_op: Second op sharing the tensor.
        dst_sizes: `dst_op`'s tile sizes, in its loop order.
        dst_shared: Shared tensor as seen by `dst_op`.
    Returns:
        True if the two tilings agree on the shared tensor, or if they cannot
        be determined.
    """
    a = tiles_on_value(src_op, src_sizes, src_shared)
    b = tiles_on_value(dst_op, dst_sizes, dst_shared)
    if a is None or b is None:
        return True
    if len(a) != len(b):
        return True
    return all(x == y for x, y in zip(a, b) if x != 0 and y != 0)


def propagate_through_value(
    src_op: ir.Operation | ir.OpView,
    src_sizes: Sequence[int],
    shared: ir.Value,
    dst_op: ir.Operation | ir.OpView,
) -> list[int] | None:
    """Propagate tile sizes from `src_op` to `dst_op` via a shared tensor.

    The shared tensor's per-dimension tiles are derived from `src_op`'s sizes and
    mapped onto `dst_op`'s iteration space; reduction dims of `dst_op` stay untiled.

    Args:
        src_op: Annotated op the tile sizes come from.
        src_sizes: `src_op`'s tile sizes, in its loop order.
        shared: Tensor value both ops use.
        dst_op: Op to derive tile sizes for.
    Returns:
        `dst_op`'s tile sizes in its loop order, or None if not possible.
    """
    return propagate_through_values(
        src_op,
        src_sizes,
        shared,
        shared,
        dst_op,
    )


def propagate_through_values(
    src_op: ir.Operation | ir.OpView,
    src_sizes: Sequence[int],
    src_shared: ir.Value,
    dst_shared: ir.Value,
    dst_op: ir.Operation | ir.OpView,
) -> list[int] | None:
    """Propagate tile sizes across possibly-aliased shared values.

    `src_shared` and `dst_shared` may be different SSA values that represent the
    same logical tensor across wrapper ops.

    Args:
        src_op: Annotated op the tile sizes come from.
        src_sizes: `src_op`'s tile sizes, in its loop order.
        src_shared: Shared tensor as seen by `src_op`.
        dst_shared: Shared tensor as seen by `dst_op`.
        dst_op: Op to derive tile sizes for.
    Returns:
        `dst_op`'s tile sizes in its loop order, or None if not possible.
    """
    tensor_tiles = tiles_on_value(src_op, src_sizes, src_shared)
    if tensor_tiles is None:
        return None
    return sizes_from_value_tiles(dst_op, dst_shared, tensor_tiles)


def sizes_from_value_tiles(
    op: ir.Operation | ir.OpView,
    value: ir.Value,
    tiles: Sequence[int],
    reject_reduced: bool = False,
) -> list[int] | None:
    """Tile sizes of `op` induced by the per-dimension `tiles` of `value`.

    Only parallel dims of `op` are tiled; reduction dims stay untiled.

    Args:
        op: Single-result structured linalg op using `value`.
        value: Operand or result of `op` the tiles apply to.
        tiles: One tile per dimension of `value` (0 = whole dim).
        reject_reduced: Fail instead of dropping tiles that land on reduction
            dims (fusing would then recompute the producer for every tile).
    Returns:
        `op`'s tile sizes in its loop order, or None if not possible.
    """
    ov = opview(op)
    maps = indexing_maps(ov)
    value_map = indexing_map_for_value(ov, value)
    if maps is None or value_map is None or len(list(ov.results)) != 1:
        return None
    if len(tiles) != len(value_map.results):
        return None
    parallel = map_dims(maps[-1])
    sizes = [0] * maps[-1].n_dims
    for tensor_dim, expr in enumerate(value_map.results):
        pos = dim_position(expr)
        if pos is None or tiles[tensor_dim] == 0:
            continue
        if pos in parallel:
            sizes[pos] = tiles[tensor_dim]
        elif reject_reduced:
            return None
    return sizes


def _slice_tiles(op: tensor.ExtractSliceOp, tiles: Sequence[int]) -> list[int]:
    """Drop the tiles of the unit dims removed by a rank-reducing slice."""
    res = ir.ShapedType(op.result.type).shape
    kept, r = [], 0
    for d, extent in enumerate(ir.ShapedType(op.source.type).shape):
        if r < len(res) and extent == res[r]:
            kept.append(tiles[d])
            r += 1
    return kept


def _packed_tiles(op: linalg.PackOp, plain: Sequence[int]) -> list[int] | None:
    """Plain -> packed tiles through a pack; only whole blocks are tiled."""
    blocks = pack_inner_blocks(op)
    if blocks is None:
        return None
    outer_plain = []
    for d, t in enumerate(plain):
        block = blocks.get(d)
        if block is not None and t % block:
            return None
        outer_plain.append(t // block if block is not None else t)
    perm = pack_outer_perm(op, len(plain))
    return [outer_plain[p] for p in perm] + [0] * len(blocks)


def _plain_tiles(op: linalg.UnPackOp, packed: Sequence[int]) -> list[int] | None:
    """Packed -> plain tiles through an unpack; only whole blocks are tiled."""
    blocks = pack_inner_blocks(op)
    if blocks is None:
        return None
    rank = len(packed) - len(blocks)
    plain = [0] * rank
    for i, p in enumerate(pack_outer_perm(op, rank)):
        plain[p] = packed[i]
    for k, d in enumerate(op.inner_dims_pos):
        inner = packed[rank + k]
        # Partial blocks: slicing inside every block is not a contiguous plain
        # tile, and a partial-block tile of one block cannot be proven to read
        # a single block by the unpack tiling (the slices turn dynamic).
        if inner not in (0, blocks[d]):
            return None
        plain[d] *= blocks[d]
    return plain


def tiles_through_relayout(
    op: ir.Operation | ir.OpView, tiles: Sequence[int]
) -> list[int] | None:
    """Result tiles of a fusable relayout op given the tiles of its source.

    Supports the ops listed by `fusion_analysis.is_fusable_relayout_op`.

    Args:
        op: Relayout op (rank-reducing full slice, pack or unpack).
        tiles: One tile per dimension of the source (0 = whole dim).
    Returns:
        One tile per dimension of the result, or None if the tiling cannot be
        expressed on the result.
    """
    ov = opview(op)
    if isinstance(ov, tensor.ExtractSliceOp):
        return _slice_tiles(ov, tiles)
    if isinstance(ov, linalg.PackOp):
        return _packed_tiles(ov, tiles)
    if isinstance(ov, linalg.UnPackOp):
        return _plain_tiles(ov, tiles)
    return None
