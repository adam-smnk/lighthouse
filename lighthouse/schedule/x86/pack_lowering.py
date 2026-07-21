from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import vector
from mlir.dialects.transform import tensor

from lighthouse.schedule import schedule_boilerplate
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext


def lower_packs(pack_ops):
    """
    Tile and lower pack ops into pad / expand_shape / transpose ops.

    Tiling comes from the tile_size_analysis framework (one inner tile per
    iteration), so it adapts to arbitrary pack dimensionality. The resulting
    transposes are vectorized separately.

    Args:
        pack_ops: Handle to (tile-size annotated) pack operations.
    """
    with lh_transform.foreach(pack_ops) as pack_op:
        tile_sizes = transform_ext.get_tile_sizes(pack_op)
        tiled_pack = structured.TileUsingForallOp(
            pack_op, tile_sizes=tile_sizes
        ).tiled_op
        structured.structured_lower_pack(
            transform.OperationType.get("tensor.pad"),
            transform.OperationType.get("tensor.expand_shape"),
            transform.OperationType.get("linalg.transpose"),
            tiled_pack,
            lower_pad_like_with_insert_slice=False,
        )
        transform.yield_()


def lower_unpacks(unpack_ops):
    """
    Tile and lower unpack ops into empty / transpose / collapse / copy ops.

    Tiling comes from the tile_size_analysis framework (one inner tile per
    iteration), so it adapts to arbitrary unpack dimensionality. The resulting
    transposes / copies are vectorized separately.

    Args:
        unpack_ops: Handle to (tile-size annotated) unpack operations.
    """
    with lh_transform.foreach(unpack_ops) as unpack_op:
        tile_sizes = transform_ext.get_tile_sizes(unpack_op)
        tiled_unpack = structured.TileUsingForallOp(
            unpack_op, tile_sizes=tile_sizes
        ).tiled_op
        structured.structured_lower_unpack(
            transform.OperationType.get("tensor.empty"),
            transform.OperationType.get("linalg.transpose"),
            transform.OperationType.get("tensor.collapse_shape"),
            transform.OperationType.get("tensor.extract_slice"),
            transform.OperationType.get("linalg.copy"),
            tiled_unpack,
            lower_unpad_like_with_extract_slice=True,
        )
        transform.yield_()


def vectorize_innermost(ops):
    """
    Tile each op's leading dims into unit loops and vectorize the innermost dim.

    Rank-agnostic: `get_leading_unit_tile_sizes` yields N-1 unit sizes for an
    N-dim op, so the outer dims become loops and the innermost dim is vectorized
    to a 1-D vector.

    Args:
        ops: Handle to structured linalg ops (e.g. transposes / copies).
    """
    with lh_transform.foreach(ops) as op:
        sizes = transform_ext.get_leading_unit_tile_sizes(op)
        tiled = structured.TileUsingForallOp(op, tile_sizes=sizes).tiled_op
        structured.structured_vectorize(tiled, [])
        transform.yield_()


def lower_packs_unpacks(tile_size: int = 32, leading_batch_dims: int = 0) -> ir.Module:
    """
    Lower pack and unpack ops into hardware-friendly, vectorized shapes.

    Pack / unpack ops are first tiled to a single inner tile per iteration (via
    the tile_size_analysis framework, adapting to any dimensionality) and lowered
    to pad / transpose / copy ops, whose leading dims are then tiled into loops
    and innermost dim vectorized.

    Args:
        tile_size: Retained for pipeline compatibility; the tiling is derived
            from each op's pack structure.
        leading_batch_dims: Retained for pipeline compatibility; batch dims are
            handled by the per-op tiling.
    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        packs = lh_transform.match_op(named_seq.bodyTarget, "linalg.pack")
        lower_packs(transform_ext.assign_tile_sizes(packs))
        lh_transform.cleanup(named_seq.bodyTarget)

        unpacks = lh_transform.match_op(named_seq.bodyTarget, "linalg.unpack")
        lower_unpacks(transform_ext.assign_tile_sizes(unpacks))

        transposes = lh_transform.match_op(named_seq.bodyTarget, "linalg.transpose")
        vectorize_innermost(transposes)
        copies = lh_transform.match_op(named_seq.bodyTarget, "linalg.copy")
        vectorize_innermost(copies)

        # Cleanup.
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            tensor.apply_patterns_tensor_fold_tensor_subset_ops_into_vector_transfers()
            transform.apply_patterns_canonicalization()
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            vector.apply_patterns_vector_cast_away_vector_leading_one_dim()
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule
