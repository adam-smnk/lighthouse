"""Shared analysis helpers for tile-size selection and propagation.

These utilities operate on linalg ops and are used to implement a generic
"tile and fuse" strategy:

    1. assign target tile sizes for key anchor ops (e.g. matmuls)
    2. propagate the tile sizes to neighbouring elementwise ops
    3. tile and fuse using the assigned sizes (sizes also act as fusion hints)

Tile sizes are stored on payload ops as a discardable attribute.
"""

from collections.abc import Sequence

from mlir import ir
from mlir.dialects import linalg

# Attribute used to annotate payload ops with their target tile sizes
# (one entry per iteration dimension, in loop order).
TILE_SIZES_ATTR_NAME = "transform_ext.tile_sizes"

# Default size used for tiled dimensions when no hint is provided.
DEFAULT_TILE_SIZE = 32

# Number of innermost parallel dimensions tiled with the full tile size.
# Other parallel dimensions are tiled with unit size to keep them sequential.
DEFAULT_PARALLEL_TILE_DIMS = 2


def _opview(op: ir.Operation | ir.OpView) -> ir.OpView:
    return op.opview if isinstance(op, ir.Operation) else op


def _dim_position(expr: ir.AffineExpr) -> int | None:
    """Return the dimension position of a plain dimension expression.

    Returns None for non-dimension expressions (constants, composite exprs).
    """
    if isinstance(expr, ir.AffineDimExpr):
        return expr.position
    return None


def indexing_maps(op: ir.Operation | ir.OpView) -> list[ir.AffineMap] | None:
    """Return the indexing maps of a structured linalg op as ``AffineMap``s.

    The returned list follows the operand order: inputs first, then outputs.
    Returns None if the op is not a structured linalg op.
    """
    try:
        raw_maps = linalg.get_indexing_maps(_opview(op))
    except (TypeError, ValueError):
        return None
    if not raw_maps:
        return None

    maps = []
    for m in raw_maps:
        maps.append(m.value if isinstance(m, ir.AffineMapAttr) else m)
    return maps


def contraction_dims(
    op: ir.Operation | ir.OpView,
) -> linalg.ContractionDimensions | None:
    """Return inferred contraction dimensions."""
    ov = _opview(op)
    try:
        if not linalg.isa_contraction_op(ov):
            return None
        return linalg.infer_contraction_dimensions(ov)
    except (TypeError, ValueError):
        return None


def _output_tensor_dim_of_iter_dim(out_map: ir.AffineMap) -> dict[int, int]:
    """Map each iteration dim to the output tensor dim it indexes (if any)."""
    mapping: dict[int, int] = {}
    for tensor_dim, expr in enumerate(out_map.results):
        pos = _dim_position(expr)
        if pos is not None:
            mapping[pos] = tensor_dim
    return mapping


def _disable_small_tiles(
    op: ir.OpView,
    out_map: ir.AffineMap,
    sizes: list[int],
    tile_size: int,
) -> None:
    """Disable tiling for parallel dims whose static extent is below tile_size.

    Tiling a dimension that is smaller than the tile size only adds loop
    overhead, so such dimensions are left untiled (size 0).
    """
    out_type = ir.ShapedType(list(op.outputs)[0].type)
    iter_to_tensor = _output_tensor_dim_of_iter_dim(out_map)
    for iter_dim, tensor_dim in iter_to_tensor.items():
        if sizes[iter_dim] <= 1:
            # Unit / untiled dimensions are always fine.
            continue
        dim = out_type.shape[tensor_dim]
        if ir.ShapedType.is_static_size(dim) and dim < tile_size:
            sizes[iter_dim] = 0


def compute_tile_sizes(
    op: ir.Operation | ir.OpView,
    tile_size: int = DEFAULT_TILE_SIZE,
    parallel_tile_dims: int = DEFAULT_PARALLEL_TILE_DIMS,
) -> list[int] | None:
    """Compute target tile sizes for an op over its iteration space.

    Selecting which ops to tile (the anchors) is an external decision; this
    routine only computes sizes for whatever op it is given.

    The innermost `parallel_tile_dims` parallel (output) dimensions are tiled
    with `tile_size`; any remaining parallel dimensions (e.g. batch dims or the
    outer M/N dims of a high-dimensional contraction) are tiled with a unit
    size; and reduction dimensions are left untiled.

    Statically sized small parallel dimensions are left untiled.

    Returns one size per iteration dimension (loop order), or None if the op
    is not supported.
    """
    ov = _opview(op)
    maps = indexing_maps(ov)
    if maps is None:
        return None
    # Only single-output ops are supported for now.
    if len(list(ov.outputs)) != 1:
        return None

    # The output operand's map is the last one (inputs first, then outputs).
    out_map = maps[-1]
    sizes = [0] * out_map.n_dims

    # Tile the innermost parallel dims with the full tile size and
    # any outer parallel dims (batch, outer M/N) with a unit size.
    parallel_dims = [
        pos for pos in (_dim_position(e) for e in out_map.results) if pos is not None
    ]
    if not parallel_dims:
        return None
    for d in parallel_dims[:-parallel_tile_dims]:
        sizes[d] = 1
    for d in parallel_dims[-parallel_tile_dims:]:
        sizes[d] = tile_size

    _disable_small_tiles(ov, out_map, sizes, tile_size)
    return sizes


def get_tile_sizes_attr(op: ir.Operation | ir.OpView) -> list[int] | None:
    """Return the tile sizes annotated on an op, or None if not annotated."""
    attr = _opview(op).operation.attributes
    if TILE_SIZES_ATTR_NAME not in attr:
        return None
    return list(ir.DenseI64ArrayAttr(attr[TILE_SIZES_ATTR_NAME]))


def set_tile_sizes_attr(op: ir.Operation | ir.OpView, sizes: Sequence[int]) -> None:
    """Annotate an op with its target tile sizes."""
    operation = _opview(op).operation
    operation.attributes[TILE_SIZES_ATTR_NAME] = ir.DenseI64ArrayAttr.get(list(sizes))


def is_propagatable(op: ir.Operation | ir.OpView) -> bool:
    """Whether tile sizes may be propagated onto this op.

    Anchor ops (fusion barriers) define tile sizes; propagation targets the
    surrounding structured linalg ops (elementwise / fill / broadcast / reduce
    ...) that can share their tiling and be fused. Any structured linalg op that
    is not a fusion barrier is a valid propagation target -- barriers are the
    heavy compute ops (contractions, convolutions / pooling) that stay as anchors
    (see `is_fusion_barrier`). Non-linalg ops have no indexing maps to translate
    tiles through and are excluded.
    """
    return indexing_maps(op) is not None and not is_fusion_barrier(op)


def _map_for_value(
    op: ir.OpView, value: ir.Value, maps: Sequence[ir.AffineMap]
) -> ir.AffineMap | None:
    """Return the indexing map associated with `value` on `op`.

    `value` may be an input/output operand or a result of `op`.
    """
    inputs = list(op.inputs)
    outputs = list(op.outputs)
    for i, operand in enumerate(inputs):
        if operand == value:
            return maps[i]
    for k, operand in enumerate(outputs):
        if operand == value:
            return maps[len(inputs) + k]
    # A result corresponds (positionally) to an output operand of a DPS op.
    for r, result in enumerate(op.results):
        if result == value:
            return maps[len(inputs) + r]
    return None


def propagate_through_value(
    src_op: ir.Operation | ir.OpView,
    src_sizes: Sequence[int],
    shared: ir.Value,
    dst_op: ir.Operation | ir.OpView,
) -> list[int] | None:
    """Propagate tile sizes from `src_op` to `dst_op` via a shared tensor.

    The shared tensor `shared` is produced/consumed by both ops. Its
    per-dimension tile sizes are derived from `src_op`'s iteration-space sizes
    and then mapped onto `dst_op`'s iteration space. Reduction dimensions of
    `dst_op` are never tiled.

    Returns the tile sizes for `dst_op` (loop order), or None if propagation is
    not possible.
    """
    src = _opview(src_op)
    dst = _opview(dst_op)

    src_maps = indexing_maps(src)
    dst_maps = indexing_maps(dst)
    if src_maps is None or dst_maps is None:
        return None

    src_map = _map_for_value(src, shared, src_maps)
    dst_map = _map_for_value(dst, shared, dst_maps)
    if src_map is None or dst_map is None:
        return None

    # Tile size per dimension of the shared tensor.
    shaped = ir.ShapedType(shared.type)
    tensor_tiles = [0] * shaped.rank
    for tensor_dim, expr in enumerate(src_map.results):
        pos = _dim_position(expr)
        if pos is not None and pos < len(src_sizes):
            tensor_tiles[tensor_dim] = src_sizes[pos]

    if len(list(dst.outputs)) != 1:
        return None
    dst_out_map = dst_maps[-1]
    dst_parallel = {
        pos
        for pos in (_dim_position(e) for e in dst_out_map.results)
        if pos is not None
    }

    dst_sizes = [0] * dst_out_map.n_dims
    for tensor_dim, expr in enumerate(dst_map.results):
        pos = _dim_position(expr)
        if pos is None:
            continue
        # Only tile parallel dims of the consumer; leave reductions untiled.
        if pos in dst_parallel:
            dst_sizes[pos] = tensor_tiles[tensor_dim]
    return dst_sizes


def is_fusion_barrier(op: ir.Operation | ir.OpView) -> bool:
    """Whether the op acts as a fusion barrier.

    Fusion groups are not fused across a barrier. The barriers are:
      * Heavy compute ops -- contractions and convolutions / pooling: on average
        it is more profitable to keep each in its own fused loop with its
        elementwise prologue and epilogue than to fuse several together. They
        also act as tiling anchors rather than propagation targets.
      * pack / unpack ops: layout changes that stay as materialization
        boundaries; they are neither propagation anchors nor fused into a group.
    """
    ov = _opview(op)
    if isinstance(ov, (linalg.PackOp, linalg.UnPackOp)):
        return True
    return linalg.isa_contraction_op(ov) or linalg.isa_convolution_op(ov)


def has_barrier_ancestor(op: ir.Operation | ir.OpView) -> bool:
    """Whether a fusion barrier is reachable backward through annotated producers.

    Used to tell an epilogue op (consumer of a barrier, e.g. a bias/relu after
    a matmul) apart from a pure prologue op (producer of a barrier, e.g. a fill).
    Only annotated ops are traversed; the barrier itself is not crossed.
    """
    visited: set = set()
    stack: list = []

    def push_producers(cur: ir.Operation | ir.OpView) -> None:
        for operand in _opview(cur).operands:
            producer = defining_op(operand)
            if producer is not None and get_tile_sizes_attr(producer) is not None:
                stack.append(producer)

    push_producers(op)
    while stack:
        cur = stack.pop()
        key = cur.operation.__hash__()
        if key in visited:
            continue
        visited.add(key)
        if is_fusion_barrier(cur):
            return True
        push_producers(cur)
    return False


def op_users(value: ir.Value) -> list[ir.Operation]:
    """Return the ops that use `value`."""
    users = []
    for use in value.uses:
        owner = use.owner
        if isinstance(owner, ir.OpView):
            users.append(owner.operation)
        elif isinstance(owner, ir.Operation):
            users.append(owner)
    return users


def defining_op(value: ir.Value) -> ir.Operation | None:
    """Return the op defining `value`, or None for block arguments."""
    owner = value.owner
    if isinstance(owner, ir.OpView):
        return owner.operation
    if isinstance(owner, ir.Operation):
        return owner
    return None
