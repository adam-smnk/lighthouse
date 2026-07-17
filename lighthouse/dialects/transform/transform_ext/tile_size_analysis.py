"""Shared analysis helpers for tile-size selection and propagation.

These utilities operate on payload linalg ops and are used by the
`assign_tile_sizes`, `propagate_tile_sizes` and `get_tile_sizes` transform ops
to implement a generic "tile and fuse" strategy:

    1. assign target tile sizes for key anchor ops (e.g. matmuls)
    2. propagate the tile sizes to neighbouring elementwise ops
    3. tile and fuse using the assigned sizes (sizes also act as fusion hints)

Tile sizes are stored on payload ops as a discardable
`transform_ext.tile_sizes` array attribute holding one entry per iteration
(loop) dimension of the op, in loop order. A size of zero means "do not tile
this dimension" (e.g. reduction dimensions).
"""

from typing import Optional, Sequence

from mlir import ir
from mlir.dialects import linalg

# Discardable attribute used to annotate payload ops with their target tile
# sizes (one entry per iteration dimension, in loop order).
TILE_SIZES_ATTR_NAME = "transform_ext.tile_sizes"

# Default size used for tiled dimensions when no hint is provided.
DEFAULT_TILE_SIZE = 32

# Number of innermost parallel dimensions tiled with the full tile size.
# Other parallel dimensions are tiled with unit size to keep them sequential.
DEFAULT_PARALLEL_TILE_DIMS = 2


def _opview(op: "ir.Operation | ir.OpView") -> ir.OpView:
    return op.opview if isinstance(op, ir.Operation) else op


def _dim_position(expr: ir.AffineExpr) -> Optional[int]:
    """Return the dimension position of a plain dimension expression.

    Returns None for non-dimension expressions (constants, composite exprs).
    """
    if isinstance(expr, ir.AffineDimExpr):
        return expr.position
    return None


def indexing_maps(op: "ir.Operation | ir.OpView") -> Optional[list[ir.AffineMap]]:
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


def contraction_dims(op: "ir.Operation | ir.OpView"):
    """Return contraction dimensions for genuine GEMM-like ops, else None.

    `linalg.infer_contraction_dimensions` only inspects indexing maps and may
    classify reduction-free elementwise ops (e.g. a broadcasted bias add) as
    contractions. A genuine GEMM has at least one reduction (K) dimension and a
    parallel (M or N) dimension, so we require both here.
    """
    try:
        cd = linalg.infer_contraction_dimensions(_opview(op))
    except (TypeError, ValueError):
        return None
    if cd is None:
        return None
    if len(cd.k) == 0 or (len(cd.m) == 0 and len(cd.n) == 0):
        return None
    return cd


def _single_output_map(maps: Sequence[ir.AffineMap], op: ir.OpView) -> ir.AffineMap:
    """Return the indexing map of the op's single output operand."""
    num_outputs = len(list(op.outputs))
    return maps[len(maps) - num_outputs]


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


def compute_anchor_tile_sizes(
    op: "ir.Operation | ir.OpView",
    tile_size: int = DEFAULT_TILE_SIZE,
    parallel_tile_dims: int = DEFAULT_PARALLEL_TILE_DIMS,
) -> Optional[list[int]]:
    """Compute target tile sizes for an anchor op over its iteration space.

    The strategy aims for 2D tiles over the innermost parallel dimensions:
      * GEMM-like ops: batch dims -> 1, the two contraction parallel dims
        (M, N) -> tile_size, reduction dims (K) -> 0.
      * Other structured ops: the innermost `parallel_tile_dims` output
        (parallel) dims -> tile_size, remaining parallel dims -> 1, and
        reduction dims (not present in the output map) -> 0.

    Statically small parallel dimensions are left untiled.

    Returns one size per iteration dimension (loop order), or None if the op
    is not a supported structured op.
    """
    ov = _opview(op)
    maps = indexing_maps(ov)
    if maps is None:
        return None
    # Only single-output ops are supported for now.
    if len(list(ov.outputs)) != 1:
        return None

    out_map = _single_output_map(maps, ov)
    num_loops = out_map.n_dims
    sizes = [0] * num_loops

    contraction = contraction_dims(ov)
    if contraction is not None:
        # GEMM-like: keep batch sequential, tile the M/N parallel dims, leave
        # the reduction (K) dims untiled so they remain inside the tile.
        for d in contraction.batch:
            sizes[d] = 1
        for d in list(contraction.m) + list(contraction.n):
            sizes[d] = tile_size
    else:
        # Generic case: tile the innermost parallel (output) dims.
        parallel_dims = [
            pos
            for pos in (_dim_position(e) for e in out_map.results)
            if pos is not None
        ]
        if not parallel_dims:
            return None
        inner = parallel_dims[-parallel_tile_dims:]
        outer = parallel_dims[:-parallel_tile_dims]
        for d in outer:
            sizes[d] = 1
        for d in inner:
            sizes[d] = tile_size

    _disable_small_tiles(ov, out_map, sizes, tile_size)
    return sizes


def get_tile_sizes_attr(op: "ir.Operation | ir.OpView") -> Optional[list[int]]:
    """Return the tile sizes annotated on an op, or None if not annotated."""
    attr = _opview(op).operation.attributes
    if TILE_SIZES_ATTR_NAME not in attr:
        return None
    return list(ir.DenseI64ArrayAttr(attr[TILE_SIZES_ATTR_NAME]))


def set_tile_sizes_attr(op: "ir.Operation | ir.OpView", sizes: Sequence[int]) -> None:
    """Annotate an op with its target tile sizes."""
    operation = _opview(op).operation
    operation.attributes[TILE_SIZES_ATTR_NAME] = ir.DenseI64ArrayAttr.get(list(sizes))


def is_propagatable(op: "ir.Operation | ir.OpView") -> bool:
    """Whether tile sizes may be propagated onto this op.

    Anchor ops (contractions) get their sizes from `compute_anchor_tile_sizes`;
    propagation targets the surrounding elementwise / fill ops that can share a
    tiling and be fused.
    """
    ov = _opview(op)
    propagatable_ops = (
        linalg.ElementwiseOp,
        linalg.AddOp,
        linalg.SubOp,
        linalg.MulOp,
        linalg.DivOp,
        linalg.ExpOp,
        linalg.MaxOp,
        linalg.MinOp,
        linalg.FillOp,
        linalg.GenericOp,
        linalg.CopyOp,
        linalg.BroadcastOp,
        linalg.TransposeOp,
        linalg.ReduceOp,
    )
    if not isinstance(ov, propagatable_ops):
        return False
    # Never re-tile a contraction through propagation.
    return contraction_dims(ov) is None


def _map_for_value(
    op: ir.OpView, value: ir.Value, maps: Sequence[ir.AffineMap]
) -> Optional[ir.AffineMap]:
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
    src_op: "ir.Operation | ir.OpView",
    src_sizes: Sequence[int],
    shared: ir.Value,
    dst_op: "ir.Operation | ir.OpView",
) -> Optional[list[int]]:
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
    dst_out_map = _single_output_map(dst_maps, dst)
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


def is_gemm(op: "ir.Operation | ir.OpView") -> bool:
    """Whether the op is a genuine GEMM-like contraction (fusion barrier)."""
    return contraction_dims(op) is not None


def has_gemm_ancestor(op: "ir.Operation | ir.OpView") -> bool:
    """Whether a GEMM is reachable going backward through annotated producers.

    Used to tell an epilogue op (downstream of a GEMM, e.g. a bias/relu after a
    matmul) apart from a pure prologue op (upstream of a GEMM, e.g. a fill).
    Only annotated ops are traversed; the GEMM itself is a barrier.
    """
    visited: set = set()
    stack: list = []

    def push_producers(cur: "ir.Operation | ir.OpView") -> None:
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
        if is_gemm(cur):
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


def defining_op(value: ir.Value) -> Optional[ir.Operation]:
    """Return the op defining `value`, or None for block arguments."""
    owner = value.owner
    if isinstance(owner, ir.OpView):
        return owner.operation
    if isinstance(owner, ir.Operation):
        return owner
    return None
