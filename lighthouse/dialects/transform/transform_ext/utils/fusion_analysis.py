from mlir import ir
from mlir.dialects import linalg, tensor

from lighthouse.utils.mlir import (
    defining_op,
    dim_position,
    indexing_maps,
    is_linalg_reduction_op,
    linalg_inputs,
    linalg_loop_extents,
    linalg_reduction_dims,
    op_users,
    opview,
)
from lighthouse.dialects.transform.transform_ext.utils import tile_size_analysis as tsa

# Attribute used to annotate an op as a new fusion separator.
FUSION_BOUNDARY_ATTR_NAME = "transform_ext.fusion_boundary"

# Attribute marking ops fused as producers into a GEMM row-panel root.
FUSION_PANEL_ATTR_NAME = "transform_ext.fusion_panel"

# Max number of times a fused reduction may be recomputed before it is split out.
MAX_REDUCTION_RECOMPUTE = 8


def is_fusion_boundary(op: ir.Operation | ir.OpView) -> bool:
    """Whether the op is marked as a fusion-group boundary (new group start)."""
    return FUSION_BOUNDARY_ATTR_NAME in opview(op).operation.attributes


def is_panel_member(op: ir.Operation | ir.OpView) -> bool:
    """Whether the op is fused into a GEMM row panel (never a fusion root)."""
    return FUSION_PANEL_ATTR_NAME in opview(op).operation.attributes


def mark_panel_member(op: ir.Operation | ir.OpView) -> None:
    opview(op).operation.attributes[FUSION_PANEL_ATTR_NAME] = ir.UnitAttr.get()


def clear_panel_member(op: ir.Operation | ir.OpView) -> None:
    attrs = opview(op).operation.attributes
    if FUSION_PANEL_ATTR_NAME in attrs:
        del attrs[FUSION_PANEL_ATTR_NAME]


def is_full_extract_slice(op: ir.Operation | ir.OpView) -> bool:
    """Whether `op` is an extract_slice of a whole tensor, i.e. a pure rank reduction."""
    ov = opview(op)
    if not isinstance(ov, tensor.ExtractSliceOp):
        return False
    shape = ir.ShapedType(ov.source.type).shape
    return (
        all(o == 0 for o in ov.static_offsets)
        and all(s == 1 for s in ov.static_strides)
        and list(ov.static_sizes) == list(shape)
    )


def is_relayout_op(op: ir.Operation | ir.OpView) -> bool:
    """Pack/unpack and reshape ops that only move data between layouts."""
    return is_full_extract_slice(op) or isinstance(
        opview(op),
        (linalg.PackOp, linalg.UnPackOp, tensor.ExpandShapeOp, tensor.CollapseShapeOp),
    )


def feeds_reduction(value: ir.Value) -> bool:
    """Whether `value` flows into a non-contraction reduction through
    elementwise and relayout ops only.
    """
    stack = [value]
    visited: set = set()
    while stack:
        for user in op_users(stack.pop()):
            key = user.__hash__()
            if key in visited:
                continue
            visited.add(key)
            if is_linalg_reduction_op(user):
                return True
            if is_relayout_op(user) or (
                indexing_maps(user) is not None
                and not linalg_reduction_dims(user)
                and not is_fusion_barrier(user)
            ):
                stack.extend(user.results)
    return False


def mark_fusion_boundary(op: ir.Operation | ir.OpView) -> None:
    """Mark the op as a fusion-group boundary (the start of a new group)."""
    opview(op).operation.attributes[FUSION_BOUNDARY_ATTR_NAME] = ir.UnitAttr.get()


def clear_fusion_boundary(op: ir.Operation | ir.OpView) -> None:
    """Remove the fusion-group boundary marker from an op, if present."""
    attrs = opview(op).operation.attributes
    if FUSION_BOUNDARY_ATTR_NAME in attrs:
        del attrs[FUSION_BOUNDARY_ATTR_NAME]


def is_fusion_barrier(op: ir.Operation | ir.OpView) -> bool:
    """Check whether the op acts as a fusion barrier (groups are not fused across it).

    Barriers are:
      * heavy compute ops: contractions and convolutions / pooling: kept in
        their own fused loop (with elementwise prologue / epilogue) and used as
        tiling anchors rather than propagation targets.
      * pack / unpack ops: layout changes that stay as materialization boundaries.
    """
    ov = opview(op)
    if isinstance(ov, (linalg.PackOp, linalg.UnPackOp)):
        return True
    return linalg.isa_contraction_op(ov) or linalg.isa_convolution_op(ov)


def has_barrier_ancestor(op: ir.Operation | ir.OpView) -> bool:
    """Check whether a fusion barrier is reachable backward through annotated producers.

    Used to tell an epilogue op (consumer of a barrier, e.g. a bias/relu after
    a matmul) apart from a pure prologue op (producer of a barrier, e.g. a fill).
    Only annotated ops are traversed; the barrier itself is not crossed.
    """
    visited: set = set()
    stack: list = []

    def push_producers(cur: ir.Operation | ir.OpView) -> None:
        for operand in opview(cur).operands:
            producer = defining_op(operand)
            if producer is not None and tsa.get_tile_sizes_attr(producer) is not None:
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


def reductions_feeding(value: ir.Value) -> list[ir.Operation]:
    """Reductions computing `value`, possibly through a chain of elementwise
    ops and reshapes (which greedy producer fusion pulls in too).
    """
    found: list[ir.Operation] = []
    stack = [value]
    visited: set = set()
    while stack:
        producer = defining_op(stack.pop())
        if producer is None:
            continue
        key = producer.__hash__()
        if key in visited:
            continue
        visited.add(key)
        ov = opview(producer)
        if isinstance(ov, (tensor.ExpandShapeOp, tensor.CollapseShapeOp)):
            stack.append(ov.src)
            continue
        if indexing_maps(ov) is None:
            continue
        if is_linalg_reduction_op(ov):
            found.append(producer)
        elif not linalg_reduction_dims(ov):
            stack.extend(linalg_inputs(ov))
    return found


def recomputes_reduction(
    consumer: ir.Operation | ir.OpView,
    consumer_sizes: list[int] | None,
    shared: ir.Value,
) -> bool:
    """Check whether fusing the producer of `shared` into `consumer` recomputes a reduction.

    That is the case when `consumer` tiles a loop dim which does not index
    `shared` (e.g. softmax columns vs. the row max): each such tile would
    recompute the entire producer slice, including a full reduction.
    A few recomputations (up to `MAX_REDUCTION_RECOMPUTE`) are tolerated as
    they hit cache-resident data and are cheaper than materializing.
    """
    if not consumer_sizes:
        return False
    ov = opview(consumer)
    maps = indexing_maps(ov)
    inputs = linalg_inputs(ov)
    extents = linalg_loop_extents(ov)
    if maps is None or inputs is None:
        return False
    shared_maps = [maps[i] for i, v in enumerate(inputs) if v == shared]
    if not shared_maps:
        return False
    used = {dim_position(e) for m in shared_maps for e in m.results}
    factor = 1
    for dim, size in enumerate(consumer_sizes[: len(extents)]):
        if size == 0 or dim in used:
            continue
        if extents[dim] is None:
            factor = MAX_REDUCTION_RECOMPUTE + 1
            break
        factor *= -(-extents[dim] // size)
    if factor <= MAX_REDUCTION_RECOMPUTE:
        return False
    return bool(reductions_feeding(shared))
