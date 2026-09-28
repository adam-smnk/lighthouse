from mlir import ir
from mlir.dialects import linalg

from lighthouse.utils.mlir import (
    defining_op,
    indexing_map_for_value,
    indexing_maps,
    is_full_extract_slice,
    is_linalg_reduction_op,
    linalg_inputs,
    linalg_loop_extents,
    linalg_reduction_dims,
    map_dims,
    op_key,
    op_users,
    opview,
)
from lighthouse.dialects.transform.transform_ext.utils import tile_size_analysis as tsa
from lighthouse.dialects.transform.transform_ext.utils.tiling.common import tile_count

# Attribute used to annotate an op as a new fusion separator.
FUSION_BOUNDARY_ATTR_NAME = "transform_ext.fusion_boundary"

# Attribute marking ops fused as producers into a GEMM row-panel root.
FUSION_PANEL_ATTR_NAME = "transform_ext.fusion_panel"

# Max number of times a fused reduction may be recomputed before it is split out.
MAX_REDUCTION_RECOMPUTE = 8


def is_fusion_boundary(op: ir.Operation | ir.OpView) -> bool:
    """Whether the op is marked as a fusion-group boundary (new group start)."""
    return FUSION_BOUNDARY_ATTR_NAME in opview(op).operation.attributes


def mark_fusion_boundary(op: ir.Operation | ir.OpView) -> None:
    """Mark the op as a fusion-group boundary (the start of a new group)."""
    opview(op).operation.attributes[FUSION_BOUNDARY_ATTR_NAME] = ir.UnitAttr.get()


def clear_fusion_boundary(op: ir.Operation | ir.OpView) -> None:
    """Remove the fusion-group boundary marker from an op, if present."""
    attrs = opview(op).operation.attributes
    if FUSION_BOUNDARY_ATTR_NAME in attrs:
        del attrs[FUSION_BOUNDARY_ATTR_NAME]


def is_panel_member(op: ir.Operation | ir.OpView) -> bool:
    """Whether the op is fused into a GEMM row panel (never a fusion root)."""
    return FUSION_PANEL_ATTR_NAME in opview(op).operation.attributes


def mark_panel_member(op: ir.Operation | ir.OpView) -> None:
    """Mark the op as fused into a GEMM row panel."""
    opview(op).operation.attributes[FUSION_PANEL_ATTR_NAME] = ir.UnitAttr.get()


def clear_panel_member(op: ir.Operation | ir.OpView) -> None:
    """Remove the GEMM row-panel marker from an op, if present."""
    attrs = opview(op).operation.attributes
    if FUSION_PANEL_ATTR_NAME in attrs:
        del attrs[FUSION_PANEL_ATTR_NAME]


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


def is_fusable_relayout_op(op: ir.Operation | ir.OpView) -> bool:
    """Data-movement ops that greedy producer fusion can tile through.

    Pack / unpack implement producer fusion and a whole-tensor rank-reducing
    slice folds into the consumer's slice. Reshapes are not fusable.
    """
    return is_full_extract_slice(op) or isinstance(
        opview(op), (linalg.PackOp, linalg.UnPackOp)
    )


def is_elementwise_like(op: ir.Operation | ir.OpView) -> bool:
    """Structured linalg op without reduction dims that is not a fusion barrier."""
    return (
        indexing_maps(op) is not None
        and not linalg_reduction_dims(op)
        and not is_fusion_barrier(op)
    )


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
        key = op_key(cur)
        if key in visited:
            continue
        visited.add(key)
        if is_fusion_barrier(cur):
            return True
        push_producers(cur)
    return False


def feeds_reduction(value: ir.Value) -> bool:
    """Whether `value` flows into a non-contraction reduction through
    elementwise and fusable relayout ops only.
    """
    stack = [value]
    visited: set = set()
    while stack:
        for user in op_users(stack.pop()):
            key = op_key(user)
            if key in visited:
                continue
            visited.add(key)
            if is_linalg_reduction_op(user):
                return True
            if is_fusable_relayout_op(user) or is_elementwise_like(user):
                stack.extend(user.results)
    return False


def reductions_feeding(value: ir.Value) -> list[ir.Operation]:
    """Reductions computing `value` through elementwise and fusable relayout ops.

    These are the reductions that greedy producer fusion pulls in together with
    the producer of `value`.
    """
    found: list[ir.Operation] = []
    stack = [value]
    visited: set = set()
    while stack:
        producer = defining_op(stack.pop())
        if producer is None:
            continue
        key = op_key(producer)
        if key in visited:
            continue
        visited.add(key)
        if is_linalg_reduction_op(producer):
            found.append(producer)
        elif is_fusable_relayout_op(producer):
            stack.append(opview(producer).operands[0])
        elif is_elementwise_like(producer):
            stack.extend(linalg_inputs(producer))
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
    shared_map = indexing_map_for_value(consumer, shared)
    extents = linalg_loop_extents(consumer)
    if shared_map is None or extents is None:
        return False
    used = map_dims(shared_map)
    unused_sizes = [0 if d in used else s for d, s in enumerate(consumer_sizes)]
    factor = tile_count(extents, unused_sizes)
    if factor is not None and factor <= MAX_REDUCTION_RECOMPUTE:
        return False
    return bool(reductions_feeding(shared))
