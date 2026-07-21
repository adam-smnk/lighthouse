"""Fusion analysis helpers.

Classify linalg ops relative to fusion groups: which ops are barriers that groups
are not fused across, and whether an op sits downstream of such a barrier.
"""

from mlir import ir
from mlir.dialects import linalg

from lighthouse.utils.mlir import opview, defining_op
from lighthouse.dialects.transform.transform_ext.utils import tile_size_analysis as tsa


def is_fusion_barrier(op: ir.Operation | ir.OpView) -> bool:
    """Whether the op acts as a fusion barrier (groups are not fused across it).

    Barriers are:
      * heavy compute ops -- contractions and convolutions / pooling: kept in
        their own fused loop (with elementwise prologue / epilogue) and used as
        tiling anchors rather than propagation targets.
      * pack / unpack ops: layout changes that stay as materialization boundaries.
    """
    ov = opview(op)
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
