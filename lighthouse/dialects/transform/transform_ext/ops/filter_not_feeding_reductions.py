from mlir import ir
from mlir.dialects import transform

from lighthouse.dialects.transform.transform_ext.utils.make_filter_handles_op import (
    make_filter_handles_op,
)
from lighthouse.dialects.transform.transform_ext.utils import fusion_analysis as fa
from lighthouse.utils.mlir import opview


def is_not_feeding_reductions(op: ir.Operation | ir.OpView) -> bool:
    """Check whether no result of the op flows into a non-contraction reduction.

    See `fusion_analysis.feeds_reduction`.
    """
    return not any(fa.feeds_reduction(r) for r in opview(op).results)


FilterNotFeedingReductionsOp = make_filter_handles_op(
    "filter_not_feeding_reductions", is_not_feeding_reductions
)


def filter_not_feeding_reductions(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a FilterNotFeedingReductionsOp.

    Keeps ops whose results do not flow into a non-contraction reduction through
    elementwise and fusable relayout ops (see `fusion_analysis.feeds_reduction`).

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the matching ops.
    """
    return FilterNotFeedingReductionsOp(target=target).ops
