from mlir import ir
from mlir.dialects import transform

from lighthouse.dialects.transform.transform_ext.utils.make_filter_handles_op import (
    make_filter_handles_op,
)
from lighthouse.dialects.transform.transform_ext.utils import fusion_analysis as fa


def is_non_panel_member(op: ir.Operation | ir.OpView) -> bool:
    """Check whether the op is not fused into a GEMM row panel."""
    return not fa.is_panel_member(op)


FilterNonPanelMembersOp = make_filter_handles_op(
    "filter_non_panel_members", is_non_panel_member
)


def filter_non_panel_members(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    """
    snake_case wrapper to create a FilterNonPanelMembersOp.

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the ops that are not fused into a GEMM row panel.
    """
    return FilterNonPanelMembersOp(target=target).ops
