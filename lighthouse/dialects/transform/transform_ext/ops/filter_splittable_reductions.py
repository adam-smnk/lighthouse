from mlir import ir
from mlir.dialects import transform

from lighthouse.dialects.transform.transform_ext.utils.make_filter_handles_op import (
    make_filter_handles_op,
)
from lighthouse.dialects.transform.transform_ext.utils.tiling.strategy_register_reduction import (
    ReductionRegisterTiling,
)
from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import is_linalg_reduction_op, linalg_outputs


def _parse_rank_and_factor(params) -> tuple[int, int] | None:
    if len(params) != 1 or not isinstance(params[0], ir.DenseI64ArrayAttr):
        return None
    values = list(params[0])
    return (values[0], values[1]) if len(values) == 2 else None


def is_splittable_reduction(
    op: ir.Operation | ir.OpView, rank_and_factor: tuple[int, int]
) -> bool:
    """Check whether the op is a reduction of the given output rank and split factor."""
    out_rank, factor = rank_and_factor
    if not is_linalg_reduction_op(op):
        return False
    if ir.ShapedType(linalg_outputs(op)[0].type).rank != out_rank:
        return False
    return ReductionRegisterTiling.split_factor(op, TargetInfo.host()) == factor


FilterSplittableReductionsOp = make_filter_handles_op(
    "filter_splittable_reductions",
    is_splittable_reduction,
    parse_param=_parse_rank_and_factor,
)


def filter_splittable_reductions(
    target: ir.Value[transform.AnyOpType],
    out_rank: int,
    split_factor: int,
) -> ir.Value:
    """
    snake_case wrapper to create a FilterSplittableReductionsOp.

    Keeps reductions whose output rank is `out_rank` and whose host-derived
    split factor (see `ReductionRegisterTiling.split_factor`) is `split_factor`.

    Args:
        target: Handle to target op(s).
        out_rank: Required output rank.
        split_factor: Required split factor.
    Returns:
        Handle to the matching reductions.
    """
    param = transform.ParamConstantOp(
        transform.AnyParamType.get(),
        ir.DenseI64ArrayAttr.get([out_rank, split_factor]),
    )
    return FilterSplittableReductionsOp(target=target, param=param).ops
