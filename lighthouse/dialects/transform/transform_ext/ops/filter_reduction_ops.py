from mlir import ir
from mlir.dialects import transform, linalg

from lighthouse.dialects.transform.transform_ext.utils.make_filter_handles_op import (
    make_filter_handles_op,
)
from lighthouse.dialects.transform.transform_ext.utils.tiling.strategy_register_reduction import (
    ReductionRegisterTiling,
)
from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import is_linalg_reduction_op, linalg_outputs


def _has_reduction_loop(op: ir.OpView) -> bool:
    """Check whether a (generic) op has at least one reduction iterator type."""
    build = ir.AttrBuilder.get("linalg.IteratorTypeEnum")
    parallel = build(linalg.IteratorType.parallel, context=op.context)
    return any(it != parallel for it in op.iterator_types)


def is_reduction_op(op: ir.Operation | ir.OpView) -> bool:
    """Check whether the op is a linalg op with at least one reduction dimension."""
    ov = op.opview if isinstance(op, ir.Operation) else op
    if not hasattr(ov, "iterator_types"):
        return False
    return _has_reduction_loop(ov)


FilterReductionOpsOp = make_filter_handles_op("filter_reduction_ops", is_reduction_op)


def filter_reduction_ops(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    """
    snake_case wrapper to create a FilterReductionOpsOp.

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the reduction-op subset of `target`.
    """
    return FilterReductionOpsOp(target=target).ops


FilterNonContractionReductionsOp = make_filter_handles_op(
    "filter_non_contraction_reductions", is_linalg_reduction_op
)


def filter_non_contraction_reductions(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a FilterNonContractionReductionsOp.

    Keeps single-output linalg reductions that are not contractions,
    convolutions or pooling ops (e.g. softmax max/sum, norm reductions).

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the non-contraction reduction subset of `target`.
    """
    return FilterNonContractionReductionsOp(target=target).ops


def _parse_rank_and_factor(params) -> tuple[int, int] | None:
    if len(params) != 1 or not isinstance(params[0], ir.DenseI64ArrayAttr):
        return None
    values = list(params[0])
    return (values[0], values[1]) if len(values) == 2 else None


def _is_splittable_reduction(
    op: ir.Operation | ir.OpView, rank_and_factor: tuple[int, int]
) -> bool:
    out_rank, factor = rank_and_factor
    if not is_linalg_reduction_op(op):
        return False
    if ir.ShapedType(linalg_outputs(op)[0].type).rank != out_rank:
        return False
    return ReductionRegisterTiling.split_factor(op, TargetInfo.host()) == factor


FilterSplittableReductionsOp = make_filter_handles_op(
    "filter_splittable_reductions",
    _is_splittable_reduction,
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
