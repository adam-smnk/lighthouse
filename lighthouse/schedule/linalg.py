from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from lighthouse.dialects.transform import transform_ext
from lighthouse.dialects.transform.transform_ext.utils.tiling.strategy_register_reduction import (
    ReductionRegisterTiling,
)
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform


def split_inner_reductions(max_rank: int = 4) -> ir.Module:
    """
    Split long inner (row) reductions into a lane-wise partial reduction and
    a short final reduction.

    E.g. a row max over 64x4096 becomes a 64x32x128 -> 64x128 partial
    reduction (128 independent lanes, no horizontal step) followed by a
    64x128 -> 64 reduction. The split factor is target-derived, see
    `ReductionRegisterTiling.split_factor`; non-matching ops are untouched.

    Args:
        max_rank: Maximum reduction output rank to consider.
    Returns:
        Schedule
    """
    # Split attributes are static: enumerate all (output rank, factor) pairs.
    factors = ReductionRegisterTiling.split_factors(TargetInfo.host())
    any_op = transform.AnyOpType.get()
    with schedule_boilerplate() as (schedule, named_seq):
        for out_rank in range(max_rank + 1):
            for factor in factors:
                ops = lh_transform.match_op(
                    named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
                )
                splittable = transform_ext.filter_splittable_reductions(
                    ops, out_rank=out_rank, split_factor=factor
                )
                with lh_transform.foreach(splittable) as op:
                    structured.SplitReductionOp(
                        any_op,
                        any_op,
                        any_op,
                        any_op,
                        op,
                        split_factor=factor,
                        insert_split_dimension=out_rank,
                        inner_parallel=True,
                    )
                    transform.yield_()
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return schedule


def linalg_contract_fold_unit_dims() -> ir.Module:
    """
    Fold unit dims of linalg contract.

    NOTE: The rewrite currently applies linalg morphism and folds all generics.

    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        # TODO: Match only contracts when the folding pattern supports them.
        ops = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ops = transform.apply_registered_pass(
            transform.any_op_t(),
            ops,
            "linalg-morph-ops",
            options={
                "category-to-generic": True,
            },
        )
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            # Works only on generics.
            structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
        transform.apply_registered_pass(
            transform.any_op_t(),
            ops,
            "linalg-morph-ops",
            options={
                "generic-to-category": True,
            },
        )
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule
