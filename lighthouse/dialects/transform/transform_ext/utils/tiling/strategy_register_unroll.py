from mlir import ir

from lighthouse.utils.mlir import is_linalg_reduction_op, opview

from .strategy_base import StrategyContext, TilingStrategy
from .strategy_register_reduction import ReductionRegisterTiling
from .common import (
    assign_parallel_tiles,
    assign_reduction_tiles,
    parallel_and_reduction_dims,
)
from .target_caps import (
    generic_parallel_tiles,
    generic_reduction_tiles,
    is_amx_bf16_contraction,
    is_f32_contraction,
)


class RegisterUnrollTilingStrategy(TilingStrategy):
    """Register-level unroll-friendly tiling; target-derived defaults."""

    def compute(
        self, op: ir.Operation | ir.OpView, ctx: StrategyContext
    ) -> list[int] | None:
        out_map = self.output_map(op)
        if out_map is None:
            return None

        ov = opview(op)
        # Reductions may have no parallel dims left (e.g. a rank-1 row reduction).
        if is_linalg_reduction_op(ov):
            return ReductionRegisterTiling.unroll_tiles(ov, ctx.target)

        sizes = [0] * out_map.n_dims
        parallel_dims, reduction_dims = parallel_and_reduction_dims(out_map)
        if not parallel_dims:
            return None

        if is_amx_bf16_contraction(ov, ctx.target):
            par_tiles = [16, 16]
            red_tiles = [32]
        elif is_f32_contraction(ov):
            par_tiles = [1, 16]
            red_tiles = [1]
        else:
            par_tiles = generic_parallel_tiles(ov, out_map, ctx.target)
            red_tiles = generic_reduction_tiles()

        assign_parallel_tiles(parallel_dims, par_tiles, sizes)
        assign_reduction_tiles(reduction_dims, red_tiles, sizes)
        return sizes
