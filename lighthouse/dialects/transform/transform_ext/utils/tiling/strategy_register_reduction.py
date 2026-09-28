from typing import NamedTuple

from mlir import ir

from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import (
    dim_position,
    has_index_ops,
    indexing_maps,
    is_linalg_reduction_op,
    linalg_inputs,
    linalg_loop_extents,
    linalg_outputs,
    linalg_reduction_dims,
    opview,
)

from .strategy_base import StrategyContext, TilingStrategy
from .common import (
    assign_reduction_tiles,
    largest_divisor,
    parallel_and_reduction_dims,
)
from .target_caps import (
    compute_bits,
    generic_reduction_tiles,
    is_amx_bf16_contraction,
    is_f32_contraction,
    register_info,
)


class _ReductionInfo(NamedTuple):
    op: ir.OpView
    # Loop dim indexing the innermost (contiguous) dim of the primary input.
    vector_dim: int
    extents: list[int | None]
    parallel_dims: list[int]
    reduction_dims: list[int]
    # Whether the vector dim is reduced (horizontal reduction across lanes).
    inner: bool
    lanes: int
    chains: int


class ReductionRegisterTiling:
    """Target-aware register tiling for non-contraction reductions (x86 CPU).

    Reductions are classified by their vector dim, i.e. the loop dim indexing
    the innermost (contiguous) dim of the primary input:
      * inner: vector dim is reduced (e.g. softmax row max/sum), lanes are
        combined by a horizontal reduction,
      * outer: vector dim is parallel (e.g. RMSNorm channel sum), every lane is
        an independent accumulator and no horizontal reduction is needed.

    Tiles target `_ACC_CHAINS` independent vector accumulators to hide the
    latency of the combiner (~4 cycles on 2 ports on recent x86 cores).
    Only static tile-divisible extents are supported; others yield None.
    """

    _ACC_CHAINS = 8
    _OUTER_RED_UNROLL = 2
    # Combiners with a known neutral element, as required by split reduction.
    _SPLITTABLE_COMBINERS = frozenset(
        {
            "arith.addf",
            "arith.mulf",
            "arith.maximumf",
            "arith.minimumf",
            "arith.maxnumf",
            "arith.minnumf",
            "arith.addi",
            "arith.muli",
            "arith.maxsi",
            "arith.minsi",
            "arith.maxui",
            "arith.minui",
        }
    )

    @classmethod
    def vector_lanes(cls, target: TargetInfo | None, elem_type: ir.Type) -> int:
        """SIMD lanes of one register; sub-32bit floats are computed as f32."""
        return max(1, register_info(target).width_bits // compute_bits(elem_type))

    @classmethod
    def acc_chains(cls, target: TargetInfo | None) -> int:
        """Independent accumulators, capped to leave room for loads/temporaries."""
        return max(1, min(cls._ACC_CHAINS, register_info(target).count // 2))

    @classmethod
    def split_factors(cls, target: TargetInfo | None) -> list[int]:
        """All split factors `split_factor` can return, one per compute width."""
        width = register_info(target).width_bits
        chains = cls.acc_chains(target)
        return sorted({max(1, width // bits) * chains for bits in (8, 16, 32, 64)})

    @staticmethod
    def vector_dim(op: ir.OpView) -> int | None:
        """Loop dim indexing the innermost dim of the least broadcast input."""
        maps = indexing_maps(op)
        best_map = None
        for value, amap in zip(linalg_inputs(op), maps):
            if not isinstance(value.type, ir.ShapedType) or not amap.results:
                continue
            if best_map is None or len(amap.results) > len(best_map.results):
                best_map = amap
        if best_map is None:
            return None
        return dim_position(best_map.results[-1])

    @staticmethod
    def _divisor_tile(extent: int | None, limit: int, quantum: int) -> int | None:
        """Largest multiple of `quantum` <= `limit` dividing `extent`.

        Extents not above `limit` are taken whole; None when nothing divides.
        """
        if extent is None:
            return None
        if extent <= limit:
            return extent
        tile = (limit // quantum) * quantum
        while tile >= quantum:
            if extent % tile == 0:
                return tile
            tile -= quantum
        return None

    @classmethod
    def _analyze(
        cls, op: ir.Operation | ir.OpView, target: TargetInfo | None
    ) -> _ReductionInfo | None:
        ov = opview(op)
        if not is_linalg_reduction_op(ov):
            return None
        vector_dim = cls.vector_dim(ov)
        if vector_dim is None:
            return None
        extents = linalg_loop_extents(ov)
        reduction_dims = linalg_reduction_dims(ov)
        elem = ir.ShapedType(linalg_outputs(ov)[0].type).element_type
        return _ReductionInfo(
            op=ov,
            vector_dim=vector_dim,
            extents=extents,
            parallel_dims=[d for d in range(len(extents)) if d not in reduction_dims],
            reduction_dims=reduction_dims,
            inner=vector_dim in reduction_dims,
            lanes=cls.vector_lanes(target, elem),
            chains=cls.acc_chains(target),
        )

    @classmethod
    def _has_splittable_combiner(cls, op: ir.OpView) -> bool:
        """Body yields `combiner(..., acc)` with a known neutral element."""
        if len(op.regions) != 1 or not op.regions[0].blocks:
            return False
        # Splitting changes the iteration space seen by linalg.index.
        if has_index_ops(op):
            return False
        block = op.regions[0].blocks[0]
        terminator = list(block.operations)[-1]
        if len(terminator.operands) != 1:
            return False
        yielded = terminator.operands[0]
        if not isinstance(yielded.owner, (ir.Operation, ir.OpView)):
            return False
        combiner = opview(yielded.owner).operation
        if combiner.name not in cls._SPLITTABLE_COMBINERS:
            return False
        acc = block.arguments[len(block.arguments) - 1]
        return any(operand == acc for operand in combiner.operands)

    @classmethod
    def split_factor(
        cls, op: ir.Operation | ir.OpView, target: TargetInfo | None
    ) -> int | None:
        """Split factor for a long inner reduction, or None if not splittable.

        Splitting turns the per-chunk horizontal reductions into lane-wise
        partial accumulation and a single horizontal reduction at the end.
        """
        info = cls._analyze(op, target)
        if info is None or info.reduction_dims != [info.vector_dim]:
            return None
        if not cls._has_splittable_combiner(info.op):
            return None
        factor = info.lanes * info.chains
        extent = info.extents[info.vector_dim]
        if extent is None or extent <= factor or extent % factor != 0:
            return None
        return factor

    @classmethod
    def parallel_tiles(
        cls, op: ir.Operation | ir.OpView, target: TargetInfo | None
    ) -> list[int] | None:
        """Register tiles of parallel dims; reduction dims are left untiled."""
        info = cls._analyze(op, target)
        if info is None or not info.parallel_dims:
            return None
        extents, lanes, chains = info.extents, info.lanes, info.chains
        sizes = [0] * len(extents)
        for d in info.parallel_dims:
            sizes[d] = 1

        if info.inner:
            red_extent = extents[info.vector_dim]
            if red_extent is None:
                return None
            unroll = max(1, min(chains, red_extent // lanes))
            rows_dim = info.parallel_dims[-1]
            sizes[rows_dim] = largest_divisor(
                extents[rows_dim], max(1, chains // unroll)
            )
            return sizes

        vec_tile = cls._divisor_tile(extents[info.vector_dim], lanes * chains, lanes)
        if vec_tile is None:
            return None
        sizes[info.vector_dim] = vec_tile
        # Spread missing accumulator chains over the next outer parallel dims.
        remaining = max(1, chains // max(1, vec_tile // lanes))
        for d in reversed([p for p in info.parallel_dims if p != info.vector_dim]):
            if remaining <= 1:
                break
            sizes[d] = largest_divisor(extents[d], remaining)
            remaining = max(1, remaining // sizes[d])
        return sizes

    @classmethod
    def reduction_tiles(
        cls, op: ir.Operation | ir.OpView, target: TargetInfo | None
    ) -> list[int] | None:
        """Register tiles of reduction dims; parallel dims are left untiled."""
        info = cls._analyze(op, target)
        if info is None:
            return None
        extents = info.extents
        sizes = [0] * len(extents)
        for d in info.reduction_dims:
            sizes[d] = 1

        if info.inner:
            limit = info.lanes * info.chains
            extent = extents[info.vector_dim]
            if extent is None:
                return None
            # Short enough for one horizontal reduction: keep it whole.
            if extent <= limit:
                sizes[info.vector_dim] = 0
            else:
                tile = cls._divisor_tile(extent, limit, info.lanes)
                if tile is None:
                    return None
                sizes[info.vector_dim] = tile
        else:
            innermost = info.reduction_dims[-1]
            sizes[innermost] = largest_divisor(
                extents[innermost], cls._OUTER_RED_UNROLL
            )
        if not any(sizes):
            return None
        return sizes

    @classmethod
    def unroll_tiles(
        cls, op: ir.Operation | ir.OpView, target: TargetInfo | None
    ) -> list[int] | None:
        """Final unrolled shape: one vector register on the vector dim."""
        info = cls._analyze(op, target)
        if info is None:
            return None
        sizes = [1] * len(info.extents)
        if info.inner:
            # Keep a single wide horizontal reduction instead of a serial chain.
            sizes[info.vector_dim] = 0
        else:
            tile = cls._divisor_tile(
                info.extents[info.vector_dim], info.lanes, info.lanes
            )
            if tile is None:
                return None
            sizes[info.vector_dim] = tile
        return sizes


class RegisterReductionTilingStrategy(TilingStrategy):
    """Register-level tiling of reduction dimensions; target-derived defaults."""

    def compute(
        self, op: ir.Operation | ir.OpView, ctx: StrategyContext
    ) -> list[int] | None:
        out_map = self.output_map(op)
        if out_map is None:
            return None

        sizes = [0] * out_map.n_dims
        _, reduction_dims = parallel_and_reduction_dims(out_map)
        if not reduction_dims:
            return None

        ov = opview(op)
        if is_amx_bf16_contraction(ov, ctx.target):
            red_tiles = [32]
        elif is_f32_contraction(ov):
            red_tiles = [2]
        elif is_linalg_reduction_op(ov):
            return ReductionRegisterTiling.reduction_tiles(ov, ctx.target)
        else:
            red_tiles = generic_reduction_tiles()

        assign_reduction_tiles(reduction_dims, red_tiles, sizes)
        return sizes
