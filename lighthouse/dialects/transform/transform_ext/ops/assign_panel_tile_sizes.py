from mlir import ir
from mlir.dialects import ext, linalg, tensor, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import fusion_analysis as fa
from lighthouse.dialects.transform.transform_ext.utils import tile_propagation as tp
from lighthouse.dialects.transform.transform_ext.utils import tile_size_analysis as tsa
from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import (
    dim_position,
    indexing_maps,
    is_linalg_reduction_op,
    linalg_inputs,
    linalg_loop_extents,
    linalg_outputs,
    linalg_reduction_dims,
    op_users,
    opview,
)

# Tile value: one entry per tensor dim, 0 = whole dim.
Tiles = list[int]


def _key(op: ir.Operation | ir.OpView) -> int:
    return opview(op).operation.__hash__()


def _contraction_sizes(op: ir.OpView, rows: int) -> list[int] | None:
    """Tile only the outermost M-like dim of a contraction (one row block)."""
    maps = indexing_maps(op)
    inputs = linalg_inputs(op)
    if maps is None or len(inputs) < 2:
        return None
    lhs_dims = {dim_position(e) for e in maps[0].results}
    rhs_dims = {dim_position(e) for e in maps[1].results}
    out_dims = [dim_position(e) for e in maps[-1].results]
    m_dims = [d for d in out_dims if d in lhs_dims and d not in rhs_dims]
    batch_dims = [d for d in out_dims if d in lhs_dims and d in rhs_dims]
    if not m_dims:
        return None
    sizes = [0] * maps[0].n_dims
    for d in batch_dims:
        sizes[d] = 1
    extents = linalg_loop_extents(op)
    # Packed (blocked) M: one outer block; plain M: `rows` rows.
    sizes[m_dims[0]] = 1 if len(m_dims) > 1 else min(rows, extents[m_dims[0]] or rows)
    return sizes


def _packed_tiles(op: ir.OpView, plain: Tiles) -> Tiles | None:
    """Plain -> packed tiles through a pack (static, no outer permutation)."""
    perm = list(op.outer_dims_perm) if op.outer_dims_perm is not None else []
    if perm != list(range(len(perm))):
        return None
    tiles = dict(zip(op.inner_dims_pos, op.static_inner_tiles))
    outer, inner = [], []
    for d, t in enumerate(plain):
        block = tiles.get(d)
        if block is None or t == 0:
            outer.append(t)
        elif t % block == 0:
            outer.append(t // block)
        else:
            return None
    inner = [0] * len(tiles)
    return outer + inner


def _plain_tiles(op: ir.OpView, packed: Tiles) -> Tiles | None:
    """Packed -> plain tiles through an unpack (static, no outer permutation)."""
    perm = list(op.outer_dims_perm) if op.outer_dims_perm is not None else []
    if perm != list(range(len(perm))):
        return None
    inner_pos = list(op.inner_dims_pos)
    blocks = list(op.static_inner_tiles)
    rank = len(packed) - len(inner_pos)
    plain = list(packed[:rank])
    for k, d in enumerate(inner_pos):
        inner_t = packed[rank + k]
        if plain[d] == 0:
            continue
        if inner_t not in (0, blocks[k]):
            return None
        plain[d] *= blocks[k]
    return plain


def _reshape_tiles(
    src: Tiles, groups: list[list[int]], expand: bool, out_rank: int
) -> Tiles | None:
    """Propagate tiles through expand (src dim -> group) or collapse (group -> dim)."""
    if expand:
        out = [0] * out_rank
        for s, group in enumerate(groups):
            if src[s] == 0:
                continue
            if len(group) != 1:
                return None
            out[group[0]] = src[s]
        return out
    out = []
    for group in groups:
        tiled = [src[d] for d in group if src[d] != 0]
        if len(tiled) > 1:
            return None
        out.append(tiled[0] if tiled else 0)
    return out


def _unit_free_groups(op: ir.OpView, expand: bool) -> list[list[int]]:
    """Reassociation groups without the unit dims of the expanded side."""
    shape = ir.ShapedType((op.result if expand else op.src).type).shape
    groups = [list(ir.ArrayAttr(g)) for g in op.reassociation]
    groups = [[int(ir.IntegerAttr(d).value) for d in g] for g in groups]
    return [[d for d in g if shape[d] != 1] or g[:1] for g in groups]


def _slice_tiles(op: ir.OpView, tiles: Tiles) -> Tiles:
    """Drop the tiles of the unit dims removed by a full rank-reducing slice."""
    src = ir.ShapedType(op.source.type).shape
    res = ir.ShapedType(op.result.type).shape
    kept, r = [], 0
    for d, extent in enumerate(src):
        if r < len(res) and extent == res[r]:
            kept.append(tiles[d])
            r += 1
    return kept


def _result_tiles(
    op: ir.OpView, sizes_or_tiles, from_operand: ir.Value
) -> Tiles | None:
    """Tiles of `op`'s (single) result given the tiles of `from_operand`."""
    if isinstance(op, tensor.ExtractSliceOp):
        return _slice_tiles(op, sizes_or_tiles)
    if isinstance(op, linalg.UnPackOp):
        return _plain_tiles(op, sizes_or_tiles)
    if isinstance(op, linalg.PackOp):
        return _packed_tiles(op, sizes_or_tiles)
    rank = ir.ShapedType(op.result.type).rank
    if isinstance(op, tensor.ExpandShapeOp):
        return _reshape_tiles(sizes_or_tiles, _unit_free_groups(op, True), True, rank)
    if isinstance(op, tensor.CollapseShapeOp):
        return _reshape_tiles(sizes_or_tiles, _unit_free_groups(op, False), False, rank)
    return None


def _loop_sizes(op: ir.OpView, operand: ir.Value, tiles: Tiles) -> list[int] | None:
    """Loop tile sizes of linalg `op` induced by the tiles of one of its inputs.

    Returns None when a tiled dim of `operand` is reduced by `op`: fusing would
    then recompute the whole panel producer for every tile.
    """
    maps = indexing_maps(op)
    idx = next((i for i, v in enumerate(op.operands) if v == operand), None)
    if idx is None or idx >= len(maps):
        return None
    reduced = set(linalg_reduction_dims(op))
    sizes = [0] * maps[0].n_dims
    for tensor_dim, expr in enumerate(maps[idx].results):
        pos = dim_position(expr)
        if pos is None or tiles[tensor_dim] == 0:
            continue
        if pos in reduced:
            return None
        sizes[pos] = tiles[tensor_dim]
    return sizes


def build_panel(
    gemm: ir.OpView, rows: int
) -> tuple[list[ir.Operation], dict[int, list[int]]] | None:
    """Collect the downstream chain of `gemm` and the row-panel tiles of its ops.

    The chain follows elementwise, reduction and relayout users; it is a panel
    only if it contains a non-contraction reduction and every op can be tiled
    by whole row blocks (no reduced dim gets split).

    Returns:
        (members in discovery order, op key -> tile sizes of linalg members)
    """
    sizes = _contraction_sizes(gemm, rows)
    if sizes is None:
        return None
    members: list[ir.Operation] = [gemm.operation]
    op_sizes: dict[int, list[int]] = {_key(gemm): sizes}
    value_tiles: dict[ir.Value, Tiles] = {}
    for result in gemm.results:
        value_tiles[result] = tp.tiles_on_value(gemm, sizes, result)
    has_reduction = False
    worklist = list(gemm.results)
    seen = {_key(gemm)}
    while worklist:
        value = worklist.pop(0)
        for user in op_users(value):
            if _key(user) in seen:
                continue
            ov = opview(user)
            if fa.is_relayout_op(ov):
                tiles = _result_tiles(ov, value_tiles[value], value)
                if tiles is None:
                    return None
                value_tiles[ov.result] = tiles
            elif indexing_maps(ov) is not None and not fa.is_fusion_barrier(ov):
                if linalg_outputs(ov) is None or len(ov.results) != 1:
                    return None
                loop_sizes = _loop_sizes(ov, value, value_tiles[value])
                if loop_sizes is None:
                    return None
                op_sizes[_key(ov)] = loop_sizes
                value_tiles[ov.results[0]] = tp.tiles_on_value(
                    ov, loop_sizes, ov.results[0]
                )
                has_reduction |= is_linalg_reduction_op(ov)
            else:
                continue
            seen.add(_key(user))
            members.append(user)
            worklist.extend(user.results)
    if not has_reduction:
        return None
    return members, op_sizes


def _num_tiles(op: ir.OpView, sizes: list[int]) -> int:
    count = 1
    for extent, size in zip(linalg_loop_extents(op) or [], sizes):
        if size and extent:
            count *= -(-extent // size)
    return count


class AssignPanelTileSizesOp(
    TransformExtensionDialect.Operation, name="assign_panel_tile_sizes"
):
    """
    Fuse a GEMM with a row-reduction epilogue (e.g. softmax, layer norm) per row panel.

    For each contraction whose downstream elementwise/relayout chain reaches a
    non-contraction reduction, the whole chain is tiled by one row block of the
    GEMM with all other dims kept whole. Terminal ops of the chain are annotated
    with their panel tile sizes and become fusion roots; all other members are
    marked as panel members so that they are only fused as producers. This way
    the GEMM is computed once per panel and never recomputed or materialized.

    Skipped when the panel count cannot keep all cores busy.

    Args:
        target: Handle to candidate contraction op(s).
        tile_size: Panel rows for unpacked (plain) GEMMs.
    Return:
        Handle to the annotated panel roots.
    """

    target: ext.Operand[transform.AnyOpType]
    tile_size: ext.Operand[transform.AnyParamType] | None = None
    roots: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "AssignPanelTileSizesOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            rows = tsa.DEFAULT_TILE_SIZE
            if op.tile_size is not None:
                params = state.get_params(op.tile_size)
                if len(params) == 1 and isinstance(params[0], ir.IntegerAttr):
                    rows = params[0].value
            cores = TargetInfo.host().core_count()

            roots = []
            for gemm in state.get_payload_ops(op.target):
                if fa.is_panel_member(gemm):
                    continue
                panel = build_panel(opview(gemm), rows)
                if panel is None:
                    continue
                members, op_sizes = panel
                member_keys = {_key(m) for m in members}
                terminals = [
                    m
                    for m in members
                    if not any(
                        _key(u) in member_keys for r in m.results for u in op_users(r)
                    )
                ]
                if any(_key(t) not in op_sizes for t in terminals):
                    continue
                if any(
                    _num_tiles(opview(t), op_sizes[_key(t)]) < cores for t in terminals
                ):
                    continue
                terminal_keys = {_key(t) for t in terminals}
                for m in members:
                    if _key(m) not in terminal_keys:
                        fa.mark_panel_member(m)
                for t in terminals:
                    tsa.set_tile_sizes_attr(t, op_sizes[_key(t)])
                    roots.append(t)

            results.set_ops(op.roots, roots)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "AssignPanelTileSizesOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def assign_panel_tile_sizes(
    target: ir.Value[transform.AnyOpType],
    tile_size: int | ir.Value | None = None,
) -> ir.Value:
    """
    snake_case wrapper to create an AssignPanelTileSizesOp.

    Args:
        target: Handle to candidate contraction op(s).
        tile_size: Panel rows for unpacked (plain) GEMMs.
    Returns:
        Handle to the annotated panel roots.
    """
    if isinstance(tile_size, int):
        param_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), tile_size)
        tile_size = transform.ParamConstantOp(transform.AnyParamType.get(), param_attr)
    return AssignPanelTileSizesOp(target=target, tile_size=tile_size).roots
