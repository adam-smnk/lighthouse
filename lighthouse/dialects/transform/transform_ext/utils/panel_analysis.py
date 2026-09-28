"""Row-panel analysis: fusing a GEMM with its reduction epilogue per row panel."""

from mlir import ir

from lighthouse.dialects.transform.transform_ext.utils import fusion_analysis as fa
from lighthouse.dialects.transform.transform_ext.utils import tile_propagation as tp
from lighthouse.dialects.transform.transform_ext.utils.tiling.common import tile_count
from lighthouse.dialects.transform.transform_ext.utils.tiling.target_caps import (
    panel_bytes,
)
from lighthouse.utils.mlir import (
    dim_position,
    in_program_order,
    indexing_maps,
    is_linalg_reduction_op,
    linalg_inputs,
    linalg_loop_extents,
    linalg_outputs,
    op_key,
    op_users,
    opview,
)

# Smallest panel height; matches the f32 register-level GEMM row tile.
MIN_PANEL_ROWS = 8


def _m_dims(op: ir.OpView) -> tuple[list[int], list[int]] | None:
    """(M-like dims in output order, batch dims) of a contraction."""
    maps = indexing_maps(op)
    inputs = linalg_inputs(op)
    if maps is None or len(inputs) < 2:
        return None
    lhs_dims = {dim_position(e) for e in maps[0].results}
    rhs_dims = {dim_position(e) for e in maps[1].results}
    out_dims = [dim_position(e) for e in maps[-1].results]
    m_dims = [d for d in out_dims if d in lhs_dims and d not in rhs_dims]
    batch_dims = [d for d in out_dims if d in lhs_dims and d in rhs_dims]
    return (m_dims, batch_dims) if m_dims else None


def panel_row_candidates(gemm: ir.OpView, rows: int) -> list[int]:
    """Panel heights to try, largest first.

    Packed M: the M block, then halved sub-blocks. Plain M: `rows`, then halved.
    Heights never go below `MIN_PANEL_ROWS` and always divide M.

    Args:
        gemm: Contraction anchoring the panel.
        rows: Panel height for plain (unpacked) M.
    Returns:
        Candidate panel heights; empty when M is dynamic or not a contraction.
    """
    dims = _m_dims(gemm)
    if dims is None:
        return []
    m_dims, _ = dims
    extents = linalg_loop_extents(gemm)
    extent = extents[m_dims[1]] if len(m_dims) > 1 else extents[m_dims[0]]
    if extent is None:
        return []
    height = extent if len(m_dims) > 1 else min(rows, extent)
    candidates = []
    while height >= MIN_PANEL_ROWS and extent % height == 0:
        candidates.append(height)
        if height % 2:
            break
        height //= 2
    return candidates


def _contraction_sizes(op: ir.OpView, rows: int) -> list[int] | None:
    """Tile the M-like dims of a contraction by one panel of `rows` rows.

    Packed M is tiled by one outer block and, for sub-block panels, part of the
    inner block; all other dims stay whole.
    """
    dims = _m_dims(op)
    if dims is None:
        return None
    m_dims, batch_dims = dims
    sizes = [0] * indexing_maps(op)[0].n_dims
    for d in batch_dims:
        sizes[d] = 1
    extents = linalg_loop_extents(op)
    if len(m_dims) > 1:
        sizes[m_dims[0]] = 1
        if extents[m_dims[1]] is not None and rows < extents[m_dims[1]]:
            sizes[m_dims[1]] = rows
    else:
        sizes[m_dims[0]] = min(rows, extents[m_dims[0]] or rows)
    return sizes


def _is_chain_op(op: ir.Operation | ir.OpView) -> bool:
    """Ops that can join a panel: fusable relayouts and single-result
    non-barrier linalg ops."""
    ov = opview(op)
    if fa.is_fusable_relayout_op(ov):
        return True
    return (
        indexing_maps(ov) is not None
        and not fa.is_fusion_barrier(ov)
        and linalg_outputs(ov) is not None
        and len(ov.results) == 1
    )


def _users_of(op: ir.Operation | ir.OpView) -> list[ir.Operation]:
    return [u for r in opview(op).results for u in op_users(r)]


class Chain:
    """Downstream chain of a GEMM through elementwise, reduction and relayout ops.

    Attributes:
        gemm: The anchoring contraction.
        members: Chain ops in program order, the GEMM first.
        keys: `op_key`s of the members.
        terminals: Members without users inside the chain (the panel roots).
        has_reduction: Whether some member is a non-contraction reduction.
        escapes: Whether an intermediate member is also used outside the chain.
    """

    def __init__(self, gemm: ir.Operation | ir.OpView):
        self.gemm = opview(gemm)
        members = [self.gemm.operation]
        keys = {op_key(gemm)}
        worklist = list(self.gemm.results)
        while worklist:
            for user in op_users(worklist.pop(0)):
                if op_key(user) in keys or not _is_chain_op(user):
                    continue
                keys.add(op_key(user))
                members.append(user)
                worklist.extend(user.results)
        # Trailing relayouts (e.g. the final unpack) stay outside of the panel
        # so that its terminals are linalg ops, which can be fusion roots.
        pruned = True
        while pruned:
            pruned = False
            for m in members[1:]:
                if fa.is_fusable_relayout_op(m) and not any(
                    op_key(u) in keys for u in _users_of(m)
                ):
                    members.remove(m)
                    keys.discard(op_key(m))
                    pruned = True
        self.members = in_program_order(members)
        self.keys = keys
        self.has_reduction = any(is_linalg_reduction_op(m) for m in self.members[1:])
        self.terminals = []
        # An intermediate also used outside the chain would be both fused into
        # the panel and materialized in full, duplicating its work.
        self.escapes = False
        for m in self.members:
            users = _users_of(m)
            inside = [u for u in users if op_key(u) in keys]
            if not inside:
                self.terminals.append(m)
            elif len(inside) != len(users):
                self.escapes = True


def _merge(a: list[int] | None, b: list[int]) -> list[int] | None:
    """Merge two loop tilings; tiled dims must agree (0 = unconstrained)."""
    if a is None:
        return b
    if any(x and y and x != y for x, y in zip(a, b)):
        return None
    return [x or y for x, y in zip(a, b)]


def build_panel(chain: Chain, rows: int) -> dict[int, list[int]] | None:
    """Row-panel tile sizes of the linalg ops of `chain`, keyed by `op_key`.

    Members are visited in program order; each derives its sizes from the
    tiles of all its chain operands, which must agree.

    Args:
        chain: Chain to tile.
        rows: Panel height.
    Returns:
        Loop tile sizes per linalg member, or None if some op cannot be tiled
        by `rows`-row panels, e.g. when a tiled dim is reduced (a reduction
        over M needs all rows).
    """
    gemm = chain.gemm
    sizes = _contraction_sizes(gemm, rows)
    if sizes is None:
        return None
    op_sizes = {op_key(gemm): sizes}
    value_tiles = {r: tp.tiles_on_value(gemm, sizes, r) for r in gemm.results}
    for member in chain.members[1:]:
        ov = opview(member)
        if fa.is_fusable_relayout_op(ov):
            src_tiles = value_tiles.get(ov.operands[0])
            if src_tiles is None:
                return None
            tiles = tp.tiles_through_relayout(ov, src_tiles)
            if tiles is None:
                return None
            value_tiles[ov.results[0]] = tiles
            continue
        sizes = None
        for operand in ov.operands:
            if value_tiles.get(operand) is None:
                continue
            induced = tp.sizes_from_value_tiles(
                ov, operand, value_tiles[operand], reject_reduced=True
            )
            sizes = None if induced is None else _merge(sizes, induced)
            if sizes is None:
                return None
        if sizes is None:
            return None
        op_sizes[op_key(ov)] = sizes
        value_tiles[ov.results[0]] = tp.tiles_on_value(ov, sizes, ov.results[0])
    return op_sizes


def fits_panel(
    chain: Chain, op_sizes: dict[int, list[int]], cores: int, budget: int
) -> bool:
    """Whether every terminal has a tile per core and a panel within `budget` bytes."""
    for terminal in chain.terminals:
        ov = opview(terminal)
        sizes = op_sizes.get(op_key(ov))
        extents = linalg_loop_extents(ov)
        if sizes is None or extents is None:
            return False
        count = tile_count(extents, sizes)
        if count is None or count < cores:
            return False
        elems = 1
        for extent, size in zip(extents, sizes):
            elems *= size or extent or 1
        elem_type = ir.ShapedType(ov.results[0].type).element_type
        if panel_bytes(elem_type, elems) > budget:
            return False
    return True


def split_after_epilogue(chain: Chain) -> None:
    """Materialize the GEMM's elementwise epilogue once when no panel is possible.

    The consumers of the epilogue result start new fusion groups, so the GEMM
    group is tiled on its own first and is not pulled into (and recomputed by)
    each reduction consumer.
    """
    value = chain.gemm.results[0]
    while True:
        users = op_users(value)
        if len(users) != 1:
            break
        user = opview(users[0])
        if not fa.is_elementwise_like(user) or len(user.results) != 1:
            break
        value = user.results[0]
    for user in op_users(value):
        if indexing_maps(user) is not None:
            fa.mark_fusion_boundary(user)
