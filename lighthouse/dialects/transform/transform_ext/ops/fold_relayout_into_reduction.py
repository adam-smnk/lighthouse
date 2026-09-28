from mlir import ir
from mlir.dialects import ext, linalg, tensor, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.utils.mlir import (
    defining_op,
    dim_position,
    has_index_ops,
    indexing_maps,
    is_linalg_reduction_op,
    linalg_inputs,
    linalg_reduction_dims,
    opview,
    pack_inner_blocks,
    pack_outer_perm,
)


def _identity_blocks(op: ir.OpView, plain: ir.Value) -> dict[int, int] | None:
    """Plain dim -> block of a static pack/unpack without outer permutation.

    None also when a blocked plain dim is not a multiple of its block, i.e.
    the packed layout holds padding.
    """
    shape = ir.ShapedType(plain.type).shape
    if pack_outer_perm(op, len(shape)) != list(range(len(shape))):
        return None
    blocks = pack_inner_blocks(op)
    if blocks is None or any(
        ir.ShapedType.is_dynamic_size(shape[d]) or shape[d] % b
        for d, b in blocks.items()
    ):
        return None
    return blocks


def fold_relayout(red: ir.OpView) -> bool:
    """Rewrite `reduction(pack_P(unpack_U(x)))` to reduce `x` directly.

    Data-layout propagation can repack an already packed tensor with a
    different blocking only to feed a reduction (e.g. a softmax row max after a
    packed matmul). Iterating over the original packed layout instead avoids
    both relayout copies. Supported when every plain dim is either blocked
    identically by U and P, not blocked at all, or blocked by U only and
    reduced (the reduced dim splits into U's outer and inner block dims).

    Returns:
        True if `red` was rewritten in place.
    """
    if not is_linalg_reduction_op(red) or not isinstance(red, linalg.GenericOp):
        return False
    inputs = linalg_inputs(red)
    if len(inputs) != 1 or has_index_ops(red):
        return False
    pack = defining_op(inputs[0])
    if pack is None or not isinstance(opview(pack), linalg.PackOp):
        return False
    pack = opview(pack)
    unpack = defining_op(pack.source)
    if unpack is None or not isinstance(opview(unpack), linalg.UnPackOp):
        return False
    unpack = opview(unpack)
    if pack.padding_value is not None:
        return False
    p_tiles = _identity_blocks(pack, pack.source)
    u_tiles = _identity_blocks(unpack, unpack.result)
    if p_tiles is None or u_tiles is None:
        return False

    maps = indexing_maps(red)
    in_map, out_map = maps
    n_loops = in_map.n_dims
    if not in_map.is_permutation or list(
        dim_position(e) for e in in_map.results
    ) != list(range(n_loops)):
        return False

    plain_rank = ir.ShapedType(unpack.result.type).rank
    reduction_dims = set(linalg_reduction_dims(red))
    p_inner = list(pack.inner_dims_pos)
    # Loop dims of `red` follow P's layout: plain (outer) dims, then P's inner dims.
    outer_loop = {d: d for d in range(plain_rank)}
    inner_loop = {d: plain_rank + k for k, d in enumerate(p_inner)}

    new_types: list[str] = []
    old_to_new: dict[int, int] = {}
    iter_types = [str(t) for t in red.iterator_types]

    def add(old: int) -> int:
        new_types.append(iter_types[old])
        return len(new_types) - 1

    # Unpack source layout: plain outer dims, then U's inner dims.
    split_inner: dict[int, int] = {}
    for d in range(plain_rank):
        pt, ut = p_tiles.get(d), u_tiles.get(d)
        if pt is not None and pt != ut:
            return False
        if ut is not None and pt is None and outer_loop[d] not in reduction_dims:
            return False
        old_to_new[outer_loop[d]] = add(outer_loop[d])
    for d in unpack.inner_dims_pos:
        if d in p_tiles:
            old_to_new[inner_loop[d]] = add(inner_loop[d])
        else:
            split_inner[d] = add(outer_loop[d])

    if any(dim_position(e) is None for e in out_map.results):
        return False
    out_exprs = [
        ir.AffineDimExpr.get(old_to_new[dim_position(e)]) for e in out_map.results
    ]
    n_new = len(new_types)
    new_out = ir.AffineMap.get(n_new, 0, out_exprs)

    # Read through an expand_shape that only adds unit dims (keep them as loops).
    source = unpack.source
    in_exprs = [ir.AffineDimExpr.get(d) for d in range(n_new)]
    expand = defining_op(source)
    if expand is not None and isinstance(opview(expand), tensor.ExpandShapeOp):
        expand = opview(expand)
        shape = ir.ShapedType(expand.result.type).shape
        kept = []
        for group in expand.reassociation:
            dims = [int(ir.IntegerAttr(d).value) for d in ir.ArrayAttr(group)]
            main = [d for d in dims if shape[d] != 1] or dims[:1]
            if len(main) != 1:
                kept = None
                break
            kept.append(main[0])
        if kept is not None:
            source = expand.src
            in_exprs = [in_exprs[d] for d in kept]
    new_in = ir.AffineMap.get(n_new, 0, in_exprs)

    red.operation.operands[0] = source
    red.operation.attributes["indexing_maps"] = ir.ArrayAttr.get(
        [ir.AffineMapAttr.get(new_in), ir.AffineMapAttr.get(new_out)]
    )
    red.operation.attributes["iterator_types"] = ir.ArrayAttr.get(
        [ir.Attribute.parse(t) for t in new_types]
    )
    return True


class FoldRelayoutIntoReductionOp(
    TransformExtensionDialect.Operation, name="fold_relayout_into_reduction"
):
    """
    Let reductions read through pack(unpack(x)) relayouts, see `fold_relayout`.

    Walks the target ops' subtrees and rewrites matching reductions in place.
    The then-dead relayout ops are left for cleanup to remove.

    Args:
        target: Handle to container op(s) (e.g. functions).
    Return:
        Pass-through handle to the target ops.
    """

    target: ext.Operand[transform.AnyOpType]
    transformed: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "FoldRelayoutIntoReductionOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = list(state.get_payload_ops(op.target))
            candidates: list[ir.OpView] = []

            def collect(visited: ir.Operation) -> ir.WalkResult:
                if isinstance(visited.opview, linalg.GenericOp):
                    candidates.append(visited.opview)
                return ir.WalkResult.ADVANCE

            for target_op in target_ops:
                target_op.walk(collect, ir.WalkOrder.PRE_ORDER)
            for candidate in candidates:
                fold_relayout(candidate)

            results.set_ops(op.transformed, target_ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "FoldRelayoutIntoReductionOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def fold_relayout_into_reduction(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    """
    snake_case wrapper to create a FoldRelayoutIntoReductionOp.

    Args:
        target: Handle to container op(s).
    Returns:
        Pass-through handle to the target ops.
    """
    return FoldRelayoutIntoReductionOp(target=target).transformed
