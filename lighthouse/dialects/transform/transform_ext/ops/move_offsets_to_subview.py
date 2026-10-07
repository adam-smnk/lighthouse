from mlir import ir
from mlir.dialects import ext, transform, vector, memref, arith, affine
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def get_base(op: ir.OpView) -> ir.Value:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.base


def get_vector_type(op: ir.OpView) -> ir.VectorType:
    if isinstance(op, vector.TransferReadOp):
        return op.vector.type
    elif isinstance(op, vector.TransferWriteOp):
        return op.valueToStore.type
    else:
        raise NotImplementedError("Unsupported op")


def get_indices(op: ir.OpView) -> ir.OpOperandList:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.indices


def get_permutation_map(op: ir.OpView) -> ir.AffineMapAttr:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.permutation_map


def get_in_bounds(op: ir.OpView) -> ir.ArrayAttr:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.in_bounds


def _enclosing_loop_iv(op: ir.OpView) -> ir.Value | None:
    """Induction variable of the innermost ``scf.for`` around ``op``."""
    parent = op.operation.parent
    while parent is not None:
        if parent.name == "scf.for":
            return parent.regions[0].blocks[0].arguments[0]
        if parent.name in ("func.func", "omp.loop_nest"):
            return None
        parent = parent.parent
    return None


def _sum_terms(expr: ir.AffineExpr, map: ir.AffineMap) -> list[int] | None:
    """Operand positions of ``expr`` if it is a sum of distinct dims/symbols."""
    if isinstance(expr, ir.AffineAddExpr):
        lhs, rhs = _sum_terms(expr.lhs, map), _sum_terms(expr.rhs, map)
        if lhs is None or rhs is None:
            return None
        return lhs + rhs
    if isinstance(expr, ir.AffineDimExpr):
        return [expr.position]
    if isinstance(expr, ir.AffineSymbolExpr):
        return [map.n_dims + expr.position]
    return None


def split_loop_offset(
    offset: ir.Value, iv: ir.Value | None
) -> tuple[ir.Value, list[ir.Value]] | None:
    """Split ``affine.apply(iv + rest...)`` into ``iv`` and the other summands.

    The AMX contraction rewrite rebuilds the reduction loop and expects a subview
    offset that is exactly its induction variable.
    """
    if iv is None:
        return None
    if isinstance(offset, ir.OpView):
        offset = offset.result
    owner = offset.owner
    if isinstance(owner, ir.Block):
        return None
    apply = owner.opview if isinstance(owner, ir.Operation) else owner
    if not isinstance(apply, affine.AffineApplyOp):
        return None
    map = apply.map.value
    if len(map.results) != 1:
        return None
    terms = _sum_terms(map.results[0], map)
    operands = list(apply.mapOperands)
    if terms is None or len(set(terms)) != len(terms):
        return None
    loop_terms = [t for t in terms if operands[t] == iv]
    if len(loop_terms) != 1:
        return None
    return iv, [operands[t] for t in terms if t != loop_terms[0]]


def _split_constant(offset: ir.Value) -> tuple[ir.Value, int]:
    """``offset`` as ``rest + constant`` when it is an ``affine.apply`` adding a
    non-negative constant: tiles of one operand then share a subview base."""
    owner = offset.owner
    if isinstance(owner, ir.Block):
        return offset, 0
    apply = owner.opview if isinstance(owner, ir.Operation) else owner
    if not isinstance(apply, affine.AffineApplyOp):
        return offset, 0
    map = apply.map.value
    if len(map.results) != 1 or not isinstance(map.results[0], ir.AffineAddExpr):
        return offset, 0
    expr = map.results[0]
    for constant, rest in ((expr.rhs, expr.lhs), (expr.lhs, expr.rhs)):
        if isinstance(constant, ir.AffineConstantExpr) and constant.value > 0:
            operands = list(apply.mapOperands)
            if isinstance(rest, ir.AffineDimExpr):
                return operands[rest.position], constant.value
            if isinstance(rest, ir.AffineSymbolExpr):
                return operands[map.n_dims + rest.position], constant.value
            rest_map = ir.AffineMap.get(map.n_dims, map.n_symbols, [rest])
            return affine.apply(rest_map, operands), constant.value
    return offset, 0


def _sum_values(values: list[ir.Value]) -> ir.Value:
    if len(values) == 1:
        return values[0]
    expr = ir.AffineSymbolExpr.get(0)
    for i in range(1, len(values)):
        expr = expr + ir.AffineSymbolExpr.get(i)
    return affine.apply(ir.AffineMap.get(0, len(values), [expr]), values)


def _unit_stride_subview(
    base: ir.Value, offsets: list[ir.Value], sizes: list[ir.Value | int]
) -> ir.Value:
    """A unit-stride ``memref.subview`` of ``base``: its strides, any offset."""
    base_type = ir.MemRefType(base.type)
    strides, _ = base_type.get_strides_and_offset()
    unit = [1] * len(sizes)
    dynamic_stride = ir.ShapedType.get_dynamic_stride_or_offset()
    if base_type.has_static_shape and dynamic_stride not in strides:
        return memref.subview(base, offsets, sizes, unit)
    dynamic = ir.ShapedType.get_dynamic_size()
    result_type = ir.MemRefType.get(
        [size if isinstance(size, int) else dynamic for size in sizes],
        base_type.element_type,
        ir.StridedLayoutAttr.get(dynamic_stride, strides),
        base_type.memory_space,
    )
    return memref.subview(base, offsets, sizes, unit, result_type=result_type)


class MoveOffsetsToSubviewOp(
    TransformExtensionDialect.Operation, name="move_offsets_to_subview"
):
    """
    Outlines non-constant indicies from vector reads and writes to subviews.
    A support rewrite to simplify offsets analysis.

    Args:
        target: Handle to target op
    Returns:
        Updated ops with offsets moved to subviews
    """

    target: ext.Operand[transform.AnyOpType]
    updated_op: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    @staticmethod
    def create_subview(target: ir.OpView) -> ir.OpView | None:
        assert isinstance(target, (vector.TransferReadOp, vector.TransferWriteOp)), (
            "Expected vector transfer op"
        )

        base_val = get_base(target)
        assert isinstance(base_val.type, ir.MemRefType), (
            "Expected memref type for base value"
        )

        memref_type: ir.MemRefType = base_val.type

        vec_type: ir.VectorType = get_vector_type(target)

        map_attr = get_permutation_map(target)
        map: ir.AffineMap = map_attr.value
        if map != ir.AffineMap.get_minor_identity(map.n_dims, len(map.results)):
            return None

        # Runtime extents are the base's own: like static ones, they span past
        # the offsets, which only shift the view's origin.
        sizes = [
            memref.dim(base_val, arith.ConstantOp(ir.IndexType.get(), dim).result)
            if memref_type.is_dynamic_dim(dim)
            else size
            for dim, size in enumerate(memref_type.shape)
        ]
        indices = get_indices(target)

        subview_offsets = []
        constant_offsets = []
        zero_offset = arith.ConstantOp(ir.IndexType.get(), 0)
        for offset in indices:
            if not isinstance(offset.owner, arith.ConstantOp):
                rest, constant = _split_constant(offset)
                subview_offsets.append(rest)
                constant_offsets.append(constant)
            else:
                subview_offsets.append(zero_offset)
                constant_offsets.append(None)

        base_subview = _unit_stride_subview(base_val, subview_offsets, sizes)

        iv = _enclosing_loop_iv(target)
        splits = [split_loop_offset(offset, iv) for offset in subview_offsets]
        if any(splits):
            outer_offsets = [
                (_sum_values(split[1]) if split[1] else zero_offset)
                if split
                else offset
                for offset, split in zip(subview_offsets, splits)
            ]
            outer = _unit_stride_subview(base_val, outer_offsets, sizes)
            inner_offsets = [split[0] if split else zero_offset for split in splits]
            base_subview = _unit_stride_subview(outer, inner_offsets, sizes)

        transfer_indices = []
        for offset, constant in zip(indices, constant_offsets):
            if isinstance(offset.owner, arith.ConstantOp):
                transfer_indices.append(offset)
            elif constant:
                transfer_indices.append(
                    arith.ConstantOp(ir.IndexType.get(), constant).result
                )
            else:
                transfer_indices.append(zero_offset)
        transfer_op = (
            vector.transfer_read(
                vec_type,
                base_subview,
                transfer_indices,
                map_attr,
                target.padding,
                get_in_bounds(target),
            ).owner
            if isinstance(target, vector.TransferReadOp)
            else vector.transfer_write(
                None,
                target.valueToStore,
                base_subview,
                transfer_indices,
                map_attr,
                get_in_bounds(target),
            )
        )

        return transfer_op

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "MoveOffsetsToSubviewOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            updated_ops = []

            for target in targets:
                if not isinstance(
                    target, (vector.TransferReadOp, vector.TransferWriteOp)
                ):
                    return DiagnosedSilenceableFailure.SilenceableFailure

                base_val = get_base(target)
                if not isinstance(base_val.type, ir.MemRefType):
                    return DiagnosedSilenceableFailure.SilenceableFailure

                with ir.InsertionPoint(target), target.location:
                    updated_op = MoveOffsetsToSubviewOp.create_subview(target)
                    if updated_op is None:
                        updated_op = target
                    else:
                        rewriter.replace_op(target, updated_op)
                updated_ops.append(updated_op)

            results.set_ops(op.updated_op, updated_ops)

            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "MoveOffsetsToSubviewOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "MoveOffsetsToSubviewOp"):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def move_offsets_to_subview(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create a MoveOffsetsToSubviewOp."""
    op = MoveOffsetsToSubviewOp(target=target)
    return op.updated_op
