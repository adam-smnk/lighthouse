from mlir import ir
from mlir.dialects import ext, transform, linalg
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext import tile_size_analysis as tsa


def _all_loops_parallel(op: ir.OpView) -> bool:
    """Whether all iterator types of a (generic) op are parallel."""
    build = ir.AttrBuilder.get("linalg.IteratorTypeEnum")
    parallel = build(linalg.IteratorType.parallel, context=op.context)
    return all(it == parallel for it in op.iterator_types)


def is_elementwise(op: ir.Operation | ir.OpView) -> bool:
    """Whether the op is an elementwise linalg op.

    Mirrors `mlir::linalg::isElementwise`: the op computes its result pointwise.
    All loops are parallel (num loops == num parallel loops), every indexing map
    is a projected permutation (identity, transpose or broadcast of the
    operands), and each output (DPS init) map is a full permutation.
    """
    ov = op.opview if isinstance(op, ir.Operation) else op
    maps = tsa.indexing_maps(ov)
    if maps is None:
        return False
    # No reduction / window loops. Only a generic can declare non-parallel
    # iterators here; other structured ops that pass the map checks below are
    # elementwise by construction (a contraction / reduction / pooling drops a
    # loop dim from its output map and so fails the permutation check).
    if isinstance(ov, linalg.GenericOp) and not _all_loops_parallel(ov):
        return False
    if not all(m.is_projected_permutation for m in maps):
        return False
    num_inputs = len(list(ov.inputs))
    return all(m.is_permutation for m in maps[num_inputs:])


class FilterElementwiseOp(
    TransformExtensionDialect.Operation, name="filter_elementwise"
):
    """
    Returns the target ops that are elementwise linalg ops.

    An op is elementwise when it computes its result pointwise (see
    `is_elementwise`): all loops are parallel, every indexing map is a projected
    permutation, and each output (DPS init) map is a full permutation. Targets
    that are not elementwise linalg ops are dropped.

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the elementwise subset of `target`.
    """

    target: ext.Operand[transform.AnyOpType]
    ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "FilterElementwiseOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            matching_ops = [t for t in targets if is_elementwise(t)]
            results.set_ops(op.ops, matching_ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "FilterElementwiseOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


def filter_elementwise(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    """
    snake_case wrapper to create a FilterElementwiseOp.

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the elementwise subset of `target`.
    """
    return FilterElementwiseOp(target=target).ops
