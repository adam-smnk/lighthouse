from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext import tile_size_analysis as tsa


class GetFusionRootsOp(TransformExtensionDialect.Operation, name="get_fusion_roots"):
    """
    Select the fusion roots among tile-size annotated ops.

    Given a set of candidate ops, keep those carrying a `transform_ext.tile_sizes`
    annotation whose result is not consumed by another annotated op. These
    "terminal" ops are the roots of each fusable group: tiling a root and
    greedily fusing its producers pulls the whole annotated group (e.g. a GEMM
    and its elementwise consumers) into a single tiled loop.

    Args:
        target: Handle to candidate op(s) (e.g. all linalg ops).
    Return:
        Handle to the annotated ops that have no annotated consumer.
    """

    target: ext.Operand[transform.AnyOpType]
    roots: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "GetFusionRootsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)

            roots = []
            for target_op in target_ops:
                if tsa.get_tile_sizes_attr(target_op) is None:
                    continue
                has_annotated_consumer = any(
                    tsa.get_tile_sizes_attr(user) is not None
                    for result in target_op.opview.results
                    for user in tsa.op_users(result)
                )
                if not has_annotated_consumer:
                    roots.append(target_op)

            results.set_ops(op.roots, roots)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GetFusionRootsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


def get_fusion_roots(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a GetFusionRootsOp.

    Args:
        target: Handle to candidate op(s).
    Returns:
        Handle to annotated ops with no annotated consumer.
    """
    return GetFusionRootsOp(target=target).roots
