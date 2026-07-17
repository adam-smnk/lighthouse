from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext import tile_size_analysis as tsa


class GetFusionRootsOp(TransformExtensionDialect.Operation, name="get_fusion_roots"):
    """
    Select the fusion roots among tile-size annotated ops, with GEMM barriers.

    Given a set of candidate ops, return the roots of each fusable group. A
    group is a GEMM together with its elementwise prologue (producers) and
    epilogue (consumers); GEMMs act as barriers so that consecutive GEMMs are
    not fused together. Tiling a returned root and greedily fusing its producers
    pulls exactly one group into a single tiled loop (when the roots are
    processed top-down, so an already-fused upstream group blocks fusion).

    An annotated op is a fusion root when its result does not feed another
    elementwise op of the same group, i.e.:
      * it has no annotated non-GEMM consumer (it is the end of an elementwise
        chain, only feeding a GEMM barrier or a non-annotated op), and
      * it is not a pure prologue op feeding a GEMM (e.g. a fill): such ops are
        fused as producers of the GEMM's root instead.

    The roots are returned in program (top-down) order.

    Args:
        target: Handle to candidate op(s) (e.g. all linalg ops).
    Return:
        Handle to the fusion roots.
    """

    target: ext.Operand[transform.AnyOpType]
    roots: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    @staticmethod
    def _is_fusion_root(target_op: ir.Operation) -> bool:
        annotated_consumers = [
            user
            for result in target_op.opview.results
            for user in tsa.op_users(result)
            if tsa.get_tile_sizes_attr(user) is not None
        ]
        # End of an elementwise chain: no annotated non-barrier consumer.
        if any(not tsa.is_fusion_barrier(user) for user in annotated_consumers):
            return False
        # A fusion barrier (e.g. a GEMM) with no elementwise epilogue is its own root.
        if tsa.is_fusion_barrier(target_op):
            return True
        # An epilogue op (downstream of a barrier) is the group's terminal root.
        if tsa.has_barrier_ancestor(target_op):
            return True
        # A pure prologue op feeding a barrier (e.g. a fill) is fused as a
        # producer of the barrier's root, not on its own.
        if annotated_consumers:
            return False
        # A terminal op of a barrier-free elementwise group.
        return True

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
                if GetFusionRootsOp._is_fusion_root(target_op):
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
        Handle to the fusion roots (one per fusable group, in program order).
    """
    return GetFusionRootsOp(target=target).roots
