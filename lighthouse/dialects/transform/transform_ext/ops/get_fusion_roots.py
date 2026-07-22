from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import tile_size_analysis as tsa
from lighthouse.dialects.transform.transform_ext.utils import fusion_analysis as fa
from lighthouse.dialects.transform.transform_ext.utils import tile_propagation as tp
from lighthouse.utils.mlir import op_users


class GetFusionRootsOp(TransformExtensionDialect.Operation, name="get_fusion_roots"):
    """
    Select the fusion roots among tile-size annotated ops, with GEMM barriers.

    Given a set of candidate ops, return the root of each fusable group. A group
    is a GEMM together with its elementwise prologue (producers, e.g. a fill) and
    epilogue (consumers, e.g. a bias-add / relu). GEMMs act as barriers, so
    consecutive GEMMs are kept in separate groups instead of being fused
    together.

    The caller tiles each returned root and greedily fuses its producers into the
    tiled loop, pulling exactly one group into a single loop. This relies on the
    roots being processed top-down: once an upstream group has been fused it
    blocks a downstream group from pulling it back in. The roots are therefore
    returned in program (top-down) order (see `_in_program_order`).

    An annotated op is a fusion root when its result does not feed another
    elementwise op of the same group, i.e.:
      * it has no annotated non-GEMM consumer of the same group (it is the end
        of an elementwise chain, only feeding a GEMM barrier or a non-annotated
        op), and
      * it is not a pure prologue op feeding a GEMM (e.g. a fill): such ops are
        fused as producers of the GEMM's root instead.

    A consumer is only in the same group when it tiles the shared tensor the same
    way (see `_same_group_consumer`): consumers marked as boundaries during
    propagation, or whose annotation induces a conflicting tiling on the shared
    tensor (e.g. hand-annotated 32x32 vs 64x64), start a separate group so their
    own tiling is not overridden by fusion.

    Args:
        target: Handle to candidate op(s) (e.g. all linalg ops).
    Return:
        Handle to the fusion roots, one per group, in program (top-down) order.
    """

    target: ext.Operand[transform.AnyOpType]
    roots: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    @staticmethod
    def _same_group_consumer(
        producer: ir.Operation, shared: ir.Value, consumer: ir.Operation
    ) -> bool:
        """Whether `consumer` shares `producer`'s fusion group across `shared`.

        A consumer explicitly marked as a boundary during propagation is a fast
        reject; otherwise the tilings both ops induce on the shared tensor are
        compared, so ops annotated outside the framework (never seen by
        propagation) are handled too. Barriers are classified by the caller.
        """
        if fa.is_fusion_boundary(consumer):
            return False
        return tp.compatible_on_value(
            producer,
            tsa.get_tile_sizes_attr(producer),
            consumer,
            tsa.get_tile_sizes_attr(consumer),
            shared,
        )

    @staticmethod
    def _is_fusion_root(target_op: ir.Operation) -> bool:
        annotated_consumers = [
            (result, user)
            for result in target_op.opview.results
            for user in op_users(result)
            if tsa.get_tile_sizes_attr(user) is not None
        ]
        # Still inside its own elementwise chain: a non-barrier consumer that
        # shares this op's tiling (compatible and not marked as a boundary) is
        # downstream in the same group, so this op is not the group's root. A
        # consumer with a conflicting tiling is a different group and is ignored.
        if any(
            not fa.is_fusion_barrier(user)
            and GetFusionRootsOp._same_group_consumer(target_op, result, user)
            for result, user in annotated_consumers
        ):
            return False
        # A fusion barrier (e.g. a GEMM) with no elementwise epilogue is its own root.
        if fa.is_fusion_barrier(target_op):
            return True
        # An epilogue op (downstream of a barrier) is the group's terminal root.
        if fa.has_barrier_ancestor(target_op):
            return True
        # A pure prologue op feeding a barrier (e.g. a fill) is fused as a
        # producer of the barrier's root, not on its own.
        if any(fa.is_fusion_barrier(user) for _, user in annotated_consumers):
            return False
        # A terminal op of a barrier-free group (or one that only feeds a
        # different group across a boundary) is its own root.
        return True

    @staticmethod
    def _in_program_order(ops: list[ir.Operation]) -> list[ir.Operation]:
        """Return `ops` sorted top-down by their position in the payload IR.

        The order of `get_payload_ops` is not guaranteed to be program order, so
        the roots are ordered explicitly. A pre-order walk of the enclosing
        payload visits ops top-down; only the given `ops` are collected (matched
        by identity) and the walk stops once all of them have been seen, so other
        payload ops are ignored.

        Note: this is a workaround for `DominanceInfo` not being exposed in the
        MLIR Python bindings. Ordering by dominance would be the natural choice;
        it is only a partial order in general, but that is sufficient here since
        a fusable group never spans control flow -- all roots share a single
        block, where dominance and program order coincide. The pre-order walk
        produces that same order directly.
        """
        if not ops:
            return ops
        remaining = {o.operation.__hash__(): o for o in ops}
        top = ops[0]
        while top.parent is not None:
            top = top.parent

        ordered: list[ir.Operation] = []

        def collect(visited: ir.Operation) -> ir.WalkResult:
            found = remaining.pop(visited.operation.__hash__(), None)
            if found is not None:
                ordered.append(found)
                if not remaining:
                    return ir.WalkResult.INTERRUPT
            return ir.WalkResult.ADVANCE

        top.walk(collect, ir.WalkOrder.PRE_ORDER)
        return ordered

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

            # Order roots to allow easy use of greedy producer fusion.
            roots = GetFusionRootsOp._in_program_order(roots)

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
