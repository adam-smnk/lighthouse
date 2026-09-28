from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import tile_size_analysis as tsa
from lighthouse.dialects.transform.transform_ext.utils import fusion_analysis as fa
from lighthouse.dialects.transform.transform_ext.utils import tile_propagation as tp
from lighthouse.utils.mlir import (
    defining_op,
    in_program_order,
    indexing_maps,
    linalg_inputs,
    op_key,
    op_users,
)


class GetFusionRootsOp(TransformExtensionDialect.Operation, name="get_fusion_roots"):
    """
    Select the fusion root of each group among tile-size annotated ops.

    A group is a barrier op (e.g. a GEMM / pack) with its elementwise prologue
    (producers, e.g. a fill) and epilogue (consumers, e.g. a bias-add / relu).

    A consumer only joins the group when it tiles the shared tensor the same way.
    An op whose annotation conflicts (e.g. 32x32 vs 64x64) or that was marked explicitly
    as a boundary starts its own group.

    The root is a group's downstream terminal: an annotated op with no same-group
    non-barrier consumer that is not itself a pure prologue feeding a barrier.

    It is assumed that the caller tiles each root and greedily fuses its producers.
    Roots are returned in program (top-down) order to help with greedy fusion. That is
    fusing an upstream group first blocks a downstream one from pulling it back in.

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
        """Check whether `consumer` shares `producer`'s fusion group across `shared` tensor."""
        if fa.is_fusion_boundary(consumer):
            return False
        consumer_sizes = tsa.get_tile_sizes_attr(consumer)
        # Fusing would recompute a reduction for every consumer tile.
        if fa.recomputes_reduction(consumer, consumer_sizes, shared):
            return False
        return tp.compatible_on_value(
            producer,
            tsa.get_tile_sizes_attr(producer),
            consumer,
            consumer_sizes,
            shared,
        )

    @staticmethod
    def _is_fusion_root(target_op: ir.Operation) -> bool:
        # Panel members are fused as producers of the panel's terminal root.
        if fa.is_panel_member(target_op):
            return False
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
    def _recomputed_reductions(root: ir.Operation) -> list[ir.Operation]:
        """Annotated reductions that fusing producers into `root` would recompute.

        Mimics greedy producer fusion: `root`'s tiles are propagated to its linalg
        producers and a reduction is reported when some fused op splits a loop dim
        that does not index the reduction-derived operand.
        """
        found: list[ir.Operation] = []
        stack = [(root, tsa.get_tile_sizes_attr(root))]
        visited: set = set()
        while stack:
            op, sizes = stack.pop()
            key = op_key(op)
            if key in visited or not sizes:
                continue
            visited.add(key)
            for operand in linalg_inputs(op) or []:
                producer = defining_op(operand)
                if producer is None or indexing_maps(producer) is None:
                    continue
                if fa.is_fusion_barrier(producer):
                    continue
                if fa.recomputes_reduction(op, sizes, operand):
                    found.extend(
                        r
                        for r in fa.reductions_feeding(operand)
                        if tsa.get_tile_sizes_attr(r) is not None
                        and not fa.is_panel_member(r)
                    )
                    continue
                producer_sizes = tp.propagate_through_value(
                    op, sizes, operand, producer
                )
                stack.append((producer, producer_sizes))
        return found

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

            # Tile reductions on their own (before their consumers) when fusing
            # them into a consumer would recompute them for every tile.
            seen = {op_key(r) for r in roots}
            pending = list(roots)
            while pending:
                for red in GetFusionRootsOp._recomputed_reductions(pending.pop()):
                    if op_key(red) not in seen:
                        seen.add(op_key(red))
                        roots.append(red)
                        pending.append(red)

            # Order roots to allow easy use of greedy producer fusion.
            roots = in_program_order(roots)

            results.set_ops(op.roots, roots)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GetFusionRootsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


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
