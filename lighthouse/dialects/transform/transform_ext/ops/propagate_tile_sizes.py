from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext import tile_size_analysis as tsa


class PropagateTileSizesOp(
    TransformExtensionDialect.Operation, name="propagate_tile_sizes"
):
    """
    Propagate tile-size annotations from anchor ops to neighbouring ops.

    This is the second step of a generic "tile and fuse" strategy. Starting from
    the already-annotated `root` ops (see `assign_tile_sizes`), the tile sizes
    are maximally propagated to surrounding elementwise / fill ops that share a
    tensor with an annotated op. Sizes are translated across each op's indexing
    maps so that a shared tensor is tiled consistently on both sides, which makes
    the annotations usable as fusion hints.

    Propagation flows in both directions (to consumers and producers) and only
    targets tileable elementwise-like ops; contraction (anchor) ops are never
    re-tiled. Reduction dimensions are never tiled. Ops that are already
    annotated keep their existing sizes and act as propagation barriers.

    Args:
        root: Handle to annotated anchor op(s).
    Return:
        Handle to all ops carrying a tile-size annotation after propagation
        (the roots plus the newly annotated ops).
    """

    root: ext.Operand[transform.AnyOpType]
    annotated: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "PropagateTileSizesOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            root_ops = list(state.get_payload_ops(op.root))

            # Worklist flood-fill across shared tensors. An op is "visited" once
            # it carries an annotation, which prevents re-processing and acts as
            # a barrier. Track ordered, de-duplicated annotated ops for results.
            annotated: list[ir.Operation] = []
            seen_results: set = set()

            def remember(target_op: ir.Operation) -> None:
                key = target_op.operation.__hash__()
                if key not in seen_results:
                    seen_results.add(key)
                    annotated.append(target_op)

            worklist: list[ir.Operation] = []
            for root in root_ops:
                if tsa.get_tile_sizes_attr(root) is not None:
                    remember(root)
                    worklist.append(root)

            def try_propagate(
                src: ir.Operation, src_sizes, shared: ir.Value, dst: ir.Operation
            ) -> None:
                if tsa.get_tile_sizes_attr(dst) is not None:
                    return
                if not tsa.is_propagatable(dst):
                    return
                dst_sizes = tsa.propagate_through_value(src, src_sizes, shared, dst)
                if dst_sizes is None or not any(dst_sizes):
                    return
                tsa.set_tile_sizes_attr(dst, dst_sizes)
                remember(dst)
                worklist.append(dst)

            while worklist:
                src = worklist.pop()
                src_sizes = tsa.get_tile_sizes_attr(src)
                if src_sizes is None:
                    continue
                src_view = src.opview

                # Forward: annotate tileable consumers of each result.
                for result in src_view.results:
                    for user in tsa.op_users(result):
                        try_propagate(src, src_sizes, result, user)

                # Backward: annotate tileable producers feeding any operand
                # (inputs and DPS init/outputs, e.g. a fill feeding a matmul).
                for operand in src_view.operands:
                    producer = tsa.defining_op(operand)
                    if producer is not None:
                        try_propagate(src, src_sizes, operand, producer)

            results.set_ops(op.annotated, annotated)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "PropagateTileSizesOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.modifies_payload(effects)


def propagate_tile_sizes(
    root: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a PropagateTileSizesOp.

    Args:
        root: Handle to annotated anchor op(s).
    Returns:
        Handle to all ops carrying a tile-size annotation after propagation.
    """
    return PropagateTileSizesOp(root=root).annotated
