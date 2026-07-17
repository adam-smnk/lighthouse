from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext import tile_size_analysis as tsa


class AssignTileSizesOp(TransformExtensionDialect.Operation, name="assign_tile_sizes"):
    """
    Assign target tile sizes to anchor ops as IR attribute annotations.

    This is the first step of a generic "tile and fuse" strategy. Each `target`
    op is analysed and annotated with a `transform_ext.tile_sizes` attribute
    holding one tile size per iteration dimension (in loop order). The sizes can
    later be propagated to neighbouring ops (see `propagate_tile_sizes`) and
    consumed by a tile-and-fuse step (see `get_tile_sizes`).

    Tiling aims for 2D tiles over the innermost parallel dimensions:
      * GEMM-like ops (matmul, batch matmul, contraction, ...): batch dims are
        tiled by 1, the two contraction parallel dims (M, N) by `tile_size`, and
        reduction dims (K) are left untiled.
      * Other structured ops (e.g. elementwise generics): the innermost two
        parallel dims are tiled by `tile_size`, other parallel dims by 1, and
        reduction dims are left untiled.

    Statically small parallel dimensions are left untiled.

    Targets for which no tile sizes can be computed are skipped (left
    unannotated). The original `target` handle is returned for chaining.

    Args:
        target: Handle to anchor op(s).
        tile_size: Optional size for tiled dimensions (default: 32).
            Acts as a user hint, e.g. tile by 64 instead of the default 32.
    Return:
        Pass-through handle to the (now annotated) target ops.
    """

    target: ext.Operand[transform.AnyOpType]
    tile_size: ext.Operand[transform.AnyParamType] | None = None
    annotated: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "AssignTileSizesOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)

            tile_size = tsa.DEFAULT_TILE_SIZE
            if op.tile_size is not None:
                tile_attr = state.get_params(op.tile_size)
                if len(tile_attr) == 1 and isinstance(tile_attr[0], ir.IntegerAttr):
                    tile_size = tile_attr[0].value

            annotated = []
            for target_op in target_ops:
                # Respect existing annotations so earlier (e.g. GEMM-derived)
                # tile sizes take precedence over later default assignments.
                if tsa.get_tile_sizes_attr(target_op) is not None:
                    annotated.append(target_op)
                    continue
                sizes = tsa.compute_anchor_tile_sizes(target_op, tile_size=tile_size)
                if sizes is None:
                    continue
                tsa.set_tile_sizes_attr(target_op, sizes)
                annotated.append(target_op)

            results.set_ops(op.annotated, annotated)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "AssignTileSizesOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.modifies_payload(effects)


def assign_tile_sizes(
    target: ir.Value[transform.AnyOpType],
    tile_size: int | ir.Value | None = None,
) -> ir.Value:
    """
    snake_case wrapper to create an AssignTileSizesOp.

    Args:
        target: Handle to anchor op(s).
        tile_size: Optional size for tiled dimensions (default: 32).
    Returns:
        Pass-through handle to the (now annotated) target ops.
    """
    if isinstance(tile_size, int):
        param_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), tile_size)
        tile_size = transform.ParamConstantOp(transform.AnyParamType.get(), param_attr)

    return AssignTileSizesOp(target=target, tile_size=tile_size).annotated
