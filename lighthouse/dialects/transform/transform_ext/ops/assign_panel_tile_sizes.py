from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import fusion_analysis as fa
from lighthouse.dialects.transform.transform_ext.utils import panel_analysis as pa
from lighthouse.dialects.transform.transform_ext.utils import tile_size_analysis as tsa
from lighthouse.dialects.transform.transform_ext.utils.tiling.target_caps import (
    panel_budget_bytes,
)
from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import op_key


class AssignPanelTileSizesOp(
    TransformExtensionDialect.Operation, name="assign_panel_tile_sizes"
):
    """
    Fuse a GEMM with a row-reduction epilogue (e.g. softmax, layer norm) per row panel.

    For each contraction whose downstream elementwise/relayout chain reaches a
    non-contraction reduction, the whole chain is tiled by one row panel of the
    GEMM with all other dims kept whole. Terminal ops of the chain are annotated
    with their panel tile sizes and become fusion roots; all other members are
    marked as panel members so that they are only fused as producers. This way
    the GEMM is computed once per panel and never recomputed or materialized.

    Panels are one M block high, halved down to `MIN_PANEL_ROWS` sub-blocks
    until there are enough panels for all cores and a panel fits half the L2
    (see `panel_analysis`). Without a feasible panel, the GEMM epilogue is
    split from the reduction chain instead (see `split_after_epilogue`).
    Chains whose intermediates are also used elsewhere, or which overlap an
    already assigned panel, are left untouched.

    Args:
        target: Handle to candidate contraction op(s).
        tile_size: Panel rows for unpacked (plain) GEMMs.
    Return:
        Handle to the annotated panel roots.
    """

    target: ext.Operand[transform.AnyOpType]
    tile_size: ext.Operand[transform.AnyParamType] | None = None
    roots: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "AssignPanelTileSizesOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            rows = tsa.DEFAULT_TILE_SIZE
            if op.tile_size is not None:
                params = state.get_params(op.tile_size)
                if len(params) == 1 and isinstance(params[0], ir.IntegerAttr):
                    rows = params[0].value
            roots = []
            target = TargetInfo.host()
            cores = target.core_count()
            budget = panel_budget_bytes(target)
            for gemm in state.get_payload_ops(op.target):
                chain = pa.Chain(gemm)
                if not chain.has_reduction or chain.escapes:
                    continue
                if any(
                    fa.is_panel_member(m) or tsa.get_tile_sizes_attr(m) is not None
                    for m in chain.members
                ):
                    continue
                op_sizes = None
                for height in pa.panel_row_candidates(chain.gemm, rows):
                    sizes = pa.build_panel(chain, height)
                    if sizes is None:
                        break
                    if pa.fits_panel(chain, sizes, cores, budget):
                        op_sizes = sizes
                        break
                if op_sizes is None:
                    pa.split_after_epilogue(chain)
                    continue
                terminal_keys = {op_key(t) for t in chain.terminals}
                for m in chain.members:
                    if op_key(m) not in terminal_keys:
                        fa.mark_panel_member(m)
                for t in chain.terminals:
                    tsa.set_tile_sizes_attr(t, op_sizes[op_key(t)])
                    roots.append(t)

            results.set_ops(op.roots, roots)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "AssignPanelTileSizesOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def assign_panel_tile_sizes(
    target: ir.Value[transform.AnyOpType],
    tile_size: int | ir.Value | None = None,
) -> ir.Value:
    """
    snake_case wrapper to create an AssignPanelTileSizesOp.

    Args:
        target: Handle to candidate contraction op(s).
        tile_size: Panel rows for unpacked (plain) GEMMs.
    Returns:
        Handle to the annotated panel roots.
    """
    if isinstance(tile_size, int):
        param_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), tile_size)
        tile_size = transform.ParamConstantOp(transform.AnyParamType.get(), param_attr)
    return AssignPanelTileSizesOp(target=target, tile_size=tile_size).roots
