from mlir import ir
from mlir.dialects import linalg

from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import (
    dim_position,
    indexing_maps,
    linalg_inputs,
    linalg_outputs,
    opview,
)

from .common import assign_reduction_tiles, parallel_and_reduction_dims


def _contraction_operand_types(
    op: ir.Operation | ir.OpView,
) -> tuple[ir.Type, ir.Type, ir.Type] | None:
    """(lhs, rhs, acc) element types of a single-output contraction, else None."""
    ov = opview(op)
    if not linalg.isa_contraction_op(ov):
        return None
    inputs = linalg_inputs(ov)
    outputs = linalg_outputs(ov)
    if inputs is None or outputs is None or len(inputs) < 2 or len(outputs) != 1:
        return None
    return (
        ir.ShapedType(inputs[0].type).element_type,
        ir.ShapedType(inputs[1].type).element_type,
        ir.ShapedType(outputs[0].type).element_type,
    )


def is_amx_bf16_contraction(
    op: ir.Operation | ir.OpView, target: TargetInfo | None
) -> bool:
    """True for a bf16 -> f32 contraction on an AMX-capable target."""
    if target is None or not target.is_supported("amx"):
        return False
    types = _contraction_operand_types(op)
    if types is None:
        return False
    lhs, rhs, acc = types
    ov = opview(op)
    inputs = linalg_inputs(ov)
    outputs = linalg_outputs(ov)
    # AMX requires matrix-shaped (rank >= 2) operands; reject vector/scalar contractions.
    if any(ir.ShapedType(v.type).rank < 2 for v in (inputs[0], inputs[1], outputs[0])):
        return False
    return (
        isinstance(lhs, ir.BF16Type)
        and isinstance(rhs, ir.BF16Type)
        and isinstance(acc, ir.F32Type)
    )


def is_f32_contraction(op: ir.Operation | ir.OpView) -> bool:
    """True for a contraction with all-f32 operands (lhs, rhs and acc)."""
    types = _contraction_operand_types(op)
    return types is not None and all(isinstance(t, ir.F32Type) for t in types)


def vnni_reduction_dims(op: ir.Operation | ir.OpView) -> tuple[int, int, int] | None:
    """``(k_dim, vnni_dim, vnni_factor)`` of a contraction in VNNI layout, else None.

    Mirrors upstream ``x86::isInVnniLayout``: A is ``[..., K/v, v]`` and B is
    ``[..., K/v, N, v]`` with both ``K/v`` and ``v`` reduction dims.
    """
    ov = opview(op)
    maps = indexing_maps(ov)
    inputs = linalg_inputs(ov)
    if maps is None or inputs is None or len(inputs) < 2 or len(maps) != 3:
        return None
    type_a, type_b = (ir.ShapedType(v.type) for v in inputs[:2])
    if type_a.rank < 3 or type_b.rank < 3:
        return None
    _, reduction = parallel_and_reduction_dims(maps[2])
    map_a, map_b = maps[0].results, maps[1].results
    vnni_a, vnni_b = dim_position(map_a[-1]), dim_position(map_b[-1])
    k_a, k_b = dim_position(map_a[-2]), dim_position(map_b[-3])
    n_b = dim_position(map_b[-2])
    if (
        vnni_a is None
        or vnni_a != vnni_b
        or vnni_a not in reduction
        or k_a is None
        or k_a != k_b
        or k_a not in reduction
        or n_b is None
        or n_b in reduction
    ):
        return None
    factor = type_b.shape[-1]
    if (
        ir.ShapedType.is_dynamic_size(factor)
        or factor == 0
        or factor % 2
        or type_a.shape[-1] != factor
        or type_a.shape[-2] != type_b.shape[-3]
    ):
        return None
    return k_a, vnni_a, factor


def assign_amx_reduction_tiles(
    op: ir.Operation | ir.OpView, reduction_dims: list[int], sizes: list[int]
) -> None:
    """AMX reduction tiles: 32 K elements per tile, 16 K-pairs x 2 in VNNI layout."""
    vnni = vnni_reduction_dims(op)
    if vnni is None:
        assign_reduction_tiles(reduction_dims, [32], sizes)
        return
    # The VNNI AMX microkernel reads a 16 x (16 * factor) tile per operand.
    k_dim, vnni_dim, factor = vnni
    for dim in reduction_dims:
        sizes[dim] = 1
    sizes[k_dim] = 16
    sizes[vnni_dim] = factor


def vector_lane_count(target: TargetInfo | None, elem_type: ir.Type) -> int:
    """SIMD lane count for a single vector register; does not include the full register bank."""
    vector_bits = (
        target.vector_register_width_bits
        if target is not None and target.vector_register_width_bits is not None
        else 512
    )
    if isinstance(elem_type, (ir.FloatType, ir.IntegerType)):
        return max(1, vector_bits // max(1, elem_type.width))
    return 16


def generic_parallel_tiles(
    op: ir.Operation | ir.OpView,
    out_map: ir.AffineMap,
    target: TargetInfo | None,
) -> list[int]:
    """SIMD-lane default parallel tiles for ops without a microkernel profile."""
    parallel_dims, _ = parallel_and_reduction_dims(out_map)
    if not parallel_dims:
        return []
    out_elem = ir.ShapedType(linalg_outputs(op)[0].type).element_type
    inner = vector_lane_count(target, out_elem)
    return [inner] if len(parallel_dims) == 1 else [1, inner]


def generic_reduction_tiles() -> list[int]:
    """Default reduction tile for ops without a microkernel profile."""
    return [1]
