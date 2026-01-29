# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HLO Code Generation

This module provides classes for building and representing HLO operations,
tensors, and modules for code generation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import ml_dtypes
import numpy as np

from nkipy.third_party.xla import xla_data_pb2
from nkipy.third_party.xla.service import hlo_pb2

# =============================================================================
# Constants
# =============================================================================

NEFF_INPUT_NAMES = "neff_input_names"
NEFF_OUTPUT_NAMES = "neff_output_names"
NKIPY_OP_TYPE = "NKIPy_Op"
TOPK_CUSTOM_CALL = "AwsNeuronTopK"

# =============================================================================
# Dtype Mappings
# =============================================================================

# Mapping from numpy dtype type to XLA PrimitiveType enum
# Includes standard numpy types and ml_dtypes types
_NUMPY_TO_XLA_TYPE: Dict[type, int] = {
    # Standard numpy integer types
    np.int8: xla_data_pb2.S8,
    np.int16: xla_data_pb2.S16,
    np.int32: xla_data_pb2.S32,
    np.int64: xla_data_pb2.S64,
    np.uint8: xla_data_pb2.U8,
    np.uint16: xla_data_pb2.U16,
    np.uint32: xla_data_pb2.U32,
    np.uint64: xla_data_pb2.U64,
    # Standard numpy float types
    np.float16: xla_data_pb2.F16,
    np.float32: xla_data_pb2.F32,
    np.float64: xla_data_pb2.F64,
    # Boolean
    np.bool_: xla_data_pb2.PRED,
    # ml_dtypes float types
    ml_dtypes.bfloat16: xla_data_pb2.BF16,
    ml_dtypes.float8_e5m2: xla_data_pb2.F8E5M2,
    ml_dtypes.float8_e4m3: xla_data_pb2.F8E4M3,
    ml_dtypes.float8_e4m3fn: xla_data_pb2.F8E4M3FN,
}

# Mapping from dtype name string to XLA PrimitiveType enum (for string-based lookups)
_DTYPE_NAME_TO_XLA_TYPE: Dict[str, int] = {
    "bfloat16": xla_data_pb2.BF16,
    "float8_e5m2": xla_data_pb2.F8E5M2,
    "float8_e4m3": xla_data_pb2.F8E4M3,
    "float8_e4m3fn": xla_data_pb2.F8E4M3FN,
}

# =============================================================================
# Helper Functions
# =============================================================================


def _normalize_dtype(dtype) -> type:
    """Normalize a dtype to its numpy type class.

    Handles various dtype representations:
    - np.dtype objects (e.g., np.dtype('float32'))
    - numpy type classes (e.g., np.float32)
    - ml_dtypes types (e.g., ml_dtypes.bfloat16)
    - string representations (e.g., 'float32', 'bfloat16')

    Returns:
        The numpy type class (e.g., np.float32, ml_dtypes.bfloat16)
    """
    # Already a type class
    if isinstance(dtype, type):
        return dtype

    # np.dtype object - get its type
    if isinstance(dtype, np.dtype):
        return dtype.type

    # String - convert via np.dtype
    if isinstance(dtype, str):
        return np.dtype(dtype).type

    # Has a 'type' attribute (like np.dtype)
    if hasattr(dtype, "type"):
        return dtype.type

    # Fallback: try to convert via np.dtype
    return np.dtype(dtype).type


def _get_dtype_name(dtype) -> str:
    """Get the string name of a dtype.

    Args:
        dtype: Any dtype representation

    Returns:
        String name of the dtype (e.g., 'float32', 'bfloat16')
    """
    if isinstance(dtype, str):
        return dtype

    if isinstance(dtype, np.dtype):
        return dtype.name

    if isinstance(dtype, type):
        return np.dtype(dtype).name

    if hasattr(dtype, "name"):
        return dtype.name

    return np.dtype(dtype).name


def _dtype_to_primitive_type(dtype) -> int:
    """Convert numpy dtype to XLA PrimitiveType enum.

    Handles standard numpy types, ml_dtypes types, and string representations.

    Args:
        dtype: Any dtype representation

    Returns:
        XLA PrimitiveType enum value
    """
    # Try direct type lookup first
    dtype_type = _normalize_dtype(dtype)
    if dtype_type in _NUMPY_TO_XLA_TYPE:
        return _NUMPY_TO_XLA_TYPE[dtype_type]

    # Try name-based lookup for special types
    dtype_name = _get_dtype_name(dtype)
    if dtype_name in _DTYPE_NAME_TO_XLA_TYPE:
        return _DTYPE_NAME_TO_XLA_TYPE[dtype_name]

    # Default to float32
    return xla_data_pb2.F32


def _make_shape_proto(
    shape: Tuple[int, ...], dtype, include_layout: bool = True
) -> xla_data_pb2.ShapeProto:
    """Create a ShapeProto from shape tuple and dtype."""
    shape_proto = xla_data_pb2.ShapeProto()
    shape_proto.element_type = _dtype_to_primitive_type(dtype)
    shape_proto.dimensions.extend(shape)
    shape_proto.is_dynamic_dimension.extend([False] * len(shape))

    if include_layout:
        layout = shape_proto.layout
        if len(shape) > 0:
            layout.minor_to_major.extend(reversed(range(len(shape))))
        layout.tail_padding_alignment_in_elements = 1

    return shape_proto


def _make_tuple_shape_proto(
    shapes_and_dtypes: List[Tuple[Tuple[int, ...], Any]],
) -> xla_data_pb2.ShapeProto:
    """Create a tuple ShapeProto from list of (shape, dtype) pairs.

    Args:
        shapes_and_dtypes: List of (shape, dtype) tuples

    Returns:
        Configured tuple ShapeProto
    """
    tuple_shape = xla_data_pb2.ShapeProto()
    tuple_shape.element_type = xla_data_pb2.TUPLE

    for shape, dtype in shapes_and_dtypes:
        elem_shape = tuple_shape.tuple_shapes.add()
        elem_shape.element_type = _dtype_to_primitive_type(dtype)
        elem_shape.dimensions.extend(shape)
        elem_shape.is_dynamic_dimension.extend([False] * len(shape))
        if len(shape) > 0:
            elem_shape.layout.minor_to_major.extend(reversed(range(len(shape))))
        elem_shape.layout.tail_padding_alignment_in_elements = 1

    return tuple_shape


def _set_literal_value(literal, value, dtype) -> None:
    """Set literal value in protobuf based on dtype.

    Handles both scalar and array values uniformly.
    Supports standard numpy types and ml_dtypes types.

    Args:
        literal: The literal protobuf to populate
        value: Scalar or numpy array value
        dtype: Data type of the value
    """
    # Setup literal shape
    is_array = isinstance(value, np.ndarray)
    shape = value.shape if is_array else ()

    literal.shape.element_type = _dtype_to_primitive_type(dtype)
    literal.shape.dimensions.extend(shape)
    literal.shape.is_dynamic_dimension.extend([False] * len(shape))
    if len(shape) > 0:
        literal.shape.layout.minor_to_major.extend(reversed(range(len(shape))))
    literal.shape.layout.tail_padding_alignment_in_elements = 1

    # Get values as flat list
    values = value.flatten().tolist() if is_array else [value]
    dtype_type = _normalize_dtype(dtype)
    dtype_name = _get_dtype_name(dtype)

    # Handle ml_dtypes types by name
    if dtype_name == "bfloat16":
        # Convert values to bfloat16 first, then get uint16 representation
        bf16_array = np.array(values, dtype=ml_dtypes.bfloat16)
        bf16_bytes = bf16_array.view(np.uint16).tobytes()
        literal.bf16s = bf16_bytes
        return

    if dtype_name == "float8_e5m2":
        # Convert values to float8_e5m2 first, then get uint8 representation
        f8_array = np.array(values, dtype=ml_dtypes.float8_e5m2)
        f8_bytes = f8_array.view(np.uint8).tobytes()
        literal.f8_e5m2s = f8_bytes
        return

    if dtype_name == "float8_e4m3":
        # NKI float8_e4m3 (distinct from float8_e4m3fn)
        # Convert values to float8_e4m3 first, then get uint8 representation
        f8_array = np.array(values, dtype=ml_dtypes.float8_e4m3)
        f8_bytes = f8_array.view(np.uint8).tobytes()
        literal.f8e4m3s = f8_bytes
        return

    if dtype_name == "float8_e4m3fn":
        # float8_e4m3fn variant (finite numbers only)
        # Convert values to float8_e4m3fn first, then get uint8 representation
        f8_array = np.array(values, dtype=ml_dtypes.float8_e4m3fn)
        f8_bytes = f8_array.view(np.uint8).tobytes()
        literal.f8e4m3fns = f8_bytes
        return

    # Handle standard numpy types
    dtype_handlers = {
        np.float16: lambda: setattr(
            literal, "f16s", b"".join(struct.pack("<e", np.float16(v)) for v in values)
        ),
        np.float32: lambda: literal.f32s.extend(float(v) for v in values),
        np.float64: lambda: literal.f64s.extend(float(v) for v in values),
        np.int8: lambda: setattr(
            literal, "s8s", bytes([int(v) & 0xFF for v in values])
        ),
        np.int16: lambda: literal.s16s.extend(int(v) for v in values),
        np.int32: lambda: literal.s32s.extend(int(v) for v in values),
        np.int64: lambda: literal.s64s.extend(int(v) for v in values),
        np.uint8: lambda: setattr(literal, "u8s", bytes([int(v) for v in values])),
        np.uint16: lambda: literal.u16s.extend(int(v) for v in values),
        np.uint32: lambda: literal.u32s.extend(int(v) for v in values),
        np.uint64: lambda: literal.u64s.extend(int(v) for v in values),
        np.bool_: lambda: literal.preds.extend(bool(v) for v in values),
    }

    handler = dtype_handlers.get(dtype_type)
    if handler:
        handler()
    else:
        # Default to float32
        literal.f32s.extend(float(v) for v in values)


# =============================================================================
# Core Classes
# =============================================================================


@dataclass
class HLOOp:
    """Represents an HLO operation."""

    op_name: str
    operands: List[HLOTensor]
    result_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]
    result_dtype: Union[np.dtype, List[np.dtype]]
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_location: Optional[Tuple[str, int]] = None
    id: Optional[int] = None


@dataclass
class HLOTensor:
    """Represents a tensor in the HLO computation graph."""

    shape: Tuple[int, ...]
    dtype: np.dtype
    source_op: Optional[HLOOp] = None
    is_parameter: bool = False
    parameter_id: Optional[int] = None
    name: str = ""
    id: Optional[int] = None


@dataclass
class TensorPlaceholder:
    """Placeholder for tensor metadata."""

    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype


# =============================================================================
# HLO Module
# =============================================================================


class HLOModule:
    """Represents an HLO module (a complete computation)."""

    def __init__(self, name: str = "main") -> None:
        self.name = name
        self.parameters: List[HLOTensor] = []
        self.operations: List[HLOOp] = []
        self.results: List[HLOTensor] = []
        self.input_output_alias: Dict[int, int] = {}
        self._next_id = 0

    @property
    def inputs(self) -> List[TensorPlaceholder]:
        """Return parameters as inputs for compatibility with IR Function interface."""
        return [
            TensorPlaceholder(name=p.name, shape=p.shape, dtype=p.dtype)
            for p in self.parameters
        ]

    @property
    def outputs(self) -> List[TensorPlaceholder]:
        """Return results as outputs for compatibility with IR Function interface."""
        return [
            TensorPlaceholder(
                name=r.name,
                shape=r.shape,
                dtype=np.uint8 if r.dtype is np.bool_ else r.dtype,
            )
            for r in self.results
        ]

    def add_parameter(
        self, shape: Tuple[int, ...], dtype: np.dtype, name: str = ""
    ) -> HLOTensor:
        """Add an input parameter to the module."""
        param = HLOTensor(
            shape=shape,
            dtype=dtype,
            is_parameter=True,
            parameter_id=len(self.parameters),
            name=name,
        )
        self.parameters.append(param)
        return param

    def add_operation(self, op: HLOOp) -> HLOTensor:
        """Add an operation and return its result tensor."""
        op.id = self._next_id
        self._next_id += 1

        result = HLOTensor(
            shape=op.result_shape, dtype=op.result_dtype, source_op=op, id=op.id
        )
        self.operations.append(op)
        return result

    def set_results(self, results: Union[HLOTensor, List[HLOTensor]]) -> None:
        """Set the output results of the module."""
        self.results = results if isinstance(results, list) else [results]

    # =========================================================================
    # Proto Generation
    # =========================================================================

    def to_proto(self) -> hlo_pb2.HloModuleProto:
        """Generate XLA HloModuleProto (protobuf)."""
        module_proto = hlo_pb2.HloModuleProto()
        module_proto.name = self.name

        # Track computation IDs and instruction IDs
        reduce_computation_map: Dict[Tuple[str, str], int] = {}
        next_comp_id = 0
        next_global_instr_id = 0

        # Create sub-computations for reduce/scatter operations
        next_comp_id, next_global_instr_id = self._create_sub_computations(
            module_proto, reduce_computation_map, next_comp_id, next_global_instr_id
        )

        # Create main computation
        computation = module_proto.computations.add()
        computation.name = "main"
        computation.id = next_comp_id

        # Build instructions
        instr_id_map: Dict[str, int] = {}
        next_instr_id = next_global_instr_id

        # Emit parameters
        next_instr_id = self._emit_parameters(computation, instr_id_map, next_instr_id)

        # Emit operations
        next_instr_id = self._emit_operations(
            computation, instr_id_map, next_instr_id, reduce_computation_map
        )

        # Emit root instruction
        next_instr_id = self._emit_root(computation, instr_id_map, next_instr_id)

        # Set computation metadata
        computation.root_id = next_instr_id - 1
        computation.program_shape.CopyFrom(self._make_program_shape())

        # Set module metadata
        module_proto.entry_computation_name = "main"
        module_proto.entry_computation_id = next_comp_id
        module_proto.host_program_shape.CopyFrom(self._make_program_shape())

        # Set input/output aliases
        self._set_aliases(module_proto)

        return module_proto

    def _create_sub_computations(
        self,
        module_proto: hlo_pb2.HloModuleProto,
        computation_map: Dict[Tuple[str, str], int],
        next_comp_id: int,
        next_instr_id: int,
    ) -> Tuple[int, int]:
        """Create sub-computations for reduce/scatter operations."""

        # Collect needed computations
        computations_needed: Dict[Tuple[str, str, str], Any] = {}

        for op in self.operations:
            if op.op_name == "reduce":
                comp_name = op.attributes.get("computation", "add")
                key = ("reduce", comp_name, str(op.result_dtype))
                computations_needed[key] = (comp_name, op.result_dtype, comp_name)

            elif op.op_name == "scatter":
                comp_name = op.attributes.get("update_computation", "add")
                if comp_name != "assign":
                    raise NotImplementedError(
                        f"Scatter update_computation '{comp_name}' not supported."
                        " Only 'assign' is implemented."
                    )
                key = ("scatter", comp_name, str(op.result_dtype))
                computations_needed[key] = (
                    f"scatter_{comp_name}",
                    op.result_dtype,
                    "copy",
                )

            elif op.op_name in ("all-reduce", "reduce-scatter"):
                comp_name = op.attributes.get("reduce_op", "add")
                key = ("collective", comp_name, str(op.result_dtype))
                computations_needed[key] = (
                    f"collective_{comp_name}",
                    op.result_dtype,
                    comp_name,
                )

        # Create each computation
        for (comp_type, comp_name, dtype_key), (
            name_prefix,
            result_dtype,
            opcode,
        ) in computations_needed.items():
            next_comp_id, next_instr_id = self._create_binary_computation(
                module_proto,
                computation_map,
                name_prefix,
                dtype_key,
                result_dtype,
                opcode,
                next_comp_id,
                next_instr_id,
            )

        return next_comp_id, next_instr_id

    def _create_binary_computation(
        self,
        module_proto: hlo_pb2.HloModuleProto,
        computation_map: Dict[Tuple[str, str], int],
        name_prefix: str,
        dtype_key: str,
        result_dtype,
        result_opcode: str,
        next_comp_id: int,
        next_instr_id: int,
    ) -> Tuple[int, int]:
        """Create a binary sub-computation (for reduce/scatter operations)."""

        comp = module_proto.computations.add()
        comp.name = f"{name_prefix}_{dtype_key}_computation"
        comp.id = next_comp_id
        computation_map[(name_prefix, dtype_key)] = next_comp_id

        # Add lhs parameter
        lhs = comp.instructions.add()
        lhs.id = next_instr_id
        lhs.name = f"{name_prefix}_{dtype_key}_lhs"
        lhs.opcode = "parameter"
        lhs.parameter_number = 0
        lhs.shape.CopyFrom(_make_shape_proto((), result_dtype))
        next_instr_id += 1

        # Add rhs parameter
        rhs = comp.instructions.add()
        rhs.id = next_instr_id
        rhs.name = f"{name_prefix}_{dtype_key}_rhs"
        rhs.opcode = "parameter"
        rhs.parameter_number = 1
        rhs.shape.CopyFrom(_make_shape_proto((), result_dtype))
        next_instr_id += 1

        # Add result operation
        result = comp.instructions.add()
        result.id = next_instr_id
        result.name = f"{name_prefix}_{dtype_key}_result"
        result.opcode = result_opcode
        result.shape.CopyFrom(_make_shape_proto((), result_dtype))

        # For scatter "assign", only use rhs; for reduce ops, use both
        if result_opcode == "copy":
            result.operand_ids.append(rhs.id)
        else:
            result.operand_ids.extend([lhs.id, rhs.id])
        next_instr_id += 1

        comp.root_id = result.id

        # Create program shape
        prog_shape = xla_data_pb2.ProgramShapeProto()
        for _ in range(2):
            param_shape = xla_data_pb2.ShapeProto()
            param_shape.element_type = _dtype_to_primitive_type(result_dtype)
            prog_shape.parameters.append(param_shape)
            prog_shape.parameter_names.append("param")

        result_shape = xla_data_pb2.ShapeProto()
        result_shape.element_type = _dtype_to_primitive_type(result_dtype)
        prog_shape.result.CopyFrom(result_shape)
        comp.program_shape.CopyFrom(prog_shape)

        return next_comp_id + 1, next_instr_id

    def _emit_parameters(
        self, computation, instr_id_map: Dict[str, int], next_instr_id: int
    ) -> int:
        """Emit parameter instructions."""
        for param in self.parameters:
            instr = computation.instructions.add()
            instr.id = next_instr_id
            instr.name = param.name or f"p{param.parameter_id}"
            instr.opcode = "parameter"
            instr.parameter_number = param.parameter_id
            instr.shape.CopyFrom(_make_shape_proto(param.shape, param.dtype))

            if param.name:
                instr.frontend_attributes.map[NEFF_INPUT_NAMES] = param.name

            instr_id_map[f"p{param.parameter_id}"] = next_instr_id
            next_instr_id += 1

        return next_instr_id

    def _emit_operations(
        self,
        computation,
        instr_id_map: Dict[str, int],
        next_instr_id: int,
        reduce_computation_map: Dict[Tuple[str, str], int],
    ) -> int:
        """Emit operation instructions."""
        for op in self.operations:
            instr = computation.instructions.add()
            instr.id = next_instr_id
            instr.name = f"op{op.id}"
            instr.opcode = op.op_name

            # Set shape (may be overwritten by specific handlers)
            is_tuple = op.attributes.get("is_tuple", False)
            if isinstance(op.result_shape, list) and is_tuple:
                instr.shape.element_type = xla_data_pb2.TUPLE
            else:
                instr.shape.CopyFrom(
                    _make_shape_proto(op.result_shape, op.result_dtype)
                )

            instr_id_map[f"op{op.id}"] = next_instr_id
            next_instr_id += 1

            # Add operand IDs
            for operand in op.operands:
                key = (
                    f"p{operand.parameter_id}"
                    if operand.is_parameter
                    else f"op{operand.id}"
                )
                instr.operand_ids.append(instr_id_map[key])

            # Add metadata
            if op.source_location:
                instr.metadata.op_type = NKIPY_OP_TYPE
                instr.metadata.op_name = f"hlo__{op.op_name}"
                instr.metadata.source_file = op.source_location[0]
                instr.metadata.source_line = op.source_location[1]

            # Handle operation-specific attributes
            self._handle_operation(instr, op, reduce_computation_map)

        return next_instr_id

    def _handle_operation(
        self, instr, op: HLOOp, reduce_computation_map: Dict[Tuple[str, str], int]
    ) -> None:
        """Handle operation-specific instruction configuration."""

        handlers = {
            "constant": self._handle_constant,
            "broadcast": self._handle_broadcast,
            "compare": self._handle_compare,
            "transpose": self._handle_transpose,
            "slice": self._handle_slice,
            "reduce": self._handle_reduce,
            "dot": self._handle_dot,
            "gather": self._handle_gather,
            "scatter": self._handle_scatter,
            "topk": self._handle_topk,
            "get-tuple-element": self._handle_get_tuple_element,
            "concatenate": self._handle_concatenate,
            "custom-call": self._handle_custom_call,
            "convolution": self._handle_convolution,
            "all-gather": self._handle_all_gather,
            "all-reduce": self._handle_all_reduce,
            "reduce-scatter": self._handle_reduce_scatter,
            "all-to-all": self._handle_all_to_all,
        }

        handler = handlers.get(op.op_name)
        if handler:
            handler(instr, op, reduce_computation_map)
        else:
            # Note: ops without handler should be simple elementwise ops
            pass

    def _handle_constant(self, instr, op: HLOOp, _) -> None:
        """Handle constant operation."""
        value = op.attributes.get("value", 0.0)
        _set_literal_value(instr.literal, value, op.result_dtype)

    def _handle_broadcast(self, instr, op: HLOOp, _) -> None:
        """Handle broadcast operation."""
        dims = op.attributes.get("broadcast_dimensions", [])
        instr.dimensions.extend(dims)

    def _handle_compare(self, instr, op: HLOOp, _) -> None:
        """Handle compare operation."""
        instr.comparison_direction = op.attributes.get("comparison_direction", "EQ")

    def _handle_transpose(self, instr, op: HLOOp, _) -> None:
        """Handle transpose operation."""
        instr.dimensions.extend(op.attributes.get("permutation", []))

    def _handle_slice(self, instr, op: HLOOp, _) -> None:
        """Handle slice operation."""
        starts = op.attributes.get("start_indices", [])
        limits = op.attributes.get("limit_indices", [])
        strides = op.attributes.get("strides", [])

        for start, limit, stride in zip(starts, limits, strides):
            dim = instr.slice_dimensions.add()
            dim.start = int(start)
            dim.limit = int(limit)
            dim.stride = int(stride)

    def _handle_reduce(self, instr, op: HLOOp, reduce_map: Dict) -> None:
        """Handle reduce operation."""
        instr.dimensions.extend(op.attributes.get("dimensions", []))

        comp_name = op.attributes.get("computation", "add")
        comp_id = reduce_map[(comp_name, str(op.result_dtype))]
        instr.called_computation_ids.append(comp_id)

    def _handle_dot(self, instr, op: HLOOp, _) -> None:
        """Handle dot operation."""
        dot_dims = instr.dot_dimension_numbers
        dot_dims.lhs_contracting_dimensions.extend(
            op.attributes.get("lhs_contracting_dimensions", [])
        )
        dot_dims.rhs_contracting_dimensions.extend(
            op.attributes.get("rhs_contracting_dimensions", [])
        )
        dot_dims.lhs_batch_dimensions.extend(
            op.attributes.get("lhs_batch_dimensions", [])
        )
        dot_dims.rhs_batch_dimensions.extend(
            op.attributes.get("rhs_batch_dimensions", [])
        )

    def _handle_gather(self, instr, op: HLOOp, _) -> None:
        """Handle gather operation."""
        gather_dims = instr.gather_dimension_numbers
        gather_dims.offset_dims.extend(op.attributes.get("offset_dims", []))
        gather_dims.collapsed_slice_dims.extend(
            op.attributes.get("collapsed_slice_dims", [])
        )
        gather_dims.start_index_map.extend(op.attributes.get("start_index_map", []))
        gather_dims.index_vector_dim = op.attributes.get("index_vector_dim", 0)

        instr.gather_slice_sizes.extend(op.attributes.get("slice_sizes", []))
        instr.indices_are_sorted = op.attributes.get("indices_are_sorted", False)

    def _handle_scatter(self, instr, op: HLOOp, reduce_map: Dict) -> None:
        """Handle scatter operation."""
        scatter_dims = instr.scatter_dimension_numbers
        scatter_dims.update_window_dims.extend(
            op.attributes.get("update_window_dims", [])
        )
        scatter_dims.inserted_window_dims.extend(
            op.attributes.get("inserted_window_dims", [])
        )
        scatter_dims.scatter_dims_to_operand_dims.extend(
            op.attributes.get("scatter_dims_to_operand_dims", [])
        )
        scatter_dims.index_vector_dim = op.attributes.get("index_vector_dim", 0)

        instr.indices_are_sorted = op.attributes.get("indices_are_sorted", False)
        instr.unique_indices = op.attributes.get("unique_indices", False)

        comp_name = op.attributes.get("update_computation", "add")
        comp_id = reduce_map[(f"scatter_{comp_name}", str(op.result_dtype))]
        instr.called_computation_ids.append(comp_id)

    def _handle_topk(self, instr, op: HLOOp, _) -> None:
        """Handle topk operation."""
        k = op.attributes.get("k", 1)
        is_tuple = op.attributes.get("is_tuple", False)

        instr.opcode = "custom-call"
        instr.custom_call_target = TOPK_CUSTOM_CALL

        if is_tuple:
            instr.shape.Clear()
            instr.shape.CopyFrom(
                _make_tuple_shape_proto(
                    [
                        (op.result_shape, op.result_dtype),
                        (op.result_shape, np.uint32),
                    ]
                )
            )

        instr.backend_config = str(k).encode("utf-8")

    def _handle_get_tuple_element(self, instr, op: HLOOp, _) -> None:
        """Handle get-tuple-element operation."""
        instr.tuple_index = op.attributes.get("tuple_index", 0)

    def _handle_concatenate(self, instr, op: HLOOp, _) -> None:
        """Handle concatenate operation."""
        instr.dimensions.append(op.attributes.get("dimension", 0))

    def _handle_custom_call(self, instr, op: HLOOp, _) -> None:
        """Handle custom-call operation."""
        instr.custom_call_target = op.attributes.get("custom_call_target", "")

        is_tuple = op.attributes.get("is_tuple", False)
        if is_tuple:
            instr.shape.Clear()
            shapes = (
                op.result_shape
                if isinstance(op.result_shape, list)
                else [op.result_shape]
            )
            dtypes = (
                op.result_dtype
                if isinstance(op.result_dtype, list)
                else [op.result_dtype]
            )
            instr.shape.CopyFrom(_make_tuple_shape_proto(list(zip(shapes, dtypes))))

        backend_config = op.attributes.get("backend_config", "")
        if backend_config:
            instr.backend_config = (
                backend_config.encode("utf-8")
                if isinstance(backend_config, str)
                else backend_config
            )

        # Handle operand-output aliasing for NKI kernels
        operand_output_aliases = op.attributes.get("operand_output_aliases", {})
        for input_idx, output_idx in operand_output_aliases.items():
            alias_proto = xla_data_pb2.OutputOperandAliasing()
            # For tuple outputs, include the output index in the shape index
            if is_tuple:
                alias_proto.output_shape_index.extend([output_idx])
            # For single output, leave output_shape_index empty
            alias_proto.operand_index = input_idx
            alias_proto.operand_shape_index.extend([])
            instr.output_operand_aliasing.append(alias_proto)

        # Handle has_collectives frontend attribute
        if op.attributes.get("has_collectives", False):
            instr.frontend_attributes.map["has_collectives"] = "1"

    def _handle_convolution(self, instr, op: HLOOp, _) -> None:
        """Handle convolution operation."""
        window = instr.window
        window_strides = op.attributes.get("window_strides", [])
        padding = op.attributes.get("padding", [])
        lhs_dilation = op.attributes.get("lhs_dilation", [])
        rhs_dilation = op.attributes.get("rhs_dilation", [])
        kernel_spatial_dims = op.attributes.get("kernel_spatial_dimensions", [])
        kernel_operand = op.operands[1]

        for i in range(len(window_strides)):
            dim = window.dimensions.add()
            dim.size = kernel_operand.shape[kernel_spatial_dims[i]]
            dim.stride = window_strides[i]
            dim.padding_low = padding[i][0]
            dim.padding_high = padding[i][1]
            dim.base_dilation = lhs_dilation[i]
            dim.window_dilation = rhs_dilation[i]

        conv_dims = instr.convolution_dimension_numbers
        conv_dims.input_batch_dimension = op.attributes.get("input_batch_dimension", 0)
        conv_dims.input_feature_dimension = op.attributes.get(
            "input_feature_dimension", 1
        )
        conv_dims.input_spatial_dimensions.extend(
            op.attributes.get("input_spatial_dimensions", [])
        )
        conv_dims.kernel_output_feature_dimension = op.attributes.get(
            "kernel_output_feature_dimension", 0
        )
        conv_dims.kernel_input_feature_dimension = op.attributes.get(
            "kernel_input_feature_dimension", 1
        )
        conv_dims.kernel_spatial_dimensions.extend(
            op.attributes.get("kernel_spatial_dimensions", [])
        )
        conv_dims.output_batch_dimension = op.attributes.get(
            "output_batch_dimension", 0
        )
        conv_dims.output_feature_dimension = op.attributes.get(
            "output_feature_dimension", 1
        )
        conv_dims.output_spatial_dimensions.extend(
            op.attributes.get("output_spatial_dimensions", [])
        )

        instr.feature_group_count = op.attributes.get("feature_group_count", 1)
        instr.batch_group_count = op.attributes.get("batch_group_count", 1)

    def _handle_all_gather(self, instr, op: HLOOp, _) -> None:
        """Handle all-gather operation."""
        instr.dimensions.append(op.attributes.get("all_gather_dim", 0))
        for group in op.attributes.get("replica_groups", [[0]]):
            replica_group = instr.replica_groups.add()
            replica_group.replica_ids.extend(group)

    def _handle_all_reduce(self, instr, op: HLOOp, reduce_map: Dict) -> None:
        """Handle all-reduce operation."""
        for group in op.attributes.get("replica_groups", [[0]]):
            replica_group = instr.replica_groups.add()
            replica_group.replica_ids.extend(group)

        reduce_op = op.attributes.get("reduce_op", "add")
        comp_id = reduce_map[(f"collective_{reduce_op}", str(op.result_dtype))]
        instr.called_computation_ids.append(comp_id)

    def _handle_reduce_scatter(self, instr, op: HLOOp, reduce_map: Dict) -> None:
        """Handle reduce-scatter operation."""
        instr.dimensions.append(op.attributes.get("reduce_scatter_dim", 0))
        for group in op.attributes.get("replica_groups", [[0]]):
            replica_group = instr.replica_groups.add()
            replica_group.replica_ids.extend(group)

        reduce_op = op.attributes.get("reduce_op", "add")
        comp_id = reduce_map[(f"collective_{reduce_op}", str(op.result_dtype))]
        instr.called_computation_ids.append(comp_id)

    def _handle_all_to_all(self, instr, op: HLOOp, _) -> None:
        """Handle all-to-all operation."""
        instr.dimensions.append(op.attributes.get("split_dimension", 0))
        for group in op.attributes.get("replica_groups", [[0]]):
            replica_group = instr.replica_groups.add()
            replica_group.replica_ids.extend(group)

    def _emit_root(
        self, computation, instr_id_map: Dict[str, int], next_instr_id: int
    ) -> int:
        """Emit root instruction."""
        if not self.results:
            return next_instr_id

        root_instr = computation.instructions.add()
        root_instr.id = next_instr_id
        root_instr.name = "result"

        if len(self.results) > 1:
            # Multiple results - create tuple
            root_instr.opcode = "tuple"
            output_names = []
            shapes_and_dtypes = []

            for result in self.results:
                key = (
                    f"p{result.parameter_id}"
                    if result.is_parameter
                    else f"op{result.id}"
                )
                root_instr.operand_ids.append(instr_id_map[key])
                if result.name:
                    output_names.append(result.name)
                shapes_and_dtypes.append((result.shape, result.dtype))

            root_instr.shape.CopyFrom(_make_tuple_shape_proto(shapes_and_dtypes))

            if output_names:
                root_instr.frontend_attributes.map[NEFF_OUTPUT_NAMES] = ",".join(
                    output_names
                )
        else:
            # Single result - copy
            result = self.results[0]
            root_instr.opcode = "copy"
            root_instr.shape.CopyFrom(_make_shape_proto(result.shape, result.dtype))

            key = f"p{result.parameter_id}" if result.is_parameter else f"op{result.id}"
            root_instr.operand_ids.append(instr_id_map[key])

            if result.name:
                root_instr.frontend_attributes.map[NEFF_OUTPUT_NAMES] = result.name

        return next_instr_id + 1

    def _set_aliases(self, module_proto: hlo_pb2.HloModuleProto) -> None:
        """Set input/output aliases in the module proto."""
        for output_index, input_param_number in self.input_output_alias.items():
            alias_entry = module_proto.input_output_alias.entries.add()
            if len(self.results) > 1:
                alias_entry.output_shape_index.append(output_index)
            alias_entry.parameter_number = input_param_number

    def _make_program_shape(self) -> xla_data_pb2.ProgramShapeProto:
        """Create ProgramShape for the computation."""
        program_shape = xla_data_pb2.ProgramShapeProto()

        for param in self.parameters:
            param_shape = _make_shape_proto(
                param.shape, param.dtype, include_layout=False
            )
            program_shape.parameters.append(param_shape)
            program_shape.parameter_names.append(param.name or f"p{param.parameter_id}")

        if self.results:
            if len(self.results) > 1:
                shapes_and_dtypes = [(r.shape, r.dtype) for r in self.results]
                program_shape.result.CopyFrom(
                    _make_tuple_shape_proto(shapes_and_dtypes)
                )
            else:
                result = self.results[0]
                program_shape.result.CopyFrom(
                    _make_shape_proto(result.shape, result.dtype, include_layout=False)
                )

        return program_shape

    def __repr__(self) -> str:
        # TODO: should convert the protobuf to text or generate text automatically
        return f"HLOModule {self.name}"


# =============================================================================
# Type Promotion
# =============================================================================

# Sentinel classes for weak types (Python scalars without explicit dtype)
# These are used as keys in the promotion table, never as outputs


class _WeakInt:
    """Sentinel for Python int (weakly typed integer)."""

    pass


class _WeakFloat:
    """Sentinel for Python float (weakly typed float)."""

    pass


# Notes:
# - Same-type promotion (A + A = A) is handled separately in _lookup_promotion
# - Table only contains one entry per symmetric pair (lookup checks both orderings)
# - float64 is NOT supported on the hardware - will raise TypeError
# - i* = weak int, f* = weak float
# - bf = bfloat16, f8_xxx = float8 variants

_TYPE_PROMOTION_TABLE: Dict[Tuple[type, type], type] = {
    # =========================================================================
    # Bool promotions - bool defers to other types
    # =========================================================================
    # bool + weak types -> concrete type (fixes the original bug!)
    (np.bool_, _WeakInt): np.int32,
    (np.bool_, _WeakFloat): np.float32,
    # bool + integers -> integer
    (np.bool_, np.int8): np.int8,
    (np.bool_, np.int16): np.int16,
    (np.bool_, np.int32): np.int32,
    (np.bool_, np.int64): np.int64,
    (np.bool_, np.uint8): np.uint8,
    (np.bool_, np.uint16): np.uint16,
    (np.bool_, np.uint32): np.uint32,
    (np.bool_, np.uint64): np.uint64,
    # bool + floats -> float
    (np.bool_, np.float16): np.float16,
    (np.bool_, np.float32): np.float32,
    (np.bool_, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    # =========================================================================
    # Weak int promotions - weak int defers to strong types
    # =========================================================================
    (_WeakInt, np.int8): np.int8,
    (_WeakInt, np.int16): np.int16,
    (_WeakInt, np.int32): np.int32,
    (_WeakInt, np.int64): np.int64,
    (_WeakInt, np.uint8): np.uint8,
    (_WeakInt, np.uint16): np.uint16,
    (_WeakInt, np.uint32): np.uint32,
    (_WeakInt, np.uint64): np.uint64,
    (_WeakInt, np.float16): np.float16,
    (_WeakInt, np.float32): np.float32,
    (_WeakInt, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    # =========================================================================
    # Weak float promotions - weak float defers to strong types
    # =========================================================================
    (_WeakFloat, np.int8): np.float32,  # int + weak_float -> float32
    (_WeakFloat, np.int16): np.float32,
    (_WeakFloat, np.int32): np.float32,
    (_WeakFloat, np.int64): np.float32,
    (_WeakFloat, np.uint8): np.float32,
    (_WeakFloat, np.uint16): np.float32,
    (_WeakFloat, np.uint32): np.float32,
    (_WeakFloat, np.uint64): np.float32,
    (_WeakFloat, np.float16): np.float16,
    (_WeakFloat, np.float32): np.float32,
    (_WeakFloat, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    # =========================================================================
    # Signed integer promotions - wider wins (only different-type pairs)
    # =========================================================================
    (np.int8, np.int16): np.int16,
    (np.int8, np.int32): np.int32,
    (np.int8, np.int64): np.int64,
    (np.int16, np.int32): np.int32,
    (np.int16, np.int64): np.int64,
    (np.int32, np.int64): np.int64,
    # =========================================================================
    # Unsigned integer promotions - wider wins (only different-type pairs)
    # =========================================================================
    (np.uint8, np.uint16): np.uint16,
    (np.uint8, np.uint32): np.uint32,
    (np.uint8, np.uint64): np.uint64,
    (np.uint16, np.uint32): np.uint32,
    (np.uint16, np.uint64): np.uint64,
    (np.uint32, np.uint64): np.uint64,
    # =========================================================================
    # Mixed signed/unsigned - promote to wider signed or float32
    # =========================================================================
    (np.int8, np.uint8): np.int16,  # u1 fits in i2
    (np.int8, np.uint16): np.int32,  # u2 fits in i4
    (np.int8, np.uint32): np.int64,  # u4 fits in i8
    (np.int8, np.uint64): np.float32,  # u8 doesn't fit in any signed
    (np.int16, np.uint8): np.int16,  # u1 fits in i2
    (np.int16, np.uint16): np.int32,  # u2 fits in i4
    (np.int16, np.uint32): np.int64,  # u4 fits in i8
    (np.int16, np.uint64): np.float32,
    (np.int32, np.uint8): np.int32,
    (np.int32, np.uint16): np.int32,
    (np.int32, np.uint32): np.int64,
    (np.int32, np.uint64): np.float32,
    (np.int64, np.uint8): np.int64,
    (np.int64, np.uint16): np.int64,
    (np.int64, np.uint32): np.int64,
    (np.int64, np.uint64): np.float32,
    # =========================================================================
    # Integer + float promotions - float wins
    # =========================================================================
    (np.int8, np.float16): np.float16,
    (np.int8, np.float32): np.float32,
    (np.int8, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    (np.int16, np.float16): np.float16,
    (np.int16, np.float32): np.float32,
    (np.int16, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    (np.int32, np.float16): np.float16,
    (np.int32, np.float32): np.float32,
    (np.int32, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    (np.int64, np.float16): np.float16,
    (np.int64, np.float32): np.float32,
    (np.int64, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    (np.uint8, np.float16): np.float16,
    (np.uint8, np.float32): np.float32,
    (np.uint8, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    (np.uint16, np.float16): np.float16,
    (np.uint16, np.float32): np.float32,
    (np.uint16, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    (np.uint32, np.float16): np.float16,
    (np.uint32, np.float32): np.float32,
    (np.uint32, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    (np.uint64, np.float16): np.float16,
    (np.uint64, np.float32): np.float32,
    (np.uint64, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    # =========================================================================
    # Float promotions - wider wins, special handling for bfloat16
    # Note: float64 is NOT supported
    # =========================================================================
    (np.float16, np.float32): np.float32,
    (np.float16, ml_dtypes.bfloat16): np.float32,  # Different semantics -> f32
    (np.float32, ml_dtypes.bfloat16): np.float32,
    # =========================================================================
    # Float8 promotions - float8 types defer to weak types and other floats
    # Note: Mixing different float8 variants is NOT allowed (ambiguous semantics)
    # =========================================================================
    # float8_e5m2 + weak types -> float8_e5m2
    (_WeakInt, ml_dtypes.float8_e5m2): ml_dtypes.float8_e5m2,
    (_WeakFloat, ml_dtypes.float8_e5m2): ml_dtypes.float8_e5m2,
    (np.bool_, ml_dtypes.float8_e5m2): ml_dtypes.float8_e5m2,
    # float8_e5m2 + integers -> float8_e5m2
    (np.int8, ml_dtypes.float8_e5m2): ml_dtypes.float8_e5m2,
    (np.int16, ml_dtypes.float8_e5m2): ml_dtypes.float8_e5m2,
    (np.int32, ml_dtypes.float8_e5m2): ml_dtypes.float8_e5m2,
    (np.int64, ml_dtypes.float8_e5m2): ml_dtypes.float8_e5m2,
    (np.uint8, ml_dtypes.float8_e5m2): ml_dtypes.float8_e5m2,
    (np.uint16, ml_dtypes.float8_e5m2): ml_dtypes.float8_e5m2,
    (np.uint32, ml_dtypes.float8_e5m2): ml_dtypes.float8_e5m2,
    (np.uint64, ml_dtypes.float8_e5m2): ml_dtypes.float8_e5m2,
    # float8_e5m2 + wider floats -> wider float
    (ml_dtypes.float8_e5m2, np.float16): np.float16,
    (ml_dtypes.float8_e5m2, np.float32): np.float32,
    (ml_dtypes.float8_e5m2, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    # float8_e4m3fn + weak types -> float8_e4m3fn
    (_WeakInt, ml_dtypes.float8_e4m3fn): ml_dtypes.float8_e4m3fn,
    (_WeakFloat, ml_dtypes.float8_e4m3fn): ml_dtypes.float8_e4m3fn,
    (np.bool_, ml_dtypes.float8_e4m3fn): ml_dtypes.float8_e4m3fn,
    # float8_e4m3fn + integers -> float8_e4m3fn
    (np.int8, ml_dtypes.float8_e4m3fn): ml_dtypes.float8_e4m3fn,
    (np.int16, ml_dtypes.float8_e4m3fn): ml_dtypes.float8_e4m3fn,
    (np.int32, ml_dtypes.float8_e4m3fn): ml_dtypes.float8_e4m3fn,
    (np.int64, ml_dtypes.float8_e4m3fn): ml_dtypes.float8_e4m3fn,
    (np.uint8, ml_dtypes.float8_e4m3fn): ml_dtypes.float8_e4m3fn,
    (np.uint16, ml_dtypes.float8_e4m3fn): ml_dtypes.float8_e4m3fn,
    (np.uint32, ml_dtypes.float8_e4m3fn): ml_dtypes.float8_e4m3fn,
    (np.uint64, ml_dtypes.float8_e4m3fn): ml_dtypes.float8_e4m3fn,
    # float8_e4m3fn + wider floats -> wider float
    (ml_dtypes.float8_e4m3fn, np.float16): np.float16,
    (ml_dtypes.float8_e4m3fn, np.float32): np.float32,
    (ml_dtypes.float8_e4m3fn, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    # float8_e4m3 (non-fn variant) + weak types -> float8_e4m3
    (_WeakInt, ml_dtypes.float8_e4m3): ml_dtypes.float8_e4m3,
    (_WeakFloat, ml_dtypes.float8_e4m3): ml_dtypes.float8_e4m3,
    (np.bool_, ml_dtypes.float8_e4m3): ml_dtypes.float8_e4m3,
    # float8_e4m3 + integers -> float8_e4m3
    (np.int8, ml_dtypes.float8_e4m3): ml_dtypes.float8_e4m3,
    (np.int16, ml_dtypes.float8_e4m3): ml_dtypes.float8_e4m3,
    (np.int32, ml_dtypes.float8_e4m3): ml_dtypes.float8_e4m3,
    (np.int64, ml_dtypes.float8_e4m3): ml_dtypes.float8_e4m3,
    (np.uint8, ml_dtypes.float8_e4m3): ml_dtypes.float8_e4m3,
    (np.uint16, ml_dtypes.float8_e4m3): ml_dtypes.float8_e4m3,
    (np.uint32, ml_dtypes.float8_e4m3): ml_dtypes.float8_e4m3,
    (np.uint64, ml_dtypes.float8_e4m3): ml_dtypes.float8_e4m3,
    # float8_e4m3 + wider floats -> wider float
    (ml_dtypes.float8_e4m3, np.float16): np.float16,
    (ml_dtypes.float8_e4m3, np.float32): np.float32,
    (ml_dtypes.float8_e4m3, ml_dtypes.bfloat16): ml_dtypes.bfloat16,
    # Note: Mixing different float8 variants is NOT in the table - will raise error
}


def _get_type_key(val) -> type:
    """Get the type key for promotion table lookup.

    Args:
        val: A value (scalar, array, or tensor)

    Returns:
        Type key for table lookup (_WeakInt, _WeakFloat, or numpy type)
    """
    # Python native scalars are weakly typed
    if isinstance(val, bool):
        # Note: bool must be checked before int since bool is subclass of int
        return np.bool_
    if isinstance(val, int):
        return _WeakInt
    if isinstance(val, float):
        return _WeakFloat

    # Numpy scalars
    if isinstance(val, (np.bool_, np.integer, np.floating)):
        return type(val)

    # Arrays and tensors with dtype attribute
    if hasattr(val, "dtype"):
        dtype = val.dtype
        if isinstance(dtype, np.dtype):
            return dtype.type
        # Handle dtype stored as type class (e.g., np.bool_, np.float32)
        if isinstance(dtype, type):
            return dtype
        # Handle ml_dtypes and other dtype-like objects
        # Try to convert to np.dtype first
        try:
            return np.dtype(dtype).type
        except (TypeError, ValueError):
            pass

    # Explicit NO Fallback
    raise ValueError(f"Unsupported or unrecognized type: {type(val)}")


# Unsupported types that should raise errors
_UNSUPPORTED_TYPES = {np.float64}


def _lookup_promotion(t0: type, t1: type) -> Optional[type]:
    """Look up promotion result in table (checking both orderings).

    Handles:
    - Same-type promotion (A + A = A) without explicit table entries
    - Symmetric lookup (checks both (t0, t1) and (t1, t0))
    - Unsupported types (raises TypeError for float64)

    Args:
        t0: First type key
        t1: Second type key

    Returns:
        Result type or None if not found

    Raises:
        TypeError: If either type is unsupported (e.g., float64)
    """
    # Check for unsupported types
    if t0 in _UNSUPPORTED_TYPES or t1 in _UNSUPPORTED_TYPES:
        unsupported = t0 if t0 in _UNSUPPORTED_TYPES else t1
        raise TypeError(
            f"Unsupported dtype: {np.dtype(unsupported)}."
            " Use .astype(dtype) to convert to a supported type."
        )

    # Same-type promotion: A + A = A (no table entry needed)
    # But NOT for weak types - two weak types cannot be promoted together
    if t0 == t1 and t0 not in (_WeakInt, _WeakFloat):
        return t0

    # Look up in table (check both orderings)
    result = _TYPE_PROMOTION_TABLE.get((t0, t1))
    if result is not None:
        return result
    return _TYPE_PROMOTION_TABLE.get((t1, t0))


def find_common_type_hlo(x, y) -> np.dtype:
    """Find common dtype for two operands in HLO operations.

    Handles weak types (Python scalars) and strong types (numpy arrays/tensors).
    Uses table-based lookup for all promotion decisions.

    Args:
        x: First operand
        y: Second operand

    Returns:
        Common numpy dtype

    Raises:
        TypeError: If types cannot be promoted
    """
    t0 = _get_type_key(x)
    t1 = _get_type_key(y)

    result = _lookup_promotion(t0, t1)
    if result is None:
        t0_name = (
            "weak_int"
            if t0 is _WeakInt
            else ("weak_float" if t0 is _WeakFloat else np.dtype(t0).name)
        )
        t1_name = (
            "weak_int"
            if t1 is _WeakInt
            else ("weak_float" if t1 is _WeakFloat else np.dtype(t1).name)
        )
        raise TypeError(
            f"No implicit dtype promotion path for {t0_name} and {t1_name}."
            " Use .astype(dtype) explicitly."
        )
    return np.dtype(result)


def scalar_dtype_hlo(scalar) -> np.dtype:
    """Infer dtype from a scalar value for HLO operations.

    Note: This returns the "strong" dtype for numpy scalars,
    but Python int/float should use _WeakInt/_WeakFloat via _get_type_key.

    Args:
        scalar: A scalar value

    Returns:
        numpy dtype
    """
    if isinstance(scalar, (bool, np.bool_)):
        return np.dtype(np.bool_)

    if isinstance(scalar, (np.integer, np.floating)):
        # Reject 64-bit types - not supported on hardware
        if isinstance(scalar, np.float64):
            raise ValueError(
                "float64 is not supported on Neuron hardware."
                " Use .astype(np.float32) to convert explicitly."
            )
        if isinstance(scalar, np.int64):
            return np.dtype(np.int32)
        if isinstance(scalar, np.uint64):
            return np.dtype(np.uint32)
        return scalar.dtype

    if isinstance(scalar, int):
        return np.dtype(np.int32)

    if isinstance(scalar, float):
        return np.dtype(np.float32)

    return np.dtype(np.float32)


# =============================================================================
# Context and Helper Functions
# =============================================================================


def get_hlo_context() -> HLOTraceContext:
    """Get the global HLO trace context.

    Returns:
        The current HLO trace context.

    Raises:
        RuntimeError: If no HLO context is available.
    """
    ctx = HLOTraceContext._global_ctx
    if ctx is None:
        raise RuntimeError("No HLO context available.")
    return ctx


def as_hlo_tensor(ctx: HLOTraceContext, value, dtype: np.dtype) -> HLOTensor:
    """Convert a value to an HLO tensor.

    Args:
        ctx: The HLO trace context
        value: Value to convert (scalar, numpy array, or HLOTensor)
        dtype: Target dtype for the tensor

    Returns:
        HLOTensor representation of the value
    """
    if isinstance(value, HLOTensor):
        return value

    if np.isscalar(value):
        op = HLOOp(
            op_name="constant",
            operands=[],
            result_shape=(),
            result_dtype=dtype,
            attributes={"value": float(value)},
        )
        return ctx.module.add_operation(op)

    if isinstance(value, np.ndarray):
        value_array = value.astype(dtype) if value.dtype != dtype else value
        op = HLOOp(
            op_name="constant",
            operands=[],
            result_shape=value_array.shape,
            result_dtype=dtype,
            attributes={"value": value_array},
        )
        return ctx.module.add_operation(op)

    raise ValueError(f"Cannot convert {type(value)} to HLO tensor")


def broadcast_to_shape_hlo(
    ctx: HLOTraceContext, x: HLOTensor, target_shape: Tuple[int, ...]
) -> HLOTensor:
    """Broadcast a tensor to a target shape.

    Args:
        ctx: The HLO trace context
        x: Input tensor to broadcast
        target_shape: Target shape to broadcast to

    Returns:
        Broadcasted HLOTensor
    """
    if not isinstance(x, HLOTensor):
        raise ValueError(f"Expected HLOTensor, got {type(x)}")

    if x.shape == target_shape:
        return x

    # Add leading dimensions if needed
    missing_dims = len(target_shape) - len(x.shape)
    if missing_dims > 0:
        new_shape = (1,) * missing_dims + x.shape
        x = ctx.build_op("reshape", [x], new_shape, x.dtype)

    # Find broadcast dimensions
    broadcast_dims = [
        d for d, (src, tgt) in enumerate(zip(x.shape, target_shape)) if src == tgt
    ]

    # Validate broadcastable
    for d, (src, tgt) in enumerate(zip(x.shape, target_shape)):
        if src != tgt and src != 1:
            raise ValueError(
                f"Cannot broadcast dimension {d} of size {src} to size {tgt}"
            )

    # Squeeze size-1 dimensions
    squeezed_shape = tuple(s for d, s in enumerate(x.shape) if d in broadcast_dims)
    if squeezed_shape != x.shape:
        x = ctx.build_op("reshape", [x], squeezed_shape, x.dtype)

    return ctx.build_op(
        "broadcast",
        [x],
        target_shape,
        x.dtype,
        {"broadcast_dimensions": broadcast_dims},
    )


def broadcast_operands_hlo(
    ctx: HLOTraceContext, x: HLOTensor, y: HLOTensor
) -> Tuple[HLOTensor, HLOTensor]:
    """Broadcast two tensors to compatible shapes.

    Args:
        ctx: The HLO trace context
        x: First tensor
        y: Second tensor

    Returns:
        Tuple of (x_broadcast, y_broadcast) with compatible shapes
    """
    if not isinstance(x, HLOTensor) or not isinstance(y, HLOTensor):
        raise ValueError("Both operands must be HLOTensor")

    if x.shape == y.shape:
        return x, y

    target_shape = tuple(np.broadcast_shapes(x.shape, y.shape))

    if x.shape != target_shape:
        x = broadcast_to_shape_hlo(ctx, x, target_shape)
    if y.shape != target_shape:
        y = broadcast_to_shape_hlo(ctx, y, target_shape)

    return x, y


class HLOTraceContext:
    """Trace Context for building HLO operations.

    Implements the TraceContext interface for HLO backend tracing.
    """

    _global_ctx: Optional[HLOTraceContext] = None

    def __init__(self, module: HLOModule) -> None:
        self.module = module
        self.current_source_location: Optional[Tuple[str, int]] = None

    def set_source_location(self, location: Optional[Tuple[str, int]]) -> None:
        """Set the current source location for operations."""
        self.current_source_location = location

    def get_source_location(self) -> Optional[Tuple[str, int]]:
        """Get the current source location."""
        return self.current_source_location

    def build_op(
        self,
        op_name: str,
        operands: List[HLOTensor],
        result_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]],
        result_dtype: Union[np.dtype, List[np.dtype]],
        attributes: Optional[Dict[str, Any]] = None,
        source_location: Optional[Tuple[str, int]] = None,
    ) -> HLOTensor:
        """Build an HLO operation.

        Args:
            op_name: Name of the HLO operation
            operands: List of input tensors
            result_shape: Shape of the result tensor
            result_dtype: Data type of the result
            attributes: Optional operation attributes
            source_location: Optional source location for debug info

        Returns:
            Result tensor from the operation
        """
        op = HLOOp(
            op_name=op_name,
            operands=operands,
            result_shape=result_shape,
            result_dtype=result_dtype,
            attributes=attributes or {},
            source_location=source_location or self.current_source_location,
        )
        return self.module.add_operation(op)
