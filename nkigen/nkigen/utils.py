
# Modified from https://github.com/cornell-zhang/allo/blob/e4ababde72803aaf156db2db86820ec817285f50/allo/utils.py

import ctypes
import numpy as np
import ml_dtypes

from mlir.runtime import to_numpy
from mlir.dialects import func as func_d
from mlir.ir import (
    MemRefType,
    RankedTensorType,
    IntegerType,
    IndexType,
    F16Type,
    F32Type,
    F64Type,
    BF16Type,
)

np_supported_types = {
    "bf16": ml_dtypes.bfloat16,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "ui1": np.bool_,
    "ui8": np.uint8,
    "ui16": np.uint16,
    "ui32": np.uint32,
    "ui64": np.uint64,
}

ctype_map = {
    # ctypes.c_float16 does not exist
    # similar implementation in _mlir/runtime/np_to_memref.py/F16
    "bf16": ctypes.c_int16,
    "f16": ctypes.c_int16,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
    "i8": ctypes.c_int8,
    "i16": ctypes.c_int16,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "ui1": ctypes.c_bool,
    "ui8": ctypes.c_uint8,
    "ui16": ctypes.c_uint16,
    "ui32": ctypes.c_uint32,
    "ui64": ctypes.c_uint64,
}

def np_type_to_str(dtype):
    return list(np_supported_types.keys())[
        list(np_supported_types.values()).index(dtype)
    ]

def get_bitwidth_from_type(dtype):
    if dtype == "index":
        return 64
    if dtype.startswith("i"):
        bitwidth = int(dtype[1:])
        assert bitwidth in [8, 16, 32, 64]
        return bitwidth
    if dtype.startswith("ui"):
        bitwidth = int(dtype[2:])
        assert bitwidth in [1, 8, 16, 32, 64]
        return bitwidth
    if dtype.startswith("f"):
        bitwidth = int(dtype[1:])
        assert bitwidth in [16, 32, 64]
        return bitwidth
    raise RuntimeError("Unsupported type")

# Locate top-level func.func entry
def find_func_in_module(module, func_name):
    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp) and op.name.value == func_name:
            return op
    return None

def extract_out_np_arrays_from_out_struct(out_struct_ptr_ptr, num_output):
    out_np_arrays = []
    for i in range(num_output):
        out_np_arrays.append(
            ranked_memref_to_numpy(getattr(out_struct_ptr_ptr[0][0], f"memref{i}"))
        )
    return out_np_arrays

def get_np_struct_type(bitwidth):
    n_bytes = int(np.ceil(bitwidth / 8))
    return np.dtype(
        {
            "names": [f"f{i}" for i in range(n_bytes)],
            # all set to unsigned byte
            "formats": ["u1"] * n_bytes,
            "offsets": list(range(n_bytes)),
            "itemsize": n_bytes,
        }
    )

def ranked_memref_to_numpy(ranked_memref):
    """Converts ranked memrefs to numpy arrays."""
    # Check rank using _length_ to avoid triggering numpy ctypes warning
    rank = ranked_memref.shape._length_

    if rank == 0:
        # Special handling for rank-0 memrefs (scalars) to avoid numpy ctypes warning
        # For rank-0 memrefs, directly read the scalar value
        contentPtr = ctypes.cast(
            ctypes.addressof(ranked_memref.aligned.contents)
            + ranked_memref.offset * ctypes.sizeof(ranked_memref.aligned.contents),
            type(ranked_memref.aligned),
        )
        # Return as a 0-d numpy array (scalar array)
        return np.array(contentPtr[0])

    # A temporary workaround for issue
    # https://discourse.llvm.org/t/setting-memref-elements-in-python-callback/72759
    contentPtr = ctypes.cast(
        ctypes.addressof(ranked_memref.aligned.contents)
        + ranked_memref.offset * ctypes.sizeof(ranked_memref.aligned.contents),
        type(ranked_memref.aligned),
    )
    np_arr = np.ctypeslib.as_array(contentPtr, shape=ranked_memref.shape)
    strided_arr = np.lib.stride_tricks.as_strided(
        np_arr,
        np.ctypeslib.as_array(ranked_memref.shape),
        np.ctypeslib.as_array(ranked_memref.strides) * np_arr.itemsize,
    )
    return to_numpy(strided_arr)


def get_signed_type_by_hint(dtype, hint):
    if hint == "u" and (dtype.startswith("i") or dtype.startswith("fixed")):
        return "u" + dtype
    return dtype


def get_dtype_and_shape_from_type(dtype):
    """
    Extract dtype, shape, and whether it's a memref from an MLIR type.

    Returns:
        tuple: (element_type: str, shape: tuple, is_memref: bool)
    """
    if MemRefType.isinstance(dtype):
        dtype = MemRefType(dtype)
        shape = dtype.shape
        ele_type, _, _ = get_dtype_and_shape_from_type(dtype.element_type)
        return ele_type, shape, True  # is_memref=True
    if RankedTensorType.isinstance(dtype):
        dtype = RankedTensorType(dtype)
        shape = dtype.shape
        ele_type, _, _ = get_dtype_and_shape_from_type(dtype.element_type)
        return ele_type, shape, True  # is_memref=True (will become memref after bufferization)
    if IndexType.isinstance(dtype):
        return "index", tuple(), False
    if IntegerType.isinstance(dtype):
        return str(IntegerType(dtype)), tuple(), False
    if F16Type.isinstance(dtype):
        return str(F16Type(dtype)), tuple(), False
    if F32Type.isinstance(dtype):
        return str(F32Type(dtype)), tuple(), False
    if F64Type.isinstance(dtype):
        return str(F64Type(dtype)), tuple(), False
    if BF16Type.isinstance(dtype):
        return str(BF16Type(dtype)), tuple(), False
    raise RuntimeError("Unsupported type")


# Get input and output type frmo func.func
def get_func_inputs_outputs(func):
    """
    Extract input and output types from func.func.

    Returns:
        tuple: (inputs, outputs) where each element is a list of tuples
               (dtype: str, shape: tuple, is_memref: bool)
    """
    # Input types
    inputs = []
    in_hints = (
        func.attributes["itypes"].value
        if "itypes" in func.attributes
        else "_" * len(func.type.inputs)
    )
    for in_type, in_hint in zip(func.type.inputs, in_hints):
        dtype, shape, is_memref = get_dtype_and_shape_from_type(in_type)
        in_type = get_signed_type_by_hint(dtype, in_hint)
        inputs.append((in_type, shape, is_memref))

    # Output types
    outputs = []
    out_hints = (
        func.attributes["otypes"].value
        if "otypes" in func.attributes
        else "_" * len(func.type.results)
    )
    for out_type, out_hint in zip(func.type.results, out_hints):
        dtype, shape, is_memref = get_dtype_and_shape_from_type(out_type)
        out_type = get_signed_type_by_hint(dtype, out_hint)
        outputs.append((out_type, shape, is_memref))
    return inputs, outputs


def create_output_struct(memref_descriptors):
    fields = [
        (f"memref{i}", memref.__class__) for i, memref in enumerate(memref_descriptors)
    ]
    # Dynamically create and return the new class
    OutputStruct = type("OutputStruct", (ctypes.Structure,), {"_fields_": fields})
    out_struct = OutputStruct()
    for i, memref in enumerate(memref_descriptors):
        setattr(out_struct, f"memref{i}", memref)
    return out_struct