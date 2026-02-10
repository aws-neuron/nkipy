# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from neuronxcc.nki.language import float8_e5m2

from nkipy.core.language import bfloat16
from spiky.utils.names import NUMPY_PKG

logger = logging.getLogger(__name__)

DTYPE_MAPPINGS = [
    (np.bool_, torch.bool),
    (np.uint8, torch.uint8),
    (np.int8, torch.int8),
    (np.int16, torch.int16),
    (np.int32, torch.int32),
    (np.uint32, torch.uint32),
    (np.int64, torch.int64),
    (bfloat16.type, torch.bfloat16),
    (float8_e5m2.type, torch.float8_e5m2),
    (np.float16, torch.float16),
    (np.float32, torch.float32),
    (np.float64, torch.float64),
    (np.complex64, torch.complex64),
    (np.complex128, torch.complex128),
]

NUMPY_TO_TORCH_DTYPE_MAP = {
    np_dtype: torch_dtype for np_dtype, torch_dtype in DTYPE_MAPPINGS
}
TORCH_TO_NUMPY_DTYPE_MAP = {
    torch_dtype: np_dtype for np_dtype, torch_dtype in DTYPE_MAPPINGS
}

# Expanded mapping of PyTorch to NumPy dtypes
TORCH_TO_NUMPY_DTYPE_STR_MAP = {
    torch.bool: "bool",
    torch.uint8: f"{NUMPY_PKG}.uint8",
    torch.int8: f"{NUMPY_PKG}.int8",
    torch.int16: f"{NUMPY_PKG}.int16",
    torch.int32: f"{NUMPY_PKG}.int32",
    torch.uint32: f"{NUMPY_PKG}.uint32",
    torch.int64: f"{NUMPY_PKG}.int64",
    torch.bfloat16: "bfloat16",
    torch.float8_e5m2: "float8_e5m2",
    torch.float16: f"{NUMPY_PKG}.float16",
    torch.float32: f"{NUMPY_PKG}.float32",
    torch.float64: f"{NUMPY_PKG}.float64",
    torch.complex64: f"{NUMPY_PKG}.complex64",
    torch.complex128: f"{NUMPY_PKG}.complex128",
}


def numpy_to_torch_dtype(np_dtype: Union[type, np.dtype]) -> torch.dtype:
    """
    Convert a NumPy dtype to its PyTorch equivalent.

    Args:
        np_dtype: A NumPy data type (can be np.dtype instance or type)

    Returns:
        The equivalent PyTorch dtype

    Raises:
        ValueError: If the dtype is not supported
    """
    if isinstance(np_dtype, np.dtype):
        np_dtype = np_dtype.type

    if np_dtype not in NUMPY_TO_TORCH_DTYPE_MAP:
        raise ValueError(f"Unsupported NumPy dtype: {np_dtype}.")

    return NUMPY_TO_TORCH_DTYPE_MAP[np_dtype]


def torch_to_numpy_dtype_str(torch_dtype) -> str:
    """
    Convert a PyTorch dtype to its NumPy string representation.

    Args:
        torch_dtype: The PyTorch dtype

    Returns:
        String representation of the corresponding NumPy dtype

    Raises:
        ValueError: If the dtype is not supported
    """
    if torch_dtype not in TORCH_TO_NUMPY_DTYPE_STR_MAP:
        logger.error(f"Unsupported dtype: {torch_dtype}")
        raise ValueError(
            f"Unsupported dtype: {torch_dtype}. "
            f"Supported dtypes: {list(TORCH_TO_NUMPY_DTYPE_STR_MAP.keys())}"
        )
    return TORCH_TO_NUMPY_DTYPE_STR_MAP[torch_dtype]


def meta_tensor_to_numpy(meta_tensor: torch.Tensor) -> np.ndarray:
    """
    Create an empty NumPy array matching the shape and dtype of a meta tensor.

    This is useful when you need to allocate space for data from a meta tensor
    without having the actual data.

    Args:
        meta_tensor: A PyTorch meta tensor whose shape and dtype to match

    Returns:
        An empty NumPy array with matching shape and dtype

    Raises:
        ValueError: If the tensor dtype cannot be converted to NumPy
    """
    shape = meta_tensor.shape

    if meta_tensor.dtype not in TORCH_TO_NUMPY_DTYPE_MAP:
        raise ValueError(f"Unsupported dtype for meta tensor: {meta_tensor.dtype}. ")

    numpy_dtype = TORCH_TO_NUMPY_DTYPE_MAP[meta_tensor.dtype]

    return np.empty(shape, dtype=numpy_dtype)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Special handling for bfloat16 tensors.

    Args:
        tensor: PyTorch tensor to convert

    Returns:
        NumPy array
    """
    if not isinstance(tensor, torch.Tensor):
        logger.warning(f"Expected torch.Tensor, got {type(tensor)}. Returning as is.")
        return tensor

    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy().astype(bfloat16)
    else:
        return tensor.numpy()


def numpy_to_tensor(array: Any) -> torch.Tensor:
    """
    Convert a NumPy array or scalar to a PyTorch tensor.

    Special handling for bfloat16 arrays.

    Args:
        array: NumPy array or NumPy scalar to convert

    Returns:
        PyTorch tensor
    """
    if array is None:
        return None

    if isinstance(array, np.generic):
        if array.dtype == bfloat16:
            return torch.tensor(array.astype(np.float32).item()).bfloat16()
        else:
            return torch.tensor(array.item())

    if not isinstance(array, np.ndarray):
        logger.warning(
            f"Expected np.ndarray or np.generic, got {type(array)}. Returning as is."
        )
        return array

    if array.dtype == bfloat16:
        return torch.from_numpy(array.astype(np.float32)).bfloat16()
    else:
        return torch.from_numpy(array)


def convert_numpy_arrays_to_tensors(
    data: Union[np.ndarray, np.generic, List[Any], Tuple[Any, ...], None],
) -> Union[torch.Tensor, List[Any], Tuple[Any, ...], None]:
    """
    Convert NumPy arrays, scalars, or nested structures to PyTorch tensors.

    Handles nested lists/tuples and None values.

    Args:
        data: NumPy array/scalar, None, or list/tuple of such items

    Returns:
        PyTorch tensor, None, or list/tuple of PyTorch tensors
    """
    if data is None:
        return None

    if isinstance(data, (list, tuple)):
        result = [convert_numpy_arrays_to_tensors(item) for item in data]
        return type(data)(result)

    elif isinstance(data, (np.ndarray, np.generic)):
        return numpy_to_tensor(data)

    else:
        logger.error(f"Unsupported type: {type(data)}")
        raise TypeError(
            f"Input must be a NumPy ndarray, scalar, None, or list/tuple of them. "
            f"Got {type(data)} instead."
        )
