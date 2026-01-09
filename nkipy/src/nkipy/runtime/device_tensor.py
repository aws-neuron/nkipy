# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes

import ml_dtypes
import numpy as np

from spike import SpikeTensor, get_spike_singleton

bfloat16 = np.dtype(ml_dtypes.bfloat16)
float8_e5m2 = np.dtype(ml_dtypes.float8_e5m2)

try:
    import torch
    # Lookup table between torch and numpy dtype
    # only support ml standard types

    torch_to_numpy_type = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.bfloat16: bfloat16,
        torch.float8_e5m2: float8_e5m2,
        torch.int32: np.int32,
        torch.uint32: np.uint32,
    }

    numpy_to_torch_type = {
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("bfloat16"): torch.bfloat16,
        np.dtype("float8_e5m2"): torch.float8_e5m2,
        np.dtype("int32"): torch.int32,
        np.dtype("uint32"): torch.uint32,
        # Also support numpy types directly (not just dtype instances)
        np.float16: torch.float16,
        np.float32: torch.float32,
        bfloat16: torch.bfloat16,
        float8_e5m2: torch.float8_e5m2,
        np.int32: torch.int32,
        np.uint32: torch.uint32,
    }
    _TORCH_ENABLED = True

except ImportError:
    _TORCH_ENABLED = False


class DeviceTensor(SpikeTensor):
    """A tensor on device"""

    @classmethod
    def from_torch(cls, tensor, name: str = None, core_id=0) -> "DeviceTensor":
        if not _TORCH_ENABLED:
            raise ImportError("torch is not available")

        spike = get_spike_singleton()

        tensor = tensor.cpu().contiguous()
        # TODO: note that this assumes the memory layout follows buffer protocol
        # even though PyTorch tensor doesn't explicitly support buffer protocol
        c_array = (ctypes.c_ubyte * tensor.nbytes).from_address(tensor.data_ptr())
        memory_view = memoryview(c_array).cast("B")

        tensor_ref = spike.allocate_tensor(
            size=tensor.nbytes, core_id=core_id, name=name
        )
        spike.tensor_write_from_pybuffer(tensor_ref, memory_view)

        return cls(
            tensor_ref=tensor_ref,
            shape=tensor.shape,
            dtype=torch_to_numpy_type[tensor.dtype],
            name=name,
        )

    def torch(self):
        if not _TORCH_ENABLED:
            raise ImportError("torch is not available")

        torch_dtype = numpy_to_torch_type[self.dtype]

        element_size = torch.tensor([], dtype=torch_dtype).element_size()
        num_elements = np.prod(self.shape) if self.shape else 1
        total_bytes = int(num_elements * element_size)
        buffer = bytearray(total_bytes)

        get_spike_singleton().tensor_read_to_pybuffer(self.tensor_ref, buffer)

        return torch.frombuffer(buffer, dtype=torch_dtype).reshape(self.shape)
