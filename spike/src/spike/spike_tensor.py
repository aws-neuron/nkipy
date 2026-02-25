from typing import Tuple

import numpy as np

from ._spike import NrtTensor
from .spike_singleton import get_spike_singleton


class SpikeTensor:
    """A tensor on device"""

    def __init__(
        self,
        tensor_ref: NrtTensor,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        name: str = None,
    ):
        """Initialize a SpikeTensor.

        Args:
            tensor_ref: Reference to the device tensor
            shape: Shape of the tensor
            dtype: Data type of the tensor, in numpy format
            name: Name of the tensor
        """
        shape = (shape,) if isinstance(shape, int) else tuple(shape)

        self.tensor_ref = tensor_ref
        self.shape = shape
        self.dtype = dtype
        self.name = name

    @classmethod
    def from_numpy(
        cls, array: np.ndarray, name: str = None, core_id=0
    ) -> "SpikeTensor":
        """Create a SpikeTensor from a numpy array.

        Args:
            array: Input numpy array
            name: Optional name for the tensor

        Returns:
            SpikeTensor instance
        """
        array = np.ascontiguousarray(array)

        spike = get_spike_singleton()
        tensor_ref = spike.allocate_tensor(
            size=array.nbytes, core_id=core_id, name=name
        )
        spike.tensor_write_from_pybuffer(tensor_ref, array)
        return cls(
            tensor_ref=tensor_ref, shape=array.shape, dtype=array.dtype, name=name
        )

    def numpy(self) -> np.ndarray:
        """Read the tensor data back as a numpy array.

        Returns:
            numpy array with the tensor data
        """

        array = np.empty(self.shape, dtype=self.dtype)
        get_spike_singleton().tensor_read_to_pybuffer(self.tensor_ref, array)
        return array

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(shape={self.shape}, dtype={self.dtype}, name={self.name})"
        )
