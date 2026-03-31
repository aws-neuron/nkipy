# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for write_from_numpy (SpikeTensor) and write_from_torch (DeviceTensor).
"""

import numpy as np
import pytest
from nkipy.runtime import is_neuron_compatible


@pytest.mark.skipif(
    not is_neuron_compatible(),
    reason="Need at least 1 Neuron core for device tensor tests",
)
class TestWriteFromNumpy:
    """Tests for SpikeTensor.write_from_numpy inherited by DeviceTensor."""

    def test_roundtrip(self):
        from nkipy.runtime.device_tensor import DeviceTensor

        original = np.array([1, 2, 3, 4], dtype=np.int32)
        dt = DeviceTensor.from_numpy(original)

        new_data = np.array([5, 6, 7, 8], dtype=np.int32)
        dt.write_from_numpy(new_data)

        np.testing.assert_array_equal(dt.numpy(), new_data)

    def test_2d(self):
        from nkipy.runtime.device_tensor import DeviceTensor

        dt = DeviceTensor.from_numpy(np.zeros((4, 4), dtype=np.float32))

        new_data = np.eye(4, dtype=np.float32)
        dt.write_from_numpy(new_data)

        np.testing.assert_array_equal(dt.numpy(), new_data)

    def test_multiple_writes(self):
        from nkipy.runtime.device_tensor import DeviceTensor

        dt = DeviceTensor.from_numpy(np.array([0], dtype=np.int32))

        for i in range(5):
            expected = np.array([i], dtype=np.int32)
            dt.write_from_numpy(expected)
            np.testing.assert_array_equal(dt.numpy(), expected)

    def test_size_mismatch(self):
        from nkipy.runtime.device_tensor import DeviceTensor

        dt = DeviceTensor.from_numpy(np.array([1, 2], dtype=np.int32))

        with pytest.raises(ValueError, match="Size mismatch"):
            dt.write_from_numpy(np.array([1, 2, 3], dtype=np.int32))

    def test_preserves_metadata(self):
        from nkipy.runtime.device_tensor import DeviceTensor

        dt = DeviceTensor.from_numpy(
            np.ones((2, 3), dtype=np.float32), name="test_tensor"
        )

        dt.write_from_numpy(np.zeros((2, 3), dtype=np.float32))

        assert dt.shape == (2, 3)
        assert dt.dtype == np.float32
        assert dt.name == "test_tensor"

    def test_non_contiguous_input(self):
        """Non-contiguous array is handled correctly (made contiguous internally)."""
        from nkipy.runtime.device_tensor import DeviceTensor

        dt = DeviceTensor.from_numpy(np.zeros((3,), dtype=np.float32))

        # Slice creates a non-contiguous view
        source = np.array([1.0, 99.0, 2.0, 99.0, 3.0], dtype=np.float32)[::2]
        assert not source.flags["C_CONTIGUOUS"]

        dt.write_from_numpy(source)
        np.testing.assert_array_equal(dt.numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))


@pytest.mark.skipif(
    not is_neuron_compatible(),
    reason="Need at least 1 Neuron core for device tensor tests",
)
class TestWriteFromTorch:
    """Tests for DeviceTensor.write_from_torch."""

    def test_roundtrip(self):
        import torch
        from nkipy.runtime.device_tensor import DeviceTensor

        dt = DeviceTensor.from_torch(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

        new_data = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
        dt.write_from_torch(new_data)

        torch.testing.assert_close(dt.torch(), new_data)

    def test_2d(self):
        import torch
        from nkipy.runtime.device_tensor import DeviceTensor

        dt = DeviceTensor.from_torch(torch.zeros(4, 4, dtype=torch.float16))

        new_data = torch.eye(4, dtype=torch.float16)
        dt.write_from_torch(new_data)

        torch.testing.assert_close(dt.torch(), new_data)

    def test_multiple_writes(self):
        import torch
        from nkipy.runtime.device_tensor import DeviceTensor

        dt = DeviceTensor.from_torch(torch.tensor([0.0], dtype=torch.float32))

        for i in range(5):
            expected = torch.tensor([float(i)], dtype=torch.float32)
            dt.write_from_torch(expected)
            torch.testing.assert_close(dt.torch(), expected)

    def test_size_mismatch(self):
        import torch
        from nkipy.runtime.device_tensor import DeviceTensor

        dt = DeviceTensor.from_torch(torch.tensor([1.0, 2.0], dtype=torch.float32))

        with pytest.raises(ValueError, match="Size mismatch"):
            dt.write_from_torch(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

    def test_preserves_metadata(self):
        import torch
        from nkipy.runtime.device_tensor import DeviceTensor

        dt = DeviceTensor.from_torch(
            torch.ones(2, 3, dtype=torch.float16), name="test_tensor"
        )

        dt.write_from_torch(torch.zeros(2, 3, dtype=torch.float16))

        assert dt.shape == (2, 3)
        assert dt.dtype == np.float16
        assert dt.name == "test_tensor"

    def test_from_numpy_then_write_torch(self):
        """Tensor created via from_numpy can be updated via write_from_torch."""
        import torch
        from nkipy.runtime.device_tensor import DeviceTensor

        dt = DeviceTensor.from_numpy(np.zeros(4, dtype=np.float32))

        new_data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        dt.write_from_torch(new_data)

        np.testing.assert_array_equal(
            dt.numpy(), np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
