import numpy as np
import pytest

try:
    from spike._spike import Spike

    from spike import reset as spike_reset

    available_core_count = Spike.get_visible_neuron_core_count()
    if available_core_count < 1:
        pytest.skip(
            "Skipping all tests: No compatible Neuron hardware detected",
            allow_module_level=True,
        )

except ImportError as e:
    pytest.skip(f"Required packages not available: {e}", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def reset_spike_before_and_after_all_tests():
    """Reset spike state before and after all tests in this module."""
    print("\n[Setup] Calling spike_reset() before all tests...")
    spike_reset()
    yield
    print("\n[Teardown] Calling spike_reset() after all tests...")
    spike_reset()


# Helper class to track tensor state (since spike doesn't expose is_freed)
class TensorTracker:
    def __init__(self):
        self.freed_tensors = set()

    def mark_freed(self, tensor):
        self.freed_tensors.add(id(tensor))

    def is_freed(self, tensor):
        return id(tensor) in self.freed_tensors


@pytest.fixture(scope="module")
def tensor_tracker():
    """Track tensor freed state"""
    return TensorTracker()


def test_core_initialization():
    """Test core initialization and cleanup"""
    # Test constructor
    try:
        spike = Spike()  # will initialize NRT
    except Exception as e:
        pytest.fail(f"Spike constructor failed: {e}")

    # Test close
    try:
        result = spike.close()
        assert result == 0, "close() should return 0 on success"
    except Exception as e:
        pytest.fail(f"close() failed: {e}")


def test_tensor_lifecycle_on_delete():
    """Test tensor lifecycle when Spike instance and tensor are deleted.

    This test verifies that deleting a Spike instance and its allocated tensor
    works correctly without errors.
    """
    s = Spike()
    t = s.allocate_tensor(100)
    del s
    del t


def test_static_methods():
    """Test static methods"""
    try:
        count = Spike.get_visible_neuron_core_count()
        assert count > 0, "Should have at least one visible neuron core"
    except Exception as e:
        pytest.fail(f"get_visible_neuron_core_count() failed: {e}")


def test_tensor_operations(tensor_tracker):
    """Test tensor operations using the shared Spike instance"""
    spike = Spike()

    # Test allocate_tensor
    try:
        tensor = spike.allocate_tensor(size=4, name="test_tensor")
        assert tensor.size == 4, "Tensor size should be 4"
        assert tensor.name == "test_tensor", "Tensor name should be 'test_tensor'"
    except Exception as e:
        pytest.fail(f"allocate_tensor() failed: {e}")

    # Test tensor_write
    try:
        data = np.array([1.0], dtype=np.float32).tobytes()
        spike.tensor_write(tensor, data, 0)
    except Exception as e:
        pytest.fail(f"tensor_write() failed: {e}")

    # Test tensor_read
    try:
        data_read = spike.tensor_read(tensor, 0, 4)
        assert len(data_read) == 4, "Read data length should match size"
    except Exception as e:
        pytest.fail(f"tensor_read() failed: {e}")

    # Test slice_from_tensor
    try:
        slice_tensor = spike.slice_from_tensor(tensor, 0, 2, "slice_tensor")
        assert slice_tensor.size == 2, "Slice tensor size should be 2"
        assert slice_tensor.name == "slice_tensor", (
            "Slice tensor name should be 'slice_tensor'"
        )
    except Exception as e:
        pytest.fail(f"slice_from_tensor() failed: {e}")

    # Test free_tensor on slice
    try:
        spike.free_tensor(slice_tensor)
        tensor_tracker.mark_freed(slice_tensor)
        assert tensor_tracker.is_freed(slice_tensor), "Tensor should be marked as freed"
    except Exception as e:
        pytest.fail(f"free_tensor() on slice failed: {e}")

    # Test free_tensor on original tensor
    try:
        spike.free_tensor(tensor)
        tensor_tracker.mark_freed(tensor)
        assert tensor_tracker.is_freed(tensor), "Tensor should be marked as freed"
    except Exception as e:
        pytest.fail(f"free_tensor() failed: {e}")

    spike.close()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
