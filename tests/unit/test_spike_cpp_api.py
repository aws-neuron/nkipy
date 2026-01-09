# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import time

import numpy as np
import pytest

try:
    from nkipy.core.compile import compile_to_neff, trace
    from spike._spike import Spike  # Internal class - not part of public API

    from spike import reset as spike_reset
except ImportError as e:
    pytest.skip(f"Required packages not available: {e}", allow_module_level=True)

from nkipy.runtime import is_neuron_compatible

# Skip all tests if not running on compatible hardware
if not is_neuron_compatible():
    pytest.skip(
        "Skipping all tests: No compatible Neuron hardware detected",
        allow_module_level=True,
    )


@pytest.fixture(scope="module", autouse=True)
def reset_spike_before_and_after_all_tests():
    """Reset spike state before and after all tests in this module."""
    print("\n[Setup] Calling spike_reset() before all tests...")
    spike_reset()
    yield
    print("\n[Teardown] Calling spike_reset() after all tests...")
    spike_reset()


# Simple test kernels
def add_kernel(x, y):
    return x + y


def multiply_kernel(x, y):
    return x * y


def subtract_kernel(x, y):
    return x - y


# This fixture will only be created after the initialization test is completed
@pytest.fixture(scope="module")
def shared_spike_instance():
    """Session-scoped fixture to provide a shared Spike instance"""
    print("\nInitializing shared Spike instance...")
    start_time = time.time()

    spike = Spike()  # RAII: constructor does nrt_init

    elapsed = time.time() - start_time
    print(f"Spike initialized in {elapsed:.2f} seconds")

    yield spike

    print("\nClosing shared Spike instance...")
    spike.close()


@pytest.fixture(scope="module")
def add_neff_model(tmpdir_factory):
    """Create NEFF model for addition tests"""
    tmpdir = tmpdir_factory.mktemp("neff_tests")
    output_dir = str(tmpdir / "test_artifacts_add")

    traced_kernel = trace(add_kernel)
    x = np.ones((1, 1), np.float32)
    y = np.ones((1, 1), np.float32)
    traced_kernel.specialize(x, y)

    neff = compile_to_neff(trace_kernel=traced_kernel, output_dir=output_dir)
    return os.path.abspath(neff)


@pytest.fixture(scope="module")
def mul_neff_model(tmpdir_factory):
    """Create NEFF model for multiplication tests"""
    tmpdir = tmpdir_factory.mktemp("neff_tests")
    output_dir = str(tmpdir / "test_artifacts_mul")

    traced_kernel = trace(multiply_kernel)
    x = np.ones((1, 1), np.float32)
    y = np.ones((1, 1), np.float32)
    traced_kernel.specialize(x, y)

    neff = compile_to_neff(trace_kernel=traced_kernel, output_dir=output_dir)
    return os.path.abspath(neff)


@pytest.fixture(scope="module")
def sub_neff_model(tmpdir_factory):
    """Create NEFF model for subtraction tests"""
    tmpdir = tmpdir_factory.mktemp("neff_tests")
    output_dir = str(tmpdir / "test_artifacts_sub")

    traced_kernel = trace(subtract_kernel)
    x = np.ones((1, 1), np.float32)
    y = np.ones((1, 1), np.float32)
    traced_kernel.specialize(x, y)

    neff = compile_to_neff(trace_kernel=traced_kernel, output_dir=output_dir)
    return os.path.abspath(neff)


# Helper class to track tensor state (since spike doesn't expose is_freed)
class TensorTracker:
    def __init__(self):
        self.freed_tensors = set()

    def mark_freed(self, tensor):
        self.freed_tensors.add(id(tensor))

    def is_freed(self, tensor):
        return id(tensor) in self.freed_tensors


# Helper class to track model state (since spike doesn't expose is_loaded)
class ModelTracker:
    def __init__(self):
        self.loaded_models = set()

    def mark_loaded(self, model):
        self.loaded_models.add(id(model))

    def mark_unloaded(self, model):
        self.loaded_models.discard(id(model))

    def is_loaded(self, model):
        return id(model) in self.loaded_models


@pytest.fixture(scope="module")
def tensor_tracker():
    """Track tensor freed state"""
    return TensorTracker()


@pytest.fixture(scope="module")
def model_tracker():
    """Track model loaded state"""
    return ModelTracker()


# This test will run first - it has order=1
@pytest.mark.order(1)
def test_tensor_lifecycle_on_delete():
    """Test tensor lifecycle when Spike instance and tensor are deleted.

    This test verifies that deleting a Spike instance and its allocated tensor
    works correctly without errors. Uses keep_alive to ensure proper cleanup order.

    Runs before shared_spike_instance is created.
    """
    try:
        s = Spike()
        t = s.allocate_tensor(100)
        del s  # Spike ref removed, but tensor keeps it alive via keep_alive
        del t  # Now Spike can be destroyed
    except Exception as e:
        pytest.fail(f"tensor_lifecycle_on_delete failed: {e}")


@pytest.mark.order(2)
def test_core_initialization():
    """Test core initialization and cleanup - runs BEFORE shared_spike fixture is created

    With RAII design, Spike() constructor initializes NRT automatically.
    """
    # Test constructor (RAII: constructor does nrt_init)
    try:
        spike = Spike()
    except Exception as e:
        pytest.fail(f"Spike constructor failed: {e}")

    # Test close
    try:
        result = spike.close()
        assert result == 0, "close() should return 0 on success"
    except Exception as e:
        pytest.fail(f"close() failed: {e}")

    # Test that we can create a new instance after close
    try:
        spike2 = Spike()
        spike2.close()
    except Exception as e:
        pytest.fail(f"Creating new Spike after close failed: {e}")


# This test will run third - it has order=3
@pytest.mark.order(3)
def test_static_methods():
    """Test static methods"""
    try:
        count = Spike.get_visible_neuron_core_count()
        assert count > 0, "Should have at least one visible neuron core"
    except Exception as e:
        pytest.fail(f"get_visible_neuron_core_count() failed: {e}")


# All remaining tests will use shared_spike and run after the initialization tests
@pytest.mark.order(4)
def test_tensor_operations(shared_spike_instance, tensor_tracker):
    """Test tensor operations using the shared Spike instance"""
    spike = shared_spike_instance

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


@pytest.mark.order(4)
def test_model_basic_operations(shared_spike_instance, add_neff_model, model_tracker):
    """Test basic model operations using the shared Spike instance"""
    spike = shared_spike_instance

    # Test load_model
    try:
        model = spike.load_model(add_neff_model)
        model_tracker.mark_loaded(model)
        assert model_tracker.is_loaded(model), "Model should be loaded"
        assert model.neff_path == add_neff_model, "Model path should match"
    except Exception as e:
        pytest.fail(f"load_model() failed: {e}")

    # Prepare tensors for execution
    x = spike.allocate_tensor(size=4, name="x")
    y = spike.allocate_tensor(size=4, name="y")
    output = spike.allocate_tensor(size=4, name="output0")

    # Write input data (both 1.0f)
    spike.tensor_write(x, np.array([1.0], dtype=np.float32).tobytes(), 0)
    spike.tensor_write(y, np.array([1.0], dtype=np.float32).tobytes(), 0)

    # Test execute (spike returns None)
    try:
        result = spike.execute(model, {"x": x, "y": y}, {"output0": output})
        # spike.execute() returns None
    except Exception as e:
        pytest.fail(f"execute() failed: {e}")

    # Read output and verify result (should be 2.0)
    data_read = spike.tensor_read(output, 0, 4)
    result_value = np.frombuffer(data_read, dtype=np.float32)[0]
    assert result_value == 2.0, f"Expected 2.0, got {result_value}"

    # Test unload_model
    try:
        spike.unload_model(model)
        model_tracker.mark_unloaded(model)
        assert not model_tracker.is_loaded(model), "Model should be unloaded"
    except Exception as e:
        pytest.fail(f"unload_model() failed: {e}")

    # Clean up tensors
    spike.free_tensor(x)
    spike.free_tensor(y)
    spike.free_tensor(output)


@pytest.mark.order(4)
def test_model_benchmark(shared_spike_instance, add_neff_model, model_tracker):
    """Test model benchmark using the shared Spike instance"""
    spike = shared_spike_instance

    model = spike.load_model(add_neff_model)
    model_tracker.mark_loaded(model)
    x = spike.allocate_tensor(size=4, name="x")
    y = spike.allocate_tensor(size=4, name="y")
    output = spike.allocate_tensor(size=4, name="output0")

    # Write input data
    spike.tensor_write(x, np.array([1.0], dtype=np.float32).tobytes(), 0)
    spike.tensor_write(y, np.array([1.0], dtype=np.float32).tobytes(), 0)

    # Test benchmark with minimal iterations
    try:
        result = spike.benchmark(
            model,
            {"x": x, "y": y},
            {"output0": output},
            warmup_iterations=1,
            benchmark_iterations=2,
        )

    except Exception as e:
        pytest.fail(f"benchmark() failed: {e}")

    # Clean up
    spike.unload_model(model)
    model_tracker.mark_unloaded(model)
    spike.free_tensor(x)
    spike.free_tensor(y)
    spike.free_tensor(output)


@pytest.mark.order(4)
def test_complex_workflow(
    shared_spike_instance, add_neff_model, mul_neff_model, model_tracker
):
    """Test a more complex workflow using multiple operations with the shared Spike instance"""
    spike = shared_spike_instance

    # Load models
    add_model = spike.load_model(add_neff_model)
    model_tracker.mark_loaded(add_model)
    mul_model = spike.load_model(mul_neff_model)
    model_tracker.mark_loaded(mul_model)

    # Create tensors
    x = spike.allocate_tensor(size=4, name="x")
    y = spike.allocate_tensor(size=4, name="y")
    add_out = spike.allocate_tensor(size=4, name="add_out")
    z = spike.allocate_tensor(size=4, name="z")
    mul_out = spike.allocate_tensor(size=4, name="mul_out")

    # Write test data: (2 + 3) * 4 = 20
    spike.tensor_write(x, np.array([2.0], dtype=np.float32).tobytes(), 0)
    spike.tensor_write(y, np.array([3.0], dtype=np.float32).tobytes(), 0)
    spike.tensor_write(z, np.array([4.0], dtype=np.float32).tobytes(), 0)

    try:
        # Execute addition
        spike.execute(add_model, {"x": x, "y": y}, {"output0": add_out})

        # Execute multiplication with addition result
        spike.execute(mul_model, {"x": add_out, "y": z}, {"output0": mul_out})

        # Read the final result
        data_read = spike.tensor_read(mul_out, 0, 4)
        result_value = np.frombuffer(data_read, dtype=np.float32)[0]
        assert result_value == 20.0, f"Expected 20.0, got {result_value}"

    except Exception as e:
        pytest.fail(f"Complex workflow failed: {e}")

    # Clean up
    spike.unload_model(add_model)
    model_tracker.mark_unloaded(add_model)
    spike.unload_model(mul_model)
    model_tracker.mark_unloaded(mul_model)
    spike.free_tensor(x)
    spike.free_tensor(y)
    spike.free_tensor(add_out)
    spike.free_tensor(z)
    spike.free_tensor(mul_out)


@pytest.mark.order(4)
def test_get_tensor_info(shared_spike_instance, add_neff_model, model_tracker):
    """Test get_tensor_info API using the shared Spike instance"""
    spike = shared_spike_instance

    # Load the model
    try:
        model = spike.load_model(add_neff_model)
        model_tracker.mark_loaded(model)
        assert model_tracker.is_loaded(model), "Model should be loaded"
    except Exception as e:
        pytest.fail(f"load_model() failed: {e}")

    # Test get_tensor_info
    try:
        tensor_info = spike.get_tensor_info(model)

        # Check structure of returned data - now it's a ModelTensorInfo object
        assert hasattr(tensor_info, "inputs"), (
            "Tensor info should have 'inputs' attribute"
        )
        assert hasattr(tensor_info, "outputs"), (
            "Tensor info should have 'outputs' attribute"
        )

        # Check input tensors
        inputs = tensor_info.inputs
        assert "x" in inputs, "Input tensor 'x' should be present"
        assert "y" in inputs, "Input tensor 'y' should be present"

        # Check output tensors
        outputs = tensor_info.outputs
        assert "output0" in outputs, "Output tensor 'output0' should be present"

        # Check tensor metadata structure - now accessing as attributes, not dict keys
        for name, metadata in inputs.items():
            assert hasattr(metadata, "size"), (
                f"Input tensor '{name}' should have 'size' attribute"
            )
            assert hasattr(metadata, "dtype"), (
                f"Input tensor '{name}' should have 'dtype' attribute"
            )
            assert hasattr(metadata, "shape"), (
                f"Input tensor '{name}' should have 'shape' attribute"
            )

            # For our simple addition model, expect float32 tensors of size 4 (single float)
            assert metadata.size == 4, (
                f"Expected size 4 for tensor '{name}', got {metadata.size}"
            )
            assert metadata.dtype == "float32", (
                f"Expected dtype 'float32' for tensor '{name}', got {metadata.dtype}"
            )
            assert len(metadata.shape) > 0, (
                f"Shape should not be empty for tensor '{name}'"
            )
            assert isinstance(metadata.shape, list), (
                f"Shape should be a list for tensor '{name}'"
            )

        # Check output tensor metadata
        for name, metadata in outputs.items():
            assert hasattr(metadata, "size"), (
                f"Output tensor '{name}' should have 'size' attribute"
            )
            assert hasattr(metadata, "dtype"), (
                f"Output tensor '{name}' should have 'dtype' attribute"
            )
            assert hasattr(metadata, "shape"), (
                f"Output tensor '{name}' should have 'shape' attribute"
            )

            # For our simple addition model, expect float32 tensors of size 4 (single float)
            assert metadata.size == 4, (
                f"Expected size 4 for tensor '{name}', got {metadata.size}"
            )
            assert metadata.dtype == "float32", (
                f"Expected dtype 'float32' for tensor '{name}', got {metadata.dtype}"
            )
            assert len(metadata.shape) > 0, (
                f"Shape should not be empty for tensor '{name}'"
            )
            assert isinstance(metadata.shape, list), (
                f"Shape should be a list for tensor '{name}'"
            )

    except Exception as e:
        pytest.fail(f"get_tensor_info() failed: {e}")

    # Print tensor info for visibility
    print("\nTensor info for addition model:")
    print(
        f"  Inputs: {[(name, f'size={m.size}, dtype={m.dtype}, shape={m.shape}') for name, m in tensor_info.inputs.items()]}"
    )
    print(
        f"  Outputs: {[(name, f'size={m.size}, dtype={m.dtype}, shape={m.shape}') for name, m in tensor_info.outputs.items()]}"
    )

    # Clean up
    spike.unload_model(model)
    model_tracker.mark_unloaded(model)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
