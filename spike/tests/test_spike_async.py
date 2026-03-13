"""
Tests for Spike async/nonblock functionality.

NOTE: These tests require actual NeuronCore hardware and a compiled NEFF model.
They serve as integration tests and usage examples.
"""

import os
import pytest
import numpy as np

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

@pytest.fixture(scope="module")
def neff_path(request):
    """Compile a matmul kernel for testing.

    Shape is chosen based on --test-mode:
      correctness  -- small (128x256) x (256x128), fast enough for CPU reference check
      overlapping  -- large (4096x8192) x (8192x4096), long enough to exhibit async overlap
    """
    from nkipy.core.compile import compile_to_neff, trace

    def my_matmul(x, y):
        return np.matmul(x, y)

    test_mode = request.config.getoption("--test-mode")
    if test_mode == "overlapping":
        x = np.zeros((4096, 8192), dtype=np.float16)
        y = np.zeros((8192, 4096), dtype=np.float16)
    else:  # correctness
        x = np.zeros((128, 256), dtype=np.float16)
        y = np.zeros((256, 128), dtype=np.float16)

    traced_kernel = trace(my_matmul)
    traced_kernel.specialize(x, y)

    output_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    neff_path = compile_to_neff(trace_kernel=traced_kernel, output_dir=output_dir)

    return neff_path


@pytest.fixture(scope="module")
def spike_async():
    """Initialize SpikeAsync for testing."""
    from spike import SpikeAsync

    spike_async = SpikeAsync()
    yield spike_async
    spike_async.spike.close()


@pytest.fixture(scope="module")
def model_and_tensors(request, neff_path, spike_async):
    """Load model and prepare input/output tensors for matmul.

    Data and shape are chosen based on --test-mode:
      correctness  -- small (128x256) x (256x128) with random values for numerical verification
      overlapping  -- large (4096x8192) x (8192x4096) with zeros to demonstrate async overlap
    """
    # Load model
    model = spike_async.load_model(neff_path, core_id=0)

    # Create input arrays for matmul
    test_mode = request.config.getoption("--test-mode")
    if test_mode == "overlapping":
        x = np.zeros((4096, 8192), dtype=np.float16)
        y = np.zeros((8192, 4096), dtype=np.float16)
    else:  # correctness
        x = np.random.rand(128, 256).astype(np.float16)
        y = np.random.rand(256, 128).astype(np.float16)

    num_pipelines = request.config.getoption("--num-pipelines")

    # Allocate input tensors for concurrent pipelines
    inputs_x = []
    inputs_y = []
    input_sets = []
    for i in range(num_pipelines):
        input_x = spike_async.allocate_tensor(size=x.nbytes, core_id=0, name=f"input_x_{i}")
        input_y = spike_async.allocate_tensor(size=y.nbytes, core_id=0, name=f"input_y_{i}")
        inputs_x.append(input_x)
        inputs_y.append(input_y)
        input_sets.append(spike_async.create_tensor_set({"x": input_x, "y": input_y}))

    # Allocate output tensors for concurrent pipelines
    output_shape = (x.shape[0], y.shape[1])
    output_size = np.prod(output_shape) * x.dtype.itemsize
    outputs = []
    output_sets = []
    output_arrays = []  # Pre-allocated CPU buffers for reading outputs
    for i in range(num_pipelines):
        out = spike_async.allocate_tensor(size=output_size, core_id=0, name=f"output_{i}")
        outputs.append(out)
        output_sets.append(spike_async.create_tensor_set({"output0": out}))
        # Pre-allocate CPU array for zero-copy tensor_read
        output_arrays.append(np.empty(output_shape, dtype=x.dtype))

    print('Done initializing inputs, outputs, and model.')

    return model, x, y, inputs_x, inputs_y, input_sets, outputs, output_sets, output_arrays


@pytest.fixture(scope="module")
def expected_output(model_and_tensors):
    """Compute expected matmul output once for all tests."""
    _, x, y, *_ = model_and_tensors
    return np.matmul(x, y)


def test_stream_api(spike_async, model_and_tensors, expected_output):
    """Test stream-based API for operation sequencing."""
    model, x, y, inputs_x, inputs_y, input_sets, outputs, output_sets, output_arrays = model_and_tensors

    # Use stream APIs - each iteration uses different input and output tensors
    streams = []

    for i in range(len(inputs_x)):
        with spike_async.create_stream() as stream:
            spike_async.tensor_write(inputs_x[i], x)
            spike_async.tensor_write(inputs_y[i], y)
            spike_async.execute(model, input_sets[i], output_sets[i])
            spike_async.tensor_read(outputs[i], output_arrays[i])
            streams.append(stream)

    # Verify outputs - data is already in pre-allocated arrays
    for i, stream in enumerate(streams):
        stream.wait()
        assert np.allclose(output_arrays[i], expected_output, rtol=1e-3), f"Output mismatch for pipeline {i}"


def test_explicit_dependency_api(spike_async, model_and_tensors, expected_output):
    """Test explicit dependency management API."""
    model, x, y, inputs_x, inputs_y, input_sets, outputs, output_sets, output_arrays = model_and_tensors

    # Use explicit dependency APIs - each iteration uses different input and output tensors
    read_futs = []

    for i in range(len(inputs_x)):
        write_x_fut = spike_async.tensor_write(inputs_x[i], x)
        write_y_fut = spike_async.tensor_write(inputs_y[i], y)
        exec_fut = spike_async.execute(model, input_sets[i], output_sets[i], deps=[write_x_fut, write_y_fut])
        read_fut = spike_async.tensor_read(outputs[i], output_arrays[i], deps=[exec_fut])
        read_futs.append(read_fut)

    # Verify outputs - data is already in pre-allocated arrays
    for i, read_fut in enumerate(read_futs):
        read_fut.wait()
        assert np.allclose(output_arrays[i], expected_output, rtol=1e-3), f"Output mismatch for pipeline {i}"


def test_coroutine_api(spike_async, model_and_tensors, expected_output):
    """Test coroutine/async-await API."""
    model, x, y, inputs_x, inputs_y, input_sets, outputs, output_sets, output_arrays = model_and_tensors

    async def inference_pipeline(pipeline_idx):
        await spike_async.tensor_write(inputs_x[pipeline_idx], x)
        await spike_async.tensor_write(inputs_y[pipeline_idx], y)
        await spike_async.execute(model, input_sets[pipeline_idx], output_sets[pipeline_idx])
        await spike_async.tensor_read(outputs[pipeline_idx], output_arrays[pipeline_idx])

    # Use coroutine APIs - each pipeline uses different input and output tensors
    futs = []
    for i in range(len(inputs_x)):
        fut = spike_async.submit(inference_pipeline(i))
        futs.append(fut)

    # Verify outputs - data is already in pre-allocated arrays
    for i, fut in enumerate(futs):
        fut.wait()
        assert np.allclose(output_arrays[i], expected_output, rtol=1e-3), f"Output mismatch for pipeline {i}"


def test_tensor_read_write(spike_async):
    """Test basic tensor read/write operations."""
    # Allocate a test tensor
    tensor_size = 1024 * 1024  # 1MB
    tensor = spike_async.allocate_tensor(size=tensor_size, core_id=0, name="test_tensor")

    # Create test data
    test_data = np.random.rand(tensor_size // 8).astype(np.float64)
    test_bytes = test_data.tobytes()

    # Submit async write and read operations
    write_fut = spike_async.tensor_write(tensor, test_bytes)
    read_fut = spike_async.tensor_read(tensor, deps=[write_fut])

    # Wait for completion
    read_data = read_fut.wait()

    # Verify data
    read_array = np.frombuffer(read_data, dtype=np.float64)
    assert np.array_equal(read_array, test_data), "Tensor read/write data mismatch"

if __name__ == "__main__":
    pytest.main(["-v", __file__])