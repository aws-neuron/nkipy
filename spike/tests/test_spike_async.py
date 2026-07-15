"""
Tests for Spike async/nonblock functionality.
"""

import os
import time
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

# Matmul shapes per --test-mode, expressed as (x_shape, y_shape) for out = x @ y.
def _matmul_shapes(test_mode):
    if test_mode == "overlapping":
        # Large enough for more overlapping opportunities
        # Run roughly as long as the tensor_read/write for a good pipeline
        return (128, 32768), (32768, 65536)
    else:
        # Small enough for a CPU reference check
        return (128, 256), (256, 128)


@pytest.fixture(scope="module")
def neff_path(request):
    """Return the path to the pre-compiled matmul NEFF for the current --test-mode.

    The NEFF is chosen based on --test-mode; its shape matches _matmul_shapes:
      correctness  -- matmul_small.neff, small (128x256) x (256x128), fast enough for CPU reference check
      overlapping  -- matmul_large.neff, large, balanced so write+read ~= exec with weights pre-written
    """
    test_mode = request.config.getoption("--test-mode")
    neff_name = "matmul_large.neff" if test_mode == "overlapping" else "matmul_small.neff"
    path = os.path.join(os.path.dirname(__file__), "artifacts", neff_name)
    if not os.path.exists(path):
        pytest.skip(f"NEFF artifact not found: {path}")
    return path


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

    Data and shape are chosen based on --test-mode (see _matmul_shapes):
      correctness  -- small with random values for numerical verification
      overlapping  -- large, with zeros, balanced so write+read ~= exec to demonstrate async overlap
    """
    # Load model
    model = spike_async.load_model(neff_path, core_id=0)

    # Create input arrays for matmul
    test_mode = request.config.getoption("--test-mode")
    x_shape, y_shape = _matmul_shapes(test_mode)
    if test_mode == "overlapping":
        x = np.zeros(x_shape, dtype=np.float16)
        y = np.zeros(y_shape, dtype=np.float16)
    else:  # correctness
        x = np.random.rand(*x_shape).astype(np.float16)
        y = np.random.rand(*y_shape).astype(np.float16)

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
def model_and_tensors_all_cores(request, neff_path, spike_async):
    """Like model_and_tensors, but allocates one pipeline per visible NeuronCore.

    Loads the model and allocates input/output tensors on every core so that one
    inference pipeline can run concurrently on each core. Mirrors the structure of
    model_and_tensors, with all per-pipeline lists indexed by core_id.

    Each core hosts --num-pipelines independent pipelines so that multiple
    inferences can overlap within a core (e.g. one pipeline's tensor_read can run
    while another's execute is in flight). Per-pipeline lists are nested and
    indexed [core_id][pipeline_idx]; per-core lists (models, inputs_y) are indexed
    [core_id]. The weights `y` are pre-written once per core (mimicking ML model
    parameters loaded ahead of inference) and shared by all pipelines on that core,
    so each pipeline only writes its own `x` and reads its own output. Shapes come
    from _matmul_shapes:
      correctness  -- small (128x256) x (256x128) with random values for numerical verification
      overlapping  -- large, balanced so write x + read out ~= exec (y excluded, pre-written)
    """
    num_cores = Spike.get_visible_neuron_core_count()
    num_pipelines = request.config.getoption("--num-pipelines")

    # Create input arrays for matmul
    test_mode = request.config.getoption("--test-mode")
    x_shape, y_shape = _matmul_shapes(test_mode)
    if test_mode == "overlapping":
        x = np.zeros(x_shape, dtype=np.float16)
        y = np.zeros(y_shape, dtype=np.float16)
    else:  # correctness
        x = np.random.rand(*x_shape).astype(np.float16)
        y = np.random.rand(*y_shape).astype(np.float16)

    output_shape = (x.shape[0], y.shape[1])
    output_size = np.prod(output_shape) * x.dtype.itemsize

    # Per-core lists (one entry per core)
    models = []
    inputs_y = []
    # Per-pipeline lists (nested: [core_id][pipeline_idx])
    inputs_x = []
    input_sets = []
    outputs = []
    output_sets = []
    output_arrays = []  # Pre-allocated CPU buffers for reading outputs
    preload_futs = []
    for core_id in range(num_cores):
        models.append(spike_async.load_model(neff_path, core_id=core_id))

        # One shared weights tensor per core, pre-written once and reused by every
        # pipeline on the core.
        input_y = spike_async.allocate_tensor(size=y.nbytes, core_id=core_id, name=f"input_y_{core_id}")
        inputs_y.append(input_y)
        preload_futs.append(spike_async.tensor_write(input_y, y))

        # Independent x/output tensors per pipeline so pipelines don't alias buffers.
        core_inputs_x = []
        core_input_sets = []
        core_outputs = []
        core_output_sets = []
        core_output_arrays = []
        for p in range(num_pipelines):
            input_x = spike_async.allocate_tensor(size=x.nbytes, core_id=core_id, name=f"input_x_{core_id}_{p}")
            core_inputs_x.append(input_x)
            core_input_sets.append(spike_async.create_tensor_set({"x": input_x, "y": input_y}))

            out = spike_async.allocate_tensor(size=output_size, core_id=core_id, name=f"output_{core_id}_{p}")
            core_outputs.append(out)
            core_output_sets.append(spike_async.create_tensor_set({"output0": out}))
            # Pre-allocate CPU array for zero-copy tensor_read
            core_output_arrays.append(np.empty(output_shape, dtype=x.dtype))

        inputs_x.append(core_inputs_x)
        input_sets.append(core_input_sets)
        outputs.append(core_outputs)
        output_sets.append(core_output_sets)
        output_arrays.append(core_output_arrays)

    # Wait for all weight pre-writes to land before any test runs.
    for fut in preload_futs:
        fut.wait()

    print(f'Done initializing {num_pipelines} pipeline(s) each on {num_cores} cores.')

    return models, x, y, inputs_x, inputs_y, input_sets, outputs, output_sets, output_arrays


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


def test_all_cores_concurrent_perf(request, spike_async, model_and_tensors_all_cores):
    """Performance test: run tensor_write -> execute -> tensor_read concurrently on all cores.

    Submits --num-pipelines coroutine pipelines per core (using tensors from the
    model_and_tensors_all_cores fixture). Running multiple pipelines concurrently on
    a single core lets independent operations overlap -- e.g. one pipeline's
    tensor_read or tensor_write (DMA-bound) can run while another pipeline's execute
    (compute-bound) is in flight -- which is the benefit of the async system.

    Two configurations are compared:
      single-core -- all pipelines on core 0 only
      all-cores   -- the same per-core pipelines on every visible core, concurrently

    Because the cores execute independently in parallel, a zero-overhead coroutine
    system would finish the all-cores run in the same time as the single-core run;
    the difference quantifies the event-loop/coroutine scheduling overhead.

    Each configuration is run for --warmup-iters untimed warmup iterations followed
    by --measure-iters timed iterations; the reported time is the mean over the
    measured iterations.
    """
    models, x, y, inputs_x, inputs_y, input_sets, outputs, output_sets, output_arrays = model_and_tensors_all_cores
    num_cores = len(models)
    if num_cores < 1:
        pytest.skip("No visible NeuronCores available")

    num_pipelines = request.config.getoption("--num-pipelines")
    warmup_iters = request.config.getoption("--warmup-iters")
    measure_iters = request.config.getoption("--measure-iters")
    if measure_iters < 1:
        pytest.skip("--measure-iters must be >= 1")

    expected = np.matmul(x, y)

    # Weights y are pre-written by the fixture (like ML model parameters), so each
    # pipeline only writes its x, executes, and reads its output. With the balanced
    # overlapping shape, write x + read out ~= exec time, so overlapping multiple
    # pipelines on a core can hide DMA behind compute (and vice versa).
    async def inference_pipeline(core_id, p):
        await spike_async.tensor_write(inputs_x[core_id][p], x)
        await spike_async.execute(models[core_id], input_sets[core_id][p], output_sets[core_id][p])
        await spike_async.tensor_read(outputs[core_id][p], output_arrays[core_id][p])

    def run_cores(core_ids):
        """One iteration: num_pipelines pipelines on each given core, all concurrent.

        Returns elapsed seconds for the whole batch to complete.
        """
        start = time.perf_counter()
        futs = [
            spike_async.submit(inference_pipeline(core_id, p))
            for core_id in core_ids
            for p in range(num_pipelines)
        ]
        for fut in futs:
            fut.wait()
        return time.perf_counter() - start

    def measure(core_ids):
        """Run warmup iterations (untimed), then measurement iterations (timed)."""
        for _ in range(warmup_iters):
            run_cores(core_ids)
        return [run_cores(core_ids) for _ in range(measure_iters)]

    def check_outputs(core_ids, label):
        for core_id in core_ids:
            for p in range(num_pipelines):
                assert np.allclose(output_arrays[core_id][p], expected, rtol=1e-3), \
                    f"Output mismatch on core {core_id} pipeline {p} ({label})"

    # Baseline: all pipelines on core 0 only. Since the all-cores run executes the
    # same per-core pipelines on every core concurrently, in a zero-overhead
    # coroutine system the two wall-clock times should be identical. Any difference
    # measures the overhead the coroutine/event-loop system adds when scheduling
    # many concurrent pipelines across cores.
    single_times = measure([0])
    check_outputs([0], "single-core baseline")

    all_core_ids = list(range(num_cores))
    all_times = measure(all_core_ids)
    check_outputs(all_core_ids, "all-cores")

    single_mean = float(np.mean(single_times))
    all_mean = float(np.mean(all_times))
    overhead = all_mean - single_mean
    print(
        f"\n({num_pipelines} pipeline(s)/core; warmup={warmup_iters}, measure={measure_iters} iters; "
        f"times are mean +/- std)\n"
        f"Single-core (core 0, {num_pipelines} pipelines):      "
        f"{single_mean * 1e3:8.2f} +/- {np.std(single_times) * 1e3:6.2f} ms "
        f"({num_pipelines / single_mean:.1f} pipelines/s)\n"
        f"All cores ({num_cores}x{num_pipelines}={num_cores * num_pipelines} pipelines): "
        f"{all_mean * 1e3:8.2f} +/- {np.std(all_times) * 1e3:6.2f} ms "
        f"({num_cores * num_pipelines / all_mean:.1f} pipelines/s)\n"
        f"Coroutine overhead:        {overhead * 1e3:8.2f} ms "
        f"({all_mean / single_mean:.2f}x single-core)"
    )


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