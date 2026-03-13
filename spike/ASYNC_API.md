# Spike Async/Nonblock API

This document describes the asynchronous and non-blocking APIs added to the Spike runtime.

## Overview

The Spike runtime supports two levels of asynchronous operation:

1. **High-level Async API** (Recommended): Python asyncio-style interface with automatic dependency management
2. **Low-level Nonblock API** (Advanced): Direct control over non-blocking operations with manual polling

**Most users should use the high-level Async API**, which provides a cleaner interface and automatic dependency management.

## High-Level Async API

The high-level API provides an asyncio-like interface with automatic dependency tracking and stream support.

### Initialization

```python
from spike import SpikeAsync

spike_async = SpikeAsync(verbose_level=1)
# Nonblock mode is automatically initialized

# Load models and allocate tensors
model = spike_async.load_model("model.neff", core_id=0)
input_tensor = spike_async.allocate_tensor(1024 * 1024, core_id=0)
output_tensor = spike_async.allocate_tensor(1024 * 1024, core_id=0)
```

### Basic Operations

All operations return a Future that can be awaited:

```python
# Tensor operations
write_fut = spike_async.tensor_write(tensor, data)
read_fut = spike_async.tensor_read(tensor)

# Model execution
exec_fut = spike_async.execute(model, input_set, output_set)

# Wait for completion (blocking)
data = read_fut.wait()
```

### Async/Await Style Programming (Recommended)

The most powerful feature is using async/await to write sequential-looking code that runs asynchronously. You can `await` futures directly, making your code clean and readable:

```python
async def inference_pipeline(model, input_data, input_tensor, output_tensor):
    # Write sequential code, but it runs asynchronously!
    # Each await releases control, allowing other operations to run

    # Prepare input data
    await spike_async.tensor_write(input_tensor, input_data)

    # Run inference
    input_set = spike_async.create_tensor_set({"input": input_tensor})
    output_set = spike_async.create_tensor_set({"output": output_tensor})
    await spike_async.execute(model, input_set, output_set)

    # Read results
    output_data = await spike_async.tensor_read(output_tensor)

    return output_data

# Run TWO pipelines concurrently
pipeline1 = spike_async.submit(
    inference_pipeline(model1, data1, input_tensor1, output_tensor1)
)
pipeline2 = spike_async.submit(
    inference_pipeline(model2, data2, input_tensor2, output_tensor2)
)

# Both pipelines are now running concurrently!
# While pipeline1 is executing the model, pipeline2 can do tensor I/O
result1 = pipeline1.wait()
result2 = pipeline2.wait()
```

**Why this is powerful:**
- Code reads like synchronous operations, but executes asynchronously
- Multiple pipelines run concurrently without manual dependency tracking
- Different operation types (tensor I/O, model execution) can overlap, maximizing hardware utilization
- You can mix async Spike operations with regular Python async operations

### Explicit Dependency Management

Alternatively, you can explicitly specify dependencies between operations:

```python
# Write, then read
write_fut = spike_async.tensor_write(tensor, data)
read_fut = spike_async.tensor_read(tensor, deps=[write_fut])

# Chain multiple operations
fut1 = spike_async.execute(model1, in1, out1)
fut2 = spike_async.execute(model2, in2, out2, deps=[fut1])
fut3 = spike_async.execute(model3, in3, out3, deps=[fut1, fut2])
```

This approach is useful when you need fine-grained control or want to build complex DAGs without writing async functions.

### Streams

Streams provide automatic sequencing of operations - all operations in a stream execute in order:

```python
# All operations in a stream are sequenced
with spike_async.create_stream() as stream:
    spike_async.tensor_write(tensor1, data1)
    spike_async.tensor_write(tensor2, data2)
    spike_async.execute(model, inputs, outputs)

# Wait for stream to complete
stream.wait()
```

Multiple streams can run concurrently:

```python
streams = []
for i in range(num_parallel):
    with spike_async.create_stream() as stream:
        # Operations in this stream
        spike_async.execute(models[i], inputs[i], outputs[i])
        streams.append(stream)

# Wait for all streams
for stream in streams:
    stream.wait()
```

### Stream Events

Synchronize between streams using events:

```python
with spike_async.create_stream() as stream1:
    spike_async.tensor_write(tensor, data)
    event = stream1.record_event()  # Capture current point

with spike_async.create_stream() as stream2:
    stream2.wait_event(event)  # Wait for stream1's event
    spike_async.execute(model, inputs, outputs)
```

### Batched Operations

For efficiency, you can batch multiple tensor operations:

**Batched Writes:**
```python
# Prepare a batch of writes
batch_id = spike_async.spike.tensor_write_nonblock_batched_prepare(
    tensors=[tensor1, tensor2, tensor3],
    data_objs=[data1, data2, data3],
    offsets=[0, 0, 0]  # Optional
)

# Start the batch operation (returns a single future)
write_fut = spike_async.tensor_write_batched_start(batch_id)
write_fut.wait()  # Wait for all writes to complete
```

**Batched Reads:**
```python
# Prepare a batch of reads
batch_id = spike_async.spike.tensor_read_nonblock_batched_prepare(
    tensors=[tensor1, tensor2, tensor3],
    dests=[dest1, dest2, dest3],  # Pre-allocated numpy arrays
    offsets=[0, 0, 0],  # Optional
    sizes=[size1, size2, size3]  # Optional
)

# Start the batch operation (returns a single future)
read_fut = spike_async.tensor_read_batched_start(batch_id)
read_fut.wait()  # Wait for all reads to complete
```

Batched operations reduce overhead when you have many small operations.

### Waiting for Multiple Operations

```python
# Launch multiple operations
futs = []
for i in range(10):
    fut = spike_async.execute(model, inputs[i], outputs[i])
    futs.append(fut)

# Wait for all
all_results = await spike_async.all(futs)
```

## Low-Level Nonblock API

The low-level API provides direct access to non-blocking operations. You must manually poll for completion. **Use this only if you need fine-grained control.**

### Initialization

```python
from spike import Spike

spike = Spike(verbose_level=1)
spike.init_nonblock()  # Initialize thread pools for async operations
```

### Nonblocking Operations

**Tensor Write:**
```python
# Returns operation ID
write_id = spike.tensor_write_nonblock(tensor, data, offset=0)
```

**Tensor Read:**
```python
# Returns operation ID
read_id = spike.tensor_read_nonblock(tensor, offset=0, size=0)
```

**Model Execution:**
```python
input_set = spike.create_tensor_set({"input": input_tensor})
output_set = spike.create_tensor_set({"output": output_tensor})
exec_id = spike.execute_nonblock(model, input_set, output_set)
```

### Polling for Results

```python
while True:
    result = spike.try_poll()  # Non-blocking poll
    if result is not None:
        # Check result type
        if isinstance(result, NonBlockTensorWriteResult):
            if result.err is None:
                print(f"Write {result.id} completed successfully")
            else:
                print(f"Write {result.id} failed: {result.err}")
        elif isinstance(result, NonBlockTensorReadResult):
            if result.err is None:
                data = result.data  # Retrieved data
            else:
                print(f"Read {result.id} failed: {result.err}")
        elif isinstance(result, NonBlockExecResult):
            if result.err is None:
                print(f"Execution {result.id} completed successfully")
            else:
                print(f"Execution {result.id} failed: {result.err}")

        # Check if this is the operation we're waiting for
        if result.id == target_id:
            break
```

### Batched Operations (Low-Level)

**Batched Writes:**
```python
# Prepare batch
batch_id = spike.tensor_write_nonblock_batched_prepare(
    tensors=[tensor1, tensor2, tensor3],
    data_objs=[data1, data2, data3],
    offsets=[0, 0, 0]  # Optional
)

# Start the batch (returns single operation ID)
write_id = spike.tensor_write_nonblock_batched_start(batch_id)

# Poll for single completion
while True:
    result = spike.try_poll()
    if result is not None and result.id == write_id:
        break
```

**Batched Reads:**
```python
# Prepare batch
batch_id = spike.tensor_read_nonblock_batched_prepare(
    tensors=[tensor1, tensor2, tensor3],
    dests=[dest1, dest2, dest3],  # Pre-allocated numpy arrays
    offsets=[0, 0, 0],  # Optional
    sizes=[size1, size2, size3]  # Optional
)

# Start the batch (returns single operation ID)
read_id = spike.tensor_read_nonblock_batched_start(batch_id)

# Poll for completion
while True:
    result = spike.try_poll()
    if result is not None and result.id == read_id:
        if result.err is None:
            # All reads completed successfully
            # Data is already in dest1, dest2, dest3
            pass
        break
```

## Architecture

### Thread Pools

When `init_nonblock()` is called, Spike creates two thread pools:

- **Tensor threads**: Handle tensor read/write operations (one per NeuronCore)
- **Execution threads**: Handle model execution (one per NeuronCore)

Operations are dispatched to the appropriate thread based on the NeuronCore ID of the tensor or model.

### Result Queue

Completed operations are pushed to a lock-free notification queue. The `try_poll()` method (or the async event loop) checks this queue for results.

### Async Event Loop

The `SpikeAsyncEventLoop` integrates with the nonblock API:

- Futures are registered with operation IDs
- The selector polls `try_poll()` periodically
- When results arrive, corresponding futures are resolved

### InternalResult → Result Pattern

To avoid GIL overheads, the worker threads use `InternalResult` structures that hold shared_ptrs to keep resources alive. When polled from the main thread (with GIL held), these are converted to `Result` structures. This ensures nanobind/Python object destructors run in GIL context, not in worker threads.

## Best Practices

1. **Use the high-level Async API**: It's simpler and handles dependency management automatically
2. **Use async/await for readable concurrent code**: Multiple pipelines can overlap tensor I/O with model execution
3. **Use streams for simple sequencing**: Streams are the easiest way to ensure operations execute in order
4. **Use explicit dependencies for complex graphs**: When you have multiple parallel streams that need to synchronize in complex patterns
5. **Batch operations when possible**: Batched operations reduce overhead for multiple small transfers
6. **Don't mix sync and async**: Once you submit async or non-blocking operations, submitting sync operations may cause unexpected behaviors.

## Migration from Old API

If you're migrating from the old `NeuronPy` codebase:

- `SpikeCore` → `Spike`
- `spike_cpp` module → `_spike` module
- `device_id` → `core_id` (parameter rename)
- API is otherwise compatible at the low level
- High-level `SpikeAsync` API is unchanged
