# Spike Async/Nonblock API

This document describes the asynchronous and non-blocking APIs added to the Spike runtime.

## Overview

The Spike runtime supports two levels of asynchronous operation:

1. **High-level Async API** (Recommended): Python asyncio-style interface with automatic dependency management
2. **Low-level Nonblock API** (Advanced): Direct control over non-blocking operations with manual polling

**Most users should use the high-level Async API**, which provides a cleaner interface and automatic dependency management.

## When to Use SpikeAsync

### Overlapping Operations for Better Hardware Utilization

A typical inference task follows a sequential pattern: **tensor write → execute → tensor read**. Done naively, each step blocks the next, leaving hardware idle.

SpikeAsync lets you overlap these stages across iterations so that, while one iteration is executing on the device, the next is writing its inputs and the previous is reading its outputs:

```
tensor_write -> execute -> tensor_read
               tensor_write -> execute -> tensor_read
                              tensor_write -> execute -> tensor_read
```

A more advanced example is **dynamic multi-LoRA**, where you can overlap loading an adapter from host memory with the execution of the current request — hiding the adapter load latency entirely.

With SpikeAsync, this looks like natural sequential code:

```python
async def inference(model, input_data, input_tensor, output_tensor):
    await spike_async.tensor_write(input_tensor, input_data)
    await spike_async.execute(model, input_set, output_set)
    return await spike_async.tensor_read(output_tensor)

for i in range(n):
    spike_async.submit(inference(model, data[i], in_tensor[i], out_tensor[i]))
```

### Why Not Just Use Stream APIs?

You may wonder whether CUDA-style stream APIs already solve this. They do provide asynchronous dispatch, but two problems make them awkward for this pattern.

#### Problem 1: CPU Work Cannot Overlap Naturally

Streams sequence GPU/device operations, but inserting CPU logic between async operations breaks the overlap. Consider a loop that runs a matmul on device, copies the result to host, then runs a CPU function on it:

```cpp
// Attempt with CUDA streams
for (int i = 0; i < n; ++i) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    matmul<<<..., stream>>>(dev_out[i], dev_in[i]);
    cudaMemcpyAsync(host_out[i], dev_out[i], stream);
    cpu_function(host_out[i]);  // WRONG: runs before matmul/copy finish
}
```

Adding `cudaStreamSynchronize` fixes correctness but kills overlap — the CPU waits for each iteration before the next begins:

```
Matmul -> Copy -> CPU
                      Matmul -> Copy -> CPU
                                            Matmul -> Copy -> CPU
```

Pulling the CPU work out after a `cudaDeviceSynchronize` improves device overlap, but all CPU work is serialized at the end — you cannot overlap CPU work with device work:

```
Matmul -> Copy
          Matmul -> Copy
                    Matmul -> Copy
                                    CPU -> CPU -> CPU
```

The core issue is that **stream APIs have no mechanism to resume CPU-side logic precisely when a specific prior operation completes**, without callbacks or restructuring the program into a state machine.

SpikeAsync achieves the fully-overlapped target pipeline naturally:

```
Matmul -> Copy -> CPU
          Matmul -> Copy -> CPU
                    Matmul -> Copy -> CPU
```

```python
async def pipeline(dev_in, dev_out):
    await spike_async.execute(matmul_model, dev_in, dev_out)
    host_out = await spike_async.tensor_read(dev_out)
    cpu_function(host_out)

for i in range(n):
    spike_async.submit(pipeline(dev_in[i], dev_out[i]))
```

Each `await` suspends only that coroutine until the awaited operation finishes, while other submitted coroutines continue to make progress — including their CPU work.

#### Problem 2: Fine-Grained Dependency Control Is Unnatural

Stream APIs model dependencies coarsely: all operations in a stream are ordered, and cross-stream synchronization requires explicit events between streams. Consider this dependency graph, which arises naturally when two independent models feed into a third, while a fourth model only needs the second:

```
model_a ──┐
          ├──► model_c
model_b ──┘
   │
   └─────────► model_d
```

With CUDA streams, you must manually create streams, record events, and wire up waits:

```cpp
cudaStream_t s1, s2, s3, s4;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);
cudaStreamCreate(&s3);
cudaStreamCreate(&s4);

// Run model_a and model_b in parallel
run_model<<<..., s1>>>(model_a, ...);
run_model<<<..., s2>>>(model_b, ...);

// Record events to mark completion
cudaEvent_t event_a, event_b;
cudaEventCreate(&event_a);
cudaEventCreate(&event_b);
cudaEventRecord(event_a, s1);
cudaEventRecord(event_b, s2);

// model_c waits for both a and b
cudaStreamWaitEvent(s3, event_a);
cudaStreamWaitEvent(s3, event_b);
run_model<<<..., s3>>>(model_c, ...);

// model_d waits only for b — easy to accidentally add event_a here too
cudaStreamWaitEvent(s4, event_b);
run_model<<<..., s4>>>(model_d, ...);

// Cleanup
cudaEventDestroy(event_a);
cudaEventDestroy(event_b);
cudaStreamDestroy(s1); cudaStreamDestroy(s2);
cudaStreamDestroy(s3); cudaStreamDestroy(s4);
```

The dependency structure is buried in scattered `cudaStreamWaitEvent` calls. A misplaced wait or a forgotten event silently introduces wrong ordering or unnecessary serialization.

With SpikeAsync's `deps=` parameter, the same graph is expressed directly:

```python
fut_a = spike_async.execute(model_a, in_a, out_a)
fut_b = spike_async.execute(model_b, in_b, out_b)

# model_c starts only after both a and b finish
fut_c = spike_async.execute(model_c, in_c, out_c, deps=[fut_a, fut_b])

# model_d starts as soon as b finishes, independent of a or c
fut_d = spike_async.execute(model_d, in_d, out_d, deps=[fut_b])
```

Each operation declares exactly what it depends on, co-located with the operation itself. There are no streams to create, no events to record, and no risk of accidentally over-constraining or under-constraining the graph.

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
