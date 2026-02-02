#!/usr/bin/env python3
"""
Simple NKI Beta2 Kernel Example using DeviceKernel and DeviceTensor
"""

import time

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from nkipy.core import nki_op  # noqa: F401, make sure monkey patch is applied
from nkipy.runtime import DeviceKernel, DeviceTensor


@nki.jit(platform_target="trn2")
def nki_tensor_add(a_input, b_input):
    a_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
    nisa.dma_copy(dst=a_tile, src=a_input)

    b_tile = sbuf.view(dtype=b_input.dtype, shape=b_input.shape)  # noqa: F821
    nisa.dma_copy(dst=b_tile, src=b_input)

    c_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

    c_output = hbm.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
    nisa.dma_copy(dst=c_output, src=c_tile)

    return c_output


#  Create an NKIPy wrapper function
# This is required because DeviceKernel.compile_and_load only supports NKIPy functions
def nkipy_wrapper(a_input, b_input):
    """NKIPy wrapper that calls the NKI kernel."""
    return nki_tensor_add(a_input, b_input)


def main():
    shape = (128, 512)  # Must fit in one SBUF tile for the simple kernel
    warmup_iterations = 10
    benchmark_iterations = 10

    print("\n[1/7] Creating test data...")
    np.random.seed(42)
    a = np.random.rand(*shape).astype(np.float32)
    b = np.random.rand(*shape).astype(np.float32)
    output = np.zeros(shape, dtype=np.float32)
    print(f"  Created tensors a: {a.shape}, b: {b.shape}")

    print("\n[2/7] Compiling kernel...")
    compile_start = time.time()

    # Compile the NKIPy wrapper (not the NKI kernel directly)
    kernel = DeviceKernel.compile_and_load(
        nkipy_wrapper,  # Pass the wrapper, not nki_tensor_add
        a,
        b,
        name="nki_tensor_add",
        use_cached_if_exists=False,  # Always recompile for this example
    )

    compile_time = time.time() - compile_start
    print(f"  Kernel compiled in {compile_time:.2f} seconds")
    print(f"  Kernel name: {kernel.name}")

    print("\n[3/7] Creating device tensors...")
    device_a = DeviceTensor.from_numpy(a)
    device_b = DeviceTensor.from_numpy(b)
    device_output = DeviceTensor.from_numpy(output)
    print("  Created device tensors for inputs and output")

    print("\n[4/7] Executing kernel...")
    kernel(
        inputs={"a_input": device_a, "b_input": device_b},
        outputs={"output0": device_output},
    )
    result = device_output.numpy()
    print("  Kernel executed successfully")
    print(f"  Output shape: {result.shape}, dtype: {result.dtype}")

    print("\n[5/7] Validating correctness...")
    reference = a + b

    try:
        np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-5)
        print("  Output matches NumPy reference within tolerance")

        # Calculate relative error
        rel_error = np.abs(result - reference) / (np.abs(reference) + 1e-8)
        max_rel_error = np.max(rel_error)
        mean_rel_error = np.mean(rel_error)
        print(f"  Max relative error: {max_rel_error:.6f}")
        print(f"  Mean relative error: {mean_rel_error:.6f}")
    except AssertionError as e:
        print(f"  Validation failed: {e}")
        return

    print("\n[6/7] Generating profile (NTFF)...")
    kernel(
        inputs={"a_input": device_a, "b_input": device_b},
        outputs={"output0": device_output},
        save_trace=True,
    )
    print(f"  Profile saved to the same directory as {kernel.neff_path}")

    print("\n[7/7] Benchmarking performance...")
    stats = kernel.benchmark(
        inputs={"a_input": device_a, "b_input": device_b},
        outputs={"output0": device_output},
        warmup_iter=warmup_iterations,
        benchmark_iter=benchmark_iterations,
    )

    print("\n  Performance Results:")
    print("  " + "-" * 40)
    print(f"  Mean time:    {stats.mean_ms:.3f} ms")
    print(f"  Min time:     {stats.min_ms:.3f} ms")
    print(f"  Max time:     {stats.max_ms:.3f} ms")
    print(f"  Std dev:      {stats.std_dev_ms:.3f} ms")
    print("  " + "-" * 40)


if __name__ == "__main__":
    main()
