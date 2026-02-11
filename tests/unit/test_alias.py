# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for tensor aliasing functionality.

Tests the mutable_tensor aliasing mechanism which allows in-place modifications
of input tensors, essential for memory-efficient operations.
"""

import nkipy.core.typing as nt
import numpy as np
import pytest
from nkipy.core.trace import NKIPyKernel
from utils import (
    NEURON_AVAILABLE,
    baremetal_assert_allclose,
    cpu_assert_allclose,
    trace_and_run,
    trace_mode,  # noqa: F401 - pytest fixture
)


def nkipy_kernel_single_alias(a_input: nt.mutable_tensor, b_input):
    """Kernel with single alias - modifies a_input in place"""
    a_input[0, :] = b_input[1, :]
    return a_input


def nkipy_kernel_multi_alias(
    a_input: nt.mutable_tensor, b_input, c_input: nt.mutable_tensor
):
    """Kernel with multiple alias pairs - modifies both a_input and c_input in place"""
    a_input[0:1, :] = b_input[0:1, :]
    c_input[2:3, :] = b_input[2:3, :]
    return a_input, c_input


def test_single_alias(trace_mode):
    """Test single alias pair on CPU and hardware"""
    np.random.seed(42)
    A = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    B = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)

    # Compute expected result
    expected = A.copy()
    expected[0, :] = B[1, :]

    # Test CPU execution
    result = trace_and_run(nkipy_kernel_single_alias, trace_mode, A.copy(), B)
    cpu_assert_allclose(result, expected)

    # Test hardware if available
    if NEURON_AVAILABLE:
        from nkipy.runtime import DeviceKernel, DeviceTensor

        # Compile kernel with appropriate backend
        if trace_mode == "hlo":
            traced_kernel = NKIPyKernel.trace(nkipy_kernel_single_alias, backend="hlo")
        else:
            raise ValueError(f"Invalid trace_mode: {trace_mode}")

        kernel = DeviceKernel.compile_and_load(
            traced_kernel,
            A,
            B,
            name=f"test_single_alias_{trace_mode}",
            use_cached_if_exists=False,
        )

        device_A = DeviceTensor.from_numpy(A)
        device_B = DeviceTensor.from_numpy(B)
        output = device_A

        # Use the .must_alias_input suffix for the mutable input parameter
        kernel(
            inputs={"a_input.must_alias_input": device_A, "b_input": device_B},
            outputs={"a_input": output},
        )

        baremetal_assert_allclose(output.numpy(), expected)


def test_multi_alias(trace_mode):
    """Test multiple alias pairs on CPU and hardware"""
    np.random.seed(43)
    A = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    B = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    C = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)

    # Compute expected results
    expected_A = A.copy()
    expected_A[0:1, :] = B[0:1, :]
    expected_C = C.copy()
    expected_C[2:3, :] = B[2:3, :]

    # Test CPU execution
    result_A, result_C = trace_and_run(
        nkipy_kernel_multi_alias, trace_mode, A.copy(), B, C.copy()
    )
    cpu_assert_allclose(result_A, expected_A)
    cpu_assert_allclose(result_C, expected_C)

    # Test hardware if available
    if NEURON_AVAILABLE:
        from nkipy.runtime import DeviceKernel, DeviceTensor

        # Compile kernel with appropriate backend
        if trace_mode == "hlo":
            traced_kernel = NKIPyKernel.trace(nkipy_kernel_multi_alias, backend="hlo")
        else:
            raise ValueError(f"Invalid trace_mode: {trace_mode}")

        kernel = DeviceKernel.compile_and_load(
            traced_kernel,
            A,
            B,
            C,
            name=f"test_multi_alias_{trace_mode}",
            use_cached_if_exists=False,
        )

        device_A = DeviceTensor.from_numpy(A)
        device_B = DeviceTensor.from_numpy(B)
        device_C = DeviceTensor.from_numpy(C)
        output0 = device_A
        output1 = device_C

        kernel(
            inputs={
                "a_input.must_alias_input": device_A,
                "b_input": device_B,
                "c_input.must_alias_input": device_C,
            },
            outputs={"a_input": output0, "c_input": output1},
        )

        baremetal_assert_allclose(output0.numpy(), expected_A)
        baremetal_assert_allclose(output1.numpy(), expected_C)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
