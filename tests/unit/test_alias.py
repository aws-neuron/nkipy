# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for tensor aliasing functionality.

Tests the automatic mutation-based aliasing mechanism which detects in-place
modifications of input tensors via __setitem__ during tracing.
"""

import numpy as np
import pytest
from nkipy.core.trace import NKIPyKernel
from utils import (
    NEURON_AVAILABLE,
    baremetal_assert_allclose,
    cpu_assert_allclose,
    trace_and_compile,
)


def nkipy_kernel_single_alias(a_input, b_input):
    """Kernel with single alias - modifies a_input in place and returns it"""
    a_input[0, :] = b_input[1, :]
    return a_input


def nkipy_kernel_multi_alias(a_input, b_input, c_input):
    """Kernel with multiple alias pairs - modifies both a_input and c_input in place"""
    a_input[0:1, :] = b_input[0:1, :]
    c_input[2:3, :] = b_input[2:3, :]
    return a_input, c_input


def nkipy_kernel_no_return(a_input, b_input):
    """Kernel that mutates a_input but does not return anything."""
    a_input[0, :] = b_input[1, :]


def nkipy_kernel_mixed_return(a_input, b_input):
    """Kernel that mutates a_input and returns a different computed value."""
    a_input[0, :] = b_input[1, :]
    return a_input + b_input


def test_single_alias(trace_mode):
    """Test single alias pair on CPU and hardware"""
    A = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    B = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)

    expected = A.copy()
    expected[0, :] = B[1, :]

    result = nkipy_kernel_single_alias(A.copy(), B)
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
    else:
        trace_and_compile(nkipy_kernel_single_alias, trace_mode, A.copy(), B)


def test_multi_alias(trace_mode):
    """Test multiple alias pairs on CPU and hardware"""

    A = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    B = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    C = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)

    # Compute expected results
    expected_A = A.copy()
    expected_A[0:1, :] = B[0:1, :]
    expected_C = C.copy()
    expected_C[2:3, :] = B[2:3, :]

    result_A, result_C = nkipy_kernel_multi_alias(A.copy(), B, C.copy())
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
    else:
        trace_and_compile(nkipy_kernel_multi_alias, trace_mode, A.copy(), B, C.copy())


def test_no_return_alias(trace_mode):
    """Test mutation-only kernel (no return statement).

    The mutated parameter should be auto-appended to outputs and aliased.
    """
    A = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    B = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)

    expected = A.copy()
    expected[0, :] = B[1, :]

    # CPU execution: function mutates A in place, returns None
    A_copy = A.copy()
    result = nkipy_kernel_no_return(A_copy, B)
    assert result is None
    cpu_assert_allclose(A_copy, expected)

    # Verify tracing: should have 1 alias (auto-added), 0 user-returned outputs
    traced_kernel = NKIPyKernel.trace(nkipy_kernel_no_return, backend="hlo")
    ir = traced_kernel.specialize(A.copy(), B)
    assert len(ir.aliases) == 1
    assert ir.aliases[0].param_name == "a_input"
    assert ir.aliases[0].is_user_returned is False
    assert len(ir.auto_aliased_indices) == 1

    if not NEURON_AVAILABLE:
        trace_and_compile(nkipy_kernel_no_return, trace_mode, A.copy(), B)


def test_mixed_return_alias(trace_mode):
    """Test kernel that mutates a parameter and returns a different value.

    a_input is mutated (aliased) but the user returns a_input + b_input.
    """
    A = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    B = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)

    expected_A = A.copy()
    expected_A[0, :] = B[1, :]
    expected_sum = expected_A + B

    # CPU execution
    A_copy = A.copy()
    result = nkipy_kernel_mixed_return(A_copy, B)
    cpu_assert_allclose(result, expected_sum)

    # Verify tracing: should have 1 alias (auto-added, not user-returned)
    # plus 1 user-returned output (the sum)
    traced_kernel = NKIPyKernel.trace(nkipy_kernel_mixed_return, backend="hlo")
    ir = traced_kernel.specialize(A.copy(), B)
    assert len(ir.aliases) == 1
    assert ir.aliases[0].param_name == "a_input"
    assert ir.aliases[0].is_user_returned is False
    # 2 outputs total: the sum (user) + a_input (auto-aliased)
    assert len(ir.outputs) == 2
    assert len(ir.auto_aliased_indices) == 1

    if not NEURON_AVAILABLE:
        trace_and_compile(nkipy_kernel_mixed_return, trace_mode, A.copy(), B)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
