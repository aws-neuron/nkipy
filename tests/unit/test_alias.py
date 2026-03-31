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


def nkipy_kernel_named_intermediate(a_input, b_input):
    """Kernel that returns an op result with an existing intermediate name."""
    out = np.add(a_input, b_input)
    out.backend_tensor.name = "intermediate0"
    return out


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


def test_non_alias_outputs_are_renamed_to_output_names():
    """Non-aliased outputs must be renamed even if tracing assigned a temp name."""

    a = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    b = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)

    traced = NKIPyKernel.trace(nkipy_kernel_named_intermediate, backend="hlo")
    hlo = traced.specialize(a, b)

    assert len(hlo.outputs) == 1
    assert hlo.outputs[0].name == "output0"


# ------------------------------------------------------------------ #
# Output naming contract tests
#
# These verify the exact I/O names that end up in compiled NEFFs,
# which callers must match when invoking kernel(inputs={...}, outputs={...}).
# The patterns below were discovered while integrating with sglang-nkipy.
# ------------------------------------------------------------------ #


def test_alias_output_naming_simple():
    """Direct alias: mutated param returned by identity keeps param name.

    Pattern: update_kv_cache(kv_cache: mutable) -> kv_cache
    Expected NEFF:
        input  = "a_input.must_alias_input"
        output = "a_input"
    """
    a = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    b = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)

    traced = NKIPyKernel.trace(nkipy_kernel_single_alias, backend="hlo")
    hlo = traced.specialize(a, b)

    # 1 output: the aliased param
    assert len(hlo.outputs) == 1
    assert hlo.outputs[0].name == "a_input"

    # Input param renamed with alias suffix
    param_names = [p.name for p in hlo.parameters]
    assert "a_input.must_alias_input" in param_names
    assert "b_input" in param_names


def test_alias_auto_appended_output_naming():
    """Broken-identity alias: mutated param NOT returned → auto-appended.

    Pattern: prefill_post_moe_fn(output: mutable, ...) where output is mutated
    but the function returns a *different* computed value. The tracer
    auto-appends the original mutated param as an extra output.

    Expected NEFF:
        inputs  = "a_input.must_alias_input", "b_input"
        outputs = "output0" (the sum), "a_input" (auto-appended alias)
    """
    a = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    b = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)

    traced = NKIPyKernel.trace(nkipy_kernel_mixed_return, backend="hlo")
    hlo = traced.specialize(a, b)

    # 2 outputs: user-returned sum (output0) + auto-appended alias (a_input)
    assert len(hlo.outputs) == 2
    assert hlo.outputs[0].name == "output0"
    assert hlo.outputs[1].name == "a_input"

    # Alias metadata
    assert len(hlo.aliases) == 1
    assert hlo.aliases[0].param_name == "a_input"
    assert hlo.aliases[0].output_index == 1
    assert hlo.aliases[0].is_user_returned is False

    # Input param renamed
    param_names = [p.name for p in hlo.parameters]
    assert "a_input.must_alias_input" in param_names


def nkipy_kernel_alias_with_multiple_outputs(a_input, b_input, c_input):
    """Kernel that aliases a_input and c_input, returns them plus a computed value.

    Pattern: fused pre_moe graph returning (kv_cache, hidden, topk, ...)
    where kv_cache is aliased but hidden/topk/... are not.
    """
    a_input[0:1, :] = b_input[0:1, :]
    c_input[2:3, :] = b_input[2:3, :]
    computed = np.add(a_input, b_input)
    return a_input, computed, c_input


def test_alias_mixed_with_non_alias_outputs():
    """Multiple outputs where some are aliased and some are not.

    Expected NEFF:
        inputs  = "a_input.must_alias_input", "b_input", "c_input.must_alias_input"
        outputs = "a_input" (alias), "output1" (computed), "c_input" (alias)
    """
    a = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    b = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    c = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)

    traced = NKIPyKernel.trace(nkipy_kernel_alias_with_multiple_outputs, backend="hlo")
    hlo = traced.specialize(a, b, c)

    assert len(hlo.outputs) == 3
    # Aliased outputs keep param names; non-aliased get output{idx}
    assert hlo.outputs[0].name == "a_input"
    assert hlo.outputs[1].name == "output1"
    assert hlo.outputs[2].name == "c_input"

    # 2 aliases
    assert len(hlo.aliases) == 2
    alias_names = {a.param_name for a in hlo.aliases}
    assert alias_names == {"a_input", "c_input"}


def test_no_return_alias_output_naming():
    """Mutation-only kernel: auto-appended alias is the sole output.

    Expected NEFF:
        input  = "a_input.must_alias_input", "b_input"
        output = "a_input"
    """
    a = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)
    b = ((np.random.rand(128, 512) - 0.5) * 2).astype(np.float16)

    traced = NKIPyKernel.trace(nkipy_kernel_no_return, backend="hlo")
    hlo = traced.specialize(a, b)

    assert len(hlo.outputs) == 1
    assert hlo.outputs[0].name == "a_input"

    param_names = [p.name for p in hlo.parameters]
    assert "a_input.must_alias_input" in param_names
    assert "b_input" in param_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
