# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Tests for NKI kernel integration with NKIPy
# Supports both legacy (neuronxcc.nki) and beta 2 (nki) frontends

import numpy as np
import pytest
from nkipy.core.nki_op import BETA2_NKI_AVAILABLE, LEGACY_NKI_AVAILABLE, wrap_nki_kernel
from nkipy.core.trace import NKIPyKernel
from utils import (
    NEURON_AVAILABLE,
    baremetal_assert_allclose,
    on_device_test,
)

# Import legacy frontend for existing tests
if LEGACY_NKI_AVAILABLE:
    import neuronxcc.nki as nki_legacy
    import neuronxcc.nki.language as nl_legacy
    import neuronxcc.nki.typing as nt_legacy

# Import beta 2 frontend for new tests
if BETA2_NKI_AVAILABLE:
    import nki as nki_beta2
    import nki.isa as nisa_beta2
    import nki.language as nl_beta2


@pytest.mark.skipif(
    not LEGACY_NKI_AVAILABLE, reason="Legacy NKI frontend (neuronxcc.nki) not available"
)
@pytest.mark.parametrize(
    "bias,add_bias",
    [
        (0.0, False),
        (5.0, True),
        (2.5, False),
        (-1.0, True),
    ],
)
def test_nki_with_grid(trace_mode, bias, add_bias):
    """Test the NKI kernel workflow with parameterized bias and add_bias values (legacy frontend)"""

    # Simple matrix add kernel for testing, now with a launch grid and some flags
    def nki_tensor_add_kernel_(a_input, b_input, bias=0.0, add_bias=False):
        c_output = nl_legacy.ndarray(
            a_input.shape, dtype=a_input.dtype, buffer=nl_legacy.shared_hbm
        )
        offset_i_x = nl_legacy.program_id(0) * 128
        offset_i_y = nl_legacy.program_id(1) * 512
        ix = offset_i_x + nl_legacy.arange(128)[:, None]
        iy = offset_i_y + nl_legacy.arange(512)[None, :]
        a_tile = nl_legacy.load(a_input[ix, iy])
        b_tile = nl_legacy.load(b_input[ix, iy])
        c_tile = a_tile + b_tile
        if add_bias:
            c_tile = c_tile + bias
        nl_legacy.store(c_output[ix, iy], value=c_tile)
        return c_output

    # Create inputs and compute reference
    a = np.random.rand(256, 1024).astype(np.float32)
    b = np.random.rand(256, 1024).astype(np.float32)
    d = np.random.rand(256, 1024).astype(np.float32)
    ref = a + b + d
    if add_bias:
        ref += bias

    # Create NKI op, currently this has to be done in advance
    grid = (256 // 128, 1024 // 512)
    nki_op = wrap_nki_kernel(
        nki_tensor_add_kernel_,
        [np.empty(a.shape, a.dtype), np.empty(b.shape, b.dtype), bias, add_bias],
        grid=grid,
        is_nki_beta_2_version=False,  # Use legacy frontend
    )

    # Hook it up with another op
    def test_func(a, b, d):
        c = nki_op(a, b)
        return np.add(c, d)

    # Test hardware - only if available
    if NEURON_AVAILABLE:
        out_baremetal = on_device_test(test_func, trace_mode, a, b, d)
        baremetal_assert_allclose(ref, out_baremetal)


@pytest.mark.skipif(
    not LEGACY_NKI_AVAILABLE, reason="Legacy NKI frontend (neuronxcc.nki) not available"
)
def test_nki_simple(trace_mode):
    """Test the simple NKI kernel workflow (legacy frontend)"""

    # Simple matrix add kernel for testing, fixed shape 128*512
    def nki_tensor_add_kernel_(a_input, b_input):
        output = nl_legacy.ndarray(
            a_input.shape, dtype=a_input.dtype, buffer=nl_legacy.shared_hbm
        )
        ix = nl_legacy.arange(128)[:, None]
        iy = nl_legacy.arange(512)[None, :]
        a_tile = nl_legacy.load(a_input[ix, iy])
        b_tile = nl_legacy.load(b_input[ix, iy])
        c_tile = a_tile + b_tile
        nl_legacy.store(output[ix, iy], value=c_tile)
        return output

    # Create inputs and compute reference
    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)
    d = np.random.rand(128, 512).astype(np.float32)
    ref = a + b + d

    # Create NKI op, currently this has to be done in advance
    nki_op = wrap_nki_kernel(
        nki_tensor_add_kernel_, [a, b], is_nki_beta_2_version=False
    )

    # Hook it up with another op
    def test_func(a, b, d):
        c = nki_op(a, b)
        return np.add(c, d)

    # Test hardware - only if available
    if NEURON_AVAILABLE:
        out_baremetal = on_device_test(test_func, trace_mode, a, b, d)
        baremetal_assert_allclose(ref, out_baremetal)


@pytest.mark.skipif(
    not BETA2_NKI_AVAILABLE, reason="Beta 2 NKI frontend (nki) not available"
)
@pytest.mark.skipif(
    not NEURON_AVAILABLE,
    reason="Hardware required - Beta 2 frontend does not support CPU execution",
)
def test_nki_simple_beta_2():
    """Test the simple NKI kernel workflow with Beta 2 frontend (hardware only)"""

    # Simple matrix add kernel using Beta 2 frontend with nisa instructions
    def nki_tensor_add_kernel_beta2(a_input, b_input):
        """
        NKI kernel to compute element-wise addition of two input tensors.
        Uses Beta 2 frontend with nisa instructions.
        """
        # Check both input tensor shapes are the same for element-wise operation.
        assert a_input.shape == b_input.shape

        # Check the first dimension's size to ensure it does not exceed on-chip
        # memory tile size, since this simple kernel does not tile inputs.
        assert a_input.shape[0] <= nl_beta2.tile_size.pmax

        # Allocate space for the input tensors in SBUF and copy the inputs from HBM
        # to SBUF with DMA copy.
        a_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
        nisa_beta2.dma_copy(dst=a_tile, src=a_input)

        b_tile = sbuf.view(dtype=b_input.dtype, shape=b_input.shape)  # noqa: F821
        nisa_beta2.dma_copy(dst=b_tile, src=b_input)

        # Allocate space for the result and use tensor_tensor to perform
        # element-wise addition.
        c_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
        nisa_beta2.tensor_tensor(
            dst=c_tile, data1=a_tile, data2=b_tile, op=nl_beta2.add
        )

        # Create a tensor in HBM and copy the result into HBM.
        c_output = hbm.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
        nisa_beta2.dma_copy(dst=c_output, src=c_tile)

        # Return kernel output as function output.
        return c_output

    # Create inputs and compute reference
    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)
    d = np.random.rand(128, 512).astype(np.float32)
    ref = a + b + d

    # Create NKI op with Beta 2 frontend
    nki_op = wrap_nki_kernel(
        nki_tensor_add_kernel_beta2,
        [a, b],
        is_nki_beta_2_version=True,  # Use Beta 2 frontend
    )

    # Hook it up with another op
    def test_func(a, b, d):
        c = nki_op(a, b)
        return np.add(c, d)

    # Test hardware only (Beta 2 frontend does not support CPU execution)
    out_baremetal = on_device_test(test_func, "hlo", a, b, d)
    baremetal_assert_allclose(ref, out_baremetal)


@pytest.mark.skipif(
    not BETA2_NKI_AVAILABLE, reason="Beta 2 NKI frontend (nki) not available"
)
def test_nki_direct_jit_beta2_called_twice_different_shapes():
    """Regression: calling the same @nki.jit beta2 kernel twice with different
    shapes during a single NKIPy trace must not fail.

    The underlying GenericKernel's C++ frontend.Kernel accumulates state during
    specialize/trace. Without the clone+reset in _generate_nki_custom_call, the
    second invocation hits stale state and raises.
    """

    @nki_beta2.jit(platform_target="trn2")
    def nki_add_kernel(a_input, b_input):
        a_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
        nisa_beta2.dma_copy(dst=a_tile, src=a_input)
        b_tile = sbuf.view(dtype=b_input.dtype, shape=b_input.shape)  # noqa: F821
        nisa_beta2.dma_copy(dst=b_tile, src=b_input)
        c_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
        nisa_beta2.tensor_tensor(
            dst=c_tile, data1=a_tile, data2=b_tile, op=nl_beta2.add
        )
        c_output = hbm.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
        nisa_beta2.dma_copy(dst=c_output, src=c_tile)
        return c_output

    # Two pairs of inputs with different second-dimension sizes
    a1 = np.random.rand(128, 512).astype(np.float32)
    b1 = np.random.rand(128, 512).astype(np.float32)
    a2 = np.random.rand(128, 256).astype(np.float32)
    b2 = np.random.rand(128, 256).astype(np.float32)

    def test_func(a1, b1, a2, b2):
        c1 = nki_add_kernel(a1, b1)  # first call: 128x512
        c2 = nki_add_kernel(a2, b2)  # second call: 128x256 — was failing
        return c1, c2

    # Tracing alone exercises the bug path (no hardware needed)
    traced = NKIPyKernel.trace(test_func, backend="hlo")
    traced.specialize(a1, b1, a2, b2)


@pytest.mark.skipif(
    not LEGACY_NKI_AVAILABLE, reason="Legacy NKI frontend (neuronxcc.nki) not available"
)
def test_nki_direct_jit(trace_mode):
    """Test using @nki.jit decorated kernel directly in NKIPy (no wrap_nki_kernel needed) - legacy frontend"""

    # Simple matrix add kernel with @nki.jit decorator
    @nki_legacy.jit
    def nki_tensor_add_kernel_jit(a_input, b_input):
        output = nl_legacy.ndarray(
            a_input.shape, dtype=a_input.dtype, buffer=nl_legacy.shared_hbm
        )
        ix = nl_legacy.arange(128)[:, None]
        iy = nl_legacy.arange(512)[None, :]
        a_tile = nl_legacy.load(a_input[ix, iy])
        b_tile = nl_legacy.load(b_input[ix, iy])
        c_tile = a_tile + b_tile
        nl_legacy.store(output[ix, iy], value=c_tile)
        return output

    # Create inputs and compute reference
    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)
    d = np.random.rand(128, 512).astype(np.float32)
    ref = a + b + d

    # Use @nki.jit kernel directly - no wrap_nki_kernel needed!
    def test_func(a, b, d):
        c = nki_tensor_add_kernel_jit(a, b)  # Direct call to @nki.jit kernel
        return np.add(c, d)

    # Test hardware - only if available
    if NEURON_AVAILABLE:
        out_baremetal = on_device_test(test_func, trace_mode, a, b, d)
        baremetal_assert_allclose(ref, out_baremetal)


@pytest.mark.skipif(
    not LEGACY_NKI_AVAILABLE, reason="Legacy NKI frontend (neuronxcc.nki) not available"
)
def test_nki_direct_jit_with_grid(trace_mode):
    """Test using @nki.jit decorated kernel with grid syntax: kernel[grid](args) - legacy frontend"""

    @nki_legacy.jit
    def nki_tensor_add_kernel_grid(a_input, b_input):
        c_output = nl_legacy.ndarray(
            a_input.shape, dtype=a_input.dtype, buffer=nl_legacy.shared_hbm
        )
        offset_i_x = nl_legacy.program_id(0) * 128
        offset_i_y = nl_legacy.program_id(1) * 512
        ix = offset_i_x + nl_legacy.arange(128)[:, None]
        iy = offset_i_y + nl_legacy.arange(512)[None, :]
        a_tile = nl_legacy.load(a_input[ix, iy])
        b_tile = nl_legacy.load(b_input[ix, iy])
        c_tile = a_tile + b_tile
        nl_legacy.store(c_output[ix, iy], value=c_tile)
        return c_output

    # Create inputs and compute reference
    a = np.random.rand(256, 1024).astype(np.float32)
    b = np.random.rand(256, 1024).astype(np.float32)
    d = np.random.rand(256, 1024).astype(np.float32)
    ref = a + b + d

    # Use @nki.jit kernel with grid syntax - kernel[grid_x, grid_y](args)
    def test_func(a, b, d):
        c = nki_tensor_add_kernel_grid[2, 2](a, b)  # Grid syntax!
        return np.add(c, d)

    # Test hardware - only if available
    if NEURON_AVAILABLE:
        out_baremetal = on_device_test(test_func, trace_mode, a, b, d)
        baremetal_assert_allclose(ref, out_baremetal)


@pytest.mark.skipif(
    not LEGACY_NKI_AVAILABLE, reason="Legacy NKI frontend (neuronxcc.nki) not available"
)
def test_nki_mutable_tensor(trace_mode):
    """Test the simple NKI kernel workflow with mutable tensor (legacy frontend)"""

    # Simple matrix add kernel for testing, fixed shape 128*512
    def nki_tensor_add_kernel_(a_input: nt_legacy.mutable_tensor, b_input):
        ix = nl_legacy.arange(128)[:, None]
        iy = nl_legacy.arange(512)[None, :]
        a_tile = nl_legacy.load(a_input[ix, iy])
        b_tile = nl_legacy.load(b_input[ix, iy])
        c_tile = a_tile + b_tile
        nl_legacy.store(a_input[ix, iy], value=c_tile)
        return a_input

    # Create inputs and compute reference
    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)
    ref = a + b

    # Create NKI op, currently this has to be done in advance
    nki_op = wrap_nki_kernel(
        nki_tensor_add_kernel_, [a, b], is_nki_beta_2_version=False
    )

    # Hook it up with another op
    def test_func(a_input: nt_legacy.mutable_tensor, b_input):
        a_input = nki_op(a_input, b_input)
        return a_input

    # Test hardware - only if available
    if NEURON_AVAILABLE:
        from nkipy.runtime import DeviceKernel, DeviceTensor

        test_func = NKIPyKernel.trace(test_func, backend=trace_mode)

        device_kernel = DeviceKernel.compile_and_load(
            test_func, a, b, use_cached_if_exists=False
        )
        t_a = DeviceTensor.from_numpy(a)
        t_b = DeviceTensor.from_numpy(b)
        device_kernel(
            inputs={"a_input.must_alias_input": t_a, "b_input": t_b},
            outputs={"a_input": t_a},
        )

        baremetal_assert_allclose(t_a.numpy(), ref)


@pytest.mark.skipif(
    not BETA2_NKI_AVAILABLE, reason="Beta 2 NKI frontend (nki) not available"
)
def test_nki_direct_jit_beta2_kwargs_operand_order():
    """Tensor operands passed as kwargs must be collected in parameter order."""

    @nki_beta2.jit(platform_target="trn2")
    def nki_add_kernel(a_input, b_input):
        a_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
        nisa_beta2.dma_copy(dst=a_tile, src=a_input)
        b_tile = sbuf.view(dtype=b_input.dtype, shape=b_input.shape)  # noqa: F821
        nisa_beta2.dma_copy(dst=b_tile, src=b_input)
        c_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
        nisa_beta2.tensor_tensor(
            dst=c_tile, data1=a_tile, data2=b_tile, op=nl_beta2.add
        )
        c_output = hbm.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
        nisa_beta2.dma_copy(dst=c_output, src=c_tile)
        return c_output

    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)

    # Pass both tensors as kwargs (b before a) — must still trace correctly
    def test_func(a, b):
        return nki_add_kernel(b_input=b, a_input=a)

    traced = NKIPyKernel.trace(test_func, backend="hlo")
    traced.specialize(a, b)


if __name__ == "__main__":
    # Allow running the test file directly
    pytest.main([__file__])
