# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Tests for NKI kernel integration with NKIPy
# Supports legacy (neuronxcc.nki), beta 2 (nki), and beta 3 (nki >= 0.4) frontends

import numpy as np
import pytest
from nkipy.core.nki_op import (
    BETA2_NKI_AVAILABLE,
    BETA3_NKI_AVAILABLE,
    LEGACY_NKI_AVAILABLE,
    wrap_nki_kernel,
)
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

# Import beta 3 frontend
if BETA3_NKI_AVAILABLE:
    import nki as nki_beta3
    import nki.isa as nisa_beta3
    import nki.language as nl_beta3


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

    # Hook it up with another op — no annotation needed on the NKIPy wrapper,
    # aliasing is detected automatically via NKI's operand_output_aliases.
    def test_func(a_input, b_input):
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


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
def test_nki_direct_jit_beta3():
    """Test direct @nki.jit kernel usage with Beta 3 frontend."""

    @nki_beta3.jit
    def nki_add_kernel(a_input, b_input):
        a_tile = nl_beta3.load(a_input[:, :])
        b_tile = nl_beta3.load(b_input[:, :])
        c_tile = nl_beta3.add(a_tile, b_tile)
        output = nl_beta3.ndarray(
            a_input.shape, dtype=a_input.dtype, buffer=nl_beta3.shared_hbm
        )
        nl_beta3.store(output[:, :], value=c_tile)
        return output

    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)

    def test_func(a, b):
        return nki_add_kernel(a, b)

    traced = NKIPyKernel.trace(test_func, backend="hlo")
    traced.specialize(a, b)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
def test_nki_direct_jit_beta3_called_twice_different_shapes():
    """Calling the same @nki.jit beta3 kernel twice with different shapes."""

    @nki_beta3.jit
    def nki_add_kernel(a_input, b_input):
        a_tile = nl_beta3.load(a_input[:, :])
        b_tile = nl_beta3.load(b_input[:, :])
        c_tile = nl_beta3.add(a_tile, b_tile)
        output = nl_beta3.ndarray(
            a_input.shape, dtype=a_input.dtype, buffer=nl_beta3.shared_hbm
        )
        nl_beta3.store(output[:, :], value=c_tile)
        return output

    a1 = np.random.rand(128, 512).astype(np.float32)
    b1 = np.random.rand(128, 512).astype(np.float32)
    a2 = np.random.rand(128, 256).astype(np.float32)
    b2 = np.random.rand(128, 256).astype(np.float32)

    def test_func(a1, b1, a2, b2):
        c1 = nki_add_kernel(a1, b1)
        c2 = nki_add_kernel(a2, b2)
        return c1, c2

    traced = NKIPyKernel.trace(test_func, backend="hlo")
    traced.specialize(a1, b1, a2, b2)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
def test_nki_direct_jit_beta3_kwargs_operand_order():
    """Tensor operands passed as kwargs must be collected in parameter order (beta 3)."""

    @nki_beta3.jit
    def nki_add_kernel(a_input, b_input):
        a_tile = nl_beta3.load(a_input[:, :])
        b_tile = nl_beta3.load(b_input[:, :])
        c_tile = nl_beta3.add(a_tile, b_tile)
        output = nl_beta3.ndarray(
            a_input.shape, dtype=a_input.dtype, buffer=nl_beta3.shared_hbm
        )
        nl_beta3.store(output[:, :], value=c_tile)
        return output

    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)

    def test_func(a, b):
        return nki_add_kernel(b_input=b, a_input=a)

    traced = NKIPyKernel.trace(test_func, backend="hlo")
    traced.specialize(a, b)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
def test_nki_wrap_kernel_beta3():
    """Test wrap_nki_kernel with is_nki_beta_3_version=True."""

    @nki_beta3.jit
    def nki_add_kernel(a_input, b_input):
        a_tile = nl_beta3.load(a_input[:, :])
        b_tile = nl_beta3.load(b_input[:, :])
        c_tile = nl_beta3.add(a_tile, b_tile)
        output = nl_beta3.ndarray(
            a_input.shape, dtype=a_input.dtype, buffer=nl_beta3.shared_hbm
        )
        nl_beta3.store(output[:, :], value=c_tile)
        return output

    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)

    nki_op = wrap_nki_kernel(
        nki_add_kernel,
        [a, b],
        is_nki_beta_3_version=True,
        platform_target="trn2",
    )

    def test_func(a, b):
        return nki_op(a, b)

    traced = NKIPyKernel.trace(test_func, backend="hlo")
    traced.specialize(a, b)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
def test_nki_mutable_tensor_beta3():
    """Test in-place (mutable) tensor aliasing with beta 3 frontend."""

    @nki_beta3.jit
    def nki_inplace_add_kernel(a_input, b_input):
        a_tile = nl_beta3.load(a_input[:, :])
        b_tile = nl_beta3.load(b_input[:, :])
        c_tile = nl_beta3.add(a_tile, b_tile)
        nl_beta3.store(a_input[:, :], value=c_tile)
        return a_input

    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)

    nki_op = wrap_nki_kernel(
        nki_inplace_add_kernel,
        [a, b],
        is_nki_beta_3_version=True,
        platform_target="trn2",
    )

    def test_func(a_input, b_input):
        a_input = nki_op(a_input, b_input)
        return a_input

    traced = NKIPyKernel.trace(test_func, backend="hlo")
    traced.specialize(a, b)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
@pytest.mark.skipif(
    not NEURON_AVAILABLE,
    reason="Hardware required - Beta 3 frontend does not support CPU execution",
)
def test_nki_mutable_tensor_beta3_hardware():
    """Test in-place (mutable) tensor on hardware with beta 3 frontend."""

    @nki_beta3.jit
    def nki_inplace_add_kernel(a_input, b_input):
        a_tile = nl_beta3.load(a_input[:, :])
        b_tile = nl_beta3.load(b_input[:, :])
        c_tile = nl_beta3.add(a_tile, b_tile)
        nl_beta3.store(a_input[:, :], value=c_tile)
        return a_input

    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)
    ref = a + b

    from nkipy.runtime import DeviceKernel, DeviceTensor

    test_func = NKIPyKernel.trace(
        lambda a_input, b_input: nki_inplace_add_kernel(a_input, b_input),
        backend="hlo",
    )

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
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
@pytest.mark.skipif(
    not NEURON_AVAILABLE,
    reason="Hardware required - Beta 3 frontend does not support CPU execution",
)
def test_nki_simple_beta3_hardware():
    """Test beta 3 NKI kernel on hardware (hardware only)."""

    @nki_beta3.jit
    def nki_add_kernel(a_input, b_input):
        a_tile = nl_beta3.load(a_input[:, :])
        b_tile = nl_beta3.load(b_input[:, :])
        c_tile = nl_beta3.add(a_tile, b_tile)
        output = nl_beta3.ndarray(
            a_input.shape, dtype=a_input.dtype, buffer=nl_beta3.shared_hbm
        )
        nl_beta3.store(output[:, :], value=c_tile)
        return output

    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)
    d = np.random.rand(128, 512).astype(np.float32)
    ref = a + b + d

    def test_func(a, b, d):
        c = nki_add_kernel(a, b)
        return np.add(c, d)

    out_baremetal = on_device_test(test_func, "hlo", a, b, d)
    baremetal_assert_allclose(ref, out_baremetal)


# ---------------------------------------------------------------------------
# Beta 3 non-tensor (trace-time constant) arguments
#
# NKI's beta 3 frontend splits a kernel's arguments into device inputs (ndarrays,
# which become HBM operands) and constants (scalars, tuples, ...), which it
# specializes into the compiled kernel. So a constant must be supplied when the
# kernel is COMPILED -- at wrap_nki_kernel time -- and must NOT be forwarded as an
# operand of the resulting custom-call.
# ---------------------------------------------------------------------------


if BETA3_NKI_AVAILABLE:

    @nki_beta3.jit
    def _beta3_scaled_kernel(a_input, scale=1.0):
        """Copy a_input * scale. `scale` is consumed at TRACE time."""
        a_tile = nl_beta3.load(a_input[:, :])
        out_tile = nl_beta3.ndarray(
            a_tile.shape, dtype=a_tile.dtype, buffer=nl_beta3.sbuf
        )
        nisa_beta3.activation(dst=out_tile, op=nl_beta3.copy, data=a_tile, scale=scale)
        output = nl_beta3.ndarray(
            a_input.shape, dtype=a_input.dtype, buffer=nl_beta3.shared_hbm
        )
        nl_beta3.store(output[:, :], value=out_tile)
        return output

    @nki_beta3.jit
    def _beta3_mid_scalar_kernel(a_input, scale, b_input):
        """(a_input + b_input) * scale, with the constant MID-signature."""
        a_tile = nl_beta3.load(a_input[:, :])
        b_tile = nl_beta3.load(b_input[:, :])
        sum_tile = nl_beta3.ndarray(
            a_tile.shape, dtype=a_tile.dtype, buffer=nl_beta3.sbuf
        )
        nisa_beta3.tensor_tensor(
            dst=sum_tile, data1=a_tile, data2=b_tile, op=nl_beta3.add
        )
        out_tile = nl_beta3.ndarray(
            a_tile.shape, dtype=a_tile.dtype, buffer=nl_beta3.sbuf
        )
        nisa_beta3.activation(
            dst=out_tile, op=nl_beta3.copy, data=sum_tile, scale=scale
        )
        output = nl_beta3.ndarray(
            a_input.shape, dtype=a_input.dtype, buffer=nl_beta3.shared_hbm
        )
        nl_beta3.store(output[:, :], value=out_tile)
        return output


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
@pytest.mark.parametrize(
    "via_kwargs", [False, True], ids=["positional", "kernel_kwargs"]
)
def test_nki_wrap_beta3_scalar_not_an_operand(via_kwargs):
    """A non-tensor arg is compiled in, not passed as a custom-call operand.

    Forwarding it as an operand would shift every later operand and make the
    kernel read the wrong buffers.
    """
    a = np.random.rand(128, 512).astype(np.float32)

    if via_kwargs:
        nki_op = wrap_nki_kernel(
            _beta3_scaled_kernel, [a], kernel_kwargs={"scale": 3.0},
            is_nki_beta_3_version=True, platform_target="trn2",
        )
    else:
        nki_op = wrap_nki_kernel(
            _beta3_scaled_kernel, [a, 3.0],
            is_nki_beta_3_version=True, platform_target="trn2",
        )

    # `scale` is not a device input, so the op takes only the one tensor.
    assert nki_op._beta3_device_input_names == ["a_input"]
    assert nki_op._beta3_constant_names == ["scale"]

    traced = NKIPyKernel.trace(lambda x: nki_op(x), backend="hlo")
    traced.specialize(a)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
@pytest.mark.parametrize(
    "via_kwargs", [False, True], ids=["positional", "kernel_kwargs"]
)
def test_nki_wrap_beta3_scalar_mid_signature(via_kwargs):
    """A constant in the MIDDLE of the signature still binds to the right param."""
    a = np.random.rand(128, 512).astype(np.float32)
    b = np.random.rand(128, 512).astype(np.float32)

    if via_kwargs:
        # kernel_kwargs fills `scale` by name, so the operands are just the tensors.
        nki_op = wrap_nki_kernel(
            _beta3_mid_scalar_kernel, [a, b], kernel_kwargs={"scale": 2.0},
            is_nki_beta_3_version=True, platform_target="trn2",
        )
    else:
        nki_op = wrap_nki_kernel(
            _beta3_mid_scalar_kernel, [a, 2.0, b],
            is_nki_beta_3_version=True, platform_target="trn2",
        )

    assert nki_op._beta3_device_input_names == ["a_input", "b_input"]

    traced = NKIPyKernel.trace(lambda x, y: nki_op(x, y), backend="hlo")
    traced.specialize(a, b)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
def test_nki_wrap_beta3_operand_mismatch_raises():
    """A wrong operand count fails loudly instead of reaching the backend.

    Without the check this surfaces from neuronx-cc as an opaque
    "Unrecognized DRAM input location", or is silently miscompiled.
    """
    a = np.random.rand(128, 512).astype(np.float32)
    nki_op = wrap_nki_kernel(
        _beta3_scaled_kernel, [a, 3.0],
        is_nki_beta_3_version=True, platform_target="trn2",
    )

    with pytest.raises(ValueError, match="expects 1 tensor operand"):
        traced = NKIPyKernel.trace(lambda x, y: nki_op(x, y), backend="hlo")
        traced.specialize(a, a)

    # A constant passed at CALL time used to be silently ignored in favour of the
    # compiled-in value; it is now rejected.
    with pytest.raises(ValueError):
        traced = NKIPyKernel.trace(lambda x: nki_op(x, 9.0), backend="hlo")
        traced.specialize(a)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
def test_nki_wrap_beta3_kernel_kwargs_validated():
    """kernel_kwargs must name real parameters, and is beta-3 only."""
    a = np.random.rand(128, 512).astype(np.float32)

    with pytest.raises(ValueError, match="not.*parameters"):
        wrap_nki_kernel(
            _beta3_scaled_kernel, [a], kernel_kwargs={"nope": 1.0},
            is_nki_beta_3_version=True, platform_target="trn2",
        )

    with pytest.raises(ValueError, match="only supported by the beta 3"):
        wrap_nki_kernel(
            _beta3_scaled_kernel, [a], kernel_kwargs={"scale": 1.0},
            platform_target="trn2",
        )


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
@pytest.mark.skipif(
    not NEURON_AVAILABLE,
    reason="Hardware required - Beta 3 frontend does not support CPU execution",
)
@pytest.mark.parametrize("scale", [1.0, 3.0])
def test_nki_wrap_beta3_scalar_applied_on_device(scale):
    """The compiled-in constant is the one the device actually uses."""
    a = np.random.rand(128, 512).astype(np.float32)

    nki_op = wrap_nki_kernel(
        _beta3_scaled_kernel, [a], kernel_kwargs={"scale": scale},
        is_nki_beta_3_version=True,
    )

    out = on_device_test(lambda x: nki_op(x), "hlo", a)
    baremetal_assert_allclose(a * scale, out)


# ---------------------------------------------------------------------------
# Beta 3 kernel SIGNATURE SHAPES
#
# wrap_nki_kernel must not constrain how a kernel AUTHOR writes their signature.
# Every kernel below computes the same thing -- ``a * c + b`` -- so a mis-binding
# shows up as a wrong number rather than passing silently. Each is exercised both
# with the constant supplied positionally in ``operands`` and by name via
# ``kernel_kwargs``.
#
# Not covered, because NKI itself rejects it: keyword-ONLY parameters (``*``).
# NKI's compiled parser frontend emits "keyword-only arguments are not supported
# in NKI" followed by "unbound variable", independently of nkipy -- see
# test_nki_beta3_keyword_only_unsupported_upstream.
# ---------------------------------------------------------------------------

# a=2, b=3, c=5  ->  2 * 5 + 3 = 13
_SIG_A_VAL, _SIG_B_VAL, _SIG_C_VAL = 2.0, 3.0, 5.0
_SIG_EXPECTED = _SIG_A_VAL * _SIG_C_VAL + _SIG_B_VAL


if BETA3_NKI_AVAILABLE:

    def _beta3_scale_add_body(a, b, c):
        """out = a * c + b, in one scalar_tensor_tensor."""
        a_tile = nl_beta3.load(a[:, :])
        b_tile = nl_beta3.load(b[:, :])
        out_tile = nl_beta3.ndarray(
            a_tile.shape, dtype=a_tile.dtype, buffer=nl_beta3.sbuf
        )
        nisa_beta3.scalar_tensor_tensor(
            dst=out_tile, data=a_tile,
            op0=nl_beta3.multiply, operand0=c,
            op1=nl_beta3.add, operand1=b_tile,
        )
        output = nl_beta3.ndarray(a.shape, dtype=a.dtype, buffer=nl_beta3.shared_hbm)
        nl_beta3.store(output[:, :], value=out_tile)
        return output

    @nki_beta3.jit
    def _sig_all_positional(a, b, c):
        return _beta3_scale_add_body(a, b, c)

    @nki_beta3.jit
    def _sig_scalar_default(a, b, c=10.0):
        return _beta3_scale_add_body(a, b, c)

    @nki_beta3.jit
    def _sig_tensor_and_scalar_default(a, b=None, c=10.0):
        return _beta3_scale_add_body(a, b, c)

    @nki_beta3.jit
    def _sig_scalar_before_tensor_default(a, c, b=None):
        return _beta3_scale_add_body(a, b, c)

    @nki_beta3.jit
    def _sig_scalar_default_before_tensor_default(a, c=10.0, b=None):
        return _beta3_scale_add_body(a, b, c)

    @nki_beta3.jit
    def _sig_all_defaulted(a=None, b=None, c=10.0):
        return _beta3_scale_add_body(a, b, c)

    @nki_beta3.jit
    def _sig_keyword_only_scalar(a, b, *, c):
        """NKI does not support keyword-only params; used as a negative test."""
        return _beta3_scale_add_body(a, b, c)


# (kernel, positional operand order) -- `A`/`B` are tensors, `C` the constant.
_SIG_CASES = {
    "all_positional": ("_sig_all_positional", "ABC"),
    "scalar_default": ("_sig_scalar_default", "ABC"),
    "tensor_and_scalar_default": ("_sig_tensor_and_scalar_default", "ABC"),
    "scalar_before_tensor_default": ("_sig_scalar_before_tensor_default", "ACB"),
    "scalar_default_before_tensor_default": (
        "_sig_scalar_default_before_tensor_default", "ACB",
    ),
    "all_defaulted": ("_sig_all_defaulted", "ABC"),
}


def _sig_operands(order, a, b, c):
    """Build the positional operand list for a signature's parameter order."""
    return [{"A": a, "B": b, "C": c}[ch] for ch in order]


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
@pytest.mark.parametrize("case", sorted(_SIG_CASES))
@pytest.mark.parametrize(
    "via_kwargs", [False, True], ids=["positional", "kernel_kwargs"]
)
def test_nki_wrap_beta3_signature_shapes(case, via_kwargs):
    """Any mix of positional/defaulted tensor and scalar params binds correctly.

    In every shape the two tensors are the only device inputs, in signature
    order -- the scalar never becomes an operand no matter where it sits.
    """
    kernel_name, order = _SIG_CASES[case]
    kernel = globals()[kernel_name]
    a = np.full((128, 512), _SIG_A_VAL, dtype=np.float32)
    b = np.full((128, 512), _SIG_B_VAL, dtype=np.float32)

    if via_kwargs:
        # Only the tensors are positional; the scalar is bound by name. Note the
        # operand list is [a, b] for EVERY shape -- naming `c` removes it from the
        # positional sequence, so the author's parameter order stops mattering.
        nki_op = wrap_nki_kernel(
            kernel, [a, b], kernel_kwargs={"c": _SIG_C_VAL},
            is_nki_beta_3_version=True, platform_target="trn2",
        )
    else:
        nki_op = wrap_nki_kernel(
            kernel, _sig_operands(order, a, b, _SIG_C_VAL),
            is_nki_beta_3_version=True, platform_target="trn2",
        )

    # The scalar is a trace-time constant, so `a` and `b` are the only operands,
    # always in signature order regardless of where `c` sits.
    assert nki_op._beta3_device_input_names == ["a", "b"]
    assert nki_op._beta3_constant_names == ["c"]

    traced = NKIPyKernel.trace(lambda x, y: nki_op(x, y), backend="hlo")
    traced.specialize(a, b)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
@pytest.mark.parametrize("case", sorted(_SIG_CASES))
def test_nki_wrap_beta3_signature_shapes_tensor_by_name(case):
    """A TENSOR may also be bound by name, mixed with positional operands."""
    kernel_name, _ = _SIG_CASES[case]
    kernel = globals()[kernel_name]
    a = np.full((128, 512), _SIG_A_VAL, dtype=np.float32)
    b = np.full((128, 512), _SIG_B_VAL, dtype=np.float32)

    nki_op = wrap_nki_kernel(
        kernel, [a], kernel_kwargs={"b": b, "c": _SIG_C_VAL},
        is_nki_beta_3_version=True, platform_target="trn2",
    )

    # `b` came from kernel_kwargs but is still a device input, in signature order.
    assert nki_op._beta3_device_input_names == ["a", "b"]

    traced = NKIPyKernel.trace(lambda x, y: nki_op(x, y), backend="hlo")
    traced.specialize(a, b)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
@pytest.mark.skipif(
    not NEURON_AVAILABLE,
    reason="Hardware required - Beta 3 frontend does not support CPU execution",
)
@pytest.mark.parametrize("case", sorted(_SIG_CASES))
@pytest.mark.parametrize(
    "via_kwargs", [False, True], ids=["positional", "kernel_kwargs"]
)
def test_nki_wrap_beta3_signature_shapes_on_device(case, via_kwargs):
    """Every signature shape computes a * c + b correctly on hardware.

    A mis-bound operand or a constant leaking into the operand list yields a
    wrong number here rather than an opaque compile failure.
    """
    kernel_name, order = _SIG_CASES[case]
    kernel = globals()[kernel_name]
    a = np.full((128, 512), _SIG_A_VAL, dtype=np.float32)
    b = np.full((128, 512), _SIG_B_VAL, dtype=np.float32)

    if via_kwargs:
        nki_op = wrap_nki_kernel(
            kernel, [a, b], kernel_kwargs={"c": _SIG_C_VAL},
            is_nki_beta_3_version=True,
        )
    else:
        nki_op = wrap_nki_kernel(
            kernel, _sig_operands(order, a, b, _SIG_C_VAL),
            is_nki_beta_3_version=True,
        )

    out = on_device_test(lambda x, y: nki_op(x, y), "hlo", a, b)
    baremetal_assert_allclose(np.full_like(a, _SIG_EXPECTED), out)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
def test_nki_beta3_keyword_only_unsupported_upstream():
    """Keyword-only params (``*``) are rejected by NKI itself, not by nkipy.

    Documents the one signature shape a kernel author cannot use. The same
    failure occurs through NKI's own execution path with nkipy uninvolved, so
    there is nothing for wrap_nki_kernel to fix; this test pins the behaviour so
    we notice if a future nki release starts supporting it.
    """
    a = np.full((128, 512), _SIG_A_VAL, dtype=np.float32)
    b = np.full((128, 512), _SIG_B_VAL, dtype=np.float32)

    with pytest.raises(RuntimeError, match="failed to compile NKI kernel"):
        wrap_nki_kernel(
            _sig_keyword_only_scalar, [a, b], kernel_kwargs={"c": _SIG_C_VAL},
            is_nki_beta_3_version=True, platform_target="trn2",
        )


# ---------------------------------------------------------------------------
# Beta 3 OPTIONAL MUTABLE OUTPUT tensor
#
# Combines a defaulted keyword tensor with output aliasing:
#
#     def add_kernel(a, b, c=10.0, out=None):
#         if out is None: <allocate>
#         out[...] = a * c + b
#         return out
#
# One source, two compiled shapes, driven purely by whether `out` is supplied:
#   out=None      -> `out` is a trace-time constant; device inputs [a, b]; no alias
#   out=<ndarray> -> `out` is a device input;         device inputs [a, b, out];
#                    alias {2: 0}
#
# The alias map is {input_idx: output_idx} indexed over the DEVICE-INPUT list, so
# it must skip the `c` constant: `out` is index 2, not 3. An off-by-one here would
# alias the wrong buffer, so these tests pin the index explicitly.
# ---------------------------------------------------------------------------


if BETA3_NKI_AVAILABLE:

    @nki_beta3.jit
    def _sig_optional_out(a, b, c=10.0, out=None):
        """out = a * c + b, writing into `out` when the caller supplies one."""
        a_tile = nl_beta3.load(a[:, :])
        b_tile = nl_beta3.load(b[:, :])
        out_tile = nl_beta3.ndarray(
            a_tile.shape, dtype=a_tile.dtype, buffer=nl_beta3.sbuf
        )
        nisa_beta3.scalar_tensor_tensor(
            dst=out_tile, data=a_tile,
            op0=nl_beta3.multiply, operand0=c,
            op1=nl_beta3.add, operand1=b_tile,
        )
        if out is None:
            out = nl_beta3.ndarray(
                a.shape, dtype=a.dtype, buffer=nl_beta3.shared_hbm
            )
        nl_beta3.store(out[:, :], value=out_tile)
        return out


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
def test_nki_wrap_beta3_optional_out_omitted():
    """`out` omitted: it is a trace-time constant and the kernel allocates."""
    a = np.full((128, 512), _SIG_A_VAL, dtype=np.float32)
    b = np.full((128, 512), _SIG_B_VAL, dtype=np.float32)

    nki_op = wrap_nki_kernel(
        _sig_optional_out, [a, b], kernel_kwargs={"c": _SIG_C_VAL},
        is_nki_beta_3_version=True, platform_target="trn2",
    )

    assert nki_op._beta3_device_input_names == ["a", "b"]
    # Both `c` and the unsupplied `out` (None) are non-tensors.
    assert nki_op._beta3_constant_names == ["c", "out"]
    assert nki_op._beta3_framework_config.operand_output_aliases == {}

    traced = NKIPyKernel.trace(lambda x, y: nki_op(x, y), backend="hlo")
    traced.specialize(a, b)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
@pytest.mark.parametrize(
    "via_kwargs", [False, True], ids=["positional", "kernel_kwargs"]
)
def test_nki_wrap_beta3_optional_out_supplied_aliases(via_kwargs):
    """`out` supplied: it becomes a third device input, aliased to output 0.

    The alias index must be 2 (position in the device-input list), NOT 3 (its
    position in the signature) -- the `c` constant does not occupy an operand.
    """
    a = np.full((128, 512), _SIG_A_VAL, dtype=np.float32)
    b = np.full((128, 512), _SIG_B_VAL, dtype=np.float32)
    out = np.zeros((128, 512), dtype=np.float32)

    if via_kwargs:
        nki_op = wrap_nki_kernel(
            _sig_optional_out, [a, b],
            kernel_kwargs={"c": _SIG_C_VAL, "out": out},
            is_nki_beta_3_version=True, platform_target="trn2",
        )
    else:
        # Positionally, `c` must be given to reach `out`.
        nki_op = wrap_nki_kernel(
            _sig_optional_out, [a, b, _SIG_C_VAL, out],
            is_nki_beta_3_version=True, platform_target="trn2",
        )

    assert nki_op._beta3_device_input_names == ["a", "b", "out"]
    assert nki_op._beta3_constant_names == ["c"]
    assert nki_op._beta3_framework_config.operand_output_aliases == {2: 0}

    traced = NKIPyKernel.trace(lambda x, y, o: nki_op(x, y, o), backend="hlo")
    traced.specialize(a, b, out)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
@pytest.mark.skipif(
    not NEURON_AVAILABLE,
    reason="Hardware required - Beta 3 frontend does not support CPU execution",
)
def test_nki_wrap_beta3_optional_out_omitted_on_device():
    """The allocating variant returns a * c + b on hardware."""
    a = np.full((128, 512), _SIG_A_VAL, dtype=np.float32)
    b = np.full((128, 512), _SIG_B_VAL, dtype=np.float32)

    nki_op = wrap_nki_kernel(
        _sig_optional_out, [a, b], kernel_kwargs={"c": _SIG_C_VAL},
        is_nki_beta_3_version=True,
    )

    out = on_device_test(lambda x, y: nki_op(x, y), "hlo", a, b)
    baremetal_assert_allclose(np.full_like(a, _SIG_EXPECTED), out)


@pytest.mark.skipif(
    not BETA3_NKI_AVAILABLE, reason="Beta 3 NKI frontend (nki >= 0.4) not available"
)
@pytest.mark.skipif(
    not NEURON_AVAILABLE,
    reason="Hardware required - Beta 3 frontend does not support CPU execution",
)
def test_nki_wrap_beta3_optional_out_aliased_on_device():
    """The caller's `out` buffer is mutated IN PLACE on hardware.

    Reads back through the caller's own device tensor, so an alias pointing at
    the wrong operand leaves it at its initial 0.0 and fails here.
    """
    from nkipy.runtime import DeviceKernel, DeviceTensor

    a = np.full((128, 512), _SIG_A_VAL, dtype=np.float32)
    b = np.full((128, 512), _SIG_B_VAL, dtype=np.float32)
    out = np.zeros((128, 512), dtype=np.float32)

    nki_op = wrap_nki_kernel(
        _sig_optional_out, [a, b], kernel_kwargs={"c": _SIG_C_VAL, "out": out},
        is_nki_beta_3_version=True,
    )

    def kernel_fn(a, b, out):
        return nki_op(a, b, out)

    traced = NKIPyKernel.trace(kernel_fn, backend="hlo")
    device_kernel = DeviceKernel.compile_and_load(
        traced, a, b, out, use_cached_if_exists=False
    )

    t_a = DeviceTensor.from_numpy(a)
    t_b = DeviceTensor.from_numpy(b)
    t_out = DeviceTensor.from_numpy(out)
    device_kernel(
        inputs={"a": t_a, "b": t_b, "out.must_alias_input": t_out},
        outputs={"out": t_out},
    )

    baremetal_assert_allclose(t_out.numpy(), np.full_like(a, _SIG_EXPECTED))


if __name__ == "__main__":
    # Allow running the test file directly
    pytest.main([__file__])
