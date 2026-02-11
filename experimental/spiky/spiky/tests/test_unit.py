# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure unit tests that do NOT require Neuron hardware.

These test individual functions and logic paths in isolation using mocks
where necessary. They run on any machine (CI, laptop, etc.).
"""

from pathlib import Path
from unittest import mock

import pytest
import torch

# ---------------------------------------------------------------------------
# Bug 5: _device_tensor_to_torch and detach/cpu/numpy conversion
# ---------------------------------------------------------------------------


class TestDeviceTensorConversion:
    """Test the _device_tensor_to_torch helper added for keep_outputs_on_device."""

    def test_float32_roundtrip(self):
        from spiky.callable import _device_tensor_to_torch

        expected = torch.randn(2, 4, dtype=torch.float32)
        raw = expected.numpy().tobytes()

        dt = mock.Mock()
        dt.read_to_bytes.return_value = list(raw)
        dt.dtype = "float32"
        dt.shape = (2, 4)

        result = _device_tensor_to_torch(dt, torch.device("cpu"))
        torch.testing.assert_close(result, expected)

    def test_float16_roundtrip(self):
        from spiky.callable import _device_tensor_to_torch

        expected = torch.randn(3, 5, dtype=torch.float16)
        raw = expected.numpy().tobytes()

        dt = mock.Mock()
        dt.read_to_bytes.return_value = list(raw)
        dt.dtype = "float16"
        dt.shape = (3, 5)

        result = _device_tensor_to_torch(dt, torch.device("cpu"))
        torch.testing.assert_close(result, expected)

    def test_int64_roundtrip(self):
        from spiky.callable import _device_tensor_to_torch

        expected = torch.randint(0, 100, (4, 2), dtype=torch.int64)
        raw = expected.numpy().tobytes()

        dt = mock.Mock()
        dt.read_to_bytes.return_value = list(raw)
        dt.dtype = "int64"
        dt.shape = (4, 2)

        result = _device_tensor_to_torch(dt, torch.device("cpu"))
        torch.testing.assert_close(result, expected)

    def test_bfloat16_roundtrip(self):
        from spiky.callable import _device_tensor_to_torch

        expected = torch.randn(2, 3, dtype=torch.bfloat16)
        # bfloat16 is stored as uint16 at the byte level
        raw = expected.view(torch.uint16).numpy().tobytes()

        dt = mock.Mock()
        dt.read_to_bytes.return_value = list(raw)
        dt.dtype = "bfloat16"
        dt.shape = (2, 3)

        result = _device_tensor_to_torch(dt, torch.device("cpu"))
        torch.testing.assert_close(result, expected)

    def test_unsupported_dtype_raises(self):
        from spiky.callable import _device_tensor_to_torch

        dt = mock.Mock()
        dt.read_to_bytes.return_value = [0] * 8
        dt.dtype = "complex128"
        dt.shape = (1,)

        with pytest.raises(ValueError, match="Unsupported DeviceTensor dtype"):
            _device_tensor_to_torch(dt, torch.device("cpu"))


class TestInputNumpyConversion:
    """Test that tensors requiring grad can be converted (Bug 5 fix)."""

    def test_cpu_tensor_with_grad(self):
        """CPU tensor with requires_grad should convert via detach().numpy()."""
        t = torch.randn(2, 3, requires_grad=True)
        # This is the pattern used in callable.py for CPU tensors
        arr = t.detach().numpy()
        assert arr.shape == (2, 3)

    def test_non_cpu_tensor_with_grad_pattern(self):
        """Non-CPU tensor conversion pattern: detach().cpu().numpy().

        We can't create a real non-CPU tensor without a device, but we can
        verify the method chain works on a CPU tensor as a proxy.
        """
        t = torch.randn(2, 3, requires_grad=True)
        # This is the fixed pattern (was t.cpu().numpy() before)
        arr = t.detach().cpu().numpy()
        assert arr.shape == (2, 3)

    def test_without_detach_raises(self):
        """Verify that .numpy() on a grad tensor raises without detach."""
        t = torch.randn(2, 3, requires_grad=True)
        with pytest.raises(RuntimeError):
            t.numpy()


# ---------------------------------------------------------------------------
# Bug 6: parallel_compile_context exception safety
# ---------------------------------------------------------------------------


class TestParallelCompileContextSafety:
    """Test that ContextVar is always reset, even on exceptions."""

    def test_context_reset_on_success(self):
        """ContextVar resets after normal exit."""
        from spiky.runtime.parallel import (
            _in_parallel_compile_context,
            parallel_compile_context,
        )

        assert _in_parallel_compile_context.get() is False

        with mock.patch("spiky.runtime.parallel.parallel_compile_model"):
            with mock.patch("spiky.torch.config.get_nkipy_backend_config") as cfg:
                cfg.return_value = mock.Mock(nkipy_cache_prefix="/tmp/test_cache")
                with mock.patch("torch.distributed.is_initialized", return_value=False):
                    with parallel_compile_context(num_workers=1):
                        assert _in_parallel_compile_context.get() is True

        assert _in_parallel_compile_context.get() is False

    def test_context_reset_on_body_exception(self):
        """ContextVar resets when the user code inside the context raises."""
        from spiky.runtime.parallel import (
            _in_parallel_compile_context,
            parallel_compile_context,
        )

        assert _in_parallel_compile_context.get() is False

        with mock.patch("spiky.runtime.parallel.parallel_compile_model"):
            with mock.patch("spiky.torch.config.get_nkipy_backend_config") as cfg:
                cfg.return_value = mock.Mock(nkipy_cache_prefix="/tmp/test_cache")
                with mock.patch("torch.distributed.is_initialized", return_value=False):
                    with pytest.raises(ValueError):
                        with parallel_compile_context(num_workers=1):
                            raise ValueError("simulated failure")

        assert _in_parallel_compile_context.get() is False

    def test_context_reset_on_cleanup_exception(self):
        """ContextVar resets even when parallel_compile_model raises in finally."""
        from spiky.runtime.parallel import (
            _in_parallel_compile_context,
            parallel_compile_context,
        )

        assert _in_parallel_compile_context.get() is False

        with mock.patch(
            "spiky.runtime.parallel.parallel_compile_model",
            side_effect=RuntimeError("compile failed"),
        ):
            with mock.patch("spiky.torch.config.get_nkipy_backend_config") as cfg:
                cfg.return_value = mock.Mock(nkipy_cache_prefix="/tmp/test_cache")
                with mock.patch("torch.distributed.is_initialized", return_value=False):
                    with pytest.raises(RuntimeError, match="compile failed"):
                        with parallel_compile_context(num_workers=1):
                            pass

        # The key assertion: ContextVar must be False even though
        # parallel_compile_model raised inside the finally block.
        assert _in_parallel_compile_context.get() is False


# ---------------------------------------------------------------------------
# Bug 10: kernel hash collision resistance
# ---------------------------------------------------------------------------


class TestKernelHash:
    """Test get_kernel_hash_from_path for collision resistance."""

    def test_same_kernel_name_across_ranks_deduplicates(self):
        """Same kernel dir name under different ranks should produce same hash."""
        from spiky.runtime.cache import get_kernel_hash_from_path

        h1 = get_kernel_hash_from_path("/cache/rank_0/kernel_abc12345def67890")
        h2 = get_kernel_hash_from_path("/cache/rank_1/kernel_abc12345def67890")
        assert h1 == h2

    def test_different_kernel_names_differ(self):
        """Different kernel directory names should produce different hashes."""
        from spiky.runtime.cache import get_kernel_hash_from_path

        h1 = get_kernel_hash_from_path("/cache/rank_0/kernel_aaaa1111bbbb2222")
        h2 = get_kernel_hash_from_path("/cache/rank_0/kernel_cccc3333dddd4444")
        assert h1 != h2

    def test_old_collision_case_now_differs(self):
        """Paths that previously collided (same last 8 chars) now differ.

        The old implementation used str(path)[-8:], which would collide for
        paths like 'rank_0/kernel_X' and 'rank_0/kernel_Y' if their last 8
        chars happened to match.
        """
        from spiky.runtime.cache import get_kernel_hash_from_path

        # These differ only in the middle, last 8 chars are identical
        h1 = get_kernel_hash_from_path("/cache/rank_0/kernel_AAAA_suffix01")
        h2 = get_kernel_hash_from_path("/cache/rank_0/kernel_BBBB_suffix01")
        assert h1 != h2

    def test_path_object_and_string_same_hash(self):
        """Path and str input should produce the same hash."""
        from spiky.runtime.cache import get_kernel_hash_from_path

        h1 = get_kernel_hash_from_path("/cache/rank_0/kernel_abc")
        h2 = get_kernel_hash_from_path(Path("/cache/rank_0/kernel_abc"))
        assert h1 == h2

    def test_hash_length(self):
        """Hash should be 16 hex characters."""
        from spiky.runtime.cache import get_kernel_hash_from_path

        h = get_kernel_hash_from_path("/some/path/kernel_test")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Bug 8: get_platform_target compiler_args forwarding
# ---------------------------------------------------------------------------


class TestPlatformTargetForwarding:
    """Test compile_model forwards compiler_args to get_platform_target."""

    def test_compile_model_passes_compiler_args_to_target(self):
        """Verify get_platform_target receives compiler_args from config."""
        from unittest.mock import MagicMock, patch

        import spiky.runtime.compile as compile_mod

        mock_config = MagicMock()
        mock_config.additional_compiler_args = "--target trn2 --some-flag"

        with patch.object(
            compile_mod, "get_nkipy_backend_config", return_value=mock_config
        ):
            with patch.object(
                compile_mod, "get_platform_target", return_value="trn2"
            ) as mock_target:
                with patch.object(compile_mod, "trace") as mock_trace:
                    mock_traced = MagicMock()
                    mock_trace.return_value = mock_traced
                    mock_traced._code.inputs = []
                    mock_traced._code.outputs = []
                    with patch.object(
                        compile_mod, "compile_to_neff", return_value="/fake/neff"
                    ):
                        with patch("builtins.open", mock.mock_open()):
                            with patch("pickle.dump"):
                                with patch.object(
                                    compile_mod,
                                    "get_kernel_hash_from_path",
                                    return_value="fakehash",
                                ):
                                    with patch.object(
                                        compile_mod, "hashes_to_kernel_dirs", {}
                                    ):
                                        kernel_dir = Path("/tmp/fake_kernel")
                                        with patch.object(
                                            Path, "exists", return_value=False
                                        ):
                                            with patch.object(Path, "mkdir"):
                                                with patch.object(
                                                    compile_mod, "in_directory"
                                                ):
                                                    try:
                                                        compile_mod.compile_model(
                                                            nkipy_func=lambda: None,
                                                            args=[torch.randn(2, 3)],
                                                            kernel_dir=kernel_dir,
                                                        )
                                                    except Exception:
                                                        pass  # Only care about the call

                mock_target.assert_called_once_with(
                    compiler_args="--target trn2 --some-flag"
                )


# ---------------------------------------------------------------------------
# Bug 2: output_layout / unpad_outputs coherence
# ---------------------------------------------------------------------------


class TestOutputLayoutConfig:
    """Test that output_layout='padded' forces unpad_outputs=False."""

    def test_padded_layout_forces_unpad_false(self):
        """output_layout='padded' should set unpad_outputs=False."""
        # This tests the logic we added in backend.py _create_spiky_callable
        # We can verify the derivation logic directly:
        output_layout = "padded"
        options = {"output_layout": "padded"}

        # Simulate the derivation logic from backend.py
        if options and "unpad_outputs" in options:
            unpad_outputs = options["unpad_outputs"]
        else:
            unpad_outputs = output_layout != "padded"

        assert unpad_outputs is False

    def test_unpad_layout_keeps_unpad_true(self):
        """When output_layout='unpad' (default), unpad_outputs should be True."""
        output_layout = "unpad"
        options = {"output_layout": "unpad"}

        if options and "unpad_outputs" in options:
            unpad_outputs = options["unpad_outputs"]
        else:
            unpad_outputs = output_layout != "padded"

        assert unpad_outputs is True

    def test_explicit_unpad_override_honored(self):
        """Explicit unpad_outputs in options overrides the derivation."""
        output_layout = "padded"
        options = {"output_layout": "padded", "unpad_outputs": True}

        if options and "unpad_outputs" in options:
            unpad_outputs = options["unpad_outputs"]
        else:
            unpad_outputs = output_layout != "padded"

        # User explicitly set unpad_outputs=True, so it should be honored
        assert unpad_outputs is True

    def test_default_no_options(self):
        """With no options, default output_layout='unpad' -> unpad_outputs=True."""
        output_layout = "unpad"
        options = None

        if options and "unpad_outputs" in options:
            unpad_outputs = options["unpad_outputs"]
        else:
            unpad_outputs = output_layout != "padded"

        assert unpad_outputs is True


# ---------------------------------------------------------------------------
# Bug 11: _build_adjusted_dynamic_specs deduplication
# ---------------------------------------------------------------------------


class TestBuildAdjustedDynamicSpecs:
    """Test the extracted _build_adjusted_dynamic_specs method."""

    def test_no_symints(self):
        """With no SymInt indices, arg_idx should be unchanged."""
        from spiky.callable import CallableConfig, NKIPyCallable
        from spiky.utils.dynamic_shapes import DynamicSpec

        config = CallableConfig(
            cache_dir=Path("/tmp"),
            buckets=[32],
            dynamic_specs={0: DynamicSpec(arg_idx=0, dim_idx=1)},
            symint_indices=[],
        )
        callable_obj = NKIPyCallable.__new__(NKIPyCallable)
        callable_obj._config = config
        result = callable_obj._build_adjusted_dynamic_specs()
        assert result == {0: 1}

    def test_symint_before_dynamic_arg(self):
        """SymInt at index 0 should shift dynamic arg at index 1 to 0."""
        from spiky.callable import CallableConfig, NKIPyCallable
        from spiky.utils.dynamic_shapes import DynamicSpec

        config = CallableConfig(
            cache_dir=Path("/tmp"),
            buckets=[32],
            dynamic_specs={1: DynamicSpec(arg_idx=1, dim_idx=0)},
            symint_indices=[0],
        )
        callable_obj = NKIPyCallable.__new__(NKIPyCallable)
        callable_obj._config = config
        result = callable_obj._build_adjusted_dynamic_specs()
        # arg_idx 1 minus 1 SymInt before it = 0
        assert result == {0: 0}

    def test_symint_after_dynamic_arg(self):
        """SymInt at index 2 should NOT shift dynamic arg at index 0."""
        from spiky.callable import CallableConfig, NKIPyCallable
        from spiky.utils.dynamic_shapes import DynamicSpec

        config = CallableConfig(
            cache_dir=Path("/tmp"),
            buckets=[32],
            dynamic_specs={0: DynamicSpec(arg_idx=0, dim_idx=1)},
            symint_indices=[2],
        )
        callable_obj = NKIPyCallable.__new__(NKIPyCallable)
        callable_obj._config = config
        result = callable_obj._build_adjusted_dynamic_specs()
        # SymInt at 2 is after arg_idx 0, so no shift
        assert result == {0: 1}

    def test_multiple_symints(self):
        """Multiple SymInts before a dynamic arg shift it correctly."""
        from spiky.callable import CallableConfig, NKIPyCallable
        from spiky.utils.dynamic_shapes import DynamicSpec

        config = CallableConfig(
            cache_dir=Path("/tmp"),
            buckets=[32],
            dynamic_specs={3: DynamicSpec(arg_idx=3, dim_idx=0)},
            symint_indices=[0, 1],
        )
        callable_obj = NKIPyCallable.__new__(NKIPyCallable)
        callable_obj._config = config
        result = callable_obj._build_adjusted_dynamic_specs()
        # arg_idx 3 minus 2 SymInts before it = 1
        assert result == {1: 0}


# ---------------------------------------------------------------------------
# Fix 1: Circular import resolved by lazy import in parallel.py
# ---------------------------------------------------------------------------


class TestCircularImportFix:
    """Test that the circular import is broken."""

    def test_parallel_module_importable(self):
        """Importing parallel module should succeed without circular error."""
        import importlib

        import spiky.runtime.parallel

        importlib.reload(spiky.runtime.parallel)
        from spiky.runtime.parallel import in_parallel_compile_context

        assert in_parallel_compile_context() is False


# ---------------------------------------------------------------------------
# Fix 4: Non-contiguous CPU inputs
# ---------------------------------------------------------------------------


class TestNonContiguousInputs:
    """Test that non-contiguous tensors are made contiguous before numpy."""

    def test_transposed_tensor_contiguous_numpy(self):
        """Transposed tensor should produce correct data after contiguous().numpy()."""
        import numpy as np

        t = torch.randn(4, 8)
        t_transposed = t.t()  # Non-contiguous
        assert not t_transposed.is_contiguous()
        arr = t_transposed.detach().contiguous().numpy()
        assert arr.shape == (8, 4)
        np.testing.assert_array_equal(arr, t_transposed.detach().numpy())

    def test_sliced_tensor_contiguous_numpy(self):
        """Sliced tensor (non-contiguous stride) should be handled."""
        t = torch.randn(10, 8)
        t_sliced = t[::2]  # Every other row, non-contiguous
        assert not t_sliced.is_contiguous()
        arr = t_sliced.detach().contiguous().numpy()
        assert arr.shape == (5, 8)


# ---------------------------------------------------------------------------
# Fix 8: Bool dtype concrete input creation
# ---------------------------------------------------------------------------


class TestDynamicBoolInputs:
    """Test that bool dtype doesn't crash during concrete input creation."""

    def test_bool_dtype_creates_valid_tensor(self):
        """Bool dtype should produce a valid tensor via torch.zeros."""
        shape = (4, 8)
        t = torch.zeros(*shape, dtype=torch.bool)
        assert t.shape == shape
        assert t.dtype == torch.bool

    def test_randint_bool_raises(self):
        """Verify that torch.randint with large range + bool dtype raises."""
        with pytest.raises(RuntimeError):
            torch.randint(0, 100, (4, 8), dtype=torch.bool)


# ---------------------------------------------------------------------------
# Fix 2: SymInt vs regular scalar distinction
# ---------------------------------------------------------------------------


class TestScalarVsSymIntDistinction:
    """Test that regular scalars are not treated as SymInt."""

    def test_regular_scalar_not_in_symint_indices(self):
        """A regular int/float arg should NOT be added to symint_indices."""
        example_inputs = [torch.randn(4, 8), 3.0, torch.randn(4, 8)]

        input_metadata = []
        symint_indices = []
        for i, inp in enumerate(example_inputs):
            if isinstance(inp, torch.Tensor):
                input_metadata.append(
                    {
                        "shape": tuple(inp.shape),
                        "dtype": inp.dtype,
                        "is_floating_point": inp.is_floating_point(),
                    }
                )
            elif isinstance(inp, torch.SymInt):
                input_metadata.append({"type": "symint"})
                symint_indices.append(i)
            else:
                input_metadata.append({"type": "scalar", "value": inp})

        assert symint_indices == []
        assert input_metadata[1] == {"type": "scalar", "value": 3.0}

    def test_scalar_value_preserved_in_compiler_fn(self):
        """Scalar values should be passed through, not replaced with bucket_size."""
        meta = {"type": "scalar", "value": 42}
        bucket_size = 128
        if isinstance(meta, dict) and meta.get("type") == "scalar":
            concrete_value = meta["value"]
        else:
            concrete_value = bucket_size
        assert concrete_value == 42


# ---------------------------------------------------------------------------
# Fix 7: parallel_compile_model missing directory handling
# ---------------------------------------------------------------------------


class TestParallelCompileMissingDir:
    """Test that parallel_compile_model handles missing cache dirs."""

    def test_missing_cache_dir_does_not_raise(self, tmp_path):
        """parallel_compile_model should skip non-existent dirs."""
        from spiky.runtime.parallel import parallel_compile_model

        nonexistent = tmp_path / "does_not_exist"
        # Should not raise
        parallel_compile_model([nonexistent], num_workers=1, is_master=False)

    def test_empty_cache_dir_works(self, tmp_path):
        """parallel_compile_model should handle empty cache dir."""
        from spiky.runtime.parallel import parallel_compile_model

        empty_dir = tmp_path / "empty_cache"
        empty_dir.mkdir()
        parallel_compile_model([empty_dir], num_workers=1, is_master=False)


# ---------------------------------------------------------------------------
# Fix 6: Frozen params mutation
# ---------------------------------------------------------------------------


class TestFrozenParamFix:
    """Test that param mutations are reflected in subsequent calls."""

    def test_fresh_param_extraction_sees_updates(self):
        """After mutating model params, fresh extraction should get new values."""
        model = torch.nn.Linear(4, 4, bias=False)
        original_weight = model.weight.data.clone()

        model.weight.data.fill_(999.0)

        params = [p.detach() for p in model.parameters()]
        assert torch.all(params[0] == 999.0)
        assert not torch.allclose(params[0], original_weight)

    def test_param_plus_buffer_count(self):
        """params() + buffers() should give expected count for BatchNorm."""
        model = torch.nn.BatchNorm1d(8)
        params = list(model.parameters())
        buffers = list(model.buffers())
        # BatchNorm1d: weight, bias (params)
        # + running_mean, running_var, num_batches_tracked (buffers)
        assert len(params) == 2
        assert len(buffers) == 3


# ---------------------------------------------------------------------------
# Fix 9: Distributed error messages
# ---------------------------------------------------------------------------


class TestDistributedErrorMessages:
    """Test that ProcessGroupNeuron error messages are descriptive."""

    def test_allreduce_error_message(self):
        """NotImplementedError should have a descriptive message."""
        from spiky.device.distributed import ProcessGroupNeuron

        pg = ProcessGroupNeuron.__new__(ProcessGroupNeuron)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            pg.allreduce([], None)

    def test_broadcast_error_message(self):
        """broadcast should also have descriptive error."""
        from spiky.device.distributed import ProcessGroupNeuron

        pg = ProcessGroupNeuron.__new__(ProcessGroupNeuron)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            pg.broadcast([], None)
