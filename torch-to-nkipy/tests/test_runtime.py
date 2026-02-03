# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NKIPy runtime module.

Tests compilation, model loading, caching, and execution functionality.
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from torch_to_nkipy.runtime.runtime import (
    IOSpecs,
    NeuronExecutable,
    TensorSpec,
    compile_model,
    get_compile_dir_and_neff_path,
    get_kernel_hash_from_path,
    hashes_to_kernel_dirs,
    in_parallel_compile_context,
    load_model,
    loaded_models,
    parallel_compile_context,
)
from torch_to_nkipy.utils.name import COMPILER_DIR, IO_SPECS_FILE, NEFF_FILE

# Check if Neuron hardware is available
try:
    from nkipy.runtime import is_neuron_compatible

    NEURON_AVAILABLE = is_neuron_compatible()
except ImportError:
    NEURON_AVAILABLE = False


class TestTensorSpec:
    """Tests for TensorSpec dataclass."""

    def test_tensor_spec_creation(self):
        """Test creating a TensorSpec with valid values."""
        spec = TensorSpec(name="input_0", shape=(2, 3, 4), dtype=torch.float32)

        assert spec.name == "input_0"
        assert spec.shape == (2, 3, 4)
        assert spec.dtype == torch.float32

    def test_tensor_spec_different_dtypes(self):
        """Test TensorSpec with different data types."""
        dtypes = [
            torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64
        ]

        for dtype in dtypes:
            spec = TensorSpec(name="test", shape=(1,), dtype=dtype)
            assert spec.dtype == dtype


class TestIOSpecs:
    """Tests for IOSpecs dataclass."""

    def test_io_specs_creation(self):
        """Test creating IOSpecs with input and output specs."""
        input_specs = [
            TensorSpec(name="input_0", shape=(2, 3), dtype=torch.float32),
            TensorSpec(name="input_1", shape=(3, 4), dtype=torch.float32),
        ]
        output_specs = [
            TensorSpec(name="output_0", shape=(2, 4), dtype=torch.float32),
        ]

        io_specs = IOSpecs(input_specs=input_specs, output_specs=output_specs)

        assert len(io_specs.input_specs) == 2
        assert len(io_specs.output_specs) == 1
        assert io_specs.input_specs[0].name == "input_0"
        assert io_specs.output_specs[0].name == "output_0"

    def test_io_specs_empty_lists(self):
        """Test IOSpecs with empty lists."""
        io_specs = IOSpecs(input_specs=[], output_specs=[])

        assert len(io_specs.input_specs) == 0
        assert len(io_specs.output_specs) == 0


class TestNeuronExecutable:
    """Tests for NeuronExecutable dataclass."""

    def test_neuron_executable_creation(self):
        """Test creating a NeuronExecutable."""
        mock_model = MagicMock()
        io_specs = IOSpecs(input_specs=[], output_specs=[])
        neff_path = "/path/to/model.neff"

        executable = NeuronExecutable(
            nkipy_model=mock_model, io_specs=io_specs, neff_path=neff_path
        )

        assert executable.nkipy_model == mock_model
        assert executable.io_specs == io_specs
        assert executable.neff_path == neff_path


class TestUtilityFunctions:
    """Tests for utility functions in runtime module."""

    def test_get_compile_dir_and_neff_path(self, tmp_path):
        """Test get_compile_dir_and_neff_path returns correct paths."""
        kernel_dir = tmp_path / "kernel_abc123"

        compile_dir, neff_path = get_compile_dir_and_neff_path(kernel_dir)

        assert compile_dir == kernel_dir / COMPILER_DIR
        assert neff_path == kernel_dir / COMPILER_DIR / NEFF_FILE

    def test_get_kernel_hash_from_path(self):
        """Test get_kernel_hash_from_path extracts correct hash."""
        path1 = Path("/cache/rank_0/kernel_abc12345")
        path2 = "/cache/rank_0/kernel_xyz98765"

        assert get_kernel_hash_from_path(path1) == "abc12345"
        assert get_kernel_hash_from_path(path2) == "xyz98765"


class TestModelCaching:
    """Tests for model caching behavior."""

    def test_loaded_models_cache_exists(self):
        """Test that loaded_models cache dictionary exists."""
        assert isinstance(loaded_models, dict)

    def test_hashes_to_kernel_dirs_cache_exists(self):
        """Test that hashes_to_kernel_dirs cache dictionary exists."""
        assert isinstance(hashes_to_kernel_dirs, dict)


class TestParallelCompileContext:
    """Tests for parallel compilation context manager."""

    def test_in_parallel_compile_context_default_false(self):
        """Test that in_parallel_compile_context returns False by default."""
        assert in_parallel_compile_context() is False

    @pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
    def test_parallel_compile_context_sets_flag(self, isolated_backend):
        """Test that parallel_compile_context sets the context flag."""
        # Backend must be initialized for parallel_compile_context to work
        isolated_backend["init"]()

        assert in_parallel_compile_context() is False

        # We can't fully test this without mocking torch.distributed
        # but we can verify the context manager structure exists
        with patch("torch.distributed.is_initialized", return_value=False):
            with patch("torch_to_nkipy.runtime.runtime.parallel_compile_model"):
                with parallel_compile_context(num_workers=1):
                    assert in_parallel_compile_context() is True

        assert in_parallel_compile_context() is False


class TestCompileModel:
    """Tests for compile_model function."""

    @pytest.fixture
    def mock_kernel_dir(self, tmp_path):
        """Create a mock kernel directory structure."""
        kernel_dir = tmp_path / "kernel_test1234"
        kernel_dir.mkdir(parents=True)
        return kernel_dir

    def test_compile_model_uses_cache_when_exists(self, mock_kernel_dir):
        """Test that compile_model uses cached artifacts when they exist."""
        compile_dir = mock_kernel_dir / COMPILER_DIR
        compile_dir.mkdir(parents=True)

        # Create mock NEFF file
        neff_path = compile_dir / NEFF_FILE
        neff_path.touch()

        # Create mock IO specs
        io_specs = IOSpecs(
            input_specs=[TensorSpec("in", (2, 3), torch.float32)],
            output_specs=[TensorSpec("out", (2, 3), torch.float32)],
        )
        io_specs_path = compile_dir / IO_SPECS_FILE
        with open(io_specs_path, "wb") as f:
            pickle.dump(io_specs, f)

        # Mock function and args
        mock_func = lambda x: x  # noqa: E731
        mock_args = (torch.randn(2, 3),)

        result_path, result_specs = compile_model(
            nkipy_func=mock_func,
            args=mock_args,
            kernel_dir=mock_kernel_dir,
        )

        assert result_path == neff_path
        assert result_specs.input_specs[0].name == "in"
        assert result_specs.output_specs[0].name == "out"


@pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
class TestCompileModelHardware:
    """Tests for compile_model that require Neuron hardware."""

    @pytest.fixture
    def setup_backend(self, isolated_backend):
        """Ensure backend is initialized for hardware tests."""
        isolated_backend["init"]()
        yield
        isolated_backend["reset"]()

    def test_compile_simple_function(self, setup_backend, tmp_path):
        """Test compiling a simple function to NEFF."""
        kernel_dir = tmp_path / "kernel_simple12"
        kernel_dir.mkdir(parents=True)

        def simple_add(a, b):
            return a + b

        args = (
            torch.randn(16, 32, dtype=torch.float32),
            torch.randn(16, 32, dtype=torch.float32),
        )

        neff_path, io_specs = compile_model(
            nkipy_func=simple_add, args=args, kernel_dir=kernel_dir
        )

        assert neff_path.exists()
        assert len(io_specs.input_specs) == 2
        assert len(io_specs.output_specs) == 1

    def test_compile_caches_neff(self, setup_backend, tmp_path):
        """Test that compilation creates NEFF cache."""
        kernel_dir = tmp_path / "kernel_cache123"
        kernel_dir.mkdir(parents=True)

        def mul_func(a, b):
            return a * b

        args = (
            torch.randn(8, 16, dtype=torch.float32),
            torch.randn(8, 16, dtype=torch.float32),
        )

        neff_path, _ = compile_model(
            nkipy_func=mul_func, args=args, kernel_dir=kernel_dir
        )

        # Verify cache artifacts exist
        assert neff_path.exists()
        io_specs_path = kernel_dir / COMPILER_DIR / IO_SPECS_FILE
        assert io_specs_path.exists()

    def test_compile_reuses_cache(self, setup_backend, tmp_path):
        """Test that second compile reuses cached artifacts."""
        kernel_dir = tmp_path / "kernel_reuse12"
        kernel_dir.mkdir(parents=True)

        def simple_func(x):
            return x * 2

        args = (torch.randn(4, 8, dtype=torch.float32),)

        # First compile
        neff_path1, io_specs1 = compile_model(
            nkipy_func=simple_func, args=args, kernel_dir=kernel_dir
        )

        # Record modification time
        mtime1 = neff_path1.stat().st_mtime

        # Second compile should reuse cache
        neff_path2, io_specs2 = compile_model(
            nkipy_func=simple_func, args=args, kernel_dir=kernel_dir
        )

        # Same path and unchanged mtime (tolerance for filesystem precision)
        assert neff_path1 == neff_path2
        assert neff_path2.stat().st_mtime <= mtime1 + 0.001

    def test_io_specs_saved_correctly(self, setup_backend, tmp_path):
        """Test that IO specs are correctly saved and loaded."""
        kernel_dir = tmp_path / "kernel_iospecs"
        kernel_dir.mkdir(parents=True)

        def func_with_multiple_outputs(a, b):
            return a + b, a - b

        args = (
            torch.randn(16, 32, dtype=torch.float32),
            torch.randn(16, 32, dtype=torch.float32),
        )

        _, io_specs = compile_model(
            nkipy_func=func_with_multiple_outputs, args=args, kernel_dir=kernel_dir
        )

        # Verify IO specs structure
        assert len(io_specs.input_specs) == 2
        for spec in io_specs.input_specs:
            assert spec.shape == (16, 32)
            assert spec.dtype == torch.float32


@pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
class TestLoadModel:
    """Tests for load_model function that require Neuron hardware."""

    @pytest.fixture
    def setup_backend(self, isolated_backend):
        """Ensure backend is initialized for hardware tests."""
        isolated_backend["init"]()
        yield
        # Clear model cache after each test
        loaded_models.clear()
        isolated_backend["reset"]()

    def test_load_model_returns_executable(self, setup_backend, tmp_path):
        """Test that load_model returns a NeuronExecutable."""
        kernel_dir = tmp_path / "kernel_load1234"
        kernel_dir.mkdir(parents=True)

        def simple_func(x):
            return x + 1

        args = (torch.randn(8, 16, dtype=torch.float32),)

        executable = load_model(
            nkipy_func=simple_func,
            kernel_hash="load1234",
            args=args,
            kernel_dir=kernel_dir,
        )

        assert isinstance(executable, NeuronExecutable)
        assert executable.nkipy_model is not None
        assert executable.io_specs is not None

    def test_load_model_caches_result(self, setup_backend, tmp_path):
        """Test that load_model caches the loaded model."""
        kernel_dir = tmp_path / "kernel_cache234"
        kernel_dir.mkdir(parents=True)
        kernel_hash = "cache234"

        def simple_func(x):
            return x * 2

        args = (torch.randn(4, 8, dtype=torch.float32),)

        # First load
        executable1 = load_model(
            nkipy_func=simple_func,
            kernel_hash=kernel_hash,
            args=args,
            kernel_dir=kernel_dir,
        )

        # Second load should return cached
        executable2 = load_model(
            nkipy_func=simple_func,
            kernel_hash=kernel_hash,
            args=args,
            kernel_dir=kernel_dir,
        )

        assert executable1 is executable2
        assert kernel_hash in loaded_models


@pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
class TestRunNeffModel:
    """Tests for run_neff_model function that require Neuron hardware."""

    @pytest.fixture
    def setup_backend(self, isolated_backend):
        """Ensure backend is initialized for hardware tests."""
        isolated_backend["init"]()
        yield
        loaded_models.clear()
        isolated_backend["reset"]()

    def test_run_neff_model_basic_output(self, setup_backend, tmp_path):
        """Test basic execution returns expected output tensors."""
        from torch_to_nkipy.runtime.runtime import compile_load_execute
        from torch_to_nkipy.utils.ntff_meta import NtffMeta

        kernel_dir = tmp_path / "kernel_run12345"
        kernel_dir.mkdir(parents=True)

        def add_func(a, b):
            return a + b

        a = torch.randn(16, 32, dtype=torch.float32, device="nkipy")
        b = torch.randn(16, 32, dtype=torch.float32, device="nkipy")

        ntff_meta = NtffMeta()

        outputs = compile_load_execute(
            nkipy_func=add_func,
            kernel_hash="run12345",
            args=(a, b),
            alias_map={},
            none_idx_list=[],
            kernel_dir=kernel_dir,
            ntff_meta=ntff_meta,
        )

        assert len(outputs) == 1
        assert outputs[0].shape == (16, 32)
        assert outputs[0].dtype == torch.float32


class TestKernelHashConsistency:
    """Tests for kernel hash consistency."""

    def test_kernel_hash_extraction_consistent(self):
        """Test that kernel hash extraction is consistent."""
        paths = [
            "/cache/rank_0/kernel_abcd1234",
            "/cache/rank_1/kernel_abcd1234",
            "/other/path/kernel_abcd1234",
        ]

        hashes = [get_kernel_hash_from_path(p) for p in paths]

        # All should extract the same hash
        assert all(h == "abcd1234" for h in hashes)

    def test_different_kernels_different_hashes(self):
        """Test that different kernels have different hashes."""
        path1 = "/cache/kernel_aaaa1111"
        path2 = "/cache/kernel_bbbb2222"

        hash1 = get_kernel_hash_from_path(path1)
        hash2 = get_kernel_hash_from_path(path2)

        assert hash1 != hash2
        assert hash1 == "aaaa1111"
        assert hash2 == "bbbb2222"
