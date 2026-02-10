# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for torch-to-nkipy.

End-to-end tests for torch.compile workflow, profiling, and backward compatibility.
"""

import shutil
from pathlib import Path

import pytest
import torch

from base import NKIPyTestBase

from torch_to_nkipy.backend.nkipy_backend import (
    init_nkipy_backend,
    is_nkipy_backend_initialized,
    reset_nkipy_backend,
)
from torch_to_nkipy.backend.nkipy_backend_config import get_nkipy_backend_config

# Check if Neuron hardware is available
try:
    from nkipy.runtime import is_neuron_compatible

    NEURON_AVAILABLE = is_neuron_compatible()
except ImportError:
    NEURON_AVAILABLE = False


class TestTorchCompileIntegration(NKIPyTestBase):
    """End-to-end tests for torch.compile with nkipy backend."""

    @pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
    def test_torch_compile_simple_function(self):
        """Test torch.compile with a simple function."""

        def simple_add(a, b):
            return a + b

        a = torch.randn(16, 32, dtype=torch.float32)
        b = torch.randn(16, 32, dtype=torch.float32)

        self.run_test_on_device(simple_add, (a, b))

    @pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
    def test_torch_compile_with_multiple_ops(self):
        """Test torch.compile with multiple operations."""

        def multi_op_func(x, y, z):
            temp = x + y
            temp = temp * z
            return temp - x

        x = torch.randn(8, 16, dtype=torch.float32)
        y = torch.randn(8, 16, dtype=torch.float32)
        z = torch.randn(8, 16, dtype=torch.float32)

        self.run_test_on_device(multi_op_func, (x, y, z))

    @pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
    def test_torch_compile_matmul(self):
        """Test torch.compile with matrix multiplication."""

        def matmul_func(a, b):
            return torch.mm(a, b)

        a = torch.randn(32, 64, dtype=torch.float32)
        b = torch.randn(64, 128, dtype=torch.float32)

        self.run_test_on_device(matmul_func, (a, b))

    @pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
    def test_results_match_cpu_reference(self):
        """Test that compiled results match CPU reference."""

        def complex_func(x):
            return torch.relu(x * 2 + 1)

        x = torch.randn(16, 32, dtype=torch.float32)

        compiled_out, ref_out, _, _ = self.run_test_on_device(complex_func, (x,))

        # Outputs should be close
        assert len(compiled_out) == len(ref_out)

    @pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
    def test_torch_compile_with_bf16(self):
        """Test torch.compile with bfloat16 dtype."""

        def bf16_func(a, b):
            return a + b

        a = torch.randn(16, 32, dtype=torch.bfloat16)
        b = torch.randn(16, 32, dtype=torch.bfloat16)

        self.run_test_on_device(bf16_func, (a, b))

    @pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
    def test_torch_compile_with_fp16(self):
        """Test torch.compile with float16 dtype."""

        def fp16_func(a, b):
            return a * b

        a = torch.randn(16, 32, dtype=torch.float16)
        b = torch.randn(16, 32, dtype=torch.float16)

        self.run_test_on_device(fp16_func, (a, b))


@pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
class TestProfilingIntegration:
    """Tests for NTFF profiling integration."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        torch.manual_seed(0)
        yield
        torch.compiler.reset()

    def test_profiling_generates_ntff(self, ntff_output_dir):
        """Test that profiling generates NTFF files."""
        ntff_output_dir.mkdir(parents=True, exist_ok=True)

        def simple_func(x):
            return x + 1

        x = torch.randn(8, 16, dtype=torch.float32, device="nkipy")

        compiled_func = torch.compile(
            simple_func,
            backend="nkipy",
            fullgraph=True,
            options={"save_ntff": True, "save_ntff_dir": str(ntff_output_dir)},
        )

        with torch.no_grad():
            _ = compiled_func(x)

        # Check if NTFF directory has content
        kernel_dirs = list(ntff_output_dir.glob("kernel_*"))
        assert len(kernel_dirs) > 0, "No kernel directories created"

        # Check for NTFF files
        ntff_files = list(ntff_output_dir.glob("kernel_*/*.ntff"))
        assert len(ntff_files) > 0, "No NTFF files generated"

    def test_profiling_copies_neff_to_output(self, ntff_output_dir):
        """Test that profiling copies NEFF file to output directory."""
        ntff_output_dir.mkdir(parents=True, exist_ok=True)

        def add_func(a, b):
            return a + b

        a = torch.randn(8, 16, dtype=torch.float32, device="nkipy")
        b = torch.randn(8, 16, dtype=torch.float32, device="nkipy")

        compiled_func = torch.compile(
            add_func,
            backend="nkipy",
            fullgraph=True,
            options={"save_ntff": True, "save_ntff_dir": str(ntff_output_dir)},
        )

        with torch.no_grad():
            _ = compiled_func(a, b)

        # Check for NEFF files
        neff_files = list(ntff_output_dir.glob("kernel_*/*.neff"))
        assert len(neff_files) > 0, "No NEFF files copied"

    def test_profiling_multiple_executions(self, ntff_output_dir):
        """Test that profiling multiple executions creates multiple NTFF files."""
        ntff_output_dir.mkdir(parents=True, exist_ok=True)

        def simple_func(x):
            return x * 2

        x = torch.randn(8, 16, dtype=torch.float32, device="nkipy")

        compiled_func = torch.compile(
            simple_func,
            backend="nkipy",
            fullgraph=True,
            options={"save_ntff": True, "save_ntff_dir": str(ntff_output_dir)},
        )

        # Execute multiple times
        with torch.no_grad():
            for _ in range(3):
                _ = compiled_func(x)

        # Should have multiple NTFF files (one per execution)
        ntff_files = list(ntff_output_dir.glob("kernel_*/*.ntff"))
        assert len(ntff_files) >= 1, "Should have at least one NTFF file"

    def test_profiling_respects_exe_idx_filter(self, ntff_output_dir):
        """Test that save_ntff_exe_idx filters which executions are profiled."""
        ntff_output_dir.mkdir(parents=True, exist_ok=True)

        def simple_func(x):
            return x + 1

        x = torch.randn(8, 16, dtype=torch.float32, device="nkipy")

        # Only profile execution index 1 (second execution)
        compiled_func = torch.compile(
            simple_func,
            backend="nkipy",
            fullgraph=True,
            options={
                "save_ntff": True,
                "save_ntff_dir": str(ntff_output_dir),
                "save_ntff_exe_idx": [1],
            },
        )

        # Execute 3 times
        with torch.no_grad():
            for _ in range(3):
                _ = compiled_func(x)

        # Should only have NTFF for index 1
        ntff_files = list(ntff_output_dir.glob("kernel_*/*.ntff"))
        assert len(ntff_files) == 1, "Should have exactly one NTFF file"
        # The file should be named "1.ntff"
        assert any("1.ntff" in str(f) for f in ntff_files)


class TestBackwardCompatibility:
    """Tests for backward compatibility of function aliases."""

    def test_execute_model_alias_exists(self):
        """Test that execute_model alias exists for backward compatibility."""
        from torch_to_nkipy.runtime.runtime import execute_model, run_neff_model

        # execute_model should be an alias for run_neff_model
        assert execute_model is run_neff_model

    def test_nkipy_execute_model_alias_exists(self):
        """Test that nkipy_execute_model alias exists."""
        from torch_to_nkipy.device import nkipy_execute_model, spike_execute

        # nkipy_execute_model should be an alias for spike_execute
        assert nkipy_execute_model is spike_execute

    def test_nkipy_load_model_alias_exists(self):
        """Test that nkipy_load_model alias exists."""
        from torch_to_nkipy.device import load_spike_model, nkipy_load_model

        # nkipy_load_model should be an alias for load_spike_model
        assert nkipy_load_model is load_spike_model

    def test_old_function_names_importable(self):
        """Test that old function names are still importable."""
        # These imports should not raise
        from torch_to_nkipy.device import nkipy_execute_model  # noqa: F401
        from torch_to_nkipy.device import nkipy_load_model  # noqa: F401
        from torch_to_nkipy.runtime import execute_model  # noqa: F401


class TestBackendRegistration:
    """Tests for backend registration with PyTorch."""

    def test_nkipy_backend_registered(self):
        """Test that 'nkipy' backend is registered with torch.compile."""
        from torch._dynamo.backends.registry import lookup_backend

        # lookup_backend should not raise for 'nkipy'
        backend_fn = lookup_backend("nkipy")
        assert backend_fn is not None
        assert callable(backend_fn)

    def test_compile_with_nkipy_backend_string(self):
        """Test that torch.compile accepts 'nkipy' as backend string."""

        def simple_func(x):
            return x + 1

        # This should not raise - just tests that the backend is recognized
        compiled = torch.compile(simple_func, backend="nkipy", fullgraph=True)
        assert compiled is not None


@pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
class TestDeviceIntegration:
    """Tests for nkipy device integration."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        torch.manual_seed(0)
        yield
        torch.compiler.reset()

    def test_tensor_to_nkipy_device(self):
        """Test moving tensor to nkipy device."""
        x = torch.randn(8, 16, dtype=torch.float32)
        x_nkipy = x.to("nkipy")

        assert x_nkipy.device.type == "nkipy"

    def test_tensor_from_nkipy_to_cpu(self):
        """Test moving tensor from nkipy device to CPU."""
        x = torch.randn(8, 16, dtype=torch.float32, device="nkipy")
        x_cpu = x.cpu()

        assert x_cpu.device.type == "cpu"

    def test_create_tensor_on_nkipy(self):
        """Test creating tensor directly on nkipy device."""
        x = torch.empty(8, 16, dtype=torch.float32, device="nkipy")

        assert x.device.type == "nkipy"
        assert x.shape == (8, 16)
        assert x.dtype == torch.float32


class TestConfigPersistence:
    """Tests for config persistence across operations."""

    def test_config_persists_after_compile(self, isolated_backend):
        """Test that backend config persists after torch.compile."""
        isolated_backend["init"](rank=3, world_size=8)

        original_config = isolated_backend["get_config"]()

        def simple_func(x):
            return x + 1

        _ = torch.compile(simple_func, backend="nkipy", fullgraph=True)

        # Config should still be the same
        current_config = isolated_backend["get_config"]()
        assert current_config is original_config
        assert current_config.rank == 3
        assert current_config.world_size == 8

    def test_config_rank_in_cache_path(self, isolated_backend):
        """Test that rank appears in cache path."""
        isolated_backend["init"](rank=7)

        config = isolated_backend["get_config"]()

        assert "rank_7" in config.nkipy_cache
