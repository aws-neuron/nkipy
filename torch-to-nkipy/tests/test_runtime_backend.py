# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the runtime backend abstraction layer."""

from unittest.mock import MagicMock


class TestRuntimeBackendProtocol:
    """Tests for the RuntimeBackend protocol and LoadedModel dataclass."""

    def test_loaded_model_creation(self):
        """Test LoadedModel dataclass creation."""
        from torch_to_nkipy.device.runtime_backend.protocol import LoadedModel

        model = LoadedModel(model_ref="test_ref", neff_path="/path/to/neff")
        assert model.model_ref == "test_ref"
        assert model.neff_path == "/path/to/neff"

    def test_loaded_model_with_various_refs(self):
        """Test LoadedModel works with different model_ref types."""
        from torch_to_nkipy.device.runtime_backend.protocol import LoadedModel

        # Test with int
        model_int = LoadedModel(model_ref=42, neff_path="/path/to/neff")
        assert model_int.model_ref == 42

        # Test with mock object (spike model)
        mock_spike_model = MagicMock()
        model_mock = LoadedModel(model_ref=mock_spike_model, neff_path="/path/to/neff")
        assert model_mock.model_ref is mock_spike_model


class TestBackendManager:
    """Tests for the backend manager functions."""

    def setup_method(self):
        """Reset backend state before each test."""
        from torch_to_nkipy.device import runtime_backend

        runtime_backend._current_backend = None

    def teardown_method(self):
        """Clean up after each test."""
        from torch_to_nkipy.device import runtime_backend

        runtime_backend._current_backend = None

    def test_get_backend_returns_spike_by_default(self):
        """Test that get_backend returns SpikeBackend by default."""
        from torch_to_nkipy.device.runtime_backend import get_backend
        from torch_to_nkipy.device.runtime_backend.spike_backend import SpikeBackend

        backend = get_backend()
        assert isinstance(backend, SpikeBackend)

    def test_get_backend_singleton(self):
        """Test that get_backend returns the same instance."""
        from torch_to_nkipy.device.runtime_backend import get_backend

        backend1 = get_backend()
        backend2 = get_backend()
        assert backend1 is backend2

    def test_reset_backend(self):
        """Test that reset_backend clears the singleton."""
        from torch_to_nkipy.device.runtime_backend import (
            get_backend,
            reset_backend,
        )

        backend1 = get_backend()
        reset_backend()
        backend2 = get_backend()
        assert backend1 is not backend2


class TestSpikeBackend:
    """Tests for SpikeBackend class."""

    def test_spike_backend_not_initialized_by_default(self):
        """Test SpikeBackend is not initialized by default."""
        from torch_to_nkipy.device.runtime_backend.spike_backend import SpikeBackend

        backend = SpikeBackend()
        assert backend.is_initialized() is False


class TestModuleExports:
    """Tests for module exports and public API."""

    def test_runtime_backend_exports(self):
        """Test runtime_backend module exports expected symbols."""
        from torch_to_nkipy.device import runtime_backend

        assert hasattr(runtime_backend, "LoadedModel")
        assert hasattr(runtime_backend, "RuntimeBackend")
        assert hasattr(runtime_backend, "get_backend")
        assert hasattr(runtime_backend, "reset_backend")

    def test_device_module_exports(self):
        """Test device module exports expected symbols."""
        # These imports should work without error
        from torch_to_nkipy.device import (
            is_nkipy_device_initialized,
            load_spike_model,
            nkipy_close,
            nkipy_init,
            spike_execute,
        )

        assert callable(nkipy_init)
        assert callable(nkipy_close)
        assert callable(is_nkipy_device_initialized)
        assert callable(load_spike_model)
        assert callable(spike_execute)

    def test_loader_returns_loaded_model_type(self):
        """Test that loader's return type annotation is LoadedModel."""
        import inspect

        from torch_to_nkipy.device.loader import load_spike_model
        from torch_to_nkipy.device.runtime_backend import LoadedModel

        sig = inspect.signature(load_spike_model)
        # Check return annotation
        assert sig.return_annotation is LoadedModel
