# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the runtime backend abstraction layer."""

import pytest
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

        # Test with int (spiky bundle_id)
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
        runtime_backend._use_spiky = False

    def teardown_method(self):
        """Clean up after each test."""
        from torch_to_nkipy.device import runtime_backend

        runtime_backend._current_backend = None
        runtime_backend._use_spiky = False

    def test_set_runtime_type_spike(self):
        """Test setting runtime type to spike."""
        from torch_to_nkipy.device.runtime_backend import (
            set_runtime_type,
            get_runtime_type,
        )

        set_runtime_type(False)
        assert get_runtime_type() is False

    def test_set_runtime_type_spiky(self):
        """Test setting runtime type to spiky."""
        from torch_to_nkipy.device.runtime_backend import (
            set_runtime_type,
            get_runtime_type,
        )

        set_runtime_type(True)
        assert get_runtime_type() is True

    def test_get_backend_returns_spike_by_default(self):
        """Test that get_backend returns SpikeBackend by default."""
        from torch_to_nkipy.device.runtime_backend import get_backend, set_runtime_type
        from torch_to_nkipy.device.runtime_backend.spike_backend import SpikeBackend

        set_runtime_type(False)
        backend = get_backend()
        assert isinstance(backend, SpikeBackend)

    def test_get_backend_returns_spiky_when_configured(self):
        """Test that get_backend returns SpikyBackend when use_spiky=True."""
        from torch_to_nkipy.device.runtime_backend import get_backend, set_runtime_type
        from torch_to_nkipy.device.runtime_backend.spiky_backend import SpikyBackend

        set_runtime_type(True)
        backend = get_backend()
        assert isinstance(backend, SpikyBackend)

    def test_get_backend_singleton(self):
        """Test that get_backend returns the same instance."""
        from torch_to_nkipy.device.runtime_backend import get_backend, set_runtime_type

        set_runtime_type(False)
        backend1 = get_backend()
        backend2 = get_backend()
        assert backend1 is backend2

    def test_reset_backend(self):
        """Test that reset_backend clears the singleton."""
        from torch_to_nkipy.device.runtime_backend import (
            get_backend,
            reset_backend,
            set_runtime_type,
        )

        set_runtime_type(False)
        backend1 = get_backend()
        reset_backend()
        backend2 = get_backend()
        assert backend1 is not backend2

    def test_set_runtime_type_resets_backend(self):
        """Test that changing runtime type resets the backend."""
        from torch_to_nkipy.device.runtime_backend import get_backend, set_runtime_type
        from torch_to_nkipy.device.runtime_backend.spike_backend import SpikeBackend
        from torch_to_nkipy.device.runtime_backend.spiky_backend import SpikyBackend

        set_runtime_type(False)
        backend_spike = get_backend()
        assert isinstance(backend_spike, SpikeBackend)

        set_runtime_type(True)
        backend_spiky = get_backend()
        assert isinstance(backend_spiky, SpikyBackend)
        assert backend_spike is not backend_spiky


class TestSpikeBackend:
    """Tests for SpikeBackend class."""

    def test_spike_backend_not_initialized_by_default(self):
        """Test SpikeBackend is not initialized by default."""
        from torch_to_nkipy.device.runtime_backend.spike_backend import SpikeBackend

        backend = SpikeBackend()
        assert backend.is_initialized() is False


class TestSpikyBackend:
    """Tests for SpikyBackend class."""

    def test_spiky_backend_not_initialized_by_default(self):
        """Test SpikyBackend is not initialized by default."""
        from torch_to_nkipy.device.runtime_backend.spiky_backend import SpikyBackend

        backend = SpikyBackend()
        assert backend._initialized is False

    def test_spiky_backend_bundle_registry_empty_by_default(self):
        """Test SpikyBackend has empty bundle registry by default."""
        from torch_to_nkipy.device.runtime_backend.spiky_backend import SpikyBackend

        backend = SpikyBackend()
        assert len(backend._bundle_registry) == 0


class TestBackendConfigIntegration:
    """Tests for backend config integration with use_spiky flag."""

    def test_config_use_spiky_default_false(self):
        """Test NKIPyBackendConfig has use_spiky=False by default."""
        from torch_to_nkipy.backend.nkipy_backend_config import NKIPyBackendConfig

        config = NKIPyBackendConfig(
            nkipy_cache_prefix="/tmp/test",
            log_level=20,
            rank=0,
            world_size=1,
            additional_compiler_args="",
        )
        assert config.use_spiky is False

    def test_config_use_spiky_true(self):
        """Test NKIPyBackendConfig with use_spiky=True."""
        from torch_to_nkipy.backend.nkipy_backend_config import NKIPyBackendConfig

        config = NKIPyBackendConfig(
            nkipy_cache_prefix="/tmp/test",
            log_level=20,
            rank=0,
            world_size=1,
            additional_compiler_args="",
            use_spiky=True,
        )
        assert config.use_spiky is True

    def test_config_repr_includes_use_spiky(self):
        """Test NKIPyBackendConfig repr includes use_spiky."""
        from torch_to_nkipy.backend.nkipy_backend_config import NKIPyBackendConfig

        config = NKIPyBackendConfig(
            nkipy_cache_prefix="/tmp/test",
            log_level=20,
            rank=0,
            world_size=1,
            additional_compiler_args="",
            use_spiky=True,
        )
        repr_str = repr(config)
        assert "use_spiky" in repr_str
        assert "True" in repr_str


class TestModuleExports:
    """Tests for module exports and public API."""

    def test_runtime_backend_exports(self):
        """Test runtime_backend module exports expected symbols."""
        from torch_to_nkipy.device import runtime_backend

        assert hasattr(runtime_backend, "LoadedModel")
        assert hasattr(runtime_backend, "RuntimeBackend")
        assert hasattr(runtime_backend, "get_backend")
        assert hasattr(runtime_backend, "set_runtime_type")
        assert hasattr(runtime_backend, "get_runtime_type")
        assert hasattr(runtime_backend, "reset_backend")

    def test_device_module_exports(self):
        """Test device module exports expected symbols."""
        # These imports should work without error
        from torch_to_nkipy.device import (
            nkipy_init,
            nkipy_close,
            is_nkipy_device_initialized,
            load_spike_model,
            spike_execute,
        )

        assert callable(nkipy_init)
        assert callable(nkipy_close)
        assert callable(is_nkipy_device_initialized)
        assert callable(load_spike_model)
        assert callable(spike_execute)

    def test_loader_returns_loaded_model_type(self):
        """Test that loader's return type annotation is LoadedModel."""
        from torch_to_nkipy.device.loader import load_spike_model
        from torch_to_nkipy.device.runtime_backend import LoadedModel
        import inspect

        sig = inspect.signature(load_spike_model)
        # Check return annotation
        assert sig.return_annotation is LoadedModel
