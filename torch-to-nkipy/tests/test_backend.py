# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NKIPy backend module.

Tests backend initialization, configuration management, and reset functionality.
"""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from torch_to_nkipy.backend.nkipy_backend import (
    init_nkipy_backend,
    reset_nkipy_backend,
)
from torch_to_nkipy.backend.nkipy_backend_config import (
    get_nkipy_backend_config,
)


class TestBackendInitialization:
    """Tests for init_nkipy_backend function."""

    def test_init_backend_twice_raises_error(self, isolated_backend):
        """Test that initializing backend twice raises RuntimeError."""
        isolated_backend["init"]()

        with pytest.raises(RuntimeError, match="already been initialized"):
            isolated_backend["init"]()

    def test_reset_backend_clears_config(self, isolated_backend):
        """Test that reset_nkipy_backend clears the config."""
        isolated_backend["init"]()
        assert isolated_backend["is_initialized"]()

        isolated_backend["reset"]()

        assert not isolated_backend["is_initialized"]()
        assert isolated_backend["get_config"]() is None

    def test_init_reset_init_cycle(self, isolated_backend):
        """Test that backend can be initialized after reset."""
        # First init
        isolated_backend["init"]()
        assert isolated_backend["is_initialized"]()

        # Reset
        isolated_backend["reset"]()
        assert not isolated_backend["is_initialized"]()

        # Second init should work
        isolated_backend["init"]()
        assert isolated_backend["is_initialized"]()

    def test_auto_detect_rank_from_env(self, isolated_backend):
        """Test that rank is auto-detected from environment variables."""
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}, clear=False):
            isolated_backend["init"]()
            config = isolated_backend["get_config"]()
            assert config.rank == 3

    def test_auto_detect_world_size_from_env(self, isolated_backend):
        """Test that world_size is auto-detected from environment variables."""
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}, clear=False):
            isolated_backend["init"]()
            config = isolated_backend["get_config"]()
            assert config.world_size == 8

    def test_explicit_rank_overrides_env(self, isolated_backend):
        """Test that explicit rank parameter overrides environment variable."""
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}, clear=False):
            isolated_backend["init"](rank=7)
            config = isolated_backend["get_config"]()
            assert config.rank == 7

    def test_explicit_world_size_overrides_env(self, isolated_backend):
        """Test that explicit world_size parameter overrides environment variable."""
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}, clear=False):
            isolated_backend["init"](world_size=4)
            config = isolated_backend["get_config"]()
            assert config.world_size == 4

    def test_core_offset_parameter(self, isolated_backend):
        """Test that core_offset affects visible core calculation."""
        # core_offset affects nkipy_init, not stored in config directly
        # We can only test that it doesn't break initialization
        isolated_backend["init"](core_offset=0)
        assert isolated_backend["is_initialized"]()

    def test_custom_cache_directory(self, isolated_backend, tmp_path):
        """Test that custom cache directory is used."""
        custom_cache = str(tmp_path / "custom_cache")

        # Need to reset first and use direct init
        if isolated_backend["is_initialized"]():
            isolated_backend["reset"]()

        init_nkipy_backend(nkipy_cache=custom_cache)

        config = get_nkipy_backend_config()
        assert config.nkipy_cache_prefix == str(Path(custom_cache).resolve())

        reset_nkipy_backend()

    def test_additional_compiler_args(self, isolated_backend):
        """Test that additional_compiler_args is stored in config."""
        if isolated_backend["is_initialized"]():
            isolated_backend["reset"]()

        init_nkipy_backend(
            nkipy_cache=isolated_backend["cache_dir"],
            additional_compiler_args="--fast-math",
        )

        config = get_nkipy_backend_config()
        assert config.additional_compiler_args == "--fast-math"


class TestBackendConfig:
    """Tests for NKIPyBackendConfig class."""

    def test_config_is_frozen(self, isolated_backend):
        """Test that config is immutable (frozen dataclass)."""
        isolated_backend["init"]()
        config = isolated_backend["get_config"]()

        with pytest.raises(AttributeError):
            config.rank = 999

    def test_config_nkipy_cache_path_includes_rank(self, isolated_backend):
        """Test that nkipy_cache includes rank in path."""
        isolated_backend["init"](rank=5)

        config = isolated_backend["get_config"]()
        assert "rank_5" in config.nkipy_cache
        assert config.nkipy_cache.endswith("/rank_5")

    def test_get_config_returns_none_when_not_initialized(self, isolated_backend):
        """Test that get_nkipy_backend_config returns None when not initialized."""
        # Ensure not initialized
        if isolated_backend["is_initialized"]():
            isolated_backend["reset"]()

        assert isolated_backend["get_config"]() is None

    def test_config_stores_all_parameters(self, isolated_backend, tmp_path):
        """Test that config correctly stores all initialization parameters."""
        cache_dir = str(tmp_path / "test_cache")

        if isolated_backend["is_initialized"]():
            isolated_backend["reset"]()

        init_nkipy_backend(
            nkipy_cache=cache_dir,
            log_level=logging.DEBUG,
            rank=2,
            world_size=16,
            additional_compiler_args="--opt-level=3",
        )

        config = get_nkipy_backend_config()
        assert config.nkipy_cache_prefix == str(Path(cache_dir).resolve())
        assert config.log_level == logging.DEBUG
        assert config.rank == 2
        assert config.world_size == 16
        assert config.additional_compiler_args == "--opt-level=3"

        reset_nkipy_backend()

    def test_config_repr(self, isolated_backend):
        """Test that config has a useful string representation."""
        isolated_backend["init"]()
        config = isolated_backend["get_config"]()

        repr_str = repr(config)
        assert "NKIPyConfig" in repr_str
        assert "nkipy_cache_prefix" in repr_str
        assert "rank" in repr_str
        assert "world_size" in repr_str


class TestBackendEdgeCases:
    """Edge case tests for backend initialization."""

    def test_default_rank_is_zero(self, isolated_backend):
        """Test that default rank is 0 when no env vars or torch.distributed."""
        # Clear relevant env vars
        env_without_rank = {
            k: v for k, v in os.environ.items() if k not in ["LOCAL_RANK", "RANK"]
        }
        with patch.dict(os.environ, env_without_rank, clear=True):
            isolated_backend["init"]()
            config = isolated_backend["get_config"]()
            assert config.rank == 0

    def test_default_world_size_is_one(self, isolated_backend):
        """Test that default world_size is 1 when no env vars or torch.distributed."""
        env_without_world_size = {
            k: v for k, v in os.environ.items() if k != "WORLD_SIZE"
        }
        with patch.dict(os.environ, env_without_world_size, clear=True):
            isolated_backend["init"]()
            config = isolated_backend["get_config"]()
            assert config.world_size == 1

    def test_rank_env_var_priority(self, isolated_backend):
        """Test that LOCAL_RANK takes priority over RANK."""
        with patch.dict(os.environ, {"LOCAL_RANK": "5", "RANK": "10"}, clear=False):
            isolated_backend["init"]()
            config = isolated_backend["get_config"]()
            assert config.rank == 5

    def test_cache_directory_created(self, isolated_backend, tmp_path):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = str(tmp_path / "new_cache_dir")
        assert not Path(cache_dir).exists()

        if isolated_backend["is_initialized"]():
            isolated_backend["reset"]()

        init_nkipy_backend(nkipy_cache=cache_dir)

        assert Path(cache_dir).exists()

        reset_nkipy_backend()
