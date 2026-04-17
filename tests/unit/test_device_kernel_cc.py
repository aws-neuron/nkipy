# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CC parameters in DeviceKernel.compile_and_load."""

from unittest.mock import MagicMock, patch

import pytest

from nkipy.runtime.device_kernel import DeviceKernel, _LOADED_KERNELS


def _dummy_kernel(x):
    return x


@pytest.fixture(autouse=True)
def clear_kernel_cache():
    _LOADED_KERNELS.clear()
    yield
    _LOADED_KERNELS.clear()


@pytest.fixture
def mock_trace_and_compile():
    with patch.object(
        DeviceKernel,
        "_trace_and_compile",
        return_value=("/tmp/test/kernel.neff", "kernel_abc123"),
    ) as m:
        yield m


@pytest.fixture
def mock_load_from_neff():
    with patch.object(
        DeviceKernel, "load_from_neff", return_value=MagicMock(spec=DeviceKernel)
    ) as m:
        yield m


@pytest.fixture
def mock_dist():
    """Mock torch.distributed as initialized with world_size=2, rank=0."""
    with (
        patch("nkipy.runtime.device_kernel._is_distributed", return_value=True),
        patch("nkipy.runtime.device_kernel.dist", create=True) as mock_d,
    ):
        mock_d.get_rank.return_value = 0
        mock_d.get_world_size.return_value = 2
        yield mock_d


@pytest.fixture
def mock_dist_rank1():
    """Mock torch.distributed as initialized with world_size=2, rank=1."""
    with (
        patch("nkipy.runtime.device_kernel._is_distributed", return_value=True),
        patch("nkipy.runtime.device_kernel.dist", create=True) as mock_d,
    ):
        mock_d.get_rank.return_value = 1
        mock_d.get_world_size.return_value = 2
        yield mock_d


class TestSPMDMode:
    """Tests for is_spmd=True (default) with torch.distributed."""

    def test_spmd_uses_broadcast(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """Default (is_spmd=True) with distributed uses rank-0 broadcast."""
        DeviceKernel.compile_and_load(_dummy_kernel)

        mock_trace_and_compile.assert_called_once()
        mock_dist.broadcast_object_list.assert_called_once()

    def test_spmd_uses_barrier(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """SPMD mode calls barrier before load."""
        DeviceKernel.compile_and_load(_dummy_kernel)

        mock_dist.barrier.assert_called_once()

    def test_spmd_resolves_rank_and_world_size(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """SPMD auto-detects rank_id and world_size from dist."""
        DeviceKernel.compile_and_load(_dummy_kernel)

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff",
            name="_dummy_kernel",
            cc_enabled=True,
            rank_id=0,
            world_size=2,
        )

    def test_spmd_barrier_still_fires_when_cc_explicit(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """SPMD barrier fires even with explicit cc_enabled (filesystem visibility)."""
        DeviceKernel.compile_and_load(_dummy_kernel, cc_enabled=True)

        mock_dist.barrier.assert_called_once()

    def test_spmd_rank1_receives_broadcast(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist_rank1
    ):
        """Non-rank-0 SPMD worker receives via broadcast, does not compile."""
        DeviceKernel.compile_and_load(_dummy_kernel)

        # Rank 1 should NOT call _trace_and_compile (receives via broadcast)
        mock_trace_and_compile.assert_not_called()
        mock_dist_rank1.broadcast_object_list.assert_called_once()


class TestMPMDMode:
    """Tests for is_spmd=False (MPMD) mode."""

    def test_mpmd_skips_broadcast(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """MPMD mode: every rank compiles, no broadcast."""
        DeviceKernel.compile_and_load(
            _dummy_kernel, is_spmd=False, cc_enabled=True, rank_id=0, world_size=2
        )

        mock_trace_and_compile.assert_called_once()
        mock_dist.broadcast_object_list.assert_not_called()

    def test_mpmd_skips_barrier(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """MPMD mode does not call barrier."""
        DeviceKernel.compile_and_load(
            _dummy_kernel, is_spmd=False, cc_enabled=True, rank_id=1, world_size=4
        )

        mock_dist.barrier.assert_not_called()

    def test_mpmd_passes_cc_to_load(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """MPMD explicit CC params are forwarded to load_from_neff."""
        DeviceKernel.compile_and_load(
            _dummy_kernel, is_spmd=False, cc_enabled=True, rank_id=3, world_size=8
        )

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff",
            name="_dummy_kernel",
            cc_enabled=True,
            rank_id=3,
            world_size=8,
        )

    def test_mpmd_namespaces_build_dir_by_rank(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """MPMD namespaces the build dir by explicit rank."""
        DeviceKernel.compile_and_load(
            _dummy_kernel, is_spmd=False, cc_enabled=True, rank_id=1, world_size=2
        )

        call_kwargs = mock_trace_and_compile.call_args
        build_dir = call_kwargs.kwargs.get("build_dir") or call_kwargs[1].get(
            "build_dir"
        )
        assert build_dir is not None
        assert build_dir.endswith("/rank_1")

    def test_mpmd_auto_namespaces_build_dir_from_dist(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """MPMD with no explicit rank_id auto-detects rank for build dir namespace."""
        DeviceKernel.compile_and_load(_dummy_kernel, is_spmd=False)

        call_kwargs = mock_trace_and_compile.call_args
        build_dir = call_kwargs.kwargs.get("build_dir") or call_kwargs[1].get(
            "build_dir"
        )
        assert build_dir is not None
        assert build_dir.endswith("/rank_0")

    def test_mpmd_auto_detects_cc_from_dist(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """MPMD with cc_enabled=None auto-detects from dist."""
        DeviceKernel.compile_and_load(_dummy_kernel, is_spmd=False)

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff",
            name="_dummy_kernel",
            cc_enabled=True,
            rank_id=0,
            world_size=2,
        )

    def test_mpmd_cc_false_disables_cc(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """MPMD with cc_enabled=False disables CC."""
        DeviceKernel.compile_and_load(_dummy_kernel, is_spmd=False, cc_enabled=False)

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff", name="_dummy_kernel"
        )


class TestExplicitCCOverride:
    """Tests for cc_enabled overriding auto-detection."""

    def test_cc_false_disables_cc_in_spmd(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """cc_enabled=False disables CC even with torch.distributed active."""
        DeviceKernel.compile_and_load(_dummy_kernel, cc_enabled=False)

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff", name="_dummy_kernel"
        )

    def test_explicit_rank_overrides_dist(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """Explicit rank_id/world_size override dist values."""
        DeviceKernel.compile_and_load(
            _dummy_kernel, cc_enabled=True, rank_id=5, world_size=16
        )

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff",
            name="_dummy_kernel",
            cc_enabled=True,
            rank_id=5,
            world_size=16,
        )


class TestValidation:
    """Tests for parameter validation."""

    def test_cc_enabled_without_rank_raises(
        self, mock_trace_and_compile, mock_load_from_neff
    ):
        """cc_enabled=True without rank_id/world_size and no dist raises ValueError."""
        with patch("nkipy.runtime.device_kernel._is_distributed", return_value=False):
            with pytest.raises(ValueError, match="rank_id and world_size are required"):
                DeviceKernel.compile_and_load(_dummy_kernel, cc_enabled=True)

    def test_cc_enabled_without_world_size_raises(
        self, mock_trace_and_compile, mock_load_from_neff
    ):
        """cc_enabled=True with rank_id but no world_size and no dist raises."""
        with patch("nkipy.runtime.device_kernel._is_distributed", return_value=False):
            with pytest.raises(ValueError, match="rank_id and world_size are required"):
                DeviceKernel.compile_and_load(_dummy_kernel, cc_enabled=True, rank_id=0)


class TestNonDistributed:
    """Tests for single-worker (non-distributed) mode."""

    def test_no_dist_no_cc(self, mock_trace_and_compile, mock_load_from_neff):
        """Without distributed, loads without CC by default."""
        with patch("nkipy.runtime.device_kernel._is_distributed", return_value=False):
            DeviceKernel.compile_and_load(_dummy_kernel)

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff", name="_dummy_kernel"
        )

    def test_no_dist_explicit_cc(self, mock_trace_and_compile, mock_load_from_neff):
        """Without torch.distributed, explicit CC params still work."""
        with patch("nkipy.runtime.device_kernel._is_distributed", return_value=False):
            DeviceKernel.compile_and_load(
                _dummy_kernel, cc_enabled=True, rank_id=0, world_size=2
            )

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff",
            name="_dummy_kernel",
            cc_enabled=True,
            rank_id=0,
            world_size=2,
        )

    def test_no_dist_mpmd(self, mock_trace_and_compile, mock_load_from_neff):
        """MPMD without torch.distributed works with explicit CC."""
        with patch("nkipy.runtime.device_kernel._is_distributed", return_value=False):
            DeviceKernel.compile_and_load(
                _dummy_kernel,
                is_spmd=False,
                cc_enabled=True,
                rank_id=1,
                world_size=4,
            )

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff",
            name="_dummy_kernel",
            cc_enabled=True,
            rank_id=1,
            world_size=4,
        )


class TestCacheWithCC:
    """Tests for in-memory cache interaction with CC params."""

    def test_cache_hit_returns_cached(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """Cache hit returns the cached kernel without reloading."""
        sentinel = MagicMock()
        _LOADED_KERNELS["kernel_abc123"] = sentinel

        result = DeviceKernel.compile_and_load(_dummy_kernel)

        assert result is sentinel
        mock_load_from_neff.assert_not_called()

    def test_cache_miss_stores_result(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """Cache miss stores the loaded kernel."""
        result = DeviceKernel.compile_and_load(_dummy_kernel)

        assert "kernel_abc123" in _LOADED_KERNELS
        assert _LOADED_KERNELS["kernel_abc123"] is result

    def test_no_cache_when_disabled(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """use_cached_if_exists=False skips caching."""
        DeviceKernel.compile_and_load(_dummy_kernel, use_cached_if_exists=False)

        assert "kernel_abc123" not in _LOADED_KERNELS
