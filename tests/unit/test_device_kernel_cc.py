# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for explicit CC parameters in DeviceKernel.compile_and_load."""

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
    """Mock torch.distributed as initialized with world_size=2."""
    with patch("nkipy.runtime.device_kernel._is_distributed", return_value=True), patch(
        "nkipy.runtime.device_kernel.dist", create=True
    ) as mock_d:
        mock_d.get_rank.return_value = 0
        mock_d.get_world_size.return_value = 2
        yield mock_d


class TestExplicitCC:
    """Tests for explicit cc_enabled / rank_id / world_size parameters."""

    def test_explicit_cc_skips_broadcast(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """When cc_enabled is explicit, every rank compiles (no broadcast)."""
        DeviceKernel.compile_and_load(
            _dummy_kernel, cc_enabled=True, rank_id=0, world_size=2
        )

        # _trace_and_compile called directly (not behind rank check)
        mock_trace_and_compile.assert_called_once()
        # No broadcast_object_list calls
        mock_dist.broadcast_object_list.assert_not_called()

    def test_explicit_cc_skips_barrier(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """When cc_enabled is explicit, barrier is not called."""
        DeviceKernel.compile_and_load(
            _dummy_kernel, cc_enabled=True, rank_id=1, world_size=4
        )

        mock_dist.barrier.assert_not_called()

    def test_explicit_cc_passes_to_load(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """Explicit CC params are forwarded to load_from_neff."""
        DeviceKernel.compile_and_load(
            _dummy_kernel, cc_enabled=True, rank_id=3, world_size=8
        )

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff",
            name="_dummy_kernel",
            cc_enabled=True,
            rank_id=3,
            world_size=8,
        )

    def test_explicit_cc_namespaces_build_dir_by_rank(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """Explicit CC namespaces the build dir by rank to avoid collisions."""
        DeviceKernel.compile_and_load(
            _dummy_kernel, cc_enabled=True, rank_id=1, world_size=2
        )

        # Check that _trace_and_compile received a rank-namespaced build_dir
        call_kwargs = mock_trace_and_compile.call_args
        build_dir = call_kwargs.kwargs.get("build_dir") or call_kwargs[1].get(
            "build_dir"
        )
        assert build_dir is not None
        assert build_dir.endswith("/rank_1")

    def test_explicit_cc_false_loads_without_cc(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """cc_enabled=False disables CC even with torch.distributed active."""
        DeviceKernel.compile_and_load(_dummy_kernel, cc_enabled=False)

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff", name="_dummy_kernel"
        )
        mock_dist.barrier.assert_not_called()


class TestAutoDetectedCC:
    """Tests for auto-detected CC via torch.distributed (cc_enabled=None)."""

    def test_auto_cc_uses_broadcast(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """Default (cc_enabled=None) with distributed uses rank-0 broadcast."""
        DeviceKernel.compile_and_load(_dummy_kernel)

        # Rank 0 compiles and broadcasts
        mock_trace_and_compile.assert_called_once()
        mock_dist.broadcast_object_list.assert_called_once()

    def test_auto_cc_uses_barrier(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """Default distributed mode calls barrier before load."""
        DeviceKernel.compile_and_load(_dummy_kernel)

        mock_dist.barrier.assert_called_once()

    def test_auto_cc_resolves_rank_and_world_size(
        self, mock_trace_and_compile, mock_load_from_neff, mock_dist
    ):
        """Auto-detected CC resolves rank_id and world_size from dist."""
        DeviceKernel.compile_and_load(_dummy_kernel)

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff",
            name="_dummy_kernel",
            cc_enabled=True,
            rank_id=0,
            world_size=2,
        )


class TestNonDistributed:
    """Tests for single-worker (non-distributed) mode."""

    def test_no_dist_no_cc(self, mock_trace_and_compile, mock_load_from_neff):
        """Without distributed, loads without CC by default."""
        with patch(
            "nkipy.runtime.device_kernel._is_distributed", return_value=False
        ):
            DeviceKernel.compile_and_load(_dummy_kernel)

        mock_load_from_neff.assert_called_once_with(
            "/tmp/test/kernel.neff", name="_dummy_kernel"
        )

    def test_no_dist_explicit_cc(self, mock_trace_and_compile, mock_load_from_neff):
        """Without torch.distributed, explicit CC params still work."""
        with patch(
            "nkipy.runtime.device_kernel._is_distributed", return_value=False
        ):
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
