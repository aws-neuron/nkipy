# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed tests for torch-to-nkipy.

Tests distributed execution with torch.distributed and collective operations.
These tests should be run with torchrun, e.g.:
    torchrun --nproc_per_node=2 -m pytest tests/test_distributed.py -v
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from torch_to_nkipy.backend.nkipy_backend import init_nkipy_backend, reset_nkipy_backend


def setup_distributed():
    """Initialize distributed backend if not already done."""
    from torch_to_nkipy.backend.nkipy_backend import is_nkipy_backend_initialized

    if not dist.is_initialized():
        dist.init_process_group("nkipy")
    if not is_nkipy_backend_initialized():
        init_nkipy_backend()


def cleanup_distributed():
    """Cleanup distributed backend."""
    from torch_to_nkipy.backend.nkipy_backend import is_nkipy_backend_initialized

    if is_nkipy_backend_initialized():
        reset_nkipy_backend()
    if dist.is_initialized():
        dist.destroy_process_group()


class TestDistributedBasic:
    """Basic distributed tests with all_reduce."""

    @classmethod
    def setup_class(cls):
        """Setup distributed environment before tests."""
        setup_distributed()
        torch.manual_seed(0)

    @classmethod
    def teardown_class(cls):
        """Cleanup distributed environment after tests."""
        cleanup_distributed()

    def test_all_reduce_sum(self):
        """Test all_reduce with SUM operation combined with computation."""

        @torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
        def reduce_func(x):
            # all_reduce needs to be combined with computation
            y = x * 2
            z = dist.all_reduce(y, op=dist.ReduceOp.SUM)
            return z

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Each rank contributes its rank value
        x = torch.full((4, 4), float(rank + 1), dtype=torch.float32, device="nkipy")

        with torch.no_grad():
            result = reduce_func(x)

        # Each rank contributes (rank+1)*2, summed over all ranks
        # Sum = 2 * sum(1..world_size) = world_size*(world_size+1)
        expected_sum = float(world_size * (world_size + 1))
        result_cpu = result.cpu()

        assert torch.allclose(
            result_cpu,
            torch.full_like(result_cpu, expected_sum),
            rtol=1e-4,
            atol=1e-4,
        ), f"Rank {rank}: Expected {expected_sum}, got {result_cpu[0, 0].item()}"

    def test_mlp_with_all_reduce(self):
        """Test MLP model with all_reduce in forward pass."""

        @torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 16, bias=False)
                self.linear2 = nn.Linear(16, 8, bias=False)
                self.act_fn = nn.SiLU()

            def forward(self, x):
                h = self.linear1(x)
                h = self.act_fn(h)
                h = self.linear2(h)
                h = dist.all_reduce(h, op=dist.ReduceOp.SUM)
                return h

        model = MLP().to("nkipy")
        x = torch.randn(2, 8, dtype=torch.float32, device="nkipy")

        with torch.no_grad():
            result = model(x)

        assert result.shape == (2, 8)
        assert result.device.type == "nkipy"

    def test_distributed_tensor_operations(self):
        """Test basic tensor operations in distributed setting."""

        @torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
        def compute_and_reduce(a, b):
            c = a + b
            d = c * 2
            e = dist.all_reduce(d, op=dist.ReduceOp.SUM)
            return e

        a = torch.randn(4, 4, dtype=torch.float32, device="nkipy")
        b = torch.randn(4, 4, dtype=torch.float32, device="nkipy")

        with torch.no_grad():
            result = compute_and_reduce(a, b)

        assert result.shape == (4, 4)

    def test_multi_layer_model(self):
        """Test multi-layer model with distributed execution."""

        @torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
        class MultiLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(4, 8, bias=False)
                self.up_proj = nn.Linear(4, 8, bias=False)
                self.down_proj = nn.Linear(8, 4, bias=False)
                self.act_fn = nn.SiLU()

            def forward(self, x):
                h0 = self.gate_proj(x)
                h1 = self.up_proj(x)
                h2 = self.act_fn(h0) * h1
                h3 = self.down_proj(h2)
                h4 = dist.all_reduce(h3, op=dist.ReduceOp.SUM)
                return h4

        model = MultiLayerModel().to("nkipy")
        x = torch.randn(2, 4, dtype=torch.float32, device="nkipy")

        with torch.no_grad():
            result = model(x)

        assert result.shape == (2, 4)
        assert result.device.type == "nkipy"

        # Verify result is valid (not NaN or Inf)
        result_cpu = result.cpu()
        assert not torch.isnan(result_cpu).any()
        assert not torch.isinf(result_cpu).any()


class TestDistributedConfig:
    """Test distributed configuration handling."""

    @classmethod
    def setup_class(cls):
        """Setup distributed environment before tests."""
        setup_distributed()

    @classmethod
    def teardown_class(cls):
        """Cleanup distributed environment after tests."""
        cleanup_distributed()

    def test_rank_detection(self):
        """Test that rank is correctly detected from torch.distributed."""
        from torch_to_nkipy.backend.nkipy_backend_config import get_nkipy_backend_config

        config = get_nkipy_backend_config()
        expected_rank = dist.get_rank()

        assert config.rank == expected_rank

    def test_world_size_detection(self):
        """Test that world_size is correctly detected from torch.distributed."""
        from torch_to_nkipy.backend.nkipy_backend_config import get_nkipy_backend_config

        config = get_nkipy_backend_config()
        expected_world_size = dist.get_world_size()

        assert config.world_size == expected_world_size

    def test_rank_in_cache_path(self):
        """Test that each rank has its own cache directory."""
        from torch_to_nkipy.backend.nkipy_backend_config import get_nkipy_backend_config

        config = get_nkipy_backend_config()
        rank = dist.get_rank()

        assert f"rank_{rank}" in config.nkipy_cache


if __name__ == "__main__":
    # Allow running directly with torchrun
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
