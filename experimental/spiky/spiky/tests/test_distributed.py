# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed tests for spiky.

Run with torchrun:
    cd nkipy/
    uv run torchrun --nproc_per_node=2 -m pytest \
        experimental/spiky/spiky/tests/test_distributed.py -v

Uses gloo backend for process coordination. CC ops (all_reduce, etc.) are
traced into the FX graph and compiled to NEFF.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

# Hardware detection via spiky (returns 0 when C++ extension unavailable)
NEURON_AVAILABLE = False
try:
    import spiky

    NEURON_AVAILABLE = spiky.device_count() > 0
except Exception:
    pass

DISTRIBUTED_ENV = (
    os.environ.get("RANK") is not None or os.environ.get("LOCAL_RANK") is not None
)

pytestmark = pytest.mark.skipif(
    not NEURON_AVAILABLE or not DISTRIBUTED_ENV,
    reason="Requires Neuron hardware and distributed environment (run with torchrun)",
)


def setup_module(module):
    """Initialize distributed backend and spiky for all tests."""
    from spiky.torch.backend import init_nkipy_backend, is_nkipy_backend_initialized

    if not dist.is_initialized():
        dist.init_process_group("gloo")
    if not is_nkipy_backend_initialized():
        init_nkipy_backend(nkipy_cache="./.nkipy_test_cache")


def teardown_module(module):
    """Cleanup distributed backend after all tests."""
    from spiky.torch.backend import is_nkipy_backend_initialized, reset_nkipy_backend

    if is_nkipy_backend_initialized():
        reset_nkipy_backend()
    if dist.is_initialized():
        dist.destroy_process_group()


class TestDistributedExecution:
    """Distributed execution tests with all_reduce."""

    def test_all_reduce_sum(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        @torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
        def reduce_fn(x):
            y = x * 2
            return dist.all_reduce(y, op=dist.ReduceOp.SUM)

        x = torch.full((4, 4), float(rank + 1), dtype=torch.float32, device="nkipy")
        with torch.no_grad():
            result = reduce_fn(x)

        # Each rank contributes (rank+1)*2, summed: 2 * sum(1..world_size)
        expected_val = float(world_size * (world_size + 1))
        result_cpu = result.cpu()
        assert torch.allclose(
            result_cpu,
            torch.full_like(result_cpu, expected_val),
            rtol=1e-4,
            atol=1e-4,
        ), f"Rank {rank}: expected {expected_val}, got {result_cpu[0, 0].item()}"

    def test_mlp_with_all_reduce(self):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 16, bias=False)
                self.act = nn.SiLU()
                self.linear2 = nn.Linear(16, 8, bias=False)

            def forward(self, x):
                h = self.act(self.linear1(x))
                h = self.linear2(h)
                return dist.all_reduce(h, op=dist.ReduceOp.SUM)

        model = MLP().to("nkipy")
        compiled = torch.compile(model, backend="nkipy", fullgraph=True, dynamic=False)
        x = torch.randn(2, 8, dtype=torch.float32, device="nkipy")

        with torch.no_grad():
            result = compiled(x)

        assert result.shape == (2, 8)
        result_cpu = result.cpu()
        assert not torch.isnan(result_cpu).any()
        assert not torch.isinf(result_cpu).any()

    def test_gated_mlp(self):
        class GatedMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(4, 8, bias=False)
                self.up_proj = nn.Linear(4, 8, bias=False)
                self.down_proj = nn.Linear(8, 4, bias=False)
                self.act = nn.SiLU()

            def forward(self, x):
                gate = self.act(self.gate_proj(x))
                up = self.up_proj(x)
                h = gate * up
                h = self.down_proj(h)
                return dist.all_reduce(h, op=dist.ReduceOp.SUM)

        model = GatedMLP().to("nkipy")
        compiled = torch.compile(model, backend="nkipy", fullgraph=True, dynamic=False)
        x = torch.randn(2, 4, dtype=torch.float32, device="nkipy")

        with torch.no_grad():
            result = compiled(x)

        assert result.shape == (2, 4)
        result_cpu = result.cpu()
        assert not torch.isnan(result_cpu).any()
        assert not torch.isinf(result_cpu).any()

    def test_add_and_reduce(self):
        @torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
        def compute(a, b):
            c = a + b
            d = c * 2
            return dist.all_reduce(d, op=dist.ReduceOp.SUM)

        a = torch.randn(4, 4, dtype=torch.float32, device="nkipy")
        b = torch.randn(4, 4, dtype=torch.float32, device="nkipy")

        with torch.no_grad():
            result = compute(a, b)

        assert result.shape == (4, 4)


class TestDistributedConfig:
    """Test that distributed config matches torch.distributed state."""

    def test_rank_matches(self):
        from spiky.torch.config import get_nkipy_backend_config

        config = get_nkipy_backend_config()
        assert config.rank == dist.get_rank()

    def test_world_size_matches(self):
        from spiky.torch.config import get_nkipy_backend_config

        config = get_nkipy_backend_config()
        assert config.world_size == dist.get_world_size()

    def test_per_rank_cache(self):
        from spiky.torch.config import get_nkipy_backend_config

        config = get_nkipy_backend_config()
        rank = dist.get_rank()
        assert f"rank_{rank}" in config.nkipy_cache


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
