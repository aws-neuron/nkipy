# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestIndexCopy(NKIPyTestBase):
    @pytest.mark.parametrize(
        "batch_size,num_heads,seq_len,head_dim,new_tokens,current_position",
        [
            (2, 12, 100, 64, 5, 10),  # Standard case
            (1, 8, 50, 32, 1, 0),  # Single batch, single token at start
            (4, 16, 200, 128, 10, 50),  # Larger dimensions
        ],
    )
    def test_index_copy(
        self, batch_size, num_heads, seq_len, head_dim, new_tokens, current_position
    ):
        """Test tensor.index_copy_ operation with different dimensions."""

        def test_func(key_states, k_out, cache_position):
            k_out.index_copy_(2, cache_position, key_states)
            # FIXME If we do not do calculation and directly return k_out
            # it triggered a neuron compiler bug: out different in simulate_kernel
            h = key_states * 2
            return h

        cache_position = torch.arange(current_position, current_position + new_tokens)

        k_out = torch.zeros(batch_size, num_heads, seq_len, head_dim)
        key_states = torch.randn(batch_size, num_heads, new_tokens, head_dim)

        self.run_test_on_host(test_func, (key_states, k_out, cache_position))
        self.run_test_on_device(test_func, (key_states, k_out, cache_position))

    def test_index_copy_different_dims(self):
        """Test index_copy_ along different dimensions."""

        def test_func(source, target, indices):
            # Copy along dimension 0
            target.index_copy_(0, indices, source)
            h = source * 2
            return h

        source = torch.randn(3, 8, 10)  # Source has 3 items in dim 0
        target = torch.zeros(10, 8, 10)  # Target has 10 items in dim 0
        indices = torch.tensor([1, 4, 7])  # Copy to these 3 positions

        self.run_test_on_host(test_func, (source, target, indices))
        self.run_test_on_device(test_func, (source, target, indices))

    def test_index_copy_dim1(self):
        """Test index_copy_ along dimension 1."""

        def test_func(source, target, indices):
            # Copy along dimension 1
            target.index_copy_(1, indices, source)
            h = source * 2
            return h

        target = torch.zeros(5, 10, 7)
        source = torch.randn(5, 3, 7)  # 3 items to copy into dim 1
        indices = torch.tensor([2, 5, 9])  # Copy to these 3 positions in dim 1

        self.run_test_on_host(test_func, (source, target, indices))
        self.run_test_on_device(test_func, (source, target, indices))
