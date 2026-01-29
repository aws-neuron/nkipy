# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenNativeLayerNorm(NKIPyTestBase):
    @pytest.mark.parametrize(
        "batch_size,seq_length,hidden_size,dtype",
        [
            (16, 32, 768, torch.float32),
            (8, 64, 1024, torch.float32),
            # (2, 256, 256, torch.bfloat16), # Precision issue
            # (4, 32, 512, torch.float16),  # Precision issue
        ],
    )
    def test_layer_norm_shapes_dtypes(self, batch_size, seq_length, hidden_size, dtype):
        """Test layer normalization with different shapes and dtypes."""

        def test_func(input_tensor, weight, bias):
            normalized_shape = [hidden_size]
            # Get only the first output (normalized values) from the tuple
            return torch.ops.aten.native_layer_norm.default(
                input_tensor, normalized_shape, weight, bias, 1e-5
            )

        input_tensor = torch.randn(
            batch_size, seq_length, hidden_size, dtype=dtype
        ).normal_(mean=0.0, std=0.02)
        weight = torch.randn(hidden_size, dtype=dtype).normal_(mean=0.0, std=0.02)
        bias = torch.randn(hidden_size, dtype=dtype).normal_(mean=0.0, std=0.02)

        if dtype != torch.bfloat16:
            self.run_test_on_host(
                test_func, (input_tensor, weight, bias), rtol=1e-2, atol=1e-2
            )
        self.run_test_on_device(
            test_func, (input_tensor, weight, bias), rtol=1e-2, atol=1e-2
        )

    def test_layer_norm_no_affine(self):
        """Test layer normalization without affine parameters."""

        def test_func(input_tensor):
            normalized_shape = [768]
            return torch.ops.aten.native_layer_norm.default(
                input_tensor, normalized_shape, None, None, 1e-5
            )

        input_tensor = torch.randn(16, 32, 768, dtype=torch.float32).normal_(
            mean=0.0, std=0.02
        )

        self.run_test_on_host(test_func, (input_tensor,))
        self.run_test_on_device(test_func, (input_tensor,))

    def test_layer_norm_eps(self):
        """Test layer normalization with different epsilon values."""

        def test_func(input_tensor, weight, bias):
            normalized_shape = [512]
            return torch.ops.aten.native_layer_norm.default(
                input_tensor, normalized_shape, weight, bias, 1e-12
            )

        input_tensor = torch.randn(8, 64, 512, dtype=torch.float32)
        weight = torch.randn(512, dtype=torch.float32)
        bias = torch.randn(512, dtype=torch.float32)

        self.run_test_on_host(test_func, (input_tensor, weight, bias))
        self.run_test_on_device(test_func, (input_tensor, weight, bias))
