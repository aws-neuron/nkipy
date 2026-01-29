# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenAddmm(NKIPyTestBase):
    @pytest.mark.parametrize(
        "batch,in_features,out_features,dtype",
        [
            (16, 32, 64, torch.float32),
            (8, 16, 32, torch.float32),
            (4, 8, 16, torch.float16),
            (2, 4, 8, torch.bfloat16),
            (1, 128, 256, torch.float32),
        ],
    )
    def test_addmm_shapes_dtypes(self, batch, in_features, out_features, dtype):
        """Test aten.addmm.default with different shapes and dtypes."""

        def test_func(bias, input1, input2):
            return torch.ops.aten.addmm.default(bias, input1, input2)

        bias = torch.randn(out_features, dtype=dtype).normal_(mean=0.0, std=0.02)
        input1 = torch.randn(batch, in_features, dtype=dtype).normal_(
            mean=0.0, std=0.02
        )
        input2 = torch.randn(in_features, out_features, dtype=dtype).normal_(
            mean=0.0, std=0.02
        )

        # FIXME BFloat16 numpy precision issue when running on host
        if dtype != torch.bfloat16:
            self.run_test_on_host(test_func, (bias, input1, input2))
        self.run_test_on_device(test_func, (bias, input1, input2))

    def test_addmm_alpha_beta(self):
        """Test addmm.default with different alpha and beta values."""

        def test_func(bias, input1, input2):
            return torch.ops.aten.addmm.default(
                bias, input1, input2, beta=0.5, alpha=2.0
            )

        bias = torch.randn(32, dtype=torch.float32).normal_(mean=0.0, std=0.02)
        input1 = torch.randn(16, 64, dtype=torch.float32).normal_(mean=0.0, std=0.02)
        input2 = torch.randn(64, 32, dtype=torch.float32).normal_(mean=0.0, std=0.02)

        self.run_test_on_host(test_func, (bias, input1, input2))
        self.run_test_on_device(test_func, (bias, input1, input2))
