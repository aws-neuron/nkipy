# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase

# FIXME: the current scatter is implemented using np.put_along_axis


class TestAtenScatterValue(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,value",
        [
            # FIXME
            # ((8, 1024), 1, -1.0),
            ((8, 1024), 1, 1.0),
        ],
    )
    def test_scatter_value_basic(self, shape, dim, value):
        """Test basic cases of scatter.value operation."""

        def test_func(x, index):
            ret = torch.ops.aten.scatter.value(x, dim, index, value)
            ret = ret.clone()
            return ret

        src = torch.ones(size=shape, dtype=torch.float32)  # full_like tensor
        index = torch.randint(0, shape[dim], size=(8, 1), dtype=torch.int64)

        self.run_test_on_host(test_func, (src, index))
        self.run_test_on_device(test_func, (src, index))
