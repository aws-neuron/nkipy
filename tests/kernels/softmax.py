# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from nkipy.core.specs import CommonTypes, KernelSpec, ShapeSpec, TensorInputSpec


def softmax_kernel(x):
    exp_x = np.exp((x - np.max(x, axis=-1, keepdims=True)))
    sum_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_x


SOFTMAX_2D = ShapeSpec(dims=[None, (2, 65536)], default=(32, 128))

kernel_specs = [
    KernelSpec(
        function=softmax_kernel,
        inputs=[
            TensorInputSpec(
                shape_spec=SOFTMAX_2D,
                dtype_spec=CommonTypes.FLOATS,
                description="Input tensor of shape (B, N)",
            ),
        ],
        is_pure_numpy=True,
        description="Applies softmax normalization with temperature",
    )
]
