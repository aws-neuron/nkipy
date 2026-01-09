# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from nkipy.core.specs import CommonTypes, KernelSpec, ShapeSpec, TensorInputSpec


def indexed_add_nkipy(input_tensor):
    a = input_tensor[2:4, :]
    b = input_tensor[0:2]
    out = np.add(a, b)

    return out


INPUT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(
        dims=[(4, None), None],  # at least 4 on the first dim
        default=(4, 4),
    ),
    dtype_spec=CommonTypes.FLOATS,
    description="Input tensor for indexed addition",
)

# Define the kernel spec
kernel_specs = [
    KernelSpec(
        function=indexed_add_nkipy,
        inputs=[INPUT_SPEC],
        description="Add slices [2:4,:] and [0:2,:] of input tensor",
        is_pure_numpy=True,
    )
]
