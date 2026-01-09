# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simple tensor addition kernel example demonstrating the NKIPy kernel specification system.

This example shows how to:
1. Define a simple numpy kernel function
2. Specify input tensor requirements
3. Create a complete kernel specification

This kernel specification will be used by integration tests.

Input Specifications:
    INPUT_A_SPEC and INPUT_B_SPEC are identical, each requiring:
    - Shape: Any 2D tensor shape ([None, None])
    - Dtype: float32 or float16 (CommonTypes.FLOATS)
    - Default shape (32, 32) for testing

The kernel_specs list contains a single KernelSpec that:
    - Links the simple_add function
    - Defines input requirements
    - Marks this as a pure numpy function
    - Provides a description

"""

import numpy as np
from nkipy.core.specs import CommonTypes, KernelSpec, ShapeSpec, TensorInputSpec


def simple_add(a, b):
    """Simple addition of two tensors.

    Args:
        a: First input tensor, any 2D shape
        b: Second input tensor, must match shape of a

    Returns:
        Element-wise sum of a and b
    """
    return np.add(a, b)


# Define TensorInputSpecs
INPUT_A_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(
        dims=[None, None],  # allow any 2D shape
        default=(32, 32),  # reasonable size for testing
    ),
    dtype_spec=CommonTypes.FLOATS,  # supports float32 and float16
    description="First input tensor",
)

INPUT_B_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[None, None], default=(32, 32)),
    dtype_spec=CommonTypes.FLOATS,
    description="Second input tensor",
)

kernel_specs = [
    KernelSpec(
        function=simple_add,
        inputs=[INPUT_A_SPEC, INPUT_B_SPEC],
        description="Simple tensor addition",
        is_pure_numpy=True,  # indicates this uses only numpy operations
    )
]
