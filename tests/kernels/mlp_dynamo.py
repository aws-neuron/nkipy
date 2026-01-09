# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# MLP generated from a torch dynamo graph

import numpy as np
from nkipy.core.specs import CommonTypes, KernelSpec, ShapeSpec, TensorInputSpec


def numpy_kernel_func(arg0_1, arg1_1, arg2_1, arg3_1):
    permute = np.transpose(arg0_1, [1, 0])
    view = np.reshape(arg1_1, [1, 2048])
    mm = np.matmul(view, permute)
    view_1 = np.reshape(mm, [1, 1, 8192])
    sigmoid = 1 / (1 + np.exp(-view_1))
    mul = np.multiply(view_1, sigmoid)
    permute_1 = np.transpose(arg2_1, [1, 0])
    view_2 = np.reshape(arg1_1, [1, 2048])
    mm_1 = np.matmul(view_2, permute_1)
    view_3 = np.reshape(mm_1, [1, 1, 8192])
    mul_1 = np.multiply(mul, view_3)
    permute_2 = np.transpose(arg3_1, [1, 0])
    view_4 = np.reshape(mul_1, [1, 8192])
    mm_2 = np.matmul(view_4, permute_2)
    view_5 = np.reshape(mm_2, [1, 1, 2048])
    return view_5


arg0_1 = np.random.uniform(0, 1, (8192, 2048)).astype(np.float32)
arg1_1 = np.random.uniform(0, 1, (2048)).astype(np.float32)
arg2_1 = np.random.uniform(0, 1, (8192, 2048)).astype(np.float32)
arg3_1 = np.random.uniform(0, 1, (2048, 8192)).astype(np.float32)

# Define TensorInputSpecs for each input
WEIGHT1_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[None, None], default=(8192, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="First weight matrix",
    default_value=arg0_1,
)

INPUT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[None], default=(2048,)),
    dtype_spec=CommonTypes.FLOATS,
    description="Input vector",
    default_value=arg1_1,
)

WEIGHT2_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[None, None], default=(8192, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="Second weight matrix",
    default_value=arg2_1,
)

WEIGHT3_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[None, None], default=(2048, 8192)),
    dtype_spec=CommonTypes.FLOATS,
    description="Third weight matrix",
    default_value=arg3_1,
)

# Define the kernel spec
kernel_specs = [
    KernelSpec(
        function=numpy_kernel_func,
        inputs=[WEIGHT1_SPEC, INPUT_SPEC, WEIGHT2_SPEC, WEIGHT3_SPEC],
        description="MLP with gated activation",
        is_pure_numpy=True,
    )
]
