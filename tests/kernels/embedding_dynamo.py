# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from nkipy.core.specs import CommonTypes, KernelSpec, ShapeSpec, TensorInputSpec


def nkipy_kernel_func(arg0_1, arg1_1):
    ge = np.greater_equal(arg0_1, 0)
    lt = np.less(arg0_1, 16000)
    bitwise_and = np.bitwise_and(ge, lt)
    ge_1 = np.greater_equal(arg0_1, 32000)
    lt_1 = np.less(arg0_1, 32000)
    bitwise_and_1 = np.bitwise_and(ge_1, lt_1)
    mul = np.multiply(bitwise_and, 0, dtype=np.int32)
    mul_1 = np.multiply(bitwise_and_1, 16000, dtype=np.int32)
    add = np.add(mul, mul_1)
    bitwise_or = np.bitwise_or(bitwise_and, bitwise_and_1)
    sub = np.subtract(arg0_1, add)
    mul_2 = np.multiply(bitwise_or, sub)

    # N.B.: Bitwise not is actually not the right one for aten's "bitwise_not"
    # bitwise_not = np.bitwise_not(bitwise_or)
    bitwise_not = np.logical_not(bitwise_or)
    embedding = np.take(arg1_1, mul_2, axis=0)
    unsqueeze = np.expand_dims(bitwise_not, -1)

    # scalar_tensor = np.array([0.0], dtype=np.float32)
    scalar_tensor = np.float32(0.0)
    where = np.where(unsqueeze, scalar_tensor, embedding)
    return where


arg0_1_numpy = np.random.uniform(0, 32000, (1, 128)).astype(np.int32)
arg1_1_numpy = np.random.uniform(0, 1, (16000, 2048)).astype(np.float32)

# Define input specifications
ARG0_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(
        dims=[
            None,
            None,
        ],  # Variable dimensions based on the example input shape (1, 128)
        default=(1, 128),
    ),
    dtype_spec=CommonTypes.INTS,
    default_value=arg0_1_numpy,
    description="First input tensor for embedding lookup",
)

ARG1_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(
        dims=[
            None,
            None,
        ],  # Variable dimensions based on the example input shape (16000, 2048)
        default=(16000, 2048),
    ),
    dtype_spec=CommonTypes.FLOATS,
    default_value=arg1_1_numpy,
    description="Second input tensor containing embedding weights",
)

# Define the kernel spec
kernel_specs = [
    KernelSpec(
        function=nkipy_kernel_func,
        inputs=[ARG0_SPEC, ARG1_SPEC],
        description="Performs embedding lookup with boundary checking and masking",
        is_pure_numpy=True,
    )
]
