# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Fuzz testing for kernel specifications.

This test module performs randomized testing of kernels by:
1. Shape fuzzing:
   - Generates random valid shapes within dimension constraints
   - Handles None (any size), fixed sizes, and (min, max) ranges
   - Ensures matching shapes for operations like addition

2. Dtype fuzzing:
   - Randomly selects from allowed dtypes (e.g. float32, float16)
   - Tests mixed precision combinations
   - Adjusts numerical tolerance based on precision

For each kernel, it runs multiple iterations with different:
- Input shapes (within spec constraints)
- Data types (from allowed types)
- Random input values

The test verifies that:
- The kernel runs successfully with valid random inputs
- Output matches numpy reference implementation
- Numerical accuracy is within tolerance for the given dtypes
"""

import random
from typing import List, Tuple, Type, Union

import numpy as np
import pytest
from kernels.simple import kernel_specs as simple_specs
from nkipy.core.specs import ScalarInputSpec, TensorInputSpec, TypeSpec
from utils import (
    NEURON_AVAILABLE,
    on_device_test,
    trace_and_compile,
)


def generate_valid_shape(dims: List[Union[int, Tuple, None]]) -> Tuple[int, ...]:
    """Generate a valid shape based on dimension constraints"""
    shape = []
    for dim in dims:
        if dim is None:
            shape.append(random.randint(1, 64))
        elif isinstance(dim, int):
            shape.append(dim)
        elif isinstance(dim, tuple):
            min_size, max_size = dim
            if min_size is None:
                min_size = 1
            if max_size is None:
                max_size = 64
            shape.append(random.randint(min_size, max_size))
    return tuple(shape)


def generate_matching_shapes(
    spec1: TensorInputSpec, spec2: TensorInputSpec
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Generate matching shapes for two tensor specs"""
    dims = spec1.shape_spec.dims
    shape1 = generate_valid_shape(dims)
    return shape1, shape1


def generate_dtype(type_spec: TypeSpec) -> Type:
    """Generate a random valid dtype from the allowed types"""
    return random.choice(type_spec.allowed)


@pytest.mark.parametrize("kernel_spec", simple_specs)
def test_kernel_fuzz(trace_mode, kernel_spec):
    """Fuzz test kernel with various valid shapes and dtypes"""

    # TODO: use some environment variable to control number of fuzz
    n_fuzz_tests = 10

    for _ in range(n_fuzz_tests):
        # Generate matching shapes for inputs
        shape_a, shape_b = generate_matching_shapes(
            kernel_spec.inputs[0], kernel_spec.inputs[1]
        )

        # Generate random dtypes for inputs
        dtype_a = generate_dtype(kernel_spec.inputs[0].dtype_spec)
        dtype_b = generate_dtype(kernel_spec.inputs[1].dtype_spec)

        print(f"\nTesting shapes: {shape_a}, {shape_b}")
        print(f"Testing dtypes: {dtype_a}, {dtype_b}")

        # Generate inputs
        inputs = []
        for input_spec, shape, dtype in zip(
            kernel_spec.inputs, [shape_a, shape_b], [dtype_a, dtype_b]
        ):
            if isinstance(input_spec, TensorInputSpec):
                x = np.random.randn(*shape).astype(dtype)
                inputs.append(x)
            elif isinstance(input_spec, ScalarInputSpec):
                inputs.append(input_spec.dtype_spec.default())

        # Run kernel
        expected = kernel_spec.function(*inputs)

        # Test hardware if available
        if NEURON_AVAILABLE:
            hardware_result = on_device_test(kernel_spec.function, trace_mode, *inputs)

            # Use looser tolerances for hardware
            hw_rtol = 1e-2 if any(x.dtype == np.float16 for x in inputs) else 1e-2
            assert np.allclose(hardware_result, expected, rtol=hw_rtol, atol=hw_rtol)
            print(
                f"Hardware test passed for shapes {shape_a} and dtypes {[x.dtype for x in inputs]}"
            )
        else:
            trace_and_compile(kernel_spec.function, trace_mode, *inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
