# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import glob
import importlib
import os
from typing import Dict, List, Tuple, Type

import numpy as np
import pytest
from nkipy.core.specs import KernelSpec, ScalarInputSpec, TensorInputSpec
from utils import (
    NEURON_AVAILABLE,
    baremetal_assert_allclose,
    on_device_test,
    trace_and_compile,
    trace_mode,  # noqa: F401 - pytest fixture
)


def collect_kernel_specs() -> Dict[str, List[KernelSpec]]:
    """Collect all kernel_specs from Python files under tests/kernels/"""
    specs_by_file = {}
    kernels_dir = os.path.join(os.path.dirname(__file__), "..", "kernels")

    # Get all .py files except __init__.py
    py_files = glob.glob(os.path.join(kernels_dir, "*.py"))
    py_files = [f for f in py_files if not f.endswith("__init__.py")]

    for py_file in py_files:
        file_name = os.path.basename(py_file)
        try:
            # Import the module
            module = importlib.import_module(
                f"kernels.{os.path.splitext(file_name)[0]}"
            )

            # Look for kernel_specs
            if hasattr(module, "kernel_specs"):
                specs = getattr(module, "kernel_specs")
                if isinstance(specs, list):
                    for i, spec in enumerate(specs):
                        specs_by_file[f"{file_name}:{i}"] = spec
                else:
                    specs_by_file[file_name] = specs
                print(f"Found kernel specs in {file_name}")
        except Exception as e:
            print(f"Warning: Could not import kernel specs from {file_name}: {e}")

    return specs_by_file


# Collect all kernel specs
SPECS_BY_FILE = collect_kernel_specs()


@pytest.mark.parametrize("file_name,kernel_spec", SPECS_BY_FILE.items())
def test_kernel_default(trace_mode, file_name, kernel_spec):
    """Test kernel with default shapes and types from spec"""

    if not kernel_spec.is_pure_numpy:
        # not pure numpy kernels are supported yet
        return

    inputs = []

    for input_spec in kernel_spec.inputs:
        # If user defines default test value, use it
        if input_spec.default_value is not None:
            inputs.append(input_spec.default_value)
            continue

        if isinstance(input_spec, TensorInputSpec):
            shape_default: Tuple[int, ...] = input_spec.shape_spec.default
            type_default: Type = input_spec.dtype_spec.default

            x = np.random.randn(*shape_default).astype(type_default)
            inputs.append(x)
        elif isinstance(input_spec, ScalarInputSpec):
            inputs.append(input_spec.dtype_spec.default)
        else:
            raise ValueError("Unknown input spec type")

    expected = kernel_spec.function(*inputs)

    if NEURON_AVAILABLE:
        hardware_result = on_device_test(kernel_spec.function, trace_mode, *inputs)

        if isinstance(hardware_result, tuple):
            for r, e in zip(hardware_result, expected):
                baremetal_assert_allclose(r, e)
        else:
            baremetal_assert_allclose(hardware_result, expected)
    else:
        trace_and_compile(kernel_spec.function, trace_mode, *inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
