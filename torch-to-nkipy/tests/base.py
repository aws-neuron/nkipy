# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Sequence

import pytest
import torch
import torch.fx as fx
from torch._decomp import core_aten_decompositions
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.utils import InputType

from torch_to_nkipy.nkipy_builder.nkipy_kernel import NKIPyKernel


def transform_arg(arg, operation=None):
    """
    Generic function to transform tensors within nested data structures.

    Args:
        arg: The argument to transform (tensor, list, tuple, dict, or other)
        operation: Function that takes a tensor and returns a transformed tensor
                  If None, the identity operation is used (just returns the tensor)

    Returns:
        Transformed argument with the same structure
    """
    if operation is None:
        operation = lambda x: x  # noqa: E731

    if isinstance(arg, torch.Tensor):
        return operation(arg)
    elif isinstance(arg, list):
        return [transform_arg(item, operation) for item in arg]
    elif isinstance(arg, tuple):
        return tuple(transform_arg(item, operation) for item in arg)
    elif isinstance(arg, dict):
        return {k: transform_arg(v, operation) for k, v in arg.items()}
    else:
        return arg


def clone_arg(arg):
    """Clone tensors within nested data structures."""
    return transform_arg(arg, lambda x: x.clone())


def move_arg_to_device(arg, device):
    """Move tensors within nested data structures to specified device."""
    return transform_arg(arg, lambda x: x.to(device=device))


def align_outputs(reference_output, compiled_output, device):
    if not isinstance(reference_output, tuple):
        reference_output = (reference_output,)
    if not isinstance(compiled_output, tuple):
        compiled_output = (compiled_output,)
    reference_output = list(reference_output)
    compiled_output = list(compiled_output)

    if device:
        compiled_output = [out.cpu() for out in compiled_output]

    compiled_output_aligned = []
    for i, out in enumerate(compiled_output):
        if not hasattr(reference_output[i], "dtype"):
            compiled_output_aligned.append(out)
            continue

        if reference_output[i].dtype == torch.bool:
            compiled_output_aligned.append(out.bool())
        elif reference_output[i].dtype == torch.int64:
            compiled_output_aligned.append(out.to(torch.int64))
        elif reference_output[i].dtype == torch.bfloat16:
            compiled_output_aligned.append(out.to(torch.bfloat16))
        else:
            compiled_output_aligned.append(out)
    return reference_output, compiled_output_aligned


def _compile_with_kernel(gm, inputs, on_device=False):
    """Common compilation helper function"""
    neuron_kernel = NKIPyKernel(gm, inputs)
    # FIXME how to control on_device will be updated
    neuron_kernel.on_device = on_device
    return neuron_kernel


def test_on_host_compile(
    gm: fx.GraphModule, example_inputs: Sequence[InputType]
) -> Callable:
    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=lambda g, i: _compile_with_kernel(g, i, False),
        decompositions=core_aten_decompositions(),
        keep_inference_input_mutations=True,
    )


def test_on_device_compile(
    gm: fx.GraphModule, example_inputs: Sequence[InputType]
) -> Callable:
    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=lambda g, i: _compile_with_kernel(g, i, True),
        decompositions=core_aten_decompositions(),
        keep_inference_input_mutations=True,
    )


class NKIPyTestBase:
    """Base class for Neuronpy tests with host and device execution capabilities"""

    @pytest.fixture(autouse=True)
    def test_setup_teardown(self):
        """Setup and teardown for each test."""
        torch.manual_seed(0)

        yield

        torch.compiler.reset()
        # FIXME we don't unload the loaded models for each test

    def _run_test(
        self, func, args, backend, device=None, rtol=1e-4, atol=1e-4, cpu_ref_func=None
    ):
        """Generic test runner for both host and device tests"""
        func_name = getattr(func, "__name__", "unknown")
        reference_args = [clone_arg(arg) for arg in args]
        compiled_args = [clone_arg(arg) for arg in args]

        if device:
            compiled_args = [move_arg_to_device(arg, device) for arg in compiled_args]

        compiled_func = torch.compile(
            func, backend=backend, fullgraph=True, dynamic=False
        )

        with torch.no_grad():
            ref_func = cpu_ref_func if cpu_ref_func is not None else func
            reference_output = ref_func(*reference_args)
            compiled_output = compiled_func(*compiled_args)

        if device:
            compiled_args = [move_arg_to_device(arg, "cpu") for arg in compiled_args]

        reference_output, compiled_output = align_outputs(
            reference_output, compiled_output, device
        )

        # Check input/output consistency
        torch.testing.assert_close(
            compiled_args,
            reference_args,
            rtol=rtol,
            atol=atol,
            msg=f"Input mismatch in '{func_name}' with args: {args}",
        )
        torch.testing.assert_close(
            compiled_output,
            reference_output,
            rtol=rtol,
            atol=atol,
            msg=f"Output mismatch in '{func_name}' with args: {args}, "
            f"compiled_output: {compiled_output}, "
            f"reference_output {reference_output}.",
        )

        return compiled_output, reference_output, reference_args, compiled_args

    def run_test_on_host(self, func, args, rtol=1e-4, atol=1e-4):
        """Run test with compiled and reference function on host"""
        return self._run_test(func, args, test_on_host_compile, None, rtol, atol, None)

    def run_test_on_device(self, func, args, rtol=1e-2, atol=1e-2, cpu_ref_func=None):
        """Run test with compiled function on Neuronpy device"""
        return self._run_test(
            func, args, test_on_device_compile, "nkipy", rtol, atol, cpu_ref_func
        )
