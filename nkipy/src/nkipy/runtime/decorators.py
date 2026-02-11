# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import functools

from nkipy.core import compile
from nkipy.core.compile import trace
from nkipy.runtime.execute import baremetal_run_traced_kernel


def baremetal_jit(
    kernel_func=None,
    *,
    additional_compiler_args="",
    target=compile.CompilationTarget.DEFAULT,
):
    """Decorator for JIT (Just-In-Time) compilation and execution on Neuron hardware.

    This decorator compiles the kernel on first invocation with a given signature.
    The compiled kernel is cached using an auto-generated name based on the function
    and argument signature, enabling persistent caching across Python sessions.

    Args:
        kernel_func: The kernel function to decorate (when used without parentheses)
        additional_compiler_args: Additional arguments to pass to the compiler
        target: Compilation target (default: CompilationTarget.DEFAULT)

    Returns:
        A wrapped function that JIT compiles and executes on Neuron hardware

    Example:
        @baremetal_jit
        def my_kernel(A, B):
            return A @ B

        # Compiles on first call with this signature
        result = my_kernel(input_a, input_b)

        # Or with compiler args:
        @baremetal_jit(additional_compiler_args="--lnc 1")
        def my_kernel(A, B):
            return A @ B
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Trace the kernel
            traced_kernel = trace(func)
            # Use baremetal_run_traced_kernel for execution
            return baremetal_run_traced_kernel(
                traced_kernel,
                *args,
                additional_compiler_args=additional_compiler_args,
                target=target,
                **kwargs,
            )

        return wrapper

    # Support both @baremetal_jit and @baremetal_jit(...)
    if kernel_func is None:
        return decorator
    else:
        return decorator(kernel_func)
