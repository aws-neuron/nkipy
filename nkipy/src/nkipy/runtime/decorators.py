# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
from typing import Optional

from nkipy.core import compile
from nkipy.core.compile import CompilerConfig, trace
from nkipy.runtime.execute import baremetal_run_traced_kernel, simulate_traced_kernel


def simulate_jit(kernel_func):
    """Decorator to execute a kernel using simulation.

    This decorator wraps a kernel function to automatically trace and simulate it.
    Useful for debugging and testing without hardware.

    Args:
        kernel_func: The kernel function to decorate

    Returns:
        A wrapped function that executes via simulation

    Example:
        @simulate
        def my_kernel(A, B):
            return A @ B

        result = my_kernel(input_a, input_b)
    """

    @functools.wraps(kernel_func)
    def wrapper(*args, **kwargs):
        traced_kernel = trace(
            kernel_func,
            debug_kernel=True,
            experimental_flags="enable-mutable-parameter",
            trace_time_unroll_builtin_range=True,
            show_compiler_tb=True,
        )
        traced_kernel.specialize(*args, **kwargs)
        return simulate_traced_kernel(traced_kernel, *args, **kwargs)

    return wrapper


def baremetal_jit(
    kernel_func=None,
    *,
    compiler_config: Optional[CompilerConfig] = None,
    additional_compiler_args="",
    target=compile.CompilationTarget.DEFAULT,
):
    """Decorator for JIT (Just-In-Time) compilation and execution on Neuron hardware.

    This decorator compiles the kernel on first invocation with a given signature.
    The compiled kernel is cached using an auto-generated name based on the function
    and argument signature, enabling persistent caching across Python sessions.

    Args:
        kernel_func: The kernel function to decorate (when used without parentheses)
        compiler_config: Structured compiler configuration (recommended).
            Use CompilerConfig.for_nkipy() or CompilerConfig.for_nki() for presets.
        additional_compiler_args: Additional arguments to pass to the compiler (legacy).
            If both compiler_config and additional_compiler_args are provided,
            additional_compiler_args will be appended to the config's args.
        target: Compilation target (default: CompilationTarget.DEFAULT)

    Returns:
        A wrapped function that JIT compiles and executes on Neuron hardware

    Example:
        @baremetal_jit
        def my_kernel(A, B):
            return A @ B

        # Compiles on first call with this signature
        result = my_kernel(input_a, input_b)

        # With structured config (recommended):
        @baremetal_jit(compiler_config=CompilerConfig.for_nkipy(model_type="transformer"))
        def my_kernel(A, B):
            return A @ B

        # Legacy string args (still supported):
        @baremetal_jit(additional_compiler_args="--model-type transformer")
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
                compiler_config=compiler_config,
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
