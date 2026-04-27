# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness and full-pipeline tests for the kernelgen backend.

Two levels of verification:
1. LLVM JIT smoke test — trace via nkipy, run through MLIR passes, execute
   via LLVM JIT to verify numerical correctness without hardware.
2. NEFF compilation — trace via nkipy with knob() annotations, compile all
   the way to NEFF to catch pass-pipeline and mem_space enum issues.

Requires nkipy_kernelgen (pass pipeline, LLVM JIT infrastructure).
"""

import numpy as np
import pytest

try:
    from nkipy_kernelgen.llvm import LLVMModule

    HAS_KERNELGEN = True
except ImportError:
    HAS_KERNELGEN = False

from nkipy.core.trace import NKIPyKernel
from nkipy.core.knob import knob

pytestmark = pytest.mark.skipif(
    not HAS_KERNELGEN, reason="nkipy-kernelgen not installed"
)


def _trace_and_run_llvm(func, *np_args):
    """Trace via nkipy kernelgen, execute via LLVM JIT, return result."""
    kernel = NKIPyKernel.trace(func, backend="kernelgen")
    ir = kernel.specialize(*np_args)
    mod = LLVMModule(ir._mlir_text, ir._func_name)
    return mod(*np_args)


def _trace_and_compile_to_neff(func, *np_args):
    """Trace a kernelgen kernel and compile all the way to NEFF.

    Exercises the full nkipy.core.knob -> builder.annotate() -> MLIR pass
    pipeline -> NISA -> neuronx-cc -> NEFF path. Raises on any failure.
    """
    import shutil
    import tempfile

    from nkipy.core import compile as nkipy_compile

    kernel = NKIPyKernel.trace(func, backend="kernelgen")
    kernel.specialize(*np_args)

    artifacts_dir = tempfile.mkdtemp(prefix="kernelgen_neff_test_")
    try:
        nkipy_compile.compile_to_neff(
            kernel,
            artifacts_dir,
            additional_compiler_args=nkipy_compile.nkipy_compiler_args,
        )
    finally:
        shutil.rmtree(artifacts_dir, ignore_errors=True)


class TestNumericalLLVMJIT:
    """Smoke tests: trace through nkipy, verify numerics via LLVM JIT."""

    def test_matmul_add(self):
        def kernel(a, b, bias):
            return np.matmul(a, b) + bias

        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8, 4).astype(np.float32)
        bias = np.random.randn(4).astype(np.float32)
        result = _trace_and_run_llvm(kernel, a, b, bias)
        np.testing.assert_allclose(result, a @ b + bias, rtol=1e-4, atol=1e-4)

    def test_sigmoid(self):
        def kernel(x):
            return np.reciprocal(1.0 + np.exp(-x))

        x = np.random.randn(4, 8).astype(np.float32)
        result = _trace_and_run_llvm(kernel, x)
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


class TestNumericalFullPipeline:
    """Compile to NEFF end-to-end via nkipy trace+compile.

    Exercises the nkipy.core.knob -> builder.annotate() -> MLIR -> NISA ->
    NEFF path, catching issues like mem_space enum mismatches between the
    Python builder and the MLIR dialect definition.
    """

    def test_add_full_pipeline(self):
        def kernel(a, b):
            C = np.add(a, b)
            C = knob(C, mem_space="SharedHbm", tile_size=[128, 128])
            return C

        a = np.random.randn(128, 128).astype(np.float32)
        b = np.random.randn(128, 128).astype(np.float32)
        _trace_and_compile_to_neff(kernel, a, b)

    def test_matmul_full_pipeline(self):
        def kernel(a, b):
            C = np.matmul(a, b)
            C = knob(C, mem_space="SharedHbm", tile_size=[128, 128],
                     reduction_tile=[128])
            return C

        a = np.random.randn(128, 256).astype(np.float32)
        b = np.random.randn(256, 128).astype(np.float32)
        _trace_and_compile_to_neff(kernel, a, b)

    def test_sigmoid_full_pipeline(self):
        def kernel(x):
            neg_x = -x
            neg_x = knob(neg_x, mem_space="Sbuf", tile_size=[128, 128])
            exp_neg = np.exp(neg_x)
            exp_neg = knob(exp_neg, mem_space="Sbuf", tile_size=[128, 128])
            denom = 1.0 + exp_neg
            denom = knob(denom, mem_space="Sbuf", tile_size=[128, 128])
            result = 1.0 / denom
            result = knob(result, mem_space="SharedHbm", tile_size=[128, 128])
            return result

        x = np.random.randn(128, 256).astype(np.float32)
        _trace_and_compile_to_neff(kernel, x)
