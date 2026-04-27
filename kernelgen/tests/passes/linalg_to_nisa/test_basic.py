"""
Tests for linalg.sqrt -> nisa.activation(op=sqrt) lowering.

The linalg-to-nisa pass should convert linalg.sqrt into nisa.activation
with op=sqrt, running on the SCALAR engine.

Run with: python -m pytest tests/passes/linalg_to_nisa/test_sqrt.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


def test_sqrt_basic():
    """
    Basic sqrt: linalg.sqrt should be lowered to nisa.activation(op=sqrt).
    """
    shape = (128, 256)
    tile_size = [64, 128]

    @trace(input_specs=[(shape, "f32")])
    def kernel(a):
        result = np.sqrt(a)
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
        return result

    np.random.seed(42)
    A = np.abs(np.random.randn(*shape)).astype(np.float32) + 0.01

    # After linalg-to-nisa: sqrt should become nisa.activation
    check_patterns = """
    CHECK: func.func
    CHECK: nisa.activation
    CHECK-SAME: op=sqrt
    CHECK-NOT: linalg.sqrt
    CHECK: return
    """
    run_kernel_test(
        kernel,
        stop_after='linalg-to-nisa',
        check_patterns=check_patterns,
        inputs=[A],
        modes=Mode.FILECHECK,
    )

    # BIR simulation: verify numerical correctness through full pipeline
    run_kernel_test(
        kernel,

        check_ir_contains=["nisa.activation", "op=sqrt"],
        inputs=[A],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK,
    )


def test_sqrt_256x256():
    """
    Sqrt on a 256x256 tensor with 128x128 tiles.
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32")])
    def kernel(a):
        result = np.sqrt(a)
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
        return result

    np.random.seed(42)
    A = np.abs(np.random.randn(*shape)).astype(np.float32) + 0.01

    check_patterns = """
    CHECK: func.func
    CHECK: nisa.activation
    CHECK-SAME: op=sqrt
    CHECK-NOT: linalg.sqrt
    CHECK: return
    """
    run_kernel_test(
        kernel,
        stop_after='linalg-to-nisa',
        check_patterns=check_patterns,
        inputs=[A],
        modes=Mode.FILECHECK,
    )

    # BIR simulation: verify numerical correctness through full pipeline
    run_kernel_test(
        kernel,

        check_ir_contains=["nisa.activation", "op=sqrt"],
        inputs=[A],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
