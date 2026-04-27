"""
Tests that AnnotateOp bufferizes correctly via BufferizableOpInterface.

After one-shot-bufferize, nkipy.annotate ops should target memrefs (not tensors),
with no leftover bufferization.to_tensor wrappers.

Run with: python -m pytest tests/passes/cleanup_bufferization_artifacts/test_basic.py -v
Or directly: python tests/passes/cleanup_bufferization_artifacts/test_basic.py
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode

# ============================================================================
# Test Cases
# ============================================================================

def test_matmul_sbuf_add_hbm():
    """
    Test matmul-add chain with:
    - matmul output -> SBUF (intermediate)
    - add output -> SharedHbm (returned result)

    After bufferization, nkipy.annotate ops should operate on memrefs directly
    (BufferizableOpInterface on AnnotateOp handles the tensor→memref conversion).
    """
    M, N, K = 256, 256, 256
    matmul_tile = [128, 128]       # TILE_M, TILE_N
    matmul_reduction_tile = [128]  # TILE_K
    add_tile = [128, 128]          # TILE_M, TILE_N

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
    def matmul_add_kernel(a, b, bias):
        # Matmul outputs to SBUF for reuse in the add
        c = np.matmul(a, b)
        knob.knob(c, mem_space="Sbuf", tile_size=matmul_tile, reduction_tile=matmul_reduction_tile)

        # Add outputs to SharedHbm (returned from kernel)
        result = c + bias
        knob.knob(result, mem_space="SharedHbm", tile_size=add_tile)

        return result

    # After bufferization + canonicalize (stop_after=9):
    # - nkipy.annotate ops should target memrefs, not tensors
    # - No bufferization.to_tensor wrappers should remain for annotate
    # - Sbuf (mem_space=3) and SharedHbm (mem_space=4) annotations preserved
    check_patterns = '''
CHECK: func.func
CHECK-NOT: bufferization.to_tensor
CHECK: nkipy.annotate
CHECK-SAME: mem_space = 3
CHECK-SAME: reduction_tile = array<i64: 128>
CHECK-SAME: tile_size = array<i64: 128, 128>
CHECK-NOT: bufferization.to_tensor
CHECK: nkipy.annotate
CHECK-SAME: mem_space = 4
CHECK-SAME: tile_size = array<i64: 128, 128>
CHECK-NOT: bufferization.to_tensor
CHECK: return
'''
    run_kernel_test(
        matmul_add_kernel,
        stop_after='one-shot-bufferize',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
