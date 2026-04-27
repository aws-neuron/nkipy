"""
Tests for annotate-memory-space pass.

The annotate-memory-space pass:
1. Annotates function inputs/outputs with SharedHbm (#nisa.mem<sharedhbm>)
2. Applies memory space attributes from nkipy.annotate to internal memrefs
3. Propagates memory spaces through subview, collapse_shape, expand_shape, reshape ops
4. Removes nkipy.annotate ops after processing

Run with: python -m pytest tests/passes/annotate_memory_space/test_basic.py -v
Or directly: python tests/passes/annotate_memory_space/test_basic.py
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

    This tests that the pass correctly:
    1. Annotates function arguments with #nisa.mem<sharedhbm>
    2. Applies 3 : i32 to matmul intermediate buffer
    3. Propagates memory space to subviews
    4. Removes all nkipy.annotate ops
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

    # FileCheck patterns to verify:
    # 1. All 3 function arguments get 4 : i32
    # 2. Matmul intermediate gets 3 : i32 from nkipy.annotate
    # 3. Transpose temp stays without memory space annotation
    # 4. LHS/RHS promote buffers already have 3 : i32
    # 5. PSUM accumulator already has 2 : i32
    # 6. Add's inputs/output promoted to SBUF (by elementwise SBUF promotion)
    # 7. Add final output gets 4 : i32 from nkipy.annotate
    # 8. All nkipy.annotate ops are removed
    # 9. Memory space propagates to subviews
    #
    # With SBUF promotion for elementwise ops, the add tiling loop contains:
    # - 3 SBUF allocs for add's promoted inputs and output
    # - Copy from matmul SBUF output to promoted input 1 (SBUF->SBUF)
    # - Copy from bias SharedHbm to promoted input 2 (SharedHbm->SBUF)
    # - Copy from promoted output to SharedHbm result (SBUF->SharedHbm)
    check_patterns = '''
CHECK: func.func @matmul_add_kernel
CHECK-SAME: memref<256x256xf32, strided<[?, ?], offset: ?>, 4 : i32>
CHECK-SAME: memref<256x256xf32, strided<[?, ?], offset: ?>, 4 : i32>
CHECK-SAME: memref<256x256xf32, strided<[?, ?], offset: ?>, 4 : i32>
CHECK-SAME: -> memref<256x256xf32, 4 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 3 : i32>
CHECK: linalg.transpose{{.*}}outs({{.*}}memref<256x256xf32, 3 : i32>)
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 3 : i32>
CHECK: memref.copy{{.*}}3 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 3 : i32>
CHECK: memref.copy{{.*}}to memref<256x256xf32, 3 : i32>
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.subview{{.*}}3 : i32>
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 2 : i32>
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.subview{{.*}}3 : i32>
CHECK: linalg.matmul
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.copy{{.*}}2 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.copy{{.*}}3 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 4 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 3 : i32>
CHECK: memref.copy{{.*}}3 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: memref.subview{{.*}}4 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 3 : i32>
CHECK: memref.copy{{.*}}4 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 3 : i32>
CHECK: linalg.add{{.*}}3 : i32>{{.*}}3 : i32>{{.*}}3 : i32>
CHECK: memref.subview{{.*}}4 : i32>
CHECK: memref.copy{{.*}}3 : i32>{{.*}}to{{.*}}4 : i32>
CHECK-NOT: nkipy.annotate
CHECK: return{{.*}}memref<256x256xf32, 4 : i32>
'''
    run_kernel_test(
        matmul_add_kernel,
        stop_after='annotate-memory-space',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_rope_reshape_propagation():
    """
    Test RoPE kernel where expand_dims produces expand_shape view ops.

    expand_dims creates a tensor.expand_shape (a view) of the HBM input.
    Since views can't cross memory spaces, the expand_shape stays on HBM.
    Tiling automatically promotes inputs of SBUF compute ops to SBUF
    by inserting HBM->SBUF copies.

    This tests that the pass correctly:
    1. expand_shape views stay in HBM (same memory space as source)
    2. Tiling creates SBUF allocs for compute inputs
    3. Intermediate elementwise results annotated as SBUF get sbuf
    4. Final concatenated output annotated as SharedHbm gets shared_hbm
    5. All nkipy.annotate ops are removed
    """
    bs = 256
    n_heads = 4
    head_dim = 128
    half_h = head_dim // 2
    tile_size = [128, 1, 64]

    @trace(input_specs=[
        ((bs, n_heads, head_dim), "f32"),
        ((bs, half_h), "f32"),
        ((bs, half_h), "f32"),
    ])
    def rope_kernel(x, freqs_cos, freqs_sin):
        # No knobs on cos/sin: they are views (expand_dims) of HBM inputs.
        # Tiling promotes them to SBUF automatically as inputs to SBUF compute.
        cos = np.expand_dims(freqs_cos, axis=1)
        sin = np.expand_dims(freqs_sin, axis=1)

        x0 = x[:, :, :half_h]
        x1 = x[:, :, half_h:]

        out_0 = x0 * cos - x1 * sin
        knob.knob(out_0, mem_space="Sbuf", tile_size=tile_size)

        out_1 = x0 * sin + x1 * cos
        knob.knob(out_1, mem_space="Sbuf", tile_size=tile_size)

        result = np.concatenate([out_0, out_1], axis=-1)
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
        return result

    # Key checks:
    # 1. Function args get 4 : i32
    # 2. expand_dims stays on HBM side (expand_shape is a view, same mem space)
    # 3. Tiling creates SBUF allocs for compute inputs (copies from HBM)
    # 4. Intermediate SBUF allocs are annotated
    # 5. Final output is 4 : i32
    # 6. No nkipy.annotate ops remain
    check_patterns = '''
CHECK: func.func @rope_kernel
CHECK-SAME: memref<256x4x128xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
CHECK-SAME: memref<256x64xf32, strided<[?, ?], offset: ?>, 4 : i32>
CHECK-SAME: memref<256x64xf32, strided<[?, ?], offset: ?>, 4 : i32>
CHECK-SAME: -> memref<256x4x128xf32, 4 : i32>
CHECK: memref.expand_shape{{.*}}4 : i32>
CHECK: memref.expand_shape{{.*}}4 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x4x64xf32, 3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x4x64xf32, 3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x4x64xf32, 3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x4x64xf32, 3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x4x64xf32, 3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x4x64xf32, 3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x4x128xf32, 4 : i32>
CHECK-NOT: nkipy.annotate
CHECK: return{{.*}}memref<256x4x128xf32, 4 : i32>
'''
    run_kernel_test(
        rope_kernel,
        stop_after='annotate-memory-space',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
