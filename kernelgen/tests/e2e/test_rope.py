"""
End-to-end tests for Rotary Position Embedding (RoPE) kernel.

RoPE applies rotary embeddings to query and key tensors:
    x_out[..., :half] = x[..., :half] * cos - x[..., half:] * sin
    x_out[..., half:] = x[..., :half] * sin + x[..., half:] * cos

This exercises:
1. Tensor slicing (split along last axis)
2. Element-wise multiply with broadcast cos/sin (via expand_dims)
3. Subtraction and addition
4. Concatenation along last axis
5. Phase 0 fold-reshape-into-alloc (2D cos/sin reshaped to 3D)
6. Trivial-broadcast generic canonicalization to named ops

Run with: pytest tests/e2e/test_rope.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Test Cases
# ============================================================================

def test_rope():
    """
    Test RoPE with 3D tensors: x(bs, n_heads, head_dim), cos/sin(bs, half_h).

    cos/sin are expanded to (bs, 1, half_h) via np.expand_dims, creating a
    broadcast multiply pattern. After tiling with tile_size=[128, 1, 64],
    the broadcast becomes trivial (size-1 on both sides).

    This exercises:
    - Phase 0: 2D alloc + copy + reshape -> 3D SBUF alloc
    - Trivial-broadcast linalg.generic -> named op canonicalization
    - 3D SBUF legalization (5D physical layout)
    - Concatenation via insert_slice
    """
    batch = 2
    seq_len = 128
    n_heads = 4
    head_dim = 128
    half_h = head_dim // 2
    bs = batch * seq_len
    tile_size = [128, 1, 64]

    @trace(input_specs=[
        ((bs, n_heads, head_dim), "f32"),
        ((bs, half_h), "f32"),
        ((bs, half_h), "f32"),
    ])
    def rope_kernel(x, freqs_cos, freqs_sin):
        # Broadcast cos/sin to (bs, 1, half_h)
        # No knobs on cos/sin: they are views (expand_dims) of HBM inputs.
        # Tiling promotes them to SBUF automatically as inputs to SBUF compute.
        cos = np.expand_dims(freqs_cos, axis=1)
        sin = np.expand_dims(freqs_sin, axis=1)

        # Split input into two halves along head_dim
        x0 = x[:, :, :half_h]
        x1 = x[:, :, half_h:]

        # Apply rotation
        out_0 = x0 * cos - x1 * sin
        knob.knob(out_0, mem_space="Sbuf", tile_size=tile_size)

        out_1 = x0 * sin + x1 * cos
        knob.knob(out_1, mem_space="Sbuf", tile_size=tile_size)

        # Concatenate back along head_dim axis
        result = np.concatenate([out_0, out_1], axis=-1)
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
        return result

    run_kernel_test(
        rope_kernel,
        stop_after="legalize-layout",
        modes=Mode.LLVM,
    )

    run_kernel_test(
        rope_kernel,
        check_ir_contains=[
            "nisa.alloc", "nisa.tensor_tensor_arith", "nisa.target",
        ],
        check_ir_not_contains=["transform.named_sequence"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


def test_rope_3d_multi_partition():
    """
    Test RoPE with 3D tensors where dim 0 spans multiple partitions:
    x(BH, seq_len, head_dim), cos/sin(1, seq_len, half_dim).

    BH=4 with tile [1, 128, 64] means 4 partitions in SBUF.
    The subtract/add ops produce 4-partition intermediates.
    The vector engine cannot selectively address individual partitions,
    so the compiler must bounce through 1-partition DMA temps.

    This exercises:
    - Multi-partition SBUF element-wise with engine=vector
    - DMA materialization for multi-partition vector operands
    - Concatenation of multi-partition results
    """
    BH = 4
    seq_len = 128
    head_dim = 128
    half_dim = head_dim // 2
    rope_tile = [1, 128, 64]
    attn_tile = [1, 128, 128]

    @trace(input_specs=[
        ((BH, seq_len, half_dim), "f32"),   # q0
        ((BH, seq_len, half_dim), "f32"),   # q1
        ((1, seq_len, half_dim), "f32"),    # freqs_cos (broadcast over BH)
        ((1, seq_len, half_dim), "f32"),    # freqs_sin (broadcast over BH)
    ])
    def rope_3d_kernel(q0, q1, freqs_cos, freqs_sin):
        # Split compound expressions so each intermediate gets an HBM knob.
        # Without this, the multiply intermediates default to SBUF, and the
        # vector engine illegally indexes into specific partitions.
        t0 = q0 * freqs_cos
        knob.knob(t0, mem_space="SharedHbm", tile_size=rope_tile)
        t1 = q1 * freqs_sin
        knob.knob(t1, mem_space="SharedHbm", tile_size=rope_tile)
        q_rot0 = t0 - t1
        knob.knob(q_rot0, mem_space="SharedHbm", tile_size=rope_tile)

        t2 = q0 * freqs_sin
        knob.knob(t2, mem_space="SharedHbm", tile_size=rope_tile)
        t3 = q1 * freqs_cos
        knob.knob(t3, mem_space="SharedHbm", tile_size=rope_tile)
        q_rot1 = t2 + t3
        knob.knob(q_rot1, mem_space="SharedHbm", tile_size=rope_tile)

        result = np.concatenate([q_rot0, q_rot1], axis=-1)
        knob.knob(result, mem_space="SharedHbm", tile_size=attn_tile)
        return result

    run_kernel_test(
        rope_3d_kernel,
        stop_after="legalize-layout",
        modes=Mode.LLVM,
    )

    run_kernel_test(
        rope_3d_kernel,
        check_ir_contains=[
            "nisa.alloc", "nisa.tensor_tensor_arith", "nisa.target",
        ],
        check_ir_not_contains=["transform.named_sequence"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


# ============================================================================
# 3D RoPE: compound expression in SBUF (no HBM workaround)
#
# Same kernel body, parametrized by partition_dim.  The shape is permuted so
# that the partition dim (seq_len=128) sits at position `pdim`.
#
# pdim=0: (seq=128, BH=4, half=64) — partition at dim 0, works today.
# pdim=1: (BH=4, seq=128, half=64) — partition at dim 1, requires
#   infer-layout to propagate partition_dim=1 to compound expression
#   intermediates.  See docs/qwen3_sbuf_partition_dim_workarounds.md
# ============================================================================

_ROPE_COMPOUND_PARAMS = [
    pytest.param(0, id="pdim0"),
    pytest.param(1, id="pdim1"),
]


@pytest.mark.parametrize("pdim", _ROPE_COMPOUND_PARAMS)
def test_rope_3d_compound(pdim):
    """
    3D RoPE compound expression with SBUF output, no intermediate knobs.

    The kernel body is identical for both partition_dim values; only the
    input shapes and tile layout change.
    """
    seq_len = 128
    BH = 4
    half_dim = 64

    # Build shapes: partition dim (seq_len) at position `pdim`,
    # batch dim (BH) at the other position.  Last dim is always half_dim.
    if pdim == 0:
        q_shape = (seq_len, BH, half_dim)
        cos_shape = (seq_len, 1, half_dim)
        tile = [128, 1, 64]
        concat_tile = [128, 1, 128]
    else:
        q_shape = (BH, seq_len, half_dim)
        cos_shape = (1, seq_len, half_dim)
        tile = [1, 128, 64]
        concat_tile = [1, 128, 128]

    @trace(input_specs=[
        (q_shape, "f32"),     # q0
        (q_shape, "f32"),     # q1
        (cos_shape, "f32"),   # freqs_cos (broadcast over BH)
        (cos_shape, "f32"),   # freqs_sin
    ])
    def kernel(q0, q1, freqs_cos, freqs_sin):
        q_rot0 = q0 * freqs_cos - q1 * freqs_sin
        knob.knob(q_rot0, mem_space="Sbuf", tile_size=tile,
                  partition_dim=pdim)

        q_rot1 = q0 * freqs_sin + q1 * freqs_cos
        knob.knob(q_rot1, mem_space="Sbuf", tile_size=tile,
                  partition_dim=pdim)

        result = np.concatenate([q_rot0, q_rot1], axis=-1)
        knob.knob(result, mem_space="SharedHbm", tile_size=concat_tile)
        return result

    run_kernel_test(
        kernel,
        stop_after="legalize-layout",
        modes=Mode.LLVM,
    )

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.alloc", "nisa.tensor_tensor_arith"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
