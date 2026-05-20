"""
End-to-end tests for attention kernels.

Covers the core attention building blocks used in Qwen3:
1. Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
2. QKV projection: matmul + split into Q, K, V
3. Attention scores: (Q @ K^T) / sqrt(d) -> softmax

Run with: pytest tests/e2e/test_attention.py -v
"""

import pytest
import numpy as np

from nkigen import trace, knob
from nkigen.apis import fori_loop
from harness import run_kernel_test, Mode


# ============================================================================
# Softmax (standalone, for easier isolated testing)
# ============================================================================


@pytest.mark.parametrize(
    "M, N, tile_size",
    [
        (128, 128, [128, 128]),
        (128, 256, [128, 128]),
        (256, 256, [128, 128]),
    ],
)
def test_softmax(M, N, tile_size):
    """
    Test softmax in isolation: exp(x - max(x)) / sum(exp(x - max(x))).

    Each intermediate step is annotated with a knob for tiling control.
    """

    @trace(input_specs=[((M, N), "f32")])
    def softmax_kernel(x):
        x_fp32 = x.astype(np.float32)

        x_max = np.max(x_fp32, axis=-1, keepdims=True)
        knob.knob(x_max).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")

        shifted = x_fp32 - x_max
        knob.knob(shifted).tile_op(tile_size=tile_size).layout(mem_space="Sbuf")

        exp_x = np.exp(shifted)
        knob.knob(exp_x).tile_op(tile_size=tile_size).layout(mem_space="Sbuf")

        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        knob.knob(sum_exp).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")

        result = exp_x / sum_exp
        knob.knob(result).tile_op(tile_size=tile_size).layout(mem_space="SharedHbm")
        return result

    run_kernel_test(
        softmax_kernel,
        check_ir_contains=["nisa.activation", "op=exp"],
        check_ir_not_contains=["transform.named_sequence"],
        rtol=1e-3,
        atol=1e-3,
        modes=Mode.HW | Mode.STRING_CHECK,
    )


# ============================================================================
# QKV Projection
# ============================================================================


@pytest.mark.parametrize(
    "M, hidden_size, matmul_tile, reduction_tile, elementwise_tile",
    [
        (256, 256, [128, 128], [128], [128, 128]),
    ],
)
def test_qkv_projection(M, hidden_size, matmul_tile, reduction_tile, elementwise_tile):
    """
    Test QKV projection: x @ weight -> split into Q, K, V.

    Returns all three outputs (Q, K, V) using multi-output support.
    The pipeline auto-inserts DMA copies for SBUF→HBM on return values.
    """

    @trace(
        input_specs=[
            ((M, hidden_size), "f32"),
            ((hidden_size, hidden_size * 3), "f32"),
        ]
    )
    def qkv_kernel(x, weight):
        qkv = np.matmul(x, weight)
        knob.knob(qkv).tile_op(tile_size=matmul_tile + reduction_tile).layout(mem_space="Sbuf")

        q, k, v = np.split(qkv, 3, axis=-1)
        return q, k, v

    run_kernel_test(
        qkv_kernel,
        check_ir_contains=["nisa.alloc", "nisa.matmul", "nisa.target"],
        check_ir_not_contains=["transform.named_sequence"],
        modes=Mode.HW | Mode.STRING_CHECK,
    )


# ============================================================================
# Attention Scores with fori_loop (batched)
# ============================================================================


@pytest.mark.parametrize(
    "batch, n_heads, seq_len, head_dim, tile_size",
    [
        (2, 4, 256, 256, [128, 128]),
    ],
)
def test_attention_scores_loop(batch, n_heads, seq_len, head_dim, tile_size):
    """
    Test batched attention scores with fori_loop: softmax((Q @ K^T) / sqrt(d)).

    Uses fori_loop to iterate over batch * n_heads, writing each result slice
    back to the output via the eliminate-same-memspace-copy HBM intermediate
    elimination pattern.
    """
    scale = 1.0 / np.sqrt(head_dim).item()

    @trace(
        input_specs=[
            ((batch * n_heads, seq_len, head_dim), "f32"),
            ((batch * n_heads, head_dim, seq_len), "f32"),
        ]
    )
    def attention_kernel_loop(q, k_transposed):
        init_result = np.empty((batch * n_heads, seq_len, seq_len), dtype=np.float32)

        def body(i, acc):
            q_i = q[i]
            k_i = k_transposed[i]

            scores = np.matmul(q_i, k_i) * scale
            knob.knob(scores).tile_op(tile_size=tile_size).layout(mem_space="Sbuf")

            scores_fp32 = scores.astype(np.float32)

            scores_max = np.max(scores_fp32, axis=-1, keepdims=True)
            knob.knob(scores_max).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")

            shifted = scores_fp32 - scores_max
            knob.knob(shifted).tile_op(tile_size=tile_size).layout(mem_space="Sbuf")

            exp_s = np.exp(shifted)
            knob.knob(exp_s).tile_op(tile_size=tile_size).layout(mem_space="Sbuf")

            sum_exp = np.sum(exp_s, axis=-1, keepdims=True)
            knob.knob(sum_exp).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")

            softmax_out = exp_s / sum_exp
            knob.knob(softmax_out).tile_op(tile_size=tile_size).layout(mem_space="SharedHbm")

            acc[i] = softmax_out
            return acc

        results = fori_loop(0, batch * n_heads, body, init_result)
        return results

    run_kernel_test(
        attention_kernel_loop,
        check_ir_contains=["nisa.dma_copy"],
        check_ir_not_contains=["memref.reshape", "transform.named_sequence"],
        rtol=1e-3,
        atol=1e-3,
        modes=Mode.HW | Mode.STRING_CHECK,
    )


# ============================================================================
# Attention Scores with SBUF BMM output + partition_dim=1
# ============================================================================


@pytest.mark.parametrize(
    "batch, n_heads, seq_len, head_dim, tile_size",
    [
        (2, 4, 256, 256, [1, 128, 128]),
    ],
)
def test_attention_scores_sbuf_bmm(batch, n_heads, seq_len, head_dim, tile_size):
    """
    Test attention scores with BMM output in SBUF and softmax using
    partition_dim=1.

    The BMM is converted to loop + matmul with MxBxN output layout in SBUF.
    Softmax ops use partition_dim=1, requiring canonicalize-partition-dim
    to insert boundary transposes. LegalizeLayout expands the 3D SBUF alloc
    to physical layout and tileCopyAndTranspose handles the HBM→SBUF copy.
    """
    scale = 1.0 / np.sqrt(head_dim).item()

    @trace(
        input_specs=[
            ((batch * n_heads, seq_len, head_dim), "f32"),
            ((batch * n_heads, head_dim, seq_len), "f32"),
        ]
    )
    def attention_kernel(q, k_transposed):
        bmm_result = np.matmul(q, k_transposed)
        knob.knob(bmm_result).tile_op(tile_size=tile_size + [128]).layout(mem_space="Sbuf")

        scores = bmm_result * scale
        knob.knob(scores).tile_op(tile_size=tile_size).layout(mem_space="Sbuf", partition_dim=1)

        scores_fp32 = scores.astype(np.float32)

        scores_max = np.max(scores_fp32, axis=-1, keepdims=True)
        knob.knob(scores_max).tile_op(tile_size=[1, 128, 128]).layout(mem_space="Sbuf", partition_dim=1)

        shifted = scores_fp32 - scores_max
        knob.knob(shifted).tile_op(tile_size=tile_size).layout(mem_space="Sbuf", partition_dim=1)

        exp_s = np.exp(shifted)
        knob.knob(exp_s).tile_op(tile_size=tile_size).layout(mem_space="Sbuf", partition_dim=1)

        sum_exp = np.sum(exp_s, axis=-1, keepdims=True)
        knob.knob(sum_exp).tile_op(tile_size=[1, 128, 128]).layout(mem_space="Sbuf", partition_dim=1)

        result = exp_s / sum_exp
        knob.knob(result).tile_op(tile_size=tile_size).layout(mem_space="SharedHbm", partition_dim=1)
        return result

    # Verify LLVM simulation matches NumPy through legalize-layout
    run_kernel_test(
        attention_kernel,
        stop_after="legalize-layout",
        rtol=1e-3,
        atol=1e-3,
        modes=Mode.LLVM,
    )

    # Verify full pipeline generates NISA dialect ops and simulates correctly
    run_kernel_test(
        attention_kernel,
        rtol=1e-3,
        atol=1e-3,
        check_ir_contains=["nisa.matmul", "nisa.dma_copy"],
        modes=Mode.STRING_CHECK | Mode.HW,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
