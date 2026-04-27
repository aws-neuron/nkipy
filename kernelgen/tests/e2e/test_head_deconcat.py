"""
End-to-end tests for head de-concatenation (reshape + transpose + reshape).

This is the pattern used in multi-head attention to merge head outputs back
into the hidden dimension:
    (BH, seq, hdim) -> (B, N, seq, hdim) -> (B, seq, N, hdim) -> (BS, hidden)

The 4D transpose [0,2,1,3] creates a 4D SBUF alloc where dim 0 = batch (small),
not the partition dim (128).  Without the SharedHbm workaround, legalize-layout
cannot tile this alloc and getBaseAndOffsets maps d0 to the batch dim, causing
OOB access in BIR simulation.

The workaround annotates the 4D transpose output as SharedHbm so the transpose
stays in HBM and the SBUF alloc is never created.

Run with: pytest tests/e2e/test_head_deconcat.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


def test_head_deconcat(request):
    """
    Minimal reproducer for the head-deconcat SBUF dim 0 bug.

    Without the SharedHbm knob on the transpose output, the compiler creates
    memref<2x128x2x128xf32, sbuf> with dim 0 = batch = 2.  NISA lowering
    assumes dim 0 = partition (128), causing OOB.

    The SharedHbm workaround keeps the transpose in HBM, sidestepping the issue.
    """
    batch = 2
    n_heads = 2
    seq_len = 128
    head_dim = 128
    BH = batch * n_heads
    BS = batch * seq_len
    hidden = n_heads * head_dim

    @trace(input_specs=[
        ((BH, seq_len, head_dim), "f32"),
        ((hidden, hidden), "f32"),
    ])
    def head_deconcat_kernel(x, w):
        # Reshape to expose batch and head dims
        x = np.reshape(x, (batch, n_heads, seq_len, head_dim))

        # Transpose to (batch, seq, heads, hdim) — the problematic op
        x = np.transpose(x, (0, 2, 1, 3))

        # Collapse back to 2D
        x = np.reshape(x, (BS, hidden))
        # Workaround: annotate the 2D result as SharedHbm so the
        # 4D transpose intermediate stays in HBM (not promoted to SBUF).
        # Without this, the 4D SBUF alloc has dim 0 = batch (not partition),
        # which legalize-layout cannot handle.
        knob.knob(x, mem_space="SharedHbm", tile_size=[128, 128])

        # Downstream matmul
        result = np.matmul(x, w)
        knob.knob(result, mem_space="SharedHbm", tile_size=[128, 128],
                  reduction_tile=[128])
        return result

    # Verify LLVM simulation through legalize-layout
    run_kernel_test(
        head_deconcat_kernel,
        stop_after="legalize-layout",
        modes=Mode.LLVM,
        request=request,
    )

    # Full pipeline: NISA generation + BIR simulation
    run_kernel_test(
        head_deconcat_kernel,
        check_ir_contains=["nisa.matmul", "nisa.alloc"],
        check_ir_not_contains=["transform.named_sequence"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
        request=request,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
