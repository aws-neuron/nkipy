"""Raw NKI kernels for device-side token sampling.

Single parameterized kernel generated via ``_make_sample_tokens_kernel``.
The ``has_filtering`` compile-time flag controls which stages are included:

**has_filtering=True** (15 vocab passes):

  Stage 1 - Prepare         load per-row params, clamp temperature
  Stage 2 - Softmax prep    row-max scan, total-exp-mass scan, inv_total
  Stage 3 - Threshold search binary-search top-k & top-p thresholds (12 iters)
  Stage 4 - Combine & clamp max(top_k, top_p, min_p), argmax guarantee
  Stage 5 - CDF sample      prefix-sum in probability space, count(cdf < target)

**has_filtering=False** (3 vocab passes — ~5x faster):

  Stage 1 - Prepare         (same, filtering params ignored)
  Stage 2 - Softmax prep    (same)
  Stage 5 - CDF sample      all tokens are candidates (no threshold masking)

Both variants share the **same tensor interface** (logits, temperatures,
top_ks, top_ps, min_ps, uniform_u) so the caller only needs to pick which
compiled NEFF to run — no input reshaping.

All batch rows are processed in parallel via ``nl.par_dim(batch_size)``
(one partition per row, SIMD across rows).  Only the sequential vocab-tile
loops run serially.

RNG contract: the kernel is deterministic given its inputs.  All randomness
enters through ``uniform_u``; see ``nkipy_serving/sampling/random_state.py`` for lifecycle.

Future: single-NEFF with runtime fast path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NKI Beta 2 adds ``nisa.register_alloc`` + ``nisa.dynamic_range`` for
runtime-controlled loop counts via hardware registers.  With these, the
threshold search loop can use a runtime iteration count (0 for unfiltered,
12 for filtered) instead of a compile-time constant — collapsing the two
NEFFs into one.  This requires migrating the kernel from Beta 1
(``neuronxcc.nki``) to Beta 2 (``nki``), which also changes indexing
(``nl.arange`` → integer slicing), load/store (``nl.load`` →
``nisa.dma_copy``), and dtype (``np.float32`` → ``nl.float32``).
Blocked on NKI Beta 2 stabilization and nkipy Beta 2 support.
"""

from __future__ import annotations

import math

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from nkipy.core.nki_op import wrap_nki_kernel

_CUMSUM_TILE = 2048
_SCAN_TILE_SMALL = 256  # for small vocabs where large tiles cause precision issues
_SCAN_TILE_LARGE = 2048  # for production vocabs (>= 4096 tokens)
_SCAN_TILE_THRESHOLD = 4096  # vocab size above which we use the large tile
_THRESHOLD_SEARCH_ITERS = 12
_TEMP_EPS = np.float32(1e-6)
_MASS_EPS = np.float32(1e-20)
_TARGET_EPS = np.float32(1e-7)


# ===================================================================
# Tile helpers — work for any partition size (1 or batch_size).
#
# Scalars are (P, 1) tiles; vectors are (P, F) tiles.  When a second
# operand has free_dim == 1 we use tensor_scalar so the ISA broadcasts
# the per-partition scalar across the free dimension automatically.
# ===================================================================


def _select_tile(predicate, on_true, on_false):
    # nl.where accepts both NKI predicates (from comparison expressions)
    # and uint8 data tensors (from _tensor_binary logical ops) directly.
    # Do NOT re-compare the predicate through _compare_tile — that sends
    # a predicate expression into nisa.tensor_scalar as data, which is
    # fragile and breaks for large tile widths.
    if hasattr(on_true, "shape"):
        shape = on_true.shape
        dtype = on_true.dtype
    else:
        shape = on_false.shape
        dtype = on_false.dtype
    if not hasattr(on_true, "shape"):
        on_true = nl.full(shape, on_true, dtype=dtype, buffer=nl.sbuf)
    if not hasattr(on_false, "shape"):
        on_false = nl.full(shape, on_false, dtype=dtype, buffer=nl.sbuf)
    return nl.where(predicate, on_true, on_false, dtype=dtype)


def _compare_tile(data, op, operand):
    out = nl.ndarray(data.shape, dtype=np.uint8, buffer=nl.sbuf)
    if hasattr(operand, "shape") and len(operand.shape) >= 2:
        if int(operand.shape[-1]) == 1:
            # (P, 1) scalar per partition — broadcast via tensor_scalar.
            out[...] = nisa.tensor_scalar(
                data=data,
                op0=op,
                operand0=operand[:, 0],
                dtype=np.uint8,
            )
        else:
            out[...] = nisa.tensor_tensor(
                data1=data,
                data2=operand,
                op=op,
                dtype=np.uint8,
            )
    else:
        out[...] = nisa.tensor_scalar(
            data=data,
            op0=op,
            operand0=operand,
            dtype=np.uint8,
        )
    return out


def _tensor_binary(data1, data2, op, *, dtype):
    out = nl.ndarray(data1.shape, dtype=dtype, buffer=nl.sbuf)
    if hasattr(data2, "shape") and len(data2.shape) >= 2 and int(data2.shape[-1]) == 1:
        # (P, 1) scalar per partition — broadcast via tensor_scalar.
        out[...] = nisa.tensor_scalar(
            data=data1,
            op0=op,
            operand0=data2[:, 0],
            dtype=dtype,
        )
    elif hasattr(data2, "shape") and len(data2.shape) > 0:
        out[...] = nisa.tensor_tensor(
            data1=data1,
            data2=data2,
            op=op,
            dtype=dtype,
        )
    else:
        out[...] = nisa.tensor_scalar(
            data=data1,
            op0=op,
            operand0=data2,
            dtype=dtype,
        )
    return out


def _tensor_scalar(data, op, operand, *, dtype):
    out = nl.ndarray(data.shape, dtype=dtype, buffer=nl.sbuf)
    out[...] = nisa.tensor_scalar(
        data=data,
        op0=op,
        operand0=operand,
        dtype=dtype,
    )
    return out


def _reduce_add(tile):
    out = nl.ndarray((tile.shape[0], 1), dtype=np.float32, buffer=nl.sbuf)
    out[...] = nisa.tensor_reduce(
        op=nl.add,
        data=tile,
        axis=[1],
        keepdims=True,
        dtype=np.float32,
    )
    return out


def _reduce_max(tile):
    out = nl.ndarray((tile.shape[0], 1), dtype=np.float32, buffer=nl.sbuf)
    out[...] = nisa.tensor_reduce(
        op=nl.maximum,
        data=tile,
        axis=[1],
        keepdims=True,
        dtype=np.float32,
    )
    return out


# ===================================================================
# Parameterized sampler kernel — par_dim(batch_size) parallelism
#
# _make_sample_tokens_kernel(has_filtering) returns an @nki.jit kernel.
# The has_filtering bool is captured in the closure and evaluated at
# trace time, so the compiler eliminates dead stages entirely.
# ===================================================================


def _make_sample_tokens_kernel(*, has_filtering: bool):
    """Factory: compile-time parameterized sampler kernel.

    has_filtering=True  → full 15-pass pipeline (threshold search + CDF)
    has_filtering=False → fast 3-pass pipeline  (CDF only, no thresholds)

    Both variants take the same 6 tensor inputs so the caller never needs
    to reshape or change the input dict.
    """

    @nki.jit(
        debug_kernel=False,
        show_compiler_tb=True,
    )
    def _sample_tokens_nki(
        logits,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        uniform_u,
    ):
        if len(logits.shape) != 2:
            raise RuntimeError(f"expected rank-2 logits, got shape={logits.shape}")

        rows = int(logits.shape[0])
        cols = int(logits.shape[1])
        TILE = _SCAN_TILE_LARGE if cols >= _SCAN_TILE_THRESHOLD else _SCAN_TILE_SMALL
        num_scan_tiles = int(math.ceil(cols / TILE))
        out = nl.ndarray((rows, 1), dtype=np.float32, buffer=nl.shared_hbm)

        i_p = nl.arange(rows)[:, None]
        i_f = nl.arange(TILE)[None, :]
        i_v = nl.arange(1)[None, :]
        ones_scan = nl.ones((rows, TILE), dtype=np.float32)

        # ============================================================
        # Stage 1 - Prepare: load per-row params, clamp temperature
        # ============================================================
        temp_in = nl.load(temperatures[i_p, i_v])
        uniform_in = nl.load(uniform_u[i_p, i_v])

        temp_safe = _tensor_scalar(
            temp_in,
            nl.maximum,
            _TEMP_EPS,
            dtype=np.float32,
        )
        row_inv_temp = nl.ndarray((rows, 1), dtype=np.float32, buffer=nl.sbuf)
        row_inv_temp[...] = nisa.activation(
            op=nl.reciprocal,
            data=temp_safe,
            dtype=np.float32,
        )
        row_uniform_safe = _tensor_scalar(
            _tensor_scalar(
                uniform_in,
                nl.maximum,
                _TARGET_EPS,
                dtype=np.float32,
            ),
            nl.minimum,
            np.float32(1.0 - _TARGET_EPS),
            dtype=np.float32,
        )

        if has_filtering:
            top_k_in = nl.load(top_ks[i_p, i_v])
            top_p_in = nl.load(top_ps[i_p, i_v])
            min_p_in = nl.load(min_ps[i_p, i_v])
            row_top_k_f = _tensor_scalar(
                top_k_in,
                nl.add,
                np.float32(0.0),
                dtype=np.float32,
            )
            row_active_topk = _compare_tile(
                top_k_in,
                nl.less,
                np.int32(cols),
            )
            row_active_topp = _compare_tile(
                top_p_in,
                nl.less,
                np.float32(1.0 - 1e-6),
            )

        # ============================================================
        # Stage 2 - Online softmax: fused row-max + total exp mass
        # ============================================================
        neg_inf_tile = nl.full(
            (rows, TILE),
            -np.inf,
            dtype=np.float32,
            buffer=nl.sbuf,
        )

        row_scaled_max = nl.full((rows, 1), -np.inf, dtype=np.float32, buffer=nl.sbuf)
        row_total_mass = nl.zeros((rows, 1), dtype=np.float32, buffer=nl.sbuf)

        for scan_tile in nl.sequential_range(num_scan_tiles):
            chunk_mask = 0 * i_p + scan_tile * TILE + i_f < cols
            chunk_logits = nl.load(
                logits[i_p, scan_tile * TILE + i_f],
                mask=chunk_mask,
            )
            chunk_logits = nl.where(
                chunk_mask,
                chunk_logits,
                neg_inf_tile,
                dtype=np.float32,
            )
            chunk_scaled = _tensor_binary(
                chunk_logits,
                row_inv_temp,
                nl.multiply,
                dtype=np.float32,
            )
            tile_max = _reduce_max(chunk_scaled)
            new_max = _tensor_binary(
                row_scaled_max,
                tile_max,
                nl.maximum,
                dtype=np.float32,
            )
            correction_exp = _tensor_binary(
                row_scaled_max,
                new_max,
                nl.subtract,
                dtype=np.float32,
            )
            correction = nl.ndarray((rows, 1), dtype=np.float32, buffer=nl.sbuf)
            correction[...] = nisa.activation(
                op=nl.exp,
                data=correction_exp,
                dtype=np.float32,
            )
            row_total_mass[...] = _tensor_binary(
                row_total_mass,
                correction,
                nl.multiply,
                dtype=np.float32,
            )
            row_scaled_max[...] = new_max
            neg_new_max = _tensor_scalar(
                new_max,
                nl.multiply,
                np.float32(-1.0),
                dtype=np.float32,
            )
            chunk_shifted = _tensor_binary(
                chunk_scaled,
                neg_new_max,
                nl.add,
                dtype=np.float32,
            )
            chunk_exp = nl.ndarray((rows, TILE), dtype=np.float32, buffer=nl.sbuf)
            chunk_exp[...] = nisa.activation(
                op=nl.exp,
                data=chunk_shifted,
                dtype=np.float32,
            )
            row_total_mass[...] = _tensor_binary(
                row_total_mass,
                _reduce_add(chunk_exp),
                nl.add,
                dtype=np.float32,
            )

        row_neg_scaled_max = _tensor_scalar(
            row_scaled_max,
            nl.multiply,
            np.float32(-1.0),
            dtype=np.float32,
        )
        row_safe_total_mass = _tensor_scalar(
            row_total_mass,
            nl.maximum,
            _MASS_EPS,
            dtype=np.float32,
        )
        row_max_prob = nl.ndarray((rows, 1), dtype=np.float32, buffer=nl.sbuf)
        row_max_prob[...] = nisa.activation(
            op=nl.reciprocal,
            data=row_safe_total_mass,
            dtype=np.float32,
        )

        # ============================================================
        # Stages 3-4 - Threshold search + combine (filtered only)
        # ============================================================
        if has_filtering:
            row_topp_target_mass = _tensor_binary(
                top_p_in,
                row_safe_total_mass,
                nl.multiply,
                dtype=np.float32,
            )

            low_topk = nl.zeros((rows, 1), dtype=np.float32, buffer=nl.sbuf)
            high_topk = _select_tile(row_active_topk, row_max_prob, np.float32(0.0))
            low_topp = nl.zeros((rows, 1), dtype=np.float32, buffer=nl.sbuf)
            high_topp = _select_tile(row_active_topp, row_max_prob, np.float32(0.0))

            for _ in nl.static_range(_THRESHOLD_SEARCH_ITERS):
                mid_topk = _tensor_scalar(
                    _tensor_binary(low_topk, high_topk, nl.add, dtype=np.float32),
                    nl.multiply,
                    np.float32(0.5),
                    dtype=np.float32,
                )
                mid_topp = _tensor_scalar(
                    _tensor_binary(low_topp, high_topp, nl.add, dtype=np.float32),
                    nl.multiply,
                    np.float32(0.5),
                    dtype=np.float32,
                )
                topk_unnorm = _tensor_binary(
                    mid_topk,
                    row_safe_total_mass,
                    nl.multiply,
                    dtype=np.float32,
                )
                topp_unnorm = _tensor_binary(
                    mid_topp,
                    row_safe_total_mass,
                    nl.multiply,
                    dtype=np.float32,
                )
                count_acc = nl.zeros((rows, 1), dtype=np.float32, buffer=nl.sbuf)
                mass_acc = nl.zeros((rows, 1), dtype=np.float32, buffer=nl.sbuf)
                for scan_tile in nl.sequential_range(num_scan_tiles):
                    chunk_mask = 0 * i_p + scan_tile * TILE + i_f < cols
                    chunk_logits = nl.load(
                        logits[i_p, scan_tile * TILE + i_f],
                        mask=chunk_mask,
                    )
                    chunk_logits = nl.where(
                        chunk_mask,
                        chunk_logits,
                        neg_inf_tile,
                        dtype=np.float32,
                    )
                    chunk_scaled = _tensor_binary(
                        chunk_logits,
                        row_inv_temp,
                        nl.multiply,
                        dtype=np.float32,
                    )
                    chunk_shifted = _tensor_binary(
                        chunk_scaled,
                        row_neg_scaled_max,
                        nl.add,
                        dtype=np.float32,
                    )
                    chunk_exp = nl.ndarray(
                        (rows, TILE), dtype=np.float32, buffer=nl.sbuf
                    )
                    chunk_exp[...] = nisa.activation(
                        op=nl.exp,
                        data=chunk_shifted,
                        dtype=np.float32,
                    )
                    chunk_topk_mask = _compare_tile(
                        chunk_exp,
                        nl.greater_equal,
                        topk_unnorm,
                    )
                    chunk_topp_mask = _compare_tile(
                        chunk_exp,
                        nl.greater_equal,
                        topp_unnorm,
                    )
                    count_acc[...] = _tensor_binary(
                        count_acc,
                        _reduce_add(
                            _select_tile(chunk_topk_mask, ones_scan, np.float32(0.0))
                        ),
                        nl.add,
                        dtype=np.float32,
                    )
                    mass_acc[...] = _tensor_binary(
                        mass_acc,
                        _reduce_add(
                            _select_tile(chunk_topp_mask, chunk_exp, np.float32(0.0))
                        ),
                        nl.add,
                        dtype=np.float32,
                    )

                can_raise_topk = _compare_tile(count_acc, nl.greater_equal, row_top_k_f)
                can_raise_topp = _compare_tile(
                    mass_acc, nl.greater_equal, row_topp_target_mass
                )
                low_topk[...] = _select_tile(
                    row_active_topk,
                    _select_tile(can_raise_topk, mid_topk, low_topk),
                    low_topk,
                )
                high_topk[...] = _select_tile(
                    row_active_topk,
                    _select_tile(can_raise_topk, high_topk, mid_topk),
                    high_topk,
                )
                low_topp[...] = _select_tile(
                    row_active_topp,
                    _select_tile(can_raise_topp, mid_topp, low_topp),
                    low_topp,
                )
                high_topp[...] = _select_tile(
                    row_active_topp,
                    _select_tile(can_raise_topp, high_topp, mid_topp),
                    high_topp,
                )

            # Stage 4 - Combine & clamp.
            row_minp_threshold = _tensor_binary(
                row_max_prob,
                min_p_in,
                nl.multiply,
                dtype=np.float32,
            )
            row_threshold = _tensor_binary(
                _tensor_binary(low_topk, low_topp, nl.maximum, dtype=np.float32),
                row_minp_threshold,
                nl.maximum,
                dtype=np.float32,
            )
            row_threshold_unnorm = _tensor_binary(
                row_threshold,
                row_safe_total_mass,
                nl.multiply,
                dtype=np.float32,
            )
            row_threshold_unnorm[...] = _tensor_scalar(
                row_threshold_unnorm,
                nl.minimum,
                np.float32(1.0 - 1e-6),
                dtype=np.float32,
            )
        else:
            # No filtering — threshold is zero, all tokens pass.
            row_threshold_unnorm = nl.zeros(
                (rows, 1),
                dtype=np.float32,
                buffer=nl.sbuf,
            )

        # ============================================================
        # Stage 5 - CDF sample: prefix-sum → count(cdf < target)
        # ============================================================

        # Pass 1: prefix-sum of (masked) probabilities to obtain total.
        cdf_carry = nl.zeros((rows, 1), dtype=np.float32)
        for scan_tile in nl.sequential_range(num_scan_tiles):
            chunk_mask = 0 * i_p + scan_tile * TILE + i_f < cols
            chunk_logits = nl.load(
                logits[i_p, scan_tile * TILE + i_f],
                mask=chunk_mask,
            )
            chunk_logits = nl.where(
                chunk_mask,
                chunk_logits,
                neg_inf_tile,
                dtype=np.float32,
            )
            chunk_scaled = _tensor_binary(
                chunk_logits,
                row_inv_temp,
                nl.multiply,
                dtype=np.float32,
            )
            chunk_shifted = _tensor_binary(
                chunk_scaled,
                row_neg_scaled_max,
                nl.add,
                dtype=np.float32,
            )
            chunk_exp = nl.ndarray((rows, TILE), dtype=np.float32, buffer=nl.sbuf)
            chunk_exp[...] = nisa.activation(
                op=nl.exp,
                data=chunk_shifted,
                dtype=np.float32,
            )
            if has_filtering:
                chunk_allowed = _compare_tile(
                    chunk_exp,
                    nl.greater_equal,
                    row_threshold_unnorm,
                )
                masked_exp = _select_tile(chunk_allowed, chunk_exp, np.float32(0.0))
            else:
                masked_exp = chunk_exp
            prob = _tensor_binary(
                masked_exp,
                row_max_prob,
                nl.multiply,
                dtype=np.float32,
            )
            local_cumsum = nisa.tensor_tensor_scan(
                data0=ones_scan,
                data1=prob,
                initial=cdf_carry,
                op0=np.multiply,
                op1=np.add,
                dtype=np.float32,
            )
            cdf_carry[:, :] = nl.copy(local_cumsum[:, TILE - 1])

        safe_cdf_total = _tensor_scalar(
            cdf_carry,
            nl.maximum,
            _MASS_EPS,
            dtype=np.float32,
        )
        row_sample_target = _tensor_binary(
            row_uniform_safe,
            safe_cdf_total,
            nl.multiply,
            dtype=np.float32,
        )

        # Pass 2: re-run prefix-sum and count entries below target.
        carry = nl.zeros((rows, 1), dtype=np.float32)
        count_below = nl.zeros((rows, 1), dtype=np.float32, buffer=nl.sbuf)

        for scan_tile in nl.sequential_range(num_scan_tiles):
            chunk_mask = 0 * i_p + scan_tile * TILE + i_f < cols
            chunk_logits = nl.load(
                logits[i_p, scan_tile * TILE + i_f],
                mask=chunk_mask,
            )
            chunk_logits = nl.where(
                chunk_mask,
                chunk_logits,
                neg_inf_tile,
                dtype=np.float32,
            )
            chunk_scaled = _tensor_binary(
                chunk_logits,
                row_inv_temp,
                nl.multiply,
                dtype=np.float32,
            )
            chunk_shifted = _tensor_binary(
                chunk_scaled,
                row_neg_scaled_max,
                nl.add,
                dtype=np.float32,
            )
            chunk_exp = nl.ndarray((rows, TILE), dtype=np.float32, buffer=nl.sbuf)
            chunk_exp[...] = nisa.activation(
                op=nl.exp,
                data=chunk_shifted,
                dtype=np.float32,
            )
            if has_filtering:
                chunk_allowed = _compare_tile(
                    chunk_exp,
                    nl.greater_equal,
                    row_threshold_unnorm,
                )
                masked_exp = _select_tile(chunk_allowed, chunk_exp, np.float32(0.0))
            else:
                masked_exp = chunk_exp
            prob = _tensor_binary(
                masked_exp,
                row_max_prob,
                nl.multiply,
                dtype=np.float32,
            )

            local_cumsum = nisa.tensor_tensor_scan(
                data0=ones_scan,
                data1=prob,
                initial=carry,
                op0=np.multiply,
                op1=np.add,
                dtype=np.float32,
            )

            below = _compare_tile(local_cumsum, nl.less, row_sample_target)
            count_below[...] = _tensor_binary(
                count_below,
                _reduce_add(_select_tile(below, ones_scan, np.float32(0.0))),
                nl.add,
                dtype=np.float32,
            )

            carry[:, :] = nl.copy(
                local_cumsum[:, TILE - 1],
                mask=scan_tile + 1 < num_scan_tiles,
            )

        row_sampled_idx = _tensor_scalar(
            count_below,
            nl.minimum,
            np.float32(cols - 1),
            dtype=np.float32,
        )
        nl.store(out[i_p, i_v], row_sampled_idx)
        return out

    return _sample_tokens_nki


_sample_tokens_filtered = _make_sample_tokens_kernel(has_filtering=True)
_sample_tokens_unfiltered = _make_sample_tokens_kernel(has_filtering=False)


# ===================================================================
# Cumsum kernel (unchanged — used by the reference sampler path)
# ===================================================================


@nki.jit(
    debug_kernel=False,
    show_compiler_tb=True,
)
def _cumsum_last_dim_nki(x_tensor):
    """Compute float32 cumsum over the last dim of a rank-2 tensor."""
    if len(x_tensor.shape) != 2:
        raise RuntimeError(f"expected rank-2 input, got shape={x_tensor.shape}")

    rows = int(x_tensor.shape[0])
    cols = int(x_tensor.shape[1])
    pmax = int(nl.tile_size.pmax)
    row_tiles = int(math.ceil(rows / pmax))
    col_tiles = int(math.ceil(cols / _CUMSUM_TILE))

    out = nl.ndarray(x_tensor.shape, dtype=np.float32, buffer=nl.shared_hbm)
    pi, fi = nl.mgrid[0:pmax, 0:_CUMSUM_TILE]
    ones = nl.ones((pmax, _CUMSUM_TILE), dtype=np.float32)

    for row_tile in nl.affine_range(row_tiles):
        init = nl.zeros((pmax, 1), dtype=np.float32)
        for col_tile in nl.sequential_range(col_tiles):
            mask = (row_tile * pmax + pi < rows) & (col_tile * _CUMSUM_TILE + fi < cols)
            data = nl.load(
                x_tensor[row_tile * pmax + pi, col_tile * _CUMSUM_TILE + fi],
                mask=mask,
            )
            result = nisa.tensor_tensor_scan(
                data0=ones,
                data1=data,
                initial=init,
                op0=np.multiply,
                op1=np.add,
                dtype=np.float32,
                mask=mask,
            )
            nl.store(
                out[row_tile * pmax + pi, col_tile * _CUMSUM_TILE + fi],
                result,
                mask=mask,
            )
            init[:, :] = nl.copy(
                result[:, _CUMSUM_TILE - 1], mask=col_tile + 1 < col_tiles
            )
    return out


# ===================================================================
# Traceable wrappers
# ===================================================================


def cumsum_last_dim(x: np.ndarray) -> np.ndarray:
    """Traceable NKIPy wrapper around the raw NKI cumsum kernel."""
    nki_op = wrap_nki_kernel(_cumsum_last_dim_nki, [x])
    return nki_op(x)


def sample_tokens(
    logits: np.ndarray,
    temperatures: np.ndarray,
    top_ks: np.ndarray,
    top_ps: np.ndarray,
    min_ps: np.ndarray,
    uniform_u: np.ndarray,
    *,
    _unfiltered: bool = False,
) -> np.ndarray:
    """Traceable raw NKI sampler from logits to sampled token ids.

    Same 6-tensor interface for both paths.  ``_unfiltered=True`` selects
    the 3-pass kernel that skips the threshold search.
    """
    kernel = _sample_tokens_unfiltered if _unfiltered else _sample_tokens_filtered
    nki_op = wrap_nki_kernel(
        kernel,
        [logits, temperatures, top_ks, top_ps, min_ps, uniform_u],
    )
    sampled = nki_op(logits, temperatures, top_ks, top_ps, min_ps, uniform_u)
    return sampled.reshape((int(logits.shape[0]),))
