"""Sparse MLA attention on Trn2.

DeepSeek-V4 sparse attention gathers at most ``K`` key/value positions
per query token (from a union of a sliding-window and a compressed-KV top-k
selector), then runs a sink-aware softmax-attention over the gathered
positions. The direct Python oracle is O(N_q) host dispatches per layer;
this module moves the compute to an NKI kernel.

Current device path:

- Kernel runs one query token at a time: ``q[h, d] x kv[K, d] -> o[h, d]``
  with an online ``(m, l)`` softmax over K tiles and an ``attn_sink`` denom
  contribution.
- The runtime wrapper (``sparse_mla_attention_host_gather``) gathers
  ``kv`` into a dense ``[N_q, K, d]`` tensor on host from ``topk_idxs``,
  zeros invalid (-1) slots and builds a boolean valid mask, then calls the
  kernel for every query. Returns ``out[N_q, h, d]`` matching
  ``_sparse_attn`` semantics.
- Scheduler interface: see ``attention/deepseek_v4/metadata.py``.
  Callers pass ``topk_indices [N_q, K_max] int32, -1 = invalid`` - the
  same contract used by the vLLM DSV4 PR (``topk_indices_buffer``).

Known performance limitations:

1. One query per kernel call - host loops over N_q.
2. Dense host-gather tensor is ``N_q * K * d`` bf16; at K=640, d=512 this
   is 640 KiB per query.
3. No fused q*k matmul across layers.

These are addressed by the device-resident query and paged-KV path below.

Kernel layout (why this shape works on Trn2):

- ``q_T``: partition=d (contract), free=h. d tiled in 128-blocks.
- ``k_T``: partition=d, free=K.
- ``v_P``: partition=K, free=d (used for pv via K-contract).
- ``mask``, ``sink`` are 2D ``[1, *]`` so all inputs are at least 2D
  (Trn2 convention).

The kernel is declared with ``@nki.jit`` at module scope. A nested
``@nki.jit`` inside a closure triggered an HLO ``args_item`` name
collision ICE during initial compilation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import ml_dtypes
import numpy as np

from nkipy_serving.runtime.device_tensor import get_device_tensor_cls
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge

P_MAX = 128
D_BLOCK = 128
K_TILE = 128

NEG_INF = np.float32(-1e9)


def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def _as_nki_bf16(a: np.ndarray) -> np.ndarray:
    if a.dtype == ml_dtypes.bfloat16:
        return np.ascontiguousarray(a)
    return np.ascontiguousarray(a.astype(ml_dtypes.bfloat16))


# ---------------------------------------------------------------------------
# CPU oracle
# ---------------------------------------------------------------------------


def sparse_mla_attention_oracle(
    q: np.ndarray,  # [N_q, h, d]
    kv_gathered: np.ndarray,  # [N_q, K, d]
    valid_mask: np.ndarray,  # [N_q, K]     bool
    attn_sink: np.ndarray,  # [h]
    softmax_scale: float,
) -> np.ndarray:
    """Reference implementation matching eager ``_sparse_attn``."""
    if q.ndim != 3:
        raise ValueError(f"q must be [N_q, h, d], got {q.shape}")
    if kv_gathered.shape[0] != q.shape[0]:
        raise ValueError(
            f"kv_gathered N_q mismatch: q={q.shape[0]}, "
            f"kv_gathered={kv_gathered.shape[0]}"
        )
    if valid_mask.shape != kv_gathered.shape[:2]:
        raise ValueError(
            f"valid_mask must be [N_q, K]={kv_gathered.shape[:2]}, "
            f"got {valid_mask.shape}"
        )
    if attn_sink.shape != (q.shape[1],):
        raise ValueError(f"attn_sink must be [h]=[{q.shape[1]}], got {attn_sink.shape}")

    qf = q.astype(np.float32)
    kvf = kv_gathered.astype(np.float32)
    scores = np.einsum("nhd,nkd->nhk", qf, kvf) * np.float32(softmax_scale)
    scores = np.where(valid_mask[:, None, :], scores, NEG_INF)
    any_valid = valid_mask.any(axis=-1)
    with np.errstate(over="ignore", invalid="ignore"):
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        e = np.where(valid_mask[:, None, :], e, 0.0)
        sink_e = np.exp(attn_sink[None, :, None] - m)
        denom = e.sum(axis=-1, keepdims=True) + sink_e
        p = e / denom
    out = np.einsum("nhk,nkd->nhd", p, kvf)
    return np.where(any_valid[:, None, None], out, 0.0)


# ---------------------------------------------------------------------------
# Host-side gather
# ---------------------------------------------------------------------------


def gather_kv_and_mask(
    kv: np.ndarray,  # [N_kv, d]
    topk_idxs: np.ndarray,  # [N_q, K_max] int
) -> tuple[np.ndarray, np.ndarray]:
    if kv.ndim != 2:
        raise ValueError(f"kv must be [N_kv, d], got {kv.shape}")
    if topk_idxs.ndim != 2:
        raise ValueError(f"topk_idxs must be [N_q, K_max], got {topk_idxs.shape}")
    N_q, K_max = topk_idxs.shape
    N_kv, d = kv.shape
    valid_mask = topk_idxs >= 0
    safe_idxs = np.where(valid_mask, topk_idxs, 0).astype(np.int64)
    if N_kv == 0:
        gathered = np.zeros((N_q, K_max, d), dtype=kv.dtype)
    else:
        gathered = kv[safe_idxs]
        gathered = np.where(
            valid_mask[..., None],
            gathered,
            np.zeros_like(gathered),
        )
    return gathered, valid_mask


# ---------------------------------------------------------------------------
# NKI kernel (module-level @nki.jit, imported lazily)
# ---------------------------------------------------------------------------


try:
    import neuronxcc.nki as _nki
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    from neuronxcc.nki.language import par_dim

    _NKI_AVAILABLE = True
except ImportError:
    _nki = None
    nisa = None
    nl = None
    nt = None
    par_dim = None
    _NKI_AVAILABLE = False


if _NKI_AVAILABLE:

    @_nki.jit
    def _sparse_attn_single_query_kernel(q_T, k_T, v_P, mask, sink):
        """Single-query sparse MLA attention (see module docstring)."""
        d = q_T.shape[0]
        h = q_T.shape[1]
        K = k_T.shape[1]
        n_d = d // D_BLOCK

        # (1) qk[h, K] = sum over d-blocks of q_tile^T @ k_tile.
        qk_psum = nl.zeros((par_dim(h), K), dtype=nl.float32, buffer=nl.psum)
        for db in nl.affine_range(n_d):
            q_sb = nl.load(q_T[nl.ds(db * D_BLOCK, D_BLOCK), :])
            k_sb = nl.load(k_T[nl.ds(db * D_BLOCK, D_BLOCK), :])
            qk_psum[...] += nisa.nc_matmul(q_sb, k_sb)

        # (2) Mask: qk_masked = qk + (mask - 1) * 1e9.
        mask_sb = nl.load(mask)
        bias_row = nl.ndarray((par_dim(1), K), dtype=nl.float32, buffer=nl.sbuf)
        bias_row[...] = nisa.tensor_scalar(
            data=mask_sb,
            op0=nl.subtract,
            operand0=nl.float32(1.0),
            op1=nl.multiply,
            operand1=nl.float32(1e9),
            dtype=nl.float32,
        )
        bias_bcast = nl.broadcast_to(bias_row, shape=(h, K))
        qk_masked = nl.ndarray((par_dim(h), K), dtype=nl.float32, buffer=nl.sbuf)
        qk_masked[...] = nl.add(qk_psum, bias_bcast)

        # (3) Row max.
        m_row = nisa.tensor_reduce(
            np.max,
            qk_masked,
            axis=(1,),
            dtype=nl.float32,
            negate=False,
        )
        neg_m = nisa.activation(nl.copy, m_row, scale=-1.0)

        # (4) exp(qk - m), row-sum l_row.
        p_fp32 = nl.ndarray((par_dim(h), K), dtype=nl.float32, buffer=nl.sbuf)
        l_row = nl.ndarray((par_dim(h), 1), dtype=nl.float32, buffer=nl.sbuf)
        p_fp32[...] = nisa.activation_reduce(
            np.exp,
            qk_masked,
            bias=neg_m,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=l_row,
            dtype=nl.float32,
        )

        # (5) Sink denom: denom = l + exp(sink - m).
        sink_sb_1h = nl.load(sink)
        sink_on_part_psum = nl.ndarray(
            (par_dim(h), 1),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        sink_on_part_psum[...] = nisa.nc_transpose(
            sink_sb_1h,
            engine=nisa.tensor_engine,
        )
        sink_on_part = nl.ndarray(
            (par_dim(h), 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        sink_on_part[...] = nl.copy(sink_on_part_psum)
        sink_term = nisa.activation(
            np.exp,
            sink_on_part,
            bias=neg_m,
            scale=1.0,
        )
        denom = nl.add(l_row, sink_term)

        # (6) Cast p to bf16.
        p_bf = nl.ndarray((par_dim(h), K), dtype=nl.bfloat16, buffer=nl.sbuf)
        p_bf[...] = nl.copy(p_fp32, dtype=nl.bfloat16)

        # (7) Transpose p_bf [h, K] -> p_T [K, h].
        p_T_psum = nl.ndarray(
            (par_dim(K), h),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        p_T_psum[...] = nisa.nc_transpose(p_bf, engine=nisa.tensor_engine)
        p_T = nl.ndarray((par_dim(K), h), dtype=nl.bfloat16, buffer=nl.sbuf)
        p_T[...] = nl.copy(p_T_psum, dtype=nl.bfloat16)

        # (8) pv: for each d-block, o_block[h, db] = p_T @ v[:, db].
        out = nl.ndarray((h, d), dtype=nl.float32, buffer=nl.shared_hbm)
        inv_denom = nl.ndarray(
            (par_dim(h), 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        inv_denom[...] = nl.divide(nl.float32(1.0), denom)

        for db in nl.affine_range(n_d):
            v_sb = nl.load(v_P[:, nl.ds(db * D_BLOCK, D_BLOCK)])
            pv_psum = nl.zeros(
                (par_dim(h), D_BLOCK),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            pv_psum[...] = nisa.nc_matmul(p_T, v_sb)
            inv_bcast = nl.broadcast_to(inv_denom, shape=(h, D_BLOCK))
            nl.store(
                out[:, nl.ds(db * D_BLOCK, D_BLOCK)],
                nl.multiply(pv_psum, inv_bcast),
            )
        return out


if _NKI_AVAILABLE:

    @_nki.jit
    def _sparse_attn_batched_kernel(q_T, k_T, v_P, mask, sink):
        """Batched sparse MLA attention over ``B`` queries.

        Inputs (partition-axis-first layout):
        - ``q_T``:  [B, d, h]   bf16.
        - ``k_T``:  [B, d, K]   bf16 (pre-scaled).
        - ``v_P``:  [B, K, d]   bf16.
        - ``mask``: [B, K]      bf16 (0/1). Partition=B(<=128) then free=K
          so the trn2 load works without the inner (1, K) convention used by
          the single-query kernel.
        - ``sink``: [1, h]      fp32 — same for all B queries; broadcast
          to partition once via nc_transpose.

        Returns ``[B, h, d]`` fp32.

        Why this shape works:
        - qk/pv matmuls stay at single-query size ([h, K] and [h, d_block]).
        - Outer ``for bi in affine_range(B)`` loop lets the compiler
          pipeline loads across queries while keeping partition sizes
          constant.
        - Sink → partition transpose moves once outside the B loop,
          saving B-1 transpose ops versus the per-call kernel.

        Performance note: on Trn2 this amortises the ~4.4 ms NEFF execute
        overhead across B queries instead of paying it B times. Each B
        value needs its own compiled NEFF (shape-keyed cache in the
        runtime wrapper); typical V4 serving sees at most a handful of
        distinct B values per layer so compile cost stays bounded.
        """
        B = q_T.shape[0]
        d = q_T.shape[1]
        h = q_T.shape[2]
        K = k_T.shape[2]
        n_d = d // D_BLOCK

        # Sink → partition axis h once.
        sink_sb_1h = nl.load(sink)
        sink_on_part_psum = nl.ndarray(
            (par_dim(h), 1),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        sink_on_part_psum[...] = nisa.nc_transpose(
            sink_sb_1h,
            engine=nisa.tensor_engine,
        )
        sink_on_part = nl.ndarray(
            (par_dim(h), 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        sink_on_part[...] = nl.copy(sink_on_part_psum)

        out = nl.ndarray((B, h, d), dtype=nl.float32, buffer=nl.shared_hbm)

        for bi in nl.affine_range(B):
            # (1) qk[h, K] = sum over d-blocks.
            qk_psum = nl.zeros((par_dim(h), K), dtype=nl.float32, buffer=nl.psum)
            for db in nl.affine_range(n_d):
                q_sb = nl.load(q_T[bi, nl.ds(db * D_BLOCK, D_BLOCK), :])
                k_sb = nl.load(k_T[bi, nl.ds(db * D_BLOCK, D_BLOCK), :])
                qk_psum[...] += nisa.nc_matmul(q_sb, k_sb)

            # (2) mask bias. mask[bi] is shape [K] on a single partition row.
            mask_sb = nl.load(mask[bi : bi + 1, :])
            bias_row = nl.ndarray((par_dim(1), K), dtype=nl.float32, buffer=nl.sbuf)
            bias_row[...] = nisa.tensor_scalar(
                data=mask_sb,
                op0=nl.subtract,
                operand0=nl.float32(1.0),
                op1=nl.multiply,
                operand1=nl.float32(1e9),
                dtype=nl.float32,
            )
            bias_bcast = nl.broadcast_to(bias_row, shape=(h, K))
            qk_masked = nl.ndarray((par_dim(h), K), dtype=nl.float32, buffer=nl.sbuf)
            qk_masked[...] = nl.add(qk_psum, bias_bcast)

            # (3) row max.
            m_row = nisa.tensor_reduce(
                np.max,
                qk_masked,
                axis=(1,),
                dtype=nl.float32,
                negate=False,
            )
            neg_m = nisa.activation(nl.copy, m_row, scale=-1.0)

            # (4) exp + row-sum.
            p_fp32 = nl.ndarray((par_dim(h), K), dtype=nl.float32, buffer=nl.sbuf)
            l_row = nl.ndarray((par_dim(h), 1), dtype=nl.float32, buffer=nl.sbuf)
            p_fp32[...] = nisa.activation_reduce(
                np.exp,
                qk_masked,
                bias=neg_m,
                scale=1.0,
                reduce_op=nl.add,
                reduce_res=l_row,
                dtype=nl.float32,
            )

            # (5) denom = l + exp(sink - m).
            sink_term = nisa.activation(
                np.exp,
                sink_on_part,
                bias=neg_m,
                scale=1.0,
            )
            denom = nl.add(l_row, sink_term)

            # (6) cast p to bf16.
            p_bf = nl.ndarray((par_dim(h), K), dtype=nl.bfloat16, buffer=nl.sbuf)
            p_bf[...] = nl.copy(p_fp32, dtype=nl.bfloat16)

            # (7) transpose p [h, K] -> p_T [K, h].
            p_T_psum = nl.ndarray(
                (par_dim(K), h),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            p_T_psum[...] = nisa.nc_transpose(p_bf, engine=nisa.tensor_engine)
            p_T = nl.ndarray((par_dim(K), h), dtype=nl.bfloat16, buffer=nl.sbuf)
            p_T[...] = nl.copy(p_T_psum, dtype=nl.bfloat16)

            # (8) pv per d-block.
            inv_denom = nl.ndarray(
                (par_dim(h), 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            inv_denom[...] = nl.divide(nl.float32(1.0), denom)
            for db in nl.affine_range(n_d):
                v_sb = nl.load(v_P[bi, :, nl.ds(db * D_BLOCK, D_BLOCK)])
                pv_psum = nl.zeros(
                    (par_dim(h), D_BLOCK),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                pv_psum[...] = nisa.nc_matmul(p_T, v_sb)
                inv_bcast = nl.broadcast_to(inv_denom, shape=(h, D_BLOCK))
                nl.store(
                    out[bi, :, nl.ds(db * D_BLOCK, D_BLOCK)],
                    nl.multiply(pv_psum, inv_bcast),
                )

        return out


if _NKI_AVAILABLE:

    @_nki.jit
    def _paged_kv_write_slots_kernel(
        kv_buffer: "nt.tensor[nt.mutable]",
        rows: "nt.tensor",
        slot_idxs: "nt.tensor",
    ):
        """In-place scatter write into the persistent KV buffer.

        Inputs:
        - ``kv_buffer``: [N_kv_max, d] bf16, mutable. Persistent DeviceTensor.
        - ``rows``:      [n_new, d]    bf16. Fresh KV rows to write.
        - ``slot_idxs``: [n_new]       int32. Target positions in ``kv_buffer``.

        Writes ``kv_buffer[slot_idxs[i]] = rows[i]`` for i in [0, n_new).

        Layout matches ``nki_paged_kv_cache.update_kv_cache``: slots and
        rows live on the partition axis (up to 128 per tile), d on free.
        """
        n_new, d = rows.shape
        MAX_T = 128
        n_tiles = (n_new + MAX_T - 1) // MAX_T

        slot_2d = slot_idxs.reshape((n_new, 1))

        for ti in nl.affine_range(n_tiles):
            if n_new <= MAX_T:
                cur = n_new
                t0 = 0
            else:
                cur = MAX_T
                t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(d)[None, :]

            slot_sb = nl.load(slot_2d[t0 : t0 + cur])  # [cur, 1]
            rows_sb = nl.load(rows[t0 : t0 + cur])  # [cur, d]

            nl.store(
                dst=kv_buffer[slot_sb[i_p, 0], i_f],
                value=rows_sb[i_p, i_f],
            )
        return kv_buffer

    @_nki.jit
    def _sparse_attn_batched_paged_kernel(
        q_T,
        kv_hbm,
        topk_T,
        mask,
        sink,
    ):
        """Batched sparse MLA with in-kernel paged gather.

        Inputs:
        - ``q_T``          : [B, d, h]     bf16. Query, d on partition.
          **Caller must pre-scale q by ``softmax_scale``** (trivial on
          host since q is small; keeps the kernel free of broadcast-scale
          plumbing).
        - ``kv_hbm``       : [N_kv, d]     bf16. Persistent KV buffer
          (flat cache; caller owns the layout).
        - ``topk_T``       : [K, B]        int32. Indices into ``kv_hbm``;
          K on partition for the parallel gather. Invalid slots must be
          safe-clamped to 0 on host — ``mask`` kills them.
        - ``mask``         : [B, K]        bf16 (0/1).
        - ``sink``         : [1, h]        fp32.

        Returns ``[B, h, d]`` fp32.

        Gather pattern (matches blocksparse_flash_attention/paged_cache.py):
        ``nl.load(kv_hbm[topk_sb[i_p, bi], i_f])`` where ``i_p = arange(K)``
        lives on the partition axis and ``topk_sb`` is ``topk_T`` loaded
        into SBUF (NKI requires indirect indices to live in SBUF, not HBM).
        """
        B = q_T.shape[0]
        d = q_T.shape[1]
        h = q_T.shape[2]
        K = topk_T.shape[0]
        n_d = d // D_BLOCK

        # Sink → partition h (once, shared across all B).
        sink_sb_1h = nl.load(sink)
        sink_on_part_psum = nl.ndarray(
            (par_dim(h), 1),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        sink_on_part_psum[...] = nisa.nc_transpose(
            sink_sb_1h,
            engine=nisa.tensor_engine,
        )
        sink_on_part = nl.ndarray(
            (par_dim(h), 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        sink_on_part[...] = nl.copy(sink_on_part_psum)

        # Preload topk into SBUF so the indirect load sees SBUF indices.
        # ``topk_T`` has K on partition; keeping the same layout in SBUF
        # means ``topk_sb[i_p, bi]`` (partition-indexed) is the gather key.
        topk_sb = nl.ndarray(
            (par_dim(K), B),
            dtype=topk_T.dtype,
            buffer=nl.sbuf,
        )
        topk_sb[...] = nl.load(topk_T)

        out = nl.ndarray((B, h, d), dtype=nl.float32, buffer=nl.shared_hbm)

        i_p = nl.arange(K)[:, None]
        i_f = nl.arange(d)[None, :]

        for bi in nl.affine_range(B):
            # ---- In-kernel gather: kv_gathered [K, d] ----
            # Indirect HBM load: each K partition-lane reads one row of
            # ``kv_hbm`` whose index comes from ``topk_sb[i_p, bi]``. The
            # index tensor must be in SBUF (checked by the NKI compiler).
            kv_gathered = nl.ndarray(
                (par_dim(K), d),
                dtype=kv_hbm.dtype,
                buffer=nl.sbuf,
            )
            kv_gathered[i_p, i_f] = nl.load(
                kv_hbm[topk_sb[i_p, bi], i_f],
            )

            # ---- qk[h, K] = sum_d q[d,h] * kv^T[d,K] ----
            # d must sit on partition for nc_matmul contract. kv_gathered
            # has K on partition and d on free, so transpose each d-block
            # (128 wide) separately via nc_transpose and accumulate.
            qk_psum = nl.zeros(
                (par_dim(h), K),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            for db in nl.affine_range(n_d):
                q_sb = nl.load(q_T[bi, nl.ds(db * D_BLOCK, D_BLOCK), :])

                # Transpose kv_gathered[:, db*128:(db+1)*128] [K=128, D_BLOCK=128]
                # → [D_BLOCK, K]. Both dims are 128 so this fits a single
                # transpose on the tensor engine.
                kv_block = kv_gathered[:, nl.ds(db * D_BLOCK, D_BLOCK)]
                kv_block_T_psum = nl.ndarray(
                    (par_dim(D_BLOCK), K),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                kv_block_T_psum[...] = nisa.nc_transpose(
                    kv_block,
                    engine=nisa.tensor_engine,
                )
                k_sb = nl.ndarray(
                    (par_dim(D_BLOCK), K),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                k_sb[...] = nl.copy(kv_block_T_psum, dtype=nl.bfloat16)
                qk_psum[...] += nisa.nc_matmul(q_sb, k_sb)

            # ---- mask + softmax ----
            mask_sb = nl.load(mask[bi : bi + 1, :])
            bias_row = nl.ndarray(
                (par_dim(1), K),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            bias_row[...] = nisa.tensor_scalar(
                data=mask_sb,
                op0=nl.subtract,
                operand0=nl.float32(1.0),
                op1=nl.multiply,
                operand1=nl.float32(1e9),
                dtype=nl.float32,
            )
            bias_bcast = nl.broadcast_to(bias_row, shape=(h, K))
            qk_masked = nl.ndarray(
                (par_dim(h), K),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            qk_masked[...] = nl.add(qk_psum, bias_bcast)

            m_row = nisa.tensor_reduce(
                np.max,
                qk_masked,
                axis=(1,),
                dtype=nl.float32,
                negate=False,
            )
            neg_m = nisa.activation(nl.copy, m_row, scale=-1.0)

            p_fp32 = nl.ndarray(
                (par_dim(h), K),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            l_row = nl.ndarray(
                (par_dim(h), 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            p_fp32[...] = nisa.activation_reduce(
                np.exp,
                qk_masked,
                bias=neg_m,
                scale=1.0,
                reduce_op=nl.add,
                reduce_res=l_row,
                dtype=nl.float32,
            )

            sink_term = nisa.activation(
                np.exp,
                sink_on_part,
                bias=neg_m,
                scale=1.0,
            )
            denom = nl.add(l_row, sink_term)

            p_bf = nl.ndarray(
                (par_dim(h), K),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            p_bf[...] = nl.copy(p_fp32, dtype=nl.bfloat16)

            p_T_psum = nl.ndarray(
                (par_dim(K), h),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            p_T_psum[...] = nisa.nc_transpose(
                p_bf,
                engine=nisa.tensor_engine,
            )
            p_T = nl.ndarray(
                (par_dim(K), h),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            p_T[...] = nl.copy(p_T_psum, dtype=nl.bfloat16)

            # ---- pv per d-block — reuse the K-partition gathered KV ----
            inv_denom = nl.ndarray(
                (par_dim(h), 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            inv_denom[...] = nl.divide(nl.float32(1.0), denom)
            for db in nl.affine_range(n_d):
                # v_P [K, d_block] reused from the gathered (pre-scaled) KV.
                # Note: v uses the UN-scaled KV, not kv_scaled; build an
                # unscaled-v view via a second tensor. Here we re-gather:
                # simpler to just cast kv_gathered (unscaled) to bf16.
                pv_psum = nl.zeros(
                    (par_dim(h), D_BLOCK),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                pv_psum[...] = nisa.nc_matmul(
                    p_T,
                    kv_gathered[:, nl.ds(db * D_BLOCK, D_BLOCK)],
                )
                inv_bcast = nl.broadcast_to(inv_denom, shape=(h, D_BLOCK))
                nl.store(
                    out[bi, :, nl.ds(db * D_BLOCK, D_BLOCK)],
                    nl.multiply(pv_psum, inv_bcast),
                )

        return out


if _NKI_AVAILABLE:

    @_nki.jit
    def _sparse_attn_batched_paged_multiK_kernel(
        q_T,
        kv_hbm,
        topk_T,
        mask,
        sink,
    ):
        """Multi-tile K paged sparse MLA with online softmax.

        Same contract as ``_sparse_attn_batched_paged_kernel`` but
        ``topk_T`` / ``mask`` carry ``K_total = n_k * K_TILE`` entries.
        Per query we walk the K tiles, keeping a running flash-attention
        ``(m, l, acc)`` triple in SBUF and rescaling the accumulator
        whenever a tile's max exceeds the running max. Sink is folded
        into the final denom once all tiles have been consumed.

        Shape constraints:
        - ``K_total % K_TILE == 0``.
        - ``K_TILE == 128`` (fits one partition axis).
        - ``d % D_BLOCK == 0``.
        """
        B = q_T.shape[0]
        d = q_T.shape[1]
        h = q_T.shape[2]
        K_total = topk_T.shape[0]
        K = K_TILE
        n_k = K_total // K
        n_d = d // D_BLOCK

        # Sink → partition h (once, shared across all B).
        sink_sb_1h = nl.load(sink)
        sink_on_part_psum = nl.ndarray(
            (par_dim(h), 1),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        sink_on_part_psum[...] = nisa.nc_transpose(
            sink_sb_1h,
            engine=nisa.tensor_engine,
        )
        sink_on_part = nl.ndarray(
            (par_dim(h), 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        sink_on_part[...] = nl.copy(sink_on_part_psum)

        out = nl.ndarray((B, h, d), dtype=nl.float32, buffer=nl.shared_hbm)

        i_p = nl.arange(K)[:, None]
        i_f = nl.arange(d)[None, :]

        for bi in nl.affine_range(B):
            # Running (m, l, acc). acc is pv accumulator in fp32.
            # Use ndarray + in-place ``[...] =`` so the state is a single
            # SBUF tile updated across K-tile iterations.
            m_state = nl.ndarray(
                (par_dim(h), 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            m_state[...] = nl.full((h, 1), np.float32(-1e30), dtype=nl.float32)
            l_state = nl.ndarray(
                (par_dim(h), 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            l_state[...] = nl.full((h, 1), np.float32(0.0), dtype=nl.float32)
            acc = nl.zeros((par_dim(h), d), dtype=nl.float32, buffer=nl.sbuf)

            for kt in nl.sequential_range(n_k):
                # Load this K-tile's topk indices on the K partition axis.
                topk_sb = nl.ndarray(
                    (par_dim(K), 1),
                    dtype=topk_T.dtype,
                    buffer=nl.sbuf,
                )
                topk_sb[...] = nl.load(
                    topk_T[nl.ds(kt * K, K), bi : bi + 1],
                )

                # Gather this K-tile's KV from HBM.
                kv_gathered = nl.ndarray(
                    (par_dim(K), d),
                    dtype=kv_hbm.dtype,
                    buffer=nl.sbuf,
                )
                kv_gathered[i_p, i_f] = nl.load(
                    kv_hbm[topk_sb[i_p, 0], i_f],
                )

                # qk[h, K] = sum_d q[d,h] * kv^T[d,K].
                qk_psum = nl.zeros(
                    (par_dim(h), K),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                for db in nl.affine_range(n_d):
                    q_sb = nl.load(q_T[bi, nl.ds(db * D_BLOCK, D_BLOCK), :])
                    kv_block = kv_gathered[:, nl.ds(db * D_BLOCK, D_BLOCK)]
                    kv_block_T_psum = nl.ndarray(
                        (par_dim(D_BLOCK), K),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    kv_block_T_psum[...] = nisa.nc_transpose(
                        kv_block,
                        engine=nisa.tensor_engine,
                    )
                    k_sb = nl.ndarray(
                        (par_dim(D_BLOCK), K),
                        dtype=nl.bfloat16,
                        buffer=nl.sbuf,
                    )
                    k_sb[...] = nl.copy(kv_block_T_psum, dtype=nl.bfloat16)
                    qk_psum[...] += nisa.nc_matmul(q_sb, k_sb)

                # Mask this tile.
                mask_sb = nl.load(mask[bi : bi + 1, kt * K : (kt + 1) * K])
                bias_row = nl.ndarray(
                    (par_dim(1), K),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                bias_row[...] = nisa.tensor_scalar(
                    data=mask_sb,
                    op0=nl.subtract,
                    operand0=nl.float32(1.0),
                    op1=nl.multiply,
                    operand1=nl.float32(1e9),
                    dtype=nl.float32,
                )
                bias_bcast = nl.broadcast_to(bias_row, shape=(h, K))
                qk_masked = nl.ndarray(
                    (par_dim(h), K),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                qk_masked[...] = nl.add(qk_psum, bias_bcast)

                # Tile max.
                m_tile = nisa.tensor_reduce(
                    np.max,
                    qk_masked,
                    axis=(1,),
                    dtype=nl.float32,
                    negate=False,
                )
                # New running max = max(m_state, m_tile).
                m_new = nl.ndarray(
                    (par_dim(h), 1),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                m_new[...] = nl.maximum(m_state, m_tile)
                # Rescale factor for previous accumulator: exp(m_state - m_new).
                alpha = nl.ndarray(
                    (par_dim(h), 1),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                alpha[...] = nisa.activation(
                    np.exp,
                    m_state,
                    bias=nisa.activation(
                        nl.copy,
                        m_new,
                        scale=-1.0,
                    ),
                    scale=1.0,
                )
                # exp(qk - m_new).
                neg_m_new = nisa.activation(nl.copy, m_new, scale=-1.0)
                p_fp32 = nl.ndarray(
                    (par_dim(h), K),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                l_tile = nl.ndarray(
                    (par_dim(h), 1),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                p_fp32[...] = nisa.activation_reduce(
                    np.exp,
                    qk_masked,
                    bias=neg_m_new,
                    scale=1.0,
                    reduce_op=nl.add,
                    reduce_res=l_tile,
                    dtype=nl.float32,
                )
                # Update l_state in place: l_state = alpha * l_state + l_tile.
                # NKI can't self-reference l_state on both sides; stash in a
                # temp first.
                l_prev_scaled = nl.ndarray(
                    (par_dim(h), 1),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                l_prev_scaled[...] = nl.multiply(alpha, l_state)
                l_state[...] = nl.add(l_prev_scaled, l_tile)

                # Rescale acc: acc = alpha * acc, then acc += p @ v_tile.
                alpha_bcast_d = nl.broadcast_to(alpha, shape=(h, d))
                acc_scaled = nl.ndarray(
                    (par_dim(h), d),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                acc_scaled[...] = nl.multiply(acc, alpha_bcast_d)
                acc[...] = nl.copy(acc_scaled)

                # Transpose p for pv matmul (K on partition, h on free).
                p_bf = nl.ndarray(
                    (par_dim(h), K),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                p_bf[...] = nl.copy(p_fp32, dtype=nl.bfloat16)
                p_T_psum = nl.ndarray(
                    (par_dim(K), h),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                p_T_psum[...] = nisa.nc_transpose(
                    p_bf,
                    engine=nisa.tensor_engine,
                )
                p_T = nl.ndarray(
                    (par_dim(K), h),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                p_T[...] = nl.copy(p_T_psum, dtype=nl.bfloat16)

                # acc += p @ v, one d-block at a time.
                for db in nl.affine_range(n_d):
                    pv_psum = nl.zeros(
                        (par_dim(h), D_BLOCK),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    pv_psum[...] = nisa.nc_matmul(
                        p_T,
                        kv_gathered[:, nl.ds(db * D_BLOCK, D_BLOCK)],
                    )
                    acc_slice = nl.ndarray(
                        (par_dim(h), D_BLOCK),
                        dtype=nl.float32,
                        buffer=nl.sbuf,
                    )
                    acc_slice[...] = nl.add(
                        acc[:, nl.ds(db * D_BLOCK, D_BLOCK)],
                        pv_psum,
                    )
                    acc[:, nl.ds(db * D_BLOCK, D_BLOCK)] = nl.copy(acc_slice)

                m_state[...] = nl.copy(m_new)

            # Fold sink into the final denom: denom = l_state + exp(sink - m_final).
            sink_term = nisa.activation(
                np.exp,
                sink_on_part,
                bias=nisa.activation(nl.copy, m_state, scale=-1.0),
                scale=1.0,
            )
            denom = nl.add(l_state, sink_term)
            inv_denom = nl.ndarray(
                (par_dim(h), 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            inv_denom[...] = nl.divide(nl.float32(1.0), denom)
            inv_bcast_d = nl.broadcast_to(inv_denom, shape=(h, d))
            nl.store(out[bi, :, :], nl.multiply(acc, inv_bcast_d))

        return out


@lru_cache(maxsize=1)
def _traced_sparse_attn_single_query():
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    ensure_nki_bridge()
    from nkipy.core.trace import NKIPyKernel

    def _entry(q_T, k_T, v_P, mask, sink):
        return _sparse_attn_single_query_kernel(q_T, k_T, v_P, mask, sink)

    return NKIPyKernel.trace(_entry, backend="hlo")


@lru_cache(maxsize=1)
def _traced_sparse_attn_batched():
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    ensure_nki_bridge()
    from nkipy.core.trace import NKIPyKernel

    def _entry(q_T, k_T, v_P, mask, sink):
        return _sparse_attn_batched_kernel(q_T, k_T, v_P, mask, sink)

    return NKIPyKernel.trace(_entry, backend="hlo")


@lru_cache(maxsize=1)
def _traced_sparse_attn_batched_paged():
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    ensure_nki_bridge()
    from nkipy.core.trace import NKIPyKernel

    def _entry(q_T, kv_hbm, topk_T, mask, sink):
        return _sparse_attn_batched_paged_kernel(
            q_T,
            kv_hbm,
            topk_T,
            mask,
            sink,
        )

    return NKIPyKernel.trace(_entry, backend="hlo")


# One-shot compile + many executes.
# ``baremetal_run_traced_kernel`` recompiles and reloads the NEFF on every
# call (1.4 s/call overhead on Trn2), which dominates per-query
# cost. We cache ``_compile_kernel``'s outputs keyed by kernel + input
# shape/dtype tuple and call ``_execute_neff`` directly for the hot path.
# Measured: 1.4 s -> 4.4 ms per call on V4 shape (h=64, d=512, K=128).


_COMPILED_CACHE: dict[tuple, tuple] = {}


def _shape_key(args: tuple) -> tuple:
    return tuple(
        (a.shape, str(a.dtype)) if isinstance(a, np.ndarray) else type(a).__name__
        for a in args
    )


def _compile_once(traced, *args, artifacts_dir: str | None):
    """Compile the kernel once for a given input signature and cache the
    NEFF + IR. Key is (id(traced), shape/dtype tuple)."""
    from nkipy.runtime.execute import _compile_kernel

    key = (id(traced), _shape_key(args))
    hit = _COMPILED_CACHE.get(key)
    if hit is not None:
        return hit
    neff, name, ir, _bound, _orig = _compile_kernel(
        traced,
        *args,
        artifacts_dir=artifacts_dir,
    )
    _COMPILED_CACHE[key] = (neff, name, ir)
    return neff, name, ir


def _run_cached(traced, *args):
    """Execute a pre-compiled kernel with fresh inputs.

    Rebuilds ``boundargs`` + output buffers per call (cheap numpy allocs);
    reuses the compiled NEFF. The assumption is that the caller previously
    ran ``_compile_once(traced, *args)`` for the same shape/dtype signature.
    """
    import inspect

    from nkipy.runtime.execute import _execute_neff

    key = (id(traced), _shape_key(args))
    neff, name, ir = _COMPILED_CACHE[key]

    sig = inspect.signature(traced.func)
    bound = sig.bind(*args)
    bound.apply_defaults()
    originals = {n: a for n, a in bound.arguments.items() if isinstance(a, np.ndarray)}
    for t in ir.outputs:
        bound.arguments[t.name] = np.empty(t.shape, dtype=t.dtype)
    return _execute_neff(neff, name, ir, bound, originals)


def _run_single_query_kernel(
    q: np.ndarray,  # [h, d] bf16
    k: np.ndarray,  # [K, d] bf16 (already scaled)
    v: np.ndarray,  # [K, d] bf16
    mask: np.ndarray,  # [K] bf16
    sink: np.ndarray,  # [h] fp32
    *,
    artifacts_dir: str | Path | None = None,
    warmup: bool = False,
) -> np.ndarray:
    traced = _traced_sparse_attn_single_query()
    q_T = np.ascontiguousarray(q.T)
    k_T = np.ascontiguousarray(k.T)
    v_P = np.ascontiguousarray(v)
    mask_2d = np.ascontiguousarray(mask.reshape(1, -1))
    sink_2d = np.ascontiguousarray(sink.reshape(1, -1))

    art = str(artifacts_dir) if artifacts_dir is not None else None
    if warmup:
        _compile_once(traced, q_T, k_T, v_P, mask_2d, sink_2d, artifacts_dir=art)

    return np.asarray(_run_cached(traced, q_T, k_T, v_P, mask_2d, sink_2d))


# ---------------------------------------------------------------------------
# Runtime wrapper
# ---------------------------------------------------------------------------


def sparse_mla_attention_host_gather(
    q: np.ndarray,  # [N_q, h, d]
    kv: np.ndarray,  # [N_kv, d]
    topk_idxs: np.ndarray,  # [N_q, K_max] int, -1 = invalid
    attn_sink: np.ndarray,  # [h]
    softmax_scale: float,
    *,
    use_device: bool = True,
    artifacts_dir: str | Path | None = None,
) -> np.ndarray:
    """Sparse MLA attention using host-gather + NKI kernel per query.

    ``use_device=False`` runs the CPU oracle.

    Returns ``[N_q, h, d]`` fp32.
    """
    if q.shape[0] != topk_idxs.shape[0]:
        raise ValueError(f"N_q mismatch: q={q.shape[0]} topk_idxs={topk_idxs.shape[0]}")
    gathered, valid_mask = gather_kv_and_mask(kv, topk_idxs)

    if not use_device:
        return sparse_mla_attention_oracle(
            q,
            gathered,
            valid_mask,
            attn_sink,
            softmax_scale,
        )

    N_q, h, d = q.shape
    K_max = topk_idxs.shape[1]
    if K_max % K_TILE:
        raise ValueError(
            f"K_max={K_max} must be a multiple of K_TILE={K_TILE}; pad topk "
            "with -1 on the caller side"
        )
    if K_max != K_TILE:
        raise NotImplementedError(
            f"K_max={K_max} != K_TILE={K_TILE}; use the multi-tile paged kernel"
        )
    if d % D_BLOCK:
        raise NotImplementedError(
            f"d={d} not a multiple of {D_BLOCK}; d must be D_BLOCK-aligned"
        )
    if h > P_MAX:
        raise ValueError(f"h={h} must be <= {P_MAX}")

    q_bf = _as_nki_bf16(q)
    scaled_kv = gathered.astype(np.float32) * np.float32(softmax_scale)
    k_bf = _as_nki_bf16(scaled_kv)
    v_bf = _as_nki_bf16(gathered.astype(np.float32))
    mask_bf = _as_nki_bf16(valid_mask.astype(np.float32))
    sink_2d = np.ascontiguousarray(attn_sink.reshape(1, -1).astype(np.float32))

    # Fast path: one kernel call for all N_q queries.
    q_T = np.ascontiguousarray(
        q_bf.transpose(0, 2, 1)  # [N_q, h, d] -> [N_q, d, h]
    )
    k_T = np.ascontiguousarray(
        k_bf.transpose(0, 2, 1)  # [N_q, K, d] -> [N_q, d, K]
    )
    v_P = np.ascontiguousarray(v_bf)  # [N_q, K, d]
    mask_2d = np.ascontiguousarray(mask_bf)  # [N_q, K]

    traced = _traced_sparse_attn_batched()
    art = str(artifacts_dir) if artifacts_dir is not None else None
    _compile_once(traced, q_T, k_T, v_P, mask_2d, sink_2d, artifacts_dir=art)
    out = np.asarray(_run_cached(traced, q_T, k_T, v_P, mask_2d, sink_2d))
    # Kernel returns [N_q, h, d] — no reshape needed.
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Device-resident KV + in-kernel paged gather
# ---------------------------------------------------------------------------


class PagedKVBuffer:
    """Persistent device-resident KV buffer.

    Owns a ``DeviceTensor`` of shape ``[N_kv_max, d]`` bf16 that stays live
    across ``sparse_mla_attention_paged`` calls. The host uploads the
    current KV into the buffer (one DMA per forward step) and passes the
    buffer handle to the kernel; the kernel gathers directly from HBM via
    indirect loads, avoiding the per-query host ``np.take`` used by the
    host-gather wrappers.

    This is the first non-weight long-lived DeviceTensor in the stack.
    The class owns its lifetime; a ``close()`` method drops the handle so
    tests can keep memory use bounded.

    Note: for now ``upload()`` rebuilds the DeviceTensor from numpy. A
    ``write_slots(slot_idxs, rows)`` writes into the existing buffer in place,
    which removes the full re-upload and lets the sliding-window KV state live
    on device.
    """

    def __init__(self, n_kv_max: int, d: int) -> None:
        self._n_kv_max = int(n_kv_max)
        self._d = int(d)
        self._device_tensor = None
        self._host_shape: tuple[int, int] | None = None

    @property
    def n_kv_max(self) -> int:
        return self._n_kv_max

    @property
    def d(self) -> int:
        return self._d

    @property
    def tensor(self):
        """The underlying DeviceTensor. ``None`` before ``upload``."""
        return self._device_tensor

    def upload(self, kv_host: np.ndarray) -> None:
        """Copy ``kv_host`` (must be bf16 ``[N, d]`` with N <= n_kv_max)
        into the persistent device buffer, padding unused rows with zeros.

        Each call currently allocates a fresh DeviceTensor because
        ``DeviceTensor.from_numpy`` is the only available constructor in
        this environment. ``write_slots`` is the in-place update path for
        incremental decode.
        """
        if kv_host.ndim != 2 or kv_host.shape[1] != self._d:
            raise ValueError(f"kv_host must be [N, d={self._d}], got {kv_host.shape}")
        if kv_host.shape[0] > self._n_kv_max:
            raise ValueError(
                f"kv_host N={kv_host.shape[0]} exceeds n_kv_max={self._n_kv_max}"
            )
        if kv_host.dtype != ml_dtypes.bfloat16:
            kv_host = kv_host.astype(ml_dtypes.bfloat16)
        # Pad to n_kv_max rows so the kernel's compile-time shape is stable.
        padded = np.zeros((self._n_kv_max, self._d), dtype=ml_dtypes.bfloat16)
        padded[: kv_host.shape[0]] = kv_host
        self._device_tensor = get_device_tensor_cls().from_numpy(
            np.ascontiguousarray(padded),
            name="paged_kv_buffer",
        )
        self._host_shape = kv_host.shape

    def close(self) -> None:
        self._device_tensor = None
        self._host_shape = None

    def write_slots(
        self,
        slot_idxs: np.ndarray,  # [n_new] int32
        rows: np.ndarray,  # [n_new, d]
        *,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        """In-place scatter write: ``self.tensor[slot_idxs[i]] = rows[i]``.

        Unlike ``upload``, this does NOT replace the buffer — it scatters
        ``n_new`` rows at caller-provided slot indices. Use this for
        incremental decode where each step only writes the newly-arrived
        KV rows and leaves the rest of the cache untouched.

        ``self.tensor`` must already be live (call ``upload`` once with a
        zero-filled or prior-state KV to allocate). The write is performed
        via an NKI scatter kernel that keeps the buffer device-resident.

        Shape constraints:
        - ``n_new <= 128`` (single tile) OR ``n_new % 128 == 0`` (multi-tile).
        - ``rows.shape == (n_new, self.d)``.
        - ``slot_idxs`` values in ``[0, self.n_kv_max)``; caller's
          responsibility.
        """
        if self._device_tensor is None:
            raise RuntimeError(
                "PagedKVBuffer has no data — call .upload() once first to "
                "allocate the device tensor (a zero-filled upload is fine)"
            )
        if slot_idxs.ndim != 1:
            raise ValueError(f"slot_idxs must be [n_new], got {slot_idxs.shape}")
        n_new = int(slot_idxs.shape[0])
        if rows.shape != (n_new, self._d):
            raise ValueError(f"rows must be [{n_new}, d={self._d}], got {rows.shape}")
        if n_new == 0:
            return
        MAX_T = 128
        if n_new > MAX_T and n_new % MAX_T != 0:
            raise NotImplementedError(
                f"n_new={n_new}: only n_new <= {MAX_T} or n_new % {MAX_T} "
                "== 0 supported today"
            )

        # bf16 rows + int32 slots.
        rows_bf = (
            np.ascontiguousarray(rows)
            if rows.dtype == ml_dtypes.bfloat16
            else np.ascontiguousarray(rows.astype(ml_dtypes.bfloat16))
        )
        slots_i32 = np.ascontiguousarray(slot_idxs.astype(np.int32))

        from nkipy.runtime import DeviceKernel

        DeviceTensor = get_device_tensor_cls()

        cache_key = (
            "write_slots",
            (self._n_kv_max, self._d),
            (n_new, self._d),
            (n_new,),
        )
        kernel = _PAGED_KERNEL_CACHE.get(cache_key)
        if kernel is None:
            zeros_kv = np.zeros(
                (self._n_kv_max, self._d),
                dtype=ml_dtypes.bfloat16,
            )
            kernel = DeviceKernel.compile_and_load(
                _paged_kv_write_slots_entry,
                zeros_kv,
                rows_bf,
                slots_i32,
                name=f"dsv4_paged_kv_write_slots_nnew{n_new}_d{self._d}",
                build_dir=(str(artifacts_dir) if artifacts_dir is not None else None),
                use_cached_if_exists=True,
            )
            _PAGED_KERNEL_CACHE[cache_key] = kernel

        rows_dev = DeviceTensor.from_numpy(rows_bf, name="rows")
        slots_dev = DeviceTensor.from_numpy(slots_i32, name="slot_idxs")

        # The kernel aliases kv_buffer as both input and output (in-place).
        # The `must_alias_input` name matches NKI's mutable tensor convention
        # used in nki_blocksparse_flash_attention.
        kernel(
            inputs={
                "kv_buffer.must_alias_input": self._device_tensor,
                "rows": rows_dev,
                "slot_idxs": slots_dev,
            },
            outputs={"kv_buffer": self._device_tensor},
        )


_PAGED_KERNEL_CACHE: dict[tuple, object] = {}


def _paged_kv_write_slots_entry(kv_buffer, rows, slot_idxs):
    """Module-scope entry wrapper for DeviceKernel.compile_and_load."""
    return _paged_kv_write_slots_kernel(kv_buffer, rows, slot_idxs)


def _sparse_attn_batched_paged_entry(q_T, kv_hbm, topk_T, mask, sink):
    """Module-scope wrapper so DeviceKernel.compile_and_load can
    `inspect.getsource` it and produce a FrameworkKernel that accepts
    named DeviceTensor inputs at call time."""
    return _sparse_attn_batched_paged_kernel(q_T, kv_hbm, topk_T, mask, sink)


def _sparse_attn_batched_paged_multiK_entry(q_T, kv_hbm, topk_T, mask, sink):
    return _sparse_attn_batched_paged_multiK_kernel(
        q_T,
        kv_hbm,
        topk_T,
        mask,
        sink,
    )


def sparse_mla_attention_paged(
    q: np.ndarray,  # [N_q, h, d]
    kv_buffer: PagedKVBuffer,  # persistent device buffer
    topk_idxs: np.ndarray,  # [N_q, K] int, -1 = invalid
    attn_sink: np.ndarray,  # [h]
    softmax_scale: float,
    *,
    artifacts_dir: str | Path | None = None,
) -> np.ndarray:
    """Sparse MLA using a persistent device KV buffer.

    Zero-copy contract: the ``kv_buffer.tensor`` ``DeviceTensor`` is fed
    directly into the compiled kernel via the
    ``DeviceKernel.compile_and_load(...)`` → ``kernel(inputs={...})``
    path (same pattern as ``nki_blocksparse_flash_attention``). No host
    round-trip on the KV. The kernel gathers rows out of the buffer via
    an indirect HBM load; invalid slots (-1) are safe-clamped to 0 on
    host and masked to ``-inf`` in the softmax.

    Caller-facing limitations (tracked elsewhere):
    - KV buffer content can be initialized via ``PagedKVBuffer.upload`` or
      updated incrementally via ``write_slots``.
    - Single K tile (``K == K_TILE == 128``); multi-tile K is a future
      kernel variant.
    - Shape envelope: d % 128 == 0 and h <= 128.
    """
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    if kv_buffer.tensor is None:
        raise RuntimeError("PagedKVBuffer has no data — call .upload(kv_host) first")
    if q.shape[0] != topk_idxs.shape[0]:
        raise ValueError(f"N_q mismatch: q={q.shape[0]} topk={topk_idxs.shape[0]}")
    N_q, h, d = q.shape
    K = topk_idxs.shape[1]
    if K != K_TILE:
        raise NotImplementedError(f"K={K} != K_TILE={K_TILE}; multi-tile K deferred")
    if d % D_BLOCK:
        raise NotImplementedError(f"d={d} not a multiple of {D_BLOCK}")
    if h > P_MAX:
        raise ValueError(f"h={h} must be <= {P_MAX}")
    if d != kv_buffer.d:
        raise ValueError(f"d mismatch: q={d}, kv_buffer={kv_buffer.d}")

    # Safe-clamp -1 indices so the indirect HBM load never addresses out
    # of bounds; mask kills them in the softmax.
    valid = topk_idxs >= 0
    safe = np.where(valid, topk_idxs, 0).astype(np.int32)
    topk_T = np.ascontiguousarray(safe.T)  # [K, N_q]
    mask_bf = _as_nki_bf16(valid.astype(np.float32))  # [N_q, K]

    # Pre-scale q by softmax_scale on host.
    q_scaled = q.astype(np.float32) * np.float32(softmax_scale)
    q_bf = _as_nki_bf16(q_scaled)
    q_T = np.ascontiguousarray(q_bf.transpose(0, 2, 1))  # [N_q, d, h]
    sink_2d = np.ascontiguousarray(
        attn_sink.reshape(1, -1).astype(np.float32)
    )  # [1, h]

    # Compile-once cached on the input signature (including kv buffer shape).
    from nkipy.runtime import DeviceKernel

    DeviceTensor = get_device_tensor_cls()
    cache_key = (
        tuple(q_T.shape),
        (kv_buffer.n_kv_max, d),
        tuple(topk_T.shape),
        tuple(mask_bf.shape),
        tuple(sink_2d.shape),
    )
    kernel = _PAGED_KERNEL_CACHE.get(cache_key)
    if kernel is None:
        zeros_kv = np.zeros(
            (kv_buffer.n_kv_max, d),
            dtype=ml_dtypes.bfloat16,
        )
        kernel = DeviceKernel.compile_and_load(
            _sparse_attn_batched_paged_entry,
            q_T,
            zeros_kv,
            topk_T,
            mask_bf,
            sink_2d,
            name=f"dsv4_sparse_attn_paged_nq{N_q}_d{d}_k{K}",
            build_dir=(str(artifacts_dir) if artifacts_dir is not None else None),
            use_cached_if_exists=True,
        )
        _PAGED_KERNEL_CACHE[cache_key] = kernel

    # Upload the per-call host tensors (q / topk / mask / sink) to device.
    q_dev = DeviceTensor.from_numpy(q_T, name="q_T")
    topk_dev = DeviceTensor.from_numpy(topk_T, name="topk_T")
    mask_dev = DeviceTensor.from_numpy(mask_bf, name="mask")
    sink_dev = DeviceTensor.from_numpy(sink_2d, name="sink")
    out_np = np.zeros((N_q, h, d), dtype=np.float32)
    out_dev = DeviceTensor.from_numpy(out_np, name="paged_out")

    # Zero-copy: kv_hbm is the live PagedKVBuffer.tensor.
    kernel(
        inputs={
            "q_T": q_dev,
            "kv_hbm": kv_buffer.tensor,
            "topk_T": topk_dev,
            "mask": mask_dev,
            "sink": sink_dev,
        },
        outputs={"output0": out_dev},
    )
    return out_dev.numpy()
