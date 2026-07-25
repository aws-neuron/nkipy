"""EagerExecutorBase: shared infrastructure for eager executors.

Provides NKI step input management, LM head dispatch, KV cache flush,
and CPU attention fallback logic.
"""

from __future__ import annotations

import numpy as np

from nkipy_serving.attention.base import AttentionMetadata
from nkipy_serving.attention.nki_blocksparse_flash_attention import NKI_MIN_Q_SEQLEN
from nkipy_serving.attention.nki_step_inputs import (
    PreparedNkiStepInputs,
    allocate_prepared_nki_step_inputs,
    initialize_prepared_nki_step_inputs,
    prepare_prepared_nki_step_inputs,
)
from nkipy_serving.models._device_utils import (
    _get_device_tensor_cls,
)
from nkipy_serving.models._device_utils import (
    alloc_device_scratch as _alloc_device_scratch,
)
from nkipy_serving.models._device_utils import (
    flush_device_kv_cache as _flush_device_kv_cache,
)
from nkipy_serving.models.reload_utils import (
    overwrite_device_tensor as _overwrite_device_tensor,
)
from nkipy_serving.ops.moe.blockwise_index import (
    BLOCK_SIZE as MOE_BLOCK_SIZE,
)
from nkipy_serving.ops.moe.blockwise_index import (
    get_n_blocks as moe_get_n_blocks,
)
from nkipy_serving.sampling.device_batch import DeviceSamplingBatch


class EagerExecutorBase:
    """Shared infrastructure for all eager executors.

    Subclasses must set:
      - ``self._kv_cache_dev``: list of per-layer device KV cache tensors
      - ``self._kv_cache_zeros``: numpy zeros array for flush
      - ``self._kv_pool``: KV pool with .block_size, .num_blocks, .clear()
      - ``self._nki_num_blocks``: kv_pool.num_blocks + 1
      - ``self._shared_tensors``: dict with "final_norm", "lm_head"
      - ``self._logits_processor``: LogitsProcessor instance
      - ``self._max_requests_per_step``: int
      - ``self._runtime_config``: with .max_context_len
      - ``self._compiler_args``: str
    """

    def _init_lm_head_scratch(self, max_bs: int, prefix: str = "eager") -> None:
        """Allocate last_token_indices device buffer and host mirror."""
        self._last_token_indices_dev = _alloc_device_scratch(
            (max_bs,),
            np.int32,
            name=f"{prefix}_last_idx",
        )
        self._last_token_indices_host = np.zeros((max_bs,), dtype=np.int32)

    def _init_nki_step_inputs_cache(self) -> None:
        self._nki_step_inputs_by_bucket: dict[int, PreparedNkiStepInputs] = {}
        self._prefill_scratch_by_bucket: dict[int, dict[str, object]] = {}

    @staticmethod
    def _require_single_rank_for_forward_cpu(
        tp_degree: int,
        ep_degree: int | None = None,
    ) -> None:
        """Reject sharded weights in ``forward_cpu``.

        ``forward_cpu`` operates on rank-local weights. On TP>1 or EP>1 it
        would silently return partial (vocab shard, expert shard, KV-head
        shard) output — a dangerous accuracy-debug hazard.
        """
        if int(tp_degree) != 1 or (ep_degree is not None and int(ep_degree) != 1):
            ep_clause = " and ep_degree == 1" if ep_degree is not None else ""
            got = f"tp_degree={tp_degree}"
            if ep_degree is not None:
                got += f", ep_degree={ep_degree}"
            raise ValueError(
                f"forward_cpu requires tp_degree == 1{ep_clause}; got {got}. "
                "Sharded weights would produce rank-local partial output."
            )

    # -- MoE prefill scratch (per-bucket) ----------------------------------------

    def _ensure_prefill_scratch(
        self,
        token_bucket: int,
        *,
        prefix: str,
    ) -> dict[str, object]:
        """Per-bucket device tensors for blockwise MoE prefill dispatch.

        Allocates token_pos, block_to_expert, moe_out, plus a host-side zero
        array used to clear moe_out before each MoE kernel call (the blockwise
        kernel does scatter-add).
        """
        cached = self._prefill_scratch_by_bucket.get(int(token_bucket))
        if cached is not None:
            return cached
        w = self._weights
        num_blocks, _ = moe_get_n_blocks(
            int(token_bucket),
            int(w.experts_per_token),
            int(w.local_num_experts),
        )
        DT = _get_device_tensor_cls()
        scr = {
            "token_pos": DT.from_numpy(
                np.zeros((num_blocks, MOE_BLOCK_SIZE), dtype=np.int32),
                name=f"{prefix}_token_pos_b{token_bucket}",
            ),
            "block_to_expert": DT.from_numpy(
                np.zeros((num_blocks,), dtype=np.int8),
                name=f"{prefix}_b2e_b{token_bucket}",
            ),
            "moe_out": DT.from_numpy(
                np.zeros((token_bucket, int(w.hidden_size)), dtype=w.dtype),
                name=f"{prefix}_moe_out_b{token_bucket}",
            ),
            "moe_out_zero": np.zeros(
                (token_bucket, int(w.hidden_size)),
                dtype=w.dtype,
            ),
        }
        self._prefill_scratch_by_bucket[int(token_bucket)] = scr
        return scr

    # -- KV cache ----------------------------------------------------------------

    def flush_cache(self) -> None:
        _flush_device_kv_cache(self._kv_cache_dev, self._kv_cache_zeros, self._kv_pool)

    # -- NKI step inputs ---------------------------------------------------------

    def _ensure_nki_step_inputs(
        self,
        token_bucket: int,
        *,
        prefix: str = "eager",
    ) -> PreparedNkiStepInputs:
        cached = self._nki_step_inputs_by_bucket.get(int(token_bucket))
        if cached is not None:
            return cached

        attn_bucket = max(int(token_bucket), int(NKI_MIN_Q_SEQLEN))
        step_inputs = allocate_prepared_nki_step_inputs(
            _alloc_device_scratch,
            token_bucket=int(token_bucket),
            attn_bucket=int(attn_bucket),
            max_context_len=int(self._runtime_config.max_context_len),
            max_requests=self._max_requests_per_step,
            num_blocks=self._nki_num_blocks,
            block_size=int(self._kv_pool.block_size),
            prefix=prefix,
        )
        initialize_prepared_nki_step_inputs(step_inputs, _overwrite_device_tensor)
        self._nki_step_inputs_by_bucket[int(token_bucket)] = step_inputs
        return step_inputs

    def _prepare_nki_step_inputs(
        self,
        *,
        token_bucket: int,
        real_total_tokens: int,
        attn_metadata: AttentionMetadata,
    ) -> dict[str, object]:
        step_inputs = self._ensure_nki_step_inputs(int(token_bucket))
        return prepare_prepared_nki_step_inputs(
            step_inputs,
            _overwrite_device_tensor,
            attn_metadata=attn_metadata,
            real_total_tokens=int(real_total_tokens),
            num_blocks=self._nki_num_blocks,
            block_size=int(self._kv_pool.block_size),
        )

    # -- LM head -----------------------------------------------------------------

    def _run_lm_head(
        self,
        hidden_dev,
        attn_metadata: AttentionMetadata,
        *,
        token_bucket: int,
        sampling_batch: DeviceSamplingBatch | None = None,
    ) -> dict[str, np.ndarray]:
        bs = int(attn_metadata.batch_size)
        self._last_token_indices_host.fill(0)
        if bs > 0:
            last_indices = (attn_metadata.query_start_loc[1 : bs + 1] - 1).astype(
                np.int32
            )
            self._last_token_indices_host[:bs] = last_indices
        _overwrite_device_tensor(
            self._last_token_indices_dev,
            self._last_token_indices_host,
        )
        lp_output = self._logits_processor.forward(
            hidden_dev,
            self._shared_tensors["final_norm"],
            self._shared_tensors["lm_head"],
            self._last_token_indices_dev,
            batch_size=bs,
            token_bucket=int(token_bucket),
            sampling_batch=sampling_batch,
            needs_logprobs=bool(sampling_batch.needs_logprobs)
            if sampling_batch
            else False,
            logprobs_k=int(sampling_batch.logprobs_k) if sampling_batch else 0,
        )
        return lp_output.to_shm_dict(
            vocab_offset=self._weights.lm_head_vocab_offset,
        )

    # -- CPU attention fallback --------------------------------------------------

    @staticmethod
    def _run_cpu_attn(
        attn_frag,
        q,
        k,
        v,
        kv_dev,
        attn_metadata: AttentionMetadata,
        effective_real_tt: int,
        effective_bucket: int,
    ):
        """Download Q/K/V, run CPU attn, writeback KV, re-pad context."""
        rt = effective_real_tt
        q_cpu = q[:rt] if hasattr(q, "__getitem__") else q.numpy()[:rt]
        k_cpu = k[:rt] if hasattr(k, "__getitem__") else k.numpy()[:rt]
        v_cpu = v[:rt] if hasattr(v, "__getitem__") else v.numpy()[:rt]
        kv_np = kv_dev.numpy() if hasattr(kv_dev, "numpy") else kv_dev

        ctx_cpu = attn_frag(q_cpu, k_cpu, v_cpu, kv_np, attn_metadata)

        # Writeback mutated KV cache to device
        _overwrite_device_tensor(kv_dev, kv_np)

        # Re-pad context to token_bucket for downstream device stages
        if rt < effective_bucket:
            pad_shape = list(ctx_cpu.shape)
            pad_shape[0] = effective_bucket - rt
            return np.concatenate(
                [ctx_cpu, np.zeros(pad_shape, dtype=ctx_cpu.dtype)],
                axis=0,
            )
        return ctx_cpu
