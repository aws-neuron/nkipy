"""Single-step model runner: thin dispatcher using model executor."""

from __future__ import annotations

from typing import Any

import numpy as np

from nkipy_serving.attention.base import (
    FORWARD_MODE_DECODE,
    FORWARD_MODE_EXTEND,
    AttentionMetadata,
)
from nkipy_serving.batching.contracts import ForwardBatch, ForwardMode
from nkipy_serving.config import RuntimeConfig
from nkipy_serving.mem_cache.memory_pool import MHATokenToKVPool
from nkipy_serving.models.registry import resolve_model_spec
from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
from nkipy_serving.sampling.logits_processor_np import NumpyLogitsProcessor


def _build_attn_metadata(
    forward_batch: ForwardBatch,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> AttentionMetadata:
    if forward_batch.forward_mode == ForwardMode.EXTEND:
        mode_int = FORWARD_MODE_EXTEND
    else:
        mode_int = FORWARD_MODE_DECODE

    # Use real_total_tokens (unpadded) for attention metadata so that
    # slot_mapping and total_tokens reflect actual tokens, not padding.
    real_tt = (
        forward_batch.real_total_tokens
        if forward_batch.real_total_tokens > 0
        else forward_batch.total_tokens
    )

    return AttentionMetadata(
        forward_mode=mode_int,
        seq_lens=forward_batch.seq_lens,
        slot_mapping=forward_batch.slot_mapping[:real_tt],
        block_tables=forward_batch.block_tables,
        query_start_loc=forward_batch.query_start_loc,
        total_tokens=real_tt,
        batch_size=forward_batch.batch_size,
        max_seq_len=forward_batch.max_seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )


class ModelRunner:
    """Single-step model runner.

    Resolves the model spec, creates an executor, and dispatches forward
    calls.  The executor owns weights, device tensors, compiled kernels,
    and the CPU/device forward logic.
    """

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        kv_pool: MHATokenToKVPool,
        tp_rank: int = 0,
        ep_rank: int = 0,
    ):
        self._runtime_config = runtime_config
        self._kv_pool = kv_pool

        spec = resolve_model_spec(runtime_config.model_id)
        model_config = spec.build_config(
            runtime_config, tp_rank=tp_rank, ep_rank=ep_rank
        )
        self._executor = spec.create_executor(model_config, kv_pool, runtime_config)

    def forward(self, forward_batch: ForwardBatch) -> dict[str, Any]:
        """Run a single forward step.

        Always returns a dict with sampled token candidates (top-1 or top-k)
        and optional logprobs.  Raw ndarray logits from any backend are
        processed via ``NumpyLogitsProcessor``.
        """
        w = self._executor.weights
        attn_metadata = _build_attn_metadata(
            forward_batch,
            num_kv_heads=w.num_kv_heads,
            head_dim=w.head_dim,
            block_size=self._executor.kv_pool.block_size,
        )
        prepare_attn = getattr(self._executor, "prepare_attention_metadata", None)
        if callable(prepare_attn):
            attn_metadata = prepare_attn(
                attn_metadata,
                positions=forward_batch.positions,
                token_bucket=forward_batch.token_bucket,
                forward_batch=forward_batch,
            )
        kv_caches = [
            self._executor.kv_pool.get_kv_cache(layer_id)
            for layer_id in range(w.num_hidden_layers)
        ]
        output = self._executor.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            kv_caches,
            attn_metadata,
            token_bucket=forward_batch.token_bucket,
            real_total_tokens=forward_batch.real_total_tokens,
            sampling_batch=DeviceSamplingBatch.from_forward_batch(forward_batch),
            attention_lane=forward_batch.attention_lane,
        )
        if isinstance(output, np.ndarray):
            vocab_offset = int(
                getattr(self._executor.weights, "lm_head_vocab_offset", 0)
            )
            sampling_batch = DeviceSamplingBatch.from_forward_batch(forward_batch)
            np_processor = NumpyLogitsProcessor(vocab_offset=vocab_offset)
            lp_output = np_processor.forward(
                output,
                forward_batch.sample_mask,
                forward_batch.query_start_loc,
                forward_batch.batch_size,
                sampling_batch=sampling_batch,
                needs_logprobs=bool(forward_batch.needs_logprobs),
                logprobs_k=int(forward_batch.logprobs_k),
            )
            return lp_output.to_shm_dict(vocab_offset=vocab_offset)
        return output

    def warmup(self, paddings=None) -> None:
        """Run the executor startup warmup stage.

        Delegates to the executor's warmup method if available. Device executors
        may both compile and execute synthetic bucketed forwards here so the
        ready state includes first-touch kernel warmup, not just compilation.
        """
        if paddings is not None and hasattr(self._executor, "warmup"):
            self._executor.warmup(paddings)

    def reload_weights_from_disk(self, model_path: str) -> None:
        if not hasattr(self._executor, "reload_weights_from_disk"):
            raise RuntimeError(
                f"Model executor does not support weight reload: {type(self._executor).__name__}"
            )
        self._executor.reload_weights_from_disk(model_path)

    def flush_cache(self) -> None:
        if hasattr(self._executor, "flush_cache"):
            self._executor.flush_cache()
            return
        self._kv_pool.clear()

    def clear_request_state(self, owner_ids: list[int]) -> None:
        clear_fn = getattr(self._executor, "clear_request_state", None)
        if callable(clear_fn):
            clear_fn(owner_ids)

    def checkpoint_request_state(
        self,
        *,
        checkpoint_id: str,
        owner_id: int,
        seq_len: int,
        num_tokens: int,
    ) -> None:
        checkpoint_fn = getattr(self._executor, "checkpoint_request_state", None)
        if not callable(checkpoint_fn):
            raise RuntimeError(
                f"Model executor does not support request-state checkpoints: "
                f"{type(self._executor).__name__}"
            )
        checkpoint_fn(
            checkpoint_id=checkpoint_id,
            owner_id=owner_id,
            seq_len=seq_len,
            num_tokens=num_tokens,
        )

    def restore_request_state(self, checkpoint_id: str) -> None:
        restore_fn = getattr(self._executor, "restore_request_state", None)
        if not callable(restore_fn):
            raise RuntimeError(
                f"Model executor does not support request-state restore: "
                f"{type(self._executor).__name__}"
            )
        restore_fn(checkpoint_id)
