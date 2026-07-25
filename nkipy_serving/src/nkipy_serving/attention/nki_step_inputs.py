"""Shared NKI attention step-input preparation utilities.

This module owns the per-step host/device state for:
  - padded slot mappings for KV update
  - unified prefill/decode tile-plan tensors for NKI attention

The prepared state is bucketed and reused across steps and layers. Callers
allocate one PreparedNkiStepInputs per token bucket, initialize it with dummy
plans once, then rewrite only the dynamic pieces each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nkipy_serving.attention.base import AttentionMetadata
from nkipy_serving.attention.nki_blocksparse_flash_attention import (
    build_decode_tile_plan,
    build_decode_tile_plan_inplace,
    build_dummy_decode_tile_plan,
    build_dummy_prefill_tile_plan,
    build_prefill_tile_plan,
    build_prefill_tile_plan_inplace,
    build_unified_tile_plans,
    compute_max_tile_counts,
)


@dataclass
class PreparedNkiStepInputs:
    token_bucket: int
    attn_bucket: int
    max_num_prefill_tiles: int
    max_num_decode_tiles: int
    slot_mapping: object
    slot_mapping_host: np.ndarray
    prefill_dummy_plan: dict[str, np.ndarray]
    decode_dummy_plan: dict[str, np.ndarray]
    prefill_plan_is_dummy: bool
    decode_plan_is_dummy: bool
    p_tqi: object
    p_tbt: object
    p_tm: object
    p_ndls: object
    p_qup: object
    p_lti: object
    d_tqi: object
    d_tbt: object
    d_tm: object
    d_ndls: object
    d_qup: object
    d_lti: object
    prefill_plan_host: dict[str, np.ndarray] | None = None
    decode_plan_host: dict[str, np.ndarray] | None = None
    slot_mapping_active_tokens: int = 0
    slot_mapping_scratch_slot: int = -1

    def as_inputs_dict(self) -> dict[str, object]:
        return {
            "slot_mapping": self.slot_mapping,
            "p_tqi": self.p_tqi,
            "p_tbt": self.p_tbt,
            "p_tm": self.p_tm,
            "p_ndls": self.p_ndls,
            "p_qup": self.p_qup,
            "p_lti": self.p_lti,
            "d_tqi": self.d_tqi,
            "d_tbt": self.d_tbt,
            "d_tm": self.d_tm,
            "d_ndls": self.d_ndls,
            "d_qup": self.d_qup,
            "d_lti": self.d_lti,
        }


def _write_prefill_plan(
    step_inputs: PreparedNkiStepInputs,
    overwrite_device_tensor,
    plan: dict[str, np.ndarray],
) -> None:
    overwrite_device_tensor(step_inputs.p_tqi, plan["tile_q_indices"])
    overwrite_device_tensor(step_inputs.p_tbt, plan["tile_block_tables"])
    overwrite_device_tensor(step_inputs.p_tm, plan["tile_masks"])
    overwrite_device_tensor(step_inputs.p_ndls, plan["num_dynamic_loop_steps"])
    overwrite_device_tensor(step_inputs.p_qup, plan["q_update_pred"])
    overwrite_device_tensor(step_inputs.p_lti, plan["last_tile_indices"])


def _write_decode_plan(
    step_inputs: PreparedNkiStepInputs,
    overwrite_device_tensor,
    plan: dict[str, np.ndarray],
) -> None:
    overwrite_device_tensor(step_inputs.d_tqi, plan["tile_q_indices"])
    overwrite_device_tensor(step_inputs.d_tbt, plan["tile_block_tables"])
    overwrite_device_tensor(step_inputs.d_tm, plan["tile_masks"])
    overwrite_device_tensor(step_inputs.d_ndls, plan["num_dynamic_loop_steps"])
    overwrite_device_tensor(step_inputs.d_qup, plan["q_update_pred"])
    overwrite_device_tensor(step_inputs.d_lti, plan["last_tile_indices"])


def _make_plan_host(plan: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.array(value, copy=True) for name, value in plan.items()}


def allocate_prepared_nki_step_inputs(
    alloc_device_scratch,
    *,
    token_bucket: int,
    attn_bucket: int,
    max_context_len: int,
    max_requests: int,
    num_blocks: int,
    block_size: int,
    prefix: str,
) -> PreparedNkiStepInputs:
    """Allocate reusable device buffers and host mirrors for one bucket."""
    max_p, max_d = compute_max_tile_counts(
        token_bucket=int(attn_bucket),
        max_context_len=int(max_context_len),
        max_requests=int(max_requests),
        block_size=int(block_size),
    )
    prefill_dummy_plan = build_dummy_prefill_tile_plan(
        max_num_prefill_tiles=max_p,
        block_size=int(block_size),
    )
    decode_dummy_plan = build_dummy_decode_tile_plan(
        max_num_decode_tiles=max_d,
        block_size=int(block_size),
    )

    return PreparedNkiStepInputs(
        token_bucket=int(token_bucket),
        attn_bucket=int(attn_bucket),
        max_num_prefill_tiles=int(max_p),
        max_num_decode_tiles=int(max_d),
        slot_mapping=alloc_device_scratch(
            (int(token_bucket),),
            np.int32,
            name=f"{prefix}_slot_mapping_t{int(token_bucket)}",
        ),
        slot_mapping_host=np.zeros((int(token_bucket),), dtype=np.int32),
        prefill_dummy_plan=prefill_dummy_plan,
        decode_dummy_plan=decode_dummy_plan,
        prefill_plan_is_dummy=False,
        decode_plan_is_dummy=False,
        p_tqi=alloc_device_scratch(
            prefill_dummy_plan["tile_q_indices"].shape,
            prefill_dummy_plan["tile_q_indices"].dtype,
            name=f"{prefix}_pf_tqi_t{int(token_bucket)}",
        ),
        p_tbt=alloc_device_scratch(
            prefill_dummy_plan["tile_block_tables"].shape,
            prefill_dummy_plan["tile_block_tables"].dtype,
            name=f"{prefix}_pf_tbt_t{int(token_bucket)}",
        ),
        p_tm=alloc_device_scratch(
            prefill_dummy_plan["tile_masks"].shape,
            prefill_dummy_plan["tile_masks"].dtype,
            name=f"{prefix}_pf_tm_t{int(token_bucket)}",
        ),
        p_ndls=alloc_device_scratch(
            prefill_dummy_plan["num_dynamic_loop_steps"].shape,
            prefill_dummy_plan["num_dynamic_loop_steps"].dtype,
            name=f"{prefix}_pf_ndls_t{int(token_bucket)}",
        ),
        p_qup=alloc_device_scratch(
            prefill_dummy_plan["q_update_pred"].shape,
            prefill_dummy_plan["q_update_pred"].dtype,
            name=f"{prefix}_pf_qup_t{int(token_bucket)}",
        ),
        p_lti=alloc_device_scratch(
            prefill_dummy_plan["last_tile_indices"].shape,
            prefill_dummy_plan["last_tile_indices"].dtype,
            name=f"{prefix}_pf_lti_t{int(token_bucket)}",
        ),
        d_tqi=alloc_device_scratch(
            decode_dummy_plan["tile_q_indices"].shape,
            decode_dummy_plan["tile_q_indices"].dtype,
            name=f"{prefix}_dc_tqi_t{int(token_bucket)}",
        ),
        d_tbt=alloc_device_scratch(
            decode_dummy_plan["tile_block_tables"].shape,
            decode_dummy_plan["tile_block_tables"].dtype,
            name=f"{prefix}_dc_tbt_t{int(token_bucket)}",
        ),
        d_tm=alloc_device_scratch(
            decode_dummy_plan["tile_masks"].shape,
            decode_dummy_plan["tile_masks"].dtype,
            name=f"{prefix}_dc_tm_t{int(token_bucket)}",
        ),
        d_ndls=alloc_device_scratch(
            decode_dummy_plan["num_dynamic_loop_steps"].shape,
            decode_dummy_plan["num_dynamic_loop_steps"].dtype,
            name=f"{prefix}_dc_ndls_t{int(token_bucket)}",
        ),
        d_qup=alloc_device_scratch(
            decode_dummy_plan["q_update_pred"].shape,
            decode_dummy_plan["q_update_pred"].dtype,
            name=f"{prefix}_dc_qup_t{int(token_bucket)}",
        ),
        d_lti=alloc_device_scratch(
            decode_dummy_plan["last_tile_indices"].shape,
            decode_dummy_plan["last_tile_indices"].dtype,
            name=f"{prefix}_dc_lti_t{int(token_bucket)}",
        ),
        prefill_plan_host=_make_plan_host(prefill_dummy_plan),
        decode_plan_host=_make_plan_host(decode_dummy_plan),
    )


def initialize_prepared_nki_step_inputs(
    step_inputs: PreparedNkiStepInputs,
    overwrite_device_tensor,
) -> None:
    """Upload the no-op prefill/decode plans once after allocation."""
    _write_prefill_plan(
        step_inputs, overwrite_device_tensor, step_inputs.prefill_dummy_plan
    )
    _write_decode_plan(
        step_inputs, overwrite_device_tensor, step_inputs.decode_dummy_plan
    )
    step_inputs.prefill_plan_is_dummy = True
    step_inputs.decode_plan_is_dummy = True


def prepare_prepared_nki_step_inputs(
    step_inputs: PreparedNkiStepInputs,
    overwrite_device_tensor,
    *,
    attn_metadata: AttentionMetadata,
    real_total_tokens: int,
    num_blocks: int,
    block_size: int,
) -> dict[str, object]:
    """Rewrite slot mapping and active tile plans for the current batch."""
    scratch_slot = (int(num_blocks) - 1) * int(block_size)
    padded_slot = step_inputs.slot_mapping_host
    real_total_tokens = int(real_total_tokens)
    if int(step_inputs.slot_mapping_scratch_slot) != int(scratch_slot):
        padded_slot.fill(int(scratch_slot))
        step_inputs.slot_mapping_scratch_slot = int(scratch_slot)
        step_inputs.slot_mapping_active_tokens = 0
    prev_active = int(step_inputs.slot_mapping_active_tokens)
    if real_total_tokens < prev_active:
        padded_slot[real_total_tokens:prev_active] = int(scratch_slot)
    padded_slot[:real_total_tokens] = np.asarray(
        attn_metadata.slot_mapping[: int(real_total_tokens)],
        dtype=np.int32,
    )
    step_inputs.slot_mapping_active_tokens = real_total_tokens
    overwrite_device_tensor(step_inputs.slot_mapping, padded_slot)

    query_lens = np.asarray(np.diff(attn_metadata.query_start_loc), dtype=np.int32)
    has_prefill = bool(np.any(query_lens > 1))
    has_decode = bool(np.any(query_lens == 1))

    if has_prefill and has_decode:
        prefill_nps, decode_nps = build_unified_tile_plans(
            attn_metadata,
            token_bucket=int(step_inputs.attn_bucket),
            max_num_prefill_tiles=int(step_inputs.max_num_prefill_tiles),
            max_num_decode_tiles=int(step_inputs.max_num_decode_tiles),
            block_size=int(block_size),
        )
        _write_prefill_plan(step_inputs, overwrite_device_tensor, prefill_nps)
        _write_decode_plan(step_inputs, overwrite_device_tensor, decode_nps)
        step_inputs.prefill_plan_is_dummy = False
        step_inputs.decode_plan_is_dummy = False
    elif has_prefill:
        can_use_prefill_fast_path = hasattr(attn_metadata, "seq_lens") and hasattr(
            attn_metadata,
            "block_tables",
        )
        if can_use_prefill_fast_path:
            if step_inputs.prefill_plan_host is None:
                step_inputs.prefill_plan_host = _make_plan_host(
                    step_inputs.prefill_dummy_plan
                )
            prefill_nps = build_prefill_tile_plan_inplace(
                attn_metadata,
                token_bucket=int(step_inputs.attn_bucket),
                max_num_prefill_tiles=int(step_inputs.max_num_prefill_tiles),
                block_size=int(block_size),
                out=step_inputs.prefill_plan_host,
            )
        else:
            prefill_nps = build_prefill_tile_plan(
                attn_metadata,
                token_bucket=int(step_inputs.attn_bucket),
                max_num_prefill_tiles=int(step_inputs.max_num_prefill_tiles),
                block_size=int(block_size),
            )
        _write_prefill_plan(step_inputs, overwrite_device_tensor, prefill_nps)
        step_inputs.prefill_plan_is_dummy = False
        if not step_inputs.decode_plan_is_dummy:
            _write_decode_plan(
                step_inputs, overwrite_device_tensor, step_inputs.decode_dummy_plan
            )
            step_inputs.decode_plan_is_dummy = True
    elif has_decode:
        can_use_decode_fast_path = hasattr(attn_metadata, "seq_lens") and hasattr(
            attn_metadata,
            "block_tables",
        )
        if can_use_decode_fast_path:
            if step_inputs.decode_plan_host is None:
                step_inputs.decode_plan_host = _make_plan_host(
                    step_inputs.decode_dummy_plan
                )
            build_decode_tile_plan_inplace(
                attn_metadata,
                max_num_decode_tiles=int(step_inputs.max_num_decode_tiles),
                block_size=int(block_size),
                out=step_inputs.decode_plan_host,
            )
            decode_plan = step_inputs.decode_plan_host
        else:
            decode_plan = build_decode_tile_plan(
                attn_metadata,
                token_bucket=int(step_inputs.attn_bucket),
                max_num_decode_tiles=int(step_inputs.max_num_decode_tiles),
                block_size=int(block_size),
            )
        _write_decode_plan(step_inputs, overwrite_device_tensor, decode_plan)
        step_inputs.decode_plan_is_dummy = False
        if not step_inputs.prefill_plan_is_dummy:
            _write_prefill_plan(
                step_inputs, overwrite_device_tensor, step_inputs.prefill_dummy_plan
            )
            step_inputs.prefill_plan_is_dummy = True
    else:
        if not step_inputs.prefill_plan_is_dummy:
            _write_prefill_plan(
                step_inputs, overwrite_device_tensor, step_inputs.prefill_dummy_plan
            )
            step_inputs.prefill_plan_is_dummy = True
        if not step_inputs.decode_plan_is_dummy:
            _write_decode_plan(
                step_inputs, overwrite_device_tensor, step_inputs.decode_dummy_plan
            )
            step_inputs.decode_plan_is_dummy = True

    return step_inputs.as_inputs_dict()
