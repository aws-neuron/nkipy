# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import os
from dataclasses import dataclass, fields
from typing import Optional

import torch
import torch.nn.functional as F

from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.constants import (  # noqa: E501
    B_FMAX_SIZE,
    B_P_SIZE,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.flash_pa_with_schedule import (  # noqa: E501
    flash_attn_varlen_blocksparse_nkifunc,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.flash_paged_attn_varlen import (  # noqa: E501
    flash_attn_varlen_nkifunc,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.scheduler import (  # noqa: E501
    FlashAttentionPlanner,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.scheduler import (  # noqa: E501
    FlashTilePlan as ContextAttnPlan,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.utils import (
    BlockDiagonalCausalFromBottomRightMask,
    ceil_div,
    get_active_block_tables,
    is_power_of_2,
    pad_to_multiple,
    pad_to_next_power_of_2,
    save_kernel_tensors_as_npy,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.wrapper import (
    flash_paged_attention_blocksparse_custom_op_nkifunc,
    flash_paged_attention_varlen_custom_op_nkifunc,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.wrapper_kv_cache import (  # noqa: E501
    flash_paged_attention_blocksparse_kv_cache_custom_op_nkifunc,
)
from vllm_nkipy.config import PagedAttnImpl, _get_paged_attn_impl


def _get_nki_custom_op(f):
    nkifunc_to_custom_op = {
        flash_attn_varlen_blocksparse_nkifunc: (
            flash_paged_attention_blocksparse_custom_op_nkifunc
        ),
        flash_attn_varlen_nkifunc: (
            flash_paged_attention_varlen_custom_op_nkifunc
        ),
        "flash_attn_varlen_blocksparse_kv_cache_nkifunc": (
            flash_paged_attention_blocksparse_kv_cache_custom_op_nkifunc
        ),
    }
    return nkifunc_to_custom_op[f]


def _decide_execution_mode(exec_mode):
    _supported_exec_mode = [
        "xla",
        "baremetal",
        "dynamo",
    ]
    if not exec_mode:
        exec_mode = os.getenv("TEST_EXEC_MODE", "")
    if exec_mode.lower() not in _supported_exec_mode:
        if exec_mode:
            print(f"Execution mode {exec_mode} is not supported.")
        exec_mode = _supported_exec_mode[0]
        print(f"Default to {exec_mode} mode for execution")
    else:
        exec_mode = exec_mode.lower()
        print(f"Use {exec_mode} mode for execution")
    return exec_mode


def _get_default_compiler_flags(save_artifact):
    compiler_flags = [
        "-O1",
        "--lnc=1",
        # "--enable-internal-data-race-checker",
    ]
    enable_branch_hint = os.getenv("ENABLE_BRANCH_HINT", "0") != "0"
    if enable_branch_hint:
        compiler_flags.append("--internal-backend-options='--enable-branch-hint'")
    if save_artifact:
        compiler_flags.extend(
            [
                "--enable-internal-birsim-after-all",
                "--internal-compiler-debug-mode=all",
                "--tensorizer-options='--print-stats --dump-after=All'",
            ]
        )
    return compiler_flags


@dataclass(frozen=True)
class ContextAttnInputs:
    tile_q_indices: torch.Tensor
    tile_block_tables: torch.Tensor
    tile_masks: torch.Tensor
    num_dynamic_loop_steps: torch.Tensor
    last_tile_indices: torch.Tensor
    q_update_pred: Optional[torch.Tensor]

    def as_dict(self, prefix=""):
        return {
            (prefix + field.name): getattr(self, field.name) for field in fields(self)
        }

    def to(self, device):
        """Move all tensor fields to the specified device."""
        field_values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                field_values[field.name] = value.to(device)
            else:
                field_values[field.name] = value
        return self.__class__(**field_values)

    @classmethod
    def create_empty(
        cls,
        max_num_tiles,
        q_tile_size,
        kv_tile_size,
        block_size,
        is_prefill,
        max_num_q_tiles,
        device=None,
    ):
        assert max_num_tiles is not None
        assert kv_tile_size % block_size == 0
        tile_block_tables = torch.zeros(
            max_num_tiles, kv_tile_size // block_size, dtype=torch.int, device=device
        )
        if is_prefill:
            tile_q_indices = torch.zeros(
                max_num_tiles, q_tile_size, dtype=torch.int32, device=device
            )
            inner_tile_size = min(B_P_SIZE, q_tile_size)
            tile_masks = torch.zeros(
                inner_tile_size,
                max_num_tiles,
                q_tile_size // inner_tile_size,
                kv_tile_size,
                dtype=torch.uint8,
                device=device,
            )
        else:
            assert q_tile_size == 1
            # pad a large enough value to trigger dma
            # skipping when merging decode back into prefill
            tile_q_indices = torch.full(
                (max_num_tiles, q_tile_size),
                (1 << 30),
                dtype=torch.int32,
                device=device,
            )
            tile_masks = torch.zeros(
                B_P_SIZE,
                max_num_tiles,
                kv_tile_size // B_P_SIZE,
                dtype=torch.uint8,
                device=device,
            )
        num_dynamic_loop_stesp = torch.zeros(1, 1, dtype=torch.int, device=device)
        # last_tile_indices = torch.arange(
        #     max_num_q_tiles, dtype=torch.int,
        #     device=device
        # ).view(max_num_q_tiles, 1)
        last_tile_indices = torch.arange(
            max_num_q_tiles, dtype=torch.int, device=device
        ).view(max_num_q_tiles, 1) + torch.zeros(
            (max_num_q_tiles, 1), dtype=torch.int, device=device
        )
        q_update_pred = torch.zeros(max_num_tiles, 1, dtype=torch.uint8, device=device)
        return cls(
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            num_dynamic_loop_steps=num_dynamic_loop_stesp,
            last_tile_indices=last_tile_indices,
            q_update_pred=q_update_pred,
        )


class NKIFlashPagedAttentionRunner:
    def __init__(
        self,
        query_lens,
        context_lens,
        large_q_tile_size,
        large_kv_tile_size,
        block_size,
        *,
        max_req_seq_len=None,
        include_prompt_in_ctx=False,
        dynamic_loop_unrolling_size=8,
        enable_separate_prefill_decode=True,
        padded_query_length=None,
        max_num_prefill_tiles=None,
        max_num_decode_tiles=None,
        skip_active=False,
        exec_mode=None,
    ):
        """
        Kernel executor to generate and cache tile-plan, and dispatch for execution
        """
        assert large_kv_tile_size >= B_P_SIZE
        self.query_lens = query_lens
        self.context_lens = context_lens
        self.max_req_seq_len = max_req_seq_len
        if self.max_req_seq_len is not None:
            assert torch.all(
                self.query_lens + self.context_lens <= self.max_req_seq_len
            )
        self.large_q_tile_size = large_q_tile_size
        self.large_kv_tile_size = large_kv_tile_size
        self.block_size = block_size
        self.dynamic_loop_unrolling_size = dynamic_loop_unrolling_size
        assert (
            max_num_prefill_tiles is None
            or max_num_prefill_tiles % dynamic_loop_unrolling_size == 0
        ), (
            f"Expecting {max_num_prefill_tiles=} to be"
            f" divisible by {dynamic_loop_unrolling_size=}"
        )
        assert (
            max_num_decode_tiles is None
            or max_num_decode_tiles % dynamic_loop_unrolling_size == 0
        ), (
            f"Expecting {max_num_decode_tiles=} to be"
            f" divisible by {dynamic_loop_unrolling_size=}"
        )
        self.max_num_prefill_tiles = max_num_prefill_tiles
        self.max_num_decode_tiles = max_num_decode_tiles
        self.num_active_tokens_after_padding = padded_query_length
        self.exec_mode = _decide_execution_mode(exec_mode)
        self.numpy_kernel_use_bf16 = True  # use ml_dtypes.bfloat16 for baremetal mode
        self.save_artifact = os.getenv("TEST_SAVE_ARTIFACTS", "0") != "0"
        self.num_actual_tokens = None
        self.prefill_ctx_inputs: Optional[ContextAttnInputs] = None
        self.decode_ctx_inputs: Optional[ContextAttnInputs] = None
        self.include_prompt_in_ctx = include_prompt_in_ctx
        self.skip_active = skip_active
        if self.include_prompt_in_ctx:
            # kernel does not perform active compute separately
            assert self.skip_active
        assert self.batch_size <= B_P_SIZE
        assert is_power_of_2(self.large_q_tile_size)
        assert is_power_of_2(self.large_kv_tile_size)
        self.prefill_plan: Optional[ContextAttnPlan] = None
        self.decode_plan: Optional[ContextAttnPlan] = None
        self.ctx_input_type = ContextAttnInputs
        self._preprocess(enable_separate_prefill_decode)
        self.kv_cache = (
            _get_paged_attn_impl()
            == PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION_KV_CACHE
        )

    @property
    def batch_size(self):
        return len(self.query_lens)

    def to(self, device):
        """Move input tensors to the specified device."""
        if self.prefill_ctx_inputs is not None:
            self.prefill_ctx_inputs = self.prefill_ctx_inputs.to(device)
        if self.decode_ctx_inputs is not None:
            self.decode_ctx_inputs = self.decode_ctx_inputs.to(device)
        if self.active_mask is not None:
            self.active_mask = self.active_mask.to(device)
        return self

    def _get_prefill_decode_batch_size(self, enable_separate_prefill_decode):
        decode_batch_size = 0
        assert torch.all(self.query_lens > 0), f"Expect nonzero {self.query_lens}"
        for x in reversed(self.query_lens):
            if x > 1:
                break
            decode_batch_size += 1
        batch_size = self.batch_size
        assert decode_batch_size <= batch_size
        prefill_batch_size = batch_size - decode_batch_size
        if (
            not enable_separate_prefill_decode
            and prefill_batch_size > 0
            and decode_batch_size > 0
        ):
            # run decode as prefill
            prefill_batch_size = batch_size
            decode_batch_size = 0
        return prefill_batch_size, decode_batch_size

    def _decide_padded_query_len(self):
        if self.num_active_tokens_after_padding is None:
            if self.num_actual_tokens < B_P_SIZE:
                num_tokens_after_padding = pad_to_next_power_of_2(
                    self.num_actual_tokens,
                )
            elif self.num_actual_tokens < B_FMAX_SIZE:
                num_tokens_after_padding = pad_to_multiple(
                    self.num_actual_tokens,
                    B_P_SIZE,
                )
            else:
                num_tokens_after_padding = pad_to_multiple(
                    self.num_actual_tokens,
                    B_FMAX_SIZE,
                )
            num_tokens_after_padding = pad_to_multiple(
                num_tokens_after_padding,
                self.large_q_tile_size,
            )
            self.num_active_tokens_after_padding = num_tokens_after_padding
        else:
            if self.num_actual_tokens > self.num_active_tokens_after_padding:
                raise ValueError(
                    f"{self.num_actual_tokens=} > "
                    f"{self.num_active_tokens_after_padding=}\n"
                    f"{self.query_lens=}"
                )
            assert self.num_actual_tokens <= self.num_active_tokens_after_padding, (
                f"{self.num_actual_tokens=} > {self.num_active_tokens_after_padding=}"
            )
            if self.num_actual_tokens < B_P_SIZE:
                assert is_power_of_2(self.num_active_tokens_after_padding)
            elif self.num_actual_tokens < B_FMAX_SIZE:
                assert self.num_active_tokens_after_padding % B_P_SIZE == 0
            else:
                assert self.num_active_tokens_after_padding % B_FMAX_SIZE == 0
            assert self.num_active_tokens_after_padding % self.large_q_tile_size == 0, (
                f"Bad {self.num_active_tokens_after_padding=} {self.large_q_tile_size=}"
            )

    def _preprocess(self, enable_separate_prefill_decode: bool):
        self.num_actual_tokens = self.query_lens.sum().item()
        self._decide_padded_query_len()
        prefill_batch_size, decode_batch_size = self._get_prefill_decode_batch_size(
            enable_separate_prefill_decode
        )
        # print(f"{prefill_batch_size=}, {decode_batch_size=}")
        self._build_kernel_plan(prefill_batch_size, decode_batch_size)
        self._pad_kernel_plan()

        self.prefill_batch_size = prefill_batch_size
        if not self.skip_active:
            self.active_mask = self._build_active_token_mask()
        else:
            self.active_mask = None

    def get_num_active_tokens_after_padding(self):
        return self.num_active_tokens_after_padding

    def prepare_tile_plan_inputs(
        self, block_tables, max_kv_cache_size, sliding_window=None
    ):
        if self.prefill_batch_size > 0:
            # prepare prefill
            self.prefill_ctx_inputs = self._build_inputs_from_plan(
                is_decode_plan=False,
                ctx_plan=self.prefill_plan,
                block_tables_2d=block_tables[: self.prefill_batch_size],
                max_kv_cache_size=max_kv_cache_size,
                sliding_window=sliding_window,
            )
        if self.prefill_batch_size < self.batch_size:
            # prepare decode
            decode_start_offset = (
                self.query_lens[: self.prefill_batch_size].sum().item()
            )
            self.decode_ctx_inputs = self._build_inputs_from_plan(
                is_decode_plan=True,
                ctx_plan=self.decode_plan,
                block_tables_2d=block_tables[self.prefill_batch_size :],
                max_kv_cache_size=max_kv_cache_size,
                query_start_offset=decode_start_offset,
                sliding_window=sliding_window,
            )

    def _build_active_token_mask(self):
        # Build attention masks
        active_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
            query_lens=self.query_lens,
            seq_lens=self.query_lens,
        )
        active_mask = F.pad(
            active_mask,
            (
                0,
                self.num_active_tokens_after_padding - active_mask.shape[1],
                0,
                self.num_active_tokens_after_padding - active_mask.shape[0],
            ),
            "constant",
            0,
        ).to(torch.uint8)
        return active_mask

    @classmethod
    def _pad_non_empty_plan_to_max(
        cls,
        plan: ContextAttnPlan,
        pad_num_tiles_to: int,
        unroll_size: int,
        q_pad_value: int = 0,
    ):
        assert plan is not None
        if plan.num_tiles == 0:
            return plan
        # make sure unrolling is correct if used in dynamic loop
        num_real_tiles = plan.num_real_tiles
        if pad_num_tiles_to is None:
            pad_num_tiles_to = pad_to_multiple(num_real_tiles, unroll_size)
        else:
            assert pad_num_tiles_to >= num_real_tiles, (
                f"{pad_num_tiles_to=} vs {num_real_tiles=}"
            )
            assert pad_num_tiles_to % unroll_size == 0, (
                f"Expecting {pad_num_tiles_to=} divisible by {unroll_size=}"
            )
        if pad_num_tiles_to != num_real_tiles:
            plan = plan.pad_plan(pad_num_tiles_to, q_pad_value=q_pad_value)
        # print(f"{plan.num_real_tiles=}")
        # print(f"After padding, {plan.num_tiles=}")
        return plan

    def _build_kernel_plan(self, prefill_batch_size, decode_batch_size):
        unroll_size = self.dynamic_loop_unrolling_size
        assert unroll_size > 0

        if prefill_batch_size > 0:
            self.prefill_plan = FlashAttentionPlanner(
                prompt_lens=self.query_lens[:prefill_batch_size].int().numpy(),
                prior_context_lens=self.context_lens[:prefill_batch_size].int().numpy(),
                tile_size_q=self.large_q_tile_size,
                tile_size_kv=self.large_kv_tile_size,
                block_size=self.block_size,
                max_seq_len=self.max_req_seq_len,
                include_prompt_in_context=self.include_prompt_in_ctx,
            ).generate_plan()
        if decode_batch_size > 0:
            self.decode_plan = FlashAttentionPlanner(
                prompt_lens=self.query_lens[prefill_batch_size:].int().numpy(),
                prior_context_lens=self.context_lens[prefill_batch_size:].int().numpy(),
                tile_size_q=1,
                tile_size_kv=self.large_kv_tile_size,
                block_size=self.block_size,
                max_seq_len=self.max_req_seq_len,
                include_prompt_in_context=self.include_prompt_in_ctx,
            ).generate_plan()

    def _pad_kernel_plan(self):
        if self.prefill_plan is not None:
            self.prefill_plan = self._pad_non_empty_plan_to_max(
                plan=self.prefill_plan,
                pad_num_tiles_to=self.max_num_prefill_tiles,
                q_pad_value=self.num_active_tokens_after_padding * 10,
                unroll_size=self.dynamic_loop_unrolling_size,
            )
        if self.decode_plan is not None:
            self.decode_plan = self._pad_non_empty_plan_to_max(
                plan=self.decode_plan,
                pad_num_tiles_to=self.max_num_decode_tiles,
                q_pad_value=0,
                unroll_size=self.dynamic_loop_unrolling_size,
            )

    def _prepare_buffer_unroll_info(self, plan: ContextAttnPlan, max_num_q_tiles: int):
        max_num_q_tiles = pad_to_next_power_of_2(max_num_q_tiles)
        q_update_pred, last_tile_indices = plan.build_tile_update_indices(
            max_num_q_tiles=max_num_q_tiles,
        )
        # reshape q_update_pred for loop unrolling
        if q_update_pred is not None:
            num_tiles = plan.num_tiles
            assert num_tiles % self.dynamic_loop_unrolling_size == 0, (
                f"{num_tiles=} {self.dynamic_loop_unrolling_size=}"
            )
            q_update_pred = q_update_pred.reshape(num_tiles, 1)
            q_update_pred = torch.tensor(q_update_pred, dtype=torch.uint8)
        # reshape last_tile_indices for fast vector dge loading
        last_tile_indices = last_tile_indices.reshape(max_num_q_tiles, 1)
        last_tile_indices = torch.tensor(last_tile_indices, dtype=torch.int32)
        return q_update_pred, last_tile_indices

    def _build_tile_block_tables(
        self,
        ctx_plan: ContextAttnPlan,
        block_tables_2d,
        context_lens,
        dma_skip_value,
    ):
        if self.max_req_seq_len is None:
            num_active_blocks = ceil_div(context_lens, self.block_size).sum().item()
            num_active_blocks = pad_to_multiple(
                num_active_blocks, self.large_kv_tile_size // self.block_size
            )
            active_block_table = get_active_block_tables(
                block_tables_2d,
                context_lens,
                self.block_size,
                num_active_blocks,
            )
        else:
            active_block_table = block_tables_2d.flatten()
        tile_block_tables = ctx_plan.build_tile_block_tables(
            active_block_table,
            skip_value=dma_skip_value,
        )
        tile_block_tables = torch.tensor(tile_block_tables)
        assert tile_block_tables.dtype == torch.int32
        return tile_block_tables

    def _build_inputs_from_plan(
        self,
        ctx_plan: ContextAttnPlan,
        block_tables_2d: torch.Tensor,
        is_decode_plan: bool,
        max_kv_cache_size: int,
        query_start_offset: int = 0,
        sliding_window: Optional[int] = None,
    ):
        skip_value = 0 if is_decode_plan else self.num_active_tokens_after_padding * 10
        tile_q_indices = torch.tensor(
            ctx_plan.build_tile_q_indices(skip_value=skip_value)
        )
        assert tile_q_indices.dtype == torch.int32
        tile_masks = ctx_plan.build_tile_masks(
            decode_kq_layout=is_decode_plan, sliding_window=sliding_window
        )
        tile_masks = torch.tensor(tile_masks).to(torch.uint8)
        num_dynamic_loop_steps = torch.empty((1, 1), dtype=torch.int32)
        num_dynamic_loop_steps[0] = ceil_div(
            ctx_plan.num_real_tiles, self.dynamic_loop_unrolling_size
        )
        # print(f"{num_dynamic_loop_steps=}")

        if is_decode_plan:
            context_lens = self.context_lens[self.prefill_batch_size :]
            tile_block_tables = self._build_tile_block_tables(
                ctx_plan=ctx_plan,
                block_tables_2d=block_tables_2d,
                context_lens=context_lens,
                dma_skip_value=0,
            )
            assert self.batch_size <= B_P_SIZE, f"{self.batch_size=} > {B_P_SIZE=}"
            q_update_pred, last_tile_indices = self._prepare_buffer_unroll_info(
                ctx_plan,
                max_num_q_tiles=B_P_SIZE,
            )
            return self.ctx_input_type(
                tile_q_indices=tile_q_indices + query_start_offset,
                tile_block_tables=tile_block_tables,
                tile_masks=tile_masks,
                num_dynamic_loop_steps=num_dynamic_loop_steps,
                q_update_pred=q_update_pred,
                last_tile_indices=last_tile_indices,
            )
        else:
            context_lens = self.context_lens[: self.prefill_batch_size]
            tile_block_tables = self._build_tile_block_tables(
                ctx_plan=ctx_plan,
                block_tables_2d=block_tables_2d,
                context_lens=context_lens,
                dma_skip_value=0,
            )
            q_update_pred, last_tile_indices = self._prepare_buffer_unroll_info(
                ctx_plan,
                max_num_q_tiles=B_P_SIZE,
            )
            return self.ctx_input_type(
                tile_q_indices=tile_q_indices,
                tile_block_tables=tile_block_tables,
                tile_masks=tile_masks,
                num_dynamic_loop_steps=num_dynamic_loop_steps,
                q_update_pred=q_update_pred,
                last_tile_indices=last_tile_indices,
            )

    def _prepare_kernel(
        self,
        query,
        k_active,
        v_active,
        kv_cache,
        sinks,
        head_size,
        num_kv_heads,
        mixed_precision,
        force_use_varlen,
    ):
        # check input shapes
        # query: (1, num_heads, seq_q, d)
        # key:   (1, num_kv_heads, d, seq_k)
        # value: (1, num_kv_heads, seq_v, d)
        num_heads = query.shape[1]
        num_active_token = self.num_active_tokens_after_padding
        assert query.shape[2] == num_active_token, (
            "QKV sequence length must be padded to"
            f" {self.get_num_active_tokens_after_padding()=}"
        )
        assert query.shape == (1, num_heads, num_active_token, head_size)
        assert k_active.shape == (1, num_kv_heads, head_size, num_active_token)
        assert v_active.shape == (1, num_kv_heads, num_active_token, head_size)

        if isinstance(kv_cache, tuple):
            k_cache, v_cache = kv_cache
            input_kwargs = dict(
                query=query,
                key=k_active,
                value=v_active,
                key_cache=k_cache,
                value_cache=v_cache,
                active_mask=self.active_mask,
                n_kv_head=num_kv_heads,
                head_size=head_size,
                dynamic_loop_unroll_factor=self.dynamic_loop_unrolling_size,
                mixed_precision=mixed_precision,
                skip_active=self.skip_active,
                sinks=sinks,
            )
        else:
            input_kwargs = dict(
                query=query,
                key=k_active,
                value=v_active,
                kv_cache=kv_cache,
                active_mask=self.active_mask,
                n_kv_head=num_kv_heads,
                head_size=head_size,
                dynamic_loop_unroll_factor=self.dynamic_loop_unrolling_size,
                mixed_precision=mixed_precision,
                skip_active=self.skip_active,
                sinks=sinks,
            )
        if force_use_varlen or 0 < self.prefill_batch_size < self.batch_size:
            prefill_ctx_inputs = self.prefill_ctx_inputs
            decode_ctx_inputs = self.decode_ctx_inputs
            # if prefill_ctx_inputs is None or self.prefill_plan.num_tiles == 0:
            if prefill_ctx_inputs is None:
                # if max_num_prefill_tiles is not provided,
                # pad to unroll size to avoid shape error
                max_num_prefill_tiles = (
                    self.dynamic_loop_unrolling_size
                    if self.max_num_prefill_tiles is None
                    else self.max_num_prefill_tiles
                )
                device = "nkipy" if str(query.device).startswith("nkipy") else None
                prefill_ctx_inputs = ContextAttnInputs.create_empty(
                    max_num_tiles=max_num_prefill_tiles,
                    q_tile_size=self.large_q_tile_size,
                    kv_tile_size=self.large_kv_tile_size,
                    block_size=self.block_size,
                    is_prefill=True,
                    max_num_q_tiles=B_P_SIZE,
                    device=device,
                )
            if decode_ctx_inputs is None:
                device = "nkipy" if str(query.device).startswith("nkipy") else None
                decode_ctx_inputs = ContextAttnInputs.create_empty(
                    max_num_tiles=self.max_num_decode_tiles,
                    q_tile_size=1,
                    kv_tile_size=self.large_kv_tile_size,
                    block_size=self.block_size,
                    is_prefill=False,
                    max_num_q_tiles=B_P_SIZE,
                    device=device,
                )
            input_kwargs.update(prefill_ctx_inputs.as_dict(prefix="prefill_"))
            input_kwargs.update(decode_ctx_inputs.as_dict(prefix="decode_"))
            self.kernel_func = flash_attn_varlen_nkifunc
        elif self.prefill_batch_size == self.batch_size:
            prefill_ctx_inputs = self.prefill_ctx_inputs
            if self.prefill_plan.num_tiles == 0:
                # if max_num_prefill_tiles is not provided,
                # pad to unroll size to avoid shape error
                max_num_prefill_tiles = (
                    self.dynamic_loop_unrolling_size
                    if self.max_num_prefill_tiles is None
                    else self.max_num_prefill_tiles
                )
                device = "nkipy" if str(query.device).startswith("nkipy") else None
                prefill_ctx_inputs = ContextAttnInputs.create_empty(
                    max_num_tiles=max_num_prefill_tiles,
                    q_tile_size=self.large_q_tile_size,
                    kv_tile_size=self.large_kv_tile_size,
                    block_size=self.block_size,
                    is_prefill=True,
                    max_num_q_tiles=B_P_SIZE,
                    device=device,
                )
            input_kwargs["decode_mode"] = False
            input_kwargs.update(prefill_ctx_inputs.as_dict())
            self.kernel_func = (
                flash_attn_varlen_blocksparse_nkifunc
                if not self.kv_cache
                else "flash_attn_varlen_blocksparse_kv_cache_nkifunc"
            )
        else:
            assert self.prefill_batch_size == 0
            input_kwargs["decode_mode"] = True
            input_kwargs.update(self.decode_ctx_inputs.as_dict())
            self.kernel_func = (
                flash_attn_varlen_blocksparse_nkifunc
                if not self.kv_cache
                else "flash_attn_varlen_blocksparse_kv_cache_nkifunc"
            )

        return input_kwargs

    def _run_nki_xla(self, **input_kwargs):
        compiler_flags = _get_default_compiler_flags(self.save_artifact)
        compiler_flags.append(" --retry_failed_compilation")
        os.environ["NEURON_CC_FLAGS"] = " ".join(compiler_flags)

        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        kernel_kwargs = {
            arg_name: (arg.to(device) if isinstance(arg, torch.Tensor) else arg)
            for arg_name, arg in input_kwargs.items()
        }
        output_nki = self.kernel_func(**kernel_kwargs)

        # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
        output_nki = output_nki.cpu()
        if self.save_artifact:
            artifact_dir = "./_artifacts"
            os.makedirs(artifact_dir, exist_ok=True)
            save_kernel_tensors_as_npy(
                artifact_dir, is_torch_tensor=True, golden=output_nki, **input_kwargs
            )

        output_nki = output_nki.permute(0, 2, 1, 3)
        output_nki = output_nki[0, : self.num_actual_tokens, :, :]
        return output_nki

    def _run_nki_numpy(self, **input_kwargs):
        compiler_flags_str = " ".join(_get_default_compiler_flags(self.save_artifact))
        os.environ["NEURON_CC_FLAGS"] = compiler_flags_str
        if self.save_artifact:
            artifact_dir = "./_artifacts"
            os.makedirs(artifact_dir, exist_ok=True)
            output_nki = self.kernel_func(
                save_artifact_dir=artifact_dir,
                **input_kwargs,
            )
            save_kernel_tensors_as_npy(
                artifact_dir,
                is_torch_tensor=False,
                golden=output_nki,
                **input_kwargs,
            )
        else:
            output_nki = self.kernel_func(**input_kwargs)

        # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
        output_nki = output_nki.transpose(0, 2, 1, 3)
        output_nki = output_nki[0, : self.num_actual_tokens, :, :]
        return output_nki

    def _run_nki_dynamo(self, **input_kwargs):
        device = "nkipy"
        kernel_kwargs = {
            arg_name: (arg.to(device) if isinstance(arg, torch.Tensor) else arg)
            for arg_name, arg in input_kwargs.items()
        }
        custom_op = _get_nki_custom_op(self.kernel_func)
        output_nki = custom_op(**kernel_kwargs)

        # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
        # output_nki = output_nki.cpu()
        # if self.save_artifact:
        #     artifact_dir = "./_artifacts"
        #     os.makedirs(artifact_dir, exist_ok=True)
        #     save_kernel_tensors_as_npy(
        #         artifact_dir, is_torch_tensor=True, golden=output_nki, **input_kwargs
        #     )

        output_nki = output_nki.permute(0, 2, 1, 3)
        # output_nki = output_nki[0, : self.num_actual_tokens, :, :]
        output_nki = output_nki[0]
        return output_nki

    def __call__(
        self,
        query,
        k_active,
        v_active,
        kv_cache,
        sinks=None,
        mixed_precision=True,
        force_use_varlen=False,
    ):
        if isinstance(kv_cache, tuple):
            k_cache, v_cache = kv_cache
            _, num_kv_heads, _, head_size = k_cache.shape
        else:
            _, _, num_kv_heads, _, head_size = kv_cache.shape
        input_kwargs = self._prepare_kernel(
            query,
            k_active,
            v_active,
            kv_cache,
            sinks,
            head_size,
            num_kv_heads,
            mixed_precision,
            force_use_varlen,
        )
        if self.exec_mode == "dynamo":
            return self._run_nki_dynamo(**input_kwargs)
        else:
            assert False


# # prepare plan based on sequence lengths
# nki_kernel_runner = NKIFlashPagedAttentionRunner(
#     query_lens, torch.ones(decode_batch_size, dtype=torch.long) for decode,
#     context_lens, (bs,), with the real context lens, no padded
#     large_q_tile_size, =1 for decode,
#         large_q_tile_size=32 for test_prefill_with_decode
#     large_kv_tile_size, (4096, 32)
#     block_size,
#     dynamic_loop_unrolling_size=dynamic_loop_unrolling_size,
#     enable_separate_prefill_decode=enable_separate_prefill_decode,
# )

# # query_lens:
# if prefill_batch_size == 0:
#     query_lens = torch.ones(decode_batch_size, dtype=torch.long)
# else:
#     prefill_query_lens = _sample_lengths(
#         prefill_batch_size, min_query_len,
#         max_query_len)
#     decode_query_lens = torch.ones(decode_batch_size, dtype=torch.long)
#     query_lens = torch.cat([prefill_query_lens, decode_query_lens])
# unified way: query_lens = S
# query_lens: num_scheduled_tokens_per_req
# context_lens: self.input_batch.num_computed_tokens_cpu
#

#
# # Prepare context token tile plan inputs
# nki_kernel_runner.prepare_tile_plan_inputs(
#     block_tables=block_table,
#     max_kv_cache_size=k_cache.shape[0],
# )

# block_table:
# max_block_per_request = ceil_div(max_model_len, block_size)
# cache_size = (batch_size * max_block_per_request) + 128
# block_tables = torch.randperm(cache_size, dtype=torch.int32) # (cache_size,)
# block_tables = block_tables[: batch_size * max_block_per_request].view(
#         batch_size, max_block_per_request
#     )

# k_cache:
# k_cache = torch.zeros(cache_size, block_size, num_kv_heads, head_size, dtype=dtype)
# k_cache.shape[0] = cache_size

# query_lens, context_lens separate by batch

# output_nki = nki_kernel_runner(
#     query=query,  (num_tokens, num_heads, head_size)
#     k_cache=k_cache, (cache_size, block_size, num_kv_heads, head_size)
#     v_cache=v_cache,
#     k_active=k_active, (sum(query_lens), num_kv_heads, head_size)
#     v_active=v_active,
#     mixed_precision=mixed_precision, True
#     force_use_varlen=force_use_varlen, False
# )


# excute it
# pad and change to kernel layout
# num_active_token_after_padding = (
#     nki_kernel_runner.get_num_active_tokens_after_padding()
# )
# pad_dims = (
#     0,
#     0,
#     0,
#     0,
#     0,
#     num_active_token_after_padding - query.shape[0],
# )
# query = F.pad(query, pad_dims, "constant", 0)
# k_active = F.pad(k_active, pad_dims, "constant", 0)
# v_active = F.pad(v_active, pad_dims, "constant", 0)
# # permute QKV tensors
# # query: (1, n_heads, seq_q, d)
# query = query.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
# # key:   (1, n_kv_heads, d, seq_k)
# k_active = k_active.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
# # value: (1, n_kv_heads, seq_v, d)
# v_active = v_active.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
# k_cache (cache_size, num_kv_heads, block_size, head_size)
# k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
# v_cache = v_cache.permute(0, 2, 1, 3).contiguous()
