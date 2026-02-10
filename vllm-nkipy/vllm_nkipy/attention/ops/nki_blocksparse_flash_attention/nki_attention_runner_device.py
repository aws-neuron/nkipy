# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
Lightweight device-ready version of NKIFlashPagedAttentionRunner.
This class stores only the essential execution data and can be transferred to device.
"""

from typing import Optional

import torch

from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.constants import (  # noqa: E501
    B_P_SIZE,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.flash_pa_with_schedule import (  # noqa: E501
    flash_attn_varlen_blocksparse_nkifunc,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.flash_paged_attn_varlen import (  # noqa: E501
    flash_attn_varlen_nkifunc,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.nki_attention_runner import (  # noqa: E501
    ContextAttnInputs,
    _get_nki_custom_op,
)
from vllm_nkipy.config import PagedAttnImpl, _get_paged_attn_impl


class NKIFlashPagedAttentionRunnerForDevice:
    """
    Lightweight version of NKIFlashPagedAttentionRunner that only stores
    essential execution data and can be transferred to device.
    Only supports dynamo execution mode.
    """

    def __init__(
        self,
        prefill_ctx_inputs: Optional[ContextAttnInputs],
        decode_ctx_inputs: Optional[ContextAttnInputs],
        active_mask: Optional[torch.Tensor],
        num_active_tokens_after_padding: int,
        large_q_tile_size: int,
        large_kv_tile_size: int,
        block_size: int,
        dynamic_loop_unrolling_size: int,
        skip_active: bool,
        max_num_prefill_tiles: Optional[int],
        max_num_decode_tiles: Optional[int],
        prefill_plan_num_tiles: int,
    ):
        """
        Initialize with essential execution data.

        Args:
            prefill_ctx_inputs: Context inputs for prefill phase
            decode_ctx_inputs: Context inputs for decode phase
            active_mask: Active token mask
            num_active_tokens_after_padding: Number of tokens after padding
            large_q_tile_size: Query tile size
            large_kv_tile_size: KV tile size
            block_size: Block size for KV cache
            dynamic_loop_unrolling_size: Loop unrolling factor
            skip_active: Whether to skip active computation
            max_num_prefill_tiles: Maximum number of prefill tiles
            max_num_decode_tiles: Maximum number of decode tiles
            prefill_plan_num_tiles: Number of tiles in prefill plan
        """
        self.prefill_ctx_inputs = prefill_ctx_inputs
        self.decode_ctx_inputs = decode_ctx_inputs
        self.active_mask = active_mask
        self.num_active_tokens_after_padding = num_active_tokens_after_padding
        self.large_q_tile_size = large_q_tile_size
        self.large_kv_tile_size = large_kv_tile_size
        self.block_size = block_size
        self.dynamic_loop_unrolling_size = dynamic_loop_unrolling_size
        self.skip_active = skip_active
        self.max_num_prefill_tiles = max_num_prefill_tiles
        self.max_num_decode_tiles = max_num_decode_tiles
        self.prefill_plan_num_tiles = prefill_plan_num_tiles

        # Kernel function will be set during execution
        self.kernel_func = None

        self.kv_cache = (
            _get_paged_attn_impl()
            == PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION_KV_CACHE
        )

    def to(self, device):
        """Move all tensor data to the specified device."""
        if self.prefill_ctx_inputs is not None:
            self.prefill_ctx_inputs = self.prefill_ctx_inputs.to(device)
        if self.decode_ctx_inputs is not None:
            self.decode_ctx_inputs = self.decode_ctx_inputs.to(device)
        if self.active_mask is not None:
            self.active_mask = self.active_mask.to(device)
        return self

    def get_num_active_tokens_after_padding(self):
        """Return the number of tokens after padding."""
        return self.num_active_tokens_after_padding

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
        """Prepare kernel inputs and determine which kernel to use."""
        num_heads = query.shape[1]
        num_active_token = self.num_active_tokens_after_padding

        # Validate input shapes
        assert query.shape[2] == num_active_token, (
            "QKV sequence length must be padded to"
            f" {self.num_active_tokens_after_padding}"
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

        # Determine kernel function and add appropriate
        # inputs based on batch composition
        # Check if we have mixed prefill/decode or need varlen kernel
        has_prefill = self.prefill_ctx_inputs is not None
        has_decode = self.decode_ctx_inputs is not None

        if force_use_varlen or (has_prefill and has_decode):
            # Use varlen kernel for mixed prefill/decode
            prefill_ctx_inputs = self.prefill_ctx_inputs
            decode_ctx_inputs = self.decode_ctx_inputs

            if prefill_ctx_inputs is None:
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

        elif has_prefill:
            # Prefill only
            prefill_ctx_inputs = self.prefill_ctx_inputs
            if self.prefill_plan_num_tiles == 0:
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
            # Decode only
            input_kwargs["decode_mode"] = True
            input_kwargs.update(self.decode_ctx_inputs.as_dict())
            self.kernel_func = (
                flash_attn_varlen_blocksparse_nkifunc
                if not self.kv_cache
                else "flash_attn_varlen_blocksparse_kv_cache_nkifunc"
            )

        return input_kwargs

    def _run_nki_dynamo(self, **input_kwargs):
        """Run kernel in dynamo mode."""
        device = "nkipy"
        kernel_kwargs = {
            arg_name: (arg.to(device) if isinstance(arg, torch.Tensor) else arg)
            for arg_name, arg in input_kwargs.items()
        }
        custom_op = _get_nki_custom_op(self.kernel_func)
        output_nki = custom_op(**kernel_kwargs)

        # Permute output: (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
        output_nki = output_nki.permute(0, 2, 1, 3)
        # Return full output (batch_size=1, so just index [0])
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
        """
        Execute the attention kernel.

        Args:
            query: Query tensor (1, num_heads, seq_q, head_size)
            k_active: Active key tensor (1, num_kv_heads, head_size, seq_k)
            v_active: Active value tensor (1, num_kv_heads, seq_v, head_size)
            kv_cache: KV cache tensor (either separate
                k_cache, v_cache tuple or combined kv_cache)
            sinks: Sink tokens (optional)
            mixed_precision: Whether to use mixed precision
            force_use_varlen: Force use of varlen kernel

        Returns:
            Output tensor with attention results
        """
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

        return self._run_nki_dynamo(**input_kwargs)
