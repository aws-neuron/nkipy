# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder

from vllm_nkipy.attention.ops.nki_flash_attn.flash_pa_with_schedule import (
    flash_attn_varlen_nki,
)
from vllm_nkipy.attention.ops.nki_flash_attn_legacy import (
    flash_paged_attention,
    reshape_and_cache2,
)
from vllm_nkipy.attention.ops.nki_paged_kv_cache_wrapper import (
    update_kv_cache_custom_op,
)
from vllm_nkipy.attention.ops.paged_kv_cache import reshape_and_cache_kv1
from vllm_nkipy.attention.ops.torch_flash_attn import reshape_and_cache_torch
from vllm_nkipy.attention.ops.torch_ragged_attn import torch_ragged_paged_attention
from vllm_nkipy.config import (
    PagedAttnImpl,
    PagedKvImpl,
    _get_paged_attn_impl,
    _get_paged_kv_impl,
)

logger = init_logger(__name__)


@torch.library.custom_op("mylib::flash_attn_varlen", mutates_args=())
def flash_attn_varlen(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: torch.Tensor,
    num_seqs: torch.Tensor,
    block_tables: torch.Tensor,
    enable_prefill: bool,
) -> torch.Tensor:
    N, _, n_kv_head, _, _ = kv_cache.shape
    return flash_attn_varlen_nki[1, n_kv_head](
        query,
        kv_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        num_seqs,
        block_tables,
        enable_prefill,
    )


@flash_attn_varlen.register_fake
def _(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: torch.Tensor,
    num_seqs: torch.Tensor,
    block_tables: torch.Tensor,
    enable_prefill: bool,
) -> torch.Tensor:
    return torch.empty_like(query.squeeze(0))


def ragged_neuron_paged_attn(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_seqs: torch.Tensor,
    enable_prefill: bool,
) -> torch.Tensor:
    _, _, _, block_size, head_size = kv_cache.shape

    (max_num_seqs,) = seq_lens.shape
    query_lens = query_start_loc[1 : max_num_seqs + 1] - query_start_loc[0:max_num_seqs]
    context_lens = seq_lens - query_lens

    # Calculate number of blocks per sequence
    num_blocks_per_seq = torch.floor(
        torch.div((context_lens + block_size - 1), block_size)
    ).to(context_lens.dtype)

    # zero = torch.tensor([0],
    #                     dtype=context_lens.dtype,
    #                     device=context_lens.device)
    # cu_ctx_lens_blockaligned = (torch.cat(
    #     (zero, num_blocks_per_seq), dim=0) * block_size).cumsum(dim=0)
    cu_ctx_lens_blockaligned = (
        torch.constant_pad_nd(num_blocks_per_seq, (1, 0), 0)  # prepend a 0
        .mul(block_size)  # scale
        .cumsum(dim=0)  # cumulative sum
    )

    output = flash_attn_varlen(
        query=query,
        kv_cache=kv_cache,
        cu_seqlens_q=query_start_loc[: max_num_seqs + 1],
        cu_seqlens_k=cu_ctx_lens_blockaligned[: max_num_seqs + 1],
        seqused_k=context_lens,
        num_seqs=num_seqs,
        block_tables=block_tables,
        enable_prefill=enable_prefill,
    )

    return output


@torch.library.custom_op("mylib::neuron_paged_attn", mutates_args=())
def neuron_paged_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    attn_mask: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    _, n_kv_head, _, _ = key.shape
    output = flash_paged_attention[1, n_kv_head](
        query,
        key,
        value,
        kv_cache,
        block_tables,
        attn_mask,
        sinks,
    )
    return output


@neuron_paged_attn.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    attn_mask: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query.transpose(-2, -1))


class NeuronAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "NEURON"

    @staticmethod
    def get_impl_cls() -> type["NeuronAttentionBackendImpl"]:
        return NeuronAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> type["NeuronAttentionMetadata"]:
        return NeuronAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["NeuronAttentionMetadataBuilder"]:
        return NeuronAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        paged_attn_impl = _get_paged_attn_impl()
        if paged_attn_impl.value.startswith("torch"):
            return (2, num_kv_heads, num_blocks, block_size, head_size)
        return (2, num_blocks, num_kv_heads, block_size, head_size)


from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.nki_attention_runner import (  # noqa: E402, E501
    NKIFlashPagedAttentionRunner,
)


@dataclass
class NeuronAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    num_seqs: torch.Tensor  # Number of actual sequences
    enable_prefill: int

    active_block_table: torch.Tensor = None
    attn_mask: torch.Tensor = None
    nki_kernel_runner: NKIFlashPagedAttentionRunner = None


class NeuronAttentionMetadataBuilder(
    AttentionMetadataBuilder[NeuronAttentionMetadata]
): ...


class NeuronAttentionBackendImpl(AttentionImpl[NeuronAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[list[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        sinks: torch.Tensor = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "KV sharing is not supported in NeuronAttentionBackend."
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.sinks = sinks
        # logger.info(f"[DEBUG] {self.sinks.device=}")
        # logger.info(f"[DEBUG] {type(self.sinks)=}")

    def _update_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        use_int32_slot_mapping: bool = False,
    ) -> torch.Tensor:
        """Update KV cache based on the configured implementation."""
        paged_kv_impl = _get_paged_kv_impl()
        if paged_kv_impl == PagedKvImpl.RESHAPE_AND_CACHE2:
            reshape_and_cache2(key, value, kv_cache, slot_mapping)
        elif paged_kv_impl == PagedKvImpl.RESHAPE_AND_CACHE_KV1:
            reshape_and_cache_kv1(key, value, kv_cache, slot_mapping)
        elif paged_kv_impl == PagedKvImpl.UPDATE_KV_CACHE_CUSTOM_OP:
            if use_int32_slot_mapping:
                slot_mapping = slot_mapping.to(dtype=torch.int32)
            kv_cache = update_kv_cache_custom_op(key, value, kv_cache, slot_mapping)
        # else: skip KV cache update when paged_kv_impl == PagedKvImpl.SKIP
        return kv_cache

    def forward_nki_ragged(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: NeuronAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[-2]
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if kv_cache.numel() > 0:
            slot_mapping = attn_metadata.slot_mapping
            self._update_kv_cache(key, value, kv_cache, slot_mapping)
        else:
            # profiling run
            return query

        query = query.view(num_tokens, self.num_heads, self.head_size)
        query = query.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

        output = ragged_neuron_paged_attn(
            query,
            kv_cache,
            attn_metadata.seq_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.num_seqs,
            attn_metadata.enable_prefill,
        )
        output = output.transpose(0, 1).reshape(
            num_tokens, self.num_heads * self.head_size
        )
        return output

    def forward_nki_masked(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: NeuronAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[-2]
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if kv_cache.numel() > 0:
            slot_mapping = attn_metadata.slot_mapping
            self._update_kv_cache(key, value, kv_cache, slot_mapping)
        else:
            # profiling run
            return query

        query = query.view(num_tokens, self.num_heads, self.head_size)
        query = query.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
        key = key.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
        value = value.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

        output = neuron_paged_attn(
            query,
            key,
            value,
            kv_cache,
            attn_metadata.active_block_table,
            attn_metadata.attn_mask,
            self.sinks,
        )
        output = output.transpose(1, 2).reshape(
            num_tokens, self.num_heads * self.head_size
        )
        return output

    def forward_nki_blocksparse_flash_attention(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: NeuronAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # forward_nki_masked, cache update
        num_tokens = query.shape[-2]
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if kv_cache.numel() > 0:
            slot_mapping = attn_metadata.slot_mapping
            kv_cache = self._update_kv_cache(
                key, value, kv_cache, slot_mapping, use_int32_slot_mapping=True
            )
        else:
            # profiling run
            return query

        padded_num_tokens = (
            attn_metadata.nki_kernel_runner.get_num_active_tokens_after_padding()
        )
        query = F.pad(query, (0, 0, 0, padded_num_tokens - num_tokens), "constant", 0)
        key = F.pad(key, (0, 0, 0, 0, 0, padded_num_tokens - num_tokens), "constant", 0)
        value = F.pad(
            value, (0, 0, 0, 0, 0, padded_num_tokens - num_tokens), "constant", 0
        )

        # forward_nki_masked, kernel
        query = query.view(padded_num_tokens, self.num_heads, self.head_size)
        query = query.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        key = key.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
        value = value.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

        paged_attn_impl = _get_paged_attn_impl()
        if paged_attn_impl == PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION:
            k_cache = kv_cache[0]
            v_cache = kv_cache[1]

            output = attn_metadata.nki_kernel_runner(
                query=query,
                k_active=key,
                v_active=value,
                kv_cache=(k_cache, v_cache),
                sinks=self.sinks,
                mixed_precision=True,
                force_use_varlen=False,
            )  # (num_tokens, h, d)
        elif paged_attn_impl == PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION_KV_CACHE:
            output = attn_metadata.nki_kernel_runner(
                query=query,
                k_active=key,
                v_active=value,
                kv_cache=kv_cache,
                sinks=self.sinks,
                mixed_precision=True,
                force_use_varlen=False,
            )  # (num_tokens, h, d)
        else:
            assert False
        output = output.contiguous().view(num_tokens, -1)
        return output

    def forward_torch_ragged(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: NeuronAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[-2]
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if kv_cache.numel() > 0:
            slot_mapping = attn_metadata.slot_mapping
            reshape_and_cache_torch(key, value, kv_cache, slot_mapping)
            # reshape_and_cache2(key, value, kv_cache, slot_mapping) shape mismatch
        else:
            # profiling run
            return query

        query_lens = torch.diff(attn_metadata.query_start_loc)

        query = query.view(num_tokens, self.num_heads, self.head_size)
        output = torch_ragged_paged_attention(
            query,
            kv_cache,
            attn_metadata.seq_lens - query_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
        )
        output = output.transpose(0, 1).reshape(
            num_tokens, self.num_heads * self.head_size
        )
        return output

    def forward_torch_masked(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: NeuronAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[-2]
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if kv_cache.numel() > 0:
            slot_mapping = attn_metadata.slot_mapping
            reshape_and_cache_torch(key, value, kv_cache, slot_mapping)
        else:
            # profiling run
            return query

        query = query.view(num_tokens, self.num_heads, self.head_size)
        query = query.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        key = key.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        value = value.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

        # FIXME: test or remove
        # if self.sinks is None:
        #     output = neuron_paged_attn_torch(
        #         query,
        #         key,
        #         value,
        #         kv_cache,
        #         attn_metadata.active_block_table,
        #         attn_metadata.attn_mask,
        #     )
        # else:
        output = neuron_paged_attn_torch_gpt_oss(  # noqa: F821
            query,
            key,
            value,
            kv_cache,
            attn_metadata.active_block_table,
            attn_metadata.attn_mask,
            self.sinks,
        )
        output = output.transpose(1, 2).reshape(
            num_tokens, self.num_heads * self.head_size
        )
        return output

    @torch.no_grad()
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: NeuronAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        paged_attn_impl = _get_paged_attn_impl()
        if paged_attn_impl == PagedAttnImpl.NKI_RAGGED:
            return self.forward_nki_ragged(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                k_scale,
                v_scale,
                attn_type,
                output,
            )
        elif paged_attn_impl == PagedAttnImpl.TORCH_MASKED:
            return self.forward_torch_masked(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                k_scale,
                v_scale,
                attn_type,
                output,
            )
        elif paged_attn_impl == PagedAttnImpl.TORCH_RAGGED:
            return self.forward_torch_ragged(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                k_scale,
                v_scale,
                attn_type,
                output,
            )
        elif paged_attn_impl == PagedAttnImpl.NKI_MASKED:
            return self.forward_nki_masked(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                k_scale,
                v_scale,
                attn_type,
                output,
            )
        elif paged_attn_impl in (
            PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION,
            PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION_KV_CACHE,
        ):
            return self.forward_nki_blocksparse_flash_attention(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                k_scale,
                v_scale,
                attn_type,
                output,
            )
        else:
            raise RuntimeError(
                f"Unknown attention kernel implementation: {paged_attn_impl}"
            )
