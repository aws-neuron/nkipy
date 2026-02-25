# import blockwise_index_cpp
import json
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import parallel_state
import torch.distributed as dist
from collective import all_gather
from config import Config
from nkipy.core.trace import NKIPyKernel
from nkipy.runtime.device_kernel import DeviceKernel
from nkipy.runtime.device_tensor import DeviceTensor
from kernels.attention import attention_module
from kernels.blockwise_index import BLOCK_SIZE, ControlType
from kernels.blockwise_nki import (
    blockwise_add_residual,
    output_init,
)
from kernels.router import expert_affinities_slice, rmsnorm_router
from logger import get_logger
from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType
from parallel_state import get_tp_size
from utils import assert_allclose

logger = get_logger()


trace = NKIPyKernel.trace(backend="hlo")

class PrefillAttentionLayer:
    # Class-level cache for tensor reuse across multiple layers
    allocated_tensors = {}

    def __init__(
        self,
        *,
        input_weight,
        qkv_weight,
        cache_k,
        cache_v,
        o_weight,
        sliding_window,
        sink,
        qkv_bias,
        o_bias,
        cos,
        sin,
        config,
        name_prefix="attention",
    ):
        compute_dtype = config.dtype
        self.sliding_window = sliding_window
        hidden_states = np.empty(
            (
                config.max_batch_size_per_dp * config.max_model_len // get_tp_size(),
                config.hidden_size,
            ),
            dtype=compute_dtype,
        )
        self.attention = DeviceKernel.compile_and_load(
            kernel=trace(attention_module),
            hidden_states=hidden_states,
            input_weight=input_weight,
            qkv_weight=qkv_weight,
            cache_k=cache_k,
            cache_v=cache_v,
            o_weight=o_weight,
            sliding_window=sliding_window,
            sink=sink,
            qkv_bias=qkv_bias,
            o_bias=o_bias,
            start_pos=None,
            cos=cos,
            sin=sin,
            config=config,
            compute_dtype=compute_dtype,
            is_neuronpy=True,
            name=f"{name_prefix}_sw{sliding_window}",
        )

    def forward(
        self,
        hidden_states_shard,
        input_weight,
        qkv_weight,
        cache_k,
        cache_v,
        o_weight,
        sink,
        qkv_bias,
        o_bias,
        cos,
        sin,
        save_trace=False,
    ):
        self.attention(
            inputs={
                "hidden_states": hidden_states_shard,
                "input_weight": input_weight,
                "qkv_weight": qkv_weight,
                "cache_k.must_alias_input": cache_k,
                "cache_v.must_alias_input": cache_v,
                "o_weight": o_weight,
                "sink": sink,
                "qkv_bias": qkv_bias,
                "o_bias": o_bias,
                "cos": cos,
                "sin": sin,
            },
            outputs={
                "output0": hidden_states_shard,
                "cache_k": cache_k,
                "cache_v": cache_v,
            },
            save_trace=save_trace,
        )


class PrefillLayer:
    # Class-level cache for tensor reuse across multiple layers
    tensors_shared_across_layers = {}
    # reference tensors to compare with
    reference_tensors = None

    def __init__(
        self,
        layer_id: int,
        config: Config,
        sliding_window: int,
        sink: DeviceTensor,
        qkv_weight: DeviceTensor,
        qkv_bias: DeviceTensor,
        o_weight: DeviceTensor,
        o_bias: DeviceTensor,
        input_weight: DeviceTensor,
        cos: DeviceTensor,
        sin: DeviceTensor,
        post_attention_weight: DeviceTensor,
        router_weight: DeviceTensor,
        router_bias: DeviceTensor,
        gate_up_proj_weight: DeviceTensor,
        down_weight: DeviceTensor,
        gate_up_bias_plus1_T: DeviceTensor,
        down_bias_broadcasted: DeviceTensor,
        cache_k: DeviceTensor,
        cache_v: DeviceTensor,
        ep_rank: DeviceTensor,
        check_against_reference: bool,
    ):
        self.layer_id = layer_id
        self.config = config
        self.sliding_window = sliding_window
        # Store weights as DeviceTensors (inputs are already DeviceTensors)
        self.sink = sink
        self.qkv_weight = qkv_weight
        self.qkv_bias = qkv_bias
        self.o_weight = o_weight
        self.o_bias = o_bias
        self.input_weight = input_weight
        self.cos = cos
        self.sin = sin
        self.post_attention_weight = post_attention_weight
        self.router_weight = router_weight
        self.router_bias = router_bias
        self.gate_up_proj_weight = gate_up_proj_weight
        self.down_weight = down_weight
        self.gate_up_bias_plus1_T = gate_up_bias_plus1_T
        self.down_bias_broadcasted = down_bias_broadcasted
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.activation_function = ActFnType.Swish
        self.ep_rank = ep_rank

        # shared tensors across layers
        if layer_id == 0:
            PrefillLayer.tensors_shared_across_layers[
                "moe_input_hidden_states_sharded"
            ] = DeviceTensor.from_numpy(
                np.zeros(
                    (
                        config.max_batch_size_per_dp
                        * config.max_model_len
                        // get_tp_size(),
                        config.hidden_size,
                    ),
                    dtype=config.dtype,
                ),
                "moe_input_hidden_states_sharded",
            )
            PrefillLayer.tensors_shared_across_layers["top_k_indices"] = (
                DeviceTensor.from_numpy(
                    np.zeros(
                        (
                            parallel_state.get_prefill_ep_size()
                            * config.max_batch_size_per_dp
                            * config.max_model_len,
                            config.num_experts_per_tok,
                        ),
                        dtype=np.int8,
                    ),
                    "top_k_indices",
                )
            )
            PrefillLayer.tensors_shared_across_layers[
                "expert_affinities_masked_all_experts"
            ] = DeviceTensor.from_numpy(
                np.zeros(
                    (
                        parallel_state.get_prefill_ep_size()
                        * config.max_batch_size_per_dp
                        * config.max_model_len,
                        config.num_experts,
                    ),
                    dtype=config.dtype,
                ),
                "expert_affinities_masked_all_experts",
            )
            PrefillLayer.tensors_shared_across_layers["expert_affinities_masked"] = (
                DeviceTensor.from_numpy(
                    np.zeros(
                        (
                            parallel_state.get_prefill_ep_size()
                            * config.max_batch_size_per_dp
                            * config.max_model_len,
                            config.num_experts // parallel_state.get_prefill_ep_size(),
                        ),
                        dtype=config.dtype,
                    ),
                    "expert_affinities_masked",
                )
            )
            PrefillLayer.tensors_shared_across_layers[
                "expert_affinities_masked_sharded"
            ] = DeviceTensor.from_numpy(
                np.zeros(
                    (
                        config.max_batch_size_per_dp
                        * config.max_model_len
                        // get_tp_size(),
                        config.num_experts,
                    ),
                    dtype=config.dtype,
                ),
                "expert_affinities_masked_sharded",
            )
            PrefillLayer.tensors_shared_across_layers["moe_input_hidden_states"] = (
                DeviceTensor.from_numpy(
                    np.zeros(
                        (
                            parallel_state.get_prefill_ep_size()
                            * config.max_batch_size_per_dp
                            * config.max_model_len,
                            config.hidden_size,
                        ),
                        dtype=config.dtype,
                    ),
                    "moe_input_hidden_states",
                )
            )
            PrefillLayer.tensors_shared_across_layers["moe_output_hidden_states"] = (
                DeviceTensor.from_numpy(
                    np.zeros(
                        (
                            parallel_state.get_prefill_ep_size()
                            * config.max_batch_size_per_dp
                            * config.max_model_len,
                            config.hidden_size,
                        ),
                        dtype=config.dtype,
                    ),
                    "moe_output_hidden_states",
                )
            )
            # self.allocated_tensors["start_block_idx"] = DeviceTensor.from_numpy(
            #     np.array([config.num_static_blocks], dtype=np.int32).reshape((1, 1)),
            #     "start_block_idx",
            # )
            if check_against_reference:
                with open("reference.json", "r") as f:
                    PrefillLayer.reference_tensors = json.load(f)

        self.moe_input_hidden_states_sharded = PrefillLayer.tensors_shared_across_layers["moe_input_hidden_states_sharded"]
        self.top_k_indices = PrefillLayer.tensors_shared_across_layers["top_k_indices"]
        self.expert_affinities_masked = PrefillLayer.tensors_shared_across_layers["expert_affinities_masked"]
        self.expert_affinities_masked_sharded = PrefillLayer.tensors_shared_across_layers[
            "expert_affinities_masked_sharded"
        ]
        self.expert_affinities_masked_all_experts = PrefillLayer.tensors_shared_across_layers[
            "expert_affinities_masked_all_experts"
        ]
        self.moe_input_hidden_states = PrefillLayer.tensors_shared_across_layers[
            "moe_input_hidden_states"
        ]
        self.moe_output_hidden_states = PrefillLayer.tensors_shared_across_layers["moe_output_hidden_states"]
        # self.start_block_idx = self.allocated_tensors["start_block_idx"]

        self.kernel_attention = PrefillAttentionLayer(
            input_weight=self.input_weight,
            qkv_weight=self.qkv_weight,
            cache_k=self.cache_k,
            cache_v=self.cache_v,
            o_weight=self.o_weight,
            sliding_window=self.sliding_window,
            sink=self.sink,
            qkv_bias=self.qkv_bias,
            o_bias=self.o_bias,
            cos=self.cos,
            sin=self.sin,
            config=self.config,
        )
        self.kernel_rmsnorm_router = DeviceKernel.compile_and_load(
            kernel=trace(rmsnorm_router),
            hidden_states_sharded=self.moe_input_hidden_states_sharded,
            post_attention_weight=self.post_attention_weight,
            router_weight=self.router_weight,
            router_bias=self.router_bias,
            norm_eps=self.config.norm_eps,
            top_k=self.config.num_experts_per_tok,
            is_neuronpy=True,
        )
        self.kernel_expert_affinities_all_gather = DeviceKernel.compile_and_load(
            kernel=trace(all_gather),
            data=self.expert_affinities_masked_sharded,
            all_gather_dim=0,
            replica_groups=parallel_state.get_prefill_ep_world_group(),
            is_neuronpy=True,
            name="expert_affinities_all_gather",
        )
        self.kernel_expert_affinities_slice = DeviceKernel.compile_and_load(
            kernel=trace(expert_affinities_slice),
            expert_affinities_masked_all_experts=self.expert_affinities_masked_all_experts,
            ep_size=parallel_state.get_prefill_ep_size(),
            ep_rank=self.ep_rank,
        )
        self.kernel_hidden_states_all_gather = DeviceKernel.compile_and_load(
            kernel=trace(all_gather),
            data=self.moe_input_hidden_states_sharded,
            all_gather_dim=0,
            replica_groups=parallel_state.get_prefill_ep_world_group(),
            is_neuronpy=True,
            name="hidden_states_all_gather",
        )
        self.output_init = DeviceKernel.compile_and_load(
            trace(output_init),
            output=self.moe_output_hidden_states,
        )
        self.kernel_blockwise_add_residual = DeviceKernel.compile_and_load(
            trace(blockwise_add_residual),
            hidden_states=self.moe_input_hidden_states,
            residual_2d_shard=self.moe_input_hidden_states_sharded,
            output=self.moe_output_hidden_states,
            expert_affinities_masked_hbm=self.expert_affinities_masked,
            gate_up_proj_weight=self.gate_up_proj_weight,
            gate_up_bias_plus1_T_hbm=self.gate_up_bias_plus1_T,
            down_proj_weight=self.down_weight,
            down_bias_broadcasted_hbm=self.down_bias_broadcasted,
            token_position_to_id=np.empty((config.num_blocks, BLOCK_SIZE), dtype=np.int32),
            block_to_expert=np.empty((config.num_blocks,), dtype=np.int8),
            num_static_blocks=self.config.num_static_blocks,
            is_neuronpy=True,
        )
        from blockwise_index_cpp import get_blockwise_expert_and_token_mapping
        self.blockwise_index_kernel = get_blockwise_expert_and_token_mapping

    def forward(
        self,
        hidden_states_sharded: DeviceTensor,
        seq_len: npt.NDArray[np.int32],
        warmup: bool,
        batch_indices_to_check: Optional[List[int]] = None,
    ):
        """Execute the MoE layer forward pass"""
        # TODO: disable device profile because of oom
        save_trace = warmup and self.layer_id <= 1 and dist.get_rank() == 0
        compare_tensor_value_with_reference = (
            warmup
            and self.layer_id <= 1
            and batch_indices_to_check is not None
            and len(batch_indices_to_check) > 0
        )
        local_batch_indices_to_compare = []  # stores (token_start_offset, num_tokens_to_compare)
        if compare_tensor_value_with_reference:
            TP_SIZE = get_tp_size()
            tp_rank = parallel_state.get_tp_rank()
            if TP_SIZE <= self.config.max_batch_size_per_dp:
                # only batch dimension is sharded
                if self.config.max_batch_size_per_dp % TP_SIZE == 0:
                    reqs_per_tp_rank = self.config.max_batch_size_per_dp // TP_SIZE
                    for batch_idx in batch_indices_to_check:
                        assigned_tp_rank = batch_idx // reqs_per_tp_rank
                        if assigned_tp_rank == tp_rank:
                            local_batch_idx = batch_idx % reqs_per_tp_rank
                            local_offset = local_batch_idx * self.config.max_model_len
                            local_batch_indices_to_compare.append((local_offset, seq_len[batch_idx]))
                else:
                    compare_tensor_value_with_reference = False
            else:
                # both batch and seqlen are sharded
                if TP_SIZE % self.config.max_batch_size_per_dp == 0:
                    tp_ranks_per_req = TP_SIZE // self.config.max_batch_size_per_dp
                    per_tp_rank_seqlen = (
                        self.config.max_batch_size_per_dp
                        * self.config.max_model_len
                        // TP_SIZE
                    )
                    for batch_idx in batch_indices_to_check:
                        # only check requests whose prefill tokens are not spread across multiple nodes
                        if tp_rank == batch_idx * tp_ranks_per_req and seq_len[batch_idx] <= per_tp_rank_seqlen:
                            local_batch_indices_to_compare.append((0, seq_len[batch_idx]))
                else:
                    compare_tensor_value_with_reference = False

        if compare_tensor_value_with_reference:
            logger.info(f"Compare tensor value with reference. {self.layer_id=}")
        self.kernel_attention.forward(
            hidden_states_shard=hidden_states_sharded,
            input_weight=self.input_weight,
            qkv_weight=self.qkv_weight,
            cache_k=self.cache_k,
            cache_v=self.cache_v,
            o_weight=self.o_weight,
            sink=self.sink,
            qkv_bias=self.qkv_bias,
            o_bias=self.o_bias,
            cos=self.cos,
            sin=self.sin,
            save_trace=save_trace,
        )

        # TODO: test other layers. Currently error too large
        # Because of sequence parallel, we only compare tp rank 0
        if compare_tensor_value_with_reference:
            hidden_states_sharded_np = hidden_states_sharded.numpy()
            # TODO: check how attention reduce scatter works
            tp_rank = parallel_state.get_tp_rank()
            ep_rank = parallel_state.get_prefill_ep_rank()
            for token_offset, num_tokens in local_batch_indices_to_compare:
                print(f"After attention {ep_rank=} {tp_rank=} checking {token_offset=}, {num_tokens=}")
                assert_allclose(
                    hidden_states_sharded_np[
                        token_offset : token_offset + num_tokens,
                        :,
                    ],
                    np.array(
                        PrefillLayer.reference_tensors[f"{self.layer_id}_attn_out"],
                        dtype=Config.dtype,
                    ),
                )

        # FIXME: router top k indices are not correct beyond seq_len
        self.kernel_rmsnorm_router(
            inputs={
                "hidden_states_sharded": hidden_states_sharded,
                "post_attention_weight": self.post_attention_weight,
                "router_weight": self.router_weight,
                "router_bias": self.router_bias,
            },
            outputs={
                "output0": self.top_k_indices,
                "output1": self.expert_affinities_masked_sharded,
                "output2": self.moe_input_hidden_states_sharded,
            },
            save_trace=save_trace,
        )
        self.kernel_expert_affinities_all_gather(
            inputs={
                "data": self.expert_affinities_masked_sharded,
            },
            outputs={
                "output0": self.expert_affinities_masked_all_experts,
            },
            save_trace=save_trace,
        )
        self.kernel_expert_affinities_slice(
            inputs={
                "expert_affinities_masked_all_experts": self.expert_affinities_masked_all_experts,
                "ep_rank": self.ep_rank,
            },
            outputs={
                "output0": self.expert_affinities_masked,
            },
            save_trace=save_trace,
        )

        # overlap with CPU
        self.output_init(
            inputs={
                "output.must_alias_input": self.moe_output_hidden_states,
            },
            outputs={
                "output": self.moe_output_hidden_states,
            },
            save_trace=save_trace,
        )
        self.kernel_hidden_states_all_gather(
            inputs={
                "data": self.moe_input_hidden_states_sharded,
            },
            outputs={
                "output0": self.moe_input_hidden_states,
            },
            save_trace=save_trace,
        )

        # FIXME: fix moe indices to check
        moe_request_idx_to_check = 0
        if compare_tensor_value_with_reference:
            moe_input_hidden_states_np = self.moe_input_hidden_states.numpy()
            for e in range(parallel_state.get_prefill_ep_size()):
                for b in range(moe_request_idx_to_check, moe_request_idx_to_check + 1):
                    #  (ep_size, tp_size, config.max_batch_size, config.max_model_len // tp_size, config.hidden_size),
                    seq_offset = self.config.max_model_len * (
                        e * self.config.max_batch_size_per_dp + b
                    )
                    assert_allclose(
                        moe_input_hidden_states_np[seq_offset : seq_offset + seq_len[b]],
                        np.array(PrefillLayer.reference_tensors[f"{self.layer_id}_normed_moe_input"], dtype=Config.dtype),
                    )
        # TODO: varlen optimization
        top_k_indices = self.top_k_indices.numpy()

        if compare_tensor_value_with_reference:
            for e in range(parallel_state.get_prefill_ep_size()):
                for b in range(moe_request_idx_to_check, moe_request_idx_to_check + 1):
                    seq_offset = self.config.max_model_len * (
                        e * self.config.max_batch_size_per_dp + b
                    )
                    np.testing.assert_array_equal(
                        top_k_indices[seq_offset : seq_offset + seq_len[b]],
                        np.array(PrefillLayer.reference_tensors[f"{self.layer_id}_expert_indices"], dtype=np.int8),
                    )

        # TODO: top_k_indices not support inplace update
        top_k_indices = top_k_indices.copy()
        # mask out tokens out of current EP rank
        n_experts_per_ep = self.config.num_experts // parallel_state.get_prefill_ep_size()
        # current EP rank expert index starts from 0
        smaller_indices = top_k_indices < n_experts_per_ep * parallel_state.get_prefill_ep_rank()
        larger_indices = top_k_indices >= n_experts_per_ep * (parallel_state.get_prefill_ep_rank() + 1)
        top_k_indices[smaller_indices | larger_indices] = ControlType.SKIP_DMA.value
        top_k_indices[~(smaller_indices | larger_indices)] -= (
            n_experts_per_ep * parallel_state.get_prefill_ep_rank()
        )
        # FIXME: assume reqs' length is all identical
        for e in range(parallel_state.get_prefill_ep_size()):
            for b in range(self.config.max_batch_size_per_dp):
                seq_offset = self.config.max_model_len * (
                    e * self.config.max_batch_size_per_dp + b
                )
                top_k_indices[
                    seq_offset + seq_len[b] : seq_offset + self.config.max_model_len
                ] = ControlType.SKIP_DMA.value

        if compare_tensor_value_with_reference:
            # only compare current ep range
            expert_affinities_masked_ref = np.array(PrefillLayer.reference_tensors[f"{self.layer_id}_expert_weights"], dtype=Config.dtype)
            expert_affinities_masked_ref[(smaller_indices | larger_indices)[: seq_len[0]]] = 0
            expert_affinities_masked_np = self.expert_affinities_masked.numpy()
            for e in range(parallel_state.get_prefill_ep_size()):
                for b in range(moe_request_idx_to_check, moe_request_idx_to_check + 1):
                    seq_offset = self.config.max_model_len * (
                        e * self.config.max_batch_size_per_dp + b
                    )            
                    expert_affinities_masked = expert_affinities_masked_np[seq_offset : seq_offset + seq_len[b]].copy()
                    expert_affinities_masked = np.concatenate(
                        [expert_affinities_masked, np.zeros_like(expert_affinities_masked, shape=(seq_len[b], 1))], axis=1
                    )
                    assert_allclose(
                        np.take_along_axis(
                            expert_affinities_masked[:seq_len[b]],
                            top_k_indices[seq_offset:seq_offset+seq_len[b]],
                            axis=1,
                        ),
                        expert_affinities_masked_ref,
                        rtol=4 * 2**-7,
                    )
        # Use C++ blockwise index implementation for better performance
        _, block_to_expert, token_position_to_id = (
            self.blockwise_index_kernel(
                top_k_indices=top_k_indices,
            )
        )
        # from kernels.blockwise_index import get_blockwise_expert_and_token_mapping
        # num_real_blocks, block_to_expert, token_position_to_id = (
        #     get_blockwise_expert_and_token_mapping(
        #         top_k_indices=top_k_indices,
        #         num_blocks=self.config.num_blocks,
        #         block_size=BLOCK_SIZE,
        #         num_experts=n_experts_per_ep,
        #         num_static_blocks=self.config.num_static_blocks,
        #     )
        # )

        # logger.info(f"num_real_blocks: {num_real_blocks}")
        # np.sum(token_position_to_id[block_to_expert != -1] >= 0, axis=1) % 128
        token_position_to_id = DeviceTensor.from_numpy(token_position_to_id)
        block_to_expert = DeviceTensor.from_numpy(block_to_expert)

        # output = blockwise_add_residual(
        #     hidden_states=self.moe_input_hidden_states.numpy(),
        #     residual_2d_shard=hidden_states_sharded.numpy(),
        #     output=self.moe_output_hidden_states.numpy(),
        #     expert_affinities_masked_hbm=self.expert_affinities_masked.numpy(),
        #     gate_up_proj_weight=self.gate_up_proj_weight.numpy(),
        #     gate_up_bias_plus1_T_hbm=self.gate_up_bias_plus1_T.numpy(),
        #     down_proj_weight=self.down_weight.numpy(),
        #     down_bias_broadcasted_hbm=self.down_bias_broadcasted.numpy(),
        #     token_position_to_id=self.token_position_to_id.numpy(),
        #     block_to_expert=self.block_to_expert.numpy(),
        #     num_static_blocks=self.config.num_static_blocks,
        #     is_neuronpy=False,
        # )
        # self.moe_output_hidden_states.copy_from_numpy(output)

        self.kernel_blockwise_add_residual(
            inputs={
                "hidden_states": self.moe_input_hidden_states,
                "residual_2d_shard": hidden_states_sharded,
                "output.must_alias_input": self.moe_output_hidden_states,
                "expert_affinities_masked_hbm": self.expert_affinities_masked,
                "gate_up_proj_weight": self.gate_up_proj_weight,
                "gate_up_bias_plus1_T_hbm": self.gate_up_bias_plus1_T,
                "down_proj_weight": self.down_weight,
                "down_bias_broadcasted_hbm": self.down_bias_broadcasted,
                "token_position_to_id": token_position_to_id,
                "block_to_expert": block_to_expert,
            },
            outputs={
                "output0": hidden_states_sharded,
                "output": self.moe_output_hidden_states,
            },
            save_trace=save_trace,
        )

        # TODO: test other layers. Currently error too large
        # Because of sequence parallel, we only compare the rank 0
        if compare_tensor_value_with_reference:
            tp_rank = parallel_state.get_tp_rank()
            ep_rank = parallel_state.get_prefill_ep_rank()
            hidden_states_sharded_np = hidden_states_sharded.numpy()
            for token_offset, num_tokens in local_batch_indices_to_compare:
                print(f"After moe {ep_rank=} {tp_rank=} checking {token_offset=}, {num_tokens=}")
                assert_allclose(
                    hidden_states_sharded_np[
                        token_offset : token_offset + num_tokens,
                        :,
                    ],
                    np.array(
                        PrefillLayer.reference_tensors[f"{self.layer_id}_moe_out"],
                        dtype=Config.dtype,
                    ),
                )

        return hidden_states_sharded
