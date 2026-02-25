import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import parallel_state
import torch
import torch.distributed as dist
from config import Config
from nkipy.core.trace import NKIPyKernel
from nkipy.runtime.device_kernel import DeviceKernel
from nkipy.runtime.device_tensor import DeviceTensor
from kernels.rope import compute_cos_sin
from kernels.sampling import greedy_sampling
from kernels.token_embedding import token_embedding
from kernels.tokengen import tokengen_fused_4layers
from logger import get_logger
from neuronxcc.nki._pre_prod_kernels.blockwise_mm import TILE_SIZE
from parallel_state import get_tp_rank, get_tp_size
from prefill_layer import PrefillLayer
from safetensors.torch import load_file
from utils import bfloat16, create_shared_pinned_tensor

logger = get_logger()

trace = NKIPyKernel.trace(backend="hlo")

@dataclass
class LayerTensors:
    qkv_weight: DeviceTensor
    o_weight: DeviceTensor
    prefill_gate_up_weight: DeviceTensor
    decode_gate_up_weight: DeviceTensor
    prefill_down_weight: DeviceTensor
    decode_down_weight: DeviceTensor
    router_weight: DeviceTensor
    router_bias: DeviceTensor
    input_weight: DeviceTensor
    post_attention_weight: DeviceTensor
    sink: DeviceTensor
    qkv_bias: DeviceTensor
    o_bias: DeviceTensor
    prefill_gate_up_bias_plus1_T: DeviceTensor
    prefill_down_bias_broadcasted: DeviceTensor
    decode_gate_up_bias_plus1_T: DeviceTensor
    decode_down_bias_broadcasted: DeviceTensor
    cache_k: DeviceTensor
    cache_v: DeviceTensor


def preprocess_moe_weights(gate_up_weight, down_weight, gate_up_bias, down_bias, config: Config, ep_size, ep_rank):
    # shard by EP
    n_experts_per_ep = config.num_experts // ep_size
    start_expert_idx = n_experts_per_ep * ep_rank
    end_expert_idx = n_experts_per_ep * (ep_rank + 1)
    gate_up_weight = gate_up_weight[start_expert_idx:end_expert_idx]
    down_weight = down_weight[start_expert_idx:end_expert_idx]
    gate_up_bias = gate_up_bias[start_expert_idx:end_expert_idx]
    down_bias = down_bias[start_expert_idx:end_expert_idx]
    # pre transpose to [E, H, 2]
    gate_up_bias_plus1_T = gate_up_bias.transpose(1, 2)
    gate_up_bias_plus1_T[..., 1] += 1
    # boardcast to reduce the cost of nc_stream_shuffle
    down_bias_broadcasted = down_bias.unsqueeze(1)  # shape: (num_experts, 1, hidden_size)
    down_bias_broadcasted = down_bias_broadcasted.expand(-1, TILE_SIZE, -1)
    return gate_up_weight, down_weight, gate_up_bias_plus1_T, down_bias_broadcasted

class GPTOSSModel:

    def __init__(self, model_weights, config: Config, check_against_reference: bool = False):
        """Initialize the model with weights and configuration"""
        self.config = config

        # Load and prepare all model resources
        self.tok_embedding = create_shared_pinned_tensor(
            model_weights.pop("tok_embedding"),
            "tok_embedding",
        )

        # Initialize kernels to None - will be loaded lazily
        self.kernel_cte_greedy_sampling = None
        # self.kernel_cte_fused_add_residual_greedy_sampling = None
        self.kernel_tkg_fuse4 = None
        self.kernel_tkg_greedy_sampling = None
        self.kernel_tkg_token_embedding = None
        self.prefill_layer = []

        # Initialize shared tensors as separate variables
        self.cos = None
        self.sin = None
        self.norm_weight = None
        self.lm_head_weight = None
        self.tok_embedding_device = None

        # Prepare model resources
        self._prepare_tensors(model_weights)
        # Ensure kernels are compiled
        self._prepare_kernels(check_against_reference=check_against_reference)

    def _prepare_tensors(self, weights):
        """Allocate device tensors ahead of time, including weights and intermediate tensors"""
        logger.info("Preparing Tensors")
        n_local_kv_heads = max(1, self.config.n_kv_heads // get_tp_size())
        cache_k = np.zeros(
            (
                self.config.max_batch_size_per_dp,
                n_local_kv_heads,
                self.config.head_dim,
                self.config.max_model_len,
            ),
            dtype=self.config.dtype,
        )
        cache_v = np.zeros(
            (
                self.config.max_batch_size_per_dp,
                self.config.max_model_len,
                n_local_kv_heads,
                self.config.head_dim,
            ),
            dtype=self.config.dtype,
        )
        norm_weight = weights.pop("norm_weight")
        lm_head_weight = weights.pop("lm_head_weight")
        self.layer_tensors: List[LayerTensors] = []
        for layer_id in range(self.config.n_layers):
            qkv_weight = weights.pop(f"layers.{layer_id}.qkv_weight")
            o_weight = weights.pop(f"layers.{layer_id}.o_weight")
            router_weight = weights.pop(f"layers.{layer_id}.router_weight")
            router_bias = weights.pop(f"layers.{layer_id}.router_bias")
            input_weight = weights.pop(f"layers.{layer_id}.attn_norm_weight")
            post_attention_weight = weights.pop(f"layers.{layer_id}.mlp_norm_weight")
            sink = weights.pop(f"layers.{layer_id}.attn_sinks")
            qkv_bias = weights.pop(f"layers.{layer_id}.qkv_bias")
            o_bias = weights.pop(f"layers.{layer_id}.o_bias")
            gate_up_weight = weights.pop(f"layers.{layer_id}.gate_up_weight")
            gate_up_bias = weights.pop(f"layers.{layer_id}.gate_up_bias")
            down_weight = weights.pop(f"layers.{layer_id}.down_weight")
            down_bias = weights.pop(f"layers.{layer_id}.down_bias")

            # shard bias by TP BEFORE EP sharding
            hidden_size_sharded = self.config.hidden_size // get_tp_size()
            tp_start_idx = hidden_size_sharded * get_tp_rank()
            tp_end_idx = tp_start_idx + hidden_size_sharded
            o_bias[:tp_start_idx] = 0
            o_bias[tp_end_idx:] = 0
            down_bias[:, :tp_start_idx] = 0
            down_bias[:, tp_end_idx:] = 0

            prefill_gate_up_weight, prefill_down_weight, prefill_gate_up_bias_plus1_T, prefill_down_bias_broadcasted = preprocess_moe_weights(gate_up_weight, down_weight, gate_up_bias, down_bias, self.config, parallel_state.get_prefill_ep_size(), parallel_state.get_prefill_ep_rank())

            # Clone tensors to avoid modifying the same underlying data twice
            decode_gate_up_weight, decode_down_weight, decode_gate_up_bias_plus1_T, decode_down_bias_broadcasted = preprocess_moe_weights(gate_up_weight, down_weight, gate_up_bias.clone(), down_bias.clone(), self.config, parallel_state.get_decode_ep_size(), parallel_state.get_decode_ep_rank())
            self.layer_tensors.append(
                LayerTensors(
                    qkv_weight=DeviceTensor.from_torch(
                        qkv_weight, f"qkv_weight_L{layer_id}"
                    ),
                    o_weight=DeviceTensor.from_torch(o_weight, f"o_weight_L{layer_id}"),
                    prefill_gate_up_weight=DeviceTensor.from_torch(
                        prefill_gate_up_weight, f"prefill_gate_up_weight_L{layer_id}"
                    ),
                    decode_gate_up_weight=DeviceTensor.from_torch(
                        decode_gate_up_weight, f"decode_gate_up_weight_L{layer_id}"
                    ),
                    prefill_down_weight=DeviceTensor.from_torch(
                        prefill_down_weight, f"prefill_down_weight_L{layer_id}"
                    ),
                    decode_down_weight=DeviceTensor.from_torch(
                        decode_down_weight, f"decode_down_weight_L{layer_id}"
                    ),
                    router_weight=DeviceTensor.from_torch(
                        router_weight, f"router_weight_L{layer_id}"
                    ),
                    router_bias=DeviceTensor.from_torch(
                        router_bias, f"router_bias_L{layer_id}"
                    ),
                    input_weight=DeviceTensor.from_torch(
                        input_weight, f"input_weight_L{layer_id}"
                    ),
                    post_attention_weight=DeviceTensor.from_torch(
                        post_attention_weight,
                        f"post_attention_weight_L{layer_id}",
                    ),
                    sink=DeviceTensor.from_torch(sink, f"sink_L{layer_id}"),
                    qkv_bias=DeviceTensor.from_torch(qkv_bias, f"qkv_bias_L{layer_id}"),
                    o_bias=DeviceTensor.from_torch(o_bias, f"o_bias_L{layer_id}"),
                    prefill_gate_up_bias_plus1_T=DeviceTensor.from_torch(
                        prefill_gate_up_bias_plus1_T,
                        f"prefill_gate_up_bias_plus1_T_L{layer_id}",
                    ),
                    prefill_down_bias_broadcasted=DeviceTensor.from_torch(
                        prefill_down_bias_broadcasted,
                        f"prefill_down_bias_broadcasted_L{layer_id}",
                    ),
                    decode_gate_up_bias_plus1_T=DeviceTensor.from_torch(
                        decode_gate_up_bias_plus1_T,
                        f"decode_gate_up_bias_plus1_T_L{layer_id}",
                    ),
                    decode_down_bias_broadcasted=DeviceTensor.from_torch(
                        decode_down_bias_broadcasted,
                        f"decode_down_bias_broadcasted_L{layer_id}",
                    ),
                    cache_k=DeviceTensor.from_numpy(cache_k, f"cache_k_L{layer_id}"),
                    cache_v=DeviceTensor.from_numpy(cache_v, f"cache_v_L{layer_id}"),
                )
            )
        cos, sin = compute_cos_sin(max_model_len=self.config.max_model_len)
        self.cos = DeviceTensor.from_numpy(cos, "cos")
        self.sin = DeviceTensor.from_numpy(sin, "sin")
        self.prefill_ep_rank = DeviceTensor.from_numpy(
            np.array([parallel_state.get_prefill_ep_rank()], dtype=np.int32), "prefill_ep_rank"
        )
        self.decode_ep_rank = DeviceTensor.from_numpy(
            np.array([parallel_state.get_decode_ep_rank()], dtype=np.int32), "decode_ep_rank"
        )
        self.norm_weight = DeviceTensor.from_torch(norm_weight, "norm_weight")
        self.lm_head_weight = DeviceTensor.from_torch(lm_head_weight, "lm_head_weight")

        # Create DeviceTensor for token embeddings
        self.tok_embedding_device = DeviceTensor.from_torch(
            self.tok_embedding, "tok_embedding_device"
        )

        logger.info("Finished Preparing Tensors")

    def _prepare_kernels(self, check_against_reference):
        """Compile and load model kernels ahead of time"""
        logger.info("Preparing kernels")

        for layer_id in range(self.config.n_layers):
            # Compute sliding window based on layer index (even layers have sliding window, odd layers don't)
            sliding_window = self.config.sliding_window if layer_id % 2 == 0 else 0
            self.prefill_layer.append(
                PrefillLayer(
                    layer_id=layer_id,
                    config=self.config,
                    sliding_window=sliding_window,
                    sink=self.layer_tensors[layer_id].sink,
                    qkv_weight=self.layer_tensors[layer_id].qkv_weight,
                    qkv_bias=self.layer_tensors[layer_id].qkv_bias,
                    o_weight=self.layer_tensors[layer_id].o_weight,
                    o_bias=self.layer_tensors[layer_id].o_bias,
                    input_weight=self.layer_tensors[layer_id].input_weight,
                    cos=self.cos,
                    sin=self.sin,
                    post_attention_weight=self.layer_tensors[
                        layer_id
                    ].post_attention_weight,
                    router_weight=self.layer_tensors[layer_id].router_weight,
                    router_bias=self.layer_tensors[layer_id].router_bias,
                    gate_up_proj_weight=self.layer_tensors[
                        layer_id
                    ].prefill_gate_up_weight,
                    down_weight=self.layer_tensors[layer_id].prefill_down_weight,
                    gate_up_bias_plus1_T=self.layer_tensors[
                        layer_id
                    ].prefill_gate_up_bias_plus1_T,
                    down_bias_broadcasted=self.layer_tensors[
                        layer_id
                    ].prefill_down_bias_broadcasted,
                    cache_k=self.layer_tensors[layer_id].cache_k,
                    cache_v=self.layer_tensors[layer_id].cache_v,
                    ep_rank=self.prefill_ep_rank,
                    check_against_reference=check_against_reference,
                )
            )

        self.kernel_cte_greedy_sampling = DeviceKernel.compile_and_load(
            trace(greedy_sampling),
            name="cte_greedy_sampling",
            hidden_states_shard=np.empty(
                shape=(
                    self.config.max_batch_size_per_dp
                    * self.config.max_model_len
                    // get_tp_size(),
                    self.config.hidden_size,
                ),
                dtype=self.config.dtype,
            ),
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            last_token_indices=np.empty(
                shape=(self.config.max_batch_size_per_dp,),
                dtype=np.int32,
            ),
            configs=self.config,
            is_neuronpy=True,
        )

        # self.kernel_cte_fused_add_residual_greedy_sampling = (
        #     DeviceKernel.compile_and_load(
        #         trace(fused_add_residual_greedy_sampling),
        #         name="cte_fused_add_residual_greedy_sampling",
        #         residual_2d_shard=self.block_wise_moe_layers[0].residual_2d_shard,
        #         hidden_states=self.block_wise_moe_layers[0].moe_hidden_states,
        #         norm_weight=self.norm_weight,
        #         lm_head_weight=self.lm_head_weight,
        #         configs=self.config,
        #     )
        # )

        # Create input tensor for compilation
        hidden_states = np.empty(
            shape=(self.config.max_batch_size_per_dp, 1, self.config.hidden_size),
            dtype=self.config.dtype,
        )
        start_pos = np.empty(shape=(self.config.max_batch_size_per_dp,), dtype=np.int32)
        # when debug, may only run 1 layer
        if self.config.n_layers >= 4:
            self.kernel_tkg_fuse4 = DeviceKernel.compile_and_load(
                trace(tokengen_fused_4layers),
                hidden_states=hidden_states,
                start_pos=start_pos,
                # Layer 0 weights (even layer - sliding_window=128)
                qkv_weight_0=self.layer_tensors[0].qkv_weight,
                o_weight_0=self.layer_tensors[0].o_weight,
                input_weight_0=self.layer_tensors[0].input_weight,
                cache_k_0=self.layer_tensors[0].cache_k,
                cache_v_0=self.layer_tensors[0].cache_v,
                post_attention_weight_0=self.layer_tensors[0].post_attention_weight,
                router_weight_0=self.layer_tensors[0].router_weight,
                router_bias_0=self.layer_tensors[0].router_bias,
                gate_up_weight_0=self.layer_tensors[0].decode_gate_up_weight,
                gate_up_bias_plus1_T_0=self.layer_tensors[
                    0
                ].decode_gate_up_bias_plus1_T,
                down_weight_0=self.layer_tensors[0].decode_down_weight,
                down_bias_broadcasted_0=self.layer_tensors[
                    0
                ].decode_down_bias_broadcasted,
                sink_0=self.layer_tensors[0].sink,
                qkv_bias_0=self.layer_tensors[0].qkv_bias,
                o_bias_0=self.layer_tensors[0].o_bias,
                # Layer 1 weights (odd layer - sliding_window=0)
                qkv_weight_1=self.layer_tensors[1].qkv_weight,
                o_weight_1=self.layer_tensors[1].o_weight,
                input_weight_1=self.layer_tensors[1].input_weight,
                cache_k_1=self.layer_tensors[1].cache_k,
                cache_v_1=self.layer_tensors[1].cache_v,
                post_attention_weight_1=self.layer_tensors[1].post_attention_weight,
                router_weight_1=self.layer_tensors[1].router_weight,
                router_bias_1=self.layer_tensors[1].router_bias,
                gate_up_weight_1=self.layer_tensors[1].decode_gate_up_weight,
                gate_up_bias_plus1_T_1=self.layer_tensors[
                    1
                ].decode_gate_up_bias_plus1_T,
                down_weight_1=self.layer_tensors[1].decode_down_weight,
                down_bias_broadcasted_1=self.layer_tensors[
                    1
                ].decode_down_bias_broadcasted,
                sink_1=self.layer_tensors[1].sink,
                qkv_bias_1=self.layer_tensors[1].qkv_bias,
                o_bias_1=self.layer_tensors[1].o_bias,
                # Layer 2 weights (even layer - sliding_window=128)
                qkv_weight_2=self.layer_tensors[2].qkv_weight,
                o_weight_2=self.layer_tensors[2].o_weight,
                input_weight_2=self.layer_tensors[2].input_weight,
                cache_k_2=self.layer_tensors[2].cache_k,
                cache_v_2=self.layer_tensors[2].cache_v,
                post_attention_weight_2=self.layer_tensors[2].post_attention_weight,
                router_weight_2=self.layer_tensors[2].router_weight,
                router_bias_2=self.layer_tensors[2].router_bias,
                gate_up_weight_2=self.layer_tensors[2].decode_gate_up_weight,
                gate_up_bias_plus1_T_2=self.layer_tensors[
                    2
                ].decode_gate_up_bias_plus1_T,
                down_weight_2=self.layer_tensors[2].decode_down_weight,
                down_bias_broadcasted_2=self.layer_tensors[
                    2
                ].decode_down_bias_broadcasted,
                sink_2=self.layer_tensors[2].sink,
                qkv_bias_2=self.layer_tensors[2].qkv_bias,
                o_bias_2=self.layer_tensors[2].o_bias,
                # Layer 3 weights (odd layer - sliding_window=0)
                qkv_weight_3=self.layer_tensors[3].qkv_weight,
                o_weight_3=self.layer_tensors[3].o_weight,
                input_weight_3=self.layer_tensors[3].input_weight,
                cache_k_3=self.layer_tensors[3].cache_k,
                cache_v_3=self.layer_tensors[3].cache_v,
                post_attention_weight_3=self.layer_tensors[3].post_attention_weight,
                router_weight_3=self.layer_tensors[3].router_weight,
                router_bias_3=self.layer_tensors[3].router_bias,
                gate_up_weight_3=self.layer_tensors[3].decode_gate_up_weight,
                gate_up_bias_plus1_T_3=self.layer_tensors[
                    3
                ].decode_gate_up_bias_plus1_T,
                down_weight_3=self.layer_tensors[3].decode_down_weight,
                down_bias_broadcasted_3=self.layer_tensors[
                    3
                ].decode_down_bias_broadcasted,
                sink_3=self.layer_tensors[3].sink,
                qkv_bias_3=self.layer_tensors[3].qkv_bias,
                o_bias_3=self.layer_tensors[3].o_bias,
                # Shared tensors (not duplicated)
                cos=self.cos,
                sin=self.sin,
                ep_rank=self.decode_ep_rank,
                ep_size=parallel_state.get_decode_ep_size(),
                configs=self.config,
            )

        self.kernel_tkg_greedy_sampling = DeviceKernel.compile_and_load(
            trace(greedy_sampling),
            name="tkg_greedy_sampling",
            hidden_states_shard=hidden_states,
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            last_token_indices=None,
            configs=self.config,
            is_neuronpy=True,
        )

        # Compile token embedding kernel
        self.kernel_tkg_token_embedding = DeviceKernel.compile_and_load(
            trace(token_embedding),
            name="token_embedding",
            tok_embedding=np.empty(self.tok_embedding.shape, dtype=bfloat16),
            token_ids=np.zeros((self.config.max_batch_size_per_dp, 1), dtype=np.uint32),
        )

        logger.info("Finished Kernel Compilation and Loading")

    def generate(
        self,
        *,
        input_ids,
        context_lens,
        max_tokens,
        batch_indices_to_check=None,
        **kwargs,
    ):
        """Run inference and generate tokens with delayed .numpy() optimization"""
        assert input_ids.shape == (
            self.config.max_batch_size_per_dp,
            self.config.max_model_len,
        )
        assert context_lens.shape == (self.config.max_batch_size_per_dp,)

        # make a copy to avoid in-place change original inputs
        input_ids = input_ids.copy()
        context_lens = context_lens.copy()
        start_pos = np.zeros_like(context_lens)
        is_prefill=True

        next_id_tensor = None
        prev_next_id_tensor = None

        for step_id in range(max_tokens + 1):
            if is_prefill:
                output_ids = self.forward(
                    input_ids_np=input_ids,
                    start_pos_np=start_pos,
                    context_len_np=context_lens,
                    is_prefill=is_prefill,
                    batch_indices_to_check=batch_indices_to_check,
                    **kwargs,
                )
                yield output_ids
                is_prefill = False
                start_pos[...] = context_lens
                context_lens += 1
                next_id_tensor = DeviceTensor.from_numpy(output_ids, "input_ids_device")
            else:
                # Token generation phase with delayed .numpy() calls (tensor read)

                prev_next_id_tensor = next_id_tensor
                # N.B. Python GC should clean up the DeviceTensor referenced by prev_next_id_tensor previously after it's overwriten (not alive)
                next_id_tensor = self._forward_tkg(
                    input_ids=next_id_tensor,
                    start_pos_np=start_pos,
                    **kwargs,
                )

                if step_id > 1:
                    # skip the first iteration (delayed)
                    # Convert previous tensor to numpy
                    output_ids = (
                        prev_next_id_tensor.numpy()
                        .reshape((self.config.max_batch_size_per_dp, 1))
                        .astype(dtype=np.uint32)
                    )
                    yield output_ids

                # Setup for next iteration
                start_pos[...] = context_lens
                context_lens += 1

        # last token if exist
        if next_id_tensor:
            output_ids = (
                next_id_tensor.numpy()
                .reshape((self.config.max_batch_size_per_dp, 1))
                .astype(dtype=np.uint32)
            )
            yield output_ids

    def forward(
        self,
        *,
        input_ids_np: npt.NDArray[np.uint32],
        start_pos_np: npt.NDArray[np.int32],
        context_len_np: npt.NDArray[np.int32],
        is_prefill: bool,
        batch_indices_to_check: Optional[List[int]] = None,
        **kwargs,
    ):
        assert input_ids_np.dtype == np.uint32
        assert start_pos_np.dtype == np.int32
        assert context_len_np.dtype == np.int32
        if is_prefill:
            return self._forward_cte(
                input_ids_np=input_ids_np,
                context_len_np=context_len_np,
                batch_indices_to_check=batch_indices_to_check,
                **kwargs,
            )
        else:
            return self._forward_tkg(
                input_ids=DeviceTensor.from_numpy(input_ids_np, "input_ids_device"),
                start_pos_np=start_pos_np,
                **kwargs,
            ).numpy()

    def _forward_cte(
        self,
        *,
        input_ids_np: npt.NDArray[np.uint32],
        context_len_np: npt.NDArray[np.int32],
        batch_indices_to_check: Optional[List[int]] = None,
        warmup: bool = False,
    ):
        # assume input tokens are sharded by EP
        tp_rank = get_tp_rank()
        tp_size = get_tp_size()
        max_batch_size = self.config.max_batch_size_per_dp
        max_model_len = self.config.max_model_len
        assert max_batch_size * max_model_len % tp_size == 0
        # for token-wise computation, reusing TP axis for sequence parallel
        num_tokens_per_tp_rank = max_batch_size * max_model_len // tp_size
        input_ids_np = input_ids_np.reshape((tp_size, num_tokens_per_tp_rank))
        input_ids_this_rank = input_ids_np[tp_rank, :]
        hidden_states_sharded = self.tok_embedding[input_ids_this_rank]
        # FIXME:
        assert hidden_states_sharded.dtype == torch.bfloat16
        assert tuple(hidden_states_sharded.shape) == (
            num_tokens_per_tp_rank,
            self.config.hidden_size,
        )

        hidden_states_sharded = DeviceTensor.from_torch(
            hidden_states_sharded,
            "hidden_states_sharded",
        )
        # Initial position - next_id tensor for storing generated tokens
        next_id = DeviceTensor.from_numpy(
            np.empty((self.config.max_batch_size_per_dp, 1), dtype=np.uint32),
            "next_id",
        )
        last_token_indices = DeviceTensor.from_numpy(
            np.maximum(context_len_np - 1, 0),
            "last_token_indices",
        )
        # Context encoding
        for i in range(self.config.n_layers):
            # TODO: last layer only need to run moe on last token
            hidden_states_sharded = self.prefill_layer[i].forward(
                hidden_states_sharded=hidden_states_sharded,
                seq_len=context_len_np,
                warmup=warmup,
                batch_indices_to_check=batch_indices_to_check,
            )
        self.kernel_cte_greedy_sampling(
            inputs={
                "hidden_states_shard": hidden_states_sharded,
                "norm_weight": self.norm_weight,
                "lm_head_weight": self.lm_head_weight,
                "last_token_indices": last_token_indices,
            },
            outputs={"output0": next_id},
        )

        if not warmup:
            logger.info(
                "--> Finished Greedy Sampling in CTE"
            )
        next_id_np = (
            next_id.numpy()
            .reshape((self.config.max_batch_size_per_dp, 1))
            .astype(dtype=np.uint32)
        )
        return next_id_np

    def _forward_tkg(
        self,
        *,
        input_ids: DeviceTensor,  # Now accepts DeviceTensor instead of numpy array
        start_pos_np: npt.NDArray[np.int32],
        warmup: bool = False,
    ):
        # Update the start position for this iteration
        t_start_pos = DeviceTensor.from_numpy(start_pos_np, "start_pos")

        # Allocate new next_id DeviceTensor for this iteration
        next_id = DeviceTensor.from_numpy(
            np.empty((self.config.max_batch_size_per_dp, 1), dtype=np.uint32), "next_id"
        )

        # Use token embedding kernel instead of direct indexing
        hidden_states = DeviceTensor.from_numpy(
            np.zeros(
                (self.config.max_batch_size_per_dp, 1, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            "h0/res1",
        )

        self.kernel_tkg_token_embedding(
            inputs={
                "tok_embedding": self.tok_embedding_device,
                "token_ids": input_ids,  # Use DeviceTensor directly
            },
            outputs={
                "output0": hidden_states,
            }
        )

        t_res1 = hidden_states  # Output becomes next layer's input

        for i in range(0, self.config.n_layers, 4):
            self.kernel_tkg_fuse4(
                inputs={
                    "hidden_states": hidden_states,
                    "start_pos": t_start_pos,
                    # Layer i weights
                    "qkv_weight_0": self.layer_tensors[i].qkv_weight,
                    "o_weight_0": self.layer_tensors[i].o_weight,
                    "input_weight_0": self.layer_tensors[i].input_weight,
                    "cache_k_0.must_alias_input": self.layer_tensors[i].cache_k,
                    "cache_v_0.must_alias_input": self.layer_tensors[i].cache_v,
                    "post_attention_weight_0": self.layer_tensors[i].post_attention_weight,
                    "router_weight_0": self.layer_tensors[i].router_weight,
                    "router_bias_0": self.layer_tensors[i].router_bias,
                    "gate_up_weight_0": self.layer_tensors[i].decode_gate_up_weight,
                    "gate_up_bias_plus1_T_0": self.layer_tensors[i].decode_gate_up_bias_plus1_T,
                    "down_weight_0": self.layer_tensors[i].decode_down_weight,
                    "down_bias_broadcasted_0": self.layer_tensors[i].decode_down_bias_broadcasted,
                    "sink_0": self.layer_tensors[i].sink,
                    "qkv_bias_0": self.layer_tensors[i].qkv_bias,
                    "o_bias_0": self.layer_tensors[i].o_bias,
                    # Layer i+1 weights
                    "qkv_weight_1": self.layer_tensors[i + 1].qkv_weight,
                    "o_weight_1": self.layer_tensors[i + 1].o_weight,
                    "input_weight_1": self.layer_tensors[i + 1].input_weight,
                    "cache_k_1.must_alias_input": self.layer_tensors[i + 1].cache_k,
                    "cache_v_1.must_alias_input": self.layer_tensors[i + 1].cache_v,
                    "post_attention_weight_1": self.layer_tensors[i + 1].post_attention_weight,
                    "router_weight_1": self.layer_tensors[i + 1].router_weight,
                    "router_bias_1": self.layer_tensors[i + 1].router_bias,
                    "gate_up_weight_1": self.layer_tensors[i + 1].decode_gate_up_weight,
                    "gate_up_bias_plus1_T_1": self.layer_tensors[i + 1].decode_gate_up_bias_plus1_T,
                    "down_weight_1": self.layer_tensors[i + 1].decode_down_weight,
                    "down_bias_broadcasted_1": self.layer_tensors[i + 1].decode_down_bias_broadcasted,
                    "sink_1": self.layer_tensors[i + 1].sink,
                    "qkv_bias_1": self.layer_tensors[i + 1].qkv_bias,
                    "o_bias_1": self.layer_tensors[i + 1].o_bias,
                    # Layer i+2 weights
                    "qkv_weight_2": self.layer_tensors[i + 2].qkv_weight,
                    "o_weight_2": self.layer_tensors[i + 2].o_weight,
                    "input_weight_2": self.layer_tensors[i + 2].input_weight,
                    "cache_k_2.must_alias_input": self.layer_tensors[i + 2].cache_k,
                    "cache_v_2.must_alias_input": self.layer_tensors[i + 2].cache_v,
                    "post_attention_weight_2": self.layer_tensors[i + 2].post_attention_weight,
                    "router_weight_2": self.layer_tensors[i + 2].router_weight,
                    "router_bias_2": self.layer_tensors[i + 2].router_bias,
                    "gate_up_weight_2": self.layer_tensors[i + 2].decode_gate_up_weight,
                    "gate_up_bias_plus1_T_2": self.layer_tensors[i + 2].decode_gate_up_bias_plus1_T,
                    "down_weight_2": self.layer_tensors[i + 2].decode_down_weight,
                    "down_bias_broadcasted_2": self.layer_tensors[i + 2].decode_down_bias_broadcasted,
                    "sink_2": self.layer_tensors[i + 2].sink,
                    "qkv_bias_2": self.layer_tensors[i + 2].qkv_bias,
                    "o_bias_2": self.layer_tensors[i + 2].o_bias,
                    # Layer i+3 weights
                    "qkv_weight_3": self.layer_tensors[i + 3].qkv_weight,
                    "o_weight_3": self.layer_tensors[i + 3].o_weight,
                    "input_weight_3": self.layer_tensors[i + 3].input_weight,
                    "cache_k_3.must_alias_input": self.layer_tensors[i + 3].cache_k,
                    "cache_v_3.must_alias_input": self.layer_tensors[i + 3].cache_v,
                    "post_attention_weight_3": self.layer_tensors[i + 3].post_attention_weight,
                    "router_weight_3": self.layer_tensors[i + 3].router_weight,
                    "router_bias_3": self.layer_tensors[i + 3].router_bias,
                    "gate_up_weight_3": self.layer_tensors[i + 3].decode_gate_up_weight,
                    "gate_up_bias_plus1_T_3": self.layer_tensors[i + 3].decode_gate_up_bias_plus1_T,
                    "down_weight_3": self.layer_tensors[i + 3].decode_down_weight,
                    "down_bias_broadcasted_3": self.layer_tensors[i + 3].decode_down_bias_broadcasted,
                    "sink_3": self.layer_tensors[i + 3].sink,
                    "qkv_bias_3": self.layer_tensors[i + 3].qkv_bias,
                    "o_bias_3": self.layer_tensors[i + 3].o_bias,
                    # Shared tensors
                    "cos": self.cos,
                    "sin": self.sin,
                    "ep_rank": self.decode_ep_rank,
                },
                outputs={
                    "output0": t_res1,
                    "cache_k_0": self.layer_tensors[i].cache_k,
                    "cache_v_0": self.layer_tensors[i].cache_v,
                    "cache_k_1": self.layer_tensors[i + 1].cache_k,
                    "cache_v_1": self.layer_tensors[i + 1].cache_v,
                    "cache_k_2": self.layer_tensors[i + 2].cache_k,
                    "cache_v_2": self.layer_tensors[i + 2].cache_v,
                    "cache_k_3": self.layer_tensors[i + 3].cache_k,
                    "cache_v_3": self.layer_tensors[i + 3].cache_v,
                },
                save_trace=True if warmup and (i == 4) and dist.get_rank() == 0 else False,
            )

        # Run greedy sampling to get next token
        self.kernel_tkg_greedy_sampling(
            inputs={
                "hidden_states_shard": t_res1,
                "norm_weight": self.norm_weight,
                "lm_head_weight": self.lm_head_weight,
            },
            outputs={"output0": next_id},
        )

        # Return DeviceTensor without .numpy() - caller will handle conversion
        return next_id


def load_gpt_oss_weights(checkpoint, config):
    logger.info("Loading Model Weights")
    shard_path = os.path.join(checkpoint, f"shard_{get_tp_rank()}.safetensors")
    weights = load_file(shard_path, device="cpu")

    # TODO: move this part to upper to unify the logic
    n_heads_per_shard = config.n_heads // get_tp_size()
    for layer_id in range(config.n_layers):
        weights[f"layers.{layer_id}.attn_sinks"] = weights[
            f"layers.{layer_id}.attn_sinks"
        ][:n_heads_per_shard]
        weights[f"layers.{layer_id}.gate_up_weight"] = weights[
            f"layers.{layer_id}.gate_up_weight"
        ].reshape(
            config.num_experts,
            config.hidden_size,
            2,
            config.intermediate_size // get_tp_size(),
        )
        weights[f"layers.{layer_id}.gate_up_bias"] = weights[
            f"layers.{layer_id}.gate_up_bias"
        ].reshape(config.num_experts, 2, config.intermediate_size // get_tp_size())
    return weights
