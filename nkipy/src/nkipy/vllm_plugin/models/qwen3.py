# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy Qwen3 model for Neuron (MoE, QK RMSNorm)."""

import logging
import time

import numpy as np
import torch
import torch.distributed as dist
from nkipy.runtime import DeviceKernel, DeviceTensor

from .config import Config
from .kernels.qwen3_transformer_layer import qwen3_transformer_layer
from .kernels.sampling import greedy_sampling

logger = logging.getLogger(__name__)
BUILD_DIR = "./build"
USE_NKI_RMSNORM = True

LAYER_WEIGHT_KEYS = [
    ("qkv_weight", "qkv_weight"),
    ("o_weight", "o_weight"),
    ("gate_up_weight", "gate_up_weight"),
    ("down_weight", "down_weight"),
    ("input_weight", "input_weight"),
    ("post_attention_weight", "post_attention_weight"),
    ("router_weight", "router_weight"),
    ("q_norm_weight", "q_norm_weight"),
    ("k_norm_weight", "k_norm_weight"),
]

_KERNEL_INPUT_KEYS = [
    ("qkv_weight", "qkv_weight"),
    ("o_weight", "o_weight"),
    ("input_weight", "input_weight"),
    ("q_norm_weight", "q_norm_weight"),
    ("k_norm_weight", "k_norm_weight"),
    ("cache_k.must_alias_input", "cache_k"),
    ("cache_v.must_alias_input", "cache_v"),
    ("post_attention_weight", "post_attention_weight"),
    ("router_weight", "router_weight"),
    ("gate_up_weight", "gate_up_weight"),
    ("down_weight", "down_weight"),
]


class Qwen3Model:
    """NKIPy Qwen3 model running on Neuron cores (MoE)."""

    def __init__(self, model_weights, config: Config, skip_kernels=False):
        self.config = config
        self.tok_embedding = model_weights.get("tok_embedding")

        self.kernel_cte = None
        self.kernel_tkg = None
        self.kernel_cte_greedy_sampling = None
        self.kernel_tkg_greedy_sampling = None
        self.norm_weight = None
        self.lm_head_weight = None
        self.layer_tensors = []

        self._prepare_tensors(model_weights)
        if not skip_kernels and config.context_len is not None:
            self._prepare_kernels()

    def _prepare_tensors(self, weights):
        t = time.time()
        logger.info("Preparing tensors")
        cfg = self.config
        n_local_kv_heads = max(1, cfg.num_kv_heads // dist.get_world_size())
        cache_shape = (cfg.max_batch_size, cfg.max_seq_len, n_local_kv_heads, cfg.head_dim)
        cache_k = np.zeros(cache_shape, dtype=cfg.dtype)
        cache_v = np.zeros(cache_shape, dtype=cfg.dtype)

        self.layer_tensors = []
        for lid in range(cfg.num_layers):
            layer = {}
            for wk, prefix in LAYER_WEIGHT_KEYS:
                layer[wk] = DeviceTensor.from_torch(weights[f"layers.{lid}.{wk}"], f"{prefix}_L{lid}")
            layer["cache_k"] = DeviceTensor.from_numpy(cache_k, f"cache_k_L{lid}")
            layer["cache_v"] = DeviceTensor.from_numpy(cache_v, f"cache_v_L{lid}")
            self.layer_tensors.append(layer)

        self.norm_weight = DeviceTensor.from_torch(weights["norm_weight"], "norm_weight")
        self.lm_head_weight = DeviceTensor.from_torch(weights["lm_head_weight"], "lm_head_weight")
        logger.info("Tensors prepared in %.2fs", time.time() - t)

    def _prepare_kernels(self):
        t = time.time()
        logger.info("Compiling kernels")
        cfg = self.config
        compiler_args = cfg.additional_compiler_args_nkipy
        L0 = self.layer_tensors[0]

        x_ctx = DeviceTensor.from_numpy(
            np.empty((cfg.max_batch_size, cfg.context_len, cfg.hidden_size), dtype=cfg.dtype), "x_ctx")
        x_tok = DeviceTensor.from_numpy(
            np.empty((cfg.max_batch_size, 1, cfg.hidden_size), dtype=cfg.dtype), "x_tok")
        sp = DeviceTensor.from_numpy(np.empty((1,), dtype=np.int32), "start_pos")

        layer_args = {wk: L0[wk] for wk, _ in LAYER_WEIGHT_KEYS}
        layer_args["cache_k"] = L0["cache_k"]
        layer_args["cache_v"] = L0["cache_v"]

        self.kernel_cte = DeviceKernel.compile_and_load(
            qwen3_transformer_layer, name="cte_layer", x=x_ctx, start_pos=None,
            **layer_args, configs=cfg, build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args)

        self.kernel_tkg = DeviceKernel.compile_and_load(
            qwen3_transformer_layer, name="tkg_layer", x=x_tok, start_pos=sp,
            **layer_args, configs=cfg, build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args)

        self.kernel_cte_greedy_sampling = DeviceKernel.compile_and_load(
            greedy_sampling, name="cte_greedy_sampling", h=x_ctx,
            norm_weight=self.norm_weight, lm_head_weight=self.lm_head_weight,
            configs=cfg, use_nki_rmsnorm=USE_NKI_RMSNORM, build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args)

        self.kernel_tkg_greedy_sampling = DeviceKernel.compile_and_load(
            greedy_sampling, name="tkg_greedy_sampling", h=x_tok,
            norm_weight=self.norm_weight, lm_head_weight=self.lm_head_weight,
            configs=cfg, use_nki_rmsnorm=USE_NKI_RMSNORM, build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args)

        logger.info("Kernels compiled in %.2fs", time.time() - t)

    def _build_inputs(self, lid, hidden, start_pos=None):
        inputs = {"x": hidden}
        if start_pos is not None:
            inputs["start_pos"] = start_pos
        for kk, lk in _KERNEL_INPUT_KEYS:
            inputs[kk] = self.layer_tensors[lid][lk]
        return inputs

    def _build_outputs(self, lid, hidden):
        return {
            "output0": hidden,
            "cache_k": self.layer_tensors[lid]["cache_k"],
            "cache_v": self.layer_tensors[lid]["cache_v"],
        }

    def generate(self, input_ids):
        """Yield one token tensor per step (prefill + decode)."""
        cfg = self.config
        hidden = DeviceTensor.from_torch(self.tok_embedding[input_ids], "hidden_states")
        next_id = DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "next_id")

        for i in range(cfg.num_layers):
            self.kernel_cte(inputs=self._build_inputs(i, hidden),
                            outputs=self._build_outputs(i, hidden))

        self.kernel_cte_greedy_sampling(
            inputs={"h": hidden, "norm_weight": self.norm_weight, "lm_head_weight": self.lm_head_weight},
            outputs={"output0": next_id})
        next_id_torch = next_id.torch().reshape(cfg.max_batch_size, 1).to(dtype=torch.int)
        yield next_id_torch

        for pos in range(cfg.context_len, cfg.context_len + cfg.max_new_tokens):
            t_sp = DeviceTensor.from_numpy(np.array([pos], dtype=np.int32))
            hidden = DeviceTensor.from_torch(self.tok_embedding[next_id_torch], "h0/res1")

            for i in range(cfg.num_layers):
                self.kernel_tkg(inputs=self._build_inputs(i, hidden, t_sp),
                                outputs=self._build_outputs(i, hidden))

            self.kernel_tkg_greedy_sampling(
                inputs={"h": hidden, "norm_weight": self.norm_weight, "lm_head_weight": self.lm_head_weight},
                outputs={"output0": next_id})
            next_id_torch = next_id.torch().reshape(cfg.max_batch_size, 1).to(dtype=torch.int)
            yield next_id_torch
