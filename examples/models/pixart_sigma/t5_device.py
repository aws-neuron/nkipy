"""On-device T5-XXL encoder for PixArt text conditioning.

Compiles the ``t5_encoder`` kernel and runs all 24 layers on Trainium. Token
embedding lookup and tokenization stay on host (a memory gather + the
sentencepiece tokenizer); this class takes token embeddings and the attention
mask and returns encoder hidden states.

Weights come from ``t5_weights.safetensors`` (produced by
``tensor_preparation.py --t5``).
"""

import os
import time

import numpy as np
import torch
from config import Config
from kernels.t5 import t5_encoder
from kernels.t5_weight_layout import T5_BLOCK_KEYS, T5_SHARED_KEYS, t5_block_key
from nkipy.runtime import DeviceKernel, DeviceTensor
from nkipy.runtime.device_tensor import bfloat16
from safetensors.torch import load_file

BUILD_DIR = "./build"


def _to_numpy(t, dtype):
    return t.detach().to(torch.float32).cpu().numpy().astype(dtype)


class T5Encoder:
    def __init__(self, weights, config: Config, seq_len, tokenizer, embed_weight):
        """Args:
            weights: dict of prepared T5 tensors (torch).
            seq_len: fixed sequence length to compile for.
            tokenizer: HF T5 tokenizer (host).
            embed_weight: (vocab, d_model) shared token-embedding table (host).
        """
        self.config = config
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.embed_weight = embed_weight  # host embedding lookup

        self.device_weights = {}
        for k, v in weights.items():
            self.device_weights[k] = DeviceTensor.from_torch(
                v.to(torch.bfloat16) if v.is_floating_point() else v, name=k
            )
        self._prepare_kernel()

    def _all_keys(self):
        keys = list(T5_SHARED_KEYS)
        for i in range(self.config.t5_num_layers):
            keys += [t5_block_key(i, s) for s in T5_BLOCK_KEYS]
        return [k for k in keys if k in self.device_weights]

    def _prepare_kernel(self):
        t = time.time()
        print(f"[t5] compiling t5_encoder (seq={self.seq_len})")
        cfg = self.config
        S = self.seq_len
        embeds = DeviceTensor.from_numpy(
            np.empty((1, S, cfg.t5_d_model), dtype=cfg.dtype), "inputs_embeds"
        )
        mask = DeviceTensor.from_numpy(np.empty((1, S), dtype=cfg.dtype), "attention_mask")
        weight_kwargs = {k: self.device_weights[k] for k in self._all_keys()}
        self.kernel = DeviceKernel.compile_and_load(
            t5_encoder,
            name=f"t5_encoder_s{S}",
            inputs_embeds=embeds,
            attention_mask=mask,
            configs=cfg,
            build_dir=BUILD_DIR,
            additional_compiler_args=cfg.additional_compiler_args_nkipy,
            **weight_kwargs,
        )
        self._out = DeviceTensor.from_numpy(
            np.empty((1, S, cfg.t5_d_model), dtype=cfg.dtype), "t5_out"
        )
        print(f"[t5] --> kernel ready in {time.time() - t:.2f}s")

    def encode(self, text):
        """Tokenize + embed on host, run 24 encoder layers on device.

        Returns (embeddings (1, S, d_model) torch, attention_mask (1, S) torch).
        """
        batch = self.tokenizer(
            text, padding="max_length", max_length=self.seq_len,
            truncation=True, return_tensors="pt",
        )
        embeds = torch.nn.functional.embedding(batch.input_ids, self.embed_weight)
        inputs = {
            "inputs_embeds": DeviceTensor.from_numpy(
                _to_numpy(embeds, bfloat16), "inputs_embeds"
            ),
            "attention_mask": DeviceTensor.from_numpy(
                _to_numpy(batch.attention_mask, bfloat16), "attention_mask"
            ),
        }
        inputs.update({k: self.device_weights[k] for k in self._all_keys()})
        self.kernel(inputs=inputs, outputs={"output0": self._out})
        return self._out.torch().to(torch.float32), batch.attention_mask
