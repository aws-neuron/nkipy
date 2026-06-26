"""Device-side P-EAGLE drafter: loads shards, compiles the parallel-draft kernel."""

import time

import numpy as np
import torch
from nkipy.runtime import DeviceKernel, DeviceTensor

from .config import EagleConfig
from .kernels.drafter import drafter_kernel

BUILD_DIR = None  # set by the caller (absolute path)


class DrafterModel:
    def __init__(self, weights, config: EagleConfig, build_dir):
        self.config = config
        self.build_dir = build_dir
        self.kernel = None
        self._prepare_tensors(weights)
        self._prepare_kernel()

    def _dt(self, t, name):
        return DeviceTensor.from_torch(t, name)

    def _prepare_tensors(self, w):
        cfg = self.config
        H = cfg.hidden_size

        # Shared tensors.
        self.embed_tokens = w["embed_tokens"]  # host, for embedding lookups
        self.fc_weight = self._dt(w["fc_weight"], "d_fc_weight")
        self.mask_hidden = self._dt(w["mask_hidden"], "d_mask_hidden")
        self.norm_weight = self._dt(w["norm_weight"], "d_norm_weight")
        self.lm_head_weight = self._dt(w["lm_head_weight"], "d_lm_head_weight")
        self.d2t = w["d2t"].to(torch.int64)
        self.ptd_emb = self._dt(
            self.embed_tokens[cfg.ptd_token_id].reshape(1, H), "d_ptd_emb"
        )

        # Fusion midlayer weights.
        self.m = {
            k: self._dt(w[f"midlayer.{k}"], f"d_m_{k}")
            for k in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "input_weight",
                "hidden_norm_weight",
                "post_attention_weight",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        }

        # Plain layers stacked on a leading axis (layers 1..N-1).
        plain_keys = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "input_weight",
            "post_attention_weight",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        self.p = {}
        for k in plain_keys:
            stacked = torch.stack(
                [w[f"layers.{i}.{k}"] for i in range(1, cfg.num_layers)]
            )
            self.p[k] = self._dt(stacked, f"d_p_{k}")

    def _prepare_kernel(self):
        cfg = self.config
        H = cfg.hidden_size
        B = cfg.max_batch_size
        target3 = DeviceTensor.from_numpy(
            np.empty((B, 1, 3 * cfg.target_hidden_size), dtype=cfg.dtype), "d_target3"
        )
        last_emb = DeviceTensor.from_numpy(
            np.empty((B, 1, H), dtype=cfg.dtype), "d_last_emb"
        )
        t = time.time()
        self.kernel = DeviceKernel.compile_and_load(
            drafter_kernel,
            name="drafter",
            target_hidden3=target3,
            last_emb=last_emb,
            ptd_emb=self.ptd_emb,
            fc_weight=self.fc_weight,
            mask_hidden=self.mask_hidden,
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            m_q_proj=self.m["q_proj"],
            m_k_proj=self.m["k_proj"],
            m_v_proj=self.m["v_proj"],
            m_o_proj=self.m["o_proj"],
            m_input_weight=self.m["input_weight"],
            m_hidden_norm_weight=self.m["hidden_norm_weight"],
            m_post_attention_weight=self.m["post_attention_weight"],
            m_gate_proj=self.m["gate_proj"],
            m_up_proj=self.m["up_proj"],
            m_down_proj=self.m["down_proj"],
            p_q_proj=self.p["q_proj"],
            p_k_proj=self.p["k_proj"],
            p_v_proj=self.p["v_proj"],
            p_o_proj=self.p["o_proj"],
            p_input_weight=self.p["input_weight"],
            p_post_attention_weight=self.p["post_attention_weight"],
            p_gate_proj=self.p["gate_proj"],
            p_up_proj=self.p["up_proj"],
            p_down_proj=self.p["down_proj"],
            cfg=cfg,
            build_dir=self.build_dir,
            additional_compiler_args=cfg.additional_compiler_args_nkipy,
        )
        self._draft_logits = DeviceTensor.from_numpy(
            np.empty(
                (B, cfg.num_draft_tokens, self.lm_head_weight.shape[1]),
                dtype=cfg.dtype,
            ),
            "d_draft_logits",
        )
        self._compile_time = time.time() - t

    def draft(self, target_hidden3, last_token_id):
        """Produce K draft token ids (global vocab) from the 3 tapped target hiddens.

        Args:
            target_hidden3: host tensor (B, 1, 3*target_hidden).
            last_token_id: int id of the last accepted token (B==1 assumed here).
        Returns:
            list[int] of K draft token ids in the target vocabulary.
        """
        cfg = self.config
        H = cfg.hidden_size
        target3_dev = DeviceTensor.from_torch(
            target_hidden3.to(torch.bfloat16), "target3_in"
        )
        last_emb = DeviceTensor.from_torch(
            self.embed_tokens[last_token_id].reshape(1, 1, H), "last_emb_in"
        )

        self.kernel(
            inputs={
                "target_hidden3": target3_dev,
                "last_emb": last_emb,
                "ptd_emb": self.ptd_emb,
                "fc_weight": self.fc_weight,
                "mask_hidden": self.mask_hidden,
                "norm_weight": self.norm_weight,
                "lm_head_weight": self.lm_head_weight,
                "m_q_proj": self.m["q_proj"],
                "m_k_proj": self.m["k_proj"],
                "m_v_proj": self.m["v_proj"],
                "m_o_proj": self.m["o_proj"],
                "m_input_weight": self.m["input_weight"],
                "m_hidden_norm_weight": self.m["hidden_norm_weight"],
                "m_post_attention_weight": self.m["post_attention_weight"],
                "m_gate_proj": self.m["gate_proj"],
                "m_up_proj": self.m["up_proj"],
                "m_down_proj": self.m["down_proj"],
                "p_q_proj": self.p["q_proj"],
                "p_k_proj": self.p["k_proj"],
                "p_v_proj": self.p["v_proj"],
                "p_o_proj": self.p["o_proj"],
                "p_input_weight": self.p["input_weight"],
                "p_post_attention_weight": self.p["post_attention_weight"],
                "p_gate_proj": self.p["gate_proj"],
                "p_up_proj": self.p["up_proj"],
                "p_down_proj": self.p["down_proj"],
            },
            outputs={"output0": self._draft_logits},
        )

        # Local (per-rank) draft logits -> draft token ids. We argmax over this
        # rank's vocab shard and remap via d2t; with vocab sharding the caller
        # reduces across ranks (see speculate.py). For the common single-vocab
        # checkpoint (lm_head replicated) this is already the global argmax.
        logits = self._draft_logits.torch().float()  # (B, K, vocab_local)
        draft_local = logits.argmax(dim=-1)[0]  # (K,)
        # Map draft-vocab id -> target-vocab id (identity when d2t is all-zero).
        draft_global = draft_local + self.d2t[draft_local]
        return draft_global.tolist()
