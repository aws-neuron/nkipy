"""Device-side P-EAGLE drafter with a KV cache, fused into one kernel per step.

Runs the whole drafter forward as a single Neuron kernel: fc-fusion +
(embed, hidden) assembly + all 4 layers + final norm + lm_head, with each layer
aliasing its own persistent KV cache. The drafter keeps a per-layer KV cache
over the full context (prompt + accepted tokens); each draft step runs
W = C + K - 1 positions that attend causally to the whole cache.

Only the token-embedding gather stays on host (dynamic ids); everything else is
on device. Kernels are compiled once per sequence width (prefill = prompt_len-1,
draft W = K..2K) at load time. Rollback of rejected speculative rows is implicit:
committed rows are re-written at their absolute positions and the causal mask
prevents any query from attending past its own position (same trick as the
target's verify).
"""

import os
import time

import numpy as np
import torch
from nkipy.runtime import DeviceKernel, DeviceTensor

from .config import EagleConfig
from .kernels.drafter import drafter_fused_kernel

BUILD_DIR = None  # set by the caller (absolute path)


class DrafterModel:
    def __init__(self, weights, config: EagleConfig, build_dir):
        self.config = config
        self.build_dir = build_dir
        self.H = config.hidden_size
        self.K = config.num_draft_tokens
        self.n_layers = config.num_layers
        assert self.n_layers == 4, "fused kernel is wired for 4 layers (1 fusion + 3)"
        self.cache_len = 0
        self._prepare_tensors(weights)
        self._prepare_kernels()

    def _dt(self, t, name):
        return DeviceTensor.from_torch(t, name)

    def _prepare_tensors(self, w):
        cfg = self.config
        H = cfg.hidden_size

        # Host: token embeddings (dynamic gather each step) + d2t remap.
        self.embed_tokens = w["embed_tokens"]  # (vocab, H) host
        self.ptd_emb_host = self.embed_tokens[cfg.ptd_token_id].reshape(1, H)
        self.d2t = w["d2t"].to(torch.int64)

        # Device: fc weight + mask hidden (fc-fuse now runs on device).
        self.fc_weight = self._dt(w["fc_weight"], "d_fc_weight")  # (3*tH, H)
        self.mask_hidden = self._dt(
            w["mask_hidden"].reshape(1, -1), "d_mask_hidden"
        )  # (1, 3*tH)
        self.norm_weight = self._dt(w["norm_weight"], "d_norm_weight")
        self.lm_head_weight = self._dt(w["lm_head_weight"], "d_lm_head_weight")

        # Fusion midlayer weights (layer 0).
        self.m = {
            k: self._dt(w[f"midlayer.{k}"], f"d_m_{k}")
            for k in [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "input_weight", "hidden_norm_weight", "post_attention_weight",
                "gate_proj", "up_proj", "down_proj",
            ]
        }

        # Plain layers 1..N-1, stacked on a leading axis.
        plain_keys = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "input_weight", "post_attention_weight",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.p = {}
        for k in plain_keys:
            stacked = torch.stack(
                [w[f"layers.{i}.{k}"] for i in range(1, cfg.num_layers)]
            )
            self.p[k] = self._dt(stacked, f"d_p_{k}")

        # Per-layer KV caches (separate named device tensors, aliased in+out).
        cache_shape = (
            cfg.max_batch_size, cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim
        )
        self.cache_k = [
            DeviceTensor.from_numpy(np.zeros(cache_shape, dtype=cfg.dtype), f"d_ck_{i}")
            for i in range(self.n_layers)
        ]
        self.cache_v = [
            DeviceTensor.from_numpy(np.zeros(cache_shape, dtype=cfg.dtype), f"d_cv_{i}")
            for i in range(self.n_layers)
        ]

    def _weight_inputs(self):
        """The (static) weight + cache inputs common to every fused kernel call."""
        d = {
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
        }
        for i in range(self.n_layers):
            d[f"cache_k{i}.must_alias_input"] = self.cache_k[i]
            d[f"cache_v{i}.must_alias_input"] = self.cache_v[i]
        return d

    def _cache_outputs(self):
        out = {}
        for i in range(self.n_layers):
            out[f"cache_k{i}"] = self.cache_k[i]
            out[f"cache_v{i}"] = self.cache_v[i]
        return out

    def _compile_fused(self, W, C, name):
        """Compile the fused kernel specialized to width W and committed-count C."""
        cfg = self.config
        H, B = cfg.hidden_size, cfg.max_batch_size
        embeds = DeviceTensor.from_numpy(
            np.empty((B, W, H), dtype=cfg.dtype), f"d_embeds_{name}"
        )
        th3 = DeviceTensor.from_numpy(
            np.empty((B, C, 3 * cfg.target_hidden_size), dtype=cfg.dtype),
            f"d_th3_{name}",
        )
        start_pos = DeviceTensor.from_numpy(np.empty((1,), dtype=np.int32), f"d_sp_{name}")
        kw = {
            "embeds": embeds,
            "target_hidden3": th3,
            "start_pos": start_pos,
        }
        kw.update(self._weight_inputs())
        # compile_and_load takes DeviceTensors positionally-by-kwarg; drop the
        # ".must_alias_input" suffix used only at call time.
        compile_kw = {k.split(".")[0]: v for k, v in kw.items()}
        return DeviceKernel.compile_and_load(
            drafter_fused_kernel,
            name=name,
            cfg=cfg,
            build_dir=self.build_dir,
            additional_compiler_args=cfg.additional_compiler_args_nkipy,
            **compile_kw,
        )

    def _prepare_kernels(self):
        t = time.time()
        # Draft step: width W = C + K - 1, C in 1..K+1, so W in K..2K. For each W
        # the committed count is C = W - K + 1. Compile all upfront (a lazy compile
        # inside the decode loop would stall a step by tens of seconds).
        self.kernel_draft = {}  # W -> kernel
        self.draft_logits = {}  # W -> (B, W, vocab) buffer
        for W in range(self.K, 2 * self.K + 1):
            C = W - self.K + 1
            self.kernel_draft[W] = self._compile_fused(W, C, f"drafter_fused_W{W}")
            self.draft_logits[W] = DeviceTensor.from_numpy(
                np.empty(
                    (self.config.max_batch_size, W, self.lm_head_weight.shape[1]),
                    dtype=self.config.dtype,
                ),
                f"d_logits_W{W}",
            )
        self._prefill_kernel = None
        self._prefill_width = None
        self._compile_time = time.time() - t

    # ── helpers ──────────────────────────────────────────────────────────────

    def _embed(self, token_ids):
        ids = torch.as_tensor(token_ids)
        return self.embed_tokens[ids].reshape(1, len(ids), self.H).to(torch.bfloat16)

    def _run_fused(self, kernel, logits_buf, embeds, target_hidden3, start_pos):
        """Invoke the fused kernel: upload inputs, run, return host logits (W,vocab)."""
        inputs = {
            "embeds": self._dt(embeds, "embeds_in"),
            "target_hidden3": self._dt(target_hidden3.to(torch.bfloat16), "th3_in"),
            "start_pos": DeviceTensor.from_numpy(
                np.array([start_pos], dtype=np.int32), "sp_in"
            ),
        }
        inputs.update(self._weight_inputs())
        outputs = {"output0": logits_buf}
        outputs.update(self._cache_outputs())
        kernel(inputs=inputs, outputs=outputs)
        return logits_buf.torch().float()[0]  # (W, vocab_local)

    # ── public API ─────────────────────────────────────────────────────────────

    def reset(self):
        self.cache_len = 0

    @torch.no_grad()
    def prefill(self, token_ids, aux_hidden_states):
        """Fill the drafter KV cache with prompt context (single fused pass).

        token_ids: (S,) prompt tokens (EAGLE-shifted by the caller).
        aux_hidden_states: (1, S, 3*target_H) target tap hiddens.
        """
        self.reset()
        S = len(token_ids)
        embeds = self._embed(token_ids)  # (1, S, H)
        th3 = aux_hidden_states  # all S rows are "committed" (C == S, no MTP slots)

        if self._prefill_width != S:
            self._prefill_kernel = self._compile_fused(S, S, f"drafter_prefill_{S}")
            self._prefill_logits = DeviceTensor.from_numpy(
                np.empty(
                    (self.config.max_batch_size, S, self.lm_head_weight.shape[1]),
                    dtype=self.config.dtype,
                ),
                f"d_plogits_{S}",
            )
            self._prefill_width = S
        self._run_fused(self._prefill_kernel, self._prefill_logits, embeds, th3, 0)
        self.cache_len = S

    @torch.no_grad()
    def draft(self, commit_token_ids, commit_aux3, base_pos):
        """Generate K draft tokens for one speculation step.

        Runs ONE fused forward of width W = C + K - 1 over
        ``[commit_0 .. commit_{C-1} | ptd_0 .. ptd_{K-2}]`` at consecutive absolute
        positions ``base_pos + [0..W-1]``, all attending to the full cache. The K
        draft logits are rows ``[C-1 .. W-1]`` (last committed slot + K-1 MTP).
        Rejected speculative rows from the previous step are overwritten in place.
        """
        H, K = self.H, self.K
        C = len(commit_token_ids)
        assert C >= 1, "draft() needs at least the NTP (last committed) token"
        W = C + K - 1

        _prof = os.environ.get("SPEC_PROFILE") == "1"
        if _prof:
            _td = time.time()

        # Embedding half: committed token embeds + K-1 ptd embeds. The hidden half
        # (fc-fuse of target_hidden3 / mask_hidden) is built inside the kernel.
        commit_emb = self._embed(commit_token_ids)  # (1, C, H)
        if K > 1:
            mtp_emb = self.ptd_emb_host.reshape(1, 1, H).expand(1, K - 1, H).to(
                torch.bfloat16
            )
            embeds = torch.cat([commit_emb, mtp_emb], dim=1)  # (1, W, H)
        else:
            embeds = commit_emb

        logits = self._run_fused(
            self.kernel_draft[W], self.draft_logits[W], embeds, commit_aux3, base_pos
        )
        # Committed rows land at their true absolute positions; the K-1 ptd rows are
        # speculative and get overwritten when the next step commits accepted tokens.
        self.cache_len = base_pos + C

        logits = logits[C - 1 :]  # (K, vocab_local)
        self.last_logits = logits  # stashed for numerical cross-checks
        draft_local = logits.argmax(dim=-1)  # (K,)
        draft_global = draft_local + self.d2t[draft_local]
        if _prof:
            self._t_forward = getattr(self, "_t_forward", 0.0) + time.time() - _td
        return draft_global.tolist()
