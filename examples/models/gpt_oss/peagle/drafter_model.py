"""Device-side P-EAGLE drafter with a KV cache, fused into one kernel per step.

Runs the whole drafter forward as a single Neuron kernel: fc-fusion +
(embed, hidden) assembly + all 4 layers + final norm + lm_head, with each layer
aliasing its own persistent KV cache. The drafter keeps a per-layer KV cache
over the full context (prompt + accepted tokens); each draft step runs
W = C + K - 1 positions that attend causally to the whole cache.

Only the token-embedding gather stays on host (dynamic ids); everything else is
on device. A single max-width draft kernel (W = 2K) is compiled up front at load
time and every step pads up to it (the acceptance count C, hence the natural
width, varies per step); the prefill kernel is compiled lazily on the first
prefill() since prompt_len isn't known until then (it runs once, outside the
decode loop). Rollback of rejected speculative rows is implicit:
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


class DrafterModel:
    def __init__(self, weights, config: EagleConfig, build_dir):
        self.config = config
        self.build_dir = build_dir
        self.H = config.hidden_size
        self.K = config.num_draft_tokens
        self.n_layers = config.num_layers
        assert self.n_layers == 4, "fused kernel is wired for 4 layers (1 fusion + 3)"
        self._prepare_tensors(weights)
        self._prepare_kernels()

    def _to_device(self, tensor, name):
        return DeviceTensor.from_torch(tensor, name)

    def _prepare_tensors(self, weights):
        cfg = self.config
        H = cfg.hidden_size

        # Host: token embeddings (dynamic gather each step) + d2t remap.
        self.embed_tokens = weights["embed_tokens"]  # (vocab, H) host
        self.ptd_emb_host = self.embed_tokens[cfg.ptd_token_id].reshape(1, H)
        self.d2t = weights["d2t"].to(torch.int64)

        # Device: fc weight + mask hidden (fc-fuse now runs on device).
        # fc_weight: (3*tH, H)
        self.fc_weight = self._to_device(weights["fc_weight"], "d_fc_weight")
        mask_hidden = weights["mask_hidden"].reshape(1, -1)  # (1, 3*tH)
        self.mask_hidden = self._to_device(mask_hidden, "d_mask_hidden")
        # Host copy of the raw mask hidden: pads target_hidden3 up to the fixed
        # committed-count C=K+1 so the single max-width kernel sees the same
        # fc(mask_hidden) MTP rows the variable-width kernels used to build.
        self.mask_hidden_host = mask_hidden.to(torch.bfloat16)
        self.norm_weight = self._to_device(weights["norm_weight"], "d_norm_weight")
        self.lm_head_weight = self._to_device(
            weights["lm_head_weight"], "d_lm_head_weight"
        )

        # Fusion midlayer weights (layer 0).
        self.fusion_weights = {
            k: self._to_device(weights[f"midlayer.{k}"], f"d_m_{k}")
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
        self.plain_weights = {}
        for k in plain_keys:
            stacked = torch.stack(
                [weights[f"layers.{i}.{k}"] for i in range(1, cfg.num_layers)]
            )
            self.plain_weights[k] = self._to_device(stacked, f"d_p_{k}")

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
            "m_q_proj": self.fusion_weights["q_proj"],
            "m_k_proj": self.fusion_weights["k_proj"],
            "m_v_proj": self.fusion_weights["v_proj"],
            "m_o_proj": self.fusion_weights["o_proj"],
            "m_input_weight": self.fusion_weights["input_weight"],
            "m_hidden_norm_weight": self.fusion_weights["hidden_norm_weight"],
            "m_post_attention_weight": self.fusion_weights["post_attention_weight"],
            "m_gate_proj": self.fusion_weights["gate_proj"],
            "m_up_proj": self.fusion_weights["up_proj"],
            "m_down_proj": self.fusion_weights["down_proj"],
            "p_q_proj": self.plain_weights["q_proj"],
            "p_k_proj": self.plain_weights["k_proj"],
            "p_v_proj": self.plain_weights["v_proj"],
            "p_o_proj": self.plain_weights["o_proj"],
            "p_input_weight": self.plain_weights["input_weight"],
            "p_post_attention_weight": self.plain_weights["post_attention_weight"],
            "p_gate_proj": self.plain_weights["gate_proj"],
            "p_up_proj": self.plain_weights["up_proj"],
            "p_down_proj": self.plain_weights["down_proj"],
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
        start_pos = DeviceTensor.from_numpy(
            np.empty((1,), dtype=np.int32), f"d_sp_{name}"
        )
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
        # Draft step: the acceptance count C is in 1..K+1, so the natural width
        # W = C + K - 1 varies over K..2K. Rather than compile one shape-static
        # kernel per width, compile ONE kernel at the max width (W = 2K, C = K+1)
        # and pad every call up to it (see draft()). Padding is safe: the extra
        # rows sit at absolute positions past the draft window, so the causal mask
        # hides them from every real row and their cache writes get overwritten by
        # a later commit -- the same invariant that already lets speculative MTP
        # rows be rewritten. Cuts cold compile from K+1 kernels to 1.
        self.draft_W = 2 * self.K
        self.draft_C = self.K + 1
        self.kernel_draft = self._compile_fused(
            self.draft_W, self.draft_C, "drafter_fused"
        )
        self.draft_logits = DeviceTensor.from_numpy(
            np.empty(
                (
                    self.config.max_batch_size,
                    self.draft_W,
                    self.lm_head_weight.shape[1],
                ),
                dtype=self.config.dtype,
            ),
            "d_logits",
        )
        self._prefill_kernel = None
        self._prefill_width = None
        self._compile_time = time.time() - t

    # ── helpers ──────────────────────────────────────────────────────────────

    def _embed(self, token_ids):
        # torch side hardcoded to bf16 to match cfg.dtype (an NKI/numpy dtype, not
        # a torch dtype, so it can't be substituted directly here).
        ids = torch.as_tensor(token_ids)
        return self.embed_tokens[ids].reshape(1, len(ids), self.H).to(torch.bfloat16)

    def _run_fused(self, kernel, logits_buf, embeds, target_hidden3, start_pos):
        """Invoke the fused kernel: upload inputs, run, return host logits (W,vocab)."""
        inputs = {
            "embeds": self._to_device(embeds, "embeds_in"),
            "target_hidden3": self._to_device(
                target_hidden3.to(torch.bfloat16), "th3_in"
            ),
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

    @torch.no_grad()
    def prefill(self, token_ids, aux_hidden_states):
        """Fill the drafter KV cache with prompt context (single fused pass).

        token_ids: (S,) prompt tokens (EAGLE-shifted by the caller).
        aux_hidden_states: (1, S, 3*target_H) target tap hiddens.

        The KV cache needs no explicit clearing between sequences: every row is
        re-written at its absolute position and the causal mask blocks any query
        from attending past its own position, so stale rows are never read.
        """
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

    @torch.no_grad()
    def draft(self, commit_token_ids, commit_aux3, base_pos):
        """Generate K draft tokens for one speculation step.

        ``C = len(commit_token_ids)`` is the number of tokens the target ACCEPTED
        in the previous verify (the acceptance length, 1..K+1; averaged over a run
        this is the "mean acceptance length"). These accepted tokens have not been
        seen by the drafter yet, so this step first commits them into the KV cache
        -- carrying their real target hidden states (``commit_aux3``, which only
        exist after verify) -- and then drafts the next K tokens, all in one pass.

        Why re-feed all C accepted tokens, not just the single newest one? The
        *previous* step wrote KV for these positions from placeholders
        (``ptd_token_id`` embeddings + ``mask_hidden``), since at draft time it did
        not yet know which of its K-1 MTP slots the target would accept. Now that
        the target has confirmed C of them, this step overwrites those stale
        mask-derived rows -- at their true absolute positions -- with the real
        token embeds + target hidden, so later positions (this step's MTP slots and
        every future step) attend to correct context. A "1 real + K-1 mask" layout
        would only be right if C were always 1 (i.e. the target accepted exactly
        one token per step, defeating speculation); the C-1 other accepted rows
        would keep their placeholder KV forever.

        The natural width is W = C + K - 1 over
        ``[commit_0 .. commit_{C-1} | ptd_0 .. ptd_{K-2}]`` at consecutive absolute
        positions ``base_pos + [0..W-1]``, all attending to the full cache. The
        width is C + K - 1 (not C + K) because the NTP draft is read from the last
        committed row itself, so only K-1 extra placeholder (ptd/MTP) slots are
        appended. The K draft logits are rows ``[C-1 .. C+K-2]`` (last committed
        slot + K-1 MTP). Rejected speculative rows from the previous step are
        overwritten in place.

        A single kernel of the max width ``draft_W = 2K`` (fixed ``draft_C = K+1``)
        handles every C: we pad the ids with extra ptd placeholders and pad
        ``target_hidden3`` with the raw ``mask_hidden`` (so the padded rows fc-fuse
        to exactly the MTP hidden the kernel would build anyway). The padding rows
        land at absolute positions >= base_pos + W, past this step's draft window,
        so no real row attends to them and their cache writes are overwritten later.
        """
        H, K = self.H, self.K
        C = len(commit_token_ids)  # tokens accepted by the target last step
        assert C >= 1, "draft() needs at least the NTP (last committed) token"
        assert C <= self.draft_C, (
            f"C={C} exceeds max acceptance length K+1={self.draft_C}"
        )

        _prof = os.environ.get("SPEC_PROFILE") == "1"
        if _prof:
            _td = time.time()

        # Embedding half: C committed token embeds, then ptd embeds padding out to
        # the fixed draft width. The hidden half (fc-fuse of target_hidden3 /
        # mask_hidden) is built inside the kernel.
        commit_emb = self._embed(commit_token_ids)  # (1, C, H)
        n_pad_emb = self.draft_W - C
        if n_pad_emb > 0:
            ptd_emb = self.ptd_emb_host.reshape(1, 1, H).expand(1, n_pad_emb, H).to(
                torch.bfloat16
            )
            embeds = torch.cat([commit_emb, ptd_emb], dim=1)  # (1, draft_W, H)
        else:
            embeds = commit_emb

        # Hidden half: C real target hiddens, padded to draft_C with raw mask_hidden
        # (which fc-fuses to the MTP hidden, matching the kernel's own MTP rows).
        n_pad_h = self.draft_C - C
        if n_pad_h > 0:
            mask_pad = self.mask_hidden_host.reshape(1, 1, -1).expand(
                1, n_pad_h, -1
            )
            target_hidden3 = torch.cat(
                [commit_aux3.to(torch.bfloat16), mask_pad], dim=1
            )
        else:
            target_hidden3 = commit_aux3

        logits = self._run_fused(
            self.kernel_draft, self.draft_logits, embeds, target_hidden3, base_pos
        )
        # Committed rows land at their true absolute positions; the ptd rows are
        # speculative and get overwritten when the next step commits accepted tokens.
        logits = logits[C - 1 : C - 1 + K]  # (K, vocab_local)
        self.last_logits = logits  # stashed for numerical cross-checks
        draft_local = logits.argmax(dim=-1)  # (K,)
        draft_global = draft_local + self.d2t[draft_local]
        if _prof:
            self._t_forward = getattr(self, "_t_forward", 0.0) + time.time() - _td
        return draft_global.tolist()
