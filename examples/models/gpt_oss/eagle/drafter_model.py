"""Device-side P-EAGLE drafter with a KV cache.

Mirrors ``DrafterCPU`` (drafter_cpu.py) but runs the layer forward on the Neuron
device. The drafter keeps a per-layer KV cache over the full context (prompt +
accepted tokens); each draft step appends K positions (1 NTP + K-1 MTP) that
attend causally to the whole cache, exactly like the CPU reference. Only the fc
fusion and the (embed, hidden) input assembly happen on host (tiny); every
attention/MLP layer runs on device via ``drafter_layer_kernel``.

The host-driven per-layer loop matches the base model (``GptOssModel``): each
layer is a compiled kernel that aliases its own ``cache_k``/``cache_v``. Kernels
are compiled once per sequence width (prefill = prompt_len-1, commit = 1 row,
draft = K rows). Rollback of rejected speculative rows is implicit: committed
rows are re-written at their absolute positions and the causal mask prevents any
query from attending past its own position (same trick as the target's verify).
"""

import time

import numpy as np
import torch
from nkipy.runtime import DeviceKernel, DeviceTensor

from .config import EagleConfig
from .kernels.drafter import drafter_head_kernel, drafter_layer_kernel

BUILD_DIR = None  # set by the caller (absolute path)


class DrafterModel:
    def __init__(self, weights, config: EagleConfig, build_dir):
        self.config = config
        self.build_dir = build_dir
        self.H = config.hidden_size
        self.K = config.num_draft_tokens
        self.n_layers = config.num_layers
        self.cache_len = 0
        self._prepare_tensors(weights)
        self._prepare_kernels()

    def _dt(self, t, name):
        return DeviceTensor.from_torch(t, name)

    def _prepare_tensors(self, w):
        cfg = self.config
        H = cfg.hidden_size

        # Host tensors used to build the (embed, hidden) input stream.
        self.embed_tokens = w["embed_tokens"]  # (vocab, H) host
        self.fc_weight_host = w["fc_weight"].to(torch.float32)  # (3*target_H, H)
        self.mask_hidden_host = w["mask_hidden"].to(torch.float32)  # (1, 3*target_H)
        self.d2t = w["d2t"].to(torch.int64)
        self.ptd_token_id = cfg.ptd_token_id

        # fc-fused shared MTP hidden (host): fc(mask_hidden) -> (1, H).
        self.mask_hidden_fused = (self.mask_hidden_host @ self.fc_weight_host).to(
            torch.bfloat16
        )
        self.ptd_emb_host = self.embed_tokens[self.ptd_token_id].reshape(1, H)

        self.norm_weight = self._dt(w["norm_weight"], "d_norm_weight")
        self.lm_head_weight = self._dt(w["lm_head_weight"], "d_lm_head_weight")

        # Per-layer weight dicts (device) + KV caches. Layer 0 is the fusion
        # midlayer; layers 1..N-1 are plain. Caches are (B, max_seq, n_kv, hd).
        cache_shape = (
            cfg.max_batch_size,
            cfg.max_seq_len,
            cfg.num_kv_heads,
            cfg.head_dim,
        )
        self.layers = []
        for i in range(self.n_layers):
            prefix = "midlayer" if i == 0 else f"layers.{i}"
            lt = {
                "q_proj": self._dt(w[f"{prefix}.q_proj"], f"d_{prefix}_q"),
                "k_proj": self._dt(w[f"{prefix}.k_proj"], f"d_{prefix}_k"),
                "v_proj": self._dt(w[f"{prefix}.v_proj"], f"d_{prefix}_v"),
                "o_proj": self._dt(w[f"{prefix}.o_proj"], f"d_{prefix}_o"),
                "input_weight": self._dt(
                    w[f"{prefix}.input_weight"], f"d_{prefix}_in"
                ),
                "post_attention_weight": self._dt(
                    w[f"{prefix}.post_attention_weight"], f"d_{prefix}_post"
                ),
                "gate_proj": self._dt(w[f"{prefix}.gate_proj"], f"d_{prefix}_gate"),
                "up_proj": self._dt(w[f"{prefix}.up_proj"], f"d_{prefix}_up"),
                "down_proj": self._dt(w[f"{prefix}.down_proj"], f"d_{prefix}_down"),
                "cache_k": DeviceTensor.from_numpy(
                    np.zeros(cache_shape, dtype=cfg.dtype), f"d_cache_k_{i}"
                ),
                "cache_v": DeviceTensor.from_numpy(
                    np.zeros(cache_shape, dtype=cfg.dtype), f"d_cache_v_{i}"
                ),
            }
            if i == 0:
                lt["hidden_norm_weight"] = self._dt(
                    w["midlayer.hidden_norm_weight"], "d_midlayer_hn"
                )
            else:
                # Plain layers have no hidden_norm; reuse input_weight as a dummy
                # so the kernel signature is uniform (unused when is_fusion=False).
                lt["hidden_norm_weight"] = lt["input_weight"]
            self.layers.append(lt)

    def _compile_layer(self, name, i, x_sample, start_pos_sample):
        lt = self.layers[i]
        is_fusion = i == 0
        return DeviceKernel.compile_and_load(
            drafter_layer_kernel,
            name=name,
            x=x_sample,
            start_pos=start_pos_sample,
            q_proj=lt["q_proj"],
            k_proj=lt["k_proj"],
            v_proj=lt["v_proj"],
            o_proj=lt["o_proj"],
            input_weight=lt["input_weight"],
            hidden_norm_weight=lt["hidden_norm_weight"],
            post_attention_weight=lt["post_attention_weight"],
            gate_proj=lt["gate_proj"],
            up_proj=lt["up_proj"],
            down_proj=lt["down_proj"],
            cache_k=lt["cache_k"],
            cache_v=lt["cache_v"],
            cfg=self.config,
            is_fusion=is_fusion,
            build_dir=self.build_dir,
            additional_compiler_args=self.config.additional_compiler_args_nkipy,
        )

    def _prepare_kernels(self):
        cfg = self.config
        H = cfg.hidden_size
        B = cfg.max_batch_size
        t = time.time()

        start_pos_sample = DeviceTensor.from_numpy(
            np.empty((1,), dtype=np.int32), "d_start_pos"
        )

        # The draft step feeds K positions (1 NTP + K-1 MTP) at a runtime offset.
        # Fusion layer sees 2H; plain layers see H.
        def x2h(S, name):
            return DeviceTensor.from_numpy(
                np.empty((B, S, 2 * H), dtype=cfg.dtype), name
            )

        def x1h(S, name):
            return DeviceTensor.from_numpy(np.empty((B, S, H), dtype=cfg.dtype), name)

        # draft-width (K) kernels.
        self.kernel_draft = [None] * self.n_layers
        for i in range(self.n_layers):
            xin = x2h(self.K, f"d_x_draft_{i}") if i == 0 else x1h(self.K, f"d_x_draft_{i}")
            self.kernel_draft[i] = self._compile_layer(
                f"drafter_draft_L{i}", i, xin, start_pos_sample
            )

        # commit-width (1 row) kernels: extend the cache by one accepted token.
        self.kernel_commit = [None] * self.n_layers
        for i in range(self.n_layers):
            xin = x2h(1, f"d_x_commit_{i}") if i == 0 else x1h(1, f"d_x_commit_{i}")
            self.kernel_commit[i] = self._compile_layer(
                f"drafter_commit_L{i}", i, xin, start_pos_sample
            )

        # Head kernel over K positions.
        h_sample = x1h(self.K, "d_h_head")
        self.kernel_head = DeviceKernel.compile_and_load(
            drafter_head_kernel,
            name="drafter_head",
            h=h_sample,
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            cfg=cfg,
            build_dir=self.build_dir,
            additional_compiler_args=cfg.additional_compiler_args_nkipy,
        )
        self._draft_logits = DeviceTensor.from_numpy(
            np.empty((B, self.K, self.lm_head_weight.shape[1]), dtype=cfg.dtype),
            "d_draft_logits",
        )
        self._compile_time = time.time() - t

    # ── helpers ────────────────────────────────────────────────────────────

    def _fc_fuse(self, hidden3):
        """(., 3*target_H) host float -> (., H) bf16 via fc weight."""
        return (hidden3.to(torch.float32) @ self.fc_weight_host).to(torch.bfloat16)

    def _run_layer(self, kernels, i, x_dev, start_pos_dev):
        lt = self.layers[i]
        inputs = {
            "x": x_dev,
            "q_proj": lt["q_proj"],
            "k_proj": lt["k_proj"],
            "v_proj": lt["v_proj"],
            "o_proj": lt["o_proj"],
            "input_weight": lt["input_weight"],
            "hidden_norm_weight": lt["hidden_norm_weight"],
            "post_attention_weight": lt["post_attention_weight"],
            "gate_proj": lt["gate_proj"],
            "up_proj": lt["up_proj"],
            "down_proj": lt["down_proj"],
            "cache_k.must_alias_input": lt["cache_k"],
            "cache_v.must_alias_input": lt["cache_v"],
        }
        if start_pos_dev is not None:
            inputs["start_pos"] = start_pos_dev
        out = DeviceTensor.from_numpy(
            np.empty((x_dev.shape[0], x_dev.shape[1], self.H), dtype=self.config.dtype),
            f"d_layer_out_{i}",
        )
        kernels[i](
            inputs=inputs,
            outputs={"output0": out, "cache_k": lt["cache_k"], "cache_v": lt["cache_v"]},
        )
        return out

    def _run_stack(self, kernels, embeds, hiddens, start_pos):
        """Run fusion (2H) + plain (H) layers over S positions at abs start_pos.

        embeds/hiddens: host bf16 (B, S, H). Returns final host hidden (B, S, H).
        """
        cfg = self.config
        x_2h = torch.cat([embeds, hiddens], dim=-1)  # (B, S, 2H)
        sp = (
            None
            if start_pos is None
            else DeviceTensor.from_numpy(np.array([start_pos], dtype=np.int32), "sp")
        )
        x_dev = DeviceTensor.from_torch(x_2h, "d_stack_in")
        x_dev = self._run_layer(kernels, 0, x_dev, sp)  # fusion 2H -> H
        for i in range(1, self.n_layers):
            x_dev = self._run_layer(kernels, i, x_dev, sp)
        return x_dev

    # ── public API (matches DrafterCPU) ──────────────────────────────────────

    def reset(self):
        self.cache_len = 0

    @torch.no_grad()
    def prefill(self, token_ids, aux_hidden_states):
        """Fill the drafter KV cache with prompt context.

        token_ids: (S,) prompt tokens (EAGLE-shifted by the caller).
        aux_hidden_states: (1, S, 3*target_H) target tap hiddens.
        """
        self.reset()
        S = len(token_ids)
        emb = self.embed_tokens[torch.as_tensor(token_ids)].reshape(1, S, self.H)
        emb = emb.to(torch.bfloat16)
        hidden = self._fc_fuse(aux_hidden_states)  # (1, S, H)

        # Compile a prefill-width stack lazily (width depends on the prompt).
        if getattr(self, "_prefill_width", None) != S:
            self._compile_prefill(S)
        self._run_stack(self.kernel_prefill, emb, hidden, start_pos=None)
        self.cache_len = S

    def _compile_prefill(self, S):
        cfg = self.config
        H = cfg.hidden_size
        B = cfg.max_batch_size
        self.kernel_prefill = [None] * self.n_layers
        for i in range(self.n_layers):
            width = 2 * H if i == 0 else H
            xin = DeviceTensor.from_numpy(
                np.empty((B, S, width), dtype=cfg.dtype), f"d_x_prefill_{i}"
            )
            self.kernel_prefill[i] = self._compile_layer(
                f"drafter_prefill_L{i}_{S}", i, xin, None
            )
        self._prefill_width = S

    @torch.no_grad()
    def draft(self, commit_token_ids, commit_aux3, base_pos):
        """Generate K draft tokens for one speculation step (see DrafterCPU.draft).

        Commits the newly accepted tokens into the cache (one row each at their
        absolute positions), then runs K positions (NTP + K-1 MTP) attending to
        the full cache. Rejected speculative rows from the previous step are
        overwritten in place, so no explicit rollback is needed.
        """
        H, K = self.H, self.K
        C = len(commit_token_ids)
        assert C >= 1, "draft() needs at least the NTP (last committed) token"

        # 1) Commit each accepted token into the cache at its absolute position.
        commit_emb = self.embed_tokens[torch.as_tensor(commit_token_ids)].reshape(
            1, C, H
        ).to(torch.bfloat16)
        commit_hidden = self._fc_fuse(commit_aux3)  # (1, C, H)
        for j in range(C):
            self._run_stack(
                self.kernel_commit,
                commit_emb[:, j : j + 1, :],
                commit_hidden[:, j : j + 1, :],
                start_pos=base_pos + j,
            )
        self.cache_len = base_pos + C
        ntp_pos = base_pos + C - 1  # abs position of the NTP (last committed) slot

        # 2) Draft: K positions. depth 0 = NTP (re-run the last committed slot so
        #    its logit is available); depths 1..K-1 = MTP (ptd embed + mask hidden).
        #    All attend to the full cache under a block-causal mask.
        ntp_emb = commit_emb[:, C - 1 : C, :]
        ntp_hidden = commit_hidden[:, C - 1 : C, :]
        if K > 1:
            mtp_emb = self.ptd_emb_host.reshape(1, 1, H).expand(1, K - 1, H).to(
                torch.bfloat16
            )
            mtp_hidden = self.mask_hidden_fused.reshape(1, 1, H).expand(1, K - 1, H)
            embs = torch.cat([ntp_emb, mtp_emb], dim=1)  # (1, K, H)
            hiddens = torch.cat([ntp_hidden, mtp_hidden], dim=1)
        else:
            embs, hiddens = ntp_emb, ntp_hidden

        x_final = self._run_stack(self.kernel_draft, embs, hiddens, start_pos=ntp_pos)

        # Restore cache_len to the committed length: the K draft rows (written at
        # ntp_pos..ntp_pos+K-1) are speculative and will be overwritten when the
        # next step commits the accepted tokens at those same absolute positions.
        self.cache_len = base_pos + C

        # 3) Head: K logit rows -> K draft tokens.
        self.kernel_head(
            inputs={
                "h": x_final,
                "norm_weight": self.norm_weight,
                "lm_head_weight": self.lm_head_weight,
            },
            outputs={"output0": self._draft_logits},
        )
        logits = self._draft_logits.torch().float()[0]  # (K, vocab_local)
        self.last_logits = logits  # stashed for numerical cross-checks
        draft_local = logits.argmax(dim=-1)  # (K,)
        draft_global = draft_local + self.d2t[draft_local]
        return draft_global.tolist()
