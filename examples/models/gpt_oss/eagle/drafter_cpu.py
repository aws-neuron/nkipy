"""CPU-side P-EAGLE drafter with KV cache for speculative decoding.

The P-EAGLE drafter maintains its own KV cache across the full context (prompt +
accepted tokens). At each draft step, K positions (1 NTP + K-1 MTP) attend to
the full accumulated cache via standard causal attention. After acceptance, the
accepted tokens' (embedding, target hidden) pairs extend the cache.

This runs entirely on CPU (the drafter is tiny — 4 layers, ~3.6 GB bf16). The
algorithm correctness is independent of where computation happens; this can be
moved to device later for throughput.
"""

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


class DrafterCPU:
    def __init__(self, model_path, target_hidden_size, num_draft_tokens=7):
        self.config = AutoConfig.from_pretrained(model_path)
        self.H = self.config.hidden_size
        self.K = num_draft_tokens
        self.target_hidden_size = target_hidden_size
        self.eps = self.config.rms_norm_eps
        self.n_heads = self.config.num_attention_heads
        self.n_kv = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim
        self.n_layers = self.config.num_hidden_layers  # 4 (midlayer + 3 plain)
        self.ptd_token_id = self.config.ptd_token_id

        # Load weights.
        with safe_open(f"{model_path}/model.safetensors", framework="pt") as f:
            self.w = {k: f.get_tensor(k).to(torch.bfloat16) for k in f.keys()}

        # Precompute RoPE.
        fn = ROPE_INIT_FUNCTIONS[self.config.rope_scaling["rope_type"]]
        inv_freq, self.rope_scaling = fn(self.config, None)
        self.inv_freq = inv_freq.float()

        # KV caches: list of (k, v) per layer, each (B, seq, n_kv, head_dim).
        self.kv_caches = None
        self.cache_len = 0

    def reset(self):
        self.kv_caches = [None] * self.n_layers
        self.cache_len = 0

    def rollback(self, new_len):
        """Truncate KV caches to new_len (discard rejected speculative entries)."""
        for i in range(self.n_layers):
            if self.kv_caches[i] is not None:
                k, v = self.kv_caches[i]
                self.kv_caches[i] = (k[:, :new_len], v[:, :new_len])
        self.cache_len = new_len

    # ── Building blocks ──────────────────────────────────────────────────────

    def _rms(self, x, w):
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w.float()).to(torch.bfloat16)

    def _rope_cos_sin(self, positions):
        """positions: 1-D int tensor of absolute positions."""
        freqs = torch.outer(positions.float(), self.inv_freq)
        cos = (freqs.cos() * self.rope_scaling).to(torch.bfloat16)
        sin = (freqs.sin() * self.rope_scaling).to(torch.bfloat16)
        return cos, sin  # (S, head_dim/2)

    def _apply_rope(self, x, cos, sin):
        """x: (B, H, S, D); cos/sin: (S, D/2)."""
        h = x.shape[-1] // 2
        cos = cos[None, None, :, :]  # (1,1,S,D/2)
        sin = sin[None, None, :, :]
        x0, x1 = x[..., :h], x[..., h:]
        return torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)

    def _attention(self, layer_idx, q_proj, k_proj, v_proj, o_proj, x, positions):
        """Self-attention with KV cache."""
        B, S, _ = x.shape
        nh, nkv, hd = self.n_heads, self.n_kv, self.head_dim
        rep = nh // nkv

        q = (x @ q_proj).view(B, S, nh, hd).transpose(1, 2)   # (B, nh, S, hd)
        k = (x @ k_proj).view(B, S, nkv, hd).transpose(1, 2)  # (B, nkv, S, hd)
        v = (x @ v_proj).view(B, S, nkv, hd).transpose(1, 2)

        # RoPE on the NEW positions only.
        cos, sin = self._rope_cos_sin(positions)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Update KV cache.
        if self.kv_caches[layer_idx] is None:
            self.kv_caches[layer_idx] = (k, v)
        else:
            pk, pv = self.kv_caches[layer_idx]
            self.kv_caches[layer_idx] = (torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2))

        # Full keys/values (cached + new).
        full_k, full_v = self.kv_caches[layer_idx]  # (B, nkv, total_len, hd)
        full_k = full_k.repeat_interleave(rep, dim=1)
        full_v = full_v.repeat_interleave(rep, dim=1)

        # Attention scores: q attends to full KV.
        total_len = full_k.shape[2]
        scores = (q @ full_k.transpose(2, 3)) / (hd ** 0.5)

        # Causal mask: position i (absolute) can attend to positions <= i.
        # Query positions are `positions`; key positions are 0..total_len-1.
        # Build mask: (S, total_len) where mask[i,j] = 0 if key_pos[j] <= query_pos[i], else -inf.
        key_pos = torch.arange(total_len, device=positions.device)
        mask = (key_pos[None, :] > positions[:, None]).float() * (-1e5)
        scores = scores + mask[None, None, :, :]  # broadcast over (B, nh)

        attn = F.softmax(scores.float(), dim=-1).to(torch.bfloat16)
        out = (attn @ full_v).transpose(1, 2).reshape(B, S, nh * hd)
        return out @ o_proj

    def _mlp(self, prefix, x):
        w = self.w
        gate = F.silu(x @ w[f"{prefix}.mlp.gate_proj.weight"].T)
        up = x @ w[f"{prefix}.mlp.up_proj.weight"].T
        return (gate * up) @ w[f"{prefix}.mlp.down_proj.weight"].T

    def _run_layers(self, x_2h, positions):
        """Run all drafter layers. x_2h: (B, S, 2H) concatenated [emb, hidden]."""
        w = self.w
        H = self.H

        # ── Fusion midlayer (layer 0) ──
        emb = x_2h[:, :, :H]
        hidden = x_2h[:, :, H:]
        residual = hidden
        hn = self._rms(hidden, w["midlayer.hidden_norm.weight"])
        en = self._rms(emb, w["midlayer.input_layernorm.weight"])
        attn_in = torch.cat([en, hn], dim=-1)  # (B, S, 2H)

        attn_out = self._attention(
            0,
            w["midlayer.self_attn.q_proj.weight"].T,
            w["midlayer.self_attn.k_proj.weight"].T,
            w["midlayer.self_attn.v_proj.weight"].T,
            w["midlayer.self_attn.o_proj.weight"].T,
            attn_in,
            positions,
        )
        x = residual + attn_out
        x = x + self._mlp("midlayer", self._rms(x, w["midlayer.post_attention_layernorm.weight"]))

        # ── Plain layers 1..N-1 ──
        for i in range(1, self.n_layers):
            p = f"layers.{i}"
            residual = x
            xn = self._rms(x, w[f"{p}.input_layernorm.weight"])
            attn_out = self._attention(
                i,
                w[f"{p}.self_attn.q_proj.weight"].T,
                w[f"{p}.self_attn.k_proj.weight"].T,
                w[f"{p}.self_attn.v_proj.weight"].T,
                w[f"{p}.self_attn.o_proj.weight"].T,
                xn,
                positions,
            )
            x = residual + attn_out
            x = x + self._mlp(p, self._rms(x, w[f"{p}.post_attention_layernorm.weight"]))

        return x

    def _fc_fuse(self, hidden3):
        """Project 3*target_hidden → hidden via fc weight."""
        return hidden3 @ self.w["fc.weight"].T

    # ── Public API ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def prefill(self, token_ids, aux_hidden_states):
        """Fill drafter KV cache with prompt context.

        Args:
            token_ids: (prompt_len,) int tensor of prompt tokens.
            aux_hidden_states: (1, prompt_len, 3*target_H) concatenated tap-layer
                hidden states from the target's prefill.
        """
        self.reset()
        S = len(token_ids)
        emb = self.w["embed_tokens.weight"][token_ids].unsqueeze(0)  # (1, S, H)
        hidden = self._fc_fuse(aux_hidden_states)  # (1, S, H)
        x_2h = torch.cat([emb, hidden], dim=-1)  # (1, S, 2H)
        positions = torch.arange(S)
        self._run_layers(x_2h, positions)
        self.cache_len = S

    @torch.no_grad()
    def draft(self, target_aux3, last_token_id, accepted_tokens=None, accepted_aux=None):
        """Generate K draft tokens attending to the full cached context.

        Args:
            target_aux3: (1, 1, 3*target_H) target hidden at the last accepted pos.
            last_token_id: int, the last accepted token.
            accepted_tokens: list[int], newly accepted tokens to add to cache first.
            accepted_aux: (1, n_accepted, 3*target_H) their target hidden states.

        Returns:
            list[int] of K draft token ids.
        """
        H = self.H
        new_positions = []

        # Step 1: Extend cache with newly accepted tokens (if any).
        if accepted_tokens is not None and len(accepted_tokens) > 0:
            A = len(accepted_tokens)
            acc_emb = self.w["embed_tokens.weight"][torch.tensor(accepted_tokens)].unsqueeze(0)
            acc_hidden = self._fc_fuse(accepted_aux)
            acc_2h = torch.cat([acc_emb, acc_hidden], dim=-1)
            acc_pos = torch.arange(self.cache_len, self.cache_len + A)
            self._run_layers(acc_2h, acc_pos)
            self.cache_len += A

        # Step 2: Build K positions (1 NTP + K-1 MTP).
        ntp_emb = self.w["embed_tokens.weight"][last_token_id].view(1, 1, H)
        ntp_hidden = self._fc_fuse(target_aux3)  # (1, 1, H)

        K = self.K
        if K > 1:
            ptd_emb = self.w["embed_tokens.weight"][self.ptd_token_id].view(1, 1, H)
            mtp_embs = ptd_emb.expand(1, K - 1, H)
            mask_hidden = self._fc_fuse(
                self.w["mask_hidden"].view(1, 1, -1)
            ).expand(1, K - 1, H)
            embs = torch.cat([ntp_emb, mtp_embs], dim=1)
            hiddens = torch.cat([ntp_hidden, mask_hidden], dim=1)
        else:
            embs = ntp_emb
            hiddens = ntp_hidden

        x_2h = torch.cat([embs, hiddens], dim=-1)  # (1, K, 2H)
        draft_positions = torch.arange(self.cache_len, self.cache_len + K)

        # Run through layers (this extends the KV cache with K new entries).
        x = self._run_layers(x_2h, draft_positions)

        # DON'T advance cache_len here — these are speculative positions.
        # They'll be rolled back if rejected. But we DO keep them in the cache
        # temporarily for the KV consistency.
        self.cache_len += K

        # Logits → draft tokens.
        x = self._rms(x, self.w["norm.weight"])
        logits = (x @ self.w["lm_head.weight"].T).float()  # (1, K, vocab)
        draft_ids = logits[0].argmax(dim=-1).tolist()

        # Map draft vocab → target vocab (identity for this checkpoint).
        d2t = self.w["d2t"].long()
        draft_ids = [(d + int(d2t[d])) for d in draft_ids]

        return draft_ids
