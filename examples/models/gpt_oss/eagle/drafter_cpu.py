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
        """Truncate KV caches to new_len (discard rejected speculative entries).

        Cache tensors are (B, n_kv, seq, head_dim); the sequence axis is dim 2.
        """
        for i in range(self.n_layers):
            if self.kv_caches[i] is not None:
                k, v = self.kv_caches[i]
                self.kv_caches[i] = (k[:, :, :new_len], v[:, :, :new_len])
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
    def draft(self, commit_token_ids, commit_aux3, base_pos):
        """Generate K draft tokens for one speculation step (parallel drafting).

        Validated against vLLM's eagle3 parallel-drafting path on GPU (the new
        positions attend to the FULL drafter KV cache via causal attention over
        absolute positions; cosine 0.9999, 100% draft-token match).

        Layout of the new positions appended to the cache (all attend to the
        full prior context):

            [ commit_0 ... commit_{C-1} | ptd_0 ... ptd_{K-2} ]
              ^ newly committed tokens     ^ K-1 MTP mask slots
              (real target hidden)         (fc(mask_hidden), ptd_token_id embed)

        The K draft predictions are the lm_head argmax at the **last committed
        slot** (NTP, depth 0) followed by the K-1 ptd slots (MTP, depths 1..K-1).

        EAGLE shift: ``commit_token_ids[i]`` is the token that *follows* the
        target hidden state ``commit_aux3[i]`` in the sequence (the drafter pairs
        ``embed(token_{p+1})`` with ``target_hidden(token_p)``).

        Args:
            commit_token_ids: list[int], the newly committed tokens (shifted; see
                above). Must be non-empty — its last entry is the NTP slot.
            commit_aux3: (1, C, 3*target_H) target tap-layer hiddens for the
                committed positions.
            base_pos: absolute position of ``commit_token_ids[0]``.

        Returns:
            list[int] of K draft token ids (target vocab).
        """
        H, K = self.H, self.K
        C = len(commit_token_ids)
        assert C >= 1, "draft() needs at least the NTP (last committed) token"

        # Roll back any speculative (ptd) slots left in the cache from the
        # previous step, so the committed tokens land at their true positions.
        if self.cache_len != base_pos:
            self.rollback(base_pos)

        # Committed slots: real embeddings + fc-fused real target hiddens.
        commit_emb = self.w["embed_tokens.weight"][
            torch.tensor(commit_token_ids)
        ].view(1, C, H)
        commit_hidden = self._fc_fuse(commit_aux3)  # (1, C, H)

        # MTP slots: ptd-token embedding + fc(mask_hidden), shared across depths.
        if K > 1:
            ptd_emb = self.w["embed_tokens.weight"][self.ptd_token_id].view(1, 1, H)
            mtp_emb = ptd_emb.expand(1, K - 1, H)
            mtp_hidden = self._fc_fuse(self.w["mask_hidden"].view(1, 1, -1)).expand(
                1, K - 1, H
            )
            embs = torch.cat([commit_emb, mtp_emb], dim=1)
            hiddens = torch.cat([commit_hidden, mtp_hidden], dim=1)
        else:
            embs, hiddens = commit_emb, commit_hidden

        x_2h = torch.cat([embs, hiddens], dim=-1)  # (1, C+K-1, 2H)
        n_new = C + K - 1
        positions = torch.arange(base_pos, base_pos + n_new)

        x = self._run_layers(x_2h, positions)
        # Keep the committed slots in the cache; the ptd slots are speculative and
        # will be rolled back at the start of the next draft() call.
        self.cache_len = base_pos + n_new

        # The K draft logits live at the last committed slot (NTP) + the K-1 ptd
        # slots (MTP). Indices into the n_new-wide window: [C-1, C, ..., C+K-2].
        x = self._rms(x, self.w["norm.weight"])
        logits = (x[0, C - 1 :] @ self.w["lm_head.weight"].T).float()  # (K, vocab)
        draft_local = logits.argmax(dim=-1)

        # Map draft vocab -> target vocab (identity when d2t is all-zero).
        d2t = self.w["d2t"].long()
        return (draft_local + d2t[draft_local]).tolist()
