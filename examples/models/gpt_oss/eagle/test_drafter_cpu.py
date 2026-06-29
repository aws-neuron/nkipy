"""Regression tests for the CPU P-EAGLE drafter (drafter_cpu.DrafterCPU).

These exercise the KV-cache bookkeeping that drives parallel-drafting acceptance.
They need the P-EAGLE checkpoint locally and are skipped otherwise:

    pytest examples/models/gpt_oss/eagle/test_drafter_cpu.py \
        --draft-model /path/to/GPT-OSS-20B-P-EAGLE

The key invariant (verified on GPU against vLLM's eagle3 parallel-drafting path,
cosine 0.9999): a draft step's output must attend to the FULL context KV cache,
and rolling back rejected speculative entries must leave the cache identical to
one that never saw them. The original bug — rollback() slicing the n_kv axis
instead of the sequence axis — silently left rejected KV in place and corrupted
every subsequent draft step.
"""
import os

import pytest
import torch
import torch.nn.functional as F

DRAFT_MODEL = os.environ.get(
    "PEAGLE_DRAFT_MODEL", "/home/ubuntu/models/GPT-OSS-20B-P-EAGLE"
)
TARGET_HIDDEN = 2880  # gpt-oss-20b hidden size


def _have_checkpoint():
    return os.path.exists(os.path.join(DRAFT_MODEL, "model.safetensors"))


pytestmark = pytest.mark.skipif(
    not _have_checkpoint(),
    reason=f"P-EAGLE checkpoint not found at {DRAFT_MODEL}",
)


def _make():
    from drafter_cpu import DrafterCPU

    return DrafterCPU(DRAFT_MODEL, TARGET_HIDDEN, num_draft_tokens=7)


def test_rollback_restores_clean_cache():
    """Processing tokens after a rollback over rejected speculative entries must
    match a stateless full-context pass (the rollback bug regression test)."""
    torch.manual_seed(0)
    m = _make()
    H = m.H
    S = 24
    ids = torch.randint(0, 1000, (S,))
    aux = torch.randn(1, S, 3 * TARGET_HIDDEN).to(torch.bfloat16)
    emb = m.w["embed_tokens.weight"][ids].unsqueeze(0)
    hid = m._fc_fuse(aux)
    x2h = torch.cat([emb, hid], dim=-1).to(torch.bfloat16)

    # Reference: one stateless full-context pass.
    m.reset()
    ref = m._run_layers(x2h, torch.arange(S))
    ref = m._rms(ref, m.w["norm.weight"])[0].float()

    # Incremental: prefill 12, push 4 *speculative* junk slots, roll back to 8,
    # then process the real continuation 8..S-1.
    m.reset()
    m._run_layers(x2h[:, :12], torch.arange(12))
    m.cache_len = 12
    junk = torch.randn(1, 4, 2 * H).to(torch.bfloat16)
    m._run_layers(junk, torch.arange(12, 16))
    m.cache_len = 16
    m.rollback(8)
    out = m._run_layers(x2h[:, 8:], torch.arange(8, S))
    out = m._rms(out, m.w["norm.weight"])[0].float()

    cos = F.cosine_similarity(out, ref[8:], dim=-1)
    assert cos.min().item() > 0.999, f"rollback inconsistency, min cos={cos.min()}"


def test_rollback_slices_sequence_axis():
    """rollback() must truncate the sequence axis (dim 2), not n_kv (dim 1)."""
    m = _make()
    m.reset()
    S = 10
    ids = torch.randint(0, 1000, (S,))
    aux = torch.randn(1, S, 3 * TARGET_HIDDEN).to(torch.bfloat16)
    x2h = torch.cat(
        [m.w["embed_tokens.weight"][ids].unsqueeze(0), m._fc_fuse(aux)], dim=-1
    ).to(torch.bfloat16)
    m._run_layers(x2h, torch.arange(S))
    m.cache_len = S
    m.rollback(4)
    k, v = m.kv_caches[0]  # (B, n_kv, seq, head_dim)
    assert k.shape[2] == 4, f"sequence axis not truncated: {tuple(k.shape)}"
    assert k.shape[1] == m.n_kv, f"n_kv axis wrongly changed: {tuple(k.shape)}"


def test_draft_uses_full_context():
    """A draft token should depend on the prefilled context: changing the context
    must change the proposed drafts (guards against context-blind drafting)."""
    m = _make()
    K = m.K
    base = 16

    def drafts_for(seed):
        torch.manual_seed(seed)
        m.reset()
        ctx_ids = torch.randint(0, 1000, (base,))
        ctx_aux = torch.randn(1, base, 3 * TARGET_HIDDEN).to(torch.bfloat16)
        x2h = torch.cat(
            [m.w["embed_tokens.weight"][ctx_ids].unsqueeze(0), m._fc_fuse(ctx_aux)],
            dim=-1,
        ).to(torch.bfloat16)
        m._run_layers(x2h, torch.arange(base))
        m.cache_len = base
        commit_aux = torch.randn(1, 1, 3 * TARGET_HIDDEN).to(torch.bfloat16)
        return m.draft([5], commit_aux, base)

    d1 = drafts_for(1)
    d2 = drafts_for(2)
    assert len(d1) == K
    assert d1 != d2, "drafts identical across different contexts (context-blind)"
