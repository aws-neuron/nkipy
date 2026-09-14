"""CPU validation of the M1 primitives (3D RoPE + RMSNorm) vs diffusers.

Run from the example dir with the project venv:

    cd examples/models/qwen_image
    python -m pytest tests/test_rope_rmsnorm.py -v

Requires ``diffusers`` (>=0.39) and ``torch`` for the reference.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_qwen = pytest.importorskip(
    "diffusers.models.transformers.transformer_qwenimage",
    reason="diffusers with Qwen-Image support required",
)
from diffusers.models.normalization import RMSNorm  # noqa: E402

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernels import rope3d, rmsnorm  # noqa: E402


def _rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


AXES = (16, 56, 56)
HEAD_DIM = sum(AXES)  # 128
THETA = 10000.0


@pytest.mark.parametrize("frame,h,w,txt_len", [(1, 8, 8, 16), (1, 16, 12, 32)])
def test_rope_freqs_match_diffusers(frame, h, w, txt_len):
    """Our comptime angle tables must reproduce QwenEmbedRope's cos/sin."""
    rope = _qwen.QwenEmbedRope(theta=int(THETA), axes_dim=list(AXES), scale_rope=True)
    with torch.no_grad():
        vid_freqs, txt_freqs = rope(
            (frame, h, w), device=torch.device("cpu"), max_txt_seq_len=txt_len
        )
    # diffusers returns complex freqs (S, head_dim//2); compare cos/sin.
    ref_vid = vid_freqs.resolve_conj().cpu().numpy()  # complex
    ref_txt = txt_freqs.resolve_conj().cpu().numpy()

    vid_ang, txt_ang = rope3d.compute_rope_freqs(
        frame, h, w, txt_len, AXES, theta=THETA, scale_rope=True
    )
    assert vid_ang.shape == (frame * h * w, HEAD_DIM // 2)
    assert txt_ang.shape == (txt_len, HEAD_DIM // 2)

    # angle -> unit complex; compare cos & sin parts
    assert _rel_l2(np.cos(vid_ang), ref_vid.real) < 1e-5
    assert _rel_l2(np.sin(vid_ang), ref_vid.imag) < 1e-5
    assert _rel_l2(np.cos(txt_ang), ref_txt.real) < 1e-5
    assert _rel_l2(np.sin(txt_ang), ref_txt.imag) < 1e-5


@pytest.mark.parametrize("frame,h,w", [(1, 8, 8), (1, 16, 12)])
def test_apply_rotary_matches_diffusers(frame, h, w):
    """Rotating q with our real-form apply must match apply_rotary_emb_qwen."""
    B, H = 2, 3
    S = frame * h * w
    rng = np.random.default_rng(0)
    # our kernel uses (B, H, S, D); diffusers' apply_rotary_emb_qwen expects
    # (B, S, H, D) here (it unsqueezes freqs at the head axis), so transpose.
    x = rng.standard_normal((B, H, S, HEAD_DIM)).astype(np.float32)

    rope = _qwen.QwenEmbedRope(theta=int(THETA), axes_dim=list(AXES), scale_rope=True)
    with torch.no_grad():
        vid_freqs, _ = rope((frame, h, w), device=torch.device("cpu"), max_txt_seq_len=8)
        x_bshd = torch.from_numpy(x).permute(0, 2, 1, 3).contiguous()  # (B,S,H,D)
        ref = _qwen.apply_rotary_emb_qwen(
            x_bshd, vid_freqs, use_real=False
        ).permute(0, 2, 1, 3).contiguous().cpu().numpy()  # back to (B,H,S,D)

    vid_ang, _ = rope3d.compute_rope_freqs(frame, h, w, 8, AXES, theta=THETA, scale_rope=True)
    cos, sin = rope3d.cos_sin_from_angles(vid_ang, dtype=np.float32)
    out = rope3d.apply_rotary_emb(x, cos, sin)

    assert _rel_l2(out, ref) < 1e-5


@pytest.mark.parametrize("dim", [128, 3584])
def test_rmsnorm_matches_diffusers(dim):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 7, dim)).astype(np.float32)
    g = rng.standard_normal((dim,)).astype(np.float32)

    ref_mod = RMSNorm(dim, eps=1e-6, elementwise_affine=True)
    with torch.no_grad():
        ref_mod.weight.copy_(torch.from_numpy(g))
        ref = ref_mod(torch.from_numpy(x)).cpu().numpy()

    out = rmsnorm.rmsnorm_kernel(x, g, eps=1e-6)
    assert _rel_l2(out, ref) < 1e-5
