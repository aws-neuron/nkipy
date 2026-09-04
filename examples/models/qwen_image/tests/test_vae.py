"""CPU validation of the Qwen-Image VAE decoder (T=1 2D collapse) vs diffusers.

The device VAE (``kernels/vae.py``) decodes a single image (T=1), where the 3D
causal *video* VAE collapses exactly to a 2D conv decoder. This test builds a
small ``AutoencoderKLQwenImage``, decodes a fixed latent through both diffusers
and our numpy kernel, and checks rel_l2 < 1e-3.

    cd examples/models/qwen_image
    python -m pytest tests/test_vae.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_diff = pytest.importorskip("diffusers", reason="diffusers with Qwen-Image required")
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (  # noqa: E402
    AutoencoderKLQwenImage,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import VAEConfig  # noqa: E402
from kernels.vae import vae_decode  # noqa: E402
from weight_extract import extract_vae_decoder_weights  # noqa: E402


def _rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _build_small_vae(base_dim, z_dim, dim_mult, num_res_blocks):
    torch.manual_seed(0)
    return AutoencoderKLQwenImage(
        base_dim=base_dim, z_dim=z_dim, dim_mult=list(dim_mult),
        num_res_blocks=num_res_blocks, temperal_downsample=[False, True, True],
    ).eval()


@pytest.mark.parametrize("base_dim,dim_mult", [(8, (1, 2, 4, 4)), (12, (1, 2, 2, 4))])
def test_vae_decode_matches_diffusers(base_dim, dim_mult):
    z_dim, nrb = 16, 2
    vae = _build_small_vae(base_dim, z_dim, dim_mult, nrb)

    rng = np.random.default_rng(0)
    lat = rng.standard_normal((1, z_dim, 1, 4, 4)).astype(np.float32)  # (B,C,T,H,W)

    with torch.no_grad():
        ref = vae.decode(torch.from_numpy(lat), return_dict=False)[0].cpu().numpy()
    ref = ref[:, :, 0]  # squeeze the single frame -> (1, 3, H, W)

    cfg = VAEConfig(z_dim=z_dim, base_dim=base_dim, dim_mult=dim_mult,
                    num_res_blocks=nrb)
    flat = extract_vae_decoder_weights(vae, dtype=np.float32)
    out = vae_decode(lat[:, :, 0], cfg, **flat)  # host squeezes T; kernel is 2D

    assert out.shape == ref.shape, (out.shape, ref.shape)
    rel = _rel_l2(out, ref)
    assert rel < 1e-3, f"rel_l2={rel:.3e}"


def test_vae_8x_upscale_shape():
    """Full Qwen VAE spatial ratio is 8x (dim_mult of length 4 -> 3 upsamples)."""
    vae = _build_small_vae(8, 16, (1, 2, 4, 4), 2)
    cfg = VAEConfig(z_dim=16, base_dim=8, dim_mult=(1, 2, 4, 4), num_res_blocks=2)
    flat = extract_vae_decoder_weights(vae, dtype=np.float32)
    lat = np.zeros((1, 16, 5, 5), dtype=np.float32)
    out = vae_decode(lat, cfg, **flat)
    assert out.shape == (1, 3, 40, 40)  # 5 * 8
