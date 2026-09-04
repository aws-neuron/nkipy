"""Validate the host-driver latent plumbing without the 20B weights.

``_unpack_latents`` in ``qwen_image.py`` must match the diffusers
``QwenImagePipeline`` static method exactly (it is a copy; this guards against
drift) and round-trip the pipeline's packed layout back to the original latent.

    cd examples/models/qwen_image
    python -m pytest tests/test_pipeline_plumbing.py -v
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_diff = pytest.importorskip("diffusers", reason="diffusers with Qwen-Image required")
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_image import _unpack_latents  # noqa: E402


def test_unpack_matches_diffusers():
    B, C, H, W = 1, 4, 8, 8
    vae_scale = 2  # so unpack recovers H,W from packed
    latent = torch.randn(B, C, 1, H, W)  # Qwen VAE latents are 5-D (B,C,frame,H,W)

    # the runtime gets already-packed latents from ``pipe.prepare_latents``; source
    # the packed input the same way (pipeline packs the frame-squeezed latent).
    packed = QwenImagePipeline._pack_latents(latent[:, :, 0], B, C, H, W)

    # unpack round-trip (channels = C*4, recover to 5-D)
    height, width = H * vae_scale, W * vae_scale
    ours_un = _unpack_latents(packed, height, width, vae_scale)
    ref_un = QwenImagePipeline._unpack_latents(packed, height, width, vae_scale)
    assert torch.allclose(ours_un, ref_un)
    # and the packed->unpacked recovers the original latent
    assert torch.allclose(ours_un, latent, atol=1e-5)
