"""Validate the NKIPy PixArt denoiser against a diffusers baseline.

Because sampling is iterative, small per-step errors compound; the most direct
correctness check is a *single-forward* comparison: feed identical (latent,
timestep, caption) into both the diffusers ``PixArtTransformer2DModel`` and the
NKIPy ``dit_forward`` kernel and compare the predicted noise (MSE / max-abs).

    python evaluate.py --validate \
        --model PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \
        --checkpoint ./tmp_pixart_sigma
"""

import argparse

import numpy as np
import torch
from config import get_config
from pixart import PixArtModel, load_model


def validate(args):
    model, config = load_model(args)

    N = 2 * config.max_batch_size
    H = config.sample_size
    gen = torch.Generator().manual_seed(args.seed)
    latent = torch.randn((N, config.in_channels, H, H), generator=gen)
    timestep = torch.tensor([500.0] * N)
    caption = torch.randn(
        (N, config.max_text_tokens, config.caption_channels), generator=gen
    ) * 0.1
    # exercise the padding mask: keep the first 32 tokens, mask the rest
    caption_mask = torch.zeros((N, config.max_text_tokens))
    caption_mask[:, :32] = 1.0

    # NKIPy device forward
    nk = model.predict_noise(latent, timestep, caption, caption_mask).float().numpy()

    # diffusers baseline (host, fp32)
    from diffusers import PixArtTransformer2DModel

    ref_model = PixArtTransformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.float32
    ).eval()
    with torch.no_grad():
        ref = ref_model(
            hidden_states=latent,
            encoder_hidden_states=caption,
            encoder_attention_mask=caption_mask,
            timestep=timestep,
            added_cond_kwargs={"resolution": None, "aspect_ratio": None},
        ).sample
    ref = ref[:, : config.in_channels].float().numpy()  # drop learned-sigma channels

    mse = float(np.mean((nk - ref) ** 2))
    max_abs = float(np.max(np.abs(nk - ref)))
    rel = float(np.linalg.norm(nk - ref) / (np.linalg.norm(ref) + 1e-8))
    print(f"[validate] MSE={mse:.4e}  max_abs={max_abs:.4e}  rel_l2={rel:.4e}")
    # bf16 device math vs fp32 host: a few percent rel error is expected
    ok = rel < args.tol
    print("[validate] PASS" if ok else "[validate] FAIL")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--model", default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--checkpoint", default="./tmp_pixart_sigma")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=0.05)
    args = parser.parse_args()

    if args.validate:
        ok = validate(args)
        raise SystemExit(0 if ok else 1)
    parser.error("nothing to do; pass --validate")


if __name__ == "__main__":
    main()
