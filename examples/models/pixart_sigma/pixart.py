"""PixArt-Sigma text-to-image on Trainium.

Correctness-first layout:

* The DiT denoiser runs on Trainium as a single compiled ``dit_forward`` kernel
  (see ``kernels/transformer.py``), called once per sampling step on a
  classifier-free-guidance batch of 2.
* The T5 text encoder and the VAE decoder run on the host (``transformers`` /
  ``diffusers``) so the example stays self-contained and the device work is
  purely the transformer.

Weights are loaded from a directory produced by ``tensor_preparation.py``
(a single ``weights.safetensors`` with the flat canonical key scheme in
``kernels/weight_layout.py``).
"""

import argparse
import os
import time

import numpy as np
import torch
from config import Config, get_config
from kernels.transformer import denoise_step, dit_forward
from kernels.weight_layout import BLOCK_KEYS, SHARED_KEYS, block_key
from nkipy.runtime import DeviceKernel, DeviceTensor
from nkipy.runtime.device_tensor import bfloat16
from safetensors.torch import load_file

BUILD_DIR = "./build"


def _to_numpy(t, dtype):
    """torch tensor -> contiguous numpy array of the given numpy dtype.

    Routes through float32 first because torch cannot export bfloat16 via
    ``.numpy()``; the final ``.astype`` produces the ml_dtypes bf16 view the
    runtime expects (and a real ``np.dtype`` so device dtype checks pass).
    """
    return t.detach().to(torch.float32).cpu().numpy().astype(dtype)


class PixArtModel:
    """The Trainium DiT denoiser (host encoder / VAE live in the driver)."""

    def __init__(self, weights, config: Config):
        self.config = config
        self._prepare_tensors(weights)
        self._prepare_kernel()

    def _prepare_tensors(self, weights):
        t = time.time()
        print("[pixart] preparing device tensors")
        self.device_weights = {}
        for key, tensor in weights.items():
            self.device_weights[key] = DeviceTensor.from_torch(
                tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor,
                name=key,
            )
        print(f"[pixart] --> tensors ready in {time.time() - t:.2f}s")

    def _all_weight_keys(self):
        keys = list(SHARED_KEYS)
        for layer_id in range(self.config.num_layers):
            keys += [block_key(layer_id, s) for s in BLOCK_KEYS]
        # only those actually present (biases may be absent)
        return [k for k in keys if k in self.device_weights]

    def _prepare_kernel(self):
        t = time.time()
        print("[pixart] compiling dit_forward")
        cfg = self.config
        N = 2 * cfg.max_batch_size  # classifier-free guidance
        H = cfg.sample_size

        latent = DeviceTensor.from_numpy(
            np.empty((N, cfg.in_channels, H, H), dtype=cfg.dtype), "latent"
        )
        timestep = DeviceTensor.from_numpy(np.empty((N,), dtype=np.float32), "timestep")
        caption = DeviceTensor.from_numpy(
            np.empty((N, cfg.max_text_tokens, cfg.caption_channels), dtype=cfg.dtype),
            "caption",
        )
        caption_mask = DeviceTensor.from_numpy(
            np.empty((N, cfg.max_text_tokens), dtype=cfg.dtype), "caption_mask"
        )

        weight_kwargs = {k: self.device_weights[k] for k in self._all_weight_keys()}

        self.kernel = DeviceKernel.compile_and_load(
            dit_forward,
            name="dit_forward",
            latent=latent,
            timestep=timestep,
            caption=caption,
            caption_mask=caption_mask,
            configs=cfg,
            build_dir=BUILD_DIR,
            additional_compiler_args=cfg.additional_compiler_args_nkipy,
            **weight_kwargs,
        )
        self._noise_pred = DeviceTensor.from_numpy(
            np.empty((N, cfg.in_channels, H, H), dtype=cfg.dtype), "noise_pred"
        )

        # ── fused sampling-step kernel (CFG + DPM update on device) ──
        B = cfg.max_batch_size
        latents1 = DeviceTensor.from_numpy(
            np.empty((B, cfg.in_channels, H, H), dtype=cfg.dtype), "latents"
        )
        prev_x0 = DeviceTensor.from_numpy(
            np.empty((B, cfg.in_channels, H, H), dtype=cfg.dtype), "prev_x0"
        )
        coeffs = DeviceTensor.from_numpy(np.empty((8,), dtype=np.float32), "coeffs")
        self.step_kernel = DeviceKernel.compile_and_load(
            denoise_step,
            name="denoise_step",
            latents=latents1,
            timestep=timestep,
            caption=caption,
            caption_mask=caption_mask,
            prev_x0=prev_x0,
            coeffs=coeffs,
            configs=cfg,
            build_dir=BUILD_DIR,
            additional_compiler_args=cfg.additional_compiler_args_nkipy,
            **weight_kwargs,
        )
        print(f"[pixart] --> kernel ready in {time.time() - t:.2f}s")

    def predict_noise(self, latent, timestep, caption, caption_mask):
        """One denoiser forward on a CFG batch.

        Args:
            latent: torch (N, C, H, W)
            timestep: torch (N,)
            caption: torch (N, Ltext, caption_channels)
            caption_mask: torch (N, Ltext) 1 for real tokens, 0 for T5 padding
        Returns:
            torch (N, C, H, W) predicted noise
        """
        inputs = {
            "latent": DeviceTensor.from_numpy(_to_numpy(latent, bfloat16), "latent"),
            "timestep": DeviceTensor.from_numpy(
                _to_numpy(timestep, np.float32), "timestep"
            ),
            "caption": DeviceTensor.from_numpy(_to_numpy(caption, bfloat16), "caption"),
            "caption_mask": DeviceTensor.from_numpy(
                _to_numpy(caption_mask, bfloat16), "caption_mask"
            ),
        }
        inputs.update({k: self.device_weights[k] for k in self._all_weight_keys()})
        self.kernel(inputs=inputs, outputs={"output0": self._noise_pred})
        return self._noise_pred.torch().to(torch.float32)

    def sample(self, init_latents, caption, caption_mask, step_coeffs):
        """Run the full denoising loop with the latent resident on device.

        Args:
            init_latents: torch (B, C, H, W) initial noise (already scaled).
            caption: torch (2B, Ltext, cc) CFG caption embeddings.
            caption_mask: torch (2B, Ltext).
            step_coeffs: list of per-step (8,) numpy coefficient arrays
                (see ``dpm_coeffs``). One entry per inference step.
        Returns:
            torch (B, C, H, W) final latent.
        """
        cfg = self.config
        B, C, H, _ = init_latents.shape

        # resident device state
        latents = DeviceTensor.from_numpy(_to_numpy(init_latents, bfloat16), "latents")
        prev_x0 = DeviceTensor.from_numpy(
            np.zeros((B, C, H, H), dtype=bfloat16), "prev_x0"
        )
        next_latents = DeviceTensor.from_numpy(
            np.empty((B, C, H, H), dtype=bfloat16), "next_latents"
        )
        next_x0 = DeviceTensor.from_numpy(
            np.empty((B, C, H, H), dtype=bfloat16), "next_x0"
        )

        cap = DeviceTensor.from_numpy(_to_numpy(caption, bfloat16), "caption")
        cap_mask = DeviceTensor.from_numpy(_to_numpy(caption_mask, bfloat16), "caption_mask")
        weight_inputs = {k: self.device_weights[k] for k in self._all_weight_keys()}

        for coeffs in step_coeffs:
            t_dev = DeviceTensor.from_numpy(
                np.full((2 * B,), coeffs[8], dtype=np.float32), "timestep"
            )
            coeffs_dev = DeviceTensor.from_numpy(
                np.asarray(coeffs[:8], dtype=np.float32), "coeffs"
            )
            inputs = {
                "latents": latents,
                "timestep": t_dev,
                "caption": cap,
                "caption_mask": cap_mask,
                "prev_x0": prev_x0,
                "coeffs": coeffs_dev,
            }
            inputs.update(weight_inputs)
            self.step_kernel(
                inputs=inputs,
                outputs={"output0": next_latents, "output1": next_x0},
            )
            # swap buffers: next -> current (no host round-trip of latent data)
            latents, next_latents = next_latents, latents
            prev_x0, next_x0 = next_x0, prev_x0

        return latents.torch().to(torch.float32)


# ── host-side pipeline pieces (T5 encoder + scheduler + VAE) ────────────────


def encode_prompt(prompt, negative_prompt, model_name, max_tokens, device="cpu"):
    """Run the T5 text encoder on the host, returning (cond, uncond) embeddings."""
    from transformers import T5EncoderModel, T5Tokenizer

    tok = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer")
    enc = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=torch.float32
    ).to(device)
    enc.eval()

    def embed(text):
        batch = tok(
            text, padding="max_length", max_length=max_tokens,
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = enc(batch.input_ids, attention_mask=batch.attention_mask)[0]
        return out, batch.attention_mask

    cond, cond_mask = embed(prompt)
    uncond, uncond_mask = embed(negative_prompt)
    return (cond, cond_mask), (uncond, uncond_mask)


def encode_prompt_device(prompt, negative_prompt, model_name, config, checkpoint):
    """Encode prompts with the T5 encoder running on Trainium.

    Loads prepared T5 weights + the host tokenizer/embedding table, compiles the
    device encoder once, and returns the same ((cond, mask), (uncond, mask))
    structure as ``encode_prompt``.
    """
    from t5_device import T5Encoder
    from transformers import T5EncoderModel, T5Tokenizer

    tok = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer")
    # host holds only the shared embedding table (for the token lookup)
    enc = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=torch.float32
    )
    embed_weight = enc.shared.weight.detach()
    del enc

    t5_weights = load_file(os.path.join(checkpoint, "t5_weights.safetensors"))
    encoder = T5Encoder(
        t5_weights, config, config.max_text_tokens, tok, embed_weight
    )
    cond, cond_mask = encoder.encode(prompt)
    uncond, uncond_mask = encoder.encode(negative_prompt)
    return (cond, cond_mask), (uncond, uncond_mask)


def dpm_coeffs(scheduler, guidance_scale):
    """Precompute per-step DPM-Solver++ (midpoint, epsilon) scalar coefficients.

    Mirrors ``DPMSolverMultistepScheduler`` for algorithm_type='dpmsolver++',
    prediction_type='epsilon', solver_type='midpoint'. Everything the device
    step needs is a scalar function of the (fixed) sigma schedule, so we compute
    it once on host. Returns a list of 9-float arrays, one per step:
        [0]=guidance_scale [1]=alpha_s [2]=sigma_ts [3]=c_sample
        [4]=c_D0 [5]=c_D1 [6]=r0 [7]=second_order_flag [8]=timestep
    """
    import numpy as _np

    sigmas = scheduler.sigmas.numpy().astype(_np.float64)  # (num_steps+1,)
    timesteps = scheduler.timesteps.numpy().astype(_np.float64)
    n = len(timesteps)

    def alpha_sigma(sigma):
        alpha_t = 1.0 / _np.sqrt(sigma * sigma + 1.0)
        return alpha_t, sigma * alpha_t

    out = []
    for i in range(n):
        sig_s0, sig_t = sigmas[i], sigmas[i + 1]
        alpha_s0, sigma_s0 = alpha_sigma(sig_s0)
        alpha_t, sigma_t = alpha_sigma(sig_t)
        # convert eps->x0 uses the *current* sigma (sig_s0)
        conv_alpha, conv_sigma = alpha_s0, sigma_s0

        # order selection: step 0 is first-order; last step is first-order when
        # final_sigmas_type='zero'; middle steps are second-order midpoint.
        lower_order_final = (i == n - 1)  # final_sigmas_type == "zero"
        first_order = (i == 0) or lower_order_final

        if lower_order_final:
            # final step to sigma_t=0: DDIM-equivalent update reduces to x0.
            # (sigma_t/sigma_s0 -> 0, alpha_t*(exp(-h)-1) -> -alpha_t*... = -1)
            out.append(_np.array([
                guidance_scale, alpha_s0, sigma_s0, 0.0, 1.0, 0.0, 1.0, 0.0,
                timesteps[i],
            ], dtype=_np.float64))
            continue

        lambda_t = _np.log(alpha_t) - _np.log(sigma_t)
        lambda_s0 = _np.log(alpha_s0) - _np.log(sigma_s0)
        h = lambda_t - lambda_s0
        em1 = _np.exp(-h) - 1.0

        c_sample = sigma_t / sigma_s0
        c_D0 = -(alpha_t * em1)
        if first_order:
            r0, c_D1, second = 1.0, 0.0, 0.0
        else:
            sig_s1 = sigmas[i - 1]
            alpha_s1, sigma_s1 = alpha_sigma(sig_s1)
            lambda_s1 = _np.log(alpha_s1) - _np.log(sigma_s1)
            h_0 = lambda_s0 - lambda_s1
            r0 = h_0 / h
            c_D1 = -0.5 * (alpha_t * em1)  # midpoint
            second = 1.0

        out.append(_np.array([
            guidance_scale, conv_alpha, conv_sigma, c_sample,
            c_D0, c_D1, r0, second, timesteps[i],
        ], dtype=_np.float64))
    return out


def run_pipeline(model, cond, uncond, config, guidance_scale=4.5, seed=0):
    """Full denoising loop with CFG + DPM step fused on device.

    The latent stays resident on device across all steps; the host only feeds
    the per-step scalar coefficients (precomputed from the fixed sigma
    schedule). ``cond``/``uncond`` are ``(embedding, attention_mask)`` tuples.
    """
    from diffusers import DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        config.model_name, subfolder="scheduler"
    )
    scheduler.set_timesteps(config.num_inference_steps)

    C, H = config.in_channels, config.sample_size
    gen = torch.Generator().manual_seed(seed)
    latents = torch.randn((config.max_batch_size, C, H, H), generator=gen)
    latents = latents * scheduler.init_noise_sigma

    cond_emb, cond_mask = cond
    uncond_emb, uncond_mask = uncond
    caption = torch.cat([uncond_emb, cond_emb], dim=0)  # (2B, Ltext, cc)
    caption_mask = torch.cat([uncond_mask, cond_mask], dim=0)  # (2B, Ltext)

    step_coeffs = dpm_coeffs(scheduler, guidance_scale)
    return model.sample(latents, caption, caption_mask, step_coeffs)


def _to_uint8(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    return (image.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)


def decode_latents(latents, model_name, device="cpu"):
    """VAE-decode latents to a PIL image on the host."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents.to(device)).sample
    return _to_uint8(image)


def decode_latents_device(latents, config, checkpoint):
    """VAE-decode latents on Trainium.

    Works up to 512px output (latent grid 64); the full 1024px decode exceeds
    the Neuron compiler's per-graph instruction limit (conv at that scale), so
    use the host path for 1024px.
    """
    from vae_device import VAEDecoder

    vae_weights = load_file(os.path.join(checkpoint, "vae_weights.safetensors"))
    decoder = VAEDecoder(vae_weights, config, latents.shape[-1])
    image = decoder.decode(latents)
    return _to_uint8(image)


def load_model(args):
    config = get_config(args.model, args.steps, args.sample_size)
    config.model_name = args.model
    print("[pixart] loading weights")
    weights = load_file(os.path.join(args.checkpoint, "weights.safetensors"))
    model = PixArtModel(weights, config)
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?",
                        default="a photo of a cat wearing a spacesuit")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--model", default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--checkpoint", default="./tmp_pixart_sigma")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="output.png")
    parser.add_argument("--t5-on-device", action="store_true",
                        help="Run the T5 encoder on Trainium (needs prepared t5_weights)")
    parser.add_argument("--vae-on-device", action="store_true",
                        help="Run the VAE decoder on Trainium (<=512px; needs vae_weights)")
    args = parser.parse_args()

    model, config = load_model(args)

    if args.t5_on_device:
        print("[pixart] encoding prompt (device T5)")
        cond, uncond = encode_prompt_device(
            args.prompt, args.negative_prompt, args.model, config, args.checkpoint
        )
    else:
        print("[pixart] encoding prompt (host T5)")
        cond, uncond = encode_prompt(
            args.prompt, args.negative_prompt, args.model, config.max_text_tokens
        )

    print("[pixart] denoising")
    start = time.time()
    latents = run_pipeline(
        model, cond, uncond, config, args.guidance_scale, args.seed
    )
    print(f"[pixart] --> {config.num_inference_steps} steps in {time.time() - start:.2f}s")

    if args.vae_on_device:
        print("[pixart] VAE decode (device)")
        images = decode_latents_device(latents, config, args.checkpoint)
    else:
        print("[pixart] VAE decode (host)")
        images = decode_latents(latents, args.model)

    from PIL import Image

    Image.fromarray(images[0]).save(args.output)
    print(f"[pixart] saved {args.output}")


if __name__ == "__main__":
    main()
