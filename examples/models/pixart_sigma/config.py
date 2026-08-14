from dataclasses import dataclass

import numpy as np
from neuronxcc.nki.language import bfloat16

# to control compiler_args
DTYPE = bfloat16


@dataclass
class Config:
    """PixArt-Sigma DiT (Transformer2DModel) configuration.

    Defaults match PixArt-alpha/PixArt-Sigma-XL-2-1024-MS. The denoiser is a
    stack of DiT blocks: self-attention (bidirectional) + cross-attention to the
    T5 text tokens + a gated feed-forward, all modulated by adaLN-single.
    """

    # DiT transformer
    hidden_size: int = 1152
    num_layers: int = 28
    num_heads: int = 16
    # head_dim derived: hidden_size // num_heads == 72
    intermediate_size: int = 4608  # 4 * hidden_size, GELU-approx MLP
    patch_size: int = 2

    # latent space (VAE-encoded image); PixArt uses the SD VAE (8x downsample, 4 ch)
    in_channels: int = 4
    out_channels: int = 8  # learn-sigma: predicts mean + variance

    # text conditioning (T5 encoder, frozen)
    caption_channels: int = 4096  # T5-XXL hidden size
    max_text_tokens: int = 300

    # T5-XXL encoder (used when running the encoder on device)
    t5_num_layers: int = 24
    t5_num_heads: int = 64
    t5_d_kv: int = 64
    t5_d_model: int = 4096
    t5_d_ff: int = 10240
    t5_eps: float = 1e-6
    t5_num_buckets: int = 32

    # VAE decoder (used when running the decoder on device)
    vae_norm_groups: int = 32
    vae_eps: float = 1e-6
    vae_layers_per_block: int = 2
    vae_block_out_channels: tuple = (128, 256, 512, 512)
    vae_scaling_factor: float = 0.13025

    # sampling
    sample_size: int = 128  # runtime latent grid side for 1024px (1024 / 8)
    # PatchEmbed positional-embedding scaling. ``native_sample_size`` is the
    # model's training resolution (a fixed model property); diffusers derives
    # base_size = native_sample_size // patch_size and holds it constant across
    # runtime resolutions. Keep these decoupled from ``sample_size``.
    native_sample_size: int = 128
    interpolation_scale: float = 2.0
    num_inference_steps: int = 20

    norm_eps: float = 1e-6
    max_batch_size: int = 1  # x2 internally for classifier-free guidance
    dtype: np.dtype = DTYPE
    additional_compiler_args_nkipy: str = "--lnc 1"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def get_config(model_name: str, num_inference_steps: int, sample_size: int) -> Config:
    """Build a Config from a diffusers PixArt-Sigma checkpoint.

    Reads the DiT (Transformer2DModel) config JSON. Falls back to the
    Sigma-XL defaults if diffusers is unavailable so the module stays importable
    on machines without the dependency.
    """
    try:
        from diffusers import PixArtTransformer2DModel

        hf = PixArtTransformer2DModel.load_config(model_name, subfolder="transformer")
        config = Config(
            hidden_size=hf["num_attention_heads"] * hf["attention_head_dim"],
            num_layers=hf["num_layers"],
            num_heads=hf["num_attention_heads"],
            intermediate_size=hf["num_attention_heads"] * hf["attention_head_dim"] * 4,
            patch_size=hf["patch_size"],
            in_channels=hf["in_channels"],
            out_channels=hf["out_channels"],
            caption_channels=hf["caption_channels"],
            interpolation_scale=hf.get("interpolation_scale", 2.0),
            native_sample_size=hf.get("sample_size", 128),
            num_inference_steps=num_inference_steps,
            sample_size=sample_size,
        )
    except Exception:
        config = Config(
            num_inference_steps=num_inference_steps,
            sample_size=sample_size,
        )
    return config
