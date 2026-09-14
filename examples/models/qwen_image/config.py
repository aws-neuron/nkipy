from dataclasses import dataclass, field

import numpy as np
from neuronxcc.nki.language import bfloat16

# to control compiler_args
DTYPE = bfloat16


@dataclass
class Config:
    """Qwen-Image (QwenImageTransformer2DModel) configuration.

    Defaults match Qwen/Qwen-Image. The denoiser is a dual-stream **MMDiT**: 60
    blocks, each running an image stream and a text stream through per-stream
    LayerNorm + adaLN-style modulation, QKV with QK-RMSNorm + 3D RoPE, a single
    joint attention over concat([text, image]), then per-stream gated MLPs.

    The text tokens participate in the same attention as the image tokens
    (joint attention over concat([text, image])), so there is no separate
    cross-attention.
    """

    # MMDiT transformer
    num_layers: int = 60
    num_heads: int = 24
    head_dim: int = 128
    # hidden/inner dim derived: num_heads * head_dim == 3072
    patch_size: int = 2

    # latent space (VAE-encoded image); Qwen-Image uses a 16-channel VAE, and
    # the DiT sees patchified latents so in_channels = 16 * patch_size**2 = 64.
    in_channels: int = 64
    out_channels: int = 16

    # text conditioning (Qwen2.5-VL encoder, frozen)
    joint_attention_dim: int = 3584  # Qwen2.5-VL hidden size (text stream input)
    pooled_projection_dim: int = 768

    # 3D RoPE axis dims (frame, height, width); sum == head_dim (16+56+56 == 128)
    axes_dims_rope: tuple = (16, 56, 56)
    rope_theta: float = 10000.0
    guidance_embeds: bool = False

    # sampling (FlowMatchEulerDiscreteScheduler, rectified flow)
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0

    norm_eps: float = 1e-6
    max_batch_size: int = 1  # x2 internally for classifier-free guidance
    dtype: np.dtype = DTYPE
    additional_compiler_args_nkipy: str = "--lnc 1"

    # tensor parallelism (required): shard attention heads + MLP intermediate
    # across tp_size cores (>=4 needed to fit the 20B weights on trn2's 24 GB/core).
    # ``all_reduce_fn`` must be set by the driver to a real collective — the
    # kernels always apply it after each row-parallel projection.
    tp_size: int = 1
    all_reduce_fn: object = None

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_dim


@dataclass
class TextEncoderConfig:
    """Qwen2.5 text-encoder config (the text-only LM inside Qwen2.5-VL).

    Defaults match Qwen-Image's text_encoder. Used by the on-device encoder
    (``kernels/text_encoder.py``); the host keeps the tokenizer, chat template,
    and embedding-table lookup.
    """

    num_layers: int = 28
    hidden_size: int = 3584
    num_heads: int = 28
    num_kv_heads: int = 4
    head_dim: int = 128  # 3584 / 28
    intermediate_size: int = 18944
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    vocab_size: int = 152064

    # chat-template wrapping (matches QwenImagePipeline)
    prompt_template: str = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, "
        "size, texture, quantity, text, spatial relationships of the objects and "
        "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    template_drop_idx: int = 34
    tokenizer_max_length: int = 1024

    dtype: np.dtype = DTYPE
    additional_compiler_args_nkipy: str = "--lnc 1"
    tp_size: int = 1
    all_reduce_fn: object = None


@dataclass
class VAEConfig:
    """AutoencoderKLQwenImage decoder config (the T=1 2D-collapsed decoder).

    Defaults match Qwen-Image's VAE. Used by the on-device VAE decoder
    (``kernels/vae.py``); the host keeps the latent denormalization and the
    single-frame squeeze/unsqueeze.
    """

    z_dim: int = 16
    base_dim: int = 96
    dim_mult: tuple = (1, 2, 4, 4)
    num_res_blocks: int = 2
    out_channels: int = 3

    dtype: np.dtype = np.float32  # VAE runs in fp32 (numerically sensitive)
    additional_compiler_args_nkipy: str = "--lnc 1"

    # kernel reads these short aliases
    @property
    def vae_dim_mult(self):
        return self.dim_mult

    @property
    def vae_num_res_blocks(self):
        return self.num_res_blocks


def get_vae_config(model_name: str) -> "VAEConfig":
    """Build a VAEConfig from a diffusers Qwen-Image checkpoint (falls back to
    Qwen-Image defaults if diffusers is unavailable)."""
    try:
        from diffusers import AutoencoderKLQwenImage

        hf = AutoencoderKLQwenImage.load_config(model_name, subfolder="vae")
        return VAEConfig(
            z_dim=hf["z_dim"],
            base_dim=hf["base_dim"],
            dim_mult=tuple(hf["dim_mult"]),
            num_res_blocks=hf["num_res_blocks"],
        )
    except Exception:
        return VAEConfig()


def get_config(model_name: str, num_inference_steps: int) -> Config:
    """Build a Config from a diffusers Qwen-Image checkpoint.

    Reads the DiT (QwenImageTransformer2DModel) config JSON. Falls back to the
    Qwen-Image defaults if diffusers is unavailable so the module stays
    importable on machines without the dependency.
    """
    try:
        from diffusers import QwenImageTransformer2DModel

        hf = QwenImageTransformer2DModel.load_config(model_name, subfolder="transformer")
        config = Config(
            num_layers=hf["num_layers"],
            num_heads=hf["num_attention_heads"],
            head_dim=hf["attention_head_dim"],
            patch_size=hf["patch_size"],
            in_channels=hf["in_channels"],
            out_channels=hf["out_channels"],
            joint_attention_dim=hf["joint_attention_dim"],
            pooled_projection_dim=hf.get("pooled_projection_dim", 768),
            axes_dims_rope=tuple(hf.get("axes_dims_rope", (16, 56, 56))),
            guidance_embeds=hf.get("guidance_embeds", False),
            num_inference_steps=num_inference_steps,
        )
    except Exception:
        config = Config(num_inference_steps=num_inference_steps)
    return config
