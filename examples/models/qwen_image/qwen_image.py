"""Qwen-Image text-to-image on Trainium — driver / entry point.

Runs the full pipeline on device (denoiser + Qwen2.5 text encoder + VAE); the
host keeps only the tokenizer + embedding lookup, latent pack/denorm, and the
scalar flow-match schedule. The host-side glue here (packing, img_shapes,
true-CFG, flow-match step) mirrors ``diffusers.QwenImagePipeline`` exactly. See
README.md for the architecture and the device/host split.

Tensor parallelism is required: 20B bf16 (~40 GB) does not fit on one 24 GB trn2
core, so launch with torchrun. TP=4 is the validated layout; TP is capped at
num_kv_heads=4 by the text encoder.
"""

import argparse
import os
import time

import numpy as np
import torch
from config import Config, get_config
from kernels.text_encoder import text_encoder_forward
from kernels.transformer import denoise_step
from kernels.vae import vae_decode
from kernels.weight_layout import BLOCK_KEYS, SHARED_KEYS, block_key


def _unpack_latents(latents, height, width, vae_scale_factor):
    """Packed tokens -> (B, C, 1, H, W) latent. Matches the pipeline (5-D for the
    3D Qwen VAE)."""
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, 1, height, width)


class _DeviceModule:
    """Shared plumbing for the on-device components (denoiser / text encoder /
    VAE): upload the pre-sharded weights once, compile a kernel per input shape
    (cached), and run it.

    The tracer prunes weights that don't reach the output, so every compiled
    kernel is invoked with only the subset in ``kernel.input_tensors_info``
    (``_compile`` returns that subset as ``wkeys``).
    """

    def __init__(self, weights, config, fp32=False):
        from nkipy.runtime import DeviceTensor
        from nkipy.runtime.device_tensor import bfloat16

        self._DeviceTensor = DeviceTensor
        self._bf16 = bfloat16
        self.config = config
        self.device_weights = self._upload(weights, fp32)
        self._weight_keys = self._select_weight_keys()
        self._kernels = {}

    def _upload(self, weights, fp32):
        """torch/numpy weights -> ``{key: DeviceTensor}`` (bf16, or fp32 for the
        numerically-sensitive VAE)."""
        DT = self._DeviceTensor
        out = {}
        for key, tensor in weights.items():
            if fp32:
                arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
                out[key] = DT.from_numpy(np.ascontiguousarray(arr.astype(np.float32)), name=key)
            else:
                src = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                out[key] = DT.from_torch(src, name=key)
        return out

    def _select_weight_keys(self):
        """Weights to feed the kernel, before tracer pruning. Default: all of
        them; the denoiser / text encoder narrow this to their key scheme."""
        return list(self.device_weights)

    def _compile(self, key, kernel_fn, name, placeholders, build_dir):
        """Compile once (cached by ``key``) and return ``(kernel, wkeys)``, where
        ``wkeys`` is the weight subset the traced graph actually reads."""
        if key in self._kernels:
            return self._kernels[key]
        from nkipy.runtime import DeviceKernel

        weight_kwargs = {k: self.device_weights[k] for k in self._weight_keys}
        kernel = DeviceKernel.compile_and_load(
            kernel_fn, name=name, configs=self.config, build_dir=build_dir,
            additional_compiler_args=self.config.additional_compiler_args_nkipy,
            **placeholders, **weight_kwargs,
        )
        wkeys = [k for k in self._weight_keys if k in set(kernel.input_tensors_info)]
        self._kernels[key] = (kernel, wkeys)
        return kernel, wkeys

    def _run(self, kernel, wkeys, host_inputs, out):
        """Assemble host inputs + the live weight subset, run into ``out``."""
        inputs = dict(host_inputs)
        inputs.update({k: self.device_weights[k] for k in wkeys})
        kernel(inputs=inputs, outputs={"output0": out})
        return out


class QwenImageDenoiser(_DeviceModule):
    """The Trainium MMDiT denoiser (host encoder / VAE live in the driver).

    Compiles the fused ``denoise_step`` once per text length; ``sample`` runs the
    full denoising loop with the packed latent resident on device (CFG +
    FlowMatchEuler on device, no per-step host round-trip).
    """

    def __init__(self, weights, config: Config, img_shape, batch_size=1):
        self.img_shape = img_shape  # (frame, gh, gw)
        self.batch_size = batch_size
        t = time.time()
        super().__init__(weights, config)
        print(f"[qwen-image] denoiser weights ready in {time.time() - t:.2f}s "
              f"(TP={getattr(config, 'tp_size', 1)})")

    def _select_weight_keys(self):
        keys = [k for k in SHARED_KEYS if k in self.device_weights]
        for i in range(self.config.num_layers):
            keys += [block_key(i, s) for s in BLOCK_KEYS
                     if block_key(i, s) in self.device_weights]
        return keys

    def _step_kernel_for(self, txt_len):
        """Compile (once) the fused ``denoise_step`` kernel for a text length.

        cond and uncond prompts can have different token counts, so we compile
        per text length on demand (usually 1-2 distinct lengths).
        """
        if txt_len in self._kernels:
            return self._kernels[txt_len]
        cfg = self.config
        frame, gh, gw = self.img_shape
        Limg = frame * gh * gw
        B = self.batch_size
        DT, bf16 = self._DeviceTensor, self._bf16

        placeholders = dict(
            latents=DT.from_numpy(np.empty((B, Limg, cfg.in_channels), dtype=bf16), "latents"),
            cond_text=DT.from_numpy(np.empty((B, txt_len, cfg.joint_attention_dim), dtype=bf16), "cond_text"),
            neg_text=DT.from_numpy(np.empty((B, txt_len, cfg.joint_attention_dim), dtype=bf16), "neg_text"),
            timestep=DT.from_numpy(np.empty((B,), dtype=np.float32), "timestep"),
            coeffs=DT.from_numpy(np.empty((2,), dtype=np.float32), "coeffs"),
            img_shape=self.img_shape,
            cond_mask=DT.from_numpy(np.empty((B, txt_len), dtype=bf16), "cond_mask"),
            neg_mask=DT.from_numpy(np.empty((B, txt_len), dtype=bf16), "neg_mask"),
        )
        print(f"[qwen-image] compiling denoise_step "
              f"({cfg.num_layers} blocks, grid {gh}x{gw}, txt {txt_len})")
        t = time.time()
        kernel, wkeys = self._compile(txt_len, denoise_step,
                                      f"denoise_step_t{txt_len}", placeholders, "./build")
        print(f"[qwen-image] --> step kernel (txt {txt_len}) ready in {time.time() - t:.2f}s")
        return kernel, wkeys

    def sample(self, init_latents, cond_text, neg_text, cond_mask, neg_mask,
               step_coeffs):
        """Full denoising loop with the latent resident on device (fused step).

        Runs ``denoise_step`` once per step; the packed latent stays on device
        across all steps (no per-step host round-trip). Text embeddings/masks are
        uploaded once. ``cond_text``/``neg_text`` must share a text length (pad +
        mask on the host). ``step_coeffs`` is a list of per-step ``[cfg_scale, dt]``.

        All tensors are torch; returns the final packed latent (torch, B×Limg×C).
        """
        DT, bf16 = self._DeviceTensor, self._bf16

        def to_np(t, dt):
            return t.detach().to(torch.float32).cpu().numpy().astype(dt)

        txt_len = cond_text.shape[1]
        kernel, wkeys = self._step_kernel_for(txt_len)

        B, Limg, C = init_latents.shape
        latents = DT.from_numpy(to_np(init_latents, bf16), "latents")
        nxt = DT.from_numpy(np.empty((B, Limg, C), dtype=bf16), "latents_next")
        cond_d = DT.from_numpy(to_np(cond_text, bf16), "cond_text")
        neg_d = DT.from_numpy(to_np(neg_text, bf16), "neg_text")
        cmask_d = DT.from_numpy(to_np(cond_mask, bf16), "cond_mask")
        nmask_d = DT.from_numpy(to_np(neg_mask, bf16), "neg_mask")

        for cfg_scale, dt, ts_val in step_coeffs:
            ts_d = DT.from_numpy(np.full((B,), ts_val, dtype=np.float32), "timestep")
            coeffs_d = DT.from_numpy(np.array([cfg_scale, dt], dtype=np.float32), "coeffs")
            host_inputs = {
                "latents": latents, "cond_text": cond_d, "neg_text": neg_d,
                "timestep": ts_d, "coeffs": coeffs_d,
                "cond_mask": cmask_d, "neg_mask": nmask_d,
            }
            self._run(kernel, wkeys, host_inputs, nxt)
            latents, nxt = nxt, latents  # swap; no host round-trip of latent data

        return latents.torch().to(torch.float32)


class DeviceTextEncoder(_DeviceModule):
    """The Trainium Qwen2.5 text encoder (``kernels/text_encoder.py``).

    Runs the text-only Qwen2.5-VL decoder LM on device and returns the last
    hidden state. The host keeps the tokenizer, chat template, and embedding-
    table lookup (huge, data-dependent gather); the device runs the 28-layer
    transformer, TP-sharded like the denoiser (the 7B encoder doesn't fit one
    core). Compiles ``text_encoder_forward`` once per sequence length.
    """

    def __init__(self, weights, te_config):
        t = time.time()
        super().__init__(weights, te_config)
        print(f"[qwen-image] text-encoder weights ready in {time.time() - t:.2f}s "
              f"(TP={getattr(te_config, 'tp_size', 1)})")

    def _select_weight_keys(self):
        from kernels.text_weight_layout import present_keys
        return present_keys(self.device_weights, self.config.num_layers)

    def _kernel_for(self, seq_len, hidden):
        if seq_len in self._kernels:
            return self._kernels[seq_len]
        DT, bf16, cfg = self._DeviceTensor, self._bf16, self.config
        placeholders = {"hidden": DT.from_numpy(np.empty((1, seq_len, hidden), dtype=bf16), "hidden")}
        print(f"[qwen-image] compiling text_encoder_forward "
              f"({cfg.num_layers} layers, seq {seq_len})")
        t = time.time()
        kernel, wkeys = self._compile(seq_len, text_encoder_forward,
                                      f"text_encoder_forward_s{seq_len}", placeholders, "./build_te")
        print(f"[qwen-image] --> text encoder (seq {seq_len}) ready in {time.time() - t:.2f}s")
        return kernel, wkeys

    def encode(self, embeds):
        """Run the encoder. ``embeds``: (1, S, hidden) torch token embeddings.

        Returns the last hidden state as a torch tensor (1, S, hidden).
        """
        B, S, H = embeds.shape
        assert B == 1, "device text encoder processes one prompt at a time"
        kernel, wkeys = self._kernel_for(S, H)
        DT = self._DeviceTensor
        hidden_np = embeds.detach().to(torch.float32).cpu().numpy().astype(self._bf16)
        host_inputs = {"hidden": DT.from_numpy(hidden_np, "hidden")}
        out = DT.from_numpy(np.empty((1, S, H), dtype=self._bf16), "out")
        self._run(kernel, wkeys, host_inputs, out)
        return out.torch().to(torch.float32)


def encode_prompt_device(pipe, device_encoder, prompt):
    """Replicate ``QwenImagePipeline._get_qwen_prompt_embeds`` with the encoder
    forward on device.

    The host does everything data-dependent (chat-template wrapping, tokenize,
    embedding lookup, masked-hidden extraction, ``drop_idx`` prefix slice); the
    device runs only the transformer. Batch=1 (cond and neg are encoded
    separately), so the tokenizer emits no padding and the encoder's causal-only
    mask matches diffusers exactly. Returns ``(prompt_embeds, prompt_embeds_mask)``
    in the same layout as ``encode_prompt``.
    """
    template = pipe.prompt_template_encode
    drop_idx = pipe.prompt_template_encode_start_idx
    txt = [template.format(prompt)]
    txt_tokens = pipe.tokenizer(
        txt, max_length=pipe.tokenizer_max_length + drop_idx,
        padding=True, truncation=True, return_tensors="pt")
    input_ids = txt_tokens.input_ids
    attn_mask = txt_tokens.attention_mask

    # host embedding lookup (the table stays on host)
    with torch.no_grad():
        embeds = pipe.text_encoder.get_input_embeddings()(input_ids)
    # device transformer -> last hidden state
    hidden = device_encoder.encode(embeds.to(torch.float32))

    # masked-hidden extraction + drop template prefix (matches diffusers)
    bool_mask = attn_mask.bool()
    valid = bool_mask.sum(dim=1).tolist()
    selected = hidden[bool_mask]
    split = torch.split(selected, valid, dim=0)
    split = [e[drop_idx:] for e in split]
    max_len = max(e.size(0) for e in split)
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_len - u.size(0), u.size(1))]) for u in split])
    mask = torch.stack(
        [torch.cat([torch.ones(u.size(0)), torch.zeros(max_len - u.size(0))]) for u in split])
    if mask.all():
        mask = None
    return prompt_embeds, mask


class DeviceVAEDecoder(_DeviceModule):
    """The Trainium VAE decoder (``kernels/vae.py``).

    Decodes a single image (T=1), where ``AutoencoderKLQwenImage``'s 3D causal
    video VAE collapses exactly to a 2D conv decoder. The host squeezes the
    temporal dim and does the latent denormalization; the device runs the conv
    stack. Runs in fp32 (numerically sensitive, one-shot). Compiles per latent
    spatial size. Replicated per rank (small, not sharded).
    """

    def __init__(self, weights, vae_config):
        super().__init__(weights, vae_config, fp32=True)

    def _kernel_for(self, h, w):
        if (h, w) in self._kernels:
            return self._kernels[(h, w)]
        DT, cfg = self._DeviceTensor, self.config
        placeholders = {"latents": DT.from_numpy(np.empty((1, cfg.z_dim, h, w), dtype=np.float32), "latents")}
        print(f"[qwen-image] compiling vae_decode (latent {h}x{w})")
        t = time.time()
        kernel, wkeys = self._compile((h, w), vae_decode,
                                      f"vae_decode_{h}x{w}", placeholders, "./build_vae")
        print(f"[qwen-image] --> vae kernel ({h}x{w}) ready in {time.time() - t:.2f}s")
        return kernel, wkeys

    def decode(self, latents):
        """Args: denormalized latents torch (1, z_dim, H, W). Returns pixels
        torch (1, 3, H*8, W*8) in [-1, 1] (unclamped; caller clamps)."""
        _, _, h, w = latents.shape
        kernel, wkeys = self._kernel_for(h, w)
        DT, cfg = self._DeviceTensor, self.config
        lat_np = latents.detach().to(torch.float32).cpu().numpy().astype(np.float32)
        host_inputs = {"latents": DT.from_numpy(lat_np, "latents")}
        scale = 2 ** (len(cfg.dim_mult) - 1)
        out = DT.from_numpy(
            np.empty((1, cfg.out_channels, h * scale, w * scale), dtype=np.float32), "pixels")
        self._run(kernel, wkeys, host_inputs, out)
        return out.torch().to(torch.float32)


def load_host_pipeline(model_name, dtype=torch.bfloat16):
    """Load the diffusers QwenImagePipeline (host encoder + VAE + scheduler)."""
    from diffusers import QwenImagePipeline

    return QwenImagePipeline.from_pretrained(model_name, torch_dtype=dtype)


def _pad_text(embeds, mask, target_len):
    """Right-pad (B, L, D) embeds to ``target_len`` and return a matching mask.

    ``encode_prompt`` returns ``mask=None`` when every token is valid; we
    materialise an all-ones mask so the padded tail can be masked to 0 (the
    fused denoise_step always takes explicit masks).
    """
    B, L, D = embeds.shape
    if mask is None:
        mask = embeds.new_ones(B, L)
    if L == target_len:
        return embeds, mask
    pad = target_len - L
    embeds = torch.cat([embeds, embeds.new_zeros(B, pad, D)], dim=1)
    mask = torch.cat([mask, mask.new_zeros(B, pad)], dim=1)
    return embeds, mask


def generate(pipe, denoiser, prompt, negative_prompt, config, height, width,
             guidance_scale, num_steps, text_encoder, vae_decoder, seed=0):
    """Full pipeline: text-encode, prepare latents + flow-match schedule, run the
    device-resident sampling loop, then unpack + VAE-decode to pixels.

    Requires true-CFG (``guidance_scale > 1.0``) — Qwen-Image always uses it, and
    the fused device step is built around the cond/neg pair.
    """
    if guidance_scale <= 1.0:
        raise ValueError("qwen-image requires true-CFG (guidance_scale > 1.0)")
    device = "cpu"
    vae_scale = pipe.vae_scale_factor

    # text encode on device
    prompt_embeds, prompt_mask = encode_prompt_device(pipe, text_encoder, prompt)
    neg_embeds, neg_mask = encode_prompt_device(pipe, text_encoder, negative_prompt)

    # init latents (packed) + flow-match schedule
    num_ch = pipe.transformer.config.in_channels // 4
    gen = torch.Generator().manual_seed(seed)
    latents = pipe.prepare_latents(
        1, num_ch, height, width, prompt_embeds.dtype, device, gen)

    # Qwen-Image's scheduler uses dynamic shifting: mu is derived from the image
    # token count (matches QwenImagePipeline.__call__).
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift

    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
    timesteps = pipe.scheduler.timesteps

    # device-resident loop: CFG + FlowMatchEuler run on device in denoise_step.
    # cond/neg share a text length (pad + mask); dt comes from the scheduler
    # sigmas (prev = sample + dt * model_output).
    tlen = max(prompt_embeds.shape[1], neg_embeds.shape[1])
    c_emb, c_mask = _pad_text(prompt_embeds, prompt_mask, tlen)
    n_emb, n_mask = _pad_text(neg_embeds, neg_mask, tlen)
    sched_sigmas = pipe.scheduler.sigmas  # (num_steps+1,)
    step_coeffs = []
    for i, t in enumerate(timesteps):
        dt = float(sched_sigmas[i + 1] - sched_sigmas[i])
        step_coeffs.append((float(guidance_scale), dt, float(t) / 1000.0))
    latents = denoiser.sample(latents, c_emb, n_emb, c_mask, n_mask, step_coeffs)

    # unpack + VAE decode on device. Denormalize exactly like QwenImagePipeline:
    # the pipeline stores latents_std as its RECIPROCAL and then does
    # ``latents / latents_std_recip`` (== latents * std) + mean. Matching that.
    latents = _unpack_latents(latents, height, width, vae_scale)  # (1, z, 1, H, W)
    latents = latents.to(pipe.vae.dtype)
    z_dim = pipe.vae.config.z_dim
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents)
    latents_std_recip = (1.0 / torch.tensor(pipe.vae.config.latents_std)).view(1, z_dim, 1, 1, 1).to(latents)
    latents = latents / latents_std_recip + latents_mean
    # device VAE: squeeze the single frame -> (1, z, H, W), decode on device
    image = vae_decoder.decode(latents[:, :, 0])
    image = (image / 2 + 0.5).clamp(0, 1)
    return (image.permute(0, 2, 3, 1).cpu().float().numpy() * 255).round().astype(np.uint8)


def _build_device_denoiser_weights(pipe, config, tp_size, rank):
    """Extract the denoiser's flat weights from the host pipeline's transformer
    and slice this rank's tensor-parallel shard (mirrors the text-encoder / VAE
    paths — no on-disk pre-bake step)."""
    from weight_extract import extract_flat_weights, shard_flat_weights

    flat = extract_flat_weights(pipe.transformer, config.num_layers, dtype=np.float32)
    shard = shard_flat_weights(flat, rank, tp_size, config.num_layers,
                               config.num_heads, config.head_dim)
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in shard.items()}


def _build_device_text_encoder(pipe, tp_size, rank, log):
    """Extract the Qwen2.5 encoder weights from the host pipeline, shard for this
    rank, and build a ``DeviceTextEncoder``. The host pipeline's encoder stays
    resident for the embedding-table lookup (the transformer layers now run on
    device, but the embedding gather remains on host)."""
    from config import TextEncoderConfig
    from weight_extract import extract_text_encoder_weights, shard_text_encoder_weights

    te_hf = pipe.text_encoder.config.text_config
    # rope_theta is top-level on older transformers, nested under
    # ``rope_parameters``/``rope_scaling`` on newer ones (e.g. 5.5.x).
    rope_theta = getattr(te_hf, "rope_theta", None)
    if rope_theta is None:
        params = getattr(te_hf, "rope_parameters", None) or getattr(te_hf, "rope_scaling", None) or {}
        rope_theta = params.get("rope_theta", 1000000.0)
    te_cfg = TextEncoderConfig(
        num_layers=te_hf.num_hidden_layers,
        hidden_size=te_hf.hidden_size,
        num_heads=te_hf.num_attention_heads,
        num_kv_heads=te_hf.num_key_value_heads,
        head_dim=getattr(te_hf, "head_dim", te_hf.hidden_size // te_hf.num_attention_heads),
        intermediate_size=te_hf.intermediate_size,
        rms_norm_eps=te_hf.rms_norm_eps,
        rope_theta=rope_theta,
    )
    from kernels.tp import make_all_reduce
    te_cfg.tp_size = tp_size
    te_cfg.all_reduce_fn = make_all_reduce(tp_size)

    log(f"[qwen-image] extracting text-encoder weights ({te_cfg.num_layers} layers)")
    lm = pipe.text_encoder.model.language_model
    flat = extract_text_encoder_weights(lm, te_cfg.num_layers, dtype=np.float32)
    flat = shard_text_encoder_weights(
        flat, rank, tp_size, te_cfg.num_layers,
        te_cfg.num_heads, te_cfg.num_kv_heads, te_cfg.head_dim)
    weights = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in flat.items()}
    return DeviceTextEncoder(weights, te_cfg)


def _build_device_vae(pipe, model_name, log):
    """Extract the VAE decoder weights from the host pipeline and build a
    ``DeviceVAEDecoder`` (T=1 2D-collapsed decoder, fp32, replicated per rank)."""
    from config import get_vae_config
    from weight_extract import extract_vae_decoder_weights

    vae_cfg = get_vae_config(model_name)
    log(f"[qwen-image] extracting VAE decoder weights (z_dim {vae_cfg.z_dim})")
    flat = extract_vae_decoder_weights(pipe.vae, dtype=np.float32)
    weights = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in flat.items()}
    return DeviceVAEDecoder(weights, vae_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="a coffee shop entrance with a chalkboard sign")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--model", default="Qwen/Qwen-Image")
    # 512px (grid 32x32) is the validated working resolution; native 1328px
    # exceeds the compiler's 2 GB HLO-proto limit (see README.md).
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="output.png")
    args = parser.parse_args()

    # ── tensor-parallel setup (torchrun) ──────────────────────────────────────
    # NEURON_RT_DISABLE_EXECUTION_BARRIER must be 0 (set before nkipy import) for
    # correct multi-rank collectives.
    import torch.distributed as dist

    tp_size = int(os.environ.get("WORLD_SIZE", "1"))
    if tp_size <= 1:
        raise SystemExit(
            "qwen-image requires tensor parallelism: the 20B model (~40 GB bf16) "
            "does not fit on one 24 GB core. Launch with e.g. "
            "`torchrun --nproc-per-node 4 qwen_image.py ...` (TP=4 fits; TP is "
            "capped at num_kv_heads=4 by the text encoder).")
    os.environ["NEURON_RT_DISABLE_EXECUTION_BARRIER"] = "0"
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "localhost:61455")
    dist.init_process_group()
    rank = dist.get_rank()
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)
    torch.set_num_threads(1)

    def log(msg):
        if rank == 0:
            print(msg)

    from kernels.tp import make_all_reduce
    config = get_config(args.model, args.steps)
    config.model_name = args.model
    config.tp_size = tp_size
    config.all_reduce_fn = make_all_reduce(tp_size)

    log(f"[qwen-image] loading host pipeline {args.model} (TP={tp_size})")
    pipe = load_host_pipeline(args.model)
    vae_scale = pipe.vae_scale_factor
    gh, gw = args.height // vae_scale // 2, args.width // vae_scale // 2

    log("[qwen-image] extracting denoiser weights from host transformer")
    weights = _build_device_denoiser_weights(pipe, config, tp_size, rank)
    denoiser = QwenImageDenoiser(weights, config, (1, gh, gw), batch_size=1)

    text_encoder = _build_device_text_encoder(pipe, tp_size, rank, log)
    vae_decoder = _build_device_vae(pipe, args.model, log)

    dist.barrier()
    log("[qwen-image] generating")
    start = time.time()
    images = generate(pipe, denoiser, args.prompt, args.negative_prompt, config,
                      args.height, args.width, args.guidance_scale, args.steps,
                      text_encoder, vae_decoder, seed=args.seed)
    log(f"[qwen-image] --> {args.steps} steps in {time.time() - start:.2f}s")

    if rank == 0:
        from PIL import Image
        Image.fromarray(images[0]).save(args.output)
        print(f"[qwen-image] saved {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
