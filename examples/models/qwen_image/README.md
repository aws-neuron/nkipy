# Qwen-Image (20B MMDiT) on Trainium

NKIPy port of [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
(`QwenImagePipeline`, Apache-2.0) — a 20B dual-stream **MMDiT** text-to-image
model with 3D RoPE, QK-RMSNorm, per-stream modulation, a Qwen2.5-VL text
encoder, a 16-channel video-style VAE, and rectified-flow sampling.

The port is **device-only** end-to-end: the denoiser, text encoder, and VAE all
run on trn2 (TP=4, fused on-device sampling), producing correct 512px images.

## Run

```bash
bash demo.sh "a coffee shop storefront with a chalkboard sign reading 'Qwen Cafe'"
```

`demo.sh` reuses the compilation cache, runs the CPU correctness tests, then
generates one image via `torchrun`. Env knobs: `TP` `HEIGHT` `WIDTH` `STEPS`
`OUTPUT`; `CLEAN=1` forces a recompile; `SKIP_TESTS=1` skips the CPU tests.

Or drive it directly:

```bash
torchrun --nproc-per-node 4 qwen_image.py "your prompt" \
    --height 512 --width 512 --steps 50
```

**Tensor parallelism is required** — 20B bf16 (~40 GB) doesn't fit on one 24 GB
core. TP=4 fits (20.4 GB/core) and is the max: the text encoder caps TP at
`num_kv_heads = 4`. The model is downloaded once by the diffusers pipeline (HF
cache); the driver extracts + shards all weights in-memory (no repack step).

CPU correctness tests: `uv run pytest tests/`. The on-device TP check needs
hardware and is opt-in: `QWEN_IMAGE_TP_DEVICE_TEST=1 uv run pytest
tests/test_tp_device.py`.

## What runs on device vs host

Everything with real FLOPs runs on device; the host keeps only trivial glue.

- **Denoiser** — 60-block MMDiT, TP=4. The fused sampling step
  (`QwenImageDenoiser.sample` / `kernels/transformer.py:denoise_step`) runs CFG +
  FlowMatchEuler on device with the packed latent resident across all steps; the
  host feeds only the per-step scalars `[true_cfg_scale, dt]` and once-uploaded
  text embeds/masks.
- **Text encoder** (`kernels/text_encoder.py`) — a prefill Qwen2.5 decoder LM (28
  layers, hidden 3584, GQA 28/4, SwiGLU 18944, RoPE θ=1e6, qkv **with** bias, no
  QK-norm), returning the last hidden state. Reuses the qwen3 block kernels;
  sharded Megatron-style (q/o by query heads, k/v by KV heads, MLP by
  intermediate). Device rel_l2 6.7e-3 bf16.
- **VAE decoder** (`kernels/vae.py`, fp32, one-shot) — the T=1 2D-collapsed
  decoder (see below). Device rel_l2 4.4e-6 at 64×64 latent → 512×512 image.
  Replicated per rank.
- **Host** keeps: tokenizer + chat template + embedding-table lookup, latent
  pack / `_unpack_latents` + denorm, and the scalar `mu`/sigma flow-match
  schedule — data-dependent / string-I/O / scalar work that buys nothing on
  device.

Weight footprint: the 7B encoder (~14 GB bf16) at TP=4 adds ~3.5 GB/core on top
of the sharded denoiser (→ ~24 GB/core, tight; TP=8 gives headroom).

The confirmed model structure lives in `config.py` (defaults match
`Qwen/Qwen-Image`); `get_config`/`get_vae_config` read the diffusers config JSON
and fall back to those defaults when diffusers is unavailable.

## Lessons / limits (durable)

Non-obvious things this port surfaced — worth knowing before touching the
kernels or scaling the config.

- **Interleaved RoPE convention.** Qwen RoPE uses the *interleaved* `(2i, 2i+1)`
  pair convention, **not** qwen3's `(i, i+half)` split. Getting this wrong is a
  subtle, hard-to-spot numerical mismatch. (`kernels/rope3d.py`, complex path
  reformulated as real interleaved for the tracer.)
- **VAE latent denormalization.** The pipeline stores `latents_std` as its
  *reciprocal* and does `latents / std_recip` (== ×std). Dividing by the raw std
  instead is off by up to std² (~10×) and produces dark/green-tinted output.
- **1024px exceeds the 2 GB HLO-proto limit.** The whole 60-block denoiser is
  unrolled into one HLO graph. At grid 64×64 (1024px, 4096 image + 34 text
  tokens) the serialized graph exceeds protobuf's 2 GB limit and compile fails in
  `to_proto().SerializeToString()`. This hits **both** the fused `denoise_step`
  and the non-fused `qwenimage_forward`. **Works today:** ≤512px (grid ≤32×32).
  The durable fix (a device-side scan/loop over one compiled block, weights
  indexed per iteration) is independent of TP — TP shrinks per-core *weights*,
  not *graph* size.
- **Tracer dead-weight pruning.** The tracer prunes weights that don't reach the
  output (e.g. the last block's text-stream MLP / out-proj — only the image
  stream feeds `final_layer`), so the driver must filter runtime inputs to
  `kernel.input_tensors_info`.
- **VAE at T=1 collapses 3D→2D.** For text-to-image we decode a single frame, and
  at T=1 the WAN-style 3D causal video VAE reduces *exactly* to a 2D conv decoder
  (`w[:, :, -1]`, `feat_cache=None`, `time_conv` skipped). This is what made the
  VAE port a day instead of a week.
- **Dynamic timestep shifting.** Qwen-Image's scheduler passes
  `mu = calculate_shift(image_seq_len)` to `set_timesteps` — match the pipeline.
