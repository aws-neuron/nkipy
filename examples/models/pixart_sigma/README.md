# PixArt-Σ on Trainium (text-to-image DiT)

A clean implementation of the [PixArt-Σ](https://pixart-alpha.github.io/PixArt-sigma-project/)
diffusion transformer (DiT) denoiser for AWS Trainium, built with NKIPy.

PixArt-Σ is architecturally a **DiT block + cross-attention**: each of the 28
layers runs bidirectional self-attention over the image latent patches, then
cross-attention onto the T5 text tokens, then a gated feed-forward — all
modulated by *adaLN-single*. This reuses the transformer kernels already in
`examples/models/qwen3/kernels` (attention, feedforward, softmax), so the only
genuinely new pieces are patch embedding, timestep + adaLN modulation, and
cross-attention (a non-causal attention with separate K/V source).

## Scope

Every component can run on Trainium; the driver lets you place each on device
or host. Validated on trn2 (bf16 device vs fp32 baseline unless noted):

| Component | Device | Validation | Flag |
|---|---|---|---|
| DiT denoiser (repeated, heavy) | **Trainium** | rel_l2 0.83% | (default) |
| CFG combine + DPM-Solver++ step | **Trainium** | matches host loop (3.5e-7 fp32) | (default, fused) |
| T5-XXL text encoder | **Trainium** | rel_l2 1.5% | `--t5-on-device` |
| VAE decoder (latents → pixels) | **Trainium** (≤512px) | rel_l2 1.3e-5 (fp32) | `--vae-on-device` |
| Tokenizer + token-embedding lookup | host | — | — |

Notes:
- The **fused sampling step** keeps the latent resident on device across all
  steps: `denoise_step` runs the DiT, applies classifier-free guidance, and
  performs the DPM-Solver++ update on device. The host only supplies per-step
  scalar coefficients precomputed from the (fixed) sigma schedule (`dpm_coeffs`).
- The **VAE decoder** is convolutional — the weakest path on the matmul-oriented
  PE array. It is correct on device up to **512px output** (latent grid 64); the
  full **1024px** decode exceeds the Neuron compiler's per-graph instruction
  limit (~10M > 5M), so use the host VAE for 1024px. Kept in fp32 (numerically
  sensitive, one-shot).
- PixArt-Σ-1024-MS is a 1024px-native model; generating at 512px produces
  degraded images regardless of where the VAE runs (a model-resolution effect,
  not an offload bug — the host and device 512px outputs match).

## Setup

```sh
cd nkipy
uv sync --all-groups
uv pip install diffusers accelerate sentencepiece   # for weight prep + baseline
source .venv/bin/activate
cd examples/models/pixart_sigma
```

## Quickstart

```sh
./test.sh            # prepare weights, validate vs baseline, generate output.png
```

Or step by step:

```sh
python tensor_preparation.py --model-name PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \
    --output-dir ./tmp_pixart_sigma
python evaluate.py --validate --checkpoint ./tmp_pixart_sigma      # MSE vs diffusers
python pixart.py "a cat in a spacesuit" --checkpoint ./tmp_pixart_sigma
```

## Files

| File | Purpose |
|---|---|
| `pixart.py` | `PixArtModel` device denoiser + host pipeline (T5, scheduler, VAE) |
| `config.py` | DiT configuration (reads the diffusers checkpoint) |
| `tensor_preparation.py` | Download HF weights, repack to the flat canonical key scheme |
| `evaluate.py` | Single-forward MSE validation vs the diffusers baseline |
| `test.sh` | Smoke test: prepare weights, validate, generate one image |
| `kernels/transformer.py` | `dit_forward` — full denoiser, compiled as one kernel |
| `kernels/dit_block.py` | One DiT block (self-attn + cross-attn + gated FF) |
| `kernels/attention.py` | Bidirectional self-attention and cross-attention |
| `kernels/embeddings.py` | Patch embed, 2D sincos pos-embed, timestep + adaLN-single, caption proj |
| `kernels/modulation.py` | adaLN-single (block + final-layer) modulation |
| `kernels/final_layer.py` | norm_out + proj_out + unpatchify |
| `kernels/{layernorm,feedforward,softmax}.py` | shared primitives |
| `kernels/weight_layout.py` | flat weight-key scheme shared by prep + kernel |

## Status

Implemented on branch `feat/dit-text-to-image`. The full forward is validated
on CPU for shape/dtype correctness; device compile + MSE-vs-diffusers validation
runs via `test.sh` on a Trainium instance (needs `diffusers` installed).

## Notes / TODO

- Single-device only for now; the weight-key scheme and per-block structure
  leave room to add tensor parallelism later (shard attention heads + FF, like
  the qwen3 example).
- `head_dim = 72` (1152/16) is not a power of two — watch for tiling
  inefficiency on the 128-lane PE array.
- The sampler and VAE run on host; a natural follow-on is moving the scheduler
  step math onto the device to cut per-step host round-trips.
