# gpt-oss on Trainium

A clean implementation of OpenAI's gpt-oss MoE models (e.g., `gpt-oss-20b`) for
AWS Trainium, built on NKIPy.

## Setup

``` sh
cd nkipy
uv sync --all-groups
source .venv/bin/activate
cd examples/models/gpt_oss
```

## Quickstart

`test.sh` handles weight preparation and runs a generation end-to-end:

``` sh
./test.sh
```

Or run generation directly (assumes weights are already prepared):

``` sh
WEIGHTS=./tmp_gpt-oss-20b
TP=4

torchrun --nproc-per-node $TP gpt_oss.py \
    -n 500 --checkpoint $WEIGHTS --model openai/gpt-oss-20b \
    "The capital of France is"
```

You can point `--model` at a local checkpoint directory too.

## Weight preparation

gpt-oss ships its experts **MXFP4-quantized** (`*_blocks` / `*_scales`). The prep
step dequantizes them to bf16 so the NKI kernels run purely in bf16, and it
shards every tensor for tensor parallelism:

``` sh
python tensor_preparation.py \
    --model-name openai/gpt-oss-20b \
    --world-size 4 \
    --output-dir ./tmp_gpt-oss-20b
```

This writes `shard_{rank}.safetensors` files. Dequantized bf16 weights are
larger than the packed checkpoint (~40 GB total), so make sure you have disk
headroom.

## Architecture notes

gpt-oss differs from the Qwen3 MoE example in several ways, all handled here:

| Feature | Handling |
|---|---|
| MXFP4 experts | Dequantized to bf16 at prep time (`tensor_preparation.py`) |
| Interleaved gate/up | De-interleaved to `[gate \| up]` at prep time |
| Clamped SwiGLU | `(up+1) * gate*sigmoid(alpha*gate)` with `clamp(limit=7)` (`kernels/feedforward.py`) |
| Attention sinks | Per-head sink logit concatenated into softmax, then dropped (`kernels/attention.py`) |
| QKV / output bias | Carried through prep and added in the attention kernel |
| Sliding-window attention | Alternating sliding (window=128) / full causal layers; one kernel compiled per attention type |
| YaRN RoPE | `inv_freq` + attention-scaling precomputed from HF config (`config.py`) and baked into the cos/sin cache |
| Router | top-k on raw logits (+bias), then softmax over the selected logits |

## Files

| File | Purpose |
|---|---|
| `gpt_oss.py` | Model definition (`GptOssModel`) and text generation |
| `config.py` | Model configuration (incl. YaRN RoPE precompute) |
| `tensor_preparation.py` | Dequantize, reshape, and shard HF weights for TP |
| `test.sh` | Smoke test: prepares weights and runs generation |
| `kernels/` | Attention, feed-forward, RoPE, RMSNorm, softmax, sampling kernels |
