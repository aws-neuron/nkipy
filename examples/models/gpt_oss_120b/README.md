# gpt-oss-120b on Trainium

A NKIPy implementation of OpenAI's **gpt-oss-120b** MoE model for AWS Trainium
(TRN2).

This is a separate example from `examples/models/gpt_oss` (which targets the
smaller `gpt-oss-20b`). The two share the same model architecture but use
different implementation stacks. The 120b example here is built for scale:

| | `gpt_oss` (20b) | `gpt_oss_120b` (this) |
|---|---|---|
| MoE | dense / batched numpy MoE | **blockwise MoE** (NKI kernel + C++ pybind11 index builder) |
| Parallelism | tensor parallel (TP) only | **TP + prefill expert-parallel (EP) + data-parallel (DP)** |
| Attention | NKIPy kernel | NKI prefill/decode kernels (`kernels/attention_nki/`) |
| Decode | per-layer | 4-layer fused decode kernel (`kernels/tokengen.py`) |

## Setup

``` sh
cd nkipy
uv sync --all-groups
source .venv/bin/activate
# The blockwise-index C++ extension is built at runtime and needs pybind11:
uv pip install pybind11
cd examples/models/gpt_oss_120b
```

## Weight preparation

gpt-oss-120b weights must be presharded for the chosen tensor-parallel degree
before running. `scripts/hf_tensor_preparation.py` reads a HuggingFace `gpt_oss`
checkpoint (e.g. [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b),
the `model.layers.*` safetensors layout with MXFP4-quantized experts),
dequantizes the MoE weights to bf16, shards every tensor for TP, and writes
`shard_{rank}.safetensors` into an output directory:

``` sh
python scripts/hf_tensor_preparation.py \
    --model-dir /path/to/openai/gpt-oss-120b \
    --world-size 8 \
    --num-layers 36 \
    --head-dim 64 \
    --output-dir ./gpt-oss-120b-bf16-TP8
```

The runner appends `-TP{tp_size}` to the `--checkpoint` base path, so the
directory above corresponds to `--checkpoint ./gpt-oss-120b-bf16` with
`--tp_size 8`. Dequantized bf16 weights are large (~240 GB total), so make sure
you have disk headroom.

<details>
<summary>Alternative: original OpenAI checkpoint</summary>

If you have the original OpenAI checkpoint instead (the dequantized `block.*`
safetensors layout — run `scripts/dequantize.py` first if it is still MXFP4),
`scripts/openai_tensor_preparation.py` takes the same arguments and produces
byte-identical shards:

``` sh
python scripts/openai_tensor_preparation.py \
    --model-dir /path/to/gpt-oss-120b \
    --world-size 8 --num-layers 36 --head-dim 64 \
    --output-dir ./gpt-oss-120b-bf16-TP8
```
</details>

## Quickstart

`chat.sh` launches `torchrun` across all `TP_SIZE * DP_SIZE` ranks and runs
generation end-to-end. It activates the project `.venv` (for both `python` and
the `neuronx-cc` compiler), builds the C++ blockwise-index extension on rank 0,
then compiles kernels.

The defaults are the layout verified on a single `trn2.48xlarge` (16 devices /
64 logical NeuronCores under LNC2): 64 ranks, `dp8`/`ep8`, one rank per logical
core, so each rank holds only `128/8 + 128/8 = 32` expert copies (~7 GB), well
under the 24 GB per-core HBM budget.

``` sh
# Defaults: TP_SIZE=8 DP_SIZE=8 PREFILL_EP_SIZE=8
CHECKPOINT=./gpt-oss-120b-bf16 ./chat.sh "The capital of France is"
```

`chat.sh` activates the repo `.venv` (set `VENV=` to use the active environment
instead) and reads `TP_SIZE`, `DP_SIZE`, `PREFILL_EP_SIZE`, and `CHECKPOINT` from
the environment. Override the parallelism layout via env vars:

``` sh
TP_SIZE=8 DP_SIZE=2 PREFILL_EP_SIZE=2 ./chat.sh "The capital of France is"
```

The heavier **128-rank** layout (`DP16`) needs 128 logical cores
(`NEURON_LOGICAL_NC_CONFIG=1`) **and** `PREFILL_EP_SIZE>=8`. The `ep4` variant
runs out of HBM during warmup: under LNC1 two ranks share one device's 24 GB, and
`ep4` leaves ~11 GB of expert tensors per rank. `ep8` halves that (~5.4 GB) and
fits:

``` sh
NEURON_LOGICAL_NC_CONFIG=1 DP_SIZE=16 PREFILL_EP_SIZE=8 \
    CHECKPOINT=./gpt-oss-120b-bf16 ./chat.sh "The capital of China is"
```

### Sizing notes (single-node TRN2)

- **HBM per rank.** gpt-oss-120b is memory-heavy: after TP sharding each rank
  still holds separate prefill and decode expert copies. If several ranks land
  on one device you get `NRT_RESOURCE: Failed to allocate tensor` at load, or an
  `ENC:__devmem_res_checkout ... failed to allocate dev mem` / `NRT_FAILURE`
  when building collective resources during warmup. Spread ranks across devices
  (`GPT_OSS_CORE_STRIDE`) and raise expert-parallelism (`DP_SIZE`,
  `PREFILL_EP_SIZE`) so `num_experts/ep` shrinks. Note `NEURON_LOGICAL_NC_CONFIG=1`
  gives more cores but **not** more memory — it splits each physical core's 24 GB
  HBM between two logical cores, so packing 2 ranks per device needs a smaller
  per-rank footprint (higher EP).
- **`max_model_len`** must satisfy `(max_model_len // tp_size) % 32 == 0` and be
  `>= 512` (the NKI flash-attention KV tile is 512).

## Architecture notes

gpt-oss specifics handled here (shared with the 20b example):

| Feature | Handling |
|---|---|
| Clamped SwiGLU | `(up+1) * gate*sigmoid(alpha*gate)` with `clamp(limit=7)` |
| Attention sinks | Per-head sink logit folded into the softmax |
| QKV / output bias | Carried through prep, added in the attention kernel |
| Sliding-window attention | Alternating sliding (window=128) / full causal layers |
| YaRN RoPE | `inv_freq` + attention-scaling precomputed and baked into cos/sin |
| Router | top-k on raw logits (+bias), then softmax over the selected logits |

## Files

| File | Purpose |
|---|---|
| `chat.py` | Generation entrypoint (tokenize, load, warm up, generate) |
| `chat.sh` | `torchrun` launcher with the parallelism layout |
| `model.py` | `GPTOSSModel`: tensor prep, kernel compilation, prefill/decode loops |
| `prefill_layer.py` | Per-layer prefill (attention + blockwise MoE) |
| `config.py` | Model configuration and parallel-aware assertions |
| `parallel_state.py` | TP / prefill-EP / DP rank and group bookkeeping |
| `collective.py` | all_gather / reduce_scatter / all_reduce / all_to_all wrappers |
| `kernels/` | Attention, MoE (blockwise), RoPE, RMSNorm, router, sampling, decode |
| `kernels/attention_nki/` | Hand-written NKI prefill/decode attention kernels |
| `kernels/blockwise_index.cpp` | C++ blockwise expert/token index builder (pybind11) |
| `scripts/` | Weight preparation and offline conversion utilities |
| `tests/` | Unit tests for the kernels (run on a Trainium host) |

## Testing

Kernel unit tests live in `tests/` and require a Trainium host for the
device-mode cases:

``` sh
pytest tests/ -n auto
```

CPU/HLO-mode tests (e.g. `-k cpu`) run without hardware; raw-NKI kernel tests
need the device.
