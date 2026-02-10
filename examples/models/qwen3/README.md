# Qwen3 on Trainium

A clean implementation of Qwen3 MoE variants (e.g., Qwen3-30B-A3B) for AWS Trainium.

## Setup

``` sh
cd nkipy
uv sync --all-groups
source .venv/bin/activate
cd examples/models/qwen3
```

## Quickstart

`test.sh` handles weight preparation and runs a generation end-to-end:

``` sh
./test.sh
```

Or run generation directly (assumes weights are already prepared):

``` sh
WEIGHTS=./tmp_qwen3-30b-a3b
TP=4

torchrun --nproc-per-node $TP qwen3.py \
    -n 500 --checkpoint $WEIGHTS --model Qwen/Qwen3-30B-A3B \
    "The capital of France is"
```

## Benchmarking

Measure TTFT, decode latency percentiles (p50/p90/p99), and throughput:

``` sh
torchrun --nproc-per-node $TP evaluate.py --benchmark \
    -n 100 --checkpoint $WEIGHTS --model Qwen/Qwen3-30B-A3B \
    --benchmark-warmup 2 --benchmark-runs 5 \
    --benchmark-output benchmark_report.json
```

Results are printed to stdout and saved as JSON.

## Accuracy validation

### 1. Generate a HF baseline (no torchrun needed)

``` sh
python evaluate.py --generate-baseline \
    -n 20 --model Qwen/Qwen3-30B-A3B \
    --output baseline_tokens.pt
```

### 2. Validate NKIPy output against the baseline

``` sh
torchrun --nproc-per-node $TP evaluate.py --validate \
    -n 20 --checkpoint $WEIGHTS --model Qwen/Qwen3-30B-A3B \
    --baseline-path baseline_tokens.pt
```

## Weight preparation

If you need to prepare weights **separately** from `test.sh`:

``` sh
python tensor_preparation.py \
    --model-name Qwen/Qwen3-30B-A3B \
    --world-size 4 --head-dim 128 \
    --output-dir ./tmp_qwen3-30b-a3b
```

## Files

| File | Purpose |
|---|---|
| `qwen3.py` | Model definition (`Qwen3Model`) and text generation |
| `evaluate.py` | Benchmarking, accuracy validation, and HF baseline generation |
| `test.sh` | Smoke test: prepares weights and runs generation |
| `config.py` | Model configuration |
| `tensor_preparation.py` | Download and shard HF weights for tensor parallelism |
