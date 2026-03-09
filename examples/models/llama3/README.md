# Llama3 on NKIPy

NKIPy implementation of Meta's Llama 3 / 3.1 models on AWS Trainium.

## Quick Start

### 1. Pre-shard weights

```bash
python tensor_preparation.py \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --world-size 8 \
    --head-dim 64 \
    --output-dir tmp_tinyllama_TP8
```

### 2. Run inference

```bash
torchrun --nproc-per-node 8 run_llama3.py \
    -n 500 \
    --checkpoint ./tmp_tinyllama_TP8 \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    "The capital of France is"
```

### 3. Evaluate

```bash
# Generate HF baseline (no torchrun needed)
python evaluate.py --generate-baseline --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 -n 16

# Benchmark
torchrun --nproc-per-node 8 evaluate.py --benchmark \
    --checkpoint ./tmp_tinyllama_TP8 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 -n 500

# Validate accuracy
torchrun --nproc-per-node 8 evaluate.py --validate \
    --checkpoint ./tmp_tinyllama_TP8 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 -n 16
```

## Architecture Differences from Qwen3

| Feature | Qwen3 | Llama3 |
|---------|-------|--------|
| FFN | MoE (router + experts) | Dense (gate/up/down) |
| QK Norm | RMSNorm on Q and K | None |
| RoPE base | 1,000,000 | 500,000 |
| RMS norm eps | 1e-6 | 1e-5 |
