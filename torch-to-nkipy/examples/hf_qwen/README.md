# HuggingFace Qwen3 Examples

This directory contains examples for running Qwen3 models on AWS Neuron devices using nkipy.

## Prerequisites

### 1. Download Models

Before running the examples, download the models from HuggingFace:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download Qwen3-0.6B (for single-device inference)
huggingface-cli download Qwen/Qwen3-0.6B

# Download Qwen3-4B (for TP=2 distributed inference)
huggingface-cli download Qwen/Qwen3-4B

# Download Qwen3-8B (for TP=8 distributed inference)
huggingface-cli download Qwen/Qwen3-8B
```

### 2. Requirements

- Python 3.10+
- transformers >= 4.51.0
- torch with nkipy backend

## Supported Models

| Model | Parameters | Recommended TP | Script |
|-------|------------|----------------|--------|
| Qwen/Qwen3-0.6B | 0.6B | 1 | `inference_single.py` |
| Qwen/Qwen3-1.7B | 1.7B | 1, 2 | `inference_single.py` or `inference_distributed.py` |
| Qwen/Qwen3-4B | 4B | 1, 2, 4 | `inference_single.py` or `inference_distributed.py` |
| Qwen/Qwen3-8B | 8B | 4, 8 | `inference_distributed.py` |
| Qwen/Qwen3-14B | 14B | 8 | `inference_distributed.py` |
| Qwen/Qwen3-32B | 32B | 8, 16 | `inference_distributed.py` |

All Qwen3 dense models have 8 KV heads, allowing flexible tensor parallelism degrees.

## Usage

### Single-Device Inference (TP=1)

```bash
python inference_single.py --model Qwen/Qwen3-0.6B
```

Or use the shell script:

```bash
./example_qwen_0.6b.sh
```

### Distributed Inference (TP>1)

Use `torchrun` to launch distributed inference:

```bash
# TP=2
torchrun --nproc-per-node 2 inference_distributed.py --model Qwen/Qwen3-4B

# TP=8
torchrun --nproc-per-node 8 inference_distributed.py --model Qwen/Qwen3-8B
```

Or use the shell scripts:

```bash
./example_qwen_4b_tp2.sh   # TP=2
./example_qwen_8b_tp8.sh   # TP=8
```

## Command-Line Options

```
--model       Model ID from HuggingFace (default: Qwen/Qwen3-0.6B for single, Qwen/Qwen3-8B for distributed)
--dtype       Model dtype: float32, bfloat16, float16 (default: float32)
--prompt      Input prompt for generation (default: "The capital of France is Paris.")
--max-length  Maximum generation length (default: 128)
--cache-dir   Directory for nkipy cache (single-device only, default: ./nkipy_cache)
```

## Example Output

```
$ python inference_single.py --model Qwen/Qwen3-0.6B --max-length 64
Output (device):
The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid...
```
