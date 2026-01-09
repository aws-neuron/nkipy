# Qwen3-Embedding on Trainium

A clean, simple implementation of Qwen3-Embedding-*B for AWS Trainium accelerators.

## Quick Start

Directly run test script (cleans cache, checks weights, runs example)
`bash test.sh`

Or, run the two steps manually

```bash
# 1. Download and convert weights (one-time setup)
python prepare_weights.py

# 2. Run retrieval example
python example_retrieval.py
```

Or to compare with huggingface example. This compares embeddings between Trainium and HuggingFace implementations. Expected cosine similarity: >0.99

```bash
python compare.py
```

## Modifying the Code

**Note:** Changing code requires recompiling kernels:
```bash
rm -rf /tmp/build/qwen3_*
```

### To change sequence length:
1. Edit `max_model_len` in `config.py`
2. Clean cache: `rm -rf /tmp/build/qwen3_*`
3. Run your code (kernels will recompile)

### To add custom pooling:
1. Add your pooling function to `embedding_utils.py`
2. Update `model.py` to use it in the `forward()` method

### To modify kernels:
1. Edit files in `kernels/` directory
2. Clean cache to force recompilation
