# Qwen3-Embedding on Trainium

A clean implementation of Qwen3-Embedding (0.6B and 8B) for AWS Trainium.

## Quick Start

```bash
# Run test script (cleans cache, checks weights, runs example)
bash test.sh

# Or run manually:
python prepare_weights.py          # Download and convert weights
python example_retrieval.py        # Run retrieval example
```

## Usage

```bash
# Basic retrieval example (0.6B model)
python example_retrieval.py

# Use 8B model
python example_retrieval.py --model-size 8b

# Run performance benchmark
python example_retrieval.py --benchmark

# Compare with HuggingFace (verify correctness)
python example_retrieval.py --compare

# Custom sequence length
python example_retrieval.py --seq-len 512

# Use LNC=1 (single NeuronCore)
python example_retrieval.py --lnc 1

# Use separate kernels instead of fused (for debugging)
python example_retrieval.py --no-fused
```

## Performance Tips

Enable async execution for better throughput:
```bash
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=16
```

Note: LNC (Logical NeuronCore) is set at compile time via `--lnc` flag, not via `NEURON_LOGICAL_NC_CONFIG` at runtime.

## Modifying the Code

After changing code, clean the kernel cache:
```bash
rm -rf build/
```

## Files

- `example_retrieval.py` - Main script with retrieval, benchmark, and compare modes
- `config.py` - Model configurations (0.6B and 8B)
- `model.py` - Trainium model implementation
- `prepare_weights.py` - Weight download and conversion
- `kernels/` - NKI kernel implementations
- `layer.py` - Alternative implementation using separate kernels (for debugging)
