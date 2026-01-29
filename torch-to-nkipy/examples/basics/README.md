# Basic NKIPy Examples

This directory contains basic examples demonstrating the core features of nkipy.

## Examples

### 1. Basic Usage (`example_basic.py`)

A simple example showing how to compile and run a PyTorch model with nkipy.

```bash
python example_basic.py
```

### 2. Distributed Example (`example_dist.py`)

Demonstrates distributed execution with tensor parallelism using `torch.distributed`.

**Important:** Use `torchrun` to launch distributed examples:

```bash
# Run with 2 processes (TP=2)
torchrun --nproc-per-node 2 example_dist.py
```

Or use the shell script:

```bash
./example_dist.sh
```

### 3. Hybrid Execution (`example_hybrid_exe.py`)

Shows how to run part of the computation on CPU and part on Neuron.

```bash
python example_hybrid_exe.py
```

### 4. Custom NKI Operator (`example_nki_op.py`)

Demonstrates how to integrate custom NKI (Neuron Kernel Interface) operators.

```bash
python example_nki_op.py
```
