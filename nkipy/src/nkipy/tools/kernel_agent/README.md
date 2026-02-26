# Kernel Agent

Test NumPy operations against the NKIPy hardware pipeline and generate kernels with an LLM.

## Commands

```bash
# List all target operations
python -m nkipy.tools.kernel_agent list-ops

# Test specific ops (numpy -> compile -> hardware)
python -m nkipy.tools.kernel_agent test --ops add,exp --dtypes float32,int32

# Discover support status for all ops
python -m nkipy.tools.kernel_agent discover --dtypes float32 -o results.json

# Generate a kernel from a natural-language prompt
python -m nkipy.tools.kernel_agent generate -p "softmax over last axis"

# Run a continuous generation-and-test sweep
python -m nkipy.tools.kernel_agent sweep --max-iterations 50 --output-dir sweep_results
```

Use `--no-hardware` on any command to skip hardware execution.

## Architecture

| Module | Purpose |
|---|---|
| `ops.py` | Op registry, deterministic input generation, per-op testing |
| `executor.py` | 3-stage pipeline: **numpy -> compile -> hardware** |
| `generator.py` | LLM kernel generation via AWS Bedrock |
| `sweep.py` | Continuous sweep loop with JSONL logging |
| `__main__.py` | CLI entry point |
