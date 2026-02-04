# Qwen3 on Trainium

A clean implementation of Qwen3 MoE variants (e.g., Qwen3-30B-A3B) for AWS Trainium.

## Quickstart

To run the model, make sure you have the environment ready by

``` sh
cd nkipy
uv sync --all-groups
source .venv/bin/activate
```

Then run the model test by

``` sh
cd examples/models/qwen3
./test.sh
```
