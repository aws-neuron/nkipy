#!/bin/bash
# Setup script for nkipy development environment.
# Installs nkipy deps via uv, then vllm via private-vllm-neuron,
# then replaces CUDA torch with CPU torch + torch-neuronx.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NEURON_INDEX="https://pip.repos.neuron.amazonaws.com"
UV_PIP="uv pip install --index-strategy unsafe-best-match"
cd "$SCRIPT_DIR"

echo "==> uv sync (all groups)..."
uv sync --all-groups

echo "==> Installing private-vllm-neuron (includes vllm)..."
$UV_PIP --extra-index-url="$NEURON_INDEX" -e ../private-vllm-neuron

echo "==> Replacing CUDA torch with CPU torch + torch-neuronx..."
$UV_PIP --extra-index-url="$NEURON_INDEX" \
    --extra-index-url=https://download.pytorch.org/whl/cpu \
    torch-neuronx torch --reinstall-package torch

echo "==> Done."
uv run python -c "import torch; print(f'torch {torch.__version__}'); import vllm; print(f'vllm {vllm.__version__}')"
