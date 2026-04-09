#!/usr/bin/env bash
# Engine A: active engine with checkpoint (cores 0-7)
set -euo pipefail

WEIGHTS=~/zhuangw/nkipy/examples/models/llama3/tmp_tinyllama_TP8
TP=8

export VLLM_PLUGINS=nkipy
export VLLM_USE_V1=1
export NKIPY_CHECKPOINT=$WEIGHTS
export OMP_NUM_THREADS=1
export VLLM_RPC_TIMEOUT=600000

python3 -m nkipy.vllm_plugin.server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tensor-parallel-size $TP \
    --max-model-len 128 \
    --max-num-seqs 1 \
    --enforce-eager \
    --dtype bfloat16 \
    --port 8000
