#!/usr/bin/env bash
# Engine B: sleeping engine, no checkpoint (cores 8-15)
# Starts sleeping, activate with /nkipy/wake_up
set -euo pipefail

TP=8

export VLLM_PLUGINS=nkipy
export VLLM_USE_V1=1
export NKIPY_MAX_RDMA_BUFS=64
export NKIPY_CORE_OFFSET=16
# export OMP_NUM_THREADS=1
export VLLM_RPC_TIMEOUT=600000

python3 -m nkipy.vllm_plugin.server \
    --model Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size $TP \
    --max-model-len 128 \
    --max-num-seqs 1 \
    --enforce-eager \
    --dtype bfloat16 \
    --port 8001
