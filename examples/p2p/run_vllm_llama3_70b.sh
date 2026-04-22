#!/usr/bin/env bash
# Engine A: active engine with checkpoint (all 32 cores)
# LLaMA-3-70B on Neuron with TP=32
set -euo pipefail

source /home/ubuntu/vllm-nkipy/nkipy/.venv/bin/activate

WEIGHTS=/home/ubuntu/models/llama-3-70b-TP32
TP=32

export VLLM_PLUGINS=nkipy
export VLLM_USE_V1=1
export NKIPY_CHECKPOINT=$WEIGHTS
export OMP_NUM_THREADS=1
export NKIPY_SKIP_CTE=1
export VLLM_RPC_TIMEOUT=600000
export VLLM_SLEEP_WHEN_IDLE=1
export HF_HUB_OFFLINE=1

python3 -m nkipy.vllm_plugin.server \
    --model /home/ubuntu/models/llama-3-70b \
    --tensor-parallel-size $TP \
    --max-model-len 128 \
    --max-num-seqs 1 \
    --enforce-eager \
    --dtype bfloat16 \
    --port 8000
