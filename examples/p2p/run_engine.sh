#!/usr/bin/env bash
# Unified vLLM+NKIPy engine launcher.
#
# Usage:
#   run_engine.sh --model MODEL --tp TP [--checkpoint PATH] [--port PORT]
#                 [--core-offset N] [--host-staging] [--lnc N]
#                 [--sleep-when-idle] [--hf-offline]
#                 [--skip-cte] [--log-level LEVEL] [--activate-venv]
#
# Examples:
#   # Qwen3 sender (Engine A) with checkpoint:
#   ./run_engine.sh --model Qwen/Qwen3-30B-A3B --tp 32 \
#       --checkpoint ~/models/Qwen3-30b-a3b_TP32 --skip-cte
#
#   # Qwen3 receiver (Engine B) in sleep mode:
#   ./run_engine.sh --model Qwen/Qwen3-30B-A3B --tp 32 --skip-cte --activate-venv
#
#   # Qwen3 receiver on same node with core offset:
#   ./run_engine.sh --model Qwen/Qwen3-30B-A3B --tp 8 --core-offset 16 --port 8001
#
#   # LLaMA-3.1-70B sender on trn2.48xlarge (cross-node):
#   ./run_engine.sh --model /fsx/models/llama-3.1-70b --tp 32 \
#       --checkpoint /fsx/models/llama_3.1_70b_TP32 \
#       --lnc 2 --skip-cte --hf-offline --activate-venv
#
#   # LLaMA-3.1-70B receiver on trn2.48xlarge (cross-node):
#   ./run_engine.sh --model /fsx/models/llama-3.1-70b --tp 32 \
#       --core-offset 32 --lnc 2 \
#       --skip-cte --hf-offline --activate-venv
#
#   # TinyLlama sender:
#   ./run_engine.sh --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp 8 \
#       --checkpoint ~/models/tinyllama_TP8 --log-level DEBUG
#
#   # TinyLlama receiver on same node:
#   ./run_engine.sh --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp 8 \
#       --core-offset 16 --port 8001
set -euo pipefail

PORT=8000
MAX_MODEL_LEN=128
MAX_NUM_SEQS=1
DTYPE=bfloat16

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)          MODEL="$2";         shift 2;;
        --tp)             TP="$2";            shift 2;;
        --checkpoint)     CHECKPOINT="$2";    shift 2;;
        --port)           PORT="$2";          shift 2;;
        --core-offset)    CORE_OFFSET="$2";   shift 2;;
        --max-model-len)  MAX_MODEL_LEN="$2"; shift 2;;
        --max-num-seqs)   MAX_NUM_SEQS="$2";  shift 2;;
        --dtype)          DTYPE="$2";         shift 2;;
        --sleep-when-idle) SLEEP_WHEN_IDLE=1; shift;;
        --hf-offline)     HF_OFFLINE=1;       shift;;
        --skip-cte)       SKIP_CTE=1;         shift;;
        --host-staging)   HOST_STAGING=1;     shift;;
        --lnc)            LNC_CONFIG="$2";    shift 2;;
        --log-level)      LOG_LEVEL="$2";     shift 2;;
        --activate-venv)  ACTIVATE_VENV=1;    shift;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [[ -z "${MODEL:-}" || -z "${TP:-}" ]]; then
    echo "Usage: $0 --model MODEL --tp TP [options]"
    echo "Run '$0 --help' or see script header for examples."
    exit 1
fi

if [[ -n "${ACTIVATE_VENV:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/../../.venv/bin/activate"
fi

export VLLM_PLUGINS=nkipy
export VLLM_USE_V1=1
export OMP_NUM_THREADS=1
export VLLM_RPC_TIMEOUT=600000

[[ -n "${CHECKPOINT:-}" ]]      && export NKIPY_CHECKPOINT="$CHECKPOINT"
[[ -n "${CORE_OFFSET:-}" ]]     && export NKIPY_CORE_OFFSET="$CORE_OFFSET"
[[ -n "${SKIP_CTE:-}" ]]        && export NKIPY_SKIP_CTE=1
[[ -n "${HOST_STAGING:-}" ]]    && export NKIPY_HOST_STAGING=1
[[ -n "${LNC_CONFIG:-}" ]]      && export NEURON_LOGICAL_NC_CONFIG="$LNC_CONFIG"
[[ -n "${SLEEP_WHEN_IDLE:-}" ]] && export VLLM_SLEEP_WHEN_IDLE=1
[[ -n "${HF_OFFLINE:-}" ]]      && export HF_HUB_OFFLINE=1
[[ -n "${LOG_LEVEL:-}" ]]       && export VLLM_LOGGING_LEVEL="$LOG_LEVEL"

exec python3 -m nkipy.vllm_plugin.server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enforce-eager \
    --dtype "$DTYPE" \
    --port "$PORT"
