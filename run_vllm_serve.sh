set -x

MAX_BATCH_SIZE=1
MAX_MODEL_LEN=10240
TP_SIZE=8
EP_SIZE=16
WORLD_SIZE=$[$TP_SIZE*$EP_SIZE]

export NEURON_LOGICAL_NC_CONFIG=1
export MODEL_CHECKPOINT="/shared/ziyangx/gpt-oss-120b-bf16-moe-fp8-TP${TP_SIZE}"
export MODEL_NAME="openai/gpt-oss-120b"

PYTHONPATH=. VLLM_USE_V1=1 vllm serve "meta-llama/Llama-3.1-405B" \
    --tensor-parallel-size ${WORLD_SIZE} \
    --no-enable-chunked-prefill \
    --max-num-seqs ${MAX_BATCH_SIZE} \
    --tokenizer ${MODEL_NAME} \
    --trust-remote-code \
    --max-model-len ${MAX_MODEL_LEN} \
    --dtype bfloat16 \
    --swap-space 0 \
    --override-neuron-config.enable_bucketing False \
    --no-enable-prefix-caching
