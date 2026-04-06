WEIGHTS=../models/qwen3/tmp_Qwen3-30b-a3b
TP=8
export NKIPY_MAX_RDMA_BUFS=64
torchrun --nproc-per-node $TP --master-port 29501 \
    server.py \
    --checkpoint $WEIGHTS \
    --model Qwen/Qwen3-30B-A3B \
    --arch qwen3 \
    --port 8000 \
    --neuron-port 62239 \
    --spread-nics \
    --core-offset 0 \
    --context-len 64 \
    --max-tokens 256
