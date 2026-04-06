WEIGHTS=../models/llama3/tmp_tinyllama_TP8
TP=8
export NKIPY_MAX_RDMA_BUFS=128
torchrun --nproc-per-node $TP --master-port 29500 \
    server.py \
    --checkpoint $WEIGHTS \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --arch llama3 \
    --port 8000 \
    --neuron-port 61239 \
    --core-offset 0 \