TP=8

torchrun --nproc-per-node $TP --master-port 29601 \
    server.py \
    --model Qwen/Qwen3-30B-A3B \
    --arch qwen3 \
    --port 8001 \
    --neuron-port 62339 \
    --core-offset 16 \
    --context-len 64 \
    --max-tokens 256
