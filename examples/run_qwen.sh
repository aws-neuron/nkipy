WEIGHTS=models/qwen3/tmp_Qwen3-30b-a3b
TP=32

torchrun --nproc-per-node $TP --master-port 29501 \
    server.py \
    --checkpoint $WEIGHTS \
    --model Qwen/Qwen3-30B-A3B \
    --arch qwen3 \
    --port 8000 \
    --neuron-port 62239