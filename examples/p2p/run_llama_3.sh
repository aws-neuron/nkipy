WEIGHTS=../models/llama3/tmp_tinyllama_TP8
TP=8

torchrun --nproc-per-node $TP --master-port 29700 \
    server.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --arch llama3 \
    --port 8002 \
    --neuron-port 61439 \
    --core-offset 0 \
    --checkpoint $WEIGHTS \