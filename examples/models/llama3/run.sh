WEIGHTS=./tmp_tinyllama_TP8
TP=8

torchrun --nproc-per-node $TP llama3.py \
    -n 500 --checkpoint $WEIGHTS --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    "The capital of France is"
