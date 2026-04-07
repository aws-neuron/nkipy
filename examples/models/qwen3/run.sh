WEIGHTS=./tmp_Qwen3-30b-a3b
TP=8

torchrun --nproc-per-node $TP qwen3.py \
    -n 500 --checkpoint $WEIGHTS --model Qwen/Qwen3-30B-A3B \
    "The capital of France is"