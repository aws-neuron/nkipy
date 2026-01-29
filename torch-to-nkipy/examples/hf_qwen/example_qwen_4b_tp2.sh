#!/bin/bash
# Distributed inference with Qwen3-4B using TP=2
torchrun --nproc-per-node 2 inference_distributed.py --model Qwen/Qwen3-4B
