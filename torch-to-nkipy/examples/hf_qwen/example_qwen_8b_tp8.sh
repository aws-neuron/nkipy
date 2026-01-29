#!/bin/bash
# Distributed inference with Qwen3-8B using TP=8
torchrun --nproc-per-node 8 inference_distributed.py --model Qwen/Qwen3-8B
