#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
set -ex

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

CACHE_DIR=$SCRIPT_DIR/cache
# CACHE_DIR=$HOME/cache
VLLM_NKIPY_CACHE_DIR=$CACHE_DIR/cache
mkdir -p $CACHE_DIR

# export TORCH_LOGS="+dynamo"
# export TORCH_LOGS="recompiles"

export NEURON_LOGICAL_NC_CONFIG=1

# export NEURON_RT_LOG_LEVEL="INFO"
# export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3

# vllm profiler
export VLLM_TORCH_PROFILER_DIR=$SCRIPT_DIR/torch_profile

# profile (NTFF artifact saving)
# export VLLM_NKIPY_COMPILE_SAVE_NTFF=1
# export VLLM_NKIPY_COMPILE_SAVE_NTFF_EXE_IDX="[20]"

# ep
export VLLM_NKIPY_MOE_2D=1

# vllm serve args
export MAX_NUM_SEQS=4
export NUM_GPU_BLOCKS_OVERRIDE=2048
export MAX_MODEL_LENGTH=10240

neuronx-cc --version
apt list | grep -E "neuronx-dkms|neuronx-c|neuronx-runtime"

VLLM_PLUGINS=nkipy VLLM_USE_V1=1 VLLM_NKIPY_CACHE_DIR="$VLLM_NKIPY_CACHE_DIR" vllm serve unsloth/gpt-oss-120b-BF16 \
  --tensor-parallel-size 8 \
  --data-parallel-size 8 \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len $MAX_MODEL_LENGTH \
  --max-num-seqs $MAX_NUM_SEQS \
  --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE \
  --additional-config '{"num_tokens_paddings": [4, 16, 1024], "compile_strategy": "merge_layers", "split_graph": false, "merge_step": 9, "large_kv_tile_size": 4096, "large_q_tile_size": 128, "dynamic_loop_unroll": 4, "paged_attn_impl": "nki_blocksparse_flash_attention_kv_cache", "paged_kv_impl": "update_kv_cache_custom_op"}'
