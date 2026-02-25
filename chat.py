# prioritize debugpy
import os

debug_rank = os.environ.get("NEURONPY_DEBUG_RANK")
local_rank = os.getenv("LOCAL_RANK")
if debug_rank and int(debug_rank) == int(local_rank):
    import signal

    import debugpy
    def signal_handler(_, __):
        print(
            f"Rank {local_rank}: Caught SIGTERM, ignoring to continue debugging"
        )
    signal.signal(signal.SIGTERM, signal_handler)
    debugpy.listen(5678)
    print(f"[Rank {local_rank}] Listening for debugpy")
    # debugpy.wait_for_client()
import argparse
import subprocess
import time
from collections import defaultdict

import numpy as np
import torch.distributed as dist
from nkipy.core.compile import _get_build_dir, _set_build_dir

import parallel_state
from config import get_config, set_env
from kernels.blockwise_index import (
    add_blockwise_index_to_path,
    build_blockwise_index_cpp,
)
from logger import get_logger
from model import GPTOSSModel, load_gpt_oss_weights

logger = get_logger()


def tokenize_prompts(tokenizer, config, args, enc=None):
    # Handle prompt input - check if it's a file path or direct text
    assert isinstance(args.prompts, list)
    if len(args.prompts) == 1 and os.path.isfile(args.prompts[0]):
        # It's a file path, read the content
        with open(args.prompts[0], "r") as f:
            prompts = [line.strip() for line in f]
    else:
        # It's direct prompt text
        prompts = args.prompts
    num_prompts = len(prompts)
    assert (
        num_prompts <= args.max_batch_size_per_dp
    ), f"{num_prompts=} provided, exceeding {args.max_batch_size_per_dp=}"

    model_inputs = tokenizer(
        prompts,
        return_tensors="np",
        return_attention_mask=True,
        max_length=config.max_model_len,
        padding="max_length",
        padding_side="right",
    )
    input_ids = model_inputs["input_ids"]
    context_lengths = model_inputs["attention_mask"].sum(axis=1, keepdims=False)
    assert np.all(args.max_tokens + context_lengths <= config.max_model_len), (
        f"{args.max_tokens + context_lengths=} should not exceed {config.max_model_len=}"
    )

    # pad to [config.max_batch_size_per_dp, config.max_model_len]
    input_ids = np.pad(
        input_ids,
        (
            (0, config.max_batch_size_per_dp - num_prompts),
            (0, config.max_model_len - input_ids.shape[1]),
        ),
    )
    context_lengths = np.pad(context_lengths, (0, config.max_batch_size_per_dp - num_prompts))

    input_ids = input_ids.astype(np.uint32)
    context_lengths = context_lengths.astype(np.int32)
    input_ids.setflags(write=False)
    context_lengths.setflags(write=False)

    return num_prompts, input_ids, context_lengths


def main():
    parser = argparse.ArgumentParser()
    # FIXME: support specify unlimited tokens
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--max_batch_size_per_dp", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument(
        "prompts",
        nargs="*",
        default=[
            "The capital of France is",
            "The most popular programming language is",
            "The first president of the United States was",
            "The capital of France is",  # duplicate to test reference comparison logic
            "The chemical symbol for water is",
            "The largest planet in our solar system is",
            "The author of the Harry Potter series is",
            "The largest ocean on Earth is",
        ],
        # default="./nki_prompt_10k.txt",
        help="Prompt text or path to a prompt file",
    )
    parser.add_argument(
        "--checkpoint",
        default="/shared/ziyangx/gpt-oss-120b-bf16-moe-fp8",
        # default="/shared/zhenyus/qwen3_shards_30B_A3B_TP8"
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--prefill_ep_size",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    args.checkpoint = f"{args.checkpoint}-TP{args.tp_size}"
    assert os.path.exists(args.checkpoint)
    logger.info(str(args))

    # model_name = args.model
    model_name = "openai/gpt-oss-120b"
    dist.init_process_group(backend="gloo")
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank())
    parallel_state.initialize_model_parallel(args.tp_size, args.prefill_ep_size)
    set_env()
    config = get_config(
        model_name,
        max_batch_size_per_dp=args.max_batch_size_per_dp,
        max_model_len=args.max_model_len,
    )

    build_dir = f"/tmp/build/gpt-oss-120b-DP{parallel_state.get_dp_size()}-PrefillEP{parallel_state.get_prefill_ep_size()}-TP{args.tp_size}-PerDPBS{args.max_batch_size_per_dp}-SEQ{config.max_model_len}"
    _set_build_dir(build_dir)
    logger.info(f"build_dir: {build_dir}")
    if dist.get_rank() == 0:
        if os.environ.get("NEURONPY_NOT_CLEAR_BUILD_CACHE") != "1":
            subprocess.run(f"rm -rf {_get_build_dir()}", shell=True)
        build_blockwise_index_cpp(
            n_experts=config.num_experts//parallel_state.get_prefill_ep_size(), # each EP locally only sees num_experts_per_ep
            top_k=config.num_experts_per_tok,
            n_blocks=config.num_blocks,
            n_static_blocks=config.num_static_blocks,
        )
        # system profile
        os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
        os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "./profile"
    dist.barrier()  # wait for remove build dir
    add_blockwise_index_to_path()

    # import is slow, so do it here
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_prompts, input_ids, context_lengths = tokenize_prompts(tokenizer, config, args)

    weights = load_gpt_oss_weights(args.checkpoint, config)

    # Create and run the model
    model = GPTOSSModel(
        weights, config, check_against_reference=True
    )  # FIXME: embed context length

    # weights can be released by now
    del weights

    ref_prompt_to_compare = "The capital of France is"
    batch_indices_to_check = []
    for batch_idx, prompt in enumerate(args.prompts):
        if ref_prompt_to_compare == prompt:
            batch_indices_to_check.append(batch_idx)
    logger.info(f"{batch_indices_to_check=}")

    # Warming up. This is because on neuron the first time run a neff is slow
    start = time.time()
    logger.info("Warming model")
    t = 0
    for token_ids in model.generate(
        input_ids=input_ids,
        context_lens=context_lengths,
        max_tokens=args.max_tokens,
        warmup=True,
        batch_indices_to_check=batch_indices_to_check,
    ):
        logger.debug(f"{t=} {token_ids.reshape(-1)=}")
        if t == 1:
            break
        t += 1
    logger.info(f"--> Finished warming the model in {time.time() - start:.2f}s")

    dist.barrier()
    # Generate tokens and measure performance
    start = time.time()
    t = 0
    req_outputs = defaultdict(list)
    for token_ids in model.generate(
        input_ids=input_ids,
        context_lens=context_lengths,
        max_tokens=args.max_tokens,
        warmup=False,
    ):
        if t == 0:
            first_token_time = time.time()
        t += 1
        for req_id in range(num_prompts):
            output_id = token_ids[req_id].tolist()
            text_token = tokenizer.decode(output_id)
            req_outputs[req_id].append(text_token)

            if output_id[-1] in [151643, 151645]:  # EOS or BOS token
                req_outputs[req_id].append("Found EOS or BOS token")

    end_time = time.time()

    if dist.get_rank() == 0:
        for req_id in range(num_prompts):
            print("===============================")
            print(f"Request {req_id}: {args.prompts[req_id]}")
            print("".join(req_outputs[req_id]))
            print()
    logger.info(f"Generated {t} tokens")

    ttft = first_token_time - start
    decoding_time = end_time - first_token_time
    tokens_per_second_per_req = t / decoding_time
    if dist.get_rank() == 0:
        print(f"\nTime to first token: {ttft:.2f}s")
        moe_matmul_per_layer_per_batch = 3 * config.max_model_len * config.num_experts_per_tok * config.intermediate_size * config.hidden_size
        qkvo_proj = (2 * config.n_heads + 2 * config.n_kv_heads) * config.max_model_len * config.hidden_size * config.head_dim
        core_attn_full = 0.5 * config.n_heads * 2 * config.max_model_len * config.head_dim * config.max_model_len
        core_attn_sliding_window = config.n_heads * 2 * config.max_model_len * config.head_dim * config.sliding_window
        avg_attn_matmul_per_layer_per_batch = qkvo_proj + (core_attn_full + core_attn_sliding_window) / 2
        flops_per_ep_per_tp = (
            2
            * config.max_batch_size_per_dp
            * config.n_layers
            * (moe_matmul_per_layer_per_batch + avg_attn_matmul_per_layer_per_batch)
            / parallel_state.get_tp_size()
        )
        print(f"Prefill MFU: {flops_per_ep_per_tp / 79e12 / ttft * 100:.2f}%")

        # moe is fp8
        weight_bytes_per_layer = config.num_experts * 3 * config.hidden_size * config.intermediate_size + 2 * 2 * (config.n_heads + config.n_kv_heads) * config.hidden_size * config.head_dim
        kv_cache_bytes_per_layer_per_batch = 2 * 0.5 * 2 * config.n_kv_heads * config.max_model_len * config.head_dim
        total_bytes = config.n_layers * (
            weight_bytes_per_layer
            + parallel_state.get_decode_ep_size()
            * config.max_batch_size_per_dp
            * kv_cache_bytes_per_layer_per_batch
        )

        print(f"Decoding tokens per second: {tokens_per_second_per_req:.2f}")
        print(f"Decode MBU: {tokens_per_second_per_req / ((320e9 * dist.get_world_size()) / total_bytes) * 100:.2f}%")


if __name__ == "__main__":
    main()
