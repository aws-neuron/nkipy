#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project

from vllm import LLM
from vllm.v1.metrics.reader import Counter, Histogram


def main():
    """Simple example demonstrating vLLM with NKIPy backend."""

    print("Initializing LLM with NKIPy backend...")
    # The NKIPy plugin will be automatically detected through the entry point
    llm = LLM(
        # model="meta-llama/Llama-3.1-70B-Instruct",
        # model="meta-llama/Llama-3.1-8B-Instruct",
        # model="meta-llama/Llama-3.2-1B-Instruct",
        model="meta-llama/Llama-3.3-70B-Instruct",
        # model="Qwen/Qwen3-0.6B",
        tensor_parallel_size=64,
        # tensor_parallel_size=8,
        # tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=8192,
        max_num_seqs=32,
        generation_config="auto",
        disable_log_stats=False,
    )

    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 512
    print("Sampling params:", sampling_params)

    # Define prompts
    prompts = [
        "Write a short poem about artificial intelligence.",
        "Write a short poem about artificial intelligence.",
        "Explain quantum computing in simple terms.",
    ]

    print("Generating text...")
    # Process the prompts
    outputs = llm.generate(prompts, sampling_params)

    # Print the results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {generated_text}")
        # print("output:", output)
        print("-" * 50)

    print("Inference complete!")

    for m in llm.get_metrics():
        # e.g. histogram of time to first token
        if isinstance(m, Histogram) and m.name in {
            "vllm:time_to_first_token_seconds",
            "vllm:time_per_output_token_seconds",
            "vllm:e2e_request_latency_seconds",
        }:
            avg = m.sum / m.count
            print(f"{m.name}: avg {avg:.4f}s over {m.count} requests")
        # total tokens generated
        if isinstance(m, Counter) and m.name == "vllm:generation_tokens_total":
            print(f"{m.name}: {m.value}")


if __name__ == "__main__":
    main()
