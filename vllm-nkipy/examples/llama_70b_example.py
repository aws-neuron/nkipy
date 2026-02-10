#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project

"""
Detailed example showing how to run meta-llama/Llama-3.1-70B-Instruct
using vLLM with the NKIPy backend.

This example demonstrates:
- Setting up tensor parallelism for distributed inference
- Running the model on AWS Neuron hardware
- Processing multi-turn conversations
- Setting appropriate generation parameters

Requirements:
- AWS Trainium instances with 32 NeuronCores
- Dynamo2Neuron + NKIPy installed
- vllm-nkipy installed

To run:
torchrun --nproc-per-node 32 llama_70b_example.py
"""

import os
import time

from vllm import LLM, SamplingParams


def setup_environment():
    """Configure environment variables for NKIPy."""
    # Enable DTensor for tensor parallelism
    os.environ["VLLM_NKIPY_ENABLE_DTENSOR"] = "1"

    # Use NKI optimized attention kernel for improved performance
    os.environ["VLLM_NKIPY_USE_NKI_FA"] = "1"

    # Set up BF16 precision (required for performance)
    os.environ["XLA_USE_BF16"] = "1"


def format_chat_message(messages):
    """Format messages for Llama-3.1 chat completion."""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            formatted += f"<|system|>\n{content}\n"
        elif role == "user":
            formatted += f"<|user|>\n{content}\n"
        elif role == "assistant":
            formatted += f"<|assistant|>\n{content}\n"

    # Add final assistant prompt to generate response
    formatted += "<|assistant|>\n"
    return formatted


def main():
    """Run Llama-3.1-70B-Instruct with vLLM-NKIPy."""
    # Set up environment
    setup_environment()

    # Initialize distributed setup (already done by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("Initializing LLM with NKIPy backend...")

    # Configure model with tensor parallelism
    llm = LLM(
        model="meta-llama/Llama-3.1-70B-Instruct",
        dtype="bfloat16",
        tensor_parallel_size=32,  # Required for 70B model
        enforce_eager=True,  # CUDA graphs not supported
        max_model_len=4096,  # Adjust based on your memory requirements
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        stop=["<|user|>", "<|system|>"],  # Stop tokens
    )

    # Define a multi-turn conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant that provides "
                "accurate, informative responses."
            ),
        },
        {
            "role": "user",
            "content": (
                "Explain how AWS Neuron SDK works for AI inference "
                "and how it integrates with PyTorch."
            ),
        },
    ]

    # Format the conversation for Llama-3.1
    prompt = format_chat_message(messages)

    # Only run on rank 0
    if local_rank == 0:
        print("Prompt:\n", prompt)
        print("\nGenerating response...")

        start_time = time.time()

        # Process the prompt
        outputs = llm.generate(prompt, sampling_params)

        end_time = time.time()

        # Print the results
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"\nGenerated response:\n{generated_text}")
            print(f"\nGeneration took {end_time - start_time:.2f} seconds")

            # Calculate token statistics
            num_input_tokens = len(output.prompt_token_ids)
            num_generated_tokens = len(output.outputs[0].token_ids)
            print(f"Input tokens: {num_input_tokens}")
            print(f"Generated tokens: {num_generated_tokens}")
            tokens_per_sec = num_generated_tokens / (end_time - start_time)
            print(f"Generation speed: {tokens_per_sec:.2f} tokens/sec")

        # Continue the conversation with a follow-up
        messages.append({"role": "assistant", "content": generated_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    "What are the key advantages of using NKIPy "
                    "for LLM inference compared to GPUs?"
                ),
            }
        )

        # Format the updated conversation
        prompt = format_chat_message(messages)
        print("\nFollow-up prompt:\n", prompt)
        print("\nGenerating follow-up response...")

        start_time = time.time()
        outputs = llm.generate(prompt, sampling_params)
        end_time = time.time()

        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"\nGenerated follow-up response:\n{generated_text}")
            print(f"\nGeneration took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
