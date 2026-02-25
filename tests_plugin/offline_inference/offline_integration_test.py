# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Optional

from vllm import LLM, SamplingParams

logger = logging.getLogger("vllm_integration")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def _init_vllm(
    model_name_or_path: str,
    tokenizer: str,
    num_workers: int,
    batch_size: int,
    max_len: int,
    block_size: int,
    enable_prefix_caching: bool,
    dtype: str,
    override_config: Optional[dict],
) -> LLM:
    logger.info(
        "Initializing vLLM %s (num_workers=%d, batch=%d, max_len=%d, block_size=%d, enable_prefix_caching=%s, dtype=%s)",
        model_name_or_path,
        num_workers,
        batch_size,
        max_len,
        block_size,
        enable_prefix_caching,
        dtype,
    )
    return LLM(
        model=model_name_or_path,
        tokenizer=tokenizer,
        trust_remote_code=True,
        dtype=dtype,
        enable_prefix_caching=enable_prefix_caching,
        swap_space=0,
        tensor_parallel_size=num_workers,  # XXX: this makes vLLM launch the specified amount of workers
        max_num_seqs=batch_size,
        max_model_len=max_len,
        block_size=block_size,
        override_neuron_config=override_config or {},
    )


def vllm_integ_test(
    title: str,
    model_name_or_path: str,
    tokenizer: str,
    n_positions: int,
    max_batch_size: int,
    num_workers: int,
    block_size: int,
    enable_prefix_caching: bool,
    dtype: str,
    top_k: int = 1,
    override_neuron_config: Optional[dict] = None,
) -> None:
    logger.info("[%s] Starting vLLM offline inference integration test", title)

    prompts = [
        "The capital of France is",
    ]

    llm = _init_vllm(
        model_name_or_path,
        tokenizer,
        num_workers,
        max_batch_size,
        n_positions,
        block_size,
        enable_prefix_caching,
        dtype,
        override_neuron_config,
    )
    outputs = llm.generate(prompts, SamplingParams(top_k=top_k))

    if top_k == 1:
        # Define expected outputs
        expected_outputs = {"The capital of France is": " Paris."}

        # Validate outputs
        validation_passed = True
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text

            logger.info(f"\n[Prompt]\n{prompt!r}\n[Generated]\n{generated_text!r}\n")

            try:
                assert generated_text.strip() == expected_outputs[prompt].strip(), (
                    f"Output mismatch for prompt '{prompt}'\n[Expected]\n"
                    f"{expected_outputs[prompt]!r}\n[Got]\n{generated_text!r}"
                )
                logger.info(f"[Validation] Prompt '{prompt}' passed\n")
            except AssertionError as e:
                logger.error(f"[Validation] Prompt '{prompt}' failed: {str(e)}\n")
                validation_passed = False

        if validation_passed:
            logger.info(f"[{title}] vLLM offline inference integration test passed")
        else:
            logger.error(f"[{title}] vLLM offline inference integration test failed")
            raise AssertionError("Test failed due to output mismatches")
