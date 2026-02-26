# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-based kernel generation using AWS Bedrock."""

import functools
import json
import re
from importlib.resources import files as _resource_files
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np

DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
DEFAULT_REGION = "us-west-2"


@functools.cache
def load_prompt(filename: str) -> str:
    """Load a prompt text file from the prompts/ subdirectory."""
    return (
        _resource_files("nkipy.tools.kernel_agent")
        .joinpath("prompts", filename)
        .read_text(encoding="utf-8")
    )


def _parse_llm_response(raw: str) -> Tuple[str, str, Dict[str, np.ndarray]]:
    """Parse LLM JSON response into (name, code, inputs).

    Extracts JSON from markdown code fences or raw text, then generates
    random numpy arrays matching the declared input specs.
    """
    json_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    json_str = json_match.group(1) if json_match else raw.strip()
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse LLM response as JSON: {e}. "
            f"Raw response (truncated): {raw[:200]}"
        ) from e

    name = data.get("name", "generated_kernel")
    code = data.get("code", "")
    input_specs = data.get("inputs", {})

    inputs = {}
    for inp_name, spec in input_specs.items():
        shape = tuple(spec.get("shape", [32, 32]))
        dtype = spec.get("dtype", "float32")
        np_dtype = getattr(np, dtype, np.float32)
        if np.issubdtype(np_dtype, np.integer):
            info = np.iinfo(np_dtype)
            lo = max(info.min, -100)
            hi = min(info.max, 100)
            inputs[inp_name] = np.random.randint(lo, hi + 1, size=shape).astype(dtype)
        else:
            size = int(np.prod(shape))
            inputs[inp_name] = (
                np.random.uniform(-1, 1, size).reshape(shape).astype(dtype)
            )

    return name, code, inputs


def generate_kernel(
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    region: str = DEFAULT_REGION,
    *,
    constrained: bool = False,
    allowed_ops: Optional[List[str]] = None,
) -> Tuple[str, str, Dict[str, np.ndarray]]:
    """Generate a kernel from a natural-language prompt via AWS Bedrock.

    Args:
        prompt: Natural language description of the kernel.
        model_id: Bedrock model identifier.
        region: AWS region for the Bedrock client.
        constrained: If True, restrict generation to allowed_ops and use
            the constrained system prompt.
        allowed_ops: Explicit op allowlist. Defaults to get_all_ops() when
            constrained=True. Ignored when constrained=False.

    Returns:
        (name, code, inputs) tuple.
    """
    if constrained:
        from nkipy.tools.kernel_agent.ops import get_all_ops

        system_prompt = load_prompt("system_constrained.txt")
        ops = allowed_ops or get_all_ops()
        ops_list = ", ".join(sorted(ops))
        user_prompt = (
            f"ALLOWED OPS: {ops_list}\n\n"
            f"REQUEST: {prompt}\n\n"
            f"Generate a kernel using only allowed operations."
        )
    else:
        system_prompt = load_prompt("system_unconstrained.txt")
        user_prompt = (
            f"REQUEST: {prompt}\n\nGenerate a numpy kernel function for this operation."
        )

    client = boto3.client("bedrock-runtime", region_name=region)
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        system=[{"text": system_prompt}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.7},
    )

    raw = response["output"]["message"]["content"][0]["text"]
    return _parse_llm_response(raw)


def compile_code(code: str):
    """Compile code string to callable."""
    namespace = {"np": np}
    exec(code, namespace)
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("_") and name != "np":
            return obj
    return None
