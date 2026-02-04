# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-based kernel generation using AWS Bedrock."""

import json
import re
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np

from nkipy.tools.kernel_agent.ops import get_all_ops

SYSTEM_PROMPT = """You are a kernel code generator for NKIPy.

Generate a Python function that:
1. Uses ONLY np.* functions from the allowed list
2. Takes numpy arrays as input, returns numpy arrays
3. No loops or conditionals based on array values

Output ONLY a JSON object such as:
{
    "name": "kernel_name",
    "code": "def kernel_name(x):\\n    return np.exp(x)",
    "inputs": {"x": {"shape": [32, 64], "dtype": "float32"}}
}"""


def generate_kernel(
    prompt: str,
    model_id: str,
    region: str,
    allowed_ops: Optional[List[str]] = None,
) -> Tuple[str, str, Dict[str, np.ndarray]]:
    """Generate kernel from prompt. Returns (name, code, inputs)."""
    allowed_ops = allowed_ops or get_all_ops()
    ops_list = ", ".join(sorted(allowed_ops))

    user_prompt = f"""ALLOWED OPS: {ops_list}

REQUEST: {prompt}

Generate a kernel using only allowed operations."""

    client = boto3.client("bedrock-runtime", region_name=region)
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        system=[{"text": SYSTEM_PROMPT}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.7},
    )

    raw = response["output"]["message"]["content"][0]["text"]

    # Parse JSON
    json_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    json_str = json_match.group(1) if json_match else raw.strip()
    data = json.loads(json_str)

    name = data.get("name", "generated_kernel")
    code = data.get("code", "")
    input_specs = data.get("inputs", {})

    # Generate inputs
    inputs = {}
    for inp_name, spec in input_specs.items():
        shape = tuple(spec.get("shape", [32, 32]))
        dtype = spec.get("dtype", "float32")
        size = int(np.prod(shape))
        # TODO: create random input that's appropriate for the dtype
        inputs[inp_name] = np.random.uniform(-1, 1, size).reshape(shape).astype(dtype)

    return name, code, inputs


def compile_code(code: str):
    """Compile code string to callable."""
    namespace = {"np": np}
    exec(code, namespace)
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("_") and name != "np":
            return obj
    return None
