# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-based kernel generation using AWS Bedrock."""

import functools
import os
from importlib.resources import files as _resource_files
from typing import Dict, List, Optional, Tuple

import boto3
import ml_dtypes  # noqa: F401 — registers bfloat16 etc. with numpy
import numpy as np

MODEL_ID_ENV = "NKIPY_KERNEL_AGENT_MODEL"
DEFAULT_REGION = "us-west-2"


def resolve_model_id(model_id: Optional[str] = None) -> str:
    """Return the Bedrock model ID to use.

    An explicit ``model_id`` wins; otherwise the ``NKIPY_KERNEL_AGENT_MODEL``
    environment variable is consulted. There is no built-in default.
    """
    if model_id:
        return model_id
    env_value = os.environ.get(MODEL_ID_ENV, "").strip()
    if env_value:
        return env_value
    raise ValueError(
        "No Bedrock model ID configured. Pass --model / model_id or set the "
        f"{MODEL_ID_ENV} environment variable."
    )

_KERNEL_TOOL = {
    "toolSpec": {
        "name": "submit_kernel",
        "description": (
            "Submit the generated kernel code, its name, and input specifications."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Function name"},
                    "code": {
                        "type": "string",
                        "description": "Python source code of the kernel function",
                    },
                    "inputs": {
                        "type": "object",
                        "description": ("Map of input name to {shape, dtype} spec"),
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "shape": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                "dtype": {"type": "string"},
                            },
                            "required": ["shape", "dtype"],
                        },
                    },
                },
                "required": ["name", "code", "inputs"],
            }
        },
    }
}


@functools.cache
def load_prompt(filename: str) -> str:
    """Load a prompt text file from the prompts/ subdirectory."""
    return (
        _resource_files("nkipy.tools.kernel_agent")
        .joinpath("prompts", filename)
        .read_text(encoding="utf-8")
    )


def build_inputs(input_specs: Dict) -> Dict[str, np.ndarray]:
    """Generate random numpy arrays matching declared input specs.

    Args:
        input_specs: Map of input name to ``{"shape": [...], "dtype": "..."}``
            spec dicts.

    Returns:
        Dict mapping input names to numpy arrays.
    """
    inputs = {}
    for inp_name, spec in input_specs.items():
        shape = tuple(spec["shape"])
        np_dtype = np.dtype(spec["dtype"])
        if np.issubdtype(np_dtype, np.integer):
            info = np.iinfo(np_dtype)
            lo = max(info.min, -100)
            hi = min(info.max, 100)
            inputs[inp_name] = np.random.randint(lo, hi + 1, size=shape).astype(
                np_dtype
            )
        else:
            size = int(np.prod(shape))
            inputs[inp_name] = (
                np.random.uniform(-1, 1, size).reshape(shape).astype(np_dtype)
            )
    return inputs


def generate_kernel(
    prompt: str,
    model_id: Optional[str] = None,
    region: str = DEFAULT_REGION,
    *,
    constrained: bool = False,
    allowed_ops: Optional[List[str]] = None,
) -> Tuple[str, str, Dict[str, np.ndarray]]:
    """Generate a kernel from a natural-language prompt via AWS Bedrock.

    Args:
        prompt: Natural language description of the kernel.
        model_id: Bedrock model identifier. Defaults to the
            NKIPY_KERNEL_AGENT_MODEL environment variable.
        region: AWS region for the Bedrock client.
        constrained: If True, restrict generation to allowed_ops and use
            the constrained system prompt.
        allowed_ops: Explicit op allowlist. Defaults to get_all_ops() when
            constrained=True. Ignored when constrained=False.

    Returns:
        (name, code, inputs) tuple.
    """
    model_id = resolve_model_id(model_id)
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
        toolConfig={"tools": [_KERNEL_TOOL]},
    )

    content_blocks = response["output"]["message"]["content"]

    for block in content_blocks:
        if "toolUse" in block:
            tool_input = block["toolUse"]["input"]
            name = tool_input["name"]
            code = tool_input["code"]
            return name, code, build_inputs(tool_input["inputs"])

    raise ValueError(
        "Model did not return a toolUse response block. "
        f"Got: {[list(b.keys()) for b in content_blocks]}"
    )


def compile_code(code: str):
    """Compile code string to callable."""
    namespace = {"np": np}
    exec(code, namespace)
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("_") and name != "np":
            return obj
    return None
