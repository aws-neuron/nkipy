# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project

import ast
import hashlib
import os
from typing import Any, Callable, Optional


def _parse_list_env(env_name: str) -> Optional[list[str]]:
    """Parse an environment variable containing a Python list literal.

    Expected format: '["item1", "item2", "item3"]'
    Returns a list of strings or None if not set or invalid.

    Args:
        env_name: Name of the environment variable to parse.

    Returns:
        List of strings if valid, None otherwise.
    """
    env_value = os.environ.get(env_name)
    if env_value is None:
        return None

    try:
        parsed = ast.literal_eval(env_value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
        return None
    except (ValueError, SyntaxError):
        return None


def _parse_int_list_env(env_name: str, default: list[int]) -> list[int]:
    """Parse an environment variable containing a Python list of integers.

    Expected format: '[0, 10, 20]' or '[0]'
    Returns a list of integers or the default if not set or invalid.

    Args:
        env_name: Name of the environment variable to parse.
        default: Default value if not set or invalid.

    Returns:
        List of integers if valid, default otherwise.
    """
    env_value = os.environ.get(env_name)
    if env_value is None:
        return default

    try:
        parsed = ast.literal_eval(env_value)
        if isinstance(parsed, list):
            return [int(item) for item in parsed]
        return default
    except (ValueError, SyntaxError):
        return default


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    # Cache directory prefix for NKIPy compiled artifacts and caching.
    # When set, vLLM-NKIPy will use this prefix followed by "_{rank}"
    # for caching compiled artifacts. If not set, defaults to "./nkipy_cache"
    # resulting in cache directories like "./nkipy_cache_{rank}".
    "VLLM_NKIPY_CACHE_DIR": lambda: os.environ.get("VLLM_NKIPY_CACHE_DIR") or None,
    # ============== MoE Environment Variables ==============
    # Use NKI-based MoE implementation instead of reference implementation.
    # When enabled (1), uses optimized NKI kernels for MoE computation.
    # When disabled (0), uses CPU/reference implementation for debugging.
    "VLLM_NKIPY_MOE_USE_NKI": lambda: int(os.getenv("VLLM_NKIPY_MOE_USE_NKI", "1")),
    # Debug: Truncate token processing at this count (0 = no truncation).
    "VLLM_NKIPY_MOE_TRUNC": lambda: int(os.getenv("VLLM_NKIPY_MOE_TRUNC", "0")),
    # Matrix multiplication mode for reference implementation.
    # Options: "sequential" (default), "batch"
    "VLLM_NKIPY_MOE_MM_MODE": lambda: os.getenv("VLLM_NKIPY_MOE_MM_MODE", "sequential"),
    # Use transposed weight matrices in reference implementation (0 or 1).
    "VLLM_NKIPY_MOE_TRANSPOSE": lambda: int(os.getenv("VLLM_NKIPY_MOE_TRANSPOSE", "0")),
    # Enable 2D parallelism (Expert Parallel + Tensor Parallel) (0 or 1).
    "VLLM_NKIPY_MOE_2D": lambda: int(os.getenv("VLLM_NKIPY_MOE_2D", "0")),
    # Blockwise implementation choice for NKI MoE.
    # Options: "nki" (default), "torch"
    "VLLM_NKIPY_MOE_BLOCKWISE_IMPL": lambda: os.getenv(
        "VLLM_NKIPY_MOE_BLOCKWISE_IMPL", "nki"
    ),
    # Use FP8 quantization for MoE weights (0 or 1).
    "VLLM_NKIPY_MOE_FP8": lambda: int(os.getenv("VLLM_NKIPY_MOE_FP8", "1")),
    # ============== Compile Environment Variables ==============
    # (MERGE_LAYERS strategy only)
    # These variables almost only apply when
    # compile_strategy="merge_layers" in NKIPyConfig.
    # They control the local_compile decorator used for layer-wise compilation.
    # Use NKIPy backend (1) or CPU fallback (0) for torch.compile.
    # This is a quick debugging/fallback switch for the compilation backend.
    "VLLM_NKIPY_COMPILE_USE_NKIPY": lambda: int(
        os.getenv("VLLM_NKIPY_COMPILE_USE_NKIPY", "1")
    ),
    # Enable/disable torch.compile for local compilation (1=enabled, 0=disabled).
    # Only used with CompileStrategy.MERGE_LAYERS.
    # When disabled, decorated functions are returned unchanged.
    "VLLM_NKIPY_ENABLE_COMPILE": lambda: int(
        os.getenv("VLLM_NKIPY_ENABLE_COMPILE", "1")
    ),
    # List of module names to selectively compile with torch.compile.
    # Only used with CompileStrategy.MERGE_LAYERS.
    # Format: Python list literal, e.g., '["OAIAttention", "MLPBlock", "RMSNorm"]'
    # When set, only modules whose names are in this list will be compiled.
    # When not set (None), all decorated modules are compiled.
    "VLLM_NKIPY_COMPILE_MODULES": lambda: _parse_list_env(
        "VLLM_NKIPY_COMPILE_MODULES"
    ),
    # Device to transfer output tensors to after compiled function execution.
    # Only used with CompileStrategy.MERGE_LAYERS.
    # Example: "cpu" or "cuda:0". When not set, outputs stay on their original device.
    "VLLM_NKIPY_COMPILE_OUTPUT_DEVICE": lambda: os.getenv(
        "VLLM_NKIPY_COMPILE_OUTPUT_DEVICE"
    ),
    # List of module names to save input/output tensors for debugging.
    # Only used with CompileStrategy.MERGE_LAYERS.
    # Format: Python list literal, e.g., '["OAIAttention", "MLPBlock"]'
    # When set, inputs and outputs of matching modules are saved to disk.
    "VLLM_NKIPY_COMPILE_SAVE_IO": lambda: _parse_list_env(
        "VLLM_NKIPY_COMPILE_SAVE_IO"
    ),
    # Debug mode for compiled modules.
    # Only used with CompileStrategy.MERGE_LAYERS.
    # Options: "no" (default, disabled), "layer" (enables layer-wise tensor debugging)
    # When "layer", intermediate tensors are logged and saved during execution.
    "VLLM_NKIPY_COMPILE_DEBUG": lambda: os.getenv("VLLM_NKIPY_COMPILE_DEBUG", "no"),
    # Directory for saving debug tensors from compiled modules.
    # Only used with CompileStrategy.MERGE_LAYERS when VLLM_NKIPY_COMPILE_DEBUG="layer".
    # Tensors are saved to {dir}/rank_{rank}/ subdirectories.
    "VLLM_NKIPY_COMPILE_DEBUG_DIR": lambda: os.getenv(
        "VLLM_NKIPY_COMPILE_DEBUG_DIR", "./debug_tensors"
    ),
    # Rank(s) to enable debugging for.
    # Only used with CompileStrategy.MERGE_LAYERS when VLLM_NKIPY_COMPILE_DEBUG="layer".
    # Options: "0" (default), "1", "2", ... or "all" for all ranks.
    "VLLM_NKIPY_COMPILE_DEBUG_RANK": lambda: os.getenv(
        "VLLM_NKIPY_COMPILE_DEBUG_RANK", "0"
    ),
    # Enable NTFF (Neuron Tensor File Format) artifact saving (1=enabled, 0=disabled).
    # Only used with CompileStrategy.MERGE_LAYERS.
    # When enabled, compiled graph artifacts are saved for profiling/debugging.
    "VLLM_NKIPY_COMPILE_SAVE_NTFF": lambda: int(
        os.getenv("VLLM_NKIPY_COMPILE_SAVE_NTFF", "0")
    ),
    # Directory for saving NTFF artifacts.
    # Only used with CompileStrategy.MERGE_LAYERS when VLLM_NKIPY_COMPILE_SAVE_NTFF=1.
    "VLLM_NKIPY_COMPILE_SAVE_NTFF_DIR": lambda: os.getenv(
        "VLLM_NKIPY_COMPILE_SAVE_NTFF_DIR", "./ntff_dir"
    ),
    # Execution indices for NTFF saving.
    # Only used with CompileStrategy.MERGE_LAYERS when VLLM_NKIPY_COMPILE_SAVE_NTFF=1.
    # Format: Python list literal, e.g., '[0]' or '[0, 10, 20]'
    # Specifies which execution indices should have their NTFF artifacts saved.
    "VLLM_NKIPY_COMPILE_SAVE_NTFF_EXE_IDX": lambda: _parse_int_list_env(
        "VLLM_NKIPY_COMPILE_SAVE_NTFF_EXE_IDX", [0]
    ),
    # ============== Debug Environment Variables ==============
    # Rank to enable debugpy remote debugging for.
    # When set to a rank number (e.g., "0"), enables debugpy listener on port 5678
    # for that specific rank, allowing remote debugging attachment.
    # When not set (None), debugpy is disabled.
    "VLLM_NKIPY_DEBUGPY_RANK": lambda: os.environ.get("VLLM_NKIPY_DEBUGPY_RANK"),
    # ============== Neuron Runtime Inspection Variables ==============
    # Enable Neuron runtime inspection for profiling/debugging.
    # When enabled (1), sets NEURON_RT_INSPECT_ENABLE=1 on master rank.
    # When disabled (0, default), no inspection overhead.
    "VLLM_NKIPY_RT_INSPECT_ENABLE": lambda: int(
        os.getenv("VLLM_NKIPY_RT_INSPECT_ENABLE", "0")
    ),
    # Output directory for Neuron runtime inspection data.
    # Only used when VLLM_NKIPY_RT_INSPECT_ENABLE=1.
    # Default: "./output"
    "VLLM_NKIPY_RT_INSPECT_OUTPUT_DIR": lambda: os.getenv(
        "VLLM_NKIPY_RT_INSPECT_OUTPUT_DIR", "./output"
    ),
}

# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def compute_hash() -> str:
    """
    Compute a hash of environment variables that affect the computation graph.

    WARNING: Whenever a new key is added to this environment variables dict,
    ensure that it is included in the factors list if it affects the
    computation graph. Different values of these variables will generate
    different computation graphs.
    """
    factors: list[Any] = []

    # summarize environment variables
    def factorize(name: str):
        if __getattr__(name):
            factors.append(__getattr__(name))
        else:
            factors.append("None")

    # The values of envs may affect the computation graph.
    environment_variables_to_hash = [
        "VLLM_NKIPY_CACHE_DIR",
    ]
    for key in environment_variables_to_hash:
        if key in environment_variables:
            factorize(key)

    hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()

    return hash_str
