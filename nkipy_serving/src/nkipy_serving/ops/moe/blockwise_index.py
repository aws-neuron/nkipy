"""CPU utilities for blockwise MoE scheduling.

This module implements the "blockwise index" construction used by the NKI MoE
kernel: given per-token top-k expert indices, build:
  - block_to_expert: [num_blocks] int8, mapping each block to its expert id
  - token_position_to_id: [num_blocks, block_size] int32, mapping each position
    in a block to a token id (or SKIP_DMA for padding)

The NKI blockwise kernel consumes these arrays to stream tokens for one expert
block at a time.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import math
import os
import subprocess
import sysconfig
import tempfile
from enum import Enum
from pathlib import Path

import numpy as np

# Kernel tile size (must match the NKI kernel's TILE_SIZE).
BLOCK_SIZE = 128


class ControlType(Enum):
    SKIP_DMA = -1
    SKIP_BLOCK = -2


_COMPILED_IMPL = None
_COMPILED_IMPL_LOAD_ERROR: str | None = None


def _source_hash() -> str:
    src_path = Path(__file__).with_name("blockwise_index_ext.c")
    digest = hashlib.sha256()
    digest.update(src_path.read_bytes())
    soabi = sysconfig.get_config_var("SOABI")
    if isinstance(soabi, str):
        digest.update(soabi.encode("utf-8"))
    else:
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if isinstance(ext_suffix, str):
            digest.update(ext_suffix.encode("utf-8"))
        else:
            digest.update(b"no-soabi")
    digest.update(np.__version__.encode("utf-8"))
    return digest.hexdigest()[:16]


def _compiled_module_name() -> str:
    return "blockwise_index_ext"


def _compiled_module_dir() -> Path:
    return (
        Path(tempfile.gettempdir())
        / "nkipy_serving_native"
        / "blockwise_index"
        / _source_hash()
    )


def _compile_native_extension(target: Path) -> None:
    src_path = Path(__file__).with_name("blockwise_index_ext.c")
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    cc = os.getenv("CC", "gcc")
    include_dirs = [np.get_include()]
    for key in ("include", "platinclude"):
        inc = sysconfig.get_paths().get(key)
        if inc:
            include_dirs.append(inc)
    cmd = [
        cc,
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        f"-I{include_dirs[0]}",
        *[f"-I{inc}" for inc in include_dirs[1:]],
        str(src_path),
        "-o",
        str(tmp_target),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    os.replace(tmp_target, target)


def _load_compiled_impl():
    global _COMPILED_IMPL, _COMPILED_IMPL_LOAD_ERROR
    if _COMPILED_IMPL is not None:
        return _COMPILED_IMPL

    module_name = _compiled_module_name()
    module_dir = _compiled_module_dir()
    module_dir.mkdir(parents=True, exist_ok=True)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        _COMPILED_IMPL_LOAD_ERROR = "python EXT_SUFFIX unavailable"
        return None
    target = module_dir / f"{module_name}{ext_suffix}"
    lock_path = module_dir / f"{module_name}.lock"

    try:
        with lock_path.open("w", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            if not target.exists():
                _compile_native_extension(target)
        spec = importlib.util.spec_from_file_location(module_name, target)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load native module spec from {target}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _COMPILED_IMPL = module
        return module
    except Exception as exc:
        _COMPILED_IMPL_LOAD_ERROR = repr(exc)
        return None


def _python_get_blockwise_expert_and_token_mapping(
    *,
    top_k_indices: np.ndarray,
    num_blocks: int,
    block_size: int,
    num_experts: int,
    num_static_blocks: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Python reference implementation used for fallback and testing."""
    if top_k_indices.ndim != 2:
        raise RuntimeError(
            f"top_k_indices must be rank-2, got shape={top_k_indices.shape}"
        )

    top_k_indices = np.asarray(top_k_indices, dtype=np.int32)

    tokens_per_expert = np.bincount(
        top_k_indices[top_k_indices != ControlType.SKIP_DMA.value].flatten(),
        minlength=int(num_experts),
    )
    blocks_per_expert = np.ceil(
        tokens_per_expert.astype(np.float32) / float(block_size)
    ).astype(np.uint32)
    cumulative_blocks_per_expert = np.cumsum(blocks_per_expert, axis=0, dtype=np.uint32)
    num_real_blocks = (
        int(cumulative_blocks_per_expert[-1])
        if cumulative_blocks_per_expert.size
        else 0
    )
    if num_real_blocks > int(num_blocks):
        raise RuntimeError(
            f"num_real_blocks exceeds num_blocks: {num_real_blocks} > {num_blocks}"
        )

    block_to_expert = np.full(
        (int(num_blocks),), ControlType.SKIP_BLOCK.value, dtype=np.int8
    )
    if num_real_blocks > 0:
        block_to_expert[:num_real_blocks] = np.repeat(
            np.arange(num_experts, dtype=np.int32), blocks_per_expert
        ).astype(np.int8)

    token_position_to_id = np.full(
        (int(num_blocks), int(block_size)), ControlType.SKIP_DMA.value, dtype=np.int32
    )

    current_block_idx = np.zeros((int(num_experts),), dtype=np.int32)
    if num_experts > 1:
        current_block_idx[1:] = cumulative_blocks_per_expert[:-1]
    current_token_id_in_block = np.zeros((int(num_experts),), dtype=np.int32)

    for t in range(int(top_k_indices.shape[0])):
        tk = top_k_indices[t]
        for k in range(int(tk.shape[0])):
            expert_id = int(tk[k])
            if expert_id == ControlType.SKIP_DMA.value:
                continue
            token_position_to_id[
                current_block_idx[expert_id], current_token_id_in_block[expert_id]
            ] = t
            current_token_id_in_block[expert_id] += 1
            if current_token_id_in_block[expert_id] == int(block_size):
                current_block_idx[expert_id] += 1
                current_token_id_in_block[expert_id] = 0

    for i in range(num_real_blocks - 1, 0, -1):
        if int(block_to_expert[i]) == int(block_to_expert[i - 1]) and i != int(
            num_static_blocks
        ):
            block_to_expert[i] = ControlType.SKIP_DMA.value

    return num_real_blocks, block_to_expert, token_position_to_id


def using_compiled_impl() -> bool:
    return _load_compiled_impl() is not None


def preload_compiled_impl() -> bool:
    """Eagerly load or build the native helper for deterministic startup latency."""
    return using_compiled_impl()


def get_n_blocks(num_tokens: int, top_k: int, num_experts: int) -> tuple[int, int]:
    """Return (num_blocks, num_static_blocks) for a fixed-shape launch.

    This matches the NeuronPyExps heuristic:
      ceil((T*TOPK - (E-1)) / BLOCK_SIZE) + (E-1)
    """
    if num_tokens < 0:
        raise RuntimeError(f"num_tokens must be >= 0, got {num_tokens}")
    if top_k <= 0:
        raise RuntimeError(f"top_k must be > 0, got {top_k}")
    if num_experts <= 0:
        raise RuntimeError(f"num_experts must be > 0, got {num_experts}")
    num_static_blocks = math.ceil(
        (num_tokens * top_k - (num_experts - 1)) / BLOCK_SIZE
    ) + (num_experts - 1)
    # This implementation currently uses only static blocks.
    return int(num_static_blocks), int(num_static_blocks)


def get_blockwise_expert_and_token_mapping(
    *,
    top_k_indices: np.ndarray,
    num_blocks: int,
    block_size: int,
    num_experts: int,
    num_static_blocks: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Build blockwise expert/token mapping from [T, top_k] expert indices.

    Args:
        top_k_indices: int array of shape [T, top_k]. Tokens outside the real
            range should already be masked to ControlType.SKIP_DMA.value.
        num_blocks: fixed number of blocks for the kernel launch.
        block_size: tokens per block (BLOCK_SIZE).
        num_experts: number of experts in this MoE partition.
        num_static_blocks: index at which SKIP_DMA behavior is enabled for
            consecutive expert blocks (see below).
    """
    compiled = _load_compiled_impl()
    if compiled is not None:
        num_real_blocks, block_to_expert, token_position_to_id = (
            compiled.get_blockwise_expert_and_token_mapping(
                top_k_indices=top_k_indices,
                num_blocks=int(num_blocks),
                block_size=int(block_size),
                num_experts=int(num_experts),
                num_static_blocks=int(num_static_blocks),
            )
        )
        return (
            int(num_real_blocks),
            np.asarray(block_to_expert),
            np.asarray(token_position_to_id),
        )

    return _python_get_blockwise_expert_and_token_mapping(
        top_k_indices=top_k_indices,
        num_blocks=num_blocks,
        block_size=block_size,
        num_experts=num_experts,
        num_static_blocks=num_static_blocks,
    )
