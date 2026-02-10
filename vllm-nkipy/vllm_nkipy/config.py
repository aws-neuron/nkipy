# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""NKIPy-specific configuration for vLLM.

This module defines configuration dataclasses that can be passed to vLLM via
the --additional-config CLI argument.

Usage:
    python -m vllm.entrypoints.openai.api_server \\
        --additional-config '{"compile_strategy": "full_graph", "split_graph": true}'
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

# Default values
DEFAULT_MIN_NUM_SEQS = 4  # Smallest output size for request padding
DEFAULT_LAYER_BARRIERS_GROUP_SIZE = 4  # Insert barriers every N layers
DEFAULT_MERGE_STEP = 4  # Merge N layers when compile_strategy="merge_layers"

# NKI blocksparse attention defaults (only effective with nki_blocksparse_* impl)
DEFAULT_LARGE_KV_TILE_SIZE = 1024  # KV tile size for NKI attention
DEFAULT_LARGE_Q_TILE_SIZE = 128  # Query tile size for NKI attention
DEFAULT_DYNAMIC_LOOP_UNROLL = 8  # Loop unrolling factor


class CompileStrategy(str, Enum):
    """Compilation strategy for the model.

    Attributes:
        FULL_FX_GRAPH: torch.compile the entire model as one FX graph.
        MERGE_LAYERS: torch.compile every N layers (controlled by merge_step).
    """

    FULL_FX_GRAPH = "full_fx_graph"
    MERGE_LAYERS = "merge_layers"


class PagedAttnImpl(str, Enum):
    """Paged attention implementation.

    Attributes:
        NKI_RAGGED: NKI-based ragged (variable-length) attention.
        NKI_MASKED: NKI-based masked attention.
        NKI_BLOCKSPARSE_FLASH_ATTENTION: NKI blocksparse flash attention.
        NKI_BLOCKSPARSE_FLASH_ATTENTION_KV_CACHE: NKI blocksparse
            flash attention with KV cache.
        TORCH_RAGGED: PyTorch-based ragged attention (for debugging).
        TORCH_MASKED: PyTorch-based masked attention (for debugging).
    """

    NKI_RAGGED = "nki_ragged"
    NKI_MASKED = "nki_masked"
    NKI_BLOCKSPARSE_FLASH_ATTENTION = "nki_blocksparse_flash_attention"
    NKI_BLOCKSPARSE_FLASH_ATTENTION_KV_CACHE = (
        "nki_blocksparse_flash_attention_kv_cache"
    )
    TORCH_RAGGED = "torch_ragged"
    TORCH_MASKED = "torch_masked"


class PagedKvImpl(str, Enum):
    """KV cache update implementation.

    Attributes:
        RESHAPE_AND_CACHE2: NKI-based reshape and cache (default).
        RESHAPE_AND_CACHE_KV1: Alternative NKI reshape and cache.
        UPDATE_KV_CACHE_CUSTOM_OP: Custom op for KV cache update.
        SKIP: Skip KV cache update (for debugging).
    """

    RESHAPE_AND_CACHE2 = "reshape_and_cache2"
    RESHAPE_AND_CACHE_KV1 = "reshape_and_cache_kv1"
    UPDATE_KV_CACHE_CUSTOM_OP = "update_kv_cache_custom_op"
    SKIP = "skip"


@dataclass
class NKIPyConfig:
    """NKIPy-specific configuration passed via --additional-config.

    This config is used by the NKIPy platform to configure compilation
    and runtime behavior.

    Attributes:
        num_tokens_paddings: List of token padding sizes for compilation.
            The max value will be used as max_num_batched_tokens.
            Example: [1, 4, 32, 128] means the model will be compiled for
            these specific batch sizes, with 128 being the maximum.
        min_num_seqs: Minimum number of sequences for padding. This is the
            smallest output size used to avoid dynamic shapes. Default: 4.
        compile_strategy: How to compile the model. Options:
            - "full_fx_graph": torch.compile the entire model as one FX graph.
            - "merge_layers": torch.compile every N layers (see merge_step).
            Default: "full_fx_graph".
        merge_step: Number of layers to merge when compile_strategy="merge_layers".
            Default: 4.
        split_graph: Whether to enable graph splitting. When True, the backend
            will automatically split the graph based on subgraph markers.
            Default: True.
        layer_barriers_group_size: Group size for layer barriers. When > 0,
            barrier ops are inserted every N layers to synchronize execution.
            Only effective when split_graph=True. Default: 4.
    """

    num_tokens_paddings: Optional[list[int]] = None
    min_num_seqs: int = DEFAULT_MIN_NUM_SEQS
    compile_strategy: CompileStrategy = CompileStrategy.FULL_FX_GRAPH
    merge_step: int = DEFAULT_MERGE_STEP
    split_graph: bool = True
    layer_barriers_group_size: int = DEFAULT_LAYER_BARRIERS_GROUP_SIZE
    # Attention implementation selection
    paged_attn_impl: PagedAttnImpl = PagedAttnImpl.NKI_RAGGED
    paged_kv_impl: PagedKvImpl = PagedKvImpl.RESHAPE_AND_CACHE2
    # NKI blocksparse attention parameters (only effective with nki_blocksparse_* impl)
    large_kv_tile_size: int = DEFAULT_LARGE_KV_TILE_SIZE
    large_q_tile_size: int = DEFAULT_LARGE_Q_TILE_SIZE
    dynamic_loop_unroll: int = DEFAULT_DYNAMIC_LOOP_UNROLL

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If configuration has invalid combinations.
        """
        # Convert string to enum if needed (for JSON deserialization)
        if isinstance(self.compile_strategy, str):
            self.compile_strategy = CompileStrategy(self.compile_strategy)
        if isinstance(self.paged_attn_impl, str):
            self.paged_attn_impl = PagedAttnImpl(self.paged_attn_impl)
        if isinstance(self.paged_kv_impl, str):
            self.paged_kv_impl = PagedKvImpl(self.paged_kv_impl)

        # MERGE_LAYERS + split_graph is an invalid combination
        if self.compile_strategy == CompileStrategy.MERGE_LAYERS and self.split_graph:
            raise ValueError(
                "compile_strategy='merge_layers' cannot be used with split_graph=True. "
                "The merge_layers strategy handles compilation per N layers itself."
            )

        # merge_step must be > 0 when using merge_layers strategy
        if (
            self.compile_strategy == CompileStrategy.MERGE_LAYERS
            and self.merge_step <= 0
        ):
            raise ValueError(
                "merge_step must be > 0 when compile_strategy='merge_layers'. "
                f"Got merge_step={self.merge_step}"
            )

    def compute_hash(self) -> str:
        """Compute a hash of the configuration for caching purposes."""
        factors: list[Any] = [
            self.num_tokens_paddings,
            self.min_num_seqs,
            self.compile_strategy.value,
            self.merge_step,
            self.split_graph,
            self.layer_barriers_group_size,
            self.paged_attn_impl.value,
            self.paged_kv_impl.value,
        ]
        return hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NKIPyConfig":
        """Create a NKIPyConfig from a dictionary.

        Args:
            d: Dictionary containing configuration values.

        Returns:
            NKIPyConfig instance with values from the dictionary.
        """
        compile_strategy = d.get("compile_strategy", CompileStrategy.FULL_FX_GRAPH)
        if isinstance(compile_strategy, str):
            compile_strategy = CompileStrategy(compile_strategy)

        paged_attn_impl = d.get("paged_attn_impl", PagedAttnImpl.NKI_RAGGED)
        if isinstance(paged_attn_impl, str):
            paged_attn_impl = PagedAttnImpl(paged_attn_impl)

        paged_kv_impl = d.get("paged_kv_impl", PagedKvImpl.RESHAPE_AND_CACHE2)
        if isinstance(paged_kv_impl, str):
            paged_kv_impl = PagedKvImpl(paged_kv_impl)

        return cls(
            num_tokens_paddings=d.get("num_tokens_paddings"),
            min_num_seqs=d.get("min_num_seqs", DEFAULT_MIN_NUM_SEQS),
            compile_strategy=compile_strategy,
            merge_step=d.get("merge_step", DEFAULT_MERGE_STEP),
            split_graph=d.get("split_graph", True),
            layer_barriers_group_size=d.get(
                "layer_barriers_group_size", DEFAULT_LAYER_BARRIERS_GROUP_SIZE
            ),
            paged_attn_impl=paged_attn_impl,
            paged_kv_impl=paged_kv_impl,
            large_kv_tile_size=d.get("large_kv_tile_size", DEFAULT_LARGE_KV_TILE_SIZE),
            large_q_tile_size=d.get("large_q_tile_size", DEFAULT_LARGE_Q_TILE_SIZE),
            dynamic_loop_unroll=d.get(
                "dynamic_loop_unroll", DEFAULT_DYNAMIC_LOOP_UNROLL
            ),
        )

    def get_max_num_batched_tokens(self) -> Optional[int]:
        """Get the maximum number of batched tokens from num_tokens_paddings.

        Returns:
            max(num_tokens_paddings), or None if not set.
        """
        if self.num_tokens_paddings:
            return max(self.num_tokens_paddings)
        return None


# Global config instance for module-level access
_global_nkipy_config: Optional[NKIPyConfig] = None


def set_global_nkipy_config(config: NKIPyConfig) -> None:
    """Set the global NKIPy configuration.

    This should be called during worker initialization to make the config
    available to components that don't have direct access to vllm_config.

    Args:
        config: The NKIPyConfig instance to set globally.
    """
    global _global_nkipy_config
    _global_nkipy_config = config


def get_global_nkipy_config() -> NKIPyConfig:
    """Get the global NKIPy configuration.

    Returns:
        The global NKIPyConfig instance, or a default config if not set.
    """
    if _global_nkipy_config is None:
        return NKIPyConfig()
    return _global_nkipy_config


def _get_paged_attn_impl() -> PagedAttnImpl:
    """Get the paged attention implementation from global config.

    Returns:
        The PagedAttnImpl enum value.
    """
    return get_global_nkipy_config().paged_attn_impl


def _get_paged_kv_impl() -> PagedKvImpl:
    """Get the paged KV cache implementation from global config.

    Returns:
        The PagedKvImpl enum value.
    """
    return get_global_nkipy_config().paged_kv_impl


def get_nkipy_config(vllm_config: Any) -> NKIPyConfig:
    """Extract NKIPyConfig from vLLM's additional_config.

    Args:
        vllm_config: The VllmConfig instance.

    Returns:
        NKIPyConfig instance parsed from additional_config.
    """
    additional_config = getattr(vllm_config, "additional_config", None)
    if additional_config is None:
        return NKIPyConfig()
    if isinstance(additional_config, dict):
        return NKIPyConfig.from_dict(additional_config)
    if isinstance(additional_config, NKIPyConfig):
        return additional_config
    return NKIPyConfig()
