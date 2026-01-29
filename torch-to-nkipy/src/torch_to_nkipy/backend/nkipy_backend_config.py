# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration management for NKIPy backend."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class NKIPyBackendConfig:
    """Global configuration class for NKIPy backend.

    Attributes:
        nkipy_cache_prefix: Directory path for cache storage of all ranks
        nkipy_cache: Directory path for cache storage of the current rank
        log_level: Logging level
        rank: Process rank for distributed setup
        world_size: Total number of processes
        additional_compiler_args: Custom compiler flags
    """

    nkipy_cache_prefix: str
    nkipy_cache: str = field(init=False)
    log_level: int
    rank: int
    world_size: int
    additional_compiler_args: str

    def __post_init__(self):
        object.__setattr__(
            self, "nkipy_cache", self.nkipy_cache_prefix + f"/rank_{self.rank}"
        )

    def __repr__(self) -> str:
        """Return string representation of the config."""
        return (
            f"\nNKIPyConfig:\n"
            f"  nkipy_cache_prefix: {self.nkipy_cache_prefix}\n"
            f"  nkipy_cache: {self.nkipy_cache}\n"
            f"  log_level: {self.log_level}\n"
            f"  rank: {self.rank}\n"
            f"  world_size: {self.world_size}\n"
            f"  additionl_compiler_flags: {self.additional_compiler_args}\n"
        )


_NKIPY_BACKEND_CONFIG: Optional[NKIPyBackendConfig] = None


def get_nkipy_backend_config() -> Optional[NKIPyBackendConfig]:
    """Get the global NKIPy configuration."""
    return _NKIPY_BACKEND_CONFIG


def set_nkipy_backend_config(config: NKIPyBackendConfig) -> None:
    """Set the global NKIPy configuration.

    Args:
        config: NKIPyConfig instance to set as global config
    """
    global _NKIPY_BACKEND_CONFIG
    _NKIPY_BACKEND_CONFIG = config


def reset_nkipy_backend_config() -> Optional[NKIPyBackendConfig]:
    """Reset the global NKIPy configuration to None.

    Returns:
        Optional[NKIPyBackendConfig]: The previous configuration
    """
    global _NKIPY_BACKEND_CONFIG
    old_config = _NKIPY_BACKEND_CONFIG
    _NKIPY_BACKEND_CONFIG = None
    return old_config
