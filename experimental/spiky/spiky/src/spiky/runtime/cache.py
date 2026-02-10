# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model caching and deduplication for Neuron execution."""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Union

from spiky.utils.names import COMPILER_DIR, NEFF_FILE

logger = logging.getLogger(__name__)

# Global cache indexed by kernel hash to guarantee kernels with the same
# hash are only compiled once across multiple ranks
hashes_to_kernel_dirs: Dict[str, Path] = {}


def get_compile_dir_and_neff_path(kernel_dir: Path):
    kernel_compile_dir = kernel_dir / COMPILER_DIR
    neff_path = kernel_compile_dir / NEFF_FILE
    return kernel_compile_dir, neff_path


def get_kernel_hash_from_path(kernel_dir: Union[str, Path]):
    """Hash the kernel directory name for cross-rank deduplication.

    Uses the directory name (not the full path) so identical kernels under
    different rank dirs (rank_0/kernel_X, rank_1/kernel_X) hash equally.
    """
    name = Path(kernel_dir).name
    return hashlib.sha256(name.encode()).hexdigest()[:16]
