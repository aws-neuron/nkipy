# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profiling utilities for Neuron execution."""

import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from torch_to_nkipy.backend.nkipy_backend_config import get_nkipy_backend_config
from torch_to_nkipy.utils.ntff_meta import NtffMeta

logger = logging.getLogger(__name__)


@contextmanager
def nkipy_profile(ntff_meta: NtffMeta, neff_path: str):
    """Context manager determining profiling settings.

    Yields (save_trace, ntff_name) tuple for use in spike_execute.
    """
    logger.debug(
        f"Profiling NEFF with hash {ntff_meta.kernel_hash}, "
        f"save_ntff_exe_idx {ntff_meta.save_ntff_exe_idx}, and "
        f"curr_exe_idx {ntff_meta.curr_exe_idx}"
    )

    exe_idx_check = (
        not ntff_meta.save_ntff_exe_idx
        or ntff_meta.curr_exe_idx in ntff_meta.save_ntff_exe_idx
    )
    rank_check = get_nkipy_backend_config().rank == 0
    should_profile = ntff_meta.save_ntff and exe_idx_check and rank_check
    ntff_file = None

    if should_profile:
        save_dir = Path(
            f"{ntff_meta.save_ntff_dir}/kernel_{ntff_meta.kernel_hash}"
        ).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        neff_filename = os.path.basename(neff_path)
        target_neff_path = save_dir / neff_filename
        if not target_neff_path.exists() and os.path.exists(neff_path):
            shutil.copy2(neff_path, target_neff_path)

        ntff_file = str(save_dir / f"{ntff_meta.curr_exe_idx}.ntff")
        if Path(ntff_file).exists():
            Path(ntff_file).unlink()

        logger.debug(f"Saving NTFF profile to {ntff_file}")

    yield should_profile, ntff_file
    ntff_meta.curr_exe_idx += 1
