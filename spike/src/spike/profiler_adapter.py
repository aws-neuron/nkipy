# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profiler Adapter for Spike"""

import inspect
import logging
import os
import sys

from ._spike import Spike

try:
    import torch.distributed as dist
except ImportError:
    dist = None

logger = logging.getLogger(__name__)


class SpikeProfiler:
    """Spike wrapper that adds debug logging to execute calls"""

    def __init__(self, verbose_level=0, target_file=None):
        self._spike = Spike(verbose_level)
        self.target_file = target_file

    def execute(self, model, inputs, outputs, ntff_name=None, save_trace=False):
        """Wrap execute to add debug logging before calling Spike execute"""
        # Only record rank 0 for performance
        rank = dist.get_rank() if dist and dist.is_initialized() else 0

        if rank == 0:
            source_info = self._find_source_loc()
            debug_file = f"debug_rank_{rank}.txt"

            with open(debug_file, "a") as f:
                f.write(f"PID: {os.getpid()}\n")
                f.write(f"SOURCE: {source_info[0]}:{source_info[1]}\n")
                if hasattr(model, "neff_path") and model.neff_path:
                    neff_filename = os.path.basename(model.neff_path)
                    model_name = neff_filename.replace(".neff", "")
                    f.write(f"KERNEL: {model_name}\n")
                f.write("---\n")

        # Call wrapped Spike execute method
        return self._spike.execute(model, inputs, outputs, ntff_name, save_trace)

    def __getattr__(self, name):
        """Delegate all other method calls to the wrapped Spike instance"""
        return getattr(self._spike, name)

    def _find_source_loc(self):
        """Find the source location of the kernel call using inspect"""
        try:
            # Determine target file
            target_file = self.target_file
            if target_file is None:
                if sys.argv:
                    target_file = os.path.basename(sys.argv[0])
                else:
                    raise RuntimeError(
                        "No target file specified and no sys.argv available"
                    )

            # Walk back through frames until we find the target file
            frame = inspect.currentframe()
            while frame:
                if target_file in frame.f_code.co_filename:
                    return (frame.f_code.co_filename, frame.f_lineno)
                frame = frame.f_back

            raise RuntimeError(
                f"Error finding target profiling file: {target_file}. "
                "Manually specify it in SpikeProfiler(target_file='your_file.py')"
            )
        except Exception as e:
            raise RuntimeError(
                "Error finding target profiling file. "
                f"Manually specify it in SpikeProfiler(target_file='your_file.py'): {e}"
            )
