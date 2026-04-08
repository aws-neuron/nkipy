# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy vLLM platform plugin."""

import glob
import warnings


def _is_neuron_dev() -> bool:
    return len(glob.glob("/dev/neuron*")) > 0


def register():
    """Register the NKIPy platform plugin with vLLM."""
    if not _is_neuron_dev():
        warnings.warn(
            "No Neuron devices found. Skipping NKIPy plugin registration.",
            category=UserWarning,
        )
        return None
    return "nkipy.vllm_plugin.platform.NKIPyPlatform"
