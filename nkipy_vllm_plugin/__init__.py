# SPDX-License-Identifier: Apache-2.0
"""VllmNeuronPlugin module."""


def register():
    """Register the Neuron platform."""

    return "nkipy_vllm_plugin.platform.NeuronPlatform"
