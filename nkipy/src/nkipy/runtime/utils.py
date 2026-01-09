# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
def is_neuron_compatible():
    """Check if machine has Neuron cores available"""
    try:
        from spike._spike import Spike

        available_core_count = Spike.get_visible_neuron_core_count()
        if available_core_count < 1:
            print("No Neuron cores found")
            return False
        print(f"Found {available_core_count} compatible Neuron core(s)")
        return True
    except Exception as e:
        print(
            f"Machine compatibility check failed: {e}. "
            "Not a Neuron machine or runtime components (e.g., Spike) not installed."
        )
        return False
