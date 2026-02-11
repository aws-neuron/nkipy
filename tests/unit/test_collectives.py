# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test distributed collectives in both IR and HLO modes"""

import nkipy.distributed.collectives as cc
import numpy as np
import pytest
from utils import (
    trace_and_run,
    trace_mode,  # noqa: F401 - pytest fixture
)


def test_all_reduce(trace_mode):
    """Test all_reduce collective in both IR and HLO modes"""

    def kernel(input_data):
        # All-reduce: sum across 2 devices
        result = cc.all_reduce(input_data, replica_groups=[[0, 1]], reduce_op=np.add)
        return result

    # Create test input
    input_data = np.random.rand(128, 256).astype(np.float32)

    # CPU execution (assumes all ranks have the same data)
    output = trace_and_run(kernel, trace_mode, input_data)

    # With 2 devices and same input, output should be 2x input (sum of 2 copies)
    expected = input_data * 2
    assert np.allclose(output, expected), (
        "All-reduce with 2 devices should return 2x input"
    )
    print(f"✓ all_reduce {trace_mode.upper()} test passed")


def test_all_gather(trace_mode):
    """Test all_gather collective in both IR and HLO modes"""

    def kernel(input_data):
        # All-gather along dimension 0 across 2 devices
        result = cc.all_gather(input_data, all_gather_dim=0, replica_groups=[[0, 1]])
        return result

    # Create test input
    input_data = np.random.rand(128, 256).astype(np.float32)

    # CPU execution (assumes all ranks have the same data)
    output = trace_and_run(kernel, trace_mode, input_data)

    # With 2 devices, output should be concatenated along dim 0: shape (256, 256)
    expected = np.concatenate([input_data, input_data], axis=0)
    assert output.shape == (256, 256), f"Expected shape (256, 256), got {output.shape}"
    assert np.allclose(output, expected), (
        "All-gather with 2 devices should concatenate inputs"
    )
    print(f"✓ all_gather {trace_mode.upper()} test passed")


def test_reduce_scatter(trace_mode):
    """Test reduce_scatter collective in both IR and HLO modes"""

    def kernel(input_data):
        # Reduce-scatter along dimension 0 across 2 devices
        result = cc.reduce_scatter(
            input_data, reduce_scatter_dim=0, replica_groups=[[0, 1]], reduce_op=np.add
        )
        return result

    # Create test input
    input_data = np.random.rand(128, 256).astype(np.float32)

    # CPU execution (assumes all ranks have the same data)
    output = trace_and_run(kernel, trace_mode, input_data)

    # With 2 devices, output should be half the size along dim 0: shape (64, 256)
    # Each device gets a slice that's been reduced across all devices
    assert output.shape == (64, 256), f"Expected shape (64, 256), got {output.shape}"
    # The output should be 2x the corresponding slice (sum of 2 copies)
    expected = input_data[:64, :] * 2
    assert np.allclose(output, expected), (
        "Reduce-scatter with 2 devices should reduce and scatter"
    )
    print(f"✓ reduce_scatter {trace_mode.upper()} test passed")


def test_all_to_all(trace_mode):
    """Test all_to_all collective in both IR and HLO modes

    With 2 devices, split_dimension=1, concat_dimension=0:
    - Each device splits input [128, 256] into 2 parts along dim 1: [128, 128] each
    - Device 0 sends part 0 to device 0, part 1 to device 1
    - Device 1 sends part 0 to device 0, part 1 to device 1
    - Each device concatenates received parts along dim 0

    Since all devices have the same input data:
    - Device 0 receives: [part 0 from dev 0, part 0 from dev 1] along dim 0
                       = [input[:, 0:128], input[:, 0:128]] concatenated on dim 0
                       = shape [256, 128]
    - Device 1 receives: [part 1 from dev 0, part 1 from dev 1] along dim 0
                       = [input[:, 128:256], input[:, 128:256]] concatenated on dim 0
                       = shape [256, 128]

    With duplicated data assumption, we get the first received chunk pattern.
    """

    def kernel(input_data):
        # All-to-all across 2 devices, split along dim 1, concat along dim 0
        result = cc.all_to_all(
            input_data, split_dimension=1, concat_dimension=0, replica_groups=[[0, 1]]
        )
        return result

    # Create test input
    input_data = np.random.rand(128, 256).astype(np.float32)

    # CPU execution (assumes all ranks have the same data)
    output = trace_and_run(kernel, trace_mode, input_data)

    # Shape should change: (128, 256) -> (256, 128)
    assert output.shape == (256, 128), f"Expected shape (256, 128), got {output.shape}"

    # Expected: split into 2 chunks along dim=1, then concatenate along dim=0
    chunks = [input_data[:, 0:128], input_data[:, 128:256]]
    expected = np.concatenate(chunks, axis=0)

    assert np.allclose(output, expected), (
        "All-to-all should split columns and stack rows"
    )
    print(f"✓ all_to_all {trace_mode.upper()} test passed")


def test_combined_ops(trace_mode):
    """Test combining collectives with regular operations in both IR and HLO modes"""

    def kernel(input_data, weight):
        # Multiply with weight
        local_result = input_data * weight

        # All-reduce across 2 devices
        reduced = cc.all_reduce(local_result, replica_groups=[[0, 1]], reduce_op=np.add)

        # Add bias
        result = reduced + 1.0

        return result

    # Create test inputs
    input_data = np.random.rand(128, 256).astype(np.float32)
    weight = np.random.rand(128, 256).astype(np.float32)

    # CPU execution (assumes all ranks have the same data)
    output = trace_and_run(kernel, trace_mode, input_data, weight)

    # Compute expected result: 2x (due to all-reduce sum across 2 devices) + 1.0
    expected = (input_data * weight) * 2 + 1.0

    assert np.allclose(output, expected), "Combined operations should work correctly"
    print(f"✓ combined_ops {trace_mode.upper()} test passed")


if __name__ == "__main__":
    # Allow running the test file directly
    pytest.main([__file__])
