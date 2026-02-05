# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel agent: NumPy op testing and LLM kernel generation.

Usage:
    python -m nkipy.tools.kernel_agent discover
    python -m nkipy.tools.kernel_agent test --ops add,exp
    python -m nkipy.tools.kernel_agent generate --prompt "softmax"
"""

from nkipy.tools.kernel_agent.executor import run_kernel, ExecutionResult, StageResult
from nkipy.tools.kernel_agent.ops import (
    TARGET_OPS,
    get_all_ops,
    test_op,
    discover_ops,
    make_kernel,
)

__all__ = [
    "run_kernel",
    "ExecutionResult",
    "StageResult",
    "TARGET_OPS",
    "get_all_ops",
    "test_op",
    "discover_ops",
    "make_kernel",
]
