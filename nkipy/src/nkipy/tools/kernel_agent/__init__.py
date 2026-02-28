# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel agent: NumPy op testing and LLM kernel generation.

Usage:
    python -m nkipy.tools.kernel_agent discover
    python -m nkipy.tools.kernel_agent test --ops add,exp
    python -m nkipy.tools.kernel_agent generate --prompt "softmax"
    python -m nkipy.tools.kernel_agent sweep
    python -m nkipy.tools.kernel_agent rerun <jsonl_path>
"""

from nkipy.tools.kernel_agent.executor import ExecutionResult, StageResult, run_kernel
from nkipy.tools.kernel_agent.generator import (
    build_inputs,
    compile_code,
    generate_kernel,
)
from nkipy.tools.kernel_agent.ops import (
    TARGET_OPS,
    discover_ops,
    get_all_ops,
    make_kernel,
    test_op,
)
from nkipy.tools.kernel_agent.sweep import (
    SweepRecord,
    get_kernel_prompts,
    run_rerun,
    run_sweep,
)

KERNEL_PROMPTS = get_kernel_prompts()

__all__ = [
    "run_kernel",
    "ExecutionResult",
    "StageResult",
    "TARGET_OPS",
    "get_all_ops",
    "test_op",
    "discover_ops",
    "make_kernel",
    "generate_kernel",
    "build_inputs",
    "compile_code",
    "run_sweep",
    "run_rerun",
    "SweepRecord",
    "get_kernel_prompts",
    "KERNEL_PROMPTS",
]
