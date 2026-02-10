# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import uuid
from pathlib import Path

import pytest

# Skip all tests in this folder if torch_to_nkipy cannot be imported
try:
    from torch_to_nkipy.backend.nkipy_backend import (
        init_nkipy_backend,
        is_nkipy_backend_initialized,
        reset_nkipy_backend,
    )
    from torch_to_nkipy.backend.nkipy_backend_config import get_nkipy_backend_config

    TORCH_TO_NKIPY_AVAILABLE = True
except ImportError:
    TORCH_TO_NKIPY_AVAILABLE = False

# Don't collect any tests in the ops/ folder if torch_to_nkipy is not available
# pytest automatically reads collect_ignore_glob from conftest.py
collect_ignore_glob = []
if not TORCH_TO_NKIPY_AVAILABLE:
    collect_ignore_glob = ["ops/*.py"]


if TORCH_TO_NKIPY_AVAILABLE:
    import torch.distributed as dist

    @pytest.fixture(scope="module", autouse=True)
    def module_setup_and_cleanup(request):
        """Auto-initialize backend for each test module.

        Uses module scope so the backend is re-initialized after test_backend.py
        tests that use isolated_backend fixture and reset the backend.

        Distributed tests are skipped as they handle their own initialization.
        """
        # Skip auto-initialization for distributed tests (they handle their own init)
        # Distributed tests are run with torchrun which sets up the process group
        if dist.is_initialized():
            yield
            return

        if not is_nkipy_backend_initialized():
            init_nkipy_backend()

        yield

        # Don't cleanup cache between modules - only at the end
        # This allows cache reuse and avoids issues with shared state

    @pytest.fixture
    def isolated_backend():
        """Fixture that provides isolated backend for each test.

        This fixture ensures a clean backend state for tests that need to
        test backend initialization/reset behavior. It provides helper
        functions for init/reset and cleans up after the test.
        """
        # Ensure clean state before test
        if is_nkipy_backend_initialized():
            reset_nkipy_backend()

        cache_dir = f"./test_cache_{os.getpid()}_{uuid.uuid4().hex[:8]}"

        def init_with_cache(**kwargs):
            return init_nkipy_backend(nkipy_cache=cache_dir, **kwargs)

        yield {
            "cache_dir": cache_dir,
            "init": init_with_cache,
            "reset": reset_nkipy_backend,
            "is_initialized": is_nkipy_backend_initialized,
            "get_config": get_nkipy_backend_config,
        }

        # Cleanup after test
        if is_nkipy_backend_initialized():
            reset_nkipy_backend()
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def ntff_output_dir(tmp_path):
        """Fixture that provides temp directory for NTFF output."""
        return tmp_path / "ntff_output"
