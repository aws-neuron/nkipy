# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest

# Skip all tests in this folder if torch_to_nkipy cannot be imported
try:
    from torch_to_nkipy import (
        get_nkipy_backend_config,
        init_nkipy_backend,
        is_nkipy_backend_initialized,
    )

    TORCH_TO_NKIPY_AVAILABLE = True
except ImportError:
    TORCH_TO_NKIPY_AVAILABLE = False

# Don't collect any tests in the ops/ folder if torch_to_nkipy is not available
# pytest automatically reads collect_ignore_glob from conftest.py
collect_ignore_glob = []
if not TORCH_TO_NKIPY_AVAILABLE:
    collect_ignore_glob = ["ops/*.py"]


if TORCH_TO_NKIPY_AVAILABLE:

    @pytest.fixture(scope="package", autouse=True)
    def package_setup_and_cleanup():
        if not is_nkipy_backend_initialized():
            init_nkipy_backend()

        yield

        CACHE_DIR = Path(get_nkipy_backend_config().nkipy_cache_prefix)
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
