# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path

from torch_to_nkipy.nkipy_builder.aten_op_registry import batch_register  # noqa: F401
from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (  # noqa: F401
    AtenOpRegistry,
)

# Automatically import all modules that start with "aten_"
_current_dir = Path(__file__).parent
for _file in _current_dir.glob("aten_*.py"):
    importlib.import_module(f".{_file.stem}", package=__name__)
