# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""spiky.torch

Optional PyTorch integration.

Importing this module registers the PyTorch PrivateUse1 backend name as "nkipy".
NRT initialization remains explicit: call `spiky.init()` before allocating any
`torch.device("nkipy")` tensors.
"""

import torch

from . import _C as _ext  # type: ignore
from .device import device_module


def _register():
    _ext._register_backend_name()
    _ext._register_hooks()
    _ext._register_allocator()

    # Make torch.nkipy available and generate standard methods (tensor.nkipy()).
    torch._register_device_module("nkipy", device_module)
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True, for_module=True, for_storage=True
    )


_register()


def device_count() -> int:
    return int(_ext._device_count())


def current_device() -> int:
    return int(_ext._current_device())


def set_device(device: int) -> None:
    _ext._set_device(int(device))


def is_available() -> bool:
    return bool(_ext._is_available())


def empty_cache() -> None:
    _ext._empty_cache()


def get_cached_blocks() -> int:
    return int(_ext._get_cached_blocks())


__all__ = [
    "device_count",
    "current_device",
    "set_device",
    "is_available",
    "empty_cache",
    "get_cached_blocks",
]
