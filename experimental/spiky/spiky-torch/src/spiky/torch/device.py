# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device module for nkipy - provides torch.nkipy.* functions."""

import torch


class DeviceModule:
    @staticmethod
    def is_available() -> bool:
        from . import _C

        return bool(_C._is_available())

    @staticmethod
    def device_count() -> int:
        from . import _C

        return int(_C._device_count())

    @staticmethod
    def current_device() -> int:
        from . import _C

        return int(_C._current_device())

    @staticmethod
    def set_device(device: int) -> None:
        from . import _C

        _C._set_device(int(device))

    @staticmethod
    def synchronize(device: int | None = None) -> None:
        # No-op: spiky.torch operations are synchronous.
        _ = device  # keep signature; silence linters without extra deps
        return None

    @staticmethod
    def empty_cache() -> None:
        from . import _C

        _C._empty_cache()

    @staticmethod
    def get_device_name(device: int | None = None) -> str:
        if device is None:
            from . import _C

            device = int(_C._current_device())
        return f"AWS Neuron Core {device}"

    @staticmethod
    def get_device_capability(device: int | None = None) -> tuple[int, int]:
        _ = device
        return (1, 0)

    @staticmethod
    def get_device_properties(device: int | None = None) -> dict:
        if device is None:
            from . import _C

            device = int(_C._current_device())
        return {"name": f"AWS Neuron Core {device}", "device_index": device}

    @staticmethod
    def get_amp_supported_dtype() -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @staticmethod
    def is_initialized() -> bool:
        return DeviceModule.is_available()

    @staticmethod
    def init() -> None:
        # Explicit NRT init is handled by spiky.init(); keep this as a no-op.
        return None


device_module = DeviceModule()
