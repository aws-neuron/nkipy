from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import nkipy_serving.runtime.device_tensor as device_tensor_module
from nkipy_serving.runtime.device_tensor import alias_device_value_first_dim_slice


class _FakeDeviceTensor:
    def __init__(self, *, tensor_ref, shape, dtype, name):
        self.tensor_ref = tensor_ref
        self.shape = tuple(shape)
        self.dtype = dtype
        self.name = name


def _install_fake_spike(monkeypatch: pytest.MonkeyPatch, singleton: object) -> None:
    spike_module = ModuleType("spike")
    spike_singleton_module = ModuleType("spike.spike_singleton")
    spike_singleton_module.get_spike_singleton = lambda: singleton
    monkeypatch.setitem(sys.modules, "spike", spike_module)
    monkeypatch.setitem(sys.modules, "spike.spike_singleton", spike_singleton_module)


def test_alias_device_value_first_dim_slice_builds_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Spike:
        def slice_from_tensor(self, tensor_ref, byte_offset, byte_size, alias_name):
            assert tensor_ref == "base_ref"
            assert byte_offset == 8
            assert byte_size == 8
            assert alias_name == "base_slice_1_1"
            return "slice_ref"

    _install_fake_spike(monkeypatch, _Spike())
    monkeypatch.setattr(
        device_tensor_module,
        "get_device_tensor_cls",
        lambda: _FakeDeviceTensor,
    )

    alias = alias_device_value_first_dim_slice(
        SimpleNamespace(
            tensor_ref="base_ref",
            shape=(3, 2),
            dtype=np.float32,
            name="base",
        ),
        start=1,
        size=1,
    )

    assert isinstance(alias, _FakeDeviceTensor)
    assert alias.tensor_ref == "slice_ref"
    assert alias.shape == (1, 2)
    assert alias.dtype == np.float32
    assert alias.name == "base_slice_1_1"


def test_alias_device_value_first_dim_slice_expected_slice_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Spike:
        def slice_from_tensor(self, *args, **kwargs):
            raise RuntimeError("slice unsupported")

    _install_fake_spike(monkeypatch, _Spike())
    monkeypatch.setattr(
        device_tensor_module,
        "get_device_tensor_cls",
        lambda: _FakeDeviceTensor,
    )

    alias = alias_device_value_first_dim_slice(
        SimpleNamespace(tensor_ref="base_ref", shape=(3, 2), dtype=np.float32),
        start=1,
        size=1,
    )

    assert alias is None


def test_alias_device_value_first_dim_slice_unexpected_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnexpectedError(Exception):
        pass

    class _Spike:
        def slice_from_tensor(self, *args, **kwargs):
            raise _UnexpectedError("boom")

    _install_fake_spike(monkeypatch, _Spike())
    monkeypatch.setattr(
        device_tensor_module,
        "get_device_tensor_cls",
        lambda: _FakeDeviceTensor,
    )

    with pytest.raises(_UnexpectedError, match="boom"):
        alias_device_value_first_dim_slice(
            SimpleNamespace(tensor_ref="base_ref", shape=(3, 2), dtype=np.float32),
            start=1,
            size=1,
        )
