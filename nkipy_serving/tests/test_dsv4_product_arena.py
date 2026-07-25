from __future__ import annotations

from types import SimpleNamespace

import ml_dtypes
import numpy as np

import nkipy_serving.models.deepseek_v4.neff_runtime.resources.manager as manager_module
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.manager import (
    Dsv4ProductBucketManagerMixin,
)


class _FakeDeviceTensor:
    allocations: list[tuple[str, tuple[int, ...], str]] = []

    def __init__(self, *, tensor_ref: str, shape, dtype, name: str):
        self.tensor_ref = tensor_ref
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype
        self.name = str(name)

    @classmethod
    def from_numpy(cls, array, name: str = "tensor"):
        tensor = cls(
            tensor_ref=f"ref:{name}",
            shape=tuple(int(dim) for dim in array.shape),
            dtype=array.dtype,
            name=name,
        )
        cls.allocations.append((name, tensor.shape, str(np.dtype(array.dtype))))
        return tensor


class _FakeProductOwner(Dsv4ProductBucketManagerMixin):
    max_requests_per_step = 8

    def __init__(self, *, num_layers: int = 5):
        self.graph = {}
        self.runtime_surface = SimpleNamespace(
            args=SimpleNamespace(dim=4096),
            v4=SimpleNamespace(hidden_size=4096),
            blocks=tuple(
                SimpleNamespace(
                    attn=SimpleNamespace(n_heads=8, head_dim=512),
                    ffn=SimpleNamespace(dim=4096),
                )
                for _ in range(int(num_layers))
            ),
        )

    def _configured_product_token_buckets(self):
        return (256, 1024, 2048, 4096)


def _fake_first_dim_alias(value, *, start: int, size: int):
    assert int(start) == 0
    return _FakeDeviceTensor(
        tensor_ref=value.tensor_ref,
        shape=(int(size), *tuple(int(dim) for dim in value.shape[1:])),
        dtype=value.dtype,
        name=f"{value.name}_alias_{int(size)}",
    )


def test_product_bucket_large_arenas_use_rolling_layer_slots(monkeypatch) -> None:
    monkeypatch.setattr(
        manager_module,
        "_get_device_tensor_cls",
        lambda: _FakeDeviceTensor,
    )
    monkeypatch.setattr(
        manager_module,
        "_alias_device_value_first_dim_slice",
        _fake_first_dim_alias,
    )
    _FakeDeviceTensor.allocations = []

    owner = _FakeProductOwner(num_layers=5)
    bucket = owner._build_product_bucket(4096)

    assert bucket.attention_outputs[0] is bucket.attention_outputs[2]
    assert bucket.attention_outputs[1] is bucket.attention_outputs[3]
    assert bucket.moe_prefill_outputs[0] is bucket.moe_prefill_outputs[2]
    assert bucket.moe_prefill_ep_outputs[0] is bucket.moe_prefill_ep_outputs[2]
    assert bucket.moe_prefill_outputs[0] is not bucket.moe_prefill_ep_outputs[0]

    arena_names = [
        name
        for name, _shape, _dtype in _FakeDeviceTensor.allocations
        if "_arena_" in name
    ]
    assert sum("attention_out_arena" in name for name in arena_names) == 2
    assert sum("moe_prefill_out_arena" in name for name in arena_names) == 2
    assert sum("moe_prefill_ep_arena" in name for name in arena_names) == 2

    arena_count_before_small_bucket = len(arena_names)
    small_bucket = owner._build_product_bucket(1024)
    arena_names_after = [
        name
        for name, _shape, _dtype in _FakeDeviceTensor.allocations
        if "_arena_" in name
    ]

    assert len(arena_names_after) == arena_count_before_small_bucket
    assert (
        small_bucket.attention_outputs[0].tensor_ref
        == bucket.attention_outputs[0].tensor_ref
    )
    assert small_bucket.attention_outputs[0].shape == (1024, 8, 512)
    assert (
        small_bucket.moe_prefill_outputs[0].tensor_ref
        == bucket.moe_prefill_outputs[0].tensor_ref
    )
    assert small_bucket.moe_prefill_outputs[0].shape == (1024, 4096)
    assert small_bucket.moe_prefill_outputs[0].dtype == ml_dtypes.bfloat16
