from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import nkipy_serving.ops.deepseek_v4.swa_state as swa_state


class _FakeDeviceTensor:
    allocations: list[tuple[str, np.ndarray]] = []

    def __init__(self, *, array: np.ndarray, name: str):
        self.array = np.ascontiguousarray(array)
        self.tensor_ref = f"ref:{name}"
        self.shape = tuple(int(dim) for dim in self.array.shape)
        self.dtype = self.array.dtype
        self.name = str(name)

    @classmethod
    def from_numpy(cls, array, name: str = "tensor"):
        tensor = cls(array=np.ascontiguousarray(array), name=name)
        cls.allocations.append((name, tensor.array.copy()))
        return tensor


def test_mirror_swa_pads_bucket_rows_metadata(monkeypatch) -> None:
    calls = []
    _FakeDeviceTensor.allocations = []
    swa_state._DEVICE_SCALAR_I32_CACHE.clear()
    monkeypatch.setattr(
        swa_state,
        "get_device_tensor_cls",
        lambda: _FakeDeviceTensor,
    )
    monkeypatch.setattr(
        swa_state,
        "run_write_kv_owner_window_device",
        lambda **kwargs: calls.append(kwargs),
    )

    state = SimpleNamespace(
        swa_kv_cache=_FakeDeviceTensor.from_numpy(
            np.zeros((8192, 8), dtype=np.float32),
            name="swa_cache",
        )
    )
    kv_rows = _FakeDeviceTensor.from_numpy(
        np.zeros((4096, 8), dtype=np.float32),
        name="bucket_rows",
    )
    live_owner_ids = _FakeDeviceTensor.from_numpy(
        np.zeros((3500,), dtype=np.int32),
        name="live_owner_ids",
    )
    live_positions = _FakeDeviceTensor.from_numpy(
        np.arange(3500, dtype=np.int32),
        name="live_positions",
    )

    swa_state.mirror_swa_kv_to_device_cache(
        kv_rows,
        0,
        window_size=4096,
        device_layer_state=state,
        owner_ids=np.zeros((3500,), dtype=np.int32),
        owner_ids_dev=live_owner_ids,
        positions_dev=live_positions,
        bsz=1,
        seqlen=3500,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["kv_new"] is kv_rows
    assert call["owner_ids"].shape == (4096,)
    assert call["positions"].shape == (4096,)
    assert call["live_rows"].shape == (1, 1)
    assert int(call["live_rows"].array[0, 0]) == 3500
    np.testing.assert_array_equal(call["positions"].array[:3500], np.arange(3500))
    np.testing.assert_array_equal(call["positions"].array[3500:], 0)
