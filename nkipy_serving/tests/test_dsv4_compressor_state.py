from __future__ import annotations

from types import SimpleNamespace

import ml_dtypes
import numpy as np

import nkipy_serving.ops.deepseek_v4.compressor_state as compressor_state


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


def test_compressor_state_mirror_pads_bucket_rows_metadata(monkeypatch) -> None:
    calls = []
    _FakeDeviceTensor.allocations = []
    compressor_state._DEVICE_SCALAR_I32_CACHE.clear()
    monkeypatch.setattr(
        compressor_state,
        "get_device_tensor_cls",
        lambda: _FakeDeviceTensor,
    )
    monkeypatch.setattr(
        compressor_state,
        "run_write_kv_score_state_device",
        lambda **kwargs: calls.append(kwargs),
    )

    compressor = SimpleNamespace(
        ape=np.zeros((4096, 8), dtype=ml_dtypes.bfloat16),
    )
    device_state = SimpleNamespace(
        kv_score_state=_FakeDeviceTensor.from_numpy(
            np.zeros((8192, 16), dtype=np.float32),
            name="kv_score_state",
        ),
        ring_size=4096,
        spec=SimpleNamespace(compress_ratio=4096, overlap=True),
    )
    kv = _FakeDeviceTensor.from_numpy(
        np.zeros((4096, 8), dtype=ml_dtypes.bfloat16),
        name="bucket_kv",
    )
    score = _FakeDeviceTensor.from_numpy(
        np.zeros((4096, 8), dtype=ml_dtypes.bfloat16),
        name="bucket_score",
    )

    compressor_state.mirror_compressor_input_to_device_state(
        compressor,
        kv,
        score,
        0,
        bsz=1,
        seqlen=3500,
        device_state=device_state,
        owner_ids=np.zeros((3500,), dtype=np.int32),
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["kv_new"] is kv
    assert call["owner_ids"].shape == (4096,)
    assert call["positions"].shape == (4096,)
    assert call["live_rows"].shape == (1, 1)
    assert int(call["live_rows"].array[0, 0]) == 3500
    np.testing.assert_array_equal(call["positions"].array[:3500], np.arange(3500))
    np.testing.assert_array_equal(call["positions"].array[3500:], 0)
