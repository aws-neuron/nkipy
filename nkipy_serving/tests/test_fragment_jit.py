from __future__ import annotations

import sys
import types

import numpy as np

from nkipy_serving.fragment_jit import jit


def test_device_fragment_forwards_device_tensor_inputs(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv(
        "NKIPY_SERVING_GLOBAL_NEFF_CACHE_DIR",
        str(tmp_path / "global_neffs"),
    )

    class FakeDeviceTensor:
        def __init__(self, array, *, name: str = "tensor") -> None:
            self.arr = np.asarray(array)
            self.tensor_ref = object()
            self.shape = tuple(self.arr.shape)
            self.dtype = self.arr.dtype
            self.name = name

        @classmethod
        def from_numpy(cls, array, name: str = "tensor"):
            return cls(array, name=name)

        def numpy(self) -> np.ndarray:
            return self.arr.copy()

    class FakeSpikeTensor:
        pass

    class FakeCompiledKernel:
        input_tensors_info: dict[str, object] = {}
        output_tensors_info: dict[str, object] = {"output0": object()}

        def allocate_output_tensors(self):
            return [
                FakeDeviceTensor.from_numpy(
                    np.zeros((2, 2), dtype=np.float32),
                    name="output0",
                )
            ]

        def __call__(self, *, inputs, outputs):
            captured["inputs"] = inputs
            outputs["output0"].arr = inputs["x"].arr + inputs["w"].arr

    class FakeDeviceKernel:
        @staticmethod
        def _trace_and_compile(_fn, _name, args, _kwargs, **_compile_kwargs):
            captured["compile_arg_types"] = tuple(type(arg) for arg in args)
            return "fake.neff", "cache"

        @staticmethod
        def load_from_neff(_neff_path, **_kwargs):
            return FakeCompiledKernel()

    class FakeCompilationTarget:
        DEFAULT = object()

    monkeypatch.setitem(
        sys.modules,
        "nkipy.core.compile",
        types.SimpleNamespace(CompilationTarget=FakeCompilationTarget),
    )
    monkeypatch.setitem(
        sys.modules,
        "nkipy.runtime.device_kernel",
        types.SimpleNamespace(DeviceKernel=FakeDeviceKernel),
    )
    monkeypatch.setitem(
        sys.modules,
        "nkipy.runtime.device_tensor",
        types.SimpleNamespace(DeviceTensor=FakeDeviceTensor),
    )
    monkeypatch.setitem(
        sys.modules,
        "spike",
        types.SimpleNamespace(SpikeTensor=FakeSpikeTensor),
    )

    def add_fn(x, w):
        return x + w

    x = FakeDeviceTensor.from_numpy(np.ones((2, 2), dtype=np.float32), name="x")
    w = FakeDeviceTensor.from_numpy(
        np.full((2, 2), 3.0, dtype=np.float32),
        name="w",
    )

    out = jit(add_fn, device=True, name="add", build_dir=str(tmp_path))(x, w)

    assert captured["compile_arg_types"] == (FakeDeviceTensor, FakeDeviceTensor)
    assert captured["inputs"] == {"x": x, "w": w}
    assert np.array_equal(out.numpy(), np.full((2, 2), 4.0, dtype=np.float32))
