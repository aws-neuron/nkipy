"""Tests for logits processor warmup compilation behavior."""

import numpy as np
import pytest

import nkipy_serving.sampling.logits_processor as logits_processor_module


def test_eager_sampler_warmup_compiles_and_loads_logprob_kernels(
    monkeypatch,
    tmp_path,
) -> None:
    eager_loaded_names: list[str] = []

    class _FakeDeviceKernel:
        @staticmethod
        def _trace_and_compile(*args, **kwargs):
            del args, kwargs
            return None

        @staticmethod
        def compile_and_load(*args, **kwargs):
            del args
            eager_loaded_names.append(str(kwargs["name"]))
            return ("loaded", kwargs["name"])

        @staticmethod
        def load_from_neff(*args, **kwargs):
            del args, kwargs
            raise AssertionError("warmup must use compile/load NEFF helper")

    def fake_compile_and_load_neff_with_lock(*args, **kwargs):
        del args
        eager_loaded_names.append(str(kwargs["name"]))
        return ("compiled-loaded", kwargs["name"])

    def fail_deferred_compile(*args, **kwargs):
        del args, kwargs
        raise AssertionError("warmup must compile/load logits kernels eagerly")

    monkeypatch.setattr(logits_processor_module, "_ensure_nkipy_runtime", lambda: None)
    monkeypatch.setattr(logits_processor_module, "_DeviceKernel", _FakeDeviceKernel)
    monkeypatch.setattr(
        logits_processor_module,
        "compile_and_load_neff_with_lock",
        fake_compile_and_load_neff_with_lock,
    )
    monkeypatch.setattr(
        logits_processor_module,
        "compile_neff_path_with_lock",
        fail_deferred_compile,
    )
    monkeypatch.setattr(
        logits_processor_module,
        "_alloc_device_scratch",
        lambda shape, dtype, name: (tuple(shape), np.dtype(dtype), str(name)),
    )

    processor = logits_processor_module.LogitsProcessor(
        vocab_size=16,
        local_vocab_size=16,
        vocab_offset=0,
        hidden_size=8,
        dtype=np.float32,
        tp_degree=1,
        dense_local_topk=1,
        build_dir=str(tmp_path),
        max_requests_per_step=2,
    )

    kernels = processor._ensure_kernels(
        4,
        include_sampler=True,
        include_logprobs=True,
        deferred_sampler_load=False,
    )

    assert eager_loaded_names == [
        "lp_top1_tp1_t4_bsmax2",
        "lp_sample_tp1_t4_bsmax2",
        "lp_sample_unf_tp1_t4_bsmax2",
        "lp_sample_logprobs_tp1_t4_bsmax2_k20",
        "lp_sample_logprobs_unf_tp1_t4_bsmax2_k20",
    ]
    assert kernels.device_sample_kernel == (
        "compiled-loaded",
        "lp_sample_tp1_t4_bsmax2",
    )
    assert kernels.logprobs_sample_kernel == (
        "compiled-loaded",
        "lp_sample_logprobs_tp1_t4_bsmax2_k20",
    )


def test_deferred_sampler_warmup_compiles_without_loading(
    monkeypatch,
    tmp_path,
) -> None:
    traced_names: list[str] = []
    loaded_names: list[str] = []
    barrier_names: list[str] = []
    called_names: list[str] = []

    class _LoadedKernel:
        def __init__(self, name: str):
            self.name = str(name)

        def __call__(self, *args, **kwargs):
            del args, kwargs
            called_names.append(self.name)

    class _FakeDeviceKernel:
        @staticmethod
        def _trace_and_compile(*args, **kwargs):
            del args, kwargs
            return None

        @staticmethod
        def compile_and_load(*args, **kwargs):
            del args, kwargs
            raise AssertionError("deferred warmup must not compile/load")

        @staticmethod
        def load_from_neff(*args, **kwargs):
            del args
            loaded_names.append(str(kwargs["name"]))
            return _LoadedKernel(str(kwargs["name"]))

    def fake_compile_and_load_neff_with_lock(*args, **kwargs):
        del args, kwargs
        raise AssertionError("deferred warmup must not load via NEFF helper")

    def fake_compile_neff_path_with_lock(*args, **kwargs):
        del args
        traced_names.append(str(kwargs["name"]))
        return str(tmp_path / f"{kwargs['name']}.neff")

    monkeypatch.setattr(logits_processor_module, "_ensure_nkipy_runtime", lambda: None)
    monkeypatch.setattr(logits_processor_module, "_DeviceKernel", _FakeDeviceKernel)
    monkeypatch.setattr(
        logits_processor_module,
        "compile_and_load_neff_with_lock",
        fake_compile_and_load_neff_with_lock,
    )
    monkeypatch.setattr(
        logits_processor_module,
        "compile_neff_path_with_lock",
        fake_compile_neff_path_with_lock,
    )
    monkeypatch.setattr(
        logits_processor_module,
        "collective_load_barrier",
        lambda *, name, **kwargs: barrier_names.append(str(name)),
    )
    monkeypatch.setattr(
        logits_processor_module,
        "_alloc_device_scratch",
        lambda shape, dtype, name: (tuple(shape), np.dtype(dtype), str(name)),
    )

    processor = logits_processor_module.LogitsProcessor(
        vocab_size=16,
        local_vocab_size=8,
        vocab_offset=0,
        hidden_size=8,
        dtype=np.float32,
        tp_degree=2,
        dense_local_topk=1,
        build_dir=str(tmp_path),
        max_requests_per_step=2,
    )

    kernels = processor._ensure_kernels(
        4,
        include_sampler=True,
        include_logprobs=True,
        deferred_sampler_load=True,
    )

    assert traced_names == [
        "lp_top1_tp2_t4_bsmax2",
        "lp_sample_tp2_t4_bsmax2",
        "lp_sample_unf_tp2_t4_bsmax2",
        "lp_sample_logprobs_tp2_t4_bsmax2_k20",
        "lp_sample_logprobs_unf_tp2_t4_bsmax2_k20",
    ]
    assert loaded_names == []
    assert barrier_names == []

    kernels.greedy_kernel(inputs={}, outputs={})
    assert loaded_names == ["lp_top1_tp2_t4_bsmax2"]
    assert barrier_names == []
    assert called_names == ["lp_top1_tp2_t4_bsmax2"]

    kernels.device_sample_kernel(inputs={}, outputs={})
    assert loaded_names == [
        "lp_top1_tp2_t4_bsmax2",
        "lp_sample_tp2_t4_bsmax2",
    ]
    assert barrier_names == ["lp_sample_tp2_t4_bsmax2"]
    assert called_names == [
        "lp_top1_tp2_t4_bsmax2",
        "lp_sample_tp2_t4_bsmax2",
    ]


def test_logits_processor_seal_rejects_uncompiled_bucket(
    monkeypatch,
    tmp_path,
) -> None:
    class _FakeDeviceKernel:
        @staticmethod
        def _trace_and_compile(*args, **kwargs):
            del args, kwargs
            return None

        @staticmethod
        def compile_and_load(*args, **kwargs):
            del args
            return ("loaded", kwargs["name"])

        @staticmethod
        def load_from_neff(*args, **kwargs):
            del args, kwargs
            raise AssertionError("test uses compile_and_load_neff_with_lock")

    def fake_compile_and_load_neff_with_lock(*args, **kwargs):
        del args
        return ("compiled-loaded", kwargs["name"])

    monkeypatch.setattr(logits_processor_module, "_ensure_nkipy_runtime", lambda: None)
    monkeypatch.setattr(logits_processor_module, "_DeviceKernel", _FakeDeviceKernel)
    monkeypatch.setattr(
        logits_processor_module,
        "compile_and_load_neff_with_lock",
        fake_compile_and_load_neff_with_lock,
    )
    monkeypatch.setattr(
        logits_processor_module,
        "_alloc_device_scratch",
        lambda shape, dtype, name: (tuple(shape), np.dtype(dtype), str(name)),
    )

    processor = logits_processor_module.LogitsProcessor(
        vocab_size=16,
        local_vocab_size=16,
        vocab_offset=0,
        hidden_size=8,
        dtype=np.float32,
        tp_degree=1,
        dense_local_topk=1,
        build_dir=str(tmp_path),
        max_requests_per_step=2,
    )

    processor._ensure_kernels(4, include_sampler=True, include_logprobs=True)
    processor.seal_precompiled_kernels()

    processor._ensure_kernels(4, include_sampler=True, include_logprobs=True)
    with pytest.raises(RuntimeError, match="uncompiled bucket=8"):
        processor._ensure_kernels(8, include_sampler=True, include_logprobs=True)


def test_logits_processor_noncollective_requires_precompiled_neff_api(
    monkeypatch,
    tmp_path,
) -> None:
    class _CompileOnlyKernel:
        @staticmethod
        def compile_and_load(*args, **kwargs):
            del args, kwargs
            raise AssertionError("non-collective logits should use NEFF load")

    monkeypatch.setattr(logits_processor_module, "_ensure_nkipy_runtime", lambda: None)
    monkeypatch.setattr(logits_processor_module, "_DeviceKernel", _CompileOnlyKernel)

    processor = logits_processor_module.LogitsProcessor(
        vocab_size=16,
        local_vocab_size=16,
        vocab_offset=0,
        hidden_size=8,
        dtype=np.float32,
        tp_degree=1,
        dense_local_topk=1,
        build_dir=str(tmp_path),
        max_requests_per_step=2,
    )

    with pytest.raises(RuntimeError, match="precompiled NEFF"):
        processor._ensure_kernels(4, include_sampler=True, include_logprobs=True)


def test_logits_processor_collective_sampler_loads_precompiled_neff(
    monkeypatch,
    tmp_path,
) -> None:
    traced_names: list[str] = []
    loaded_names: list[str] = []
    barriers: list[str] = []

    class _FakeDeviceKernel:
        @staticmethod
        def _trace_and_compile(*args, **kwargs):
            del args, kwargs
            return None

        @staticmethod
        def compile_and_load(*args, **kwargs):
            del args, kwargs
            raise AssertionError("collective logits should load precompiled NEFF")

        @staticmethod
        def load_from_neff(*args, **kwargs):
            del args
            loaded_names.append(str(kwargs["name"]))
            return ("loaded-neff", kwargs["name"])

    def fake_compile_and_load_neff_with_lock(*args, **kwargs):
        del args
        return ("compiled-loaded", kwargs["name"])

    def fake_compile_neff_path_with_lock(*args, **kwargs):
        del args
        traced_names.append(str(kwargs["name"]))
        return str(tmp_path / f"{kwargs['name']}.neff")

    monkeypatch.setattr(logits_processor_module, "_ensure_nkipy_runtime", lambda: None)
    monkeypatch.setattr(logits_processor_module, "_DeviceKernel", _FakeDeviceKernel)
    monkeypatch.setattr(
        logits_processor_module,
        "compile_and_load_neff_with_lock",
        fake_compile_and_load_neff_with_lock,
    )
    monkeypatch.setattr(
        logits_processor_module,
        "compile_neff_path_with_lock",
        fake_compile_neff_path_with_lock,
    )
    monkeypatch.setattr(
        logits_processor_module,
        "collective_load_barrier",
        lambda *, name, **kwargs: barriers.append(str(name)),
    )
    monkeypatch.setattr(
        logits_processor_module,
        "_alloc_device_scratch",
        lambda shape, dtype, name: (tuple(shape), np.dtype(dtype), str(name)),
    )

    processor = logits_processor_module.LogitsProcessor(
        vocab_size=16,
        local_vocab_size=8,
        vocab_offset=0,
        hidden_size=8,
        dtype=np.float32,
        tp_degree=2,
        dense_local_topk=1,
        build_dir=str(tmp_path),
        max_requests_per_step=2,
    )

    kernels = processor._ensure_kernels(
        4,
        include_sampler=True,
        include_logprobs=True,
    )

    assert traced_names == [
        "lp_sample_tp2_t4_bsmax2",
        "lp_sample_unf_tp2_t4_bsmax2",
        "lp_sample_logprobs_tp2_t4_bsmax2_k20",
        "lp_sample_logprobs_unf_tp2_t4_bsmax2_k20",
    ]
    assert barriers == traced_names
    assert loaded_names == traced_names
    assert kernels.device_sample_kernel == ("loaded-neff", traced_names[0])
    assert kernels.logprobs_sample_kernel == ("loaded-neff", traced_names[2])
