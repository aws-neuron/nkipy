from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("neuronxcc.nki")

from nkipy.runtime.device_kernel import DeviceKernel
from nkipy.runtime.device_tensor import DeviceTensor

import nkipy_serving.ops.lm_head_sampling as ref_math
from nkipy_serving.sampling.nki_kernels import sample_tokens

pytestmark = pytest.mark.integration


def _make_logits(seed: int, *, batch_size: int, vocab_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.normal(loc=0.0, scale=1.0, size=(batch_size, vocab_size)).astype(
        np.float32
    )
    # Add a tiny monotonic offset so ties are vanishingly unlikely.
    logits += np.linspace(
        np.float32(-1e-3),
        np.float32(1e-3),
        vocab_size,
        dtype=np.float32,
    )[None, :]
    return logits


def _assert_sampler_matches_reference(
    *,
    logits: np.ndarray,
    temperatures: np.ndarray,
    top_ks: np.ndarray,
    top_ps: np.ndarray,
    min_ps: np.ndarray,
    uniform_u: np.ndarray,
    build_dir,
) -> None:
    expected = ref_math._sample_from_logits_reference(
        logits=logits,
        temperatures=temperatures,
        top_ks=top_ks,
        top_ps=top_ps,
        min_ps=min_ps,
        uniform_u=uniform_u,
    )

    logits_2d = logits.astype(np.float32)
    temperatures_2d = temperatures.astype(np.float32).reshape((-1, 1))
    top_ks_2d = top_ks.astype(np.int32).reshape((-1, 1))
    top_ps_2d = top_ps.astype(np.float32).reshape((-1, 1))
    min_ps_2d = min_ps.astype(np.float32).reshape((-1, 1))
    uniform_u_2d = uniform_u.astype(np.float32).reshape((-1, 1))

    def sampler_kernel(logits, temperatures, top_ks, top_ps, min_ps, uniform_u):
        return sample_tokens(logits, temperatures, top_ks, top_ps, min_ps, uniform_u)

    kernel = DeviceKernel.compile_and_load(
        sampler_kernel,
        logits_2d,
        temperatures_2d,
        top_ks_2d,
        top_ps_2d,
        min_ps_2d,
        uniform_u_2d,
        name=f"sample_tokens_accuracy_bs{logits.shape[0]}_v{logits.shape[1]}",
        use_cached_if_exists=False,
        build_dir=str(build_dir),
    )
    outputs = kernel(
        {
            "logits": DeviceTensor.from_numpy(logits_2d),
            "temperatures": DeviceTensor.from_numpy(temperatures_2d),
            "top_ks": DeviceTensor.from_numpy(top_ks_2d),
            "top_ps": DeviceTensor.from_numpy(top_ps_2d),
            "min_ps": DeviceTensor.from_numpy(min_ps_2d),
            "uniform_u": DeviceTensor.from_numpy(uniform_u_2d),
        }
    )
    actual = next(iter(outputs.values())).numpy().reshape((-1,)).astype(np.int32)

    assert np.array_equal(actual, expected), (
        f"sampler mismatch\nexpected={expected.tolist()}\nactual={actual.tolist()}"
    )


def test_sample_tokens_qwen3_06b_default_generation(tmp_path) -> None:
    """Qwen3-0.6B full vocab (151936) with default generation settings.

    Default params: temperature=1.0, top_k=all, top_p=1.0, min_p=0.0.
    This is pure multinomial sampling from softmax — the most common
    non-greedy path in serving.
    """
    batch_size = 4
    vocab_size = 151936
    logits = _make_logits(2025, batch_size=batch_size, vocab_size=vocab_size)

    _assert_sampler_matches_reference(
        logits=logits,
        temperatures=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        top_ks=np.asarray(
            [vocab_size, vocab_size, vocab_size, vocab_size], dtype=np.int32
        ),
        top_ps=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        min_ps=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        uniform_u=np.asarray([0.25, 0.50, 0.75, 0.99], dtype=np.float32),
        build_dir=tmp_path / "qwen3_default_build",
    )


def _assert_unfiltered_matches_reference(
    *,
    logits: np.ndarray,
    temperatures: np.ndarray,
    uniform_u: np.ndarray,
    build_dir,
) -> None:
    """Run the unfiltered kernel and compare against the reference with no filtering.

    Uses the same 6-tensor interface as the filtered kernel — the unfiltered
    variant just ignores top_ks/top_ps/min_ps internally.
    """
    vocab_size = logits.shape[1]
    batch_size = logits.shape[0]

    expected = ref_math._sample_from_logits_reference(
        logits=logits,
        temperatures=temperatures,
        top_ks=np.full(batch_size, vocab_size, dtype=np.int32),
        top_ps=np.ones(batch_size, dtype=np.float32),
        min_ps=np.zeros(batch_size, dtype=np.float32),
        uniform_u=uniform_u,
    )

    logits_2d = logits.astype(np.float32)
    temperatures_2d = temperatures.astype(np.float32).reshape((-1, 1))
    top_ks_2d = np.full((batch_size, 1), vocab_size, dtype=np.int32)
    top_ps_2d = np.ones((batch_size, 1), dtype=np.float32)
    min_ps_2d = np.zeros((batch_size, 1), dtype=np.float32)
    uniform_u_2d = uniform_u.astype(np.float32).reshape((-1, 1))

    def sampler_kernel(logits, temperatures, top_ks, top_ps, min_ps, uniform_u):
        return sample_tokens(
            logits, temperatures, top_ks, top_ps, min_ps, uniform_u, _unfiltered=True
        )

    kernel = DeviceKernel.compile_and_load(
        sampler_kernel,
        logits_2d,
        temperatures_2d,
        top_ks_2d,
        top_ps_2d,
        min_ps_2d,
        uniform_u_2d,
        name=f"sample_unfiltered_accuracy_bs{batch_size}_v{vocab_size}",
        use_cached_if_exists=False,
        build_dir=str(build_dir),
    )
    outputs = kernel(
        {
            "logits": DeviceTensor.from_numpy(logits_2d),
            "temperatures": DeviceTensor.from_numpy(temperatures_2d),
            "top_ks": DeviceTensor.from_numpy(top_ks_2d),
            "top_ps": DeviceTensor.from_numpy(top_ps_2d),
            "min_ps": DeviceTensor.from_numpy(min_ps_2d),
            "uniform_u": DeviceTensor.from_numpy(uniform_u_2d),
        }
    )
    actual = next(iter(outputs.values())).numpy().reshape((-1,)).astype(np.int32)

    assert np.array_equal(actual, expected), (
        f"unfiltered sampler mismatch\nexpected={expected.tolist()}\nactual={actual.tolist()}"
    )


def test_sample_tokens_qwen3_06b_unfiltered(tmp_path) -> None:
    """Qwen3-0.6B with unfiltered fast-path kernel (3 passes vs 15).

    Same logits and params as the default generation test, but uses
    the stripped-down kernel that skips threshold search entirely.
    Must produce bit-exact identical results.
    """
    batch_size = 4
    vocab_size = 151936
    logits = _make_logits(2025, batch_size=batch_size, vocab_size=vocab_size)

    _assert_unfiltered_matches_reference(
        logits=logits,
        temperatures=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        uniform_u=np.asarray([0.25, 0.50, 0.75, 0.99], dtype=np.float32),
        build_dir=tmp_path / "qwen3_unfiltered_build",
    )


def test_sample_tokens_gpt_oss_generation(tmp_path) -> None:
    """GPT-OSS full vocab (201088) with typical chat generation settings.

    GPT-OSS uses vocab_size=201088, TP=8.  The gathered vocab after
    all-gather is 201088.  Tests both the default serving scenario
    (temperature=0.6, top_p=0.95 — common for chat) and a mixed batch
    with greedy + creative rows.
    """
    batch_size = 4
    vocab_size = 201088
    logits = _make_logits(3001, batch_size=batch_size, vocab_size=vocab_size)

    _assert_sampler_matches_reference(
        logits=logits,
        # Row 0: typical chat (temp=0.6, top_p=0.95, top_k=40)
        # Row 1: greedy (top_k=1)
        # Row 2: creative writing (temp=1.2, top_p=0.9, min_p=0.05)
        # Row 3: default (temp=1.0, no filtering)
        temperatures=np.asarray([0.6, 0.1, 1.2, 1.0], dtype=np.float32),
        top_ks=np.asarray([40, 1, vocab_size, vocab_size], dtype=np.int32),
        top_ps=np.asarray([0.95, 1.0, 0.90, 1.0], dtype=np.float32),
        min_ps=np.asarray([0.0, 0.0, 0.05, 0.0], dtype=np.float32),
        uniform_u=np.asarray([0.42, 0.50, 0.87, 0.15], dtype=np.float32),
        build_dir=tmp_path / "gpt_oss_build",
    )
