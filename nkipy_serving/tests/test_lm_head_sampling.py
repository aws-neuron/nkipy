from __future__ import annotations

import numpy as np

import nkipy_serving.ops.lm_head_sampling as ref_math
import nkipy_serving.sampling.lm_head_sampling as sampling_module


def test_sample_from_probs_contracts() -> None:
    cases = [
        (
            np.asarray(
                [
                    [0.10, 0.20, 0.70],
                    [0.50, 0.30, 0.20],
                    [0.40, 0.35, 0.25],
                ],
                dtype=np.float32,
            ),
            np.asarray([3, 3, 3], dtype=np.int32),
            np.asarray([0.05, 0.65, 0.99], dtype=np.float32),
            np.asarray([0, 1, 2], dtype=np.int32),
        ),
        (
            np.asarray(
                [
                    [0.15, 0.75, 0.10],
                    [0.05, 0.20, 0.75],
                ],
                dtype=np.float32,
            ),
            np.asarray([1, 1], dtype=np.int32),
            np.asarray([0.30, 0.70], dtype=np.float32),
            np.asarray([1, 2], dtype=np.int32),
        ),
    ]
    for probs, top_ks, uniform_u, expected in cases:
        sampled = ref_math._sample_from_probs(
            probs,
            top_ks=top_ks,
            top_ps=np.ones_like(top_ks, dtype=np.float32),
            min_ps=np.zeros_like(top_ks, dtype=np.float32),
            uniform_u=uniform_u,
        )
        assert np.array_equal(sampled, expected)

    rng = np.random.default_rng(99)
    probs = np.asarray([[0.05, 0.10, 0.60, 0.25]], dtype=np.float32)
    for _ in range(50):
        sampled = ref_math._sample_from_probs(
            probs,
            top_ks=np.asarray([1], dtype=np.int32),
            top_ps=np.asarray([1.0], dtype=np.float32),
            min_ps=np.asarray([0.0], dtype=np.float32),
            uniform_u=np.asarray([np.float32(rng.random())], dtype=np.float32),
        )
        assert sampled[0] == np.int32(2)


def test_sampled_tokens_respect_filter_allowed_sets() -> None:
    """Samples must stay inside the active top-k, top-p, and min-p filters."""
    cases = []
    vocab = 100

    rng = np.random.default_rng(42)
    probs = rng.dirichlet(np.ones(vocab), size=1).astype(np.float32)
    k = 10
    cases.append(
        (
            rng,
            probs,
            np.asarray([k], dtype=np.int32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            set(np.argsort(probs[0])[-k:].tolist()),
            "top-k",
        )
    )

    rng = np.random.default_rng(77)
    probs = rng.dirichlet(np.ones(vocab), size=1).astype(np.float32)
    p = np.float32(0.5)
    topp_set = set()
    cum = np.float32(0.0)
    for idx in np.argsort(probs[0])[::-1]:
        topp_set.add(int(idx))
        cum += probs[0, idx]
        if cum >= p:
            break
    cases.append(
        (
            rng,
            probs,
            np.asarray([vocab], dtype=np.int32),
            np.asarray([p], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            topp_set,
            "top-p",
        )
    )

    rng = np.random.default_rng(55)
    probs = rng.dirichlet(np.ones(vocab), size=1).astype(np.float32)
    min_p = np.float32(0.1)
    minp_set = set(np.where(probs[0] >= float(np.max(probs)) * float(min_p))[0])
    cases.append(
        (
            rng,
            probs,
            np.asarray([vocab], dtype=np.int32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([min_p], dtype=np.float32),
            minp_set,
            "min-p",
        )
    )

    for rng, probs, top_ks, top_ps, min_ps, allowed, label in cases:
        for _ in range(200):
            sampled = ref_math._sample_from_probs(
                probs,
                top_ks=top_ks,
                top_ps=top_ps,
                min_ps=min_ps,
                uniform_u=np.asarray([np.float32(rng.random())], dtype=np.float32),
            )
            assert int(sampled[0]) in allowed, (
                f"sampled {sampled[0]} outside {label} set"
            )


def test_uniform_logits_samples_all_positions() -> None:
    """With uniform probs, sweeping u from 0->1 should cover all tokens."""
    vocab = 20
    probs = np.full((1, vocab), 1.0 / vocab, dtype=np.float32)
    seen = set()
    for i in range(1, 200):
        u = np.float32(i / 200.0)
        sampled = ref_math._sample_from_probs(
            probs,
            top_ks=np.asarray([vocab], dtype=np.int32),
            top_ps=np.asarray([1.0], dtype=np.float32),
            min_ps=np.asarray([0.0], dtype=np.float32),
            uniform_u=np.asarray([u], dtype=np.float32),
        )
        seen.add(int(sampled[0]))
    assert len(seen) == vocab, (
        f"Expected all {vocab} tokens, got {len(seen)}: {sorted(seen)}"
    )


def test_seed_determinism_via_draw_uniform() -> None:
    """Same seed + same position -> same uniform -> same sample."""
    from nkipy_serving.sampling.random_state import draw_uniform

    u_a = draw_uniform(seed=42, position=7)
    u_b = draw_uniform(seed=42, position=7)
    assert u_a == u_b

    # Different position -> different draw.
    u_c = draw_uniform(seed=42, position=8)
    assert u_a != u_c


def test_device_sampling_batch_needs_filtering_contracts() -> None:
    from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
    from nkipy_serving.sampling.params import TOP_K_ALL

    cases = [
        (
            np.asarray([TOP_K_ALL, TOP_K_ALL], dtype=np.int32),
            np.asarray([1.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float32),
            False,
        ),
        (
            np.asarray([40], dtype=np.int32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            True,
        ),
        (
            np.asarray([TOP_K_ALL], dtype=np.int32),
            np.asarray([0.95], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            True,
        ),
        (
            np.asarray([TOP_K_ALL], dtype=np.int32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0.05], dtype=np.float32),
            True,
        ),
    ]

    for top_ks, top_ps, min_ps, expected in cases:
        batch = DeviceSamplingBatch(
            use_full_sampler=True,
            temperatures=np.ones_like(top_ps, dtype=np.float32),
            top_ks=top_ks,
            top_ps=top_ps,
            min_ps=min_ps,
            uniform_u=np.full_like(top_ps, 0.5, dtype=np.float32),
        )
        assert batch.needs_filtering is expected


def test_lm_head_sample_tokens_uses_shared_sampler(monkeypatch) -> None:
    captured: dict[str, np.ndarray] = {}

    def fake_sampler(
        logits, temperatures, top_ks, top_ps, min_ps, uniform_u, *, _unfiltered=False
    ):
        captured["logits"] = np.asarray(logits, dtype=np.float32)
        captured["temperatures"] = np.asarray(temperatures, dtype=np.float32)
        captured["top_ks"] = np.asarray(top_ks, dtype=np.int32)
        captured["top_ps"] = np.asarray(top_ps, dtype=np.float32)
        captured["min_ps"] = np.asarray(min_ps, dtype=np.float32)
        captured["uniform_u"] = np.asarray(uniform_u, dtype=np.float32)
        captured["_unfiltered"] = _unfiltered
        return np.asarray([7, 3], dtype=np.int32)

    monkeypatch.setattr(sampling_module, "sample_tokens", fake_sampler)

    hidden = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    final_norm = np.asarray([1.0, 1.0], dtype=np.float32)
    lm_head = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    sampled = sampling_module.lm_head_sample_tokens(
        hidden,
        final_norm,
        lm_head,
        last_token_indices=np.asarray([1, 2], dtype=np.int32),
        temperatures=np.asarray([0.7, 0.8], dtype=np.float32),
        top_ks=np.asarray([5, 9], dtype=np.int32),
        top_ps=np.asarray([0.9, 0.95], dtype=np.float32),
        min_ps=np.asarray([0.0, 0.05], dtype=np.float32),
        uniform_u=np.asarray([0.2, 0.4], dtype=np.float32),
        rms_norm_eps=1e-6,
        tp_degree=1,
    )

    assert np.array_equal(sampled, np.asarray([7, 3], dtype=np.int32))
    assert captured["logits"].shape == (2, 3)
    assert captured["temperatures"].shape == (2, 1)
    assert captured["top_ks"].shape == (2, 1)
    assert captured["top_ps"].shape == (2, 1)
    assert captured["min_ps"].shape == (2, 1)
    assert captured["uniform_u"].shape == (2, 1)
