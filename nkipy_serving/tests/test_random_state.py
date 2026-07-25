"""Contract tests for stateless per-request RNG."""

from __future__ import annotations

import numpy as np

from nkipy_serving.sampling.random_state import assign_seed, draw_uniform


def test_seed_assignment_contracts() -> None:
    assert assign_seed(42) == 42
    assert assign_seed(0) == 0

    s = assign_seed(None)
    assert isinstance(s, int)
    assert s >= 0


def test_draw_uniform_stateless_contract() -> None:
    seed = 77
    forward = [draw_uniform(seed, pos) for pos in range(50)]
    reverse = [draw_uniform(seed, pos) for pos in reversed(range(50))]
    reverse.reverse()
    assert forward == reverse

    values = np.asarray(
        [draw_uniform(seed, pos) for pos in range(100)], dtype=np.float32
    )
    assert values.min() >= np.float32(1e-7)
    assert values.max() <= np.float32(1.0) - np.float32(1e-7)
    assert len(set(float(v) for v in values)) == len(values)
