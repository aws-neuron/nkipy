"""Contract tests for profiling helpers."""

from __future__ import annotations

import json

from nkipy_serving.profiling import ProfileWriter, StepTimer


def _make_writer(path, *, flush_every=50):
    writer = ProfileWriter.__new__(ProfileWriter)
    writer._path = path
    writer._fh = open(path, "a")
    writer._flush_every = flush_every
    writer._count = 0
    return writer


def test_profile_writer_jsonl_contract(tmp_path):
    path = tmp_path / "profile.jsonl"
    writer = _make_writer(path, flush_every=2)

    writer.write({"step": 1, "path": tmp_path})
    writer.write({"step": 2, "value": 0.5})
    writer.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {"step": 1, "path": str(tmp_path)},
        {"step": 2, "value": 0.5},
    ]


def test_step_timer_elapsed_contract():
    assert list(StepTimer().elapsed()) == ["t_total"]

    timer = StepTimer()
    timer.mark("batch_build")
    timer.mark("dispatch")
    result = timer.elapsed()

    assert set(result) == {"t_batch_build", "t_dispatch", "t_total"}
    assert result["t_total"] >= result["t_batch_build"] + result["t_dispatch"]
    assert all(value == round(value, 6) and value >= 0.0 for value in result.values())
