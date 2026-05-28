# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for concurrent (broadcast) push support in NKIPyWorker.

Tests the worker-level push thread management without requiring Neuron
hardware or RDMA connections.  Uses a minimal stub class that replicates
the push thread logic to avoid importing torch/vllm (which may not be
available in the test environment).
"""

import threading
import time

import pytest


class _PushThreadManager:
    """Extracted push-thread logic mirroring NKIPyWorker for testability.

    This replicates the exact thread management from worker.py without
    importing torch or vllm dependencies.
    """

    def __init__(self):
        self._push_threads: list[threading.Thread] = []
        self._push_errors: list[BaseException] = []
        self._push_lock = threading.Lock()
        self.rank = 0

    def push_weights(self, push_fn) -> dict:
        """Start push_fn in a background thread (mirrors nkipy_push_weights)."""
        def _bg_push():
            try:
                push_fn()
            except BaseException as exc:
                with self._push_lock:
                    self._push_errors.append(exc)

        t = threading.Thread(target=_bg_push, daemon=True)
        t.start()
        with self._push_lock:
            self._push_threads.append(t)
        return {"status": "started"}

    def push_status(self) -> dict:
        """Mirrors nkipy_push_status."""
        with self._push_lock:
            alive = [t for t in self._push_threads if t.is_alive()]
            if alive:
                self._push_threads = alive
                return {"status": "running"}
            self._push_threads = []
            if self._push_errors:
                err = self._push_errors[0]
                self._push_errors = []
                return {"status": "error", "message": str(err)}
        return {"status": "done"}

    def wait_all_pushes(self):
        """Mirrors the push-wait logic in nkipy_sleep."""
        with self._push_lock:
            threads = list(self._push_threads)
        for t in threads:
            t.join()
        with self._push_lock:
            self._push_threads = []
            self._push_errors = []


@pytest.fixture
def manager():
    return _PushThreadManager()


class TestConcurrentPush:

    def test_single_push_starts_thread(self, manager):
        """A single push starts one background thread."""
        manager.push_weights(lambda: time.sleep(0.05))
        with manager._push_lock:
            assert len(manager._push_threads) == 1
        manager.wait_all_pushes()

    def test_multiple_concurrent_pushes(self, manager):
        """Multiple pushes spawn independent threads that run concurrently."""
        push_count = 5
        barrier = threading.Barrier(push_count, timeout=5)
        completed = []

        def _concurrent_push():
            barrier.wait()
            completed.append(threading.current_thread().ident)

        for _ in range(push_count):
            result = manager.push_weights(_concurrent_push)
            assert result["status"] == "started"

        with manager._push_lock:
            assert len(manager._push_threads) == push_count

        manager.wait_all_pushes()
        assert len(completed) == push_count

    def test_push_status_running_while_threads_alive(self, manager):
        """Status returns 'running' while any push thread is alive."""
        event = threading.Event()
        manager.push_weights(lambda: event.wait(timeout=5))
        manager.push_weights(lambda: event.wait(timeout=5))

        status = manager.push_status()
        assert status["status"] == "running"

        event.set()
        time.sleep(0.1)

        status = manager.push_status()
        assert status["status"] == "done"

    def test_push_status_reports_first_error(self, manager):
        """If any push fails, status reports the error after all complete."""
        call_count = [0]
        lock = threading.Lock()

        def _maybe_fail():
            with lock:
                call_count[0] += 1
                idx = call_count[0]
            if idx == 2:
                raise RuntimeError("RDMA write failed")

        for _ in range(3):
            manager.push_weights(_maybe_fail)

        time.sleep(0.3)

        status = manager.push_status()
        assert status["status"] == "error"
        assert "RDMA write failed" in status["message"]

    def test_push_status_done_clears_state(self, manager):
        """After reporting 'done', subsequent calls also return 'done'."""
        manager.push_weights(lambda: None)
        time.sleep(0.1)

        status = manager.push_status()
        assert status["status"] == "done"

        status = manager.push_status()
        assert status["status"] == "done"

    def test_wait_all_blocks_until_complete(self, manager):
        """wait_all_pushes blocks until all concurrent threads finish."""
        events = [threading.Event() for _ in range(3)]
        completed = []

        for i, event in enumerate(events):
            def _blocking(e=event, idx=i):
                e.wait(timeout=5)
                completed.append(idx)
            manager.push_weights(_blocking)

        with manager._push_lock:
            assert len(manager._push_threads) == 3

        for e in events:
            e.set()

        manager.wait_all_pushes()

        assert len(completed) == 3
        with manager._push_lock:
            assert len(manager._push_threads) == 0

    def test_error_cleared_after_wait_all(self, manager):
        """Errors are cleared by wait_all_pushes."""
        manager.push_weights(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        time.sleep(0.1)
        manager.wait_all_pushes()

        with manager._push_lock:
            assert len(manager._push_errors) == 0

    def test_new_pushes_after_previous_complete(self, manager):
        """New pushes can start after previous ones complete."""
        manager.push_weights(lambda: None)
        time.sleep(0.1)
        assert manager.push_status()["status"] == "done"

        # Start a second batch
        manager.push_weights(lambda: None)
        time.sleep(0.1)
        assert manager.push_status()["status"] == "done"
