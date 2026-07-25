from __future__ import annotations

import zmq

from nkipy_serving.entrypoints.engine import PortArgs, RuntimeProcessGroup


class _FakeProcess:
    def __init__(self) -> None:
        self.join_calls: list[float | None] = []
        self.terminated = False

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        self.terminated = True


def test_runtime_process_group_shutdown_ignores_zmq_send_failure() -> None:
    scheduler = _FakeProcess()
    detokenizer = _FakeProcess()
    group = RuntimeProcessGroup(
        scheduler_process=scheduler,
        detokenizer_process=detokenizer,
        port_args=PortArgs.create("test-engine-shutdown"),
    )

    class _FailingSocket:
        def send_pyobj(self, _payload) -> None:
            raise zmq.ZMQError("send failed")

        def close(self, *, linger: int) -> None:
            raise zmq.ZMQError("close failed")

    class _FailingContext:
        def term(self) -> None:
            raise zmq.ZMQError("term failed")

    group._send_socket = _FailingSocket()
    group._zmq_context = _FailingContext()

    group.shutdown()

    assert scheduler.join_calls
    assert detokenizer.terminated is True
