"""Engine bootstrap helpers for HTTP server launch."""

import multiprocessing as mp
import os
import time
from contextlib import suppress
from dataclasses import asdict, dataclass

import zmq

from nkipy_serving.config import (
    RuntimeConfig,
    configure_runtime_environment,
    load_runtime_config,
    validate_runtime_config,
)
from nkipy_serving.managers.detokenizer_manager import run_detokenizer_process
from nkipy_serving.managers.scheduler import run_scheduler_process
from nkipy_serving.managers.tokenizer_manager import TokenizerManager
from nkipy_serving.server_args import ServerArgs

_DEFAULT_READY_TIMEOUT_S = 1800


@dataclass(frozen=True)
class PortArgs:
    """IPC endpoint names for ZMQ sockets."""

    scheduler_input_ipc_name: str
    scheduler_output_ipc_name: str
    detokenizer_ipc_name: str

    @staticmethod
    def create(suffix: str | None = None) -> "PortArgs":
        pid = os.getpid()
        tag = suffix or str(pid)
        return PortArgs(
            scheduler_input_ipc_name=f"ipc:///tmp/nkipy_serving_{tag}_in",
            scheduler_output_ipc_name=f"ipc:///tmp/nkipy_serving_{tag}_out",
            detokenizer_ipc_name=f"ipc:///tmp/nkipy_serving_{tag}_detok",
        )


@dataclass
class RuntimeProcessGroup:
    scheduler_process: mp.Process
    detokenizer_process: mp.Process | None
    port_args: PortArgs
    _zmq_context: zmq.Context | None = None
    _send_socket: zmq.Socket | None = None

    def _ensure_send_socket(self) -> zmq.Socket:
        if self._send_socket is None:
            self._zmq_context = zmq.Context()
            self._send_socket = self._zmq_context.socket(zmq.PUSH)
            self._send_socket.connect(self.port_args.scheduler_input_ipc_name)
        return self._send_socket

    def shutdown(self) -> None:
        # Best-effort shutdown still terminates child processes below.
        with suppress(zmq.ZMQError):
            sock = self._ensure_send_socket()
            sock.send_pyobj({"cmd": "shutdown"})

        deadline = time.time() + 10.0
        remaining = max(0.0, deadline - time.time())
        self.scheduler_process.join(timeout=remaining if remaining > 0 else 0.1)
        if self.scheduler_process.is_alive():
            self.scheduler_process.terminate()
            self.scheduler_process.join(timeout=1.0)

        if self.detokenizer_process is not None:
            self.detokenizer_process.terminate()
            self.detokenizer_process.join(timeout=2.0)

        if self._send_socket is not None:
            with suppress(zmq.ZMQError):
                self._send_socket.close(linger=0)
        if self._zmq_context is not None:
            with suppress(zmq.ZMQError):
                self._zmq_context.term()


@dataclass(frozen=True)
class EngineInitResult:
    tokenizer_manager: TokenizerManager
    runtime_config: RuntimeConfig
    variant_count: int
    warmup_summary: dict[str, object] | None
    process_group: RuntimeProcessGroup


def _launch_runtime_process_group(
    runtime_config: RuntimeConfig,
) -> tuple[RuntimeProcessGroup, int, dict[str, object] | None]:
    ctx = mp.get_context("spawn")
    port_args = PortArgs.create()
    ready_reader, ready_writer = ctx.Pipe(duplex=False)

    runtime_config_dict = asdict(runtime_config)
    port_args_dict = asdict(port_args)

    # Spawn detokenizer first — it must bind before the scheduler connects.
    detokenizer_ready = ctx.Event()
    detokenizer_process = ctx.Process(
        target=run_detokenizer_process,
        args=(runtime_config_dict, port_args_dict, detokenizer_ready),
        daemon=True,
    )
    detokenizer_process.start()

    # Wait for detokenizer to bind ZMQ sockets before starting the scheduler.
    if not detokenizer_ready.wait(timeout=30):
        detokenizer_process.terminate()
        raise RuntimeError("Detokenizer process failed to start within 30 seconds")

    scheduler_process = ctx.Process(
        target=run_scheduler_process,
        args=(
            runtime_config_dict,
            port_args_dict,
            ready_writer,
        ),
    )
    scheduler_process.start()
    ready_writer.close()

    process_group = RuntimeProcessGroup(
        scheduler_process=scheduler_process,
        detokenizer_process=detokenizer_process,
        port_args=port_args,
    )

    ready_timeout_s = int(
        os.getenv(
            "NKIPY_SERVING_SCHEDULER_READY_TIMEOUT_S", str(_DEFAULT_READY_TIMEOUT_S)
        )
    )
    if not ready_reader.poll(timeout=ready_timeout_s):
        process_group.shutdown()
        raise RuntimeError("Timed out waiting for scheduler subprocess to become ready")

    ready_info = ready_reader.recv()
    status = str(ready_info.get("status", ""))
    if status != "ready":
        process_group.shutdown()
        raise RuntimeError(
            "Scheduler subprocess initialization failed: "
            f"{ready_info.get('error', 'unknown error')}"
        )

    variant_count = int(ready_info.get("variant_count", 0))
    warmup_summary = ready_info.get("warmup_summary")
    if not isinstance(warmup_summary, dict):
        warmup_summary = None
    return process_group, variant_count, warmup_summary


def _launch_subprocesses_or_threads(server_args: ServerArgs) -> EngineInitResult:
    """Bootstrap runtime components with scheduler subprocess."""
    server_args.check_server_args()

    overrides = {}
    if server_args.max_model_len is not None:
        overrides["max_context_len"] = server_args.max_model_len
    if server_args.device_offset is not None:
        overrides["device_offset"] = server_args.device_offset
    runtime_config = load_runtime_config(
        config_path=server_args.config_path,
        overrides=overrides or None,
    )
    validate_runtime_config(runtime_config)
    configure_runtime_environment(runtime_config)

    process_group, variant_count, warmup_summary = _launch_runtime_process_group(
        runtime_config
    )
    tokenizer_manager = TokenizerManager(
        runtime_config=runtime_config,
        port_args=asdict(process_group.port_args),
    )
    return EngineInitResult(
        tokenizer_manager=tokenizer_manager,
        runtime_config=runtime_config,
        variant_count=variant_count,
        warmup_summary=warmup_summary,
        process_group=process_group,
    )
