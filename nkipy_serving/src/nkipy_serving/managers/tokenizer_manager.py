import asyncio
import dataclasses
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import zmq
import zmq.asyncio

from nkipy_serving.config import RuntimeConfig
from nkipy_serving.managers.io_struct import GenerateReqInput
from nkipy_serving.profiling import PROFILING_ENABLED, ProfileWriter
from nkipy_serving.runtime import PrecompilePaddings, build_precompile_paddings
from nkipy_serving.tokenization.hf_tokenizer import HfTokenizer

if TYPE_CHECKING:
    from nkipy_serving.runtime.worker_coordinator import WorkerCoordinator

_DEFAULT_SCHEDULER_TIMEOUT_S = 1800


class SchedulerError(RuntimeError):
    """Raised when the scheduler returns an error for a request."""

    def __init__(self, message: str, aborted: bool = False):
        super().__init__(message)
        self.aborted = aborted


@dataclass
class _ProxyRequestState:
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    submit_ts: float = 0.0
    first_token_ts: float = 0.0
    token_count: int = 0


class TokenizerManager:
    """Tokenizer + runtime adapter manager (Python-first, no JAX/Torch)."""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        kernel_adapter: object | None = None,
        worker_coordinator: "WorkerCoordinator | None" = None,
        scheduler_timeout_s: int | None = None,
        # ZMQ-based IPC
        port_args: dict[str, str] | None = None,
    ):
        self.runtime_config = runtime_config
        self.kernel_adapter = kernel_adapter
        self.worker_coordinator = worker_coordinator
        self.precompile_paddings: PrecompilePaddings = build_precompile_paddings(
            runtime_config
        )

        # Proxy mode requires ZMQ port_args.
        self._proxy_mode = port_args is not None

        if self._proxy_mode:
            self._scheduler_timeout_s = (
                int(scheduler_timeout_s)
                if scheduler_timeout_s is not None
                else int(
                    os.getenv(
                        "NKIPY_SERVING_SCHEDULER_TIMEOUT_S",
                        str(_DEFAULT_SCHEDULER_TIMEOUT_S),
                    )
                )
            )
            # Keep a local tokenizer instance for prompt formatting (e.g. HF chat templates).
            # Actual tokenization/detokenization for generation happens in the scheduler process.
            self.tokenizer = HfTokenizer(
                model_id=runtime_config.tokenizer_model_id,
                revision=runtime_config.tokenizer_revision,
                local_files_only=runtime_config.tokenizer_local_files_only,
            )

            self._port_args = port_args
            self._zmq_context: zmq.asyncio.Context | None = None
            self._zmq_send_socket: zmq.asyncio.Socket | None = None
            self._zmq_recv_socket: zmq.asyncio.Socket | None = None

            self._proxy_response_router_task: asyncio.Task | None = None
            self._proxy_request_states: dict[str, _ProxyRequestState] = {}
            self._proxy_request_states_lock = asyncio.Lock()
            self._proxy_control_waiters: dict[str, asyncio.Future] = {}
            self._proxy_control_waiters_lock = asyncio.Lock()
        else:
            raise RuntimeError(
                "TokenizerManager requires proxy mode (port_args). "
                "Local execution mode is no longer supported."
            )
        self.served_model_name = runtime_config.model_id
        self.last_receive_tstamp = time.time()

        # HTTP-layer profiling (gated by NKIPY_SERVING_PROFILE=1).
        self._http_profile_writer: ProfileWriter | None = None
        if PROFILING_ENABLED:
            self._http_profile_writer = ProfileWriter("http_events")

    # -- ZMQ socket management --

    def _ensure_zmq_sockets(self) -> tuple[zmq.asyncio.Socket, zmq.asyncio.Socket]:
        """Lazily create ZMQ async sockets."""
        if self._zmq_send_socket is None:
            self._zmq_context = zmq.asyncio.Context()
            self._zmq_send_socket = self._zmq_context.socket(zmq.PUSH)
            self._zmq_send_socket.connect(self._port_args["scheduler_input_ipc_name"])
            self._zmq_recv_socket = self._zmq_context.socket(zmq.PULL)
            self._zmq_recv_socket.connect(self._port_args["scheduler_output_ipc_name"])
        return self._zmq_send_socket, self._zmq_recv_socket

    def run_precompile_warmup(self) -> dict[str, object]:
        """No-op warmup in the new architecture (scheduler drives forward)."""
        return {
            "max_padded_batch_size": self.precompile_paddings.max_padded_batch_size,
            "max_padded_num_tokens": self.precompile_paddings.max_padded_num_tokens,
        }

    async def _ensure_proxy_response_router_started(self) -> None:
        if not self._proxy_mode:
            raise RuntimeError("Proxy response router is only available in proxy mode")
        if self._proxy_response_router_task is None:
            self._proxy_response_router_task = asyncio.create_task(
                self._proxy_response_router_loop()
            )

    async def _proxy_response_router_loop(self) -> None:
        await self._proxy_response_router_loop_zmq()

    async def _route_response(self, response: object) -> None:
        """Route a single response to the appropriate waiter or request state."""
        if not isinstance(response, dict):
            return

        # Timestamp when this response arrived at the HTTP process.
        if PROFILING_ENABLED:
            response["_http_recv_ts"] = time.time()

        control_id = response.get("control_id")
        if isinstance(control_id, str) and control_id:
            async with self._proxy_control_waiters_lock:
                waiter = self._proxy_control_waiters.pop(control_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(response)
            return

        request_id = str(response.get("request_id", ""))
        if not request_id:
            return

        async with self._proxy_request_states_lock:
            state = self._proxy_request_states.get(request_id)
        if state is not None:
            await state.event_queue.put(response)

    async def _proxy_response_router_loop_zmq(self) -> None:
        """Response router using ZMQ PULL socket."""
        _, recv_socket = self._ensure_zmq_sockets()
        poller = zmq.asyncio.Poller()
        poller.register(recv_socket, zmq.POLLIN)
        while True:
            try:
                events = await poller.poll(timeout=1000)
                if not events:
                    continue
                response = await recv_socket.recv_pyobj(zmq.NOBLOCK)
            except asyncio.CancelledError:
                return
            except zmq.ZMQError:
                continue
            await self._route_response(response)

    async def _register_proxy_request(self, request_id: str) -> _ProxyRequestState:
        state = _ProxyRequestState()
        async with self._proxy_request_states_lock:
            if request_id in self._proxy_request_states:
                raise RuntimeError(
                    f"Duplicate proxy request_id registration: {request_id}"
                )
            self._proxy_request_states[request_id] = state
        return state

    async def _unregister_proxy_request(self, request_id: str) -> None:
        async with self._proxy_request_states_lock:
            self._proxy_request_states.pop(request_id, None)

    async def _send_scheduler_payload(self, payload: dict[str, Any]) -> None:
        send_socket, _ = self._ensure_zmq_sockets()
        await send_socket.send_pyobj(payload)

    async def _start_proxy_generation(
        self, req: GenerateReqInput, request_id: str | None = None
    ) -> tuple[str, _ProxyRequestState]:
        await self._ensure_proxy_response_router_started()
        request_id = str(request_id) if request_id is not None else uuid.uuid4().hex
        state = await self._register_proxy_request(request_id)
        state.submit_ts = time.time()
        try:
            # Pre-tokenize text prompts so the scheduler thread never blocks on
            # tokenization.  The scheduler already accepts ``input_ids`` and skips
            # its own encode path when the field is populated.  Keep prompt/text
            # intact so they flow through to the response metadata unchanged.
            if req.input_ids is None:
                prompt_text = req.prompt if req.prompt is not None else req.text
                if prompt_text is not None:
                    token_ids = self.tokenizer.encode(prompt_text)
                    req = dataclasses.replace(req, input_ids=token_ids.tolist())
            payload = {
                "cmd": "generate",
                "request_id": request_id,
                "req": asdict(req),
            }
            await self._send_scheduler_payload(payload)
        except BaseException:
            await self._unregister_proxy_request(request_id)
            raise
        return request_id, state

    async def _await_proxy_final(
        self, request_id: str, state: _ProxyRequestState
    ) -> dict[str, Any]:
        while True:
            try:
                response = await asyncio.wait_for(
                    state.event_queue.get(), timeout=float(self._scheduler_timeout_s)
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Timed out waiting for scheduler response") from exc

            message_type = str(response.get("type", ""))
            if message_type != "final":
                continue
            if not bool(response.get("ok", False)):
                raise SchedulerError(
                    f"Scheduler generation failed: {response.get('error', 'unknown error')}",
                    aborted=bool(response.get("aborted", False)),
                )
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Scheduler response missing result payload")
            return result

    async def _request_scheduler_control(
        self,
        cmd: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self._ensure_proxy_response_router_started()
        control_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        async with self._proxy_control_waiters_lock:
            self._proxy_control_waiters[control_id] = future

        try:
            message = {"cmd": cmd, "control_id": control_id}
            if payload is not None:
                message.update(payload)
            await self._send_scheduler_payload(message)
            response = await asyncio.wait_for(
                future, timeout=float(self._scheduler_timeout_s)
            )
        except asyncio.TimeoutError as exc:
            async with self._proxy_control_waiters_lock:
                self._proxy_control_waiters.pop(control_id, None)
            raise RuntimeError(
                f"Timed out waiting for scheduler {cmd} response"
            ) from exc
        except BaseException:
            async with self._proxy_control_waiters_lock:
                self._proxy_control_waiters.pop(control_id, None)
            raise

        if not bool(response.get("ok", False)):
            raise RuntimeError(
                f"Scheduler control cmd failed: cmd={cmd}, error={response.get('error', 'unknown error')}"
            )
        return response

    async def abort_request(self, request_id: str) -> None:
        if not self._proxy_mode:
            return
        await self._request_scheduler_control(
            cmd="abort",
            payload={"request_id": str(request_id)},
        )

    async def pause_generation(self) -> None:
        if not self._proxy_mode:
            return
        await self._request_scheduler_control(cmd="pause")

    async def continue_generation(self) -> None:
        if not self._proxy_mode:
            return
        await self._request_scheduler_control(cmd="resume")

    async def flush_cache(self, abort_all_requests: bool = True) -> dict[str, Any]:
        if not self._proxy_mode:
            return {"aborted_count": 0}
        response = await self._request_scheduler_control(
            cmd="flush_cache",
            payload={"abort_all_requests": bool(abort_all_requests)},
        )
        return {"aborted_count": int(response.get("aborted_count", 0))}

    async def reload_weights_from_disk(
        self,
        model_path: str,
        *,
        abort_all_requests: bool = True,
    ) -> dict[str, Any]:
        if not self._proxy_mode:
            raise RuntimeError("Weight reload requires scheduler proxy mode")
        response = await self._request_scheduler_control(
            cmd="reload_weights_from_disk",
            payload={
                "model_path": str(model_path),
                "abort_all_requests": bool(abort_all_requests),
            },
        )
        return {"aborted_count": int(response.get("aborted_count", 0))}

    async def checkpoint_request_state(
        self,
        request_id: str,
        *,
        num_tokens: int,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        """Checkpoint DSV4 request-owned state for an internal rollback policy."""
        if not self._proxy_mode:
            raise RuntimeError("Request-state checkpoint requires scheduler proxy mode")
        payload: dict[str, Any] = {
            "request_id": str(request_id),
            "num_tokens": int(num_tokens),
        }
        if checkpoint_id is not None:
            payload["checkpoint_id"] = str(checkpoint_id)
        response = await self._request_scheduler_control(
            cmd="checkpoint_request_state",
            payload=payload,
        )
        return {
            "checkpoint_id": str(response.get("checkpoint_id", "")),
            "request_id": str(response.get("request_id", request_id)),
            "owner_id": int(response.get("owner_id", -1)),
            "seq_len": int(response.get("seq_len", 0)),
            "num_tokens": int(response.get("num_tokens", num_tokens)),
        }

    async def restore_request_state(self, checkpoint_id: str) -> dict[str, Any]:
        """Restore a DSV4 request-state checkpoint created before speculation."""
        if not self._proxy_mode:
            raise RuntimeError("Request-state restore requires scheduler proxy mode")
        response = await self._request_scheduler_control(
            cmd="restore_request_state",
            payload={"checkpoint_id": str(checkpoint_id)},
        )
        return {
            "checkpoint_id": str(response.get("checkpoint_id", checkpoint_id)),
            "request_id": str(response.get("request_id", "")),
            "owner_id": int(response.get("owner_id", -1)),
            "seq_len": int(response.get("seq_len", 0)),
        }

    async def get_scheduler_metrics(self) -> dict[str, Any]:
        if not self._proxy_mode:
            return {
                "proxy_mode": False,
                "last_receive_tstamp": self.last_receive_tstamp,
            }
        response = await self._request_scheduler_control(cmd="get_metrics")
        metrics = response.get("metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError("Scheduler get_metrics response missing metrics payload")
        return metrics

    async def get_lane_metadata(self) -> dict[str, Any]:
        """Per-rank DP-attention lane / group metadata (DeepSeek-V4 path)."""
        if not self._proxy_mode:
            return {"proxy_mode": False, "lane_metadata": {}, "lane_routes": []}
        response = await self._request_scheduler_control(cmd="get_lane_metadata")
        return {
            "lane_metadata": response.get("lane_metadata", {}),
            "lane_routes": response.get("lane_routes", []),
            "attention_dp_degree": int(response.get("attention_dp_degree", 1)),
            "tp_degree": int(response.get("tp_degree", 1)),
            "ep_degree": int(response.get("ep_degree", 1)),
            "replica_degree": int(response.get("replica_degree", 1)),
            "total_workers": int(response.get("total_workers", 1)),
        }

    def _write_completion_profile(
        self,
        request_id: str,
        state: _ProxyRequestState,
        result: dict,
        *,
        stream: bool,
    ) -> None:
        pw = self._http_profile_writer
        if pw is None:
            return
        complete_ts = time.time()
        first_scheduled_ts = float(result.get("first_scheduled_ts", 0.0))
        first_token_ts = (
            float(result.get("first_token_ts", 0.0))
            if not stream
            else state.first_token_ts
        )
        entry: dict[str, object] = {
            "event": "request_completed",
            "request_id": request_id,
            "submit_ts": state.submit_ts,
            "first_scheduled_ts": first_scheduled_ts
            if first_scheduled_ts > 0
            else None,
            "first_token_ts": first_token_ts if first_token_ts > 0 else None,
            "complete_ts": complete_ts,
            "ttft_ms": round((first_token_ts - state.submit_ts) * 1000, 3)
            if first_token_ts > 0
            else None,
            "scheduled_ttft_ms": round((first_token_ts - first_scheduled_ts) * 1000, 3)
            if first_token_ts > 0 and first_scheduled_ts > 0
            else None,
            "total_ms": round((complete_ts - state.submit_ts) * 1000, 3),
            "prompt_tokens": int(result.get("prompt_tokens", 0)),
            "completion_tokens": int(result.get("completion_tokens", 0)),
        }
        if stream:
            entry["token_count"] = state.token_count
        else:
            entry["stream"] = False
        pw.write(entry)

    async def generate_once(
        self, req: GenerateReqInput, request_id: str | None = None
    ) -> dict:
        self.last_receive_tstamp = time.time()
        request_id, state = await self._start_proxy_generation(
            req, request_id=request_id
        )
        try:
            result = await self._await_proxy_final(request_id=request_id, state=state)

            self._write_completion_profile(request_id, state, result, stream=False)
            return result
        finally:
            await self._unregister_proxy_request(request_id)

    async def generate_stream(
        self, req: GenerateReqInput, request_id: str | None = None
    ):
        self.last_receive_tstamp = time.time()
        request_id, state = await self._start_proxy_generation(
            req, request_id=request_id
        )
        pw = self._http_profile_writer
        try:
            while True:
                try:
                    response = await asyncio.wait_for(
                        state.event_queue.get(),
                        timeout=float(self._scheduler_timeout_s),
                    )
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(
                        "Timed out waiting for scheduler stream response"
                    ) from exc

                message_type = str(response.get("type", ""))
                if message_type == "token":
                    now = time.time()
                    state.token_count += 1
                    if state.token_count == 1:
                        state.first_token_ts = now

                    if pw is not None:
                        http_recv_ts = response.get("_http_recv_ts", now)
                        pw.write(
                            {
                                "event": "token_delivered",
                                "request_id": request_id,
                                "token_idx": state.token_count,
                                "ts": now,
                                "zmq_recv_ts": http_recv_ts,
                                "queue_to_yield_ms": round(
                                    (now - http_recv_ts) * 1000, 3
                                ),
                            }
                        )

                    yield {
                        "text": str(response.get("text", "")),
                        "token_id": int(response.get("token_id", 0)),
                        "finish_reason": None,
                    }
                    continue
                if message_type != "final":
                    continue
                if not bool(response.get("ok", False)):
                    raise SchedulerError(
                        "Scheduler stream generation failed: "
                        f"{response.get('error', 'unknown error')}",
                        aborted=bool(response.get("aborted", False)),
                    )
                result = response.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError(
                        "Scheduler stream response missing result payload"
                    )

                self._write_completion_profile(request_id, state, result, stream=True)

                yield {
                    "text": "",
                    "finish_reason": str(result.get("finish_reason", "length")),
                    "prompt_tokens": int(result.get("prompt_tokens", 0)),
                    "completion_tokens": int(result.get("completion_tokens", 0)),
                }
                return
        finally:
            await self._unregister_proxy_request(request_id)
