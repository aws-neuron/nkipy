"""DetokenizerManager — separate process for text decoding and response routing.

Sits between the scheduler and the tokenizer manager.  The scheduler sends
compact token-level events; this process decodes them into text and forwards
formatted responses to the tokenizer manager.

IPC topology::

    Scheduler --(PUSH)--> detokenizer_ipc --(PULL)-->  DetokenizerManager
    DetokenizerManager --(PUSH)--> scheduler_output_ipc --(PULL)--> TokenizerManager
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Per-request incremental decode state
# ---------------------------------------------------------------------------


@dataclass
class _DecodeState:
    request_id: str
    stream: bool
    generated_ids: list[int] = field(default_factory=list)
    decode_offset: int = 0
    decoded_text: str = ""
    generated_token_texts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DetokenizerManager
# ---------------------------------------------------------------------------


class DetokenizerManager:
    """Process that converts token IDs to text and routes responses."""

    def __init__(
        self,
        tokenizer_model_id: str,
        tokenizer_revision: str | None = None,
        tokenizer_local_files_only: bool = True,
    ):
        from nkipy_serving.tokenization.hf_tokenizer import HfTokenizer

        self._tokenizer = HfTokenizer(
            model_id=tokenizer_model_id,
            revision=tokenizer_revision,
            local_files_only=tokenizer_local_files_only,
        )
        self._states: dict[str, _DecodeState] = {}

    # -- Text decoding helpers ------------------------------------------------

    def _decode_ids(self, token_ids: np.ndarray) -> str:
        return self._tokenizer.decode(token_ids)

    def _decode_one_token(self, token_id: int) -> str:
        return self._tokenizer.decode(np.asarray([int(token_id)], dtype=np.int32))

    # -- Message handlers -----------------------------------------------------

    def handle_message(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Process one scheduler message. Returns list of responses to forward."""
        msg_type = msg.get("type")

        if msg_type == "batch_tokens":
            return self._handle_batch_tokens(msg)
        if msg_type == "finish":
            return self._handle_finish(msg)
        # Control / error / abort messages — clean up any decode state for the
        # request (abort after batch_tokens would otherwise leak), then forward.
        request_id = msg.get("request_id")
        if request_id is not None:
            self._states.pop(str(request_id), None)
        return [msg]

    def _handle_batch_tokens(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        responses: list[dict[str, Any]] = []
        for out in msg.get("outputs", []):
            request_id = str(out["request_id"])
            token_id = int(out["token_id"])
            stream = bool(out.get("stream", False))

            state = self._states.get(request_id)
            if state is None:
                state = _DecodeState(request_id=request_id, stream=stream)
                self._states[request_id] = state

            state.generated_ids.append(token_id)

            # Incremental decode with UTF-8 buffering.
            suffix_ids = np.asarray(
                state.generated_ids[state.decode_offset :], dtype=np.int32
            )
            new_text = self._decode_ids(suffix_ids)

            if new_text and not new_text.endswith("\ufffd"):
                state.decoded_text += new_text
                state.decode_offset = len(state.generated_ids)
                token_text = new_text
            else:
                token_text = ""

            state.generated_token_texts.append(token_text)

            if state.stream and token_text:
                responses.append(
                    {
                        "type": "token",
                        "request_id": request_id,
                        "token_id": token_id,
                        "text": token_text,
                    }
                )

        return responses

    def _handle_finish(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        request_id = str(msg["request_id"])
        generated_ids_list: list[int] = msg.get("generated_ids", [])
        prompt_ids_list: list[int] = msg.get("prompt_ids", [])
        finish_reason = str(msg.get("finish_reason", "length"))
        stop_strs = msg.get("stop_strs") or []
        no_stop_trim = bool(msg.get("no_stop_trim", False))

        # Bulk decode entire completion.
        if generated_ids_list:
            generated_text = self._decode_ids(
                np.asarray(generated_ids_list, dtype=np.int32)
            )
        else:
            generated_text = ""

        # Stop-string trimming.
        if finish_reason == "stop" and stop_strs and not no_stop_trim:
            for stop_str in stop_strs:
                idx = generated_text.find(stop_str)
                if idx >= 0:
                    generated_text = generated_text[:idx]
                    break

        # Build token_texts from the decode state if available.
        state = self._states.pop(request_id, None)
        token_texts = list(state.generated_token_texts) if state else []

        # Logprob token text decode.
        logprob_data = msg.get("logprob_data")
        if logprob_data is not None:
            return_text = bool(logprob_data.get("return_text_in_logprobs", True))
            if return_text:
                logprob_data = self._decode_logprob_texts(logprob_data)

        meta = dict(msg.get("metadata") or {})
        # The tokenizer manager injects input_ids alongside the original
        # prompt/text for scheduler efficiency.  Strip injected IDs so the
        # public response matches the original request, but keep user-supplied
        # input_ids (where prompt and text are both None).
        if meta.get("input_ids") is not None and (
            meta.get("prompt") is not None or meta.get("text") is not None
        ):
            meta.pop("input_ids", None)
        meta_info = {
            "finish_reason": finish_reason,
            "prompt_tokens": len(prompt_ids_list),
            "completion_tokens": len(generated_ids_list),
            "cached_tokens": int(msg.get("cached_tokens", 0)),
        }
        if logprob_data is not None:
            top_logprobs_num = int(logprob_data.get("top_logprobs_num", 0))
            meta_info["output_token_logprobs"] = logprob_data.get("token_logprobs")
            meta_info["output_top_logprobs"] = (
                logprob_data.get("top_logprobs") if top_logprobs_num > 0 else None
            )
            meta_info["input_token_logprobs"] = logprob_data.get("input_token_logprobs")
            meta_info["input_top_logprobs"] = logprob_data.get("input_top_logprobs")

        result: dict[str, Any] = {
            "text": generated_text,
            "finish_reason": finish_reason,
            "prompt_ids": prompt_ids_list,
            "completion_ids": generated_ids_list,
            "output_ids": prompt_ids_list + generated_ids_list,
            "meta": meta,
            "meta_info": meta_info,
            "token_texts": token_texts,
            "prompt_tokens": len(prompt_ids_list),
            "completion_tokens": len(generated_ids_list),
            "first_scheduled_ts": float(msg.get("first_scheduled_ts", 0.0)),
            "first_token_ts": float(msg.get("first_token_ts", 0.0)),
        }
        if logprob_data is not None:
            result["token_logprobs"] = logprob_data.get("token_logprobs")
            result["top_logprobs"] = logprob_data.get("top_logprobs")

        return [
            {
                "type": "final",
                "request_id": request_id,
                "ok": True,
                "result": result,
            }
        ]

    def _decode_logprob_texts(self, logprob_data: dict[str, Any]) -> dict[str, Any]:
        """Decode token IDs to text in logprob tuples.

        token_logprobs: list of (prob, token_id, text|None)
        top_logprobs:   list of (list[(prob, token_id, text|None)] | None)
        """
        result = dict(logprob_data)

        def _decode_entry(entry: tuple | list) -> tuple:
            """Decode a single (prob, token_id, text|None) tuple."""
            prob, tid = entry[0], int(entry[1])
            return (prob, tid, self._decode_one_token(tid))

        # token_logprobs: flat list of tuples.
        raw_tl = result.get("token_logprobs")
        if raw_tl is not None:
            result["token_logprobs"] = [_decode_entry(e) for e in raw_tl]

        # top_logprobs: list of (list[tuple] | None) per position.
        raw_top = result.get("top_logprobs")
        if raw_top is not None:
            decoded_top = []
            for per_pos in raw_top:
                if per_pos is None:
                    decoded_top.append(None)
                else:
                    decoded_top.append([_decode_entry(e) for e in per_pos])
            result["top_logprobs"] = decoded_top

        return result


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------


def run_detokenizer_process(
    runtime_config_dict: dict[str, Any],
    port_args_dict: dict[str, str],
    ready_event=None,
) -> None:
    """Entry point for the detokenizer subprocess.

    If ``ready_event`` is a :class:`multiprocessing.Event`, it is set after
    ZMQ sockets are bound and the tokenizer is loaded, so the parent process
    can wait for startup to succeed before proceeding.
    """
    import sys
    import traceback

    import zmq as _zmq

    from nkipy_serving.config import RuntimeConfig

    zmq_context = None
    recv_socket = None
    send_socket = None
    try:
        runtime_config = RuntimeConfig(**runtime_config_dict)

        detokenizer = DetokenizerManager(
            tokenizer_model_id=runtime_config.tokenizer_model_id,
            tokenizer_revision=runtime_config.tokenizer_revision,
            tokenizer_local_files_only=runtime_config.tokenizer_local_files_only,
        )

        zmq_context = _zmq.Context()
        recv_socket = zmq_context.socket(_zmq.PULL)
        recv_socket.bind(port_args_dict["detokenizer_ipc_name"])
        # Bind the output endpoint that the tokenizer manager connects to.
        send_socket = zmq_context.socket(_zmq.PUSH)
        send_socket.bind(port_args_dict["scheduler_output_ipc_name"])

        if ready_event is not None:
            ready_event.set()

        while True:
            msg = recv_socket.recv_pyobj()
            responses = detokenizer.handle_message(msg)
            for resp in responses:
                send_socket.send_pyobj(resp)
    except KeyboardInterrupt:
        return
    except Exception:
        traceback.print_exc(file=sys.stderr)
    finally:
        for socket in (recv_socket, send_socket):
            if socket is not None:
                with suppress(_zmq.ZMQError):
                    socket.close(linger=0)
        if zmq_context is not None:
            with suppress(_zmq.ZMQError):
                zmq_context.term()
