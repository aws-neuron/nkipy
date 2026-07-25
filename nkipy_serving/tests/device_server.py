"""Shared helpers for opt-in live device server tests."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError


def load_model_id(config: Path, *, default: str) -> str:
    if not config.exists():
        return default
    with config.open("r", encoding="utf-8") as f:
        return str(json.load(f)["model_id"])


def require_config(config: Path) -> None:
    if not config.exists():
        pytest.skip(f"Missing config file: {config}")


def require_local_snapshot(repo_id: str) -> str:
    try:
        return str(snapshot_download(repo_id=repo_id, local_files_only=True))
    except (HFValidationError, LocalEntryNotFoundError) as exc:
        pytest.skip(f"Local HF snapshot unavailable for {repo_id}: {exc!r}")


@contextmanager
def launch_server(
    *,
    config: Path,
    port: int,
    env: Mapping[str, str] | None = None,
    terminate_timeout_s: int = 20,
) -> Iterator[subprocess.Popen[Any]]:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "nkipy_serving.launch_server",
            "--config",
            str(config),
            "--port",
            str(port),
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=dict(env) if env is not None else None,
    )
    try:
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=terminate_timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=terminate_timeout_s)


def get_json(base_url: str, path: str, *, timeout_s: int) -> dict[str, Any]:
    req = urllib.request.Request(f"{base_url}{path}")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read())


def post_json(
    base_url: str,
    path: str,
    body: dict[str, Any],
    *,
    timeout_s: int,
) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read()
        try:
            payload = json.loads(body_bytes)
        except json.JSONDecodeError:
            payload = {"raw": body_bytes.decode("utf-8", errors="replace")}
        return int(exc.code), payload


def post_stream(
    base_url: str,
    path: str,
    body: dict[str, Any],
    *,
    timeout_s: int,
) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return int(resp.status), resp.read().decode("utf-8", errors="replace")


def parse_sse_events(sse_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in sse_text.strip().splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


def ready_error(payload: dict[str, Any]) -> str:
    return str(
        payload.get("error") or payload.get("message") or payload.get("raw") or payload
    )


def probe_ready(
    base_url: str,
    *,
    timeout_s: int = 10,
) -> tuple[int | None, dict[str, Any] | None]:
    req = urllib.request.Request(f"{base_url}/ready")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
            payload = json.loads(body) if body else {}
            return int(resp.status), payload
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read()
        try:
            payload = json.loads(body_bytes)
        except json.JSONDecodeError:
            payload = {"raw": body_bytes.decode("utf-8", errors="replace")}
        return int(exc.code), payload
    except (urllib.error.URLError, OSError):
        return None, None


def wait_ready(
    base_url: str,
    *,
    timeout_s: int,
    terminal_error_snippets: tuple[str, ...] = (),
    any_500_is_terminal: bool = False,
) -> tuple[bool, str | None]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, payload = probe_ready(base_url)
        if status == 200:
            return True, None
        if status == 500 and payload is not None:
            error_text = ready_error(payload).lower()
            if any_500_is_terminal or any(
                snippet in error_text for snippet in terminal_error_snippets
            ):
                return False, error_text
        time.sleep(2)
    return False, None
