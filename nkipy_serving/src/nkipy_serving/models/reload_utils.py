from __future__ import annotations

from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from spike import get_spike_singleton


def resolve_model_snapshot_path(
    model_source: str,
    *,
    revision: str | None = None,
    local_files_only: bool = True,
) -> Path:
    """Resolve a local snapshot directory or a cached HuggingFace snapshot."""
    path = Path(model_source).expanduser()
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"Local model source must be a directory, got {path}")
        return path.resolve()
    return Path(
        snapshot_download(
            repo_id=model_source,
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def overwrite_device_tensor(dst, src: np.ndarray) -> None:
    """Overwrite an existing device tensor allocation in place."""
    arr = np.asarray(src, dtype=dst.dtype)
    if tuple(arr.shape) != tuple(dst.shape):
        raise RuntimeError(
            f"shape mismatch rewriting device tensor {dst.name!r}: "
            f"{tuple(arr.shape)} != {tuple(dst.shape)}"
        )
    get_spike_singleton().tensor_write_from_pybuffer(
        dst.tensor_ref,
        np.ascontiguousarray(arr),
    )


def overwrite_device_tensor_if_changed(
    dst,
    host: np.ndarray,
    value,
    *,
    prefix_len: int | None = None,
    error_context: str = "device tensor metadata sync",
) -> bool:
    """Overwrite ``dst`` only when ``value`` differs from the cached host array."""

    src = np.asarray(value, dtype=host.dtype)
    if prefix_len is None:
        if tuple(src.shape) != tuple(host.shape):
            raise RuntimeError(
                f"{error_context} shape mismatch: "
                f"got {tuple(src.shape)}, expected {tuple(host.shape)}"
            )
        if np.array_equal(host, src):
            return False
        host[...] = src
    else:
        n = int(prefix_len)
        if n < 0 or n > int(host.shape[0]):
            raise RuntimeError(
                f"{error_context} prefix length is invalid: "
                f"prefix_len={n}, host_shape={tuple(host.shape)}"
            )
        src = src.reshape(-1)
        if int(src.shape[0]) != n:
            raise RuntimeError(
                f"{error_context} prefix sync shape mismatch: "
                f"got {tuple(src.shape)}, expected ({n},)"
            )
        view = host[:n]
        if np.array_equal(view, src):
            return False
        view[...] = src
    overwrite_device_tensor(dst, host)
    return True


def upsert_device_tensor(
    device_tensor_cls, src: np.ndarray, *, name: str, existing=None
):
    """Create a new device tensor or rewrite an existing one in place."""
    if existing is None:
        return device_tensor_cls.from_numpy(
            np.ascontiguousarray(np.asarray(src)), name=name
        )
    overwrite_device_tensor(existing, src)
    return existing
