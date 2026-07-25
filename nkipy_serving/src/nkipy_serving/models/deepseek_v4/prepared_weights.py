"""Prepared-weight staging helpers for DeepSeek-V4."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nkipy_serving.config import RuntimeConfig

logger = logging.getLogger(__name__)


def runtime_config_looks_like_deepseek_v4(runtime_config: RuntimeConfig) -> bool:
    if str(getattr(runtime_config, "attention_backend", "")) == "Dsv4SparseAttention":
        return True
    model = str(getattr(runtime_config, "model_id", "")).lower()
    hf_model = str(getattr(runtime_config, "hf_model_id", "") or "").lower()
    return "deepseek-v4" in model or "deepseek-v4" in hf_model


def prepared_weight_source_dirs(
    runtime_config: RuntimeConfig,
    source_root: Path,
) -> list[Path]:
    """Return unique prepared-weight source dirs used by this worker group."""
    if (source_root / "metadata.json").exists():
        return [source_root]

    from nkipy_serving.models.deepseek_v4.device_weights import (
        prepared_weight_rank_dir_for,
    )

    tp = int(runtime_config.tp_degree)
    ep = int(runtime_config.ep_degree)
    replica = int(runtime_config.replica_degree)
    total = int(runtime_config.total_workers)
    dirs: list[Path] = []
    seen: set[str] = set()
    for rank in range(total):
        tp_rank = rank % tp
        lane = rank // tp
        candidates = [
            prepared_weight_rank_dir_for(
                source_root,
                tp_degree=tp,
                ep_degree=ep,
                replica_degree=replica,
                lane=lane,
                tp_rank=tp_rank,
            ),
        ]
        if replica > 1:
            candidates.append(
                prepared_weight_rank_dir_for(
                    source_root,
                    tp_degree=tp,
                    ep_degree=ep,
                    replica_degree=replica,
                    lane=lane % ep,
                    tp_rank=tp_rank,
                )
            )
        for candidate in candidates:
            if not (candidate / "metadata.json").exists():
                continue
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                dirs.append(candidate)
            break
    return dirs


def prestage_prepared_weights(runtime_config: RuntimeConfig) -> None:
    """Stage prepared DSV4 weight dirs locally before worker spawn.

    The per-rank loader can stage lazily, but doing it after 128 workers start
    creates a thundering herd against shared storage. This hook is intentionally opt-in:
    cold full-model staging can copy hundreds of GB and is better done offline.
    """
    if not runtime_config.dsv4_prepared_weight_prestage:
        return
    if not runtime_config_looks_like_deepseek_v4(runtime_config):
        return
    source_raw = (runtime_config.dsv4_prepared_weight_dir or "").strip()
    local_raw = (runtime_config.dsv4_prepared_weight_local_dir or "").strip()
    if not source_raw or not local_raw:
        return
    source_root = Path(source_raw).expanduser()
    local_root = Path(local_raw).expanduser()
    if not source_root.exists():
        return

    from nkipy_serving.models.deepseek_v4.device_weights import (
        stage_prepared_weight_rank_dir_local,
    )

    rank_dirs = prepared_weight_source_dirs(runtime_config, source_root)
    if not rank_dirs:
        return
    workers = max(
        1,
        min(
            len(rank_dirs),
            int(runtime_config.dsv4_prepared_weight_prestage_workers),
        ),
    )
    logger.info(
        "DSV4 weight prestage start dirs=%d workers=%d src=%s local=%s",
        len(rank_dirs),
        workers,
        source_root,
        local_root,
    )
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                stage_prepared_weight_rank_dir_local,
                source_root,
                rank_dir,
                local_root,
                log_fn=lambda message: logger.info("DSV4 weight prestage %s", message),
            )
            for rank_dir in rank_dirs
        ]
        for future in as_completed(futures):
            future.result()
    logger.info(
        "DSV4 weight prestage done dirs=%d elapsed=%.1fs",
        len(rank_dirs),
        time.monotonic() - t0,
    )
