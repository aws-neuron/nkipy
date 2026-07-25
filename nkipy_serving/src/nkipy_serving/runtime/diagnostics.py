"""Small environment helpers for optional runtime diagnostics."""

from __future__ import annotations

import os

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_ENV_VALUES


def current_rank(default: int = -1) -> int:
    try:
        return int(os.getenv("RANK", str(int(default))))
    except ValueError:
        return int(default)


def env_rank_filter_allows(rank: int, *env_names: str) -> bool:
    raw = ""
    for env_name in env_names:
        raw = os.getenv(env_name, "").strip()
        if raw:
            break
    if not raw:
        return True
    return rank_filter_allows(rank, raw)


def rank_filter_allows(rank: int, raw_filter: str) -> bool:
    rank_i = int(rank)
    for part in str(raw_filter).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if start <= rank_i <= end:
                return True
            continue
        try:
            if int(item) == rank_i:
                return True
        except ValueError:
            continue
    return False
