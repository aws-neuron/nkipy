"""Common utilities used by non-JAX runtime modules."""

from __future__ import annotations

import functools
import logging
import os
import random
import re
from collections import OrderedDict
from typing import Any, Callable, Mapping

import numpy as np

logger = logging.getLogger(__name__)

_warned_bool_env_values: set[str] = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    truthy_values = {"1", "true", "yes", "y", "on"}
    falsy_values = {"0", "false", "no", "n", "off"}

    if value in truthy_values:
        return True
    if value in falsy_values:
        return False

    if value not in _warned_bool_env_values:
        logger.warning(
            "Environment variable %s has non-bool value=%r; treating as false",
            name,
            value,
        )
        _warned_bool_env_values.add(value)
    return False


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def nullable_str(val: str | None) -> str | None:
    if val is None:
        return None
    stripped = val.strip()
    if not stripped or stripped == "None":
        return None
    return stripped


_REMOTE_URL_RE = re.compile(r"^(https?|s3|gs|hf)://", re.IGNORECASE)


def is_remote_url(path: str) -> bool:
    return bool(_REMOTE_URL_RE.match(path.strip()))


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(v) for v in value)
    return value


def lru_cache_frozenset(
    maxsize: int = 128,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """LRU cache wrapper that accepts dict/list/set arguments."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache: OrderedDict[Any, Any] = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (_freeze_value(args), _freeze_value(kwargs))
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            out = func(*args, **kwargs)
            cache[key] = out
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return out

        setattr(wrapper, "cache_clear", cache.clear)
        return wrapper

    return decorator
