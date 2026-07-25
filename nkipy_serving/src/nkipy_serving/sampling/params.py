"""Sampling parameter validation for text generation."""

from __future__ import annotations

from typing import Any

from nkipy_serving.utils import get_bool_env_var

_SAMPLING_EPS = 1e-6
TOP_K_ALL = 1 << 30
DEFAULT_SAMPLING_SEED = 42


class SamplingParams:
    def __init__(
        self,
        max_new_tokens: int = 128,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        n: int = 1,
        json_schema: str | None = None,
        regex: str | None = None,
        ebnf: str | None = None,
        structural_tag: str | None = None,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        no_stop_trim: bool = False,
        stream_interval: int | None = None,
        sampling_seed: int | None = None,
    ) -> None:
        self.max_new_tokens = int(max_new_tokens)
        self.stop_strs = stop
        self.stop_token_ids = set(stop_token_ids) if stop_token_ids else None
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.min_p = float(min_p)
        self.frequency_penalty = float(frequency_penalty)
        self.presence_penalty = float(presence_penalty)
        self.repetition_penalty = float(repetition_penalty)
        self.min_new_tokens = int(min_new_tokens)
        self.n = int(n)
        self.regex = regex
        self.json_schema = json_schema
        self.ebnf = ebnf
        self.structural_tag = structural_tag
        self.ignore_eos = bool(ignore_eos)
        self.skip_special_tokens = bool(skip_special_tokens)
        self.spaces_between_special_tokens = bool(spaces_between_special_tokens)
        self.no_stop_trim = bool(no_stop_trim)
        self.stream_interval = stream_interval

        if (
            get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_SAMPLING")
            and sampling_seed is None
        ):
            sampling_seed = DEFAULT_SAMPLING_SEED
        self.sampling_seed = sampling_seed

        if 0 <= self.temperature < _SAMPLING_EPS:
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            self.top_k = TOP_K_ALL

    def verify(self, vocab_size: int) -> None:
        if self.temperature < 0.0:
            raise RuntimeError(
                f"temperature must be non-negative, got {self.temperature}"
            )
        if not 0.0 < self.top_p <= 1.0:
            raise RuntimeError(f"top_p must be in (0, 1], got {self.top_p}")
        if not 0.0 <= self.min_p <= 1.0:
            raise RuntimeError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.top_k < 1:
            raise RuntimeError(f"top_k must be >= 1 or -1, got {self.top_k}")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise RuntimeError(
                f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}"
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise RuntimeError(
                f"presence_penalty must be in [-2, 2], got {self.presence_penalty}"
            )
        if not 0.0 <= self.repetition_penalty <= 2.0:
            raise RuntimeError(
                f"repetition_penalty must be in [0, 2], got {self.repetition_penalty}"
            )
        if self.max_new_tokens < 0:
            raise RuntimeError(
                f"max_new_tokens must be >= 0, got {self.max_new_tokens}"
            )
        if self.min_new_tokens < 0:
            raise RuntimeError(
                f"min_new_tokens must be >= 0, got {self.min_new_tokens}"
            )
        if self.min_new_tokens > self.max_new_tokens:
            raise RuntimeError(
                "min_new_tokens must be <= max_new_tokens, "
                f"got {self.min_new_tokens} > {self.max_new_tokens}"
            )

        grammars = [self.json_schema, self.regex, self.ebnf]
        if sum(x is not None for x in grammars) > 1:
            raise RuntimeError("Only one of json_schema, regex, or ebnf can be set")

    def normalize(self, tokenizer: Any | None) -> None:
        if self.stop_strs is None:
            self.stop_strs = []
            self.stop_str_max_len = 0
            return

        if isinstance(self.stop_strs, str):
            self.stop_strs = [self.stop_strs]

        max_len = 0
        for stop_str in self.stop_strs:
            if tokenizer is not None and hasattr(tokenizer, "encode"):
                try:
                    stop_ids = tokenizer.encode(stop_str)
                    max_len = max(max_len, len(stop_ids))
                    continue
                except (TypeError, ValueError):
                    # Fall back to character length only for invalid tokenizer input.
                    # Runtime tokenizer failures should surface to the caller.
                    max_len = max(max_len, len(stop_str))
                    continue
            max_len = max(max_len, len(stop_str))
        self.stop_str_max_len = max_len

    def convert_to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in self.__dict__.items():
            if key.startswith("_"):
                continue
            if key == "stop_token_ids":
                out[key] = sorted(list(val)) if val is not None else None
            elif key == "stop_strs":
                out["stop"] = val
            else:
                out[key] = val
        return out
