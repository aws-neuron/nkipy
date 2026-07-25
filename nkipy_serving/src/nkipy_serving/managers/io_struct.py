from dataclasses import dataclass
from typing import Any


@dataclass
class GenerateReqInput:
    prompt: str | None = None
    text: str | None = None
    input_ids: list[int] | None = None
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stream: bool = False
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    return_logprob: bool = False
    logprob_start_len: int = -1
    top_logprobs_num: int = 0
    return_text_in_logprobs: bool = True
    seed: int | None = None
    no_stop_trim: bool = False
    ignore_eos: bool = False
    metadata: dict[str, Any] | None = None
    score: bool = False
