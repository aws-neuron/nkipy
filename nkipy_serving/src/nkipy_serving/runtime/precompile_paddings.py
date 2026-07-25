from __future__ import annotations

from dataclasses import dataclass

from nkipy_serving.config import RuntimeConfig


@dataclass(frozen=True)
class PrecompilePaddings:
    token_paddings: tuple[int, ...]
    bs_paddings: tuple[int, ...]
    max_padded_num_tokens: int
    max_padded_batch_size: int


def build_precompile_paddings(config: RuntimeConfig) -> PrecompilePaddings:
    token_paddings = tuple(config.token_buckets)
    bs_paddings = tuple(config.request_buckets)
    max_padded_num_tokens = max(token_paddings)
    if (
        config.chunked_prefill_size > 0
        and max_padded_num_tokens > config.chunked_prefill_size
    ):
        if config.enable_mixed_chunk:
            # Mixed chunk adds up to max_requests decode tokens (1 per request).
            mixed_cap = config.chunked_prefill_size + config.max_requests
            max_padded_num_tokens = min(max_padded_num_tokens, mixed_cap)
        else:
            max_padded_num_tokens = config.chunked_prefill_size

    max_padded_batch_size = min(max(bs_paddings), max_padded_num_tokens)

    # Minimum bucket size of 2: nkipy's compiler squeezes leading
    # dimension-1 in kernel outputs, causing shape mismatches for
    # token_bucket=1.  Padding 1→2 is negligible overhead.
    _MIN_BUCKET = 2
    normalized_bs = sorted(
        {max(b, _MIN_BUCKET) for b in bs_paddings if b <= max_padded_batch_size}
    )
    if not normalized_bs or normalized_bs[-1] < max_padded_batch_size:
        normalized_bs.append(max_padded_batch_size)

    normalized_token = sorted(
        {
            t
            for t in token_paddings
            if t >= max_padded_batch_size and t <= max_padded_num_tokens
        }
    )
    if not normalized_token or normalized_token[-1] < max_padded_num_tokens:
        normalized_token.append(max_padded_num_tokens)

    return PrecompilePaddings(
        token_paddings=tuple(normalized_token),
        bs_paddings=tuple(normalized_bs),
        max_padded_num_tokens=max_padded_num_tokens,
        max_padded_batch_size=max_padded_batch_size,
    )
