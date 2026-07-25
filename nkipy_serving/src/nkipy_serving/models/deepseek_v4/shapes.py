"""Shape contracts shared by DSV4 execution and product kernel factories."""

from __future__ import annotations


def token_topk_prep_widths(
    x_shape: tuple[int, ...],
    *,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
) -> tuple[int, int, int, int]:
    if len(x_shape) < 2:
        raise RuntimeError(
            f"DSV4 token top-k prep expects x [batch, seqlen, ...], got {x_shape}"
        )
    if int(ratio) <= 0:
        raise RuntimeError(f"DSV4 token top-k ratio must be > 0, got {ratio}")
    if int(window_size) <= 0:
        raise RuntimeError(
            f"DSV4 token top-k prep requires window_size > 0, got {window_size}"
        )
    if int(k_tile) <= 0:
        raise RuntimeError(f"DSV4 token top-k prep requires k_tile > 0, got {k_tile}")
    bsz, seqlen = int(x_shape[0]), int(x_shape[1])
    expected_offset = int(window_size)
    if int(start_pos) == 0 and int(seqlen) > int(window_size):
        expected_offset = int(seqlen)
    if int(offset) != int(expected_offset):
        raise RuntimeError(
            "DSV4 token top-k prep offset must match the two-source "
            "attention layout; expected "
            f"{int(expected_offset)} for x={x_shape}, "
            f"window_size={int(window_size)}, start_pos={int(start_pos)}, "
            f"got {int(offset)}"
        )
    win_width = (
        int(window_size) if int(start_pos) > 0 else min(int(seqlen), int(window_size))
    )
    if (int(start_pos) == 0 and int(seqlen) // int(ratio) == 0) or (
        int(start_pos) > 0 and int(max_c_len) <= 0
    ):
        comp_width = 1
    else:
        comp_width = int(max_c_len) if int(start_pos) > 0 else int(seqlen) // int(ratio)
    if win_width <= 0 or comp_width <= 0:
        raise RuntimeError(
            "DSV4 token top-k prep requires positive widths, "
            f"win={win_width}, comp={comp_width}"
        )
    n_rows = bsz * (1 if int(start_pos) > 0 else int(seqlen))
    if int(rows) < n_rows:
        raise RuntimeError(
            "DSV4 token top-k prep rows must cover query rows, "
            f"got rows={int(rows)} for x={x_shape}"
        )
    k_raw = int(win_width) + int(comp_width)
    k_padded = ((k_raw + int(k_tile) - 1) // int(k_tile)) * int(k_tile)
    return win_width, comp_width, n_rows, k_padded


def prefill_token_topk_offset(*, seqlen: int, window_size: int) -> int:
    return int(window_size) if int(seqlen) <= int(window_size) else int(seqlen)


def bucketed_prefill_token_topk_compile_shape(
    x_shape: tuple[int, ...],
    *,
    canonical_rows: int,
    q_token_bucket: int,
    kv_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    k_tile: int,
) -> tuple[int, int, int, int] | None:
    """Return bucketed prefill ``(bsz, seqlen, rows, offset)`` if safe.

    The plain compressor-token-topk fallback has two static shape classes for
    common DSV4 settings: tiny prompts whose top-k width pads to one tile, and
    normal prompts whose width pads to the larger bucket class. Pick the largest
    compile seqlen with the same padded top-k width and two-source offset so live
    short prompt lengths do not become NEFF keys.
    """
    if len(x_shape) != 3:
        return None
    bsz, seqlen, hidden = (int(x_shape[0]), int(x_shape[1]), int(x_shape[2]))
    rows = int(canonical_rows)
    if (
        int(start_pos) != 0
        or bsz != 1
        or seqlen <= 0
        or hidden <= 0
        or rows <= bsz * seqlen
        or int(q_token_bucket) != rows
        or int(kv_token_bucket) != rows
    ):
        return None
    try:
        _, _, _, active_k_padded = token_topk_prep_widths(
            x_shape,
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=rows,
            k_tile=int(k_tile),
        )
    except RuntimeError:
        return None

    for compile_seqlen in range(rows, seqlen, -1):
        compile_offset = prefill_token_topk_offset(
            seqlen=int(compile_seqlen),
            window_size=int(window_size),
        )
        if int(compile_offset) != int(offset):
            continue
        try:
            _, _, full_rows, full_k_padded = token_topk_prep_widths(
                (bsz, int(compile_seqlen), hidden),
                window_size=int(window_size),
                ratio=int(ratio),
                offset=int(compile_offset),
                start_pos=int(start_pos),
                max_c_len=int(max_c_len),
                rows=rows,
                k_tile=int(k_tile),
            )
        except RuntimeError:
            continue
        if int(full_rows) == int(compile_seqlen) and int(full_k_padded) == int(
            active_k_padded
        ):
            return bsz, int(compile_seqlen), int(full_rows), int(compile_offset)
    return None


def prefill_token_topk_compile_bucket_lengths(
    *,
    token_bucket: int,
    window_size: int,
    ratio: int,
    k_tile: int,
) -> tuple[int, ...]:
    bucket_i = int(token_bucket)
    if bucket_i <= 0:
        return ()
    by_width: dict[int, int] = {}
    for seqlen in range(1, bucket_i + 1):
        offset = prefill_token_topk_offset(
            seqlen=int(seqlen),
            window_size=int(window_size),
        )
        try:
            _, _, _, k_padded = token_topk_prep_widths(
                (1, int(seqlen), 1),
                window_size=int(window_size),
                ratio=int(ratio),
                offset=int(offset),
                start_pos=0,
                max_c_len=0,
                rows=bucket_i,
                k_tile=int(k_tile),
            )
        except RuntimeError:
            continue
        by_width[int(k_padded)] = int(seqlen)
    return tuple(sorted(set(by_width.values())))


def bucketed_prefill_token_topk_shape(
    x_shape: tuple[int, ...],
    *,
    canonical_rows: int,
    q_token_bucket: int,
    kv_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    k_tile: int,
) -> tuple[int, int, int] | None:
    """Return a safe bucketed prefill ``(bsz, seqlen, rows)`` shape."""
    if len(x_shape) != 3:
        return None
    bsz, seqlen, hidden = (int(x_shape[0]), int(x_shape[1]), int(x_shape[2]))
    rows = int(canonical_rows)
    if (
        int(start_pos) != 0
        or bsz != 1
        or seqlen <= 0
        or hidden <= 0
        or rows <= bsz * seqlen
        or int(q_token_bucket) != rows
        or int(kv_token_bucket) != rows
    ):
        return None
    if rows % bsz != 0:
        return None
    full_seqlen = rows // bsz
    if full_seqlen <= seqlen:
        return None
    try:
        _, _, _, active_k_padded = token_topk_prep_widths(
            x_shape,
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=rows,
            k_tile=int(k_tile),
        )
        _, _, full_n_rows, full_k_padded = token_topk_prep_widths(
            (bsz, full_seqlen, hidden),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=rows,
            k_tile=int(k_tile),
        )
    except RuntimeError:
        return None
    if int(full_n_rows) != rows or int(active_k_padded) != int(full_k_padded):
        return None
    return bsz, full_seqlen, rows


__all__ = [
    "bucketed_prefill_token_topk_shape",
    "bucketed_prefill_token_topk_compile_shape",
    "prefill_token_topk_compile_bucket_lengths",
    "prefill_token_topk_offset",
    "token_topk_prep_widths",
]
