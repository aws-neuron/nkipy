from nkipy_serving.batching.contracts import ForwardBatch, ForwardMode


def select_bucket(required: int, buckets: tuple[int, ...], axis_name: str) -> int:
    if required <= 0:
        raise RuntimeError(f"{axis_name} required size must be > 0, got {required}")
    for bucket in buckets:
        if required <= bucket:
            return bucket
    raise RuntimeError(
        f"{axis_name} bucket miss. required={required}, available={buckets}"
    )


def validate_forward_batch_shape(
    batch: ForwardBatch,
    token_buckets: tuple[int, ...],
    request_buckets: tuple[int, ...],
) -> None:
    """Validate that a ForwardBatch fits configured padding buckets.

    EXTEND mode: token_bucket must be in token_buckets.
    DECODE mode: token_bucket may be in either:
      - token_buckets: current scheduler behavior pads decode with token buckets
      - request_buckets: specialized decode paths may use request buckets directly
    """
    if batch.forward_mode == ForwardMode.EXTEND:
        if batch.token_bucket not in token_buckets:
            raise RuntimeError(
                "ForwardBatch extend token_bucket is not configured: "
                f"{batch.token_bucket}, allowed={token_buckets}"
            )
        return

    if batch.forward_mode == ForwardMode.DECODE:
        allowed = tuple(
            sorted(
                {
                    int(bucket)
                    for bucket in (tuple(token_buckets) + tuple(request_buckets))
                }
            )
        )
        if batch.token_bucket not in allowed:
            raise RuntimeError(
                "ForwardBatch decode token_bucket is not configured: "
                f"{batch.token_bucket}, allowed={allowed}"
            )
        return

    raise RuntimeError(f"Unknown ForwardBatch mode: {batch.forward_mode}")
