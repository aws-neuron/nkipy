# Qwen3 Dense

Qwen3 dense support in `nkipy-serving` targets Neuron device execution with
tensor parallelism and the NKI blocksparse attention backend.

## Requirements

- Model id pattern: `Qwen/Qwen3-*` without the MoE `-A<size>B` suffix
- TP NKI attention expects one local KV head per rank on the current dense path
  - `Qwen/Qwen3-0.6B` should run with `tp_degree=8`
- Attention backend: `NKIBlockSparseFlashAttention` (required)
- Execution mode: all-layers-one-graph with embedding outside (no mode selection)

Typical NKI launch:

```bash
export PATH=$(pwd)/../.venv/bin:$PATH
python -m nkipy_serving.launch_server \
  --config tests/runtime.tp8.qwen3.serving.test.json \
  --port 30103
```

## Execution Mode

Qwen3 Dense always uses all-layers-one-graph with embedding outside:

- One graph for all transformer layers per step
- Embedding and LM-head run as separate kernels
- NKI attention step inputs are shared across the whole step
- Best steady-state device path

## Implemented Runtime Optimizations

The current Qwen3 dense device path now includes the main scratch-buffer reuse
pattern that GPT-OSS already used.

### Shared prepared attention inputs

- `PreparedNkiStepInputs` are allocated once per token bucket
- Slot mapping and tile plans are rewritten in place for each step
- Worker warmup includes synthetic extend/decode execution, not just compile

### Persistent per-bucket step scratch

Qwen3 dense reuses per-bucket device buffers for:

- `input_ids`
- RoPE `cos` / `sin`
- `last_token_indices`
- local top-1 output buffers
- local top-k output buffers

This removes repeated `DeviceTensor.from_numpy(...)` allocation on the hot path
for LM-head sampling and full-graph decode inputs.

### Decode behavior

- Uses the scheduler-visible token bucket
- Attention preparation still pads decode Q/K/V up to the NKI minimum attention
  bucket when needed
- The runtime reuses persistent input/output step buffers instead
  of recreating them every request

## Decode Bucket Behavior

- Pure decode now uses scheduler request buckets (`bs_paddings`) instead of
  always normalizing back up to token buckets
- Synthetic warmup now first-touches those same decode bucket shapes, so startup
  and steady-state serving use the same decode surfaces
- NKI attention still pads the internal attention side to at least 128 tokens
  when required, but model-side decode compute can stay on the smaller request
  bucket

## Benchmark Guidance

For TP8 NKI serving on `Qwen/Qwen3-0.6B`, the current best-balanced config is:

```text
tp_degree         = 8
request_buckets   = [32]
token_buckets     = [128,512,1024,2048,4096]
attention_backend = NKIBlockSparseFlashAttention
```

Minimal benchmark-oriented config:

```json
{
  "model_id": "Qwen/Qwen3-0.6B",
  "max_context_len": 4096,
  "tp_degree": 8,
  "attention_backend": "NKIBlockSparseFlashAttention",
  "paged_attn_impl": "nki_blocksparse_flash_attention",
  "request_buckets": [32],
  "token_buckets": [128, 512, 1024, 2048, 4096]
}
```

Repo config file:

```text
tests/runtime.tp8.qwen3.nki.bench.json
```

Equivalent environment overrides:

```bash
export NKIPY_SERVING_REQUEST_BUCKETS=32
export NKIPY_SERVING_TOKEN_BUCKETS=128,512,1024,2048,4096
```

`4096` is included intentionally because the default chunked prefill length is
`4096`, so the recommended token bucket set should cover the full prefill chunk.

### Performance Reference

Current TP8 NKI numbers on `request_buckets=[32]` and
`token_buckets=[128,512,1024,2048,4096]` (all-layers-one-graph mode):

### Request-Bucket Sweep

Current 100-prompt saturated ShareGPT sweep on `2026-03-19`, all with
`token_buckets=[128,512,1024,2048,4096]`:

- `request_buckets=[32]`
  - request throughput: `2.12 req/s`
  - output throughput: `434.51 tok/s`
  - mean TTFT: `838.77 ms`
  - median TTFT: `269.95 ms`
  - mean E2E: `7679.29 ms`
- `request_buckets=[16,32]`
  - request throughput: `2.19 req/s`
  - output throughput: `444.79 tok/s`
  - mean TTFT: `855.89 ms`
  - median TTFT: `269.38 ms`
  - mean E2E: `7734.10 ms`
- `request_buckets=[64]`
  - request throughput: `2.03 req/s`
  - output throughput: `416.71 tok/s`
  - median TTFT: `293.83 ms`
- `request_buckets=[16]`
  - request throughput: `2.01 req/s`
  - output throughput: `414.09 tok/s`
  - median TTFT: `4070.26 ms`
- `request_buckets=[8]`
  - request throughput: `1.62 req/s`
  - output throughput: `335.55 tok/s`
  - median TTFT: `9782.09 ms`

Current recommendation:

- `request_buckets=[32]` is the best balanced setting across TTFT, E2E latency,
  and output throughput
- `[16,32]` is the only mixed request-bucket set close enough to consider if
  you want a slightly more elastic serving surface
- including `8` or dropping to a single `16` bucket hurts saturated serving
- larger single buckets such as `[64]` were worse than `[32]` on the tested
  workload
- That sweep was run on a prior configuration, but the same `[32]` setting
  carried cleanly to the current all-layers-one-graph mode

### Latest Rerun

Latest validated rerun on `2026-04-09`, using cached build and the
recommended config above (post model-subpackage refactor, detokenizer
process, pre-tokenization, overlap scheduling):

```text
200-prompt ShareGPT, request_rate=inf
  request throughput      5.30 req/s
  output throughput    1074.12 tok/s
  mean TTFT            1033.08 ms
  median TTFT           924.64 ms
  mean ITL               18.96 ms
  median ITL             16.00 ms
  p99 ITL                64.29 ms
  mean E2E             4854.87 ms
  median E2E           3707.82 ms
```

## Notes

- For dense local sampling, the output remains rank-local and uses
  `vocab_offset` for TP reconstruction

## Validation

The current optimization port was validated with:

- targeted unit coverage in `tests/test_model_runner.py`
- scheduler decode-bucket coverage in `tests/test_scheduler.py`
- TP=8 NKI serving validation in `tests/test_tp8_serving.py`
- TP=8 live reload validation in `tests/test_tp8_reload_weights.py`
