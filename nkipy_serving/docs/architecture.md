# Architecture

## System Overview

nkipy-serving is a Python-first serving runtime for text generation on AWS Neuron hardware (Trn2/Inf2). Key design principles:

- **No PyTorch or JAX** in the runtime/model path. Weights are numpy arrays; compute goes through NKI kernels compiled to NEFFs via neuronx-cc.
- **Fail-fast** — no silent fallbacks. Errors surface immediately.
- **No backward compatibility** — refactor freely without deprecation shims.
- **bf16-first** — fp32 only for numerically sensitive reductions.

## Process Model

```
┌──────────────────────────────────────────────────────────┐
│  Main Process                                            │
│                                                          │
│  ┌──────────────────────────┐                            │
│  │  HTTP Server (FastAPI)   │                            │
│  │  14 routes + CORS        │                            │
│  └────────────┬─────────────┘                            │
│               │ ZMQ PUSH/PULL (ipc://)                   │
│  ┌────────────▼─────────────┐                            │
│  │  TokenizerManager        │  Pre-tokenize prompts,     │
│  │  (ZMQ bridge)            │  async proxy, streaming    │
│  └──────────────────────────┘                            │
└───────────────┬──────────────────────────────────────────┘
                │ ZMQ IPC (scheduler_input_ipc)
┌───────────────▼──────────────────────────────────────────┐
│  Scheduler Subprocess (pure CPU coordinator)             │
│  rank=-1, never claims a Neuron core                     │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  ScheduleBatch → build_forward_batch → forward     │  │
│  │  → process_results → token IDs to detokenizer      │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌───────────────────────┐ ┌────────────────────────┐    │
│  │  SchedulerKVPoolStub  │ │  Prefix Cache (radix/  │    │
│  │  (no per-layer arrays)│ │  chunk), LRU eviction  │    │
│  └───────────────────────┘ └────────────────────────┘    │
└───────────────┬──────────────────────────────────────────┘
                │ ZMQ IPC (detokenizer_ipc)
┌───────────────▼──────────────────────────────────────────┐
│  Detokenizer Subprocess                                  │
│  Incremental text decode, stop-string trimming,          │
│  streaming token events, final response formatting       │
└───────────────┬──────────────────────────────────────────┘
                │ ZMQ IPC (scheduler_output_ipc)
┌───────────────▼──────────────────────────────────────────┐
│  TokenizerManager (receives responses)                   │
└───────────────┬──────────────────────────────────────────┘
                │ Shared memory (batch tensors + command/status slots)
┌───────────────▼──────────────────────────────────────────┐
│  Workers (global_rank 0..N-1)                            │
│  Each owns Neuron cores, forward passes in lockstep      │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Worker 0 │  │ Worker 1 │  │ Worker N │  NRT execute,  │
│  │ (rank 0) │  │ (rank 1) │  │ (rank N) │  all-reduce    │
│  └──────────┘  └──────────┘  └──────────┘                │
│                                                          │
│  Output ranks return compact sampled outputs             │
└──────────────────────────────────────────────────────────┘
```

**TP=1**: Scheduler runs the executor directly (no worker subprocess).
**TP>1**: Scheduler spawns N worker processes via `WorkerCoordinator`. Workers share `ForwardBatch` data through pre-allocated shared memory buffers. The scheduler publishes each step into a shared command block with a monotonically increasing generation counter plus compact fixed-schema `msgspec`/msgpack batch metadata, workers poll that state with short spin/sleep backoff, and each worker publishes completion into its own status slot. Distributed nkipy sampled-output models (Qwen3 dense, `qwen3-moe`, `gpt-oss`) keep the LM head TP-sharded. Pure greedy batches stay on the compact TP-local top-1/top-k path, while any non-greedy batch all-gathers full LM-head logits across TP and runs one raw NKI sampler kernel on device to emit final `next_token_ids` directly. The result queue is reserved for worker lifecycle events.
**TP+EP(+replica)**: With expert parallelism enabled (`ep_degree > 1`), `total_workers = tp_degree * ep_degree * replica_degree` processes are spawned. Each worker derives `tp_rank = global_rank % tp_degree`; model-specific layout code maps the remaining row index into EP, replica, and attention-DP lanes. Qwen3/GPT-OSS keep `replica_degree=1`, so their topology remains `tp_degree * ep_degree`. DeepSeek-V4 uses replica/attention-lane metadata to decide which output slots publish sampled outputs to the scheduler. Requires `NEURON_LOGICAL_NC_CONFIG=1` for high-rank EP layouts.

IPC between the HTTP layer and the scheduler uses ZMQ PUSH/PULL (`ipc://`). The tokenizer manager pre-tokenizes text prompts before sending to the scheduler, so the scheduler thread never blocks on tokenization. The scheduler sends compact token-level events (batched token IDs and finish messages) to the detokenizer process, which converts them to text and forwards formatted responses back to the tokenizer manager. Control messages (shutdown, metrics, abort) are forwarded through the detokenizer unchanged. Scheduler/worker IPC uses shared memory for step data/control. Both nkipy and numpy backends write compact sampled-output dicts to SHM output slots. A single `mp.Queue` is used only for startup lifecycle (worker ready/crash reporting) — shutdown uses SHM broadcast, and no queues exist on the per-step forward path.

The scheduler still performs lightweight incremental text decoding for requests with text stop strings (to detect when to stop generation). This is the same pattern as upstream SGLang, where the scheduler decodes for stop checking and the detokenizer independently decodes for user output.

With overlap scheduling enabled (default), the scheduler receives new requests during the device-wait window between dispatch and collect, hiding request handling latency behind device execution. This improved Qwen3 throughput by ~5% and GPT-OSS by ~8% versus the non-overlap path.

In the latest GPT-OSS TP8 serving profile, pure decode overhead (scheduler + IPC, excluding device time) measured ~`0.9 ms` at p50 per decode step.

## Request Lifecycle

```
HTTP request
  → FastAPI route (http_server.py)
  → Pydantic validation (protocol.py)
  → GenerateReqInput (io_struct.py)
  → [ZMQ] TokenizerManager pre-tokenizes prompt → Scheduler
  → _RequestState created with SamplingParams
  → Admission control (token budget check)
  → Prefix cache lookup (optional, skip cached prefix)
  → ScheduleBatch._get_next_batches()
    → EXTEND batch (new/extending requests, chunked if needed)
    → DECODE batch (requests generating tokens)
    → MIXED batch (if enable_mixed_chunk: extend+decode combined)
  → build_forward_batch() → ForwardBatch
  → Bucket selection (shape_guard.py)
  → forward() → executor/workers
  → process_results() → sampling → token IDs
  → [ZMQ] batch_tokens / finish → DetokenizerManager
  → Incremental detokenization (offset-based, UTF-8 buffered)
  → [ZMQ] token / final → TokenizerManager
  → Stream SSE events (OpenAI delta protocol) or collect final response
  → HTTP response
```

## Module Map

All paths relative to `src/nkipy_serving/`.

| Module | Role |
|--------|------|
| `nkipy_serving/entrypoints/http_server.py` | FastAPI routes, CORS, lifespan, readiness |
| `nkipy_serving/entrypoints/engine.py` | ZMQ IPC bootstrap, `PortArgs`, process group |
| `nkipy_serving/entrypoints/openai/protocol.py` | Pydantic models: requests, responses, `LogProbs`, `ChoiceLogprobs` |
| `nkipy_serving/entrypoints/openai/serving_completions.py` | `/v1/completions` handler, logprobs formatting |
| `nkipy_serving/entrypoints/openai/serving_chat.py` | `/v1/chat/completions` handler, OpenAI delta streaming |
| `nkipy_serving/managers/scheduler.py` | `ScheduleBatch`, batching, mixed chunk, LPM scheduling, admission control, logprobs, metrics |
| `nkipy_serving/managers/tokenizer_manager.py` | Pre-tokenize prompts, ZMQ async proxy, streaming |
| `nkipy_serving/managers/detokenizer_manager.py` | Separate process: incremental text decode, stop-string trimming, response formatting |
| `nkipy_serving/managers/io_struct.py` | `GenerateReqInput` with full sampling params |
| `nkipy_serving/sampling/params.py` | `SamplingParams` validation and normalization |
| `nkipy_serving/config.py` | `RuntimeConfig`, validation, config loading |
| `nkipy_serving/runtime/worker_coordinator.py` | Shared memory, worker dispatch, generation-polled control plane |
| `nkipy_serving/runtime/parallel_groups.py` | TP/EP replica group builders for collective ops |
| `nkipy_serving/runtime/shape_guard.py` | Bucket selection, forward batch shape validation |
| `nkipy_serving/runtime/variant_registry.py` | Variant registry for graph compilation |
| `nkipy_serving/runtime/precompile_catalog.py` | Precompilation catalog for warmup |
| `nkipy_serving/runtime/warmup.py` | Shared synthetic startup warmup planner/executor |
| `nkipy_serving/models/registry.py` | `ModelSpec` with `build_kv_metadata` and per-model `config_defaults` |
| `nkipy_serving/models/_device_utils.py` | Shared device utilities: nkipy runtime, KV cache allocation/flush |
| `nkipy_serving/models/qwen3_dense/` | Qwen3 dense model: config, weights, graph fns, executor, CPU ref forward |
| `nkipy_serving/models/gpt_oss/` | GPT-OSS MoE model: config, weights, graph fns, executor |
| `nkipy_serving/models/qwen3_moe/` | Qwen3 MoE model: config, weights, graph fns, executor |
| `nkipy_serving/ops/nn.py` | Shared NN primitives: RMS norm, RoPE, SiLU, MLP, TP utilities |
| `nkipy_serving/attention/_kernel_cache.py` | `AttentionKernelCache`: compiled NKI attention/KV-update kernels |
| `nkipy_serving/attention/nki_blocksparse_flash_attention.py` | NKI attention backend: unified prefill+decode kernel |
| `nkipy_serving/attention/nki_step_inputs.py` | Shared prepared slot-mapping and tile-plan buffers reused across models |
| `nkipy_serving/attention/vanilla.py` | Vanilla paged attention (CPU, for numpy backend) |
| `nkipy_serving/attention/blocksparse_flash_attention/` | NKI kernel implementations: flash attention core, scheduler, paged cache |
| `nkipy_serving/ops/lm_head_sampling.py` | Reference sampling math: softmax, threshold search, CDF sample (pure numpy) |
| `nkipy_serving/sampling/logits_processor.py` | `LogitsProcessor`: device LM-head → sampling → logprobs pipeline, owns kernel compilation/warmup |
| `nkipy_serving/sampling/lm_head_sampling.py` | Device LM-head sampling entry points: greedy top-k, NKI CDF sampler, sampler+logprobs |
| `nkipy_serving/sampling/nki_kernels.py` | Raw NKI kernels for device-side token sampling (filtered/unfiltered) |
| `nkipy_serving/sampling/logits_processor_np.py` | `NumpyLogitsProcessor`: CPU reference logits processor (accuracy baseline) |
| `nkipy_serving/ops/moe/blockwise_nki.py` | NKI MoE kernels (prefill + decode paths) |
| `nkipy_serving/ops/moe/blockwise_index.py` | CPU-side MoE block scheduling |
| `nkipy_serving/mem_cache/memory_pool.py` | `MHATokenToKVPool`, `ReqToTokenPool`, `SchedulerKVPoolStub` |
| `nkipy_serving/mem_cache/radix_cache.py` | Radix prefix cache with LRU eviction |
| `nkipy_serving/mem_cache/allocator.py` | Paged token-to-KV pool allocator |
| `nkipy_serving/batching/contracts.py` | `ForwardBatch`, `ForwardMode` (EXTEND/DECODE) |
| `nkipy_serving/tokenization/hf_tokenizer.py` | HuggingFace tokenizer wrapper |
| `nkipy_serving/profiling.py` | `PROFILING_ENABLED` gate, `ProfileWriter` (JSONL), `StepTimer` |
| `nkipy_serving/model_executor/model_runner.py` | Model runner for worker execution |

## Data Flow

```
GenerateReqInput (HTTP layer, Pydantic)
  ↓
_RequestState (scheduler, per-request tracking)
  - prompt_ids, generated_ids, seq_len, extend_offset
  - req_pool_idx, out_cache_loc (KV pool tracking)
  - SamplingParams, stop tokens, logprobs config
  ↓
ScheduleBatch (batch abstraction)
  - list of _RequestState, forward_mode (EXTEND/DECODE/MIXED)
  - build_forward_batch() produces ForwardBatch
  ↓
ForwardBatch (runtime contract)
  - input_ids, positions, slot_mapping, seq_lens
  - block_tables, query_start_loc
  - token_bucket, forward_mode
  ↓
forward() → executor or workers
  ↓
sampled output dict (next_token_ids + optional logprobs)
  ↓
process_results() → new token IDs
  ↓
Incremental detokenization → SSE stream events
```

## Scheduling

The scheduler supports three modes:

- **EXTEND**: Processes new or continuing prompts. Token count may be chunked by `chunked_prefill_size` (default 4096).
- **DECODE**: Auto-regressive token generation. Each request contributes 1 token.
- **MIXED** (when `enable_mixed_chunk=true`): Single EXTEND batch where decode states contribute 1 token each, overlapping extend and decode in one forward pass.

Scheduling policy:
- **FIFO** by default
- **LPM** (longest prefix match) when prefix cache is enabled — prioritizes requests with longer cache hits
- Admission control with token budget pre-checks

Main loop: `_get_next_batches()` → `build_forward_batch()` → `forward()` → `process_results()`.

## Memory Management

### KV Cache

Block-paged KV cache (`MHATokenToKVPool`):
- Storage layout per layer: `[2, num_blocks, num_kv_heads, block_size, head_dim]` (axis 0 = K/V)
- `kv_pool_size` total token slots, `kv_cache_block_size` tokens per block (default 32)
- `PagedTokenToKVPoolAllocator` manages block allocation/deallocation

### Scheduler KV Pool

`SchedulerKVPoolStub` — lightweight stub used by the scheduler process. Does not allocate per-layer arrays, saving ~1 GB for 8B models. The scheduler only needs slot index tracking, not actual KV data.

### Prefix Cache

- **Radix cache** (`radix_cache.py`): Trie-based prefix matching with LRU eviction. On hit, cached KV slot indices are reused and `extend_offset` skips the prefix.
- On completion, prompt slots are donated to the cache. Full hits still re-run the last prompt token for sampling logits.

### Request Pool

`ReqToTokenPool` maps request indices to token slot indices. Tracks which KV slots belong to each active request.

## Profiling

Optional per-step JSONL profiling, gated by `NKIPY_SERVING_PROFILE=1`. Zero overhead when disabled.

Core utilities in `nkipy_serving/profiling.py`:
- `PROFILING_ENABLED` — module-level bool from env var
- `ProfileWriter` — buffered JSONL writer (one per trace file, flush every 50 records)
- `StepTimer` — interval-based phase timing (mark named phases, get `{t_phase: dur, t_total: dur}`)

Instrumentation points:

| Trace file | Writer location | What it captures |
|---|---|---|
| `scheduler_steps.jsonl` | `_SchedulerCore._run_single_batching_step` | Per-step phases: admit, classify, batch_build, device_wait, process_results, retire. Includes overhead % (non-device time / total). |
| `scheduler_poll.jsonl` | `run_scheduler_process` main loop | ZMQ poll + message handling duration per iteration. |
| `ipc_breakdown.jsonl` | `_SchedulerCore._forward_step` | Per-step IPC overhead from `WorkerCoordinator`: shm_write, SHM publish, completion wait, first_result, combine. |
| `worker_{rank}_steps.jsonl` | `_worker_main` | Per-worker: shm_read, model_forward, result_put. |
| `http_events.jsonl` | `TokenizerManager` | Request lifecycle: submit_ts, TTFT, total_ms, per-token queue_to_yield delay. |

All traces write to `NKIPY_SERVING_PROFILE_DIR` (default `/tmp/nkipy_serving_profile/`).

## Key Design Constraints

1. Scheduler is a pure CPU coordinator — never runs NRT, never claims a Neuron core. Must NOT set `RANK`, `LOCAL_RANK`, `NEURON_RT_VISIBLE_CORES`, or `WORLD_SIZE`.
2. TP settings come from config (`tp_degree` only), not env vars. Rank/world/profile are derived internally.
3. Tokenizer/server IPC uses ZMQ PUSH/PULL (`ipc://`). Scheduler/worker IPC uses shared memory plus a multiprocessing queue for lifecycle events.
4. `chunked_prefill_size` cannot be `0` (`-1` disables); `enable_mixed_chunk` requires `> 0`.
5. Python package code belongs under `src/nkipy_serving` (not repo root).
6. HTTP readiness: `/health` is liveness (process up), `/ready` returns 200 only after warmup/init completes. Worker warmup now includes both compilation and one synthetic execution per configured startup bucket path, so 503/500 covers first-touch kernel warmup as well as pure init.
7. Streaming SSE follows OpenAI delta protocol: first chunk = role only, middle chunks = content delta only, final chunk = empty delta + finish_reason + usage.
8. Incremental detokenization uses offset-based decode. Stream events are suppressed when the decoded text ends with the Unicode replacement character (incomplete multi-byte sequence).
9. Error propagation: `SchedulerError` carries `aborted: bool`. Serving endpoints catch it and return proper HTTP error codes.
10. Per-model config defaults via `ModelSpec.config_defaults`, injected at lowest precedence: `global defaults < model defaults < config file < env vars < overrides`.
11. TP/EP worker coordination uses a shared-memory generation protocol, not a per-step global barrier. Workers must observe a new generation, run the step, then publish status/output before the scheduler reuses the slot. Distributed nkipy sampled-output batches also carry a per-request `sample_mask` plus `requested_topk`, so partial prefill rows do not emit sampled tokens and TP ranks can publish compact top-k candidates instead of logits.
