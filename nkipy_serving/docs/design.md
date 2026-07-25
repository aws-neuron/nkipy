# Design

Five design topics unique to nkipy-serving, each self-contained.

---

## 1. NKI Attention Backend

### Overview

nkipy-serving uses NKI (Neuron Kernel Interface) kernels for attention, compiled to NEFFs via neuronx-cc. There is no CUDA or Pallas — attention runs directly on NeuronCore v2 hardware.

Source: `nkipy_serving/attention/nki_blocksparse_flash_attention.py`, `nkipy_serving/attention/blocksparse_flash_attention/`

### Unified Kernel Design

A single compiled kernel handles both prefill and decode. The kernel signature is always the unified mixed (prefill+decode) form with parameters:

- `include_prompt_in_ctx=True` — prompt tokens are planned as context tiles
- `skip_active=True` — no separate active self-attention path
- `active_mask=None` — no active block-diagonal mask tensor

Decode-only or prefill-only steps pass a **dummy tile plan** for the missing direction (0 loop steps + out-of-bounds write-back indices), so the unused path becomes a no-op. This avoids maintaining separate kernels or recompiling per batch composition.

### Tile Plan Scheduling

The `FlashAttentionPlanner.MakeTilePlan()` builds per-step tile plans on CPU. A tile plan specifies which Q tiles attend to which KV blocks, with masks for partial tiles.

To keep kernel shapes fixed per `token_bucket`, tile plan tensors are padded to deterministic maximum sizes computed by `compute_max_tile_counts()`:

```python
def compute_max_tile_counts(token_bucket, max_context_len, max_requests, block_size):
    # Returns (max_num_prefill_tiles, max_num_decode_tiles)
    # These pad tile plans so each token_bucket compiles once
```

### Dummy Plans

When a step is decode-only, a dummy prefill plan is passed (and vice versa). A dummy plan has:
- 0 loop steps (the dynamic loop body never executes)
- Out-of-bounds write-back indices (writes go to scratch, not output)

This means every forward step uses the same compiled kernel regardless of whether the batch is pure-prefill, pure-decode, or mixed.

### Prepared Step Inputs

The tile-plan tensors and padded slot mapping are now cached per bucket in `nkipy_serving/attention/nki_step_inputs.py`.

- Models allocate one `PreparedNkiStepInputs` object per bucket and keep its device tensors resident
- Init uploads dummy prefill/decode plans once
- Each step rewrites only the active slot mapping and tile-plan tensors
- If a step switches from prefill-only to decode-only (or vice versa), the inactive side is restored to the cached dummy plan instead of allocating fresh tensors

This utility is shared by GPT-OSS, Qwen3 dense, and Qwen3 MoE so attention-step preparation uses the same API and caching behavior everywhere.

For pure decode steps, the shared path now bypasses the generic tile planner and fills a reusable host-side decode plan buffer in place. Decode masks come from a cached lookup table, then the prepared host tensors are uploaded into the resident device buffers. Prefill and mixed steps still use the generic planner path.

GPT-OSS prefill still has a per-layer CPU MoE scheduling boundary, but the device work on each side of that boundary is now fused per layer: the pre-router segment runs `pre_attn + kv_update + attention + post_attn + router`, then the post-CPU segment runs `output_init + blockwise_add_residual`.

### seqlen_q >= 128 Constraint

The NKI attention kernel requires `seqlen_q >= 128` (`B_P_SIZE = nl.tile_size.pmax`). For small decode buckets (e.g., 1, 2, 4, 8), Q/K/V are padded to 128 tokens for the attention call only. After attention, the output is sliced back to the original `token_bucket` size for the rest of the model.

### Compilation and Caching

- One attention kernel compiled per `token_bucket` value (using fixed tile-plan shapes)
- Kernels are cached per executor by token_bucket (each executor owns its compiled kernel dicts)
- Q, K, V and the KV cache stay on device throughout
- Tile plans are still prepared on CPU per step, but decode uses a specialized in-place fast path and the corresponding device tensors are cached and rewritten in place via `PreparedNkiStepInputs`

---

## 2. Bucket-Based Static Compilation

### Why Buckets

Neuron hardware requires static tensor shapes for compilation. Each distinct shape produces a separate NEFF (Neuron Executable File Format). Buckets discretize the shape space so that a bounded number of NEFFs covers all runtime batch sizes.

Source: `nkipy_serving/config.py`, `nkipy_serving/runtime/shape_guard.py`

### Two Dimensions

- **`token_buckets`** (default: `32, 128, 1024, 4096`) — used for EXTEND mode. The total number of tokens in the batch is rounded up to the smallest bucket that fits.
- **`request_buckets`** (default: `1, 2, 4, 8, 16, 32`) — used for DECODE mode. Since each decode request contributes exactly 1 token, `token_bucket == batch_size`, so the request count is bucketed.
- **`max_context_len`** — when unset, `load_runtime_config()` derives it from the resolved HF model config when possible. GPT-OSS is intentionally pinned to `4096` by default to avoid implicitly compiling/loading 128K long-context surfaces.

### Selection

`select_bucket()` picks the smallest bucket >= the required size:

```python
def select_bucket(required, buckets, axis_name):
    for bucket in buckets:
        if required <= bucket:
            return bucket
    raise RuntimeError(f"{axis_name} bucket miss")
```

Inputs are padded to the selected bucket size.

### Minimum Bucket = 2

The Neuron compiler squeezes dimensions of size 1, which can break shape assumptions. The minimum useful bucket is 2. Single-request decode pads to bucket 2.

### Precompilation (Warmup)

During warmup, the runtime compiles and then executes representative synthetic `(bucket, mode)` steps to avoid first-touch kernel stalls on the first real request. This startup warmup is always enabled and is part of the normal `/ready` path.

### Kernel Caching

Most model families still own their compiled kernels through executor-local structures. Dense models use `_CompiledKernels` with embed + full-graph kernels. MoE models use `_CompiledKernels` with per-layer + full-decode kernels. NKI attention kernels are cached in `AttentionKernelCache` for per-layer execution in MoE prefill.

Some runtime kernels also use a signature-based global NEFF catalog so fresh build directories can reuse already-compiled artifacts:

- Shared support kernels use `NKIPY_SERVING_GLOBAL_NEFF_CACHE_DIR`, default `/tmp/nkipy_serving_neff_catalog`.
- Model-specific fused-kernel paths may add their own global catalogs when their signatures are stable across build directories.

The support catalog is namespace-scoped and covers reusable attention/KV kernels, compressor kernels, MoE support kernels, and logits processor kernels. Local build directories still write canonical records under the config hash, but a populated global catalog avoids per-build compile locks and avoids generating duplicate local `.neff` files.

When reading startup profiles, do not equate warmup time with total `/ready` time. Readiness includes parent/server orchestration, worker process setup, model/runtime setup, weight loading, device allocation, kernel load, and synthetic warmup. The worker `*_warmup_rank_*.jsonl` span isolates only the final first-touch warmup stage.

### Build Directory Layout

Compiled NEFFs are stored under a config-hash subdirectory so that different configurations never collide:

```
{nkipy_build_dir}/
├── a3f1c8e2d1/                        ← MD5 hash of config fields (first 10 hex chars)
│   ├── config.json                    ← human-readable manifest of hashed fields
│   ├── rank0/
│   │   ├── gpt_oss_embed_s16_819aa67f3239/
│   │   │   └── gpt_oss_embed_s16.neff
│   │   └── ...
│   └── rank7/
│       └── ...
└── b7e2f4a9c3/                        ← different config = new directory
    ├── config.json
    └── ...
```

`RuntimeConfig.config_build_dir()` computes the hash from fields that affect compiled kernel identity: `model_id`, `model_dtype`, `attention_backend`, `attention_backend_version`, `paged_attn_impl`, `moe_kernel_version`, `nkipy_compiler_args`, `compile_options_hash`, TP/EP/replica/attention-lane degrees, DSV4 state size and product warmup/fusion knobs, `kv_cache_block_size`, `max_context_len`, `decode_graph_scope`, HF checkpoint fields, and synthetic prototype fields. The manifest is written lazily on first access.

### Config Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `token_buckets` | `32,128,1024,4096` | EXTEND buckets |
| `request_buckets` | `1,2,4,8,16,32` | DECODE buckets |

---

## 3. Shared-Memory Worker Coordination

### Design Goals

Minimal per-step overhead for tensor-parallel execution. The scheduler never touches NRT or claims Neuron cores.

Source: `nkipy_serving/runtime/worker_coordinator.py`, `nkipy_serving/mem_cache/memory_pool.py`

### Process Topology

```
Scheduler (rank=-1, CPU only)
  │
  ├── Worker 0  (global_rank=0,  tp_rank=0, ep_rank=0, core=device_offset+0)
  ├── Worker 1  (global_rank=1,  tp_rank=1, ep_rank=0, core=device_offset+1)
  ├── ...
  ├── Worker 7  (global_rank=7,  tp_rank=7, ep_rank=0, core=device_offset+7)
  ├── Worker 8  (global_rank=8,  tp_rank=0, ep_rank=1, core=device_offset+8)   [EP only]
  ├── ...
  └── Worker N-1
```

The scheduler creates `WorkerCoordinator` which spawns `total_workers = tp_degree * ep_degree * replica_degree` worker processes. Each worker:
- Sets `NEURON_RT_VISIBLE_CORES` to `device_offset + global_rank` (default `device_offset=0`)
- Initializes NRT and loads the model
- Enters a forward loop waiting for commands

### Shared Memory IPC

ForwardBatch data is passed via pre-allocated `multiprocessing.shared_memory.SharedMemory` buffers:

- `input_ids`, `positions`, `slot_mapping`, `seq_lens`, `block_tables`, `query_start_loc`
- Buffer sizes computed once from `RuntimeConfig` (max token/request buckets)
- A separate shared control segment holds the step command block, one status slot per worker, and output slots for scheduler-visible TP ranks/lane rows
- The nkipy control path now uses the multiprocessing queue only for worker lifecycle events (`worker_ready`, crashes, shutdown acks)

This avoids per-step serialization of numpy arrays.

### Tokenizer / Detokenizer Architecture

Tokenization and detokenization run outside the scheduler's main loop to avoid
blocking the decode hot path:

- **TokenizerManager** (HTTP process): pre-tokenizes text prompts into
  `input_ids` before sending to the scheduler via ZMQ. The scheduler receives
  pre-tokenized requests and never calls `tokenizer.encode()`.
- **DetokenizerManager** (separate subprocess): receives compact token-level
  events from the scheduler (`batch_tokens` for per-step token IDs, `finish`
  for completion metadata). Performs incremental UTF-8-buffered text decoding,
  stop-string trimming, logprob token text decoding, and final response
  formatting. Forwards formatted `token` and `final` responses to the
  tokenizer manager. Control messages (shutdown, metrics) are forwarded
  unchanged.

The scheduler still performs a lightweight incremental decode for requests with
text stop strings (`state.stop_strs`), since stop-string detection requires
decoded text. Requests without text stop strings skip this decode entirely.
This matches upstream SGLang's pattern where the scheduler decodes for stop
checking and the detokenizer independently decodes for user output.

ZMQ topology:

```
TokenizerManager --(PUSH/connect)--> scheduler_input_ipc --(PULL/bind)--> Scheduler
Scheduler --(PUSH/connect)--> detokenizer_ipc --(PULL/bind)--> DetokenizerManager
DetokenizerManager --(PUSH/bind)--> scheduler_output_ipc --(PULL/connect)--> TokenizerManager
```

### Overlap Scheduling

When `overlap_schedule=True` (default), the scheduler interleaves request
handling with device execution.  Between `dispatch_forward_step` (non-blocking
SHM write, ~1ms) and `collect_forward_step` (blocking poll, ~36ms), the
scheduler calls `recv_fn()` to drain ZMQ messages.  This hides the cost of
`handle_message` behind the device wait window.

```
Normal:   recv → admit → build → dispatch → collect [36ms idle] → process → retire
Overlap:  admit → build → dispatch → [recv during 36ms] → collect → process → retire
```

This matches SGLang upstream's `event_loop_overlap` pattern but without
deferred `process_results` — on Neuron, `spike_execute` is synchronous so
there is no GPU-CPU stream overlap.  Results are processed immediately after
collect, same as the normal path.

When async spike becomes available, the `poll_forward_step()` /
`collect_forward_step_result()` methods on `WorkerCoordinator` provide a
non-blocking poll interface.  The overlap loop can then check for completion
without blocking, enabling true CPU-device overlap.

Config: `overlap_schedule` field in RuntimeConfig, env var
`NKIPY_SERVING_OVERLAP_SCHEDULE`.  Disable with `=0` or `=false`.

### SchedulerKVPoolStub

The scheduler process uses `SchedulerKVPoolStub` instead of a full `MHATokenToKVPool`. The stub tracks slot indices and block allocation but does NOT allocate per-layer KV arrays. This saves ~1 GB of memory for 8B models, since actual KV data only exists in worker processes.

### Chunked Prefill

Chunked prefill is scheduler-driven. For `EXTEND` requests that are split into
multiple chunks, the scheduler must build each request's block table from the
effective post-chunk sequence length, not from the pre-chunk `state.seq_len`.

Without that rule, later chunks of a long request can build a block table that
is too short for the active chunk and fail inside the blocksparse prefill tile
planner.

This fix is model-agnostic. It was validated on long-context GPT-OSS runs,
including exact `4096 -> 1` and `8192 -> 1` prompts on a `10k`-context setup.

### Lockstep Execution

All workers execute the same forward step in lockstep:
1. Scheduler writes ForwardBatch to shared memory
2. Scheduler writes a new generation + command into the shared control block
3. Workers independently observe the generation change, read shared memory, and execute NRT forward (with all-reduce)
4. Each worker publishes completion into its status slot; scheduler-visible output ranks also publish SHM outputs for distributed top-1/top-k
5. Distributed nkipy sampled-output workers (Qwen3 dense, `qwen3-moe`, `gpt-oss`) keep the LM head TP-sharded and choose one of two device paths per batch: TP-local top-1/top-k candidate outputs for pure greedy decode, or TP all-gather followed by one raw NKI sampler kernel for any non-greedy batch
6. The default `dense_local_topk=1` path now uses dedicated top-1 output buffers and a top-1 merge fast path; larger `k` opts into the wider greedy candidate path only when requested
7. The command metadata is compact and fixed-schema: workers derive array dtypes and most shapes locally instead of receiving per-field dtype/shape descriptors every step
8. The scheduler merges per-rank candidates only for the greedy path; non-greedy batches receive final `next_token_ids` directly from the shared device sampler
9. Scheduler waits on completions / output availability, then finalizes token selection

This keeps the control plane off `mp.Barrier`, so worker wakeup is staggered and overlaps with real execution instead of becoming a stop-the-world futex rendezvous.

### Distributed Sampled Outputs

Distributed nkipy sampled-output models (Qwen3 dense, `qwen3-moe`, `gpt-oss`) have three decode paths, all managed by `LogitsProcessor`:

- **Greedy, no logprobs**: rank-local top-1/top-k candidates, scheduler merges across TP ranks.
- **Non-greedy, no logprobs**: all-gather LM-head logits across TP on device, NKI CDF sampler, emit `next_token_ids`.
- **Logprobs requested** (greedy or non-greedy): force all-gather path, NKI sampler + device log_softmax + top-k extraction, emit `next_token_ids` + `chosen_logprobs` + `topk_logprob_vals` + `topk_logprob_ids` through SHM.

When logprobs are requested, the batch pays one extra TP all-gather (if greedy) plus ~0.2 ms for log_softmax + top-k at TP=8. Zero overhead when logprobs are not requested.

On TP+EP model variants, these LM-head kernels still load against the global communicator (`rank_id=global_rank`, `world_size=total_workers`), not a TP-local communicator size. The TP/EP scoping still comes from `tp_replica_groups`, but the NEFF load must match the process-wide communicator or NRT rejects the kernel with a world-size mismatch during worker warmup.

The logprobs-specific sampler kernels compile eagerly with the rest of the bucket's LM-head kernels, so warmup covers the full sampled-output path up front.

For dense models, the LM-head stage must use the padded hidden buffer's leading dimension as `token_bucket`. The scheduler's `real_total_tokens` is smaller on padded prefill/decode steps, but `LogitsProcessor` kernels are specialized to the actual device buffer shape.

### NRT Environment Setup

For `total_workers > 1`, the coordinator configures:
- `NEURON_RT_VISIBLE_CORES` — per-worker core assignment (one core per global rank)
- `device_offset` / `NKIPY_SERVING_DEVICE_OFFSET` / `--device-offset` — optional base core index used to shift those per-worker assignments without changing logical TP/EP rank numbering
- `NEURON_RT_ROOT_COMM_ID` — defaults to `localhost:62182` for collective communication
- `RANK` / `WORLD_SIZE` — global rank and total workers for CC bootstrap

### Expert Parallelism (EP)

When `ep_degree > 1`, MoE expert weights are sharded across EP ranks:
- Each rank holds `local_num_experts = num_experts // ep_degree` experts
- Router weight stays full (global top-k needed for correct routing)
- After local expert computation, `cc.all_reduce` across EP groups sums partial outputs
- Then `cc.reduce_scatter` across TP groups for sequence parallelism

Replica groups (built by `nkipy_serving/runtime/parallel_groups.py`):
- **TP groups**: consecutive ranks `[[0..7], [8..15], ...]` — used for all_gather/reduce_scatter
- **EP groups**: strided ranks `[[0,8,16,...], [1,9,17,...], ...]` — used for all_reduce

EP requires `NEURON_LOGICAL_NC_CONFIG=1` for per-core addressing on trn2.

---

## 4. Blockwise MoE Kernels

### Overview

GPT-OSS is a 120B Mixture-of-Experts model. The MoE layer routes tokens to a subset of experts (`experts_per_token`). nkipy-serving implements MoE via NKI blockwise kernels that process one expert-block at a time.

Source: `nkipy_serving/ops/moe/blockwise_index.py`, `nkipy_serving/ops/moe/blockwise_index_ext.c`, `nkipy_serving/ops/moe/blockwise_nki.py`, `nkipy_serving/models/gpt_oss/`

### CPU-Side Scheduling

`blockwise_index.py` builds the scheduling arrays consumed by NKI kernels. The fast path uses a small native C extension (`blockwise_index_ext.c`) that is built lazily, cached under `/tmp/nkipy_serving_native/`, and preloaded during worker warmup so the first real MoE-prefill request does not pay the host extension build cost.

- `block_to_expert`: `[num_blocks]` int8 — maps each block to its expert id
- `token_position_to_id`: `[num_blocks, block_size]` int32 — maps each position in a block to a token id (or `SKIP_DMA` for padding)

Given per-token `top_k_indices`, `get_blockwise_expert_and_token_mapping()` bins tokens into blocks of size 128 (`TILE_SIZE`), sorted by expert. Sentinel values control the kernel:
- `SKIP_DMA (-1)` — skip this position (padding within a block)
- `SKIP_BLOCK (-2)` — skip the entire block

### Prefill Kernel

`blockwise_nki_static` (+ `blockwise_add_residual`):
- CPU builds dynamic `(block_to_expert, token_position_to_id)` per step based on actual routing decisions
- Supports arbitrary token counts
- Used for all EXTEND steps and DECODE steps where `token_bucket > 128`

### Decode Kernel

`blockwise_nki_decode` (+ `blockwise_decode_add_residual`):
- Static block mappings baked at compile time (1 block per expert, all tokens replicated)
- Hidden states loaded once and reused across all experts
- `BUFFER_DEGREE=3` for weight prefetching
- No CPU scheduling overhead
- Requires `token_bucket <= 128` (TILE_SIZE)
- Activated when `forward_mode==DECODE` and `token_bucket <= 128`

### TP Strategy

- Expert dimension is kept intact per TP group (not sharded by TP)
- Intermediate dimension (I) is sharded across TP ranks
- Collective: `reduce_scatter` (not all_reduce) in both prefill and decode paths

### EP Strategy (Expert Parallelism)

- Expert dimension is sharded across EP ranks: each holds `num_experts // ep_degree`
- Intermediate dimension is still sharded across TP ranks
- Router affinities are sliced to local experts on-device
- CPU-side block scheduling remaps global expert IDs to local IDs
- `all_reduce` across EP groups sums partial MoE outputs before TP reduce_scatter

### Sequence Parallelism

Per-layer pattern matching NeuronPyExps:
- `all_gather` over the packed token axis (dim=0) before the MoE layer
- EP `all_reduce` after local expert computation (when `ep_degree > 1`)
- `reduce_scatter` after the MoE layer for TP sequence parallelism

---

## 5. Per-Model Executor Pattern

### Philosophy

Each model is a self-contained subpackage. The model owns its forward pass, compilation strategy, and hardware-specific decisions.

Source: `nkipy_serving/models/registry.py`, `nkipy_serving/models/qwen3_dense/`, `nkipy_serving/models/gpt_oss/`, `nkipy_serving/models/qwen3_moe/`, `nkipy_serving/models/deepseek_v4/`

### ModelSpec Registry

`ModelSpec` is the contract between the runtime and a model:

```python
@dataclass(frozen=True)
class ModelSpec:
    build_config: Callable     # (runtime_config, tp_rank, ep_rank) -> model_config
    init_weights: Callable     # (model_config) → weights
    create_executor: Callable  # (model_config, kv_pool, runtime_config) → executor
    build_kv_metadata: Callable  # (model_config) → (num_layers, num_kv_heads, head_dim, dtype)
    config_defaults: dict      # injected into RuntimeConfig at lowest precedence
```

Models are registered in `_MODEL_SPECS`:

```python
_MODEL_SPECS = {
    "<qwen3_dense_family>": ModelSpec(..., config_defaults={"attention_backend": "NKIBlockSparseFlashAttention", ...}),
    "gpt-oss": ModelSpec(..., config_defaults={"execution_backend": "nkipy", ...}),
}
```

`resolve_model_spec()` resolves model IDs including HF-style ids (`Qwen/Qwen3-*` dense ids → the Qwen3 dense family spec, `unsloth/gpt-oss-*` → `gpt-oss`).

### Anatomy of a Model Subpackage

Each model subpackage (e.g., `qwen3_dense/`) contains:

1. **`config.py`** — Config dataclass (e.g., `Qwen3DenseModelConfig`). Zero nkipy imports (backend-agnostic).
2. **`weights.py`** — Weight dataclasses, HF loading, TP sharding, `init_*_weights()`. Zero nkipy imports (backend-agnostic).
3. **`graph_fns.py`** — Device-traceable NKI functions passed to `DeviceKernel.compile_and_load()`.
4. **`executor.py`** — Executor class: owns forward pass, kernel compilation, KV cache interaction.
5. **`eager_executor.py`** (Qwen3 dense, Qwen3 MoE, and GPT-OSS) — Stage-level composable executor for debug: `forward_cpu()` + swappable attn fragment + prefill/decode mode dispatch. See [§6](#6-eager-executors-debug-surface).
6. **`__init__.py`** — Re-exports public names so `registry.py` and generated kernel sources work unchanged.

The config/weights boundary is intentionally kept free of device imports to leave space for adding a CPU backend.

### Execution Modes

Each model has a fixed execution mode:

- **Qwen3 Dense**: All-layers-one-graph with embedding outside. NKI attention required. No mode selection.
- **GPT-OSS**: Full decode graph for decode; per-layer prefill kernels for prefill (CPU MoE scheduling boundary per layer). No mode selection.
- **Qwen3 MoE**: Per-layer prefill graphs (fixed mode).

### Adding a New Model

1. Create `nkipy_serving/models/<model_name>.py` with config, weights, loader, executor
2. Implement `get_<model>_kv_metadata()` returning `(num_layers, num_kv_heads, head_dim, dtype)`
3. Register in `_MODEL_SPECS` in `registry.py` with appropriate `config_defaults`
4. The executor's `__init__` can enforce model-specific constraints (e.g., required attention backend)

### Comparison: Qwen3 Dense vs GPT-OSS

| Aspect | Qwen3 Dense | GPT-OSS |
|--------|-------------|---------|
| Architecture | Dense transformer | MoE (120B, 128 experts) |
| Execution mode | All-layers-one-graph (embedding outside) | Full decode graph + per-layer prefill kernels |
| Attention | NKI only | NKI only |
| MoE | — | Blockwise NKI kernels |
| TP baseline | Flexible | `tp_degree=8` (1 KV head per rank) |
| Logprobs | Supported | Supported (output-token logprobs via LogitsProcessor) |
| Weight loading | Full numpy arrays | Metadata only (direct device upload) |

---

## 6. Eager Executors (Debug Surface)

Per-model debug class (`Qwen3DenseEagerExecutor`, `Qwen3MoeEagerExecutor`,
`GptOssEagerExecutor`, shared base in `models/common/eager_executor_base.py`)
built from stage-level `@jit` fragments. Not a serving path — single-process,
TP=1/EP=1 in-process; no warmup, no reload, no fused all-layer graph.

**Layer shape**:
```
dense:       pre_attn → attn → post_attn
MoE decode:  pre_attn → attn → post_attn → router_moe           (fused)
MoE prefill: pre_attn → attn → post_attn → router_prefill
                   → [CPU block schedule] → moe_dispatch_prefill
```
Each arrow is a fragment, lazily compiled. The MoE prefill CPU boundary is
the same one production uses; `build_prefill_moe_schedule` is shared.

**Knobs**:
- `forward_cpu(...)` — pure-numpy body (no device dependency in the method;
  `__init__` still compiles fragments, so CPU-only tests either call
  `forward_cpu` after a Neuron-backed build or use the `__new__` bypass
  pattern in `tests/test_moe_forward_cpu.py`).
- `exe.attn = jit(cpu_attn_fn, device=False)` — swap attention to CPU while
  every other stage stays on device.
- `hf_num_hidden_layers=N` on the model config loads a truncated checkpoint.

**Constraints**: no-SP only (full hidden per rank, `all_reduce`); production
SP prefill correctness is covered by the multi-process serving tests. Decode
MoE requires `token_bucket ≤ MOE_BLOCK_SIZE=128`. Prefill mode is uncapped.

**Per-model capability at TP=1 in-process**:
| Model | `forward_cpu` | CPU-attn swap | Full device `forward()` |
|---|---|---|---|
| Qwen3 Dense | yes | yes | yes |
| Qwen3 MoE | yes | yes | no (NKI attn needs TP≥4) |
| GPT-OSS | yes | no* | no (NKI attn needs TP≥8; no-SP MoE kernel overflows SBUF for 128 experts) |

\* GPT-OSS CPU-attn swap would work in principle but isn't exercised in-process
because the device `forward()` it feeds into can't run at TP=1. The GPT-OSS
eager device `forward()` is **not** covered by the multi-process TP8 serving
test (which exercises the production `GptOssExecutor`, not the eager
composition). Phase 1 leaves this intentionally unvalidated; a TP8 eager test
would need its own opt-in subprocess harness.

**Retained test files**:
| File | Neuron? | Uses real HF weights? |
|---|---|---|
| `test_moe_forward_cpu.py` — MoE `forward_cpu` + numeric tests for `cpu_moe_dispatch_{swish,swiglu_oai}` | No | No (synthetic, `__new__` bypass) |
| `test_prefill_moe_schedule.py` — `build_prefill_moe_schedule` (SKIP_DMA, EP remap, partitioning) | No | No |
| `test_moe_eager_executor_contracts.py` — eager executor wiring contracts that previously regressed | No | No |
| `test_qwen3_moe_tp4_device.py` and `test_gpt_oss_tp8_device.py` — production serving paths | Yes | Yes |

**Typical bisect ladder** (Qwen3 only — GPT-OSS stops at step 1):
1. `forward_cpu` vs reference implementation (synthetic or truncated real)
2. device `forward()` with CPU-attn swap vs `forward_cpu` top-1 match
3. full device `forward()` vs the swap (Qwen3 Dense only at TP=1; MoE at TP≥4)
