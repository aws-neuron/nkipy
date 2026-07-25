# Getting Started

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- Neuron SDK 2.31 host packages: driver 2.29.0.0, runtime 2.33.10.0,
  and collectives 2.33.10.0. Install them from the
  [Neuron package repository](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup.html).
  The compatible `neuronx-cc` and `nki` Python wheels are pinned in
  `pyproject.toml` and installed by `uv sync`.
- AWS Trn2 or Inf2 instance (for device execution)
- HuggingFace model checkpoints (local or cached)

## Install

```bash
cd nkipy
uv sync --group serving --group test
cd nkipy_serving
```

## Configuration

### RuntimeConfig

All runtime behavior is controlled by `RuntimeConfig` (defined in `nkipy_serving/config.py`). Configuration sources are merged in order of precedence:

```
global defaults < model defaults < config file < env vars < explicit overrides
```

- **Global defaults**: Hardcoded in `config.py`
- **Model defaults**: `ModelSpec.config_defaults` in `registry.py` (e.g., Qwen3 sets attention backend defaults)
- **Config file**: JSON file passed via `--config` or `NKIPY_SERVING_CONFIG_FILE`
- **Env vars**: `NKIPY_SERVING_*` environment variables
- **Explicit overrides**: Programmatic overrides dict

### Config Reference

| Setting | Env var | Default | Description |
|---------|---------|---------|-------------|
| `execution_backend` | `NKIPY_SERVING_EXECUTION_BACKEND` | `nkipy` | `nkipy` (Neuron) |
| `model_id` | `NKIPY_SERVING_MODEL_ID` | `Qwen/Qwen3-0.6B` | Dense Qwen3 uses HF model IDs directly. Other supported direct keys are `gpt-oss`, `qwen3-moe`, and `deepseek-v4`. |
| `decode_graph_scope` | `NKIPY_SERVING_DECODE_GRAPH_SCOPE` | `embed_layers` | GPT-OSS full-decode scope: `embed_layers` (LM head always separate via LogitsProcessor) |
| `model_dtype` | `NKIPY_SERVING_MODEL_DTYPE` | `bf16` | Model dtype |
| `attention_backend` | `NKIPY_SERVING_ATTENTION_BACKEND` | `NKIBlockSparseFlashAttention` | Attention backend |
| `hf_model_id` | `NKIPY_SERVING_HF_MODEL_ID` | — | HF checkpoint (e.g. `Qwen/Qwen3-0.6B`) |
| `tp_degree` | `NKIPY_SERVING_TP_DEGREE` | `1` | Tensor parallelism degree |
| `ep_degree` | `NKIPY_SERVING_EP_DEGREE` | `1` | Expert parallelism degree (MoE models only, requires `NEURON_LOGICAL_NC_CONFIG=1`) |
| `replica_degree` | `NKIPY_SERVING_REPLICA_DEGREE` | `1` | Full-model replica count. Qwen3/GPT-OSS keep `1`; DeepSeek-V4 can use replicas as part of its TP/EP/attention-lane layout. |
| `attention_dp_degree` | `NKIPY_SERVING_ATTENTION_DP_DEGREE` | `1` | Attention-DP lane count for DeepSeek-V4-style hybrid layouts. |
| `device_offset` | `NKIPY_SERVING_DEVICE_OFFSET` | `0` | Base Neuron core index for worker placement. Worker `global_rank=r` binds to core `device_offset + r`. Also settable via `--device-offset`. This changes placement only; TP/EP logical ranks and config-hash build directories stay unchanged. |
| `token_buckets` | `NKIPY_SERVING_TOKEN_BUCKETS` | `32,128,1024,4096` | EXTEND token buckets |
| `request_buckets` | `NKIPY_SERVING_REQUEST_BUCKETS` | `1,2,4,8,16,32` | DECODE request buckets |
| `chunked_prefill_size` | `NKIPY_SERVING_CHUNKED_PREFILL_SIZE` | `4096` | Max tokens per prefill chunk (`-1` to disable) |
| `enable_mixed_chunk` | `NKIPY_SERVING_ENABLE_MIXED_CHUNK` | `false` | Overlap extend+decode in single batch |
| `prefix_cache_enabled` | `NKIPY_SERVING_PREFIX_CACHE_ENABLED` | `false` | Enable radix/chunk prefix cache |
| `prefix_cache_type` | `NKIPY_SERVING_PREFIX_CACHE_TYPE` | `radix` | `radix` or `chunk` |
| `kv_pool_size` | `NKIPY_SERVING_KV_POOL_SIZE` | `16384` | Total token slots in shared KV pool |
| `kv_cache_block_size` | `NKIPY_SERVING_KV_CACHE_BLOCK_SIZE` | `32` | KV cache block size |
| `max_context_len` | `NKIPY_SERVING_MAX_CONTEXT_LEN` | model-derived (`4096` fallback; GPT-OSS pinned to `4096`) | Max context length per request. Also settable via `--max-model-len` CLI flag. When unset, runtime config derives it from the resolved HF model config when possible; GPT-OSS stays on `4096` by default. This affects compiled attention kernel shapes (tile plan padding) and is included in the config hash. |
| `request_timeout_s` | `NKIPY_SERVING_REQUEST_TIMEOUT_S` | `600` | Request timeout in seconds (`0` to disable) |
| `nkipy_build_dir` | `NKIPY_SERVING_BUILD_DIR` | `/tmp/build` | Root build directory for compiled NEFFs. NEFFs are stored under `{nkipy_build_dir}/{config_hash}/rank{N}/`. The config hash is derived from model/compiler/parallelism settings; a `config.json` manifest is written per hash for inspection. |

TP config uses `tp_degree` only; rank, world, and profile fields are derived internally.

### DeepSeek-V4 Config Reference

DeepSeek-V4 uses the common runtime fields above plus these product-path fields.

| Setting | Env var | Default | Description |
|---------|---------|---------|-------------|
| `dsv4_disable_mtp` | `NKIPY_SERVING_DSV4_DISABLE_MTP` | model default `true` for DeepSeek-V4 | Must remain true; `false` is rejected because serving currently targets the main model path only. |
| `dsv4_prepared_weight_dir` | `NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR` | — | Source root for prepared per-rank weights produced by `scripts.prepare_dsv4_rank_weights` from a converted `scripts.convert_dsv4_checkpoint` snapshot. The root must match the selected TP/EP/replica/attention-DP topology. Required for the opt-in full-model smoke. |
| `dsv4_prepared_weight_local_dir` | `NKIPY_SERVING_DSV4_PREPARED_WEIGHT_LOCAL_DIR` | — | Optional local staging root for prepared weights, usually populated by `scripts.stage_dsv4_prepared_weights`; requires `dsv4_prepared_weight_dir`. |
| `dsv4_prepared_weight_prestage` | `NKIPY_SERVING_DSV4_PREPARED_WEIGHT_PRESTAGE` | `false` | Copy prepared rank weights to `dsv4_prepared_weight_local_dir` before worker spawn to avoid rank-local shared-storage thundering herds. |
| `dsv4_prepared_weight_prestage_workers` | `NKIPY_SERVING_DSV4_PREPARED_WEIGHT_PRESTAGE_WORKERS` | `8` | Maximum host copy workers used by prepared-weight pre-stage; must be positive. |
| `dsv4_state_size` | `NKIPY_SERVING_DSV4_STATE_SIZE` | `0` | Explicit mutable DSV4 state capacity in token slots. DeepSeek-V4 serving configs must set it to cover `max_context_len` and the largest token bucket. |
| `dsv4_product_prefill_moe_blockwise_fusion_max_rows` | `NKIPY_SERVING_DSV4_PRODUCT_PREFILL_MOE_BLOCKWISE_FUSION_MAX_ROWS` | `0` | Optional row cap for the heaviest prefill MoE blockwise fusion; `0` keeps it unlimited. |
| `dsv4_product_prefill_moe_dispatch_fusion_max_rows` | `NKIPY_SERVING_DSV4_PRODUCT_PREFILL_MOE_DISPATCH_FUSION_MAX_ROWS` | `0` | Optional row cap for the lighter prefill MoE dispatch fusion. |
| `dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows` | `NKIPY_SERVING_DSV4_PRODUCT_PREFILL_DP_ATTENTION_POST_PRE_FUSION_MAX_ROWS` | `0` | Optional row cap for fusing DP-attention all-reduce with mHC post/pre. |
| `dsv4_warmup_execute_forwards` | `NKIPY_SERVING_DSV4_WARMUP_EXECUTE_FORWARDS` | `true` | Must remain true; `false` is rejected because product startup must first-touch kernels before `/ready`. |

## Launch Examples

### nkipy TP=8

```bash
uv run python -m nkipy_serving.launch_server --config runtime.nkipy.tp8.json --port 30000
```

### nkipy TP=8 on cores 8-15

```bash
uv run python -m nkipy_serving.launch_server \
  --config runtime.nkipy.tp8.json \
  --device-offset 8 \
  --port 30000
```

Equivalent config/env forms:

```bash
# JSON config
{
  "tp_degree": 8,
  "device_offset": 8
}

# Environment override
NKIPY_SERVING_DEVICE_OFFSET=8 uv run python -m nkipy_serving.launch_server ...
```

### Multiple independent replicas with `device_offset`

There is no built-in DP controller/router in `nkipy-serving` today. The supported scale-out pattern is to launch multiple independent server instances, each on its own HTTP port and Neuron core range.

Example: two independent TP=8 replicas on one host:

```bash
# Replica 0: cores 0-7
uv run python -m nkipy_serving.launch_server \
  --config runtime.nkipy.tp8.json \
  --device-offset 0 \
  --port 30000

# Replica 1: cores 8-15
uv run python -m nkipy_serving.launch_server \
  --config runtime.nkipy.tp8.json \
  --device-offset 8 \
  --port 30001
```

For a general replica index `i`, use:

```text
device_offset = base_device_offset + i * (tp_degree * ep_degree * replica_degree)
```

For example, with `tp_degree=8, ep_degree=1, replica_degree=1`, successive server replicas would use offsets `0`, `8`, `16`, `24`, ...

If you want a single public endpoint, put a simple external load balancer in front of these replicas and route only to backends whose `/ready` endpoint returns `200`.

### Kernel-cache reuse across replicas

Independent replicas can reuse compiled kernels as long as they share the same non-placement runtime config and the same `nkipy_build_dir`.

- `device_offset` affects worker placement only; it does not change the config-hash build directory.
- If one replica has already warmed up and compiled the NEFFs, a later replica launched with the same model/compiler/parallelism config can reuse that cache even if its `device_offset` is different.
- In practice, keep `model_id`, `model_dtype`, attention backend, TP/EP settings, compiler args, and `nkipy_build_dir` the same across replicas if you want cache reuse.
- Some runtime kernels also use a signature-based global NEFF catalog (`NKIPY_SERVING_GLOBAL_NEFF_CACHE_DIR`, default `/tmp/nkipy_serving_neff_catalog`). This can let a fresh build directory reuse already-compiled support kernels by writing local canonical records that point at the global NEFFs.

### Cold-start cache race caveat

Be careful when launching multiple replicas in parallel against the same empty `nkipy_build_dir`.

- Startup compilation is not documented as cross-process serialized.
- Two cold-start replicas with the same config hash may both try to compile and write the same artifacts at the same time.
- The safe pattern is: warm one replica to `/ready` first, then launch additional replicas that reuse the populated cache.
- If you need fully parallel cold starts, give each replica a different `nkipy_build_dir` for the initial compile, or pre-populate the shared/global cache ahead of time.

### In-place weight reload

`nkipy-serving` can rewrite weights in place on a running server for the currently supported model families (`Qwen3` dense, `qwen3-moe`, and `gpt-oss`).

- Use `POST /reload_weights_from_disk`.
- `model_path` may be a local HF snapshot directory or a cached HF repo id.
- Reload is same-architecture and same-shape only. Do not use it to switch model family, hidden size, TP/EP layout, vocab, or layer count.
- Reload reuses the existing compiled kernels, then clears KV/prefix/request state before serving resumes.
- `abort_all_requests` defaults to `true`. Set it to `false` only when you know there are no active requests.
- Reload is fail-closed today: if worker reload or cache flush fails, the scheduler pauses rather than continuing to serve on uncertain state.

Example: reload a running Qwen3 server from a local cached snapshot:

```bash
SNAPSHOT_DIR="$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='Qwen/Qwen3-0.6B', local_files_only=True))
PY
)"

curl -s http://127.0.0.1:30000/reload_weights_from_disk \
  -H 'content-type: application/json' \
  -d "{\"model_path\":\"${SNAPSHOT_DIR}\",\"abort_all_requests\":true}" | jq .
```

You can also flush runtime cache state without changing weights:

```bash
curl -s http://127.0.0.1:30000/flush_cache \
  -H 'content-type: application/json' \
  -d '{"abort_all_requests":true}' | jq .
```

### GPT-OSS (MoE, TP=8)

```bash
PATH="$(pwd)/../.venv/bin:$PATH" ../.venv/bin/python -m nkipy_serving.launch_server \
  --config tests/runtime.tp8.gpt_oss.serving.test.json --port 30000
```

### GPT-OSS with Expert Parallelism (TP=8, EP=16, 128 cores)

```bash
NEURON_LOGICAL_NC_CONFIG=1 uv run python -m nkipy_serving.launch_server \
  --config tests/runtime.tp8_ep16.gpt_oss.serving.test.json --port 30000
```

Requires trn2.48xlarge (128 NeuronCores). EP shards the 128 experts across 16 EP ranks (8 experts per rank), reducing per-core HBM usage and enabling larger KV cache budgets.

## Smoke Requests

```bash
# Liveness: 200 as soon as the process is up (runtime may still be warming up).
curl -s http://127.0.0.1:30000/health

# Readiness: 200 only after the runtime has completed warmup/init.
curl -s http://127.0.0.1:30000/ready

# Version and tokenizer info
curl -s http://127.0.0.1:30000/version | jq .
curl -s http://127.0.0.1:30000/tokenizer_info | jq .

# Optional probe that runs a tiny generate request (requires readiness).
curl -s http://127.0.0.1:30000/health_generate

# List models
curl -s http://127.0.0.1:30000/v1/models | jq .

# Completions
curl -s http://127.0.0.1:30000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":2,"temperature":0.0}' | jq .

# With stop sequence
curl -s http://127.0.0.1:30000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":10,"temperature":0.0,"stop":"."}' | jq .

# Chat streaming (SSE)
curl -N http://127.0.0.1:30000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"stream":true}'
```

For the full endpoint reference, including native `/generate`, tokenization utilities, reload/flush control routes, and the explicit `501` unsupported-task routes, see [HTTP API](http_api.md).

## GPT-OSS Notes

- Model ID: `unsloth/gpt-oss-120b-BF16` (expected to be in local HF cache)
- Requires `attention_backend="NKIBlockSparseFlashAttention"`.
- GPT-OSS uses a full decode graph for decode and per-layer prefill kernels for prefill.
- `decode_graph_scope` (GPT-OSS only):
  - `embed_layers` (default): embed + fused decode body in one graph, LM-head as a separate kernel (supports filtered/unfiltered sampling dispatch)
- Model defaults (`execution_backend`, `attention_backend`, `paged_attn_impl`) are declared in `ModelSpec.config_defaults` and applied automatically when `model_id` resolves to gpt-oss.
- NKI attention baseline assumes `num_key_value_heads == tp_degree` (1 KV head per rank), so use `tp_degree=8` for the provided checkpoint.
- MoE uses two kernel paths: prefill and decode. Prefill still has a CPU scheduling boundary, but the device work around it is fused per layer (`prefill_pre_moe` before CPU routing and `prefill_post_moe` after it), and the blockwise index builder now uses a native C extension preloaded during warmup. Decode keeps the static block-mapping path with `BUFFER_DEGREE=3` when `forward_mode==DECODE` and `token_bucket <= 128`.
- GPT-OSS full decode is no-SP.
- `/v1/chat/completions` uses the checkpoint's `chat_template.jinja` via `transformers` + `jinja2`.
- Distributed nkipy serving supports greedy decode and exact device-side non-greedy sampling for `temperature`, `top_k`, `top_p`, and `min_p`.
- Pure greedy batches stay on the TP-local top-1/top-k fast path. If any request in a batch is non-greedy, the whole batch falls back to a shared full-vocab all-gather plus one raw NKI sampler kernel on device.
- Distributed nkipy serving also supports token logprobs on sampled-output models through `LogitsProcessor`; prompt logprobs are still not implemented.
- On TP+EP runs, LM-head sampling/logprobs kernels must join the global worker communicator (`total_workers = tp_degree * ep_degree * replica_degree`) even though the actual LM-head collectives are scoped to TP replica groups.

## Logprobs Notes

- `logprobs` works on distributed nkipy sampled-output models (`Qwen3` dense, `qwen3-moe`, `gpt-oss`) through the device-side `LogitsProcessor` path.
- Native `/generate` also supports SGLang-style `return_logprob`, `logprob_start_len`, and `top_logprobs_num`. See [HTTP API](http_api.md#native-generate-api).
- `NKIPY_SERVING_DENSE_LOCAL_TOPK` controls the local candidate width for greedy distributed nkipy batches. It defaults to `1`, which keeps the serving path on the dedicated top-1 fast path; larger values opt into wider candidate buffers for greedy TP merge only.

## Profiling

Set `NKIPY_SERVING_PROFILE=1` to enable per-step JSONL profiling traces. Traces are written to `NKIPY_SERVING_PROFILE_DIR` (default: `/tmp/nkipy_serving_profile/`).

```bash
NKIPY_SERVING_PROFILE=1 uv run python -m nkipy_serving.launch_server --config runtime.nkipy.tp8.json --port 30000

# After running traffic, inspect traces:
ls /tmp/nkipy_serving_profile/
# scheduler_steps.jsonl   — per-step scheduler timing (admit, build, device_wait, overhead %)
# scheduler_poll.jsonl    — poll loop timing (zmq recv + message handling)
# ipc_breakdown.jsonl     — IPC overhead per step (shm write, broadcast, collect, combine)
# worker_0_steps.jsonl    — per-worker forward timing (shm_read, model_forward, result_put)
# http_events.jsonl       — HTTP-layer request/token delivery timing (TTFT, total_ms, queue delay)

# Each line is JSON; analyze with jq:
cat /tmp/nkipy_serving_profile/scheduler_steps.jsonl | jq -c '{step, mode, overhead_pct, t_device_wait, t_total}'
```

Override the output directory:

```bash
NKIPY_SERVING_PROFILE=1 NKIPY_SERVING_PROFILE_DIR=/my/traces uv run python -m nkipy_serving.launch_server ...
```

Profiling adds negligible overhead (a few `time.perf_counter()` calls per step) and is fully gated — zero cost when disabled.

For the SHM worker loop, the default is pure `sched_yield()` polling. Set `NKIPY_SERVING_SPIN_SLEEP_WHEN_IDLE=1` to enable spin-then-sleep backoff, and use `NKIPY_SERVING_SPIN_BUSY_LOOP_S` plus `NKIPY_SERVING_SPIN_IDLE_SLEEP_S` to tune it.
