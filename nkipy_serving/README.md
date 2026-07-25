# nkipy-serving

Python-first Neuron serving runtime for text generation on AWS Trn2/Inf2.

**What makes it different:**
- **No PyTorch or JAX** in the runtime path — weights are numpy, compute is NKI kernels compiled to NEFFs
- **NKI kernels** — unified flash attention (prefill+decode), blockwise MoE, all running natively on NeuronCore v2
- **Bucket-based static compilation** — discretized shape space with precompilation for zero-JIT serving
- **SHM-polled TP control plane** — workers read per-step commands from shared memory without global barriers; the nkipy path keeps queues for lifecycle events only
- **Near-zero TP control-plane overhead** — shared-memory worker coordination keeps scheduler-side serving overhead negligible relative to model execution

## Quick Start

```bash
# From the NKIPy monorepo root
uv sync --group serving --group test
cd nkipy_serving

# Launch TP=8 on Neuron cores 8-15
uv run python -m nkipy_serving.launch_server \
  --config tests/runtime.tp8.qwen3.serving.test.json \
  --device-offset 8 \
  --port 30000

# Test
curl -s http://127.0.0.1:30000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-dense","prompt":"Hello","max_tokens":2,"temperature":0.0}' | jq .
```

`nkipy-serving` does not have a built-in DP controller/router yet. To scale out, launch multiple independent replicas on different ports with different `device_offset` values. See [Getting Started](docs/getting_started.md) for a manual multi-replica example, kernel-cache reuse notes, and the caveat that cold-start replicas sharing the same empty build cache may race during first compile.

`nkipy-serving` also supports in-place weight reload on the currently supported model families. Use `POST /reload_weights_from_disk` with `{"model_path": "...", "abort_all_requests": true}` to rewrite weights on running workers without restarting the HTTP server. Reload is same-architecture and same-shape only, flushes KV/prefix cache state, and reuses the existing compiled kernels.

The serving surface includes:
- OpenAI-compatible generation routes such as `/v1/completions` and `/v1/chat/completions`
- native SGLang-style `/generate` route for rollout-oriented use
- tokenizer and info utilities such as `/v1/tokenize`, `/v1/detokenize`, `/version`, and `/tokenizer_info`

See [HTTP API](docs/http_api.md) for the full endpoint reference and examples.

## Documentation

| Doc | Contents |
|-----|----------|
| [Getting Started](docs/getting_started.md) | Install, config reference, launch examples, smoke tests |
| [HTTP API](docs/http_api.md) | Endpoint reference for OpenAI, native, control, tokenization, and unsupported routes |
| [Architecture](docs/architecture.md) | Process model, request lifecycle, module map, data flow, scheduling, memory management |
| [Design](docs/design.md) | NKI attention, bucket compilation, shared-memory TP, blockwise MoE, per-model executors |
| [Features](docs/features.md) | Feature matrix vs upstream SGLang, supported models, open tasks |

## Tests

```bash
# Run all tests (unit + integration + device):
bash scripts/run_all_tests.sh

# Same but clear compiled kernel caches first:
bash scripts/run_all_tests.sh --clean

# Unit tests only (fast, no device):
uv run pytest -v
```

See `scripts/run_all_tests.sh` for the full test matrix.
