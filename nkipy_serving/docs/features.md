# Features

## Feature Matrix

Comparison against **upstream SGLang** (the canonical CUDA/GPU version at `sgl-project/sglang`).

| Category | Feature | SGLang (GPU) | nkipy-serving (Neuron) | Notes |
|----------|---------|:---:|:---:|-------|
| **Scheduling** | Continuous batching | Yes | Yes | |
| | Chunked prefill | Yes | Yes | Default 4096 tokens |
| | Mixed chunk (prefill+decode) | — | Yes | `enable_mixed_chunk` |
| | Zero-overhead CPU scheduler | Yes | Yes | Pure-CPU coordinator |
| | Overlap scheduling | Yes | — | |
| | Prefill-decode disaggregation | Yes | — | |
| **Memory** | Paged KV cache | Yes | Yes | Block-based, configurable `kv_cache_block_size` |
| | RadixAttention prefix cache | Yes | Yes | Radix cache with LRU eviction |
| | Quantized KV cache | Yes | — | |
| | HiCache (hierarchical) | Yes | — | |
| **Attention** | FlashInfer / FlashAttention | Yes | — | GPU-specific |
| | NKI BlockSparse FlashAttention | — | Yes | Neuron-specific, unified prefill+decode kernel |
| **Parallelism** | Tensor parallelism | Yes | Yes | Explicit workers + shared memory |
| | Expert parallelism (EP) | — | Yes | Uniform EP: shards experts across ranks, `ep_degree` config |
| | Data parallelism | Yes | — | No built-in DP controller/router yet. Manual multi-replica launch is supported by running independent instances on different ports with different `device_offset` values. |
| | Blockwise MoE kernels | Yes | Yes (NKI) | CPU-scheduled prefill + static-mapping decode |
| | Pipeline parallelism | Yes | — | |
| | Multi-node inference | Yes | — | |
| **Decoding** | Sampling (temp/top-k/top-p) | Yes | Yes (NKI device sampler) | Filtered/unfiltered fast-path dispatch |
| | Structured output (JSON/EBNF/regex) | Yes | — | |
| | Speculative decoding (EAGLE) | Yes | — | |
| | Multi-LoRA batching | Yes | — | |
| **Quantization** | FP8/INT8/FP4/AWQ/GPTQ | Yes | — | |
| **API** | OpenAI-compatible API | Yes | Yes | `/v1/completions`, `/v1/chat/completions` |
| | Tokenize / detokenize HTTP utilities | Yes | Yes | `/v1/tokenize`, `/v1/detokenize` |
| | Streaming SSE | Yes | Yes | OpenAI delta protocol |
| | In-place weight reload from disk | Yes | Yes | `POST /reload_weights_from_disk`; current Neuron path is same-architecture/same-shape only and flushes KV/prefix state after reload |
| | Logprobs | Yes | Yes | Output-token logprobs via `logprobs` parameter in chat/completion requests |
| | Native token-id generate API | Yes | Yes | Native `/generate` accepts `input_ids`, batching, and `n` |
| | Embedding / reranking | Yes | — | |
| | Responses API | Yes | — | |
| **Models** | Language models | 50+ families | 3 (Qwen3 dense, Qwen3 MoE, GPT-OSS MoE) | |
| | Multimodal models | Yes | — | |
| | Diffusion models | Yes | — | |
| | Reward / embedding models | Yes | — | |
| **Infra** | Hardware | NVIDIA/AMD GPU, Intel CPU, TPU, Ascend | AWS Neuron (Trn2/Inf2) | |
| | Framework | PyTorch + CUDA | numpy + NKI (no PyTorch/JAX) | |
| | CUDA graphs | Yes | — | N/A for Neuron |
| | Observability / metrics | Yes | Basic (throughput, TTFT, cache hit) + JSONL profiling | `NKIPY_SERVING_PROFILE=1` |

See [HTTP API](http_api.md) for the endpoint reference and request examples.

## Supported Models

Model-specific notes live under `docs/models/`.

| Model | Checkpoint | TP | Attention Backend | Execution Mode |
|-------|-----------|:---:|-------------------|----------------|
| Qwen3 Dense | `Qwen/Qwen3-0.6B` (and other Qwen3 sizes) | 1+ | NKI BlockSparse FlashAttention | All-layers-one-graph with embedding outside |
| Qwen3 MoE | `Qwen/Qwen3-*-A<size>B*` | 4+ baseline, EP supported | NKI BlockSparse FlashAttention | Per-layer prefill graphs |
| GPT-OSS MoE | `unsloth/gpt-oss-120b-BF16` | 8 baseline, EP supported | NKI BlockSparse FlashAttention (required) | Full decode graph; per-layer prefill kernels |

## Open Tasks

### Correctness

- [x] Wire sampling into scheduler for `temperature`, `top_k`, `top_p`, and `min_p`
- [x] EOS token detection (check tokenizer's `eos_token_id` during generation)
- [ ] GPT-OSS: sliding-window attention support (checkpoint alternates sliding/full layers)
- [ ] Reverify MoE kernel correctness across all supported MoE model families, including prefill/decode gather-scatter behavior
- [x] Distributed nkipy models: logprobs support via LogitsProcessor (device log_softmax + top-k, OpenAI endpoint formatting)
- [x] Distributed nkipy sampled-output models: add device-side local top-k for greedy TP merge
- [ ] Penalties / logit bias on distributed nkipy sampled-output models
- [ ] Prompt logprobs on distributed nkipy serving

### Serving

- [ ] Add DP controller / `dp_size > 1` serving path
- [ ] Watchdog (detect hung scheduler, kill process tree)
- [ ] Host KV eviction (radix cache `host_value` field exists but unused)

### Performance

- [x] GPT-OSS: dedicated decode MoE kernel (no CPU scheduling, BUFFER_DEGREE=3, static block mappings)
- [x] GPT-OSS: full decode graph
- [ ] Fused multi-request kernel path (currently loops request-by-request)
- [x] NKI attention: reduce per-step tile-plan upload overhead (shared prepared step-input cache in `nkipy_serving/attention/nki_step_inputs.py`)
- [x] Worker warmup: compile + synthetic first-touch startup stage shared across GPT-OSS, Qwen3 dense, and Qwen3 MoE
- [ ] GPT-OSS: reduce per-layer weight memory (avoid storing down_bias as [E,128,H] broadcast on device)
