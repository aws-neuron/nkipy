# GPT-OSS

GPT-OSS support in `nkipy-serving` is aimed at Neuron device execution with TP/EP and NKI attention.

## Requirements

- Model id: `unsloth/gpt-oss-120b-BF16` or another `unsloth/gpt-oss-*` checkpoint supported by the runtime
- Attention backend: `NKIBlockSparseFlashAttention`
- EP runs require `NEURON_LOGICAL_NC_CONFIG=1`

Typical EP launch:

```bash
export PATH=$(pwd)/../.venv/bin:$PATH
export NEURON_LOGICAL_NC_CONFIG=1
python -m nkipy_serving.launch_server \
  --config tests/runtime.tp8_ep16.gpt_oss.serving.test.json \
  --port 30000
```

## Execution Mode

GPT-OSS uses a fixed execution mode:

- **Decode**: Full decode graph (embed + all layers fused). `decode_graph_scope` controls scope:
  - `embed_layers` (default): embed + fused decode body in one graph, LM-head as a separate kernel via `LogitsProcessor` (supports per-batch filtered/unfiltered sampling dispatch)
- **Prefill**: Per-layer prefill kernels with CPU MoE scheduling boundary per layer

Cold compile/warmup for the full decode graph can take several minutes.

## Warmup And Compile Behavior

Worker warmup now includes:

1. compiling startup buckets
2. executing one synthetic extend/decode step for each startup bucket path

That means `/ready` includes first-touch kernel warmup, not just NEFF compilation.

This matters because:

- the full decode graph is large
- the first real decode step should not be the first time the graph is loaded/executed
- compile cache reuse is important for repeated profiling

Recommended knobs:

```bash
export NKIPY_SERVING_BUILD_DIR=/tmp/build_gpt_oss_full
```

If you want a cold compile measurement, point `NKIPY_SERVING_BUILD_DIR` at a fresh directory.

## Runtime Notes

- Internal decode compute buckets are TP-aligned and can be smaller than the scheduler-visible input token bucket.
- Full decode is now a true no-SP path, so small decode buckets such as `t8` can compile as real full-decode graphs instead of being normalized up to `t16`.
- Attention still pads decode Q/K/V up to 128 tokens when required by the NKI attention kernel.
- Attention-step inputs are prepared once per bucket and rewritten in place through the shared `PreparedNkiStepInputs` path.
- On TP+EP runs, `LogitsProcessor` kernels must load with the global communicator (`global_rank`, `tp_degree * ep_degree`) even though LM-head collectives are restricted to TP replica groups. Using TP-local world size on EP workers fails during warmup with an NRT NEFF communicator mismatch.
- Prefill still crosses a CPU MoE scheduling boundary per layer, but the device work on each side of that boundary is fused:
  - pre-router graph: `pre_attn + kv_update + attention + post_attn + router`
  - post-CPU graph: `output_init + blockwise_add_residual`

## Implemented Optimizations

The current GPT-OSS path includes the following model/runtime optimizations.

### Shared runtime improvements

- Worker startup warmup is unconditional. `/ready` now means:
  - startup buckets compiled
  - one synthetic first-touch extend/decode step executed for each startup bucket path
- GPT-OSS startup bucket progress logs are emitted on rank 0 with artifact paths for the compiled kernels.
- Startup does not compile unused fallback kernels for normal serving paths.
- The runtime fails fast if a required fused kernel is missing.

### Memory and buffer reuse

- GPT-OSS reuses per-bucket scratch instead of allocating `DeviceTensor`
  buffers per layer in the forward loop.
- Reused scratch covers:
  - Q/K/V
  - post-attention outputs
  - router scratch
  - MoE outputs
  - prefill mapping tensors
- Attention step inputs are cached per bucket through `PreparedNkiStepInputs`.
- Prefill and decode tile plans are rewritten in place instead of being rebuilt and reuploaded from fresh buffers on every step.

### Decode path optimizations

- Full decode graph (embed + all layers fused) with `decode_graph_scope=embed_layers`.
- Full decode is a true no-SP path.
- Small internal decode buckets such as `t8` are supported as real full-decode graphs on TP8.
- GPT-OSS embedding uses no-SP vocab-parallel embedding for full decode paths.
- Mixed request-bucket warmup/runtime for no-SP LM-head paths now uses per-`bs_bucket` scratch, so combinations such as `[16,32]` warm up and serve correctly.

### Attention preparation optimizations

- Decode attention metadata preparation has a decode-only fast path.
- Decode masks are cached and reused.
- Prefill tile-plan filling also has an in-place reusable host-buffer path.
- The shared prepared-input API is now used by GPT-OSS, Qwen3-dense, and Qwen3-MoE.

### Prefill path optimizations

- Prefill is partially fused per layer, while keeping CPU MoE scheduling in the middle:
  - pre-CPU fused device graph:
    `pre_attn + kv_update + attention + post_attn + router`
  - CPU MoE scheduling
  - post-CPU fused device graph:
    `output_init + blockwise_add_residual`
- Prefill attention step-prep uses reusable host-side plan filling.
- Blockwise MoE CPU scheduling uses a native C extension for
  `block_to_expert` / `token_position_to_id` construction rather than a pure
  Python loop path.
- The native `blockwise_index` helper is preloaded during worker warmup, so the
  first real MoE-prefill request does not pay the host extension build cost.

## Current Execution Shape

The execution path is:

```text
prefill/extend
  embed                separate
  prefill_pre_moe      fused per layer
  cpu moe scheduling   CPU
  prefill_post_moe     fused per layer
  lm_head              separate

decode
  full_decode          embed + all layers fused
  lm_head              separate no-SP graph
```

The full decode graph covers the decode path. Prefill is not a full-graph path because it still crosses a per-layer CPU MoE scheduling boundary.

## Profiling

Enable JSONL profiling with:

```bash
export NKIPY_SERVING_PROFILE=1
export NKIPY_SERVING_PROFILE_DIR=/tmp/nkipy_serving_profile_gpt_oss
```

Useful fields in `gpt_oss_model_rank*_steps.jsonl`:

- `token_bucket`: internal compute bucket
- `input_token_bucket`: scheduler-visible padded input bucket
- `attn_token_bucket`: attention-side bucket after any decode padding
- `decode_graph_scope`: active full-decode scope
- `t_attn_prep`: host/device attention-step preparation time
- `t_prefill_pre_moe_graph`: fused prefill device time before CPU MoE scheduling
- `t_prefill_post_moe_graph`: fused prefill device time after CPU MoE scheduling
- `t_total`: full model step time on that rank

## Benchmark Guidance

The most useful current serving benchmark setting for GPT-OSS is:

```text
tp_degree         = 8
ep_degree         = 16
request_buckets   = [32]
token_buckets     = [128,512,1024,2048,4096]
decode_graph_scope = embed_layers
attention_backend = NKIBlockSparseFlashAttention
```

### Best config now

Minimal benchmark-oriented config:

```json
{
  "model_id": "unsloth/gpt-oss-120b-BF16",
  "max_context_len": 4096,
  "tp_degree": 8,
  "ep_degree": 16,
  "attention_backend": "NKIBlockSparseFlashAttention",
  "request_buckets": [32],
  "token_buckets": [128, 512, 1024, 2048, 4096],
  "decode_graph_scope": "embed_layers",
  "nkipy_build_dir": "/tmp/build_gpt_oss_bench"
}
```

Repo config file:

```text
tests/runtime.tp8_ep16.gpt_oss.bench.json
```

Equivalent environment overrides:

```bash
export NEURON_LOGICAL_NC_CONFIG=1
export NKIPY_SERVING_REQUEST_BUCKETS=32
export NKIPY_SERVING_TOKEN_BUCKETS=128,512,1024,2048,4096
export NKIPY_SERVING_DECODE_GRAPH_SCOPE=embed_layers
export NKIPY_SERVING_BUILD_DIR=/tmp/build_gpt_oss_bench
```

Rationale from the current 200-prompt saturated ShareGPT sweep:

- `request_buckets=[32]` is the best tested setting for TTFT, E2E latency, request throughput, and output-token throughput.
- `[16,32]` is the only mixed request-bucket set that is still close enough to consider for more elastic serving.
- Including `8` in the benchmark request-bucket set hurts the saturated ShareGPT benchmark.
- Larger single request buckets such as `[64]`, `[96]`, and `[128]` were all worse than `[32]` on the tested workload.

This recommendation is for warm serving benchmarks after `/ready`. Cold compile/startup time should be measured separately with a fresh `NKIPY_SERVING_BUILD_DIR`.

### Latest rerun

Latest validated rerun on `2026-04-09`, using cached build and the config
above (post model-subpackage refactor, detokenizer process,
pre-tokenization, overlap scheduling):

```text
200-prompt ShareGPT, request_rate=inf
  request throughput      1.91 req/s
  output throughput     358.60 tok/s
  mean TTFT            2801.35 ms
  median TTFT          2399.54 ms
  mean ITL               54.58 ms
  median ITL             39.83 ms
  p99 ITL               264.63 ms
  mean E2E            13680.65 ms
  median E2E          10719.93 ms

server-side queue-free summary for the same run
  mean scheduled TTFT   237.55 ms
  median scheduled TTFT 177.52 ms
  p90 scheduled TTFT   544.73 ms
  mean queue delay    2546.93 ms
  median queue delay  2192.13 ms
```

Note: throughput is ~6% below the 2026-03-20 baseline (2.02 req/s, 380.89
tok/s). Overlap scheduling recovered most of the gap (no-overlap was ~12%
below). The remaining difference is in model execution time from the
model-subpackage refactor.

## Benchmark Methodology

The current GPT-OSS benchmark numbers in this repo are based on a ShareGPT
serving workload that intentionally mirrors the vLLM nightly ShareGPT serving
setup closely enough to make directional comparison meaningful.

### Dataset

- Dataset family: `sharegpt`
- Dataset file: `ShareGPT_V3_unfiltered_cleaned_split.json`
- Dataset source: the same ShareGPT JSON used by the local vLLM nightly and
  serving benchmark configs, for example:
  - `vllm-repo/.buildkite/scripts/run-benchmarks.sh`
  - `vllm-repo/examples/online_serving/prometheus_grafana/README.md`
- Sampling source: first two conversation turns from each ShareGPT record
  - turn 0 -> prompt
  - turn 1 -> expected completion length
- Tokenizer: the benchmark tokenizes prompt and completion with the model
  tokenizer for `unsloth/gpt-oss-120b-BF16`

Download command:

```bash
wget \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Example local path used in current reruns:

```text
/tmp/ShareGPT_V3_unfiltered_cleaned_split.json
```

### Filtering

The benchmark uses the same sequence filter as the vLLM ShareGPT serving
benchmark:

- `prompt_len >= 4`
- `output_len >= 4`
- `prompt_len <= 1024`
- `prompt_len + output_len <= 2048`

This is the filter implemented in `scripts/bench_serving.py --dataset-name sharegpt` and it matches
the default ShareGPT pruning logic in
`vllm/benchmarks/datasets.py:is_valid_sequence(...)`.

### Request Shape

The current main benchmark run uses:

- `num_prompts = 200`
- `request_rate = inf`
- endpoint: `/v1/completions`
- streaming enabled
- `temperature = 0.0`
- `max_tokens = expected_output_len`
- EOS is not forcibly ignored, so generation may still stop early on EOS

### Length Distribution

The current best-config 200-prompt run is:

- result file:
  `/tmp/gptoss_req_bucket_sweep_results/req32_inf_blockwise_native.json`

Its sampled length distribution is:

```text
prompt tokens
  min   4
  p50   106.5
  p90   627.4
  p95   723.5
  max   1013
  mean  219.7

output tokens
  min   4
  p50   157.0
  p90   515.2
  p95   601.3
  max   794
  mean  217.3

prompt + output tokens
  min   14
  p50   362.0
  p90   836.9
  p95  1001.1
  max  1577
  mean  437.0
```

### Metrics

The benchmark currently reports:

- `TTFT`: client-observed time to first streamed token
- `scheduled_ttft`: server-side time from first scheduler admission to first token
- `ITL`: client-observed inter-token latency between streamed token chunks
- `E2E`: client-observed request start to request finish
- `req/s`: completed requests per second
- `tok/s`: generated output tokens per second

For `/v1/chat/completions` benchmark runs, `scripts/bench_serving.py` now uses
the same chat-template prompt rendering path as the server for prompt-length
accounting, and it defaults to `max_completion_tokens` instead of `max_tokens`
unless the caller overrides either field explicitly.

These are request-level serving metrics. In a saturated run, `TTFT` includes
queueing delay as well as real service time. Model-only prefill timing should be
read from `gpt_oss_model_rank*_steps.jsonl`, especially the `extend` steps and
their `t_total`, `t_prefill_pre_moe_graph`, `t_prefill_post_moe_graph`, and
`t_attn_prep` fields.

For current ShareGPT serving reruns, `scheduled_ttft` is collected from
`http_events.jsonl` written by `NKIPY_SERVING_PROFILE=1`, using the
`request_completed` events emitted by the tokenizer manager.

### Benchmark Notes

- Use a fixed `NKIPY_SERVING_BUILD_DIR` and benchmark only after `/ready` if the
  goal is warm-serving TTFT / ITL / throughput.
- Measure cold compile/startup separately with a fresh build dir.
- The current best request-bucket recommendation is tuned for the saturated
  200-prompt ShareGPT run above. It is not automatically the best setting for
  low-concurrency or interactive traffic.

## Current Scope

- GPT-OSS uses a full decode graph for decode
- GPT-OSS prefill keeps CPU MoE scheduling per layer but fuses the surrounding device work
- Pure greedy decode stays on the TP-local top-1/top-k fast path
- Any non-greedy batch uses the shared exact device sampler: full-vocab TP all-gather followed by one raw NKI sampler kernel for `temperature`, `top_k`, `top_p`, and `min_p`
