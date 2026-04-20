# nkipy vLLM Plugin — Roadmap

Gap analysis comparing nkipy vLLM plugin against [private-vllm-neuron](../private-vllm-neuron/) (vllm-neuron), with a phased plan to reach feature parity.

## Feature Gap

| Feature | vllm-neuron | nkipy | Priority |
|---|---|---|---|
| Continuous Batching | ✅ Bucket-aware scheduler | ❌ batch_size=1, sequential | **Critical** |
| PagedAttention / KV Cache Mgmt | ✅ vLLM block manager | ❌ Internal KV cache | **Critical** |
| On-Device Sampling (top-k/p, temp) | ✅ Full sampler | ❌ Greedy only | **High** |
| Chunked / Segmented Prefill | ✅ Segmented CTE | ❌ | **High** |
| Speculative Decoding (Eagle3) | ✅ | ❌ | Medium |
| Quantization (MXFP4 / BF16) | ✅ | ❌ | Medium |
| Sliding Window Attention | ✅ | ❌ | Medium |
| Expert Parallelism (EP) | ✅ | ❌ (TP only for MoE) | Medium |
| Attention Data Parallelism (ADP) | ✅ | ❌ | Medium |
| Async Scheduling | ✅ (opt-in) | ❌ (disabled) | Low |
| Disaggregated Inference (NiXL) | ✅ | ❌ (has P2P, different) | Low |
| Multi-LoRA | ❌ (TODO) | ❌ | Neither |
| Prefix Caching | ❌ (TODO) | ❌ | Neither |
| P2P Weight Transfer / Sleep-Wake | ❌ | ✅ | **nkipy advantage** |

## Phase 1: Serving Fundamentals (Weeks 1–4)

Blockers for production use — without these, throughput is 1 request at a time.

### 1.1 Continuous Batching

- Replace the sequential generator pattern in `model_runner.py` with batched prefill/decode.
- Add batch dimension to CTE/TKG kernels (currently `max_batch_size=1`).
- Implement bucket-aware admission control (reference: vllm-neuron `NeuronScheduler`).
- Separate prefill and decode phases with a state machine (IDLE → PREFILL → DECODE).

### 1.2 vLLM-Managed KV Cache

- Replace internal KV cache with vLLM's block manager (`initialize_from_config` is currently a no-op).
- Map vLLM `KVCacheConfig` blocks to DeviceTensor allocations.
- Wire `SchedulerOutput.scheduled_cached_reqs` for KV reuse.

### 1.3 Sampling Parameters

- Wire vLLM `SamplingParams` (temperature, top_k, top_p) through to the kernel.
- Replace `greedy_sampling` with a configurable on-device sampler.
- Minimum: greedy, top-k, top-p, temperature scaling.

## Phase 2: Prefill Efficiency (Weeks 5–8)

Needed for long-context workloads and mixed prefill/decode batches.

### 2.1 Chunked Prefill

- Split long prompts into segments that fit in compiled bucket sizes.
- Implement segmented CTE attention kernel (reference: vllm-neuron `attention_segmented_cte.py`).
- Enable mixing prefill and decode in the same scheduling cycle.

### 2.2 Sliding Window Attention

- Add `sliding_window` parameter to attention kernels.
- Apply per-layer (e.g., even layers only, as in Qwen3).
- Reduces KV cache memory for long contexts.

## Phase 3: Performance (Weeks 9–12)

Throughput and latency optimizations.

### 3.1 Expert Parallelism (EP)

- Currently MoE uses TP only — each rank has all experts but sharded.
- Distribute experts across ranks, route tokens to the correct rank.
- Reduces per-rank memory for large MoE models (e.g., Qwen3-30B-A3B with 128 experts).

### 3.2 Attention Data Parallelism (ADP)

- Replicate attention across a subset of ranks for higher decode throughput.
- Complementary to TP — ADP for attention, TP for FFN/MoE.

### 3.3 Quantization (MXFP4)

- Add MXFP4 weight loading and dequantization.
- Modify FFN/attention kernels to accept quantized inputs.
- ~2× memory reduction, enables larger models per instance.

## Phase 4: Advanced (Weeks 13+)

### 4.1 Speculative Decoding (Eagle3)

- Implement Eagle3 draft model (smaller Llama variant).
- Add rejection sampler kernel.
- Prerequisite: continuous batching (Phase 1).

### 4.2 Async Scheduling

- Overlap scheduling with model execution.
- Requires on-device sampling (Phase 1) so the scheduler doesn't wait for CPU-side token IDs.

### 4.3 FX Graph Compilation Backend

- Current approach: manual HLO/NKI trace per kernel.
- Alternative: FX graph capture like vllm-neuron, auto-compile via neuroncc.
- Enables supporting new model architectures without hand-writing kernels.

## Priority Order

```
Phase 1 (Critical)     → Continuous Batching → KV Cache → Sampling
Phase 2 (High)         → Chunked Prefill → Sliding Window
Phase 3 (Performance)  → EP → ADP → Quantization
Phase 4 (Advanced)     → Spec Decode → Async → FX Backend
```

Phase 1 is the minimum viable gap to close. Everything else builds on continuous batching.
