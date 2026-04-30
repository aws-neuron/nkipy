# P2P RDMA Sleep/Wake Optimization

Documentation for the P2P weight transfer optimization work: reducing `/sleep` latency from 46.5s to ~1s, fixing wake-up correctness, and validating scalability.

## Architecture & Design

- **[ARCHITECTURE_CONSTRAINTS.md](ARCHITECTURE_CONSTRAINTS.md)** — MR registration strategy for server vs receiver, why receiver must call `dereg_async()`
- **[SPIKE_RESET_MR_ANALYSIS.md](SPIKE_RESET_MR_ANALYSIS.md)** — Root cause analysis: active RDMA MRs cause 60x `spike_reset()` slowdown; three controlled tests proving it
- **[observations_and_ideas.md](observations_and_ideas.md)** — Original design tradeoff analysis (naive vs optimized vs alternative MR strategies)

## Key Fixes

- **[WAKE_FROM_CHECKPOINT_FIX.md](WAKE_FROM_CHECKPOINT_FIX.md)** — Fix for garbage output after sleep/wake: must write into existing tensors, not recreate them
- **[CTRL_C_CLEANUP_FIX.md](CTRL_C_CLEANUP_FIX.md)** — Fix for RDMA resource leakage on Ctrl+C; cleanup ordering (RDMA before spike_reset)

## Performance Results

- **[NRT_INIT_SKIP_RESET.md](NRT_INIT_SKIP_RESET.md)** — Skip NC firmware reset (`NEURON_RT_RESET_CORES=0`, nrt_init 5–15s → 0.15s) + sender MR pre-registration (P2P 7.7s → 5.2s) + overlapped sender connect (P2P 5.2s → 3.25s). Total wake-up 30s → 5.0s
- **[P2P_LATENCY_LLAMA3_70B.md](P2P_LATENCY_LLAMA3_70B.md)** — LLaMA-3-70B benchmarks: tok_embedding TP sharding (2.1 GB → 65.7 MB/rank), wake/sleep/inference latency, non-blocking push
- **[SCALABILITY_TEST_QWEN3.md](SCALABILITY_TEST_QWEN3.md)** — Qwen3-30B TP=32: 50 standby engines, resource costs, dynamic lifecycle, Gloo teardown fix
- **[SCALABILITY_TEST_TINYLLAMA.md](SCALABILITY_TEST_TINYLLAMA.md)** — TinyLlama TP=8: 100 engines, bidirectional P2P, context_len bug fix

## Roadmap

- **[ROADMAP.md](ROADMAP.md)** — Feature gap analysis vs vllm-neuron; phased plan for continuous batching, KV cache, sampling, etc.

## Summary of Results

| Metric | Before | After |
|--------|--------|-------|
| `/sleep` (deferred, >60s after wake) | 46.5s | ~2s |
| `/sleep` (immediate, <60s after wake) | 46.5s | ~63-65s (waits for MR dereg) |
| `/wake_up` (P2P, LLaMA-3-70B TP=32, cold) | — | ~22s |
| `/wake_up` (P2P, TP=32, optimized) | ~28-30s | ~5.0s |
| `/wake_up` (P2P, TP=32, 10-engine avg) | — | 5.02s (warm), 21.6s (cold) |
| `nrt_init` (subsequent, `reset_cores=0`) | 5-15s | 0.15-0.18s |
| P2P transfer (all optimizations) | 7.7s | 3.21-3.25s |
| `spike_reset` (with active MRs) | 7-25s | avoided |
| `spike_reset` (MRs deregistered) | — | 0.12-0.15s |
| Standby engines per trn1 (TP=32) | ~58 (TCP limit) | ~110 (memory limit) |
| Checkpoint size (LLaMA-3-70B) | 214 GB | 139 GB (35% smaller) |
| Non-blocking push stalls | — | 0 |
