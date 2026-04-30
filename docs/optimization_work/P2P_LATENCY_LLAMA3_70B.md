# Token Embedding TP Sharding

## Problem

LLaMA-3-70B's token embedding table is `[128256, 8192]` in bfloat16 = **2.1 GB**. Before this change, every TP rank held a full copy:

- **Device memory**: 2.1 GB per rank duplicated across 32 ranks = 67.2 GB total HBM wasted on redundant copies.
- **Checkpoint disk**: Each of 32 shard files stored the full 2.1 GB embedding, inflating total checkpoint from 149 GB (unique data) to 214 GB.
- **RDMA transfer**: The 2.1 GB buffer exceeded the EFA `ibv_reg_mr` registration limit (~1 GB), requiring a separate HTTP fallback path for tok_embedding during P2P weight transfer.

## Solution: Hidden-Dimension Parallel Sharding

Shard tok_embedding along the **hidden dimension** (columns), not the vocab dimension (rows). Each TP rank stores `tok_embedding[:, rank*cols:(rank+1)*cols]` where `cols = hidden_size // tp`.

For LLaMA-3-70B with TP=32: `[128256, 8192]` -> `[128256, 256]` per rank = **65.7 MB**.

### Embedding Lookup

```python
def _tp_embedding_lookup(shard, input_ids):
    partial = shard[input_ids.long()]        # [*, hidden_per_rank]
    chunks = [torch.empty_like(partial) for _ in range(world_size)]
    dist.all_gather(chunks, partial.contiguous())
    return torch.cat(chunks, dim=-1)         # [*, hidden_size]
```

Every rank does a full-vocab local lookup producing a partial hidden vector. `all_gather` + `cat` reconstructs the full hidden dimension.

### Why All-Gather over All-Reduce

Vocab-dim sharding (the alternative) requires masking out-of-range tokens, zeroing non-owned entries, and `all_reduce(SUM)` to combine. Hidden-dim sharding avoids all of that:

| Aspect | Vocab-dim (all_reduce) | Hidden-dim (all_gather) |
|--------|----------------------|------------------------|
| Local lookup | Masked indexing + zero fill | Direct indexing |
| Collective | `all_reduce(SUM)` | `all_gather` + `cat` |
| Arithmetic | Reduction across ranks | None (just concatenation) |
| Correctness | Requires careful masking for edge ranks | Always correct |

Both move the same number of bytes over the network, but all_gather skips the reduction arithmetic.

### Checkpoint Resharding

A resharding script (`scripts/reshard_tok_embedding.py`) replaces the full embedding in each shard file with only that rank's column slice:

```
python scripts/reshard_tok_embedding.py /path/to/checkpoint --tp 32
```

The model loading code auto-detects whether tok_embedding is already sharded (by checking `shape[1] == hidden_size // tp`) and skips re-slicing if so. This means both pre-sharded and legacy (full) checkpoints work.

## Results

### LLaMA-3-70B, TP=32, trn1.32xlarge, 2 instances

#### Checkpoint and Disk

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| tok_embedding per rank | 2,101 MB | 65.7 MB | 32x smaller |
| Per-shard file size | 6.7 GB | 4.4 GB | 34% smaller |
| Total checkpoint disk | 214 GB | 139 GB | 75 GB saved (35%) |

#### Engine Startup (Cold Start from Checkpoint)

| Metric | Value |
|--------|-------|
| Engine A cold start (NEFF + checkpoint page-cached) | **32s** |
| Engine A cold start (page cache dropped) | **86s** |
| Engine B (sleep mode, no checkpoint) | **12s** |

The 32s cold start requires both NEFF kernels and checkpoint files to be in the OS page cache. When the page cache is cold (e.g. after `echo 3 > /proc/sys/vm/drop_caches`), the same startup takes ~86s because 32 workers must re-read 149 GB of checkpoint data plus 1 GB of NEFF cache from disk concurrently. Without any NEFF cache on disk (first-ever start), compilation adds ~10 minutes.

#### Checkpoint Disk-to-CPU Loading (Microbenchmark)

| Scenario | Wall Clock | Per-Rank | Throughput |
|----------|-----------|----------|------------|
| 1 shard (4.4 GB), cold | 0.22s | 0.22s | 21.7 GB/s |
| 1 shard (4.4 GB), warm | 0.20s | 0.20s | 23.2 GB/s |
| 32 shards concurrent (149 GB), cold | 44.3s | avg 39.2s, max 40.5s | 3.4 GB/s |
| 32 shards concurrent (149 GB), warm | 43.1s | avg 37.7s, max 40.5s | 3.5 GB/s |

A single shard reads at 21.7 GB/s (near NVMe sequential limit), but 32-way concurrency reduces aggregate throughput to ~3.4 GB/s. Cold vs warm makes little difference because 149 GB far exceeds available page cache.

The microbenchmark (44s) is slower than the actual engine cold start (86s includes all startup overhead, not just checkpoint I/O) because the engine uses **mmap + DMA pipelining**: `safetensors.load_file()` returns mmap-backed tensors without copying data to user space, and `DeviceTensor.from_torch()` reads each tensor directly from the mmap region and DMA-transfers it to device memory. Disk reads overlap with DMA transfers, so checkpoint loading is faster than the microbenchmark that forces all data into user-space RAM with `.clone()`.

#### P2P Wake-Up Latency (Engine B activated from Engine A)

**Before `NEURON_RT_RESET_CORES=0`:**

| Phase | Cycle 1 | Cycle 2 | Cycle 3 (non-blocking) |
|-------|---------|---------|----------------------|
| Gloo init | 0.95s | 0.70s | - |
| NRT init | 5.78s | 14.29s | - |
| Alloc tensors | 0.06s | - | - |
| P2P transfer | 10.29s | 10.17s | 10.48s |
| Kernel load | 0.11s | 0.16s | - |
| tok_embedding | 0.11s | - | - |
| **Total wake-up** | **28.9s** | **27.7s** | **30.6s** |

**After `NEURON_RT_RESET_CORES=0` + sender MR pre-registration** (see [NRT_INIT_SKIP_RESET.md](NRT_INIT_SKIP_RESET.md)):

| Phase | Cycle 1 (cold) | Cycle 2+ (optimized) |
|-------|---------------|---------------------|
| Gloo init | 0.95s | 0.94s |
| NRT init | 15.56s | **0.17s** |
| P2P transfer | 7.75s | **5.18s** |
| Kernel load | 0.09s | 0.07s |
| **Total wake-up** | **30.8s** | **7.1s** |

Subsequent wake-ups are **~4× faster**: NRT init drops from 5–15s to 0.17s (skip NC reset), P2P transfer drops from 7.7s to 5.2s (sender MRs pre-registered at model load).

P2P transfer volume: ~139 GB across 32 ranks (4.35 GB/rank), completing in ~10s = **~14 GB/s aggregate RDMA throughput**.

#### Sleep Latency

**Immediate sleep** (called shortly after wake-up, dereg still in progress):

| Cycle | dereg_wait | spike_reset | gloo_destroy | Total |
|-------|-----------|-------------|-------------|-------|
| 1 | 31.5s | 0.13s | 0.78s | 33.2s |
| 2 | 67.7s | 0.13s | 1.24s | 69.5s |
| 3 | 55.7s | 0.13s | 0.91s | 57.2s |

When `/sleep` is called before RDMA deregistration finishes (`dereg_waited: True`), the engine blocks 31–68s waiting for it.

**Deferred sleep** (called 120s+ after wake-up, dereg already finished):

| Cycle | dereg_wait | spike_reset | gloo_destroy | gc_collect | Total |
|-------|-----------|-------------|-------------|-----------|-------|
| 1 | 0.0s | 0.15s | 1.35s | 0.36s | **1.87s** |
| 2 | 0.0s | 0.12s | 1.52s | 0.31s | **1.96s** |
| 3 | 0.0s | 0.12s | 1.45s | 0.31s | **1.88s** |

When RDMA deregistration has already completed (`dereg_waited: False`), sleep takes **~1.9s**. The remaining cost is gloo_destroy (~1.4s), spike_reset (~0.13s), and GC (~0.33s).

#### Non-Blocking Push (Inference on A During Transfer)

| Metric | Baseline (pre-push) | During/After Push |
|--------|-------------------|-------------------|
| Avg latency | 1,585 ms | 1,626 ms |
| Min latency | 1,525 ms | 1,515 ms |
| Max latency | 1,677 ms | 1,702 ms |
| Stalls | 0 | 0 |

**Zero inference stalls on Engine A during P2P transfer to Engine B.** The background RDMA push does not interfere with serving.

#### Inference Correctness

All outputs match exactly across Engine A, Engine B (all 3 wake cycles):

| Prompt | Output |
|--------|--------|
| "The capital of France is" | " Paris. It is the largest city in France. It is the capital of the Île-de-F" |
| "Albert Einstein was born in" | " Ulm, Germany on March 14, 1879. He was the first child of the famous physicist..." |

## Files Changed

- `nkipy/src/nkipy/vllm_plugin/models/llama.py` — `_tp_embedding_lookup`, `_shard_tok_embedding`, `embed_tokens`, `_allocate_empty_tensors`
- `nkipy/src/nkipy/vllm_plugin/models/qwen3.py` — Same changes as llama.py
- `nkipy/src/nkipy/vllm_plugin/model_runner.py` — Uses `model.embed_tokens()` instead of direct `tok_embedding[input_ids]`
- `nkipy/src/nkipy/vllm_plugin/worker.py` — Wake-up converts sharded DeviceTensor to CPU tensor
- `nkipy/src/nkipy/vllm_plugin/server.py` — Updated pre-cache size calculation for sharded embedding
- `scripts/reshard_tok_embedding.py` — New: reshards checkpoint tok_embedding for hidden-dim TP

## Date

2026-04-22
