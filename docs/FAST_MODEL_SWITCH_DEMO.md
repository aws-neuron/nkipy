# Fast On-demand Model Serving Scaling Up on Neuron

## 1. Motivation

### Over-Provision or Cold-Start Latency When Scaling Up LLM in Multi-Model Serving

Production LLM serving increasingly requires multiple models on shared cluster. Traffic patterns are bursty — a model may see 10x load during peak and near-zero at off-peak. On Trainium instances, operators face a choice: keep every model always-loaded (expensive) or load models on-demand (slow).

The **always-loaded** approach over-provisions instances to each model, keeping it running and ready to serve at all times. This eliminates cold-start latency entirely and requests are served immediately. However, in practice, most models sit idle most of the time — a fleet serving 10 models may see only 1–2 active at any moment, yet pays for all 10 continuously. This makes always-loaded impractical as the model catalog grows.

The **on-demand** path avoids this cost by loading models only when traffic arrives, but introduces a **cold start** bottleneck. When a request arrives for a model that isn't currently loaded, the system must initialize an engine and load weights before serving the first token. On Neuron, this cold-start latency has several components:

### Cold-start Latency Breakdown on Neuron

We profiled cold-start latency across two trn1.32xlarge instances (each with 32 NeuronCores, 4 instance store NVMe drives, 8 EFA NICs) running LLaMA-3-70B with TP=32 under the vLLM serving framework. The model checkpoint size is 139 GB.

| Phase | Latency | Bottleneck |
|-------|---------|------------|
| Prior model release | ~65s | RDMA MR deregistration (~63s) + spike_reset + Gloo destroy (measured) |
| Python imports + framework init + worker spawn | ~22s | vLLM, PyTorch, plugin loading, fork 32 worker processes (measured) |
| Gloo distributed init | ~1s | Establish Gloo mesh across 32 ranks (measured: ~1s without CPU contention) |
| NeuronCore runtime init | 5–20s | Firmware reset, device allocation via NRT (measured: 5–20s first init) |
| Weight loading from local NVMe | ~35s | Instance store NVMe aggregate ~4 GB/s for 139 GB (measured: ~1 GB/s per drive × 4 drives) |
| Weight loading from FSx for Lustre | 70–140s | Shared bandwidth (~1 GB/s/TiB); degrades under concurrent access |
| Weight loading from S3 | 105–175s | Download to local disk (~2–3 GB/s, 50–70s) + load from disk (~35s); sequential |
| NEFF kernel compilation | 5–30min | Neuron compiler translates model graph to NeuronCore binaries (first run only) |
| NEFF kernel load + warmup | ~18s | Load pre-compiled NEFF, profile, allocate KV cache (measured) |
| **Total cold start (local NVMe)** | **~141s** | |
| **Total cold start (FSx)** | **176–246s** | |
| **Total cold start (S3)** | **211–301s** | |

Weight loading dominates: reading 139 GB of Neuron-compiled checkpoint from disk is bounded by storage bandwidth. trn1.32xlarge has 4 instance store NVMe drives (~1 GB/s each, ~4 GB/s aggregate as measured by `dd iflag=direct`), giving a best-case load time of ~35s. In practice, checkpoint format overhead and 32-worker contention push this to 44s+. From networked storage (FSx/S3), it can exceed 2–3 minutes. Engine startup overhead (imports, worker spawn, Gloo init) adds another 15–23s before weight loading even begins.

The operational challenge: **How do we scale up LLM model serving on demand with second-level cold-start latency?**

---

## 2. System Design

### 2.1 Design Principles

We reduce cold-start latency from minutes to seconds through three key design principles:

**1. Decouple engine initialization from weight loading and compilation.** A traditional cold start serializes everything: process spawn → NRT init → compile → load weights → serve. We separate these concerns so that CPU-heavy work (Python imports, framework init, NEFF compilation) happens once at deployment time, not on the critical path of each scale-up event.

**2. Standby engine pool with shared hardware.** Each instance hosts 100+ standby engines of different models sharing the same NeuronCores, with only one active at a time. This enables fast model switch among hundreds of models on a single instance. Every instance in the cluster independently initializes its own standby pool, so the number of instances that can serve any given model scales linearly with cluster size.

**3. P2P RDMA weight transfer.** Instead of reading weights from disk (bounded by NVMe at ~4 GB/s) or network storage (FSx/S3), we push weights directly from an active engine's device memory over EFA (e.g., 800 Gbps on trn1.32xlarge). This reduces weight loading from 35–175s to ~2s and eliminates the requirement for a local checkpoint entirely.

**Expected result**: Cold start drops from **3–5 minutes** to **< 10 seconds** for LLaMA-3-70B at TP=32.

### 2.2 System Overview

Each instance in the cluster runs a **standby engine pool** — 100+ pre-initialized engines of different models sharing the same NeuronCores. Only one engine is active at a time; others sleep in a minimal state (zero device resources). An external scheduler orchestrates sleep/wake transitions via HTTP endpoints.


```
Traditional cold start (serialized):

┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌───────┐
│  Prior   │ │ Imports  │ │ NRT init │ │   NEFF   │ │ Weight load  │ │       │
│  model   │→│ + spawn  │→│ + Gloo   │→│ compile  │→│ (from disk)  │→│ Serve │
│ release  │ │  ~22s    │ │  ~6–21s  │ │ 5–30min  │ │   35–175s    │ │       │
│  ~65s    │ │          │ │          │ │          │ │              │ │       │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘ └───────┘
                              Total: 3–5 minutes

Our approach (decoupled):

┌──────────────────────────────────────────────────────────────────────────┐
│ Deploy time: standby engine pool (per instance)                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────┐                  │
│  │ Model A (active — serving)                         │                  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │                  │
│  │  │ Imports  │→ │ NRT init │→ │   NEFF   │→ serve   │                  │
│  │  │ + spawn  │  │          │  │ compile  │          │                  │
│  │  └──────────┘  └──────────┘  └──────────┘          │                  │
│  └────────────────────────────────────────────────────┘                  │
│  ┌────────────────────────────────────────────────────┐                  │
│  │ Model B (sleeping)                                 │ + 100+ sleeping  │
│  │  ┌──────────┐  ┌──────────┐                        │   models         │
│  │  │ Imports  │→ │   NEFF   │→ sleep                 │                  │
│  │  │ + spawn  │  │ compile  │                        │                  │
│  │  └──────────┘  └──────────┘                        │                  │
│  └────────────────────────────────────────────────────┘                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                              ║
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ On-demand wake_up: scale up Model B (critical path)                      │
│                                                                          │
│  Model A (sleeping):                                                     │
│  ┌─────────────┐  ┌─────────────┐                                        │
│  │ Release NRT │→ │    Sleep    │                                        │
│  │             │  │     ~2s     │                                        │
│  └─────────────┘  └─────────────┘                                        │
│                                                                          │
│  Model B (activating):                                                   │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────┐  │
│  │ Wait sleep │→ │ NRT init │→ │   Gloo   │→ │  P2P weight  │→ │Serve │  │
│  │    ~2s     │  │  ~0.2s   │  │   init   │  │ transfer ~2s │  │      │  │
│  │            │  │          │  │   ~1s    │  └──────────────┘  └──────┘  │
│  └────────────┘  └──────────┘  └──────────┘                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                        Total: ~5s
```


We assume at least one engine of a given model is always active for serving. When scaling up that model, the active **sender engine** pushes weights directly into the standby **receiver engine's** device memory via per-rank RDMA writes over EFA. This is possible because model weights remain unchanged during inference — the sender's device memory always holds a valid, up-to-date copy that can be read at any time without coordination. This bypasses disk I/O entirely and allows the sender engine to continue serving requests during the transfer.


```
                          ┌───────────┐
                          │   Users   │
                          └───────────┘
                     requests │   ▲ generations
                              ▼   │
┌────────────────────────────────────────────────────────────────┐
│                    LLM Serving Scheduler                       │
│                                                                │
│  ┌─────────────┐    ┌───────────────┐    ┌──────────────┐      │
│  │   Request   │───▶│ Load Monitor  │───▶│ Auto-scaler  │      │
│  │   Router    │    │               │    │ /wake_up     │      │
│  │             │◀───│ (per-model    │    │ /sleep       │      │
│  └──────┬──────┘    │  metrics)     │    └──────┬───────┘      │
│         │           └───────────────┘           │              │
└─────────┼───────────────────────────────────────┼──────────────┘
          │  /v1/completions                      │  /wake_up, /sleep
          │                                       │  /v1/completions
          ▼                                       ▼
┌────────────────────────┐    RDMA WRITE    ┌────────────────────────┐
│   Instance A           │═════════════════▶│   Instance B           │
│                        │   EFA 800 Gbps   │                        │
│  Active engine:        │                  │  Standby engine:       │
│  Model B (serving)     │  Per-rank push   │  Model B (waking)      │
│  ┌──────────────────┐  │                  │  ┌──────────────────┐  │
│  │ NeuronCore 0    │──┼────────────────────┼─▶│ NeuronCore 0    │  │
│  │ NeuronCore 1    │──┼────────────────────┼─▶│ NeuronCore 1    │  │
│  │     ...         │  │                    │  │     ...         │  │
│  │ NeuronCore 31   │──┼────────────────────┼─▶│ NeuronCore 31   │  │
│  └──────────────────┘  │                  │  └──────────────────┘  │
│                        │                  │                        │
│  Standby pool:         │                  │  Standby pool:         │
│  100+ sleeping engines │                  │  100+ sleeping engines │
└────────────────────────┘                  └────────────────────────┘
```

### 2.3 P2P Weight Transfer: Data Plane

The data plane uses a custom RDMA library ("Relay") built on AWS EFA. Each TP rank on both sender and receiver has its own RDMA endpoint with up to 8 NIC contexts for parallelism. The sender pushes weights from device memory while continuing to serve inference — no downtime on the sender side.

**Transfer protocol (receiver-initiated):**

```
Receiver (standby engine waking up)               Sender (active engine)
───────────────────────────────────               ──────────────────────
1. POST /p2p_preconnect
   (endpoint metadata)  ─────────────────────────▶ Start RDMA QP connect (~2s)
                                                       │
2. Register MRs (~2.3s)                               │ (overlapped)
   ┌─────────────────────┐                            │
   │ ibv_reg_mr() x 435  │                            ▼
   │ buffers in chunks    │                       Connection established
   └─────────────────────┘                            │
                                                       │
3. POST /p2p_push_weights                             │
   (transfer descriptors) ────────────────────────────▶
                                                       │
                                                  4. RDMA WRITE (~0.6s)
                          ◀════════════════════════════╡ (one-sided, no CPU
                            Direct memory write         │  involvement on receiver)
                                                       │
5. Acknowledge writes                             5. Done (background thread)
   (read-back forces                                   │
    NRT cache coherence)                               │  Sender continues serving
                                                       ▼  inference throughout
6. Load cached NEFF kernels                       Return success
```

**Key design decisions:**

- **One-sided RDMA WRITE**: The sender writes directly into the receiver's registered memory regions. No receiver CPU involvement during transfer — the receiver just waits for completion.
- **Chunked MR registration**: EFA has per-device MR limits. We register buffers in chunks of 1024 to stay within limits while keeping registration fast.
- **Overlapped connect + register**: The sender's 2.0s QP establishment runs in parallel with the receiver's 2.3s MR registration (via async HTTP preconnect). Critical path = max(2.0, 2.3) + 0.6s = ~2.9s.
- **Non-blocking push**: The sender runs RDMA writes in a background thread per worker. The sender's inference loop is never blocked — zero stalls during transfer.
- **Deferred resource cleanup**: After a push, MR deregistration and endpoint reset run asynchronously in the background (~63s), completely off the critical path.

### 2.4 Control Plane: Sleep/Wake Lifecycle

Each engine exposes HTTP endpoints for lifecycle management:

| Endpoint | Action |
|----------|--------|
| `POST /nkipy/wake_up` | Allocate tensors, receive weights via P2P, reload kernels |
| `POST /nkipy/sleep` | Deregister MRs, release NeuronCores, destroy Gloo |
| `GET /nkipy/health` | Returns `{sleeping: true/false, ...}` |
| `POST /nkipy/p2p_preconnect` | Early RDMA connection establishment |
| `POST /nkipy/p2p_push_weights` | Trigger weight push from active engine |

**Sleep state**: A sleeping engine holds ~2.8 GB of host memory (Python process + framework state) but **zero** device resources — NeuronCores are fully released and available for other engines.

**Wake sequence (Qwen3-30B-A3B TP=32, measured from 100-engine test):**

| Step | Qwen3-30B | LLaMA-3-70B | Description |
|------|-----------|-------------|-------------|
| Gloo init | 0.92s | 0.92s | Recreate distributed process group |
| NRT init | 0.15s | 0.15s | NeuronCore runtime (skip firmware reset) |
| Alloc tensors | 0.05s | 0.05s | Empty device memory for model weights |
| P2P transfer | 3.23s | 4.95s | RDMA receive (1977 MB / 4350 MB per rank) |
| RDMA ack | 0.63s | 0.67s | Read-back for NRT cache coherence |
| Kernel load | 0.08s | 0.08s | Reload cached NEFF binaries from disk |
| **Total** | **5.1s** | **7.0s** | |

**Standby engine pool**: Multiple engines share the same NeuronCores. Only one is active at a time; others sleep. An external scheduler calls `/sleep` on the current engine and `/wake_up` on the next one.

### 2.5 Token Embedding: Sharding Optimization

LLaMA-3-70B's token embedding is 2.1 GB — too large for a single RDMA MR registration (EFA limit ~1 GB). Rather than using an HTTP fallback, we shard the embedding along the hidden dimension:

- **Before**: `[128256, 8192]` full copy per rank = 2.1 GB x 32 = 67 GB redundant
- **After**: `[128256, 256]` per rank = 65.7 MB — fits in a single MR, transfers via RDMA

This also reduced checkpoint disk from 214 GB to 139 GB (35% savings).

---

## 3. Implementation Challenges

### 3.1 Active RDMA MRs Cause 60x `spike_reset()` Slowdown

**Discovery**: Sleep latency was 46.5s instead of the expected ~1s. Root cause: the NeuronCore firmware reset (`spike_reset()`) takes 7–25s when RDMA memory regions are still registered, vs 0.12s when MRs are deregistered first.

**Fix**: Deregister all MRs asynchronously immediately after transfer completes. Sleep path waits for deregistration if still in progress. If sleep is called >60s after wake (typical in production), MR deregistration has already finished and sleep takes ~2s.

### 3.2 NRT Init Firmware Reset Dominates Wake Latency (5–15s)

**Discovery**: `nrt_init()` reboots NeuronCore firmware on every wake cycle, even though a clean `nrt_close()` leaves cores in a known-good state.

**Fix**: Coordinate via a marker file (`/tmp/nkipy_nrt_closed`). After any clean shutdown, write the marker. On next init, check for the marker and set `NEURON_RT_RESET_CORES=0` to skip the firmware reboot. Cold starts (no marker) still get a full reset. Result: `nrt_init` drops from 5–15s to **0.15s**.

**Safety**: HBM scrub (zero-fill) still runs. Cross-model correctness verified — cores that ran LLaMA can immediately run Qwen3 without firmware reset.

### 3.3 TCP Port Exhaustion at Scale

**Discovery**: Each standby engine's Gloo distributed backend consumed ~482 TCP ephemeral ports. At 30 engines on one instance, all 28,232 ports were exhausted.

**Fix**: Destroy Gloo process groups during sleep, recreate during wake. Sleeping engines hold ~1 TCP port (the HTTP server) instead of 482. This raised the scalability limit from ~58 engines (TCP-bound) to **~110 engines** (memory-bound at 2.8 GB each).

### 3.4 CPU Spinning in Idle Workers

**Discovery**: vLLM's SHM message queue polling loop busy-spins at 99% CPU per worker. With 29 engines x 32 workers = 928 spinning threads, new engine startup slowed from 15s to 185s due to scheduling contention.

**Fix**: Enable `VLLM_SLEEP_WHEN_IDLE=1` — after 3s of inactivity, workers call `time.sleep(0.1)` instead of `sched_yield()`. Startup time stays constant at ~15s regardless of standby pool size.

### 3.5 Stale Tensor References After Wake

**Discovery**: After sleep/wake, model output was garbage. The model's internal tensor references pointed to deallocated memory from the previous wake cycle.

**Fix**: During wake, write new weights into the existing allocated tensors rather than creating new tensor objects. The model graph retains valid references to the same memory.

### 3.6 RDMA Resource Leak on Ctrl+C

**Discovery**: Killing an engine with Ctrl+C left RDMA resources registered, causing the next `spike_reset()` on the same cores to take 25s+ (same root cause as 3.1).

**Fix**: Register cleanup handlers that deregister RDMA MRs **before** calling `spike_reset()`. Ordering is critical — RDMA cleanup must precede firmware reset.

### 3.7 Sender RDMA QP Exhaustion After Repeated Pushes

**Discovery**: The sender crashed with `Assertion '*qp' failed` in `RDMAChannelImpl::initQP()` after ~11 consecutive pushes. Each push created a new RDMA Queue Pair connection to the receiver, but connections were never freed — they accumulated until the EFA device's QP pool was exhausted.

**Root cause**: With pre-registered MRs, `push_to_peer()` kept the sender's `RankEndpoint` alive across pushes (to avoid 2.3s re-registration). But the endpoint accumulated stale QP connections from all previous receivers. The relay library has no per-connection `disconnect()` API.

**Fix**: After each push completes, call `reset_endpoint_async()` which destroys the old endpoint (freeing all QPs) and re-registers MRs on a fresh endpoint in a background thread. The 54s re-registration runs concurrently with the receiver's sleep/next-wake cycle. The next `preconnect_to_peer()` call blocks on `_ensure_endpoint()` until the background reset finishes — if less than 60s have passed since the last push, the preconnect waits; in production (>60s between pushes), no waiting occurs.

### 3.8 Partial NRT Init Leaves NeuronCores in Split-Brain State

**Discovery**: With 32 TP workers, some ranks succeed at NRT init while others fail (e.g., due to stale neuron driver state). Successful ranks hold NeuronCores, failed ranks report error. The engine returns an error response but the successful ranks' cores are now orphaned — no subsequent engine can claim them.

**Fix**: After NRT init, all ranks participate in a collective check via `dist.all_gather_object()`. If ANY rank failed, the successful ranks explicitly release their cores (`spike_reset()`) and the whole engine returns to sleep state. This ensures NeuronCores are never left in a partial-claim state.

### 3.9 Gloo TCP Thundering Herd at 50+ Engines

**Discovery**: Starting 50+ standby engines simultaneously caused Gloo distributed backend initialization to fail with TCP connection errors. All engines start their TP=32 Gloo groups concurrently, creating thousands of simultaneous TCP connections.

**Fix**: Two-part mitigation:
1. **Stagger engine launches** by 2s in the orchestration script (eliminates startup failures entirely)
2. **Retry logic in `_init_distributed()`** with exponential backoff (3 retries, 1s/2s/4s). If Gloo init fails, partial state is destroyed and retried. This handles transient connection failures during wake-up even when multiple engines wake near-simultaneously.

### 3.10 Concurrent Wake/Sleep Requests Cause Race Conditions

**Discovery**: If a scheduler sends `/wake_up` while a previous `/sleep` is still processing (MR deregistration takes 60s), both operations run simultaneously and corrupt internal state.

**Fix**: A `_nkipy_transitioning` flag in the server guards against concurrent lifecycle operations. The second request receives HTTP 409 (Conflict) immediately. Health checks during transitions return `{"status": "transitioning"}` without blocking.

### 3.11 P2P Transfer Failure Orphans RDMA State

**Discovery**: If the sender crashes or the network fails mid-transfer, the receiver's RDMA MRs, Gloo process groups, and NRT runtime are left allocated. The receiver appears healthy but cannot wake again because resources are stuck.

**Fix**: Wrap the P2P receive in a try/except that explicitly:
1. Deregisters RDMA MRs (`dereg_async`)
2. Releases NeuronCores (`spike_reset`)
3. Destroys Gloo process groups
4. Returns the engine to sleeping state

The error is propagated to the caller as a 503 response with detailed latency breakdown for debugging.

### 3.12 HuggingFace Rate Limiting at Scale

**Discovery**: Starting 100 engines that each hit the HuggingFace API for model config caused HTTP 429 (Too Many Requests) for the last ~10 engines.

**Fix**: Set `HF_HUB_OFFLINE=1` for all standby engines. Model configs are cached locally from the first download. This also speeds up engine startup by ~2s per engine.

### 3.13 NRT Init Fails After Dirty Process Kills

**Discovery**: After force-killing engines (`kill -9`), the neuron kernel module retains stale references. New engines start successfully (Python level) but `nrt_init()` fails with `NRT_FAILURE(1)` even with `NEURON_RT_RESET_CORES=0`, because the underlying cores are in a corrupted state that requires a firmware reset to recover.

**Fix**: Two-tier NRT init strategy:
1. **Fast path**: Try with `NEURON_RT_RESET_CORES=0` (0.15s when cores are clean)
2. **Retry with reset**: If fast path fails, retry without the env var (allows firmware reset, takes 2-5s on clean systems)
3. **Operational requirement**: After dirty kills, a machine reboot is required to clear the neuron kernel module's stale state. The cleanup script should always use graceful shutdown (SIGTERM → wait → SIGKILL only as last resort).

---

## 4. Performance Results

### 4.1 End-to-End Wake-Up Latency

**LLaMA-3-70B, TP=32, cross-instance (trn1.32xlarge)**:

| Configuration | Wake-Up Latency | Improvement |
|---|---|---|
| Checkpoint reload (baseline) | 86s (cold) / 32s (cached) | — |
| P2P transfer (no optimizations) | 28–31s | — |
| + Skip NRT firmware reset | 9.5–12.5s | 2.5x |
| + Sender MR pre-registration | 7.1s | 3.8x |
| + Overlapped connect + QP reset | **7.0s** | **4.6x** |

**Qwen3-30B-A3B, TP=32, cross-instance (trn1.32xlarge)**:

| Configuration | Wake-Up Latency | Improvement |
|---|---|---|
| Checkpoint reload (baseline) | ~45s (cold) | — |
| P2P transfer (optimized) | **5.1s** | **~9x** |

### 4.2 Sleep Latency

| Scenario | Latency | Notes |
|---|---|---|
| Deferred (>60s after wake) | **~2s** | MR deregistration already complete |
| Immediate (<60s after wake) | 63–65s | Must wait for 435 MR deregistration |

In production, engines serve traffic for minutes to hours between switches, so the ~2s fast path is the expected case.

### 4.3 RDMA Transfer Breakdown (Optimized)

**Qwen3-30B-A3B, TP=32, 1977 MB/rank, cross-instance:**

| Phase | Time |
|---|---|
| Receiver MR registration | 2.36s |
| Sender RDMA connect (overlapped) | 2.0s (hidden) |
| RDMA write | 0.63s |
| HTTP + gather | 0.09s |
| **P2P total** | **3.25s** |

Effective aggregate throughput: ~14 GB/s across 32 ranks (limited by single-NIC per rank on trn1).

### 4.4 Scalability: 10-Engine Sequential Wake Test

10 standby engines on a single trn1.32xlarge, woken sequentially from a sender on a separate instance. Each engine: wake, verify inference correctness, sleep.

| Engine | P2P Transfer | Total Wake | Inference |
|--------|---|---|---|
| 1 (cold start) | 3.22s | 21.63s | Correct |
| 2 | 3.25s | 4.75s | Correct |
| 3 | 3.23s | 5.03s | Correct |
| 4 | 3.23s | 4.99s | Correct |
| 5 | 3.23s | 5.07s | Correct |
| 6 | 3.21s | 5.01s | Correct |
| 7 | 3.23s | 5.10s | Correct |
| 8 | 3.21s | 5.04s | Correct |
| 9 | 3.22s | 5.06s | Correct |
| 10 | 3.24s | 5.11s | Correct |
| **Warm avg** | **3.23s** | **5.02s** | **10/10** |

P2P transfer is rock-solid at 3.21–3.25s with zero degradation across engines.

### 4.5 Scalability: Standby Engine Density

| Metric | Per Engine | At 100 Engines (measured) | At 110 Engines (limit) |
|---|---|---|---|
| Host memory | 3.0 GB | 303 GB | 330 GB |
| TCP ports (sleeping) | ~0 | 4 total | ~4 |
| Python processes | 2 | 200 | 220 |
| Startup time (staggered 2s) | ~15s | ~215s (total) | ~235s (total) |

**Measured**: 100 engines (50 Qwen3 + 50 LLaMA) on trn1.32xlarge uses 303/495 GB (61%). Theoretical max remains ~110+ (memory-limited). TCP connections drop to near-zero because Gloo is destroyed during sleep.

### 4.6 Non-Blocking Inference During Push

The sender continues serving inference requests with **zero stalls** while pushing weights to a receiver:

| Phase | Avg Latency | Max Latency | Stalls |
|---|---|---|---|
| Before push (baseline) | 775 ms | 787 ms | 0 |
| During push | 800 ms | 829 ms | 0 |
| After push | 800 ms | 808 ms | 0 |

3% latency increase during push, well within normal variance. The RDMA push runs in background threads with no lock contention on the inference path.

### 4.7 Scalability: 100-Engine Multi-Model Test

100 standby engines (50 Qwen3-30B-A3B + 50 LLaMA-3-70B) on a single trn1.32xlarge (Instance B), with senders on Instance A. Sequential wake/infer/sleep cycles across both models.

**Setup:**
- Instance A (172.31.44.131): Qwen3 sender (port 8000) + LLaMA sender (port 8001, time-shared)
- Instance B (172.31.40.200): 50 Qwen3 standby (ports 9000-9049) + 50 LLaMA standby (ports 9050-9099)

**Phase 1 — Qwen3 sequential (10 engines):**

| Engine | Wake | P2P Transfer | Inference | Sleep (deferred) |
|--------|------|---|---|---|
| 9000 | 9.6s (cold) | 3.23s | Correct | ~2s |
| 9001 | 6.5s | 4.60s | Correct | ~2s |
| 9002 | 5.1s | 3.23s | Correct | ~2s |
| 9003 | 6.8s | 3.28s | Correct | ~2s |
| 9004 | 5.0s | 3.23s | Correct | ~2s |
| 9005–9009 | 5.1–5.2s | 3.2s | Correct | ~2s |

**Phase 2 — LLaMA sequential (5 engines):**

| Engine | Wake | P2P Transfer | Inference | Sleep (deferred) |
|--------|------|---|---|---|
| 9050 | 7.2s | 4.92s | Correct | ~2s |
| 9051 | 7.0s | 4.88s | Correct | ~2s |
| 9052 | 7.0s | 4.96s | Correct | ~2s |
| 9053 | 6.5s | 5.00s | Correct | ~2s |
| 9054 | 7.2s | 4.97s | Correct | ~2s |

In production, engines serve traffic for minutes between switches. MR deregistration runs asynchronously and completes within ~60s of wake, so deferred sleep takes only ~2s.

**Phase 3 — Cross-model switching:**

| Step | Latency | Details |
|------|---------|---------|
| Sleep Qwen3 sender (deferred) | ~2s | MR dereg already done; release NeuronCores |
| Wake LLaMA sender | 6.6s | Reload LLaMA on same cores |
| Wake LLaMA receiver | 7.2s | P2P transfer from LLaMA sender |
| Sleep LLaMA sender (deferred) | ~2s | |
| Wake Qwen3 sender | 4.3s | |
| Wake Qwen3 receiver | 5.1s | P2P from Qwen3 sender |
| **Full model switch** | **~12s** | sleep(~2s) + sender wake + receiver wake |

**Resource usage with 100 sleeping engines:**

| Metric | Value |
|---|---|
| Memory (100 engines) | 303 GB / 495 GB (61%) |
| Per-engine memory | ~3.0 GB |
| TCP connections (sleeping) | 4 total |
| Python processes | 200 (2 per engine: server + resource tracker) |

### 4.8 Summary

| Metric | Value |
|---|---|
| Wake-up latency (warm, LLaMA-3-70B TP=32) | **7.0s** |
| Wake-up latency (warm, Qwen3-30B TP=32) | **5.1s** |
| Sleep latency (deferred) | **~2s** |
| Cross-model switch (end-to-end) | **~12s** |
| Checkpoint size reduction (LLaMA-3-70B) | **35%** (214 GB → 139 GB) |
| Inference impact during push | **0 stalls** |
| Max standby engines per instance | **~110** |
| P2P consistency (10-engine test) | **10/10 correct** |
| Multi-model consistency (100-engine test) | **15/15 correct** |
| Bidirectional P2P | **Verified** (A→B and B→A) |

---

## Appendix: Comparison with Alternatives

| | P2P RDMA (this work) | Checkpoint from local disk | Checkpoint from S3/EFS | CPU Offload |
|---|---|---|---|---|
| **Prerequisite** | Active sender loaded | Compiled checkpoint on local NVMe | Compiled checkpoint in S3 | Weights in CPU RAM |
| **Switch latency (70B, TP=32)** | 7s | 44–86s | 90–180s | 30–60s |
| **Switch latency (30B MoE, TP=32)** | 5s | ~45s | ~90s | ~20s |
| **Bandwidth** | 100 Gbps EFA | 3.4 GB/s NVMe | 1–2 GB/s network | 25 GB/s PCIe |
| **Sender impact** | 0 stalls | N/A | N/A | N/A |
| **Standby cost** | 3 GB RAM | Full instance | Full instance | Full instance + CPU RAM |
| **Multi-model switching** | ~12s | 60–120s | 120–240s | 40–80s |
| **Max standby per instance** | 100+ | 1 | 1 | 2–3 |
| **Requires local checkpoint** | No | Yes | No (but slow) | No |
| **Hardware requirement** | EFA NICs | Large NVMe | Network bandwidth | Large CPU RAM |
