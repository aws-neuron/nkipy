# Fast On-demand Model Serving Scaling Up on Neuron

## 1. Motivation

### Over-Provision or Cold-Start Latency When Scaling Up LLM in Multi-Model Serving

Production LLM serving increasingly requires multiple models on shared cluster. Traffic patterns are bursty — a model may see 10x load during peak and near-zero at off-peak. On Trainium instances, operators face a choice: keep every model always-loaded (expensive) or load models on-demand (slow).

The **always-loaded** approach over-provisions instances to each model, keeping it running and ready to serve at all times. This eliminates cold-start latency entirely and requests are served immediately. However, in practice, most models sit idle most of the time — a fleet serving 10 models may see only 1–2 active at any moment, yet pays for all 10 continuously. This makes always-loaded impractical as the model catalog grows.

The **on-demand** path avoids this cost by loading models only when traffic arrives, but introduces a **cold start** bottleneck. When a request arrives for a model that isn't currently loaded, the system must initialize an engine and load weights before serving the first token. On Neuron, this cold-start latency has several components:

### Cold-start Latency Breakdown on Neuron

We profiled cold-start latency across two trn1.32xlarge instances (each with 32 NeuronCores, 4 instance store NVMe drives, 8 EFA NICs) running LLaMA-3-70B with TP=32 under the vLLM serving framework. The model checkpoint size is 139 GB.

<table>
<tr><th>Category</th><th>Phase</th><th>Latency</th><th>Bottleneck</th></tr>
<tr><td rowspan="5"><b>Software init</b></td>
    <td>Python imports + framework init + worker spawn</td><td>~22s</td><td>vLLM, PyTorch, plugin loading, fork 32 worker processes (measured)</td></tr>
<tr><td>Gloo distributed init</td><td>~1s</td><td>Establish Gloo mesh across 32 ranks (measured: ~1s without CPU contention)</td></tr>
<tr><td>NeuronCore runtime init</td><td>5–20s</td><td>Firmware reset, device allocation via NRT (measured: 5–20s first init)</td></tr>
<tr><td>NEFF kernel compilation</td><td>14–83s</td><td>Neuron compiler traces + compiles kernels (measured: ~9.2s/bucket; 8 buckets → ~14s cached, ~83s cold)</td></tr>
<tr><td>NEFF kernel load + warmup</td><td>~18s</td><td>Load pre-compiled NEFF, profile, allocate KV cache (measured)</td></tr>
<tr><td rowspan="3"><b>Weight loading</b></td>
    <td>From local NVMe</td><td>~35s</td><td>Instance store NVMe aggregate ~4 GB/s for 139 GB (measured: ~1 GB/s per drive × 4 drives)</td></tr>
<tr><td>From FSx for Lustre</td><td>70–140s</td><td>Shared bandwidth (~1 GB/s/TiB); degrades under concurrent access</td></tr>
<tr><td>From S3</td><td>105–175s</td><td>Download to local disk (~2–3 GB/s, 50–70s) + load from disk (~35s); sequential</td></tr>
<tr><td><b>Hardware init</b></td>
    <td>Provisioning accelerator</td><td>~5min</td><td>Readying NeuronCores for new engine (e.g., ~5min in Mantle)</td></tr>
<tr><td rowspan="3"><b>Total</b></td>
    <td>Cold start (local NVMe)</td><td><b>~7min</b></td><td></td></tr>
<tr><td>Cold start (FSx)</td><td><b>~8–9min</b></td><td></td></tr>
<tr><td>Cold start (S3)</td><td><b>~9–10min</b></td><td></td></tr>
</table>

Weight loading dominates: reading 139 GB of Neuron-compiled checkpoint from disk is bounded by storage bandwidth. trn1.32xlarge has 4 instance store NVMe drives (~1 GB/s each, ~4 GB/s aggregate as measured by `dd iflag=direct`), giving a best-case load time of ~35s. In practice, checkpoint format overhead and 32-worker contention push this to 44s+. From networked storage (FSx/S3), it can exceed 2–3 minutes. Engine startup overhead (imports, worker spawn, Gloo init) adds another 15–23s before weight loading even begins.

The operational challenge: **How do we scale up LLM model serving on demand with second-level cold-start latency?**

---

## 2. System Design

### 2.1 Design Principles

We reduce cold-start latency from minutes to seconds through three key design principles:

**1. Decouple engine initialization from weight loading and compilation.** A traditional cold start serializes everything: process spawn → NRT init → compile → load weights → serve. We separate these concerns so that CPU-heavy work (Python imports, framework init, NEFF compilation) happens once at deployment time, not on the critical path of each scale-up event.

**2. Standby engine pool with shared hardware.** Each instance hosts 100+ standby engines of different models sharing the same NeuronCores, with only one active at a time. This is made possible by the decoupling in principle 1: since sleeping engines hold only CPU-resident state (no device resources), hundreds can coexist on a single instance. Because the hardware is already provisioned and shared across all engines in the pool, scaling up a new model does not require provisioning new accelerator hardware — it simply switches which engine holds the NeuronCores. This eliminates the ~5min hardware init latency entirely. Every instance in the cluster independently initializes its own standby pool, so the number of instances that can serve any given model scales linearly with cluster size.

**3. P2P RDMA weight transfer.** Instead of reading weights from disk (bounded by NVMe at ~4 GB/s) or network storage (FSx/S3), we push weights directly from an active engine's device memory over EFA (e.g., 800 Gbps on trn1.32xlarge). This reduces weight loading from 35–175s to ~2s and eliminates the requirement for a local checkpoint entirely.

**Expected result**: Cold start drops from **8–10 minutes** to **< 10 seconds** for LLaMA-3-70B at TP=32.

### 2.2 System Overview

#### 2.2.1 Standby Engine Pool

A traditional cold start serializes every phase — hardware provisioning, software initialization, kernel compilation, and weight loading — into a single blocking pipeline. Each phase must complete before the next can begin, and every scale-up event pays the full cost from scratch. This makes on-demand scaling impractical for latency-sensitive serving.

```
Traditional cold start (serialized):

┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌───────┐
│ Hardware │ │ Imports  │ │ NRT init │ │   NEFF   │ │ Weight load  │ │       │
│   init   │→│ + spawn  │→│ + Gloo   │→│ compile  │→│ (from disk)  │→│ Serve │
│          │ │  ~22s    │ │  ~6–21s  │ │  ~83s    │ │   35–175s    │ │       │
│  ~5min   │ │          │ │          │ │          │ │              │ │       │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘ └───────┘
                              Total: 8–10 minutes
```

In our design, each instance in the cluster runs a **standby engine pool** — 100+ pre-initialized engines of different models sharing the same NeuronCores. Only one engine is active at a time; others sleep in a minimal state (zero device resources). Sleeping engines hold only CPU-resident state (Python process, compiled NEFFs in memory, framework metadata) and consume no NeuronCore or device memory resources. This makes it feasible to initialize hundreds of engines concurrently at deploy time — the bottleneck is CPU and host memory (~3 GB RAM per sleeping engine), not accelerator capacity.


```
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
│  │    ~2s     │  │          │  │   init   │  │ transfer ~2s │  │      │  │
│  │            │  │          │  │   ~1s    │  └──────────────┘  └──────┘  │
│  └────────────┘  └──────────┘  └──────────┘                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                        Total: ~5s
```

An external scheduler orchestrates sleep/wake transitions via HTTP endpoints.
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

### 2.3 P2P Weight Transfer

The data plane uses a custom RDMA library ("Relay") built on AWS EFA. Each TP rank on both sender and receiver has its own RDMA endpoint that distributes transfers across all available NICs for maximum bandwidth utilization. The sender pushes weights directly from device memory while continuing to serve inference — zero downtime on the sender side.

**Transfer protocol (receiver-initiated):**

The receiver orchestrates the entire transfer. It sends two HTTP requests to the sender: one to establish the RDMA connection, and one to trigger the weight push. The sender is purely reactive — it connects, writes, and returns. Only the RDMA WRITE moves bulk weight data; the HTTP calls carry only lightweight metadata (endpoint addresses, buffer descriptors).

```
         Receiver                                 Sender
         ────────                                 ──────
            │                                        │
  ┌─────────┴────────────────────────────────────────┴────────────┐
  │ Phase 1: Setup                                                │
  │                                                               │
  │  Receiver tells sender "here is my RDMA address, connect"     │
  │                                                               │
  │  Receiver: POST /preconnect ────────────────────▶ Sender      │
  │                                                               │
  │  Both sides work in parallel:                                 │
  │    Receiver: register weight buffers with NIC                 │
  │    Sender:   establish RDMA connection to receiver            │
  │                                                               │
  │  Sender: 200 OK ◀──────────────────────────────── Sender      │
  └─────────┬────────────────────────────────────────┬────────────┘
            │                                        │
  ┌─────────┴────────────────────────────────────────┴────────────┐
  │ Phase 2: Weight Transfer                                      │
  │                                                               │
  │  Receiver tells sender "push weights into these buffers"      │
  │                                                               │
  │  Receiver: POST /push_weights ──────────────────▶ Sender      │
  │                                                               │
  │  Sender performs one-sided RDMA WRITE:                        │
  │    ◀═══════════════ weights (bulk data) ═══════════════       │
  │    Written directly into receiver's device memory             │
  │    No receiver CPU involvement during transfer                │
  │                                                               │
  │  Sender: 200 OK ◀──────────────────────────────── Sender      │
  └─────────┬────────────────────────────────────────┬────────────┘
            │                                        │
  ┌─────────┴────────────────────────────────────────┴────────────┐
  │ Phase 3: Cleanup (async, off critical path)                   │
  │                                                               │
  │  Receiver: deregister memory      Sender: close RDMA          │
  │  regions in background            connection in background    │
  │                                                               │
  │  Neither side blocks — cleanup runs asynchronously while      │
  │  both engines continue normal operation.                      │
  └───────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **One-sided RDMA WRITE**: The sender writes directly into the receiver's registered memory regions. No receiver CPU involvement during transfer — the receiver simply waits for completion.
- **Overlapped setup**: Connection establishment and memory registration run in parallel via an async preconnect phase. The critical path is the maximum of the two, not the sum.
- **Non-blocking push**: The sender runs RDMA writes in a background thread. The inference loop is never blocked — the sender continues serving requests throughout the entire transfer.
- **Deferred resource cleanup**: After transfer completes, connection teardown and memory deregistration run asynchronously in the background, completely off the critical path.

### 2.4 Engine HTTP Endpoints

Each engine exposes HTTP endpoints for lifecycle management. These fall into two categories:

**Scheduler-facing endpoints** — initiated by the LLM Serving Scheduler based on per-model workload metrics:

| Endpoint | Action |
|----------|--------|
| `POST /nkipy/wake_up` | Allocate tensors, receive weights via P2P, reload kernels |
| `POST /nkipy/sleep` | Release NeuronCores, destroy Gloo, enter standby |
| `GET /nkipy/health` | Returns engine state (sleeping/active) |

**Internal endpoints** — used by the P2P weight transfer protocol between engines:

| Endpoint | Action |
|----------|--------|
| `POST /nkipy/p2p_preconnect` | Exchange RDMA connection metadata (Phase 1) |
| `POST /nkipy/p2p_push_weights` | Trigger one-sided RDMA weight write (Phase 2) |

**Sleep state**: A sleeping engine holds ~3 GB of host memory (Python process + framework state) but **zero** device resources — NeuronCores are fully released and available for other engines.

### 2.5 LLM Serving Scheduler

**Why traditional schedulers cannot reschedule at runtime.** The fundamental issue is initialization latency. Every time NeuronCores are reallocated to a different model, the new engine must go through software initialization (imports, NRT init, NEFF compilation, weight loading) — 2–3 minutes during which the NeuronCores sit idle, unable to serve any traffic. This 2–3 minute reallocation time exceeds the timescale of traffic fluctuations — by the time a new engine is ready, the traffic burst that triggered it has likely subsided. Reactive scheduling is therefore pointless, and the scheduler cannot respond to real-time demand. Instead, traditional infrastructure (e.g., Mantle) avoids rescheduling entirely by provisioning dedicated hardware for each model ahead of time, incurring ~5 minutes of hardware provisioning latency. The scheduler is trapped: provision conservatively and risk under-serving during traffic spikes, or provision aggressively and pay for idle resources that were allocated for demand that never materialized.

**Why our approach enables real-time rescheduling.** Our system reduces the reallocation cost from 2–3 minutes to ~5 seconds. Because initialization is decoupled from the wake-up path (done once at deploy time), switching NeuronCores to a different model only requires NRT init (~0.2s), Gloo init (~1s), and P2P weight transfer (~2–4s). The fundamental change is that 5 seconds is shorter than the timescale of traffic fluctuations. Traffic bursts typically last tens of seconds to minutes — with a 5-second reallocation, the scheduler can observe a spike, reallocate, and begin serving before the burst subsides. With a 3-minute reallocation, the burst is likely over before the new engine is ready, making reactive scheduling pointless. This is what transforms scheduling from a prediction problem into a reaction problem: the reallocation latency is now below the response threshold, so the scheduler can act on observed demand rather than forecasted demand.

**Auto-scaler (future work):** The scheduler monitors per-model request rates and queue depths in real time. When a model's load exceeds its current serving capacity, the auto-scaler triggers `/wake_up` on additional standby engines for that model. When load drops, it calls `/sleep` to release resources back to the pool. The exact scaling policy (thresholds, cooldown periods, preemptive warm-up) is an area of ongoing work.

---

## 3. Performance Results on Trn1 Instances

### 3.1 End-to-End Wake-Up Latency

| Model | Traditional Cold Start | Our Approach |
|---|---|---|
| LLaMA-3-70B (TP=32) | ~8 min | **7.0s** |
| Qwen3-30B-A3B (TP=32) | ~7 min | **5.1s** |

### 3.2 Sleep Latency

| Model | Sleep Latency |
|---|---|
| LLaMA-3-70B (TP=32) | **~2s** |
| Qwen3-30B-A3B (TP=32) | **~2s** |

In production, engines serve traffic for minutes to hours between switches. RDMA resource cleanup runs asynchronously in the background and completes well before the next sleep, so the ~2s fast path is the expected case.

### 3.3 Scalability: Standby Engine Density

| Metric | Per Engine | At 100 Engines (measured) |
|---|---|---|
| Host memory | 3.0 GB | 303 GB |
| TCP ports (sleeping) | 1 | 100 |
| Python processes (server + 32 workers) | 33 | 3300 |
| Total pool launch time | — | ~215s |

Each sleeping engine holds 1 TCP port for its HTTP server (Gloo is destroyed during sleep). Engines are launched one at a time with a 2-second delay between consecutive launches to avoid CPU and TCP contention. Each engine initializes in the background (~15s), so multiple engines are initializing concurrently. Total time to bring up the full pool: 100 × 2s = ~200s, plus ~15s for the last engine to finish.

**Measured**: 100 engines (50 Qwen3 + 50 LLaMA) on trn1.32xlarge uses 303/495 GB host memory (61%). The trn1.32xlarge has 512 GB total host RAM, of which ~495 GB is available after OS and system services.

### 3.4 Non-Blocking Inference During P2P Weight Transfer

The sender continues serving inference requests while pushing weights to a receiver. End-to-end latency measured per request (prompt + full generation, max_tokens=128):

| Phase | Avg E2E Latency | Max E2E Latency |
|---|---|---|
| Before push (baseline) | 775 ms | 787 ms |
| During push | 800 ms | 829 ms |
| After push | 800 ms | 808 ms |

3% latency increase during push, well within normal variance. The RDMA push runs in background threads with no lock contention on the inference path.

### 3.5 Scalability: 100-Engine Multi-Model Test

100 standby engines (50 Qwen3-30B-A3B + 50 LLaMA-3-70B) on a single trn1.32xlarge (Instance B), with senders on Instance A. Sequential wake/infer/sleep cycles across both models.

**Setup:**
- Instance A (172.31.44.131): Qwen3 sender (port 8000) + LLaMA sender (port 8001, time-shared)
- Instance B (172.31.40.200): 50 Qwen3 standby (ports 9000-9049) + 50 LLaMA standby (ports 9050-9099)

**Phase 1 — Qwen3 sequential (10 engines):**

| Engine | Wake | P2P Transfer | Inference | Sleep |
|--------|------|---|---|---|
| 9000 | 9.6s (cold) | 3.23s | Correct | ~2s |
| 9001 | 6.5s | 4.60s | Correct | ~2s |
| 9002 | 5.1s | 3.23s | Correct | ~2s |
| 9003 | 6.8s | 3.28s | Correct | ~2s |
| 9004 | 5.0s | 3.23s | Correct | ~2s |
| 9005–9009 | 5.1–5.2s | 3.2s | Correct | ~2s |

**Phase 2 — LLaMA sequential (5 engines):**

| Engine | Wake | P2P Transfer | Inference | Sleep |
|--------|------|---|---|---|
| 9050 | 7.2s | 4.92s | Correct | ~2s |
| 9051 | 7.0s | 4.88s | Correct | ~2s |
| 9052 | 7.0s | 4.96s | Correct | ~2s |
| 9053 | 6.5s | 5.00s | Correct | ~2s |
| 9054 | 7.2s | 4.97s | Correct | ~2s |

**Phase 3 — Cross-model switching:**

| Step | Latency | Details |
|------|---------|---------|
| Sleep Qwen3 sender | ~2s | Release NeuronCores |
| Wake LLaMA sender | 6.6s | Reload LLaMA on sender instance |
| Wake LLaMA receiver | 7.2s | P2P transfer from LLaMA sender |
| Sleep LLaMA sender | ~2s | Release NeuronCores |
| Wake Qwen3 sender | 4.3s | Reload QWen on sender instance|
| Wake Qwen3 receiver | 5.1s | P2P from Qwen3 sender |

### 3.6 Latency Breakdown

**Wake-up latency breakdown (LLaMA-3-70B, TP=32, cross-instance):**

| Phase | Latency | Notes |
|---|---|---|
| NRT init (skip firmware reset) | ~0.2s | |
| Gloo distributed init | ~1s | |
| RDMA MR registration | 2.36s | |
| Sender RDMA connect (overlapped) | 2.0s | Hidden behind MR registration |
| RDMA write (4.3 GB/rank) | 0.63s | |
| HTTP + gather | 0.09s | |
| NEFF kernel reload | ~0.8s | |
| **Total wake-up** | **~7.0s** | |

Effective aggregate throughput: ~14 GB/s across 32 ranks (limited by single-NIC per rank on trn1).

### 3.7 Summary

| Metric | Value |
|---|---|
| Wake-up latency (warm, LLaMA-3-70B TP=32) | **7.0s** |
| Wake-up latency (warm, Qwen3-30B TP=32) | **5.1s** |
| Sleep latency | **~2s** |
| Inference impact during push | **0 stalls** |
| Max standby engines per instance | **100+** |
| P2P consistency (10-engine test) | **10/10 correct** |
| Multi-model consistency (100-engine test) | **15/15 correct** |
| Bidirectional P2P | **Verified** (A→B and B→A) |

---

## 4. Implementation Challenges

### 4.1 Active RDMA MRs Cause 60x `spike_reset()` Slowdown

**Discovery**: Sleep latency was 46.5s instead of the expected ~1s. Root cause: the NeuronCore firmware reset (`spike_reset()`) takes 7–25s when RDMA memory regions are still registered, vs 0.12s when MRs are deregistered first.

**Fix**: Deregister all MRs asynchronously immediately after transfer completes. Sleep path waits for deregistration if still in progress. If sleep is called >60s after wake (typical in production), MR deregistration has already finished and sleep takes ~2s.

### 4.2 NRT Init Firmware Reset Dominates Wake Latency (5–15s)

**Discovery**: `nrt_init()` reboots NeuronCore firmware on every wake cycle, even though a clean `nrt_close()` leaves cores in a known-good state.

**Fix**: Coordinate via a marker file (`/tmp/nkipy_nrt_closed`). After any clean shutdown, write the marker. On next init, check for the marker and set `NEURON_RT_RESET_CORES=0` to skip the firmware reboot. Cold starts (no marker) still get a full reset. Result: `nrt_init` drops from 5–15s to **0.15s**.

**Safety**: HBM scrub (zero-fill) still runs. Cross-model correctness verified — cores that ran LLaMA can immediately run Qwen3 without firmware reset.

### 4.3 TCP Port Exhaustion at Scale

**Discovery**: Each standby engine's Gloo distributed backend consumed ~482 TCP ephemeral ports. At 30 engines on one instance, all 28,232 ports were exhausted.

**Fix**: Destroy Gloo process groups during sleep, recreate during wake. Sleeping engines hold ~1 TCP port (the HTTP server) instead of 482. This raised the scalability limit from ~58 engines (TCP-bound) to **100+ engines** (memory-bound at ~3 GB each).

### 4.4 CPU Spinning in Idle Workers

**Discovery**: vLLM's SHM message queue polling loop busy-spins at 99% CPU per worker. With 29 engines x 32 workers = 928 spinning threads, new engine startup slowed from 15s to 185s due to scheduling contention.

**Fix**: Enable `VLLM_SLEEP_WHEN_IDLE=1` — after 3s of inactivity, workers call `time.sleep(0.1)` instead of `sched_yield()`. Startup time stays constant at ~15s regardless of standby pool size.

### 4.5 Stale Tensor References After Wake

**Discovery**: After sleep/wake, model output was garbage. The model's internal tensor references pointed to deallocated memory from the previous wake cycle.

**Fix**: During wake, write new weights into the existing allocated tensors rather than creating new tensor objects. The model graph retains valid references to the same memory.

### 4.6 RDMA Resource Leak on Ctrl+C

**Discovery**: Killing an engine with Ctrl+C left RDMA resources registered, causing the next `spike_reset()` on the same cores to take 25s+ (same root cause as 4.1).

**Fix**: Register cleanup handlers that deregister RDMA MRs **before** calling `spike_reset()`. Ordering is critical — RDMA cleanup must precede firmware reset.

### 4.7 Sender RDMA QP Exhaustion After Repeated Pushes

**Discovery**: The sender crashed with `Assertion '*qp' failed` in `RDMAChannelImpl::initQP()` after ~11 consecutive pushes. Each push created a new RDMA Queue Pair connection to the receiver, but connections were never freed — they accumulated until the EFA device's QP pool was exhausted.

**Root cause**: With pre-registered MRs, `push_to_peer()` kept the sender's `RankEndpoint` alive across pushes (to avoid 2.3s re-registration). But the endpoint accumulated stale QP connections from all previous receivers. The relay library has no per-connection `disconnect()` API.

**Fix**: After each push completes, call `reset_endpoint_async()` which destroys the old endpoint (freeing all QPs) and re-registers MRs on a fresh endpoint in a background thread. The 54s re-registration runs concurrently with the receiver's sleep/next-wake cycle. The next `preconnect_to_peer()` call blocks on `_ensure_endpoint()` until the background reset finishes — if less than 60s have passed since the last push, the preconnect waits; in production (>60s between pushes), no waiting occurs.

### 4.8 Partial NRT Init Leaves NeuronCores in Split-Brain State

**Discovery**: With 32 TP workers, some ranks succeed at NRT init while others fail (e.g., due to stale neuron driver state). Successful ranks hold NeuronCores, failed ranks report error. The engine returns an error response but the successful ranks' cores are now orphaned — no subsequent engine can claim them.

**Fix**: After NRT init, all ranks participate in a collective check via `dist.all_gather_object()`. If ANY rank failed, the successful ranks explicitly release their cores (`spike_reset()`) and the whole engine returns to sleep state. This ensures NeuronCores are never left in a partial-claim state.

### 4.9 Gloo TCP Thundering Herd at 50+ Engines

**Discovery**: Starting 50+ standby engines simultaneously caused Gloo distributed backend initialization to fail with TCP connection errors. All engines start their TP=32 Gloo groups concurrently, creating thousands of simultaneous TCP connections.

**Fix**: Two-part mitigation:
1. **Stagger engine launches** by 2s in the orchestration script (eliminates startup failures entirely)
2. **Retry logic in `_init_distributed()`** with exponential backoff (3 retries, 1s/2s/4s). If Gloo init fails, partial state is destroyed and retried. This handles transient connection failures during wake-up even when multiple engines wake near-simultaneously.

### 4.10 Concurrent Wake/Sleep Requests Cause Race Conditions

**Discovery**: If a scheduler sends `/wake_up` while a previous `/sleep` is still processing (MR deregistration takes 60s), both operations run simultaneously and corrupt internal state.

**Fix**: A `_nkipy_transitioning` flag in the server guards against concurrent lifecycle operations. The second request receives HTTP 409 (Conflict) immediately. Health checks during transitions return `{"status": "transitioning"}` without blocking.

### 4.11 P2P Transfer Failure Orphans RDMA State

**Discovery**: If the sender crashes or the network fails mid-transfer, the receiver's RDMA MRs, Gloo process groups, and NRT runtime are left allocated. The receiver appears healthy but cannot wake again because resources are stuck.

**Fix**: Wrap the P2P receive in a try/except that explicitly:
1. Deregisters RDMA MRs (`dereg_async`)
2. Releases NeuronCores (`spike_reset`)
3. Destroys Gloo process groups
4. Returns the engine to sleeping state

The error is propagated to the caller as a 503 response with detailed latency breakdown for debugging.

### 4.12 HuggingFace Rate Limiting at Scale

**Discovery**: Starting 100 engines that each hit the HuggingFace API for model config caused HTTP 429 (Too Many Requests) for the last ~10 engines.

**Fix**: Set `HF_HUB_OFFLINE=1` for all standby engines. Model configs are cached locally from the first download. This also speeds up engine startup by ~2s per engine.

### 4.13 NRT Init Fails After Dirty Process Kills

**Discovery**: After force-killing engines (`kill -9`), the neuron kernel module retains stale references. New engines start successfully (Python level) but `nrt_init()` fails with `NRT_FAILURE(1)` even with `NEURON_RT_RESET_CORES=0`, because the underlying cores are in a corrupted state that requires a firmware reset to recover.

**Fix**: Two-tier NRT init strategy:
1. **Fast path**: Try with `NEURON_RT_RESET_CORES=0` (0.15s when cores are clean)
2. **Retry with reset**: If fast path fails, retry without the env var (allows firmware reset, takes 2-5s on clean systems)
3. **Operational requirement**: After dirty kills, a machine reboot is required to clear the neuron kernel module's stale state. The cleanup script should always use graceful shutdown (SIGTERM → wait → SIGKILL only as last resort).

### 4.14 Token Embedding Exceeds RDMA MR Limit

**Discovery**: LLaMA-3-70B's token embedding is 2.1 GB — too large for a single RDMA MR registration (EFA limit ~1 GB). The transfer would fail without special handling.

**Fix**: Shard the embedding along the hidden dimension:
- **Before**: `[128256, 8192]` full copy per rank = 2.1 GB × 32 = 67 GB redundant
- **After**: `[128256, 256]` per rank = 65.7 MB — fits in a single MR, transfers via RDMA

This also reduced checkpoint disk from 214 GB to 139 GB (35% savings).

---

## Appendix

### A.1 10-Engine Sequential Wake Test

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

### A.2 Comparison with Alternatives

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
