# On-Demand Multi-Model LLM Serving with Second-Level Cold Start on Neuron

## 1. Motivation

### 1.1 Background: Multi-Model Serving Is the Norm

Production LLM platforms serve tens to hundreds of models simultaneously — diverse in model family (LLaMA, Qwen, Mistral, DeepSeek, …), size (8B to 600B+), and fine-tuned variants per customer. This is the fundamental operating reality, not an edge case.

AWS Mantle — the inference infrastructure behind Amazon Bedrock (100+ foundation models) — dynamically allocates a shared accelerator fleet across hundreds of deployments with independent, highly bursty traffic patterns. Production LLM traces show strong diurnal and weekly periodicity, with request volumes capable of doubling within minutes and fluctuating by orders of magnitude between peak working hours and off-peak periods. Fixed hardware allocation per model is economically untenable at this scale. The infrastructure must reallocate accelerators across models in real time for viable multi-model serving.

### 1.2 The Multi-Model Scaling Dilemma

LLM platfroms face a concrete resource allocation problem. Traffic patterns are bursty — a model may see orders-of-magnitude more load during peak hours and near-zero at off-peak. On Trainium instances, there are two choices: keep models always-loaded (expensive) or load models on-demand (slow).

The **always-loaded** approach dedicates instances to each model, keeping it running and ready to serve at all times. Each model must be provisioned for its peak traffic to avoid dropping requests or long waiting time during bursts. Since peaks are brief, most reserved instances sit idle most of the time, especially during off-peak hours. This makes always-loaded approach economically sub-optimal.

The **on-demand** path avoids this cost by loading models only when traffic arrives, but introduces a **cold start** bottleneck. When a request burst arrives for a model that isn't sufficiently loaded, the platform must initialize LLM engines and load weights to serve the burst.

### 1.3 Why Cold Start Is Fundamentally Slow

A traditional cold start serializes every phase — hardware provisioning, software initialization, kernel compilation (can be skipped), and weight loading — into a single blocking pipeline. Each phase must complete before the next can begin, and every scale-up event pays the full cost from scratch.

```
Traditional cold start (serialized):

┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐
│ Hardware │  │ Imports  │  │ NRT init │  │   NEFF   │  │  Weight  │  │       │
│   init   │→ │ + spawn  │→ │ + Gloo   │→ │ compile  │→ │   load   │→ │ Serve │
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └───────┘
                              Total: ~7–13 minutes
```

This makes on-demand scaling impractical — by the time a new engine is ready, the traffic burst that triggered it has likely subsided.

### 1.4 Cold-start Latency Breakdown on Neuron

We profiled cold-start latency on a trn2.48xlarge instance (32 logical NeuronCores with LNC=2, 4 instance store NVMe drives, 16 EFA NICs) running **LLaMA-3.1-70B** with TP=32 under the vLLM+NKIPy serving framework. The model checkpoint size is 139 GB.

<table>
<tr><th>Category</th><th>Phase</th><th>Latency</th><th>Bottleneck</th></tr>
<tr><td rowspan="5"><b>Software init</b></td>
    <td>Python imports + framework init + worker spawn</td><td>~20s</td><td>vLLM, PyTorch, plugin loading, fork 32 worker processes</td></tr>
<tr><td>Gloo distributed init</td><td>~1s</td><td>Establish Gloo mesh across 32 ranks</td></tr>
<tr><td>NeuronCore runtime init</td><td>~19s</td><td>Firmware reset + device allocation across 32 workers (measured: 4–16s per worker, 19s wall)</td></tr>
<tr><td>NEFF kernel compilation</td><td>14–83s</td><td>Neuron compiler traces + compiles kernels (~14s cached, ~83s cold)</td></tr>
<tr><td>NEFF kernel load + warmup</td><td>~20s</td><td>Load pre-compiled NEFF, profile, allocate KV cache</td></tr>
<tr><td><b>Weight loading</b></td>
    <td>From FSx for Lustre</td><td>~121s</td><td>Measured: 139 GB at 1.15 GB/s (4.5 TiB PERSISTENT_2, 32 shards in parallel)</td></tr>
<tr><td><b>Hardware init</b></td>
    <td>Provisioning accelerator</td><td>~5min</td><td>Readying NeuronCores for new engine (e.g., ~5min in Mantle)</td></tr>
<tr><td><b>Total</b></td>
    <td>Cold start (FSx, cached kernels)</td><td><b>~8.3min</b></td><td></td></tr>
</table>

**Measured end-to-end** (FSx, cached kernels, no hardware provisioning): **195s** (~3.3 min) from process start to first token. Weight loading from FSx dominates at 121s (62% of total).

### 1.5 Checkpoint Read Latency from FSx

Measured on **trn2.48xlarge** with FSx for Lustre (4.5 TiB PERSISTENT_2, 4 OSTs). Page cache dropped before each test; 32 shards read in parallel (one per TP rank).

| Model | Checkpoint Size | Cold Read Time | Throughput |
|---|---|---|---|
| Qwen3-30B-A3B (TP=32) | 59 GB | 51s | 1.15 GB/s |
| LLaMA-3.1-70B (TP=32) | 139 GB | 121s | 1.15 GB/s |
| Qwen3-235B-A22B (TP=32) | 448 GB | **386s** | 1.16 GB/s |

### 1.6 The Operational Challenge

**How do we scale up LLM model serving on demand with second-level cold-start latency?**

---

## 2. System Design

### 2.1 Design Principles

We reduce cold-start latency from minutes to seconds through three key design principles:

**1. Decouple engine initialization from weight loading and compilation.** A traditional cold start serializes everything: process spawn → NRT init → compile → load weights → serve. We separate these concerns so that CPU-heavy work (Python imports, framework init, NEFF compilation) happens once at deployment time, not on the critical path of each scale-up event.

**2. Standby engine pool with shared hardware.** Each instance hosts 100+ standby (sleeping) engines of different models sharing the same NeuronCores, with only one active at a time. This is made possible by the decoupling in principle 1: since sleeping engines hold only CPU-resident state (no device resources), hundreds can coexist on a single instance. Because the hardware is already provisioned and shared across all engines in the pool, scaling up a new model does not require provisioning new accelerator hardware — it simply switches which engine holds the NeuronCores. This eliminates the ~5min hardware init latency entirely. Every instance in the cluster independently initializes its own standby pool, so the number of instances that can serve any given model scales linearly with cluster size.

**3. P2P RDMA weight transfer.** Instead of reading weights from disk (bounded by NVMe at ~4 GB/s) or network storage (FSx/S3), we push weights directly from an active engine's device memory over EFA (e.g., 3.2 Tbps on trn2.48xlarge). This reduces weight loading from 51–386s to ~5s and eliminates the requirement for a local checkpoint entirely.

**Expected result**: Cold start drops from **8–13 minutes** to **< 10 seconds** for all models up to 235B parameters at TP=32.

### 2.2 System Overview

The system has three components:

- **Standby engine pool (§2.3)**: Each instance maintains 100+ pre-initialized engines sharing the same NeuronCores, with only one active at a time. Scaling up a model means switching which engine holds the cores (~9s), not provisioning new hardware (~5min).
- **P2P weight transfer (§2.4)**: When a standby engine wakes, an active engine of the same model on another instance pushes weights directly into the receiver's device memory via per-rank RDMA over EFA. The sender continues serving during the transfer — model weights are immutable during inference, so no coordination is needed.
- **LLM serving scheduler (§2.5)**: An external scheduler monitors per-model load and orchestrates scaling decisions. It routes user requests to active engines and triggers `/wake_up` or `/sleep` on standby engines based on real-time demand.

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
          │                                       │  
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

### 2.3 Standby Engine Pool

Each instance in the cluster runs a **standby engine pool** — 100+ pre-initialized engines of different models sharing the same NeuronCores. Only one engine is active at a time; others sleep in a minimal state (zero device resources). Sleeping engines hold only CPU-resident state (Python process, compiled NEFFs in memory, framework metadata) and consume no NeuronCore or device memory resources. This makes it feasible to initialize hundreds of engines concurrently at deploy time — the bottleneck is CPU and host memory (~3 GB RAM per sleeping engine), not accelerator capacity.

```
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
│  │ Wait sleep │→ │NRT switch│→ │   Gloo   │→ │  P2P weight  │→ │Serve │  │
│  │    ~2s     │  │          │  │   init   │  │ transfer ~5s │  │      │  │
│  │            │  │          │  │   ~1s    │  └──────────────┘  └──────┘  │
│  └────────────┘  └──────────┘  └──────────┘                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                        Total: <10s
```

**NRT switch vs NRT init.** Traditional cold starts require **NRT init** — a full firmware reset (~19s) that zeroes device memory and reprovisions the runtime. This is necessary because the prior core state is unknown (crash, force-kill, stale data). Our standby pool guarantees cores are always left in a known-good state via clean shutdown (`nrt_close()`), so the wake-up path uses **NRT switch** which skips the firmware reset entirely: 0.2s device acquisition instead of ~19s.

### 2.4 P2P Weight Transfer

The data plane uses **NIXL** (Network Interface eXchange Library) with the **LIBFABRIC** backend over AWS EFA for direct device-to-device RDMA. Each TP rank on both sender and receiver has its own NIXL agent that manages memory registration and transfer operations. The sender pushes weights directly from device memory while continuing to serve inference — zero downtime on the sender side.

**Transfer protocol (single-round-trip, receiver-initiated):**

The receiver orchestrates the entire transfer in a single HTTP request. It registers its device memory, gathers RDMA metadata from all ranks, and POSTs it to the sender. The sender adds the receiver as a remote agent, performs RDMA WRITE into the receiver's device memory, and returns. Only one HTTP call is needed — the bulk data moves via one-sided RDMA WRITE.

```
         Receiver                                 Sender
         ────────                                 ──────
            │                                        │
  ┌─────────┴────────────────────────────────────────┴────────────┐
  │ Step 1: Receiver setup (all ranks in parallel)                │
  │                                                               │
  │  Each rank:                                                   │
  │    - Register device buffers with NIXL (VRAM, LIBFABRIC)      │
  │    - Obtain agent metadata (RDMA addresses)                   │
  │  Rank 0 gathers metadata + buffer VAs from all ranks          │
  └─────────┬────────────────────────────────────────┬────────────┘
            │                                        │
  ┌─────────┴────────────────────────────────────────┴────────────┐
  │ Step 2: Transfer (single HTTP round-trip)                     │
  │                                                               │
  │  Rank 0: POST /nkipy/push_weights {receivers} ────▶ Sender     │
  │                                                               │
  │  Sender (all ranks in parallel):                              │
  │    - Add receiver as remote NIXL agent                        │
  │    - RDMA WRITE: local VRAM ═══════════════▶ receiver VRAM    │
  │    - Direct device-to-device, no CPU copies                   │
  │                                                               │
  │  Sender: 200 OK ◀──────────────────────────────── Sender      │
  └─────────┬────────────────────────────────────────┬────────────┘
            │                                        │
  ┌─────────┴────────────────────────────────────────┴────────────┐
  │ Step 3: Done                                                  │
  │                                                               │
  │  Receiver: weights are in device memory, ready for inference  │
  │  Memory registration persists until sleep (no cleanup needed  │
  │  on critical path — destroyed during next sleep cycle)        │
  └───────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **One-sided RDMA WRITE**: The sender writes directly into the receiver's registered device memory regions. No receiver CPU involvement during transfer — the receiver simply waits for the HTTP response.
- **Single contiguous MR registration**: All weight buffers are registered as one contiguous VRAM region (one `ibv_reg_mr` / dmabuf FD instead of N), minimizing registration overhead.
- **Persistent registration**: Memory regions stay registered until sleep. Repeated wake-ups with the same memory layout skip registration entirely (fast-path no-op).
- **Epoch-tagged agents**: Each wake cycle creates a fresh NIXL agent with an incremented epoch suffix (e.g., `nkipy_host_r0_e3`). The sender adds each new receiver as a distinct remote agent — no need to invalidate stale connections.
- **Non-blocking sender**: The sender's inference loop is never blocked. RDMA writes execute on the NIXL progress thread; the HTTP handler returns only after completion.

### 2.5 Engine Interface and Scheduling

#### 2.5.1 Engine HTTP Endpoints

Each engine exposes HTTP endpoints for lifecycle management:

**Scheduler-facing endpoints** — initiated by the LLM Serving Scheduler based on per-model workload metrics:

| Endpoint | Action |
|----------|--------|
| `POST /nkipy/wake_up` | Allocate tensors, receive weights via P2P, reload kernels |
| `POST /nkipy/sleep` | Release NeuronCores, destroy NIXL agent + Gloo, enter standby |
| `GET /nkipy/health` | Returns engine state (sleeping/active/transitioning) |

**Internal endpoints** — used by the P2P weight transfer protocol between engines:

| Endpoint | Action |
|----------|--------|
| `POST /nkipy/push_weights` | Sender performs parallel RDMA WRITEs to 1–N receivers and returns |
| `GET /nkipy/rdma_metadata` | Receiver exposes per-rank NIXL agent metadata for broadcast orchestration |

#### 2.5.2 LLM Serving Scheduler

**Why traditional schedulers cannot reschedule at runtime.** Every reallocation requires a full cold start (2–8 minutes depending on model size), during which NeuronCores sit idle. This exceeds the timescale of traffic bursts — by the time a new engine is ready, the spike has likely subsided. Traditional infrastructure (e.g., Mantle) therefore avoids rescheduling entirely, instead provisioning dedicated hardware per model periodically (e.g., every 5 minutes). The scheduler is trapped: provision conservatively and risk under-serving during spikes, or provision aggressively and pay for idle resources.

**Why our approach enables real-time rescheduling.** We reduce reallocation to <10 seconds (NRT switch 0.2s + Gloo 1s + P2P transfer ~5s). Since traffic bursts typically last tens of seconds, the scheduler can now observe a spike, reallocate, and begin serving before the burst subsides. **This transforms scheduling from a prediction problem into a reaction problem** — the scheduler acts on observed demand rather than forecasted demand.

**Auto-scaler (future work):** The scheduler monitors per-model request rates and queue depths in real time, triggering `/wake_up` on standby engines when load exceeds capacity, and `/sleep` when load drops. The exact scaling policy (thresholds, cooldown periods, preemptive warm-up) is ongoing work. More details in [SERVING_SCHEDULER_DESIGN.md](SERVING_SCHEDULER_DESIGN.md).

---

## 3. Performance Results

> **Platform**: All results are measured on **trn2.48xlarge** exclusively. The NIXL LIBFABRIC backend relies on the dmabuf ioctl for VRAM registration, which has only been validated on Trn2's kernel driver. Trn1 support is untested and not planned (trn1 instances are being deprecated).

### 3.1 End-to-End Latency

Measured on **trn2.48xlarge** (32 logical NeuronCores, 16 EFA NICs, 2 TB RAM).

| Model | Traditional Cold Start | Wake-Up | Sleep |
|---|---|---|---|
| Qwen3-30B-A3B (TP=32) | ~7 min | **3.9–5.8s** | **~2.5s** |
| LLaMA-3.1-70B (TP=32) | ~8 min | **4.4–6.2s** | **~2.5s** |
| Qwen3-235B-A22B (TP=32) | ~13 min | **6.6–9.0s** | **~2.5s** |

### 3.2 Latency Breakdown

**Qwen3-30B-A3B (TP=32, trn2.48xlarge, cross-instance, direct device RDMA):**

| Phase | Latency | Notes |
|---|---|---|
| Gloo distributed init | 0.5s | |
| NRT init + tensor alloc | 0.4s | Fast path (skip firmware reset) |
| **P2P transfer** (1.8 GB/rank) | 3.4s | Direct device→device via NIXL LIBFABRIC |
| Kernel load + barrier | 0.4s | |
| **Total wake-up** | **3.9–5.8s** (avg 4.8s) | |

Effective aggregate throughput: 59 GB in 3.4s = ~139 Gbps across 32 ranks × 16 EFA NICs (~37 Gbps per rank).

**LLaMA-3.1-70B (TP=32, trn2.48xlarge, cross-instance, direct device RDMA):**

| Phase | Latency | Notes |
|---|---|---|
| Gloo distributed init | 0.2s | |
| NRT init + tensor alloc | 0.5s | Fast path (skip firmware reset) |
| **P2P transfer** (4.6 GB/rank) | 3.6s | Breakdown below |
| Kernel load + barrier | 0.4s | |
| **Total wake-up** | **4.4–6.2s** (avg 5.0s) | |

P2P transfer breakdown (receiver perspective):

| Sub-phase | Latency | Notes |
|---|---|---|
| Receiver VRAM MR registration | 2.4s | 1 dmabuf FD + 1 ibv_reg_mr for 4.6 GB |
| Metadata gather | 0.4s | dist.gather_object (32 ranks) |
| **RDMA write** | **0.8s** | 146 GB, 1,422 Gbps (89% of 1,600 Gbps wire speed) |

**Qwen3-235B-A22B (TP=32, trn2.48xlarge, cross-instance, direct device RDMA):**

| Phase | Latency | Notes |
|---|---|---|
| Gloo distributed init | 0.3s | |
| NRT init + tensor alloc | 0.5s | Fast path (skip firmware reset) |
| **P2P transfer** (14 GB/rank) | 5.5s | Breakdown below |
| Kernel load + barrier | 0.4s | |
| **Total wake-up** | **6.6–9.0s** (avg 7.6s) | |

P2P transfer breakdown (receiver perspective):

| Sub-phase | Latency | Notes |
|---|---|---|
| Receiver VRAM MR registration | 2.6s | 1 dmabuf FD + 1 ibv_reg_mr for 14 GB |
| Metadata gather | 0.3s | dist.gather_object (32 ranks) |
| **RDMA write** | **2.6s** | 448 GB, 1,384 Gbps (86% of 1,600 Gbps wire speed) |

All transfers use direct device RDMA — no host staging or CPU copies on the data path. The RDMA write achieves 86% of unidirectional EFA bandwidth; the primary optimization opportunity is reducing MR registration overhead.

**P2P RDMA vs FSx cold load:**

| Model | FSx Cold Read | P2P Transfer | Speedup |
|---|---|---|---|
| Qwen3-30B-A3B (TP=32) | 51s | 3.4s | **15×** |
| LLaMA-3.1-70B (TP=32) | 121s | 3.9s | **31×** |
| Qwen3-235B-A22B (TP=32) | 386s | 5.7s | **68×** |

### 3.3 Scalability

#### 3.3.1 Standby Engine Density

Measured on **trn2.48xlarge** (2 TB host RAM, 32 logical NeuronCores).

| Metric | Value (500 engines) |
|---|---|
| Host memory per engine | ~2.7 GB |
| Total memory used | 1,329 GB / 1,991 GB (67%) |
| TCP ports (sleeping) | 1 per engine |
| Python processes (server + 32 workers) | 16,500 |
| Total pool launch time | ~22 min |

Each sleeping engine holds minimal TCP ports (Gloo destroyed during sleep). Engines are launched with a 1–2 second delay between consecutive launches to avoid CPU and TCP contention.

500 engines (LLaMA-3.1-70B, TP=32) on a single trn2.48xlarge. At 500 engines, 611 GB memory still available — theoretical limit is ~730 engines per instance.


### 3.4 Broadcast Weight Transfer (1-to-N)

When scaling up multiple engines simultaneously, the sender broadcasts weights to N receivers in a single `/nkipy/push_weights` call. All RDMA WRITEs are issued concurrently.

**LLaMA-3-8B (TP=8, 1 sender → 8 receivers, all receivers on same trn2.48xlarge):**

| Phase | Latency | Notes |
|---|---|---|
| Wake + MR register | 8.2s | 8 engines contend on shared NRT/EFA resources |
| **Broadcast transfer** | **2.8s** | 128 GB total, 367 Gbps aggregate |
| **Total wake-up** | **~11s** | 8 engines on same instance |

**Qwen3-235B-A22B (TP=32, 1 sender → 2 receivers on separate trn2 instances):**

| Phase | Latency | Notes |
|---|---|---|
| Wake + MR register | 4.4s | No contention (separate instances) |
| **Broadcast transfer** | **5.0s** | 448 GB × 2 receivers, 86% wire speed |
| **Total wake-up** | **~9.5s** | 2 engines on separate instances |

> **Note**: Same-instance results above (8.2s wake for LLaMA-3-8B × 8) are inflated by resource contention. With receivers on separate instances, wake drops from 8.2s to 4.4s. Transfer time is unchanged (network-bound).

**Broadcast scaling (all models, trn2.48xlarge, all receivers on same instance):**

| Model | TP | Receivers | Total wake-up | Transfer time | Aggregate throughput |
|---|---|---|---|---|---|
| LLaMA-3-8B | 8 | 1 | 3.5s | 0.55s | 29 Gbps |
| LLaMA-3-8B | 8 | 8 | **11.0s** | 2.79s | 367 Gbps |
| Qwen3-30B-A3B | 32 | 1 | 4.8s | 0.55s | 31 Gbps |
| LLaMA-3.1-70B | 32 | 1 | 5.0s | 0.55s | 54 Gbps |
| LLaMA-3.1-70B | 32 | 2 | **10.0s** | 1.82s | 54 Gbps |
| Qwen3-235B-A22B | 32 | 1 | 7.6s | 2.6s | 86% wire |
| Qwen3-235B-A22B | 32 | 2 | **13.6s** | 5.74s | 86% wire |

Broadcast overhead is sub-linear: pushing to 8 receivers takes ~5× longer than 1 receiver (not 8×), because the EFA network has sufficient bandwidth to serve multiple concurrent RDMA WRITEs. Total wake-up includes Gloo init, NRT init, tensor allocation, MR registration, and RDMA transfer.

### 3.5 Summary

| Metric | Value |
|---|---|
| Wake-up latency (Qwen3-30B-A3B, TP=32) | **3.9–5.8s** |
| Wake-up latency (LLaMA-3.1-70B, TP=32) | **4.4–6.2s** |
| Wake-up latency (Qwen3-235B-A22B, TP=32) | **6.6–9.0s** |
| Sleep latency | **~2.5s** |
| Max standby engines per instance | **500** (2.7 GB/engine) |
| P2P correctness | Verified (all iterations) |

---

## 4. Implementation Challenges

### 4.1 Active RDMA MRs Cause 60x `spike_reset()` Slowdown

**Discovery**: Sleep latency was 46.5s instead of the expected ~1s. Root cause: the NeuronCore firmware reset (`spike_reset()`) takes 7–25s when RDMA memory regions are still registered, vs 0.12s when MRs are deregistered first.

**Fix**: Destroy the NIXL agent (which deregisters all VRAM) before calling `spike_reset()`. The sleep path calls `endpoint.destroy()` first, then releases NeuronCores. Since the agent is destroyed synchronously before `spike_reset()`, sleep latency is consistently ~2s.

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

**Discovery**: Killing an engine with Ctrl+C left NIXL VRAM registrations active, causing the next `spike_reset()` on the same cores to take 25s+ (same root cause as 4.1).

**Fix**: Register cleanup handlers (atexit + SIGINT/SIGTERM) that call `endpoint.destroy()` **before** `spike_reset()`. Ordering is critical — NIXL agent destruction must precede firmware reset.

### 4.7 Stale Remote Agent Accumulation After Repeated Pushes

**Discovery**: After many consecutive pushes to different receiver instances, the sender's NIXL agent accumulated remote agent metadata entries that were never cleaned up.

**Root cause**: Each receiver creates a fresh agent with a new epoch on every wake cycle (e.g., `nkipy_host_r0_e0`, `nkipy_host_r0_e1`, ...). The sender calls `add_remote_agent()` for each new receiver, but never removes old entries.

**Fix**: Epoch-tagged agent names make each receiver identity unique. The `add_remote_agent()` call is idempotent — if the same agent name is already known, it's a no-op. Old entries are tiny metadata pointers (~KB) so accumulation is bounded by the total number of distinct receivers the sender has ever served, not the number of push operations. At 500 standby engines this is ~16 KB of metadata — negligible.

### 4.8 Partial NRT Init Leaves NeuronCores in Split-Brain State

**Discovery**: With 32 TP workers, some ranks succeed at NRT init while others fail (e.g., due to stale neuron driver state). Successful ranks hold NeuronCores, failed ranks report error. The engine returns an error response but the successful ranks' cores are now orphaned — no subsequent engine can claim them.

**Fix**: After NRT init, all ranks participate in a collective check via `dist.all_gather_object()`. If ANY rank failed, the successful ranks explicitly release their cores (`spike_reset()`) and the whole engine returns to sleep state. This ensures NeuronCores are never left in a partial-claim state.

### 4.9 Gloo TCP Thundering Herd at 50+ Engines

**Discovery**: Starting 50+ standby engines simultaneously caused Gloo distributed backend initialization to fail with TCP connection errors. All engines start their TP=32 Gloo groups concurrently, creating thousands of simultaneous TCP connections.

**Fix**: Two-part mitigation:
1. **Stagger engine launches** by 2s in the orchestration script (eliminates startup failures entirely)
2. **Retry logic in `_init_distributed()`** with exponential backoff (3 retries, 1s/2s/4s). If Gloo init fails, partial state is destroyed and retried. This handles transient connection failures during wake-up even when multiple engines wake near-simultaneously.

### 4.10 Concurrent Wake/Sleep Requests Cause Race Conditions

**Discovery**: If a scheduler sends `/wake_up` while a previous `/sleep` is still processing (NIXL agent destruction + spike_reset), both operations run simultaneously and corrupt internal state.

**Fix**: A `_nkipy_transitioning` flag in the server guards against concurrent lifecycle operations. The second request receives HTTP 409 (Conflict) immediately. Health checks during transitions return `{"status": "transitioning"}` without blocking.

### 4.11 P2P Transfer Failure Orphans RDMA State

**Discovery**: If the sender crashes or the network fails mid-transfer, the receiver's NIXL agent, Gloo process groups, and NRT runtime are left allocated. The receiver appears healthy but cannot wake again because resources are stuck.

**Fix**: Wrap the P2P receive in a try/except that explicitly:
1. Deregisters VRAM asynchronously (`endpoint.deregister_async()`)
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

### 4.14 Token Embedding Redundancy Wastes Transfer Bandwidth

**Discovery**: LLaMA-3-70B's token embedding is 2.1 GB and was replicated identically across all 32 ranks. Transferring 2.1 GB × 32 = 67 GB of redundant data wasted bandwidth and inflated checkpoint size.

**Fix**: Shard the embedding along the hidden dimension:
- **Before**: `[128256, 8192]` full copy per rank = 2.1 GB × 32 = 67 GB redundant
- **After**: `[128256, 256]` per rank = 65.7 MB — unique per rank, transfers via RDMA

This reduced checkpoint from 214 GB to 139 GB (35% savings) and eliminates 67 GB of redundant RDMA traffic.

### 4.15 NRT Init Blocks Scalability on Trn2

**Discovery**: The 500-engine scalability test on Trn2 failed immediately — every engine after the first crashed with `NRT_FAILURE: nrt_allocate_neuron_cores: Logical Neuron Core(s) not available`. The first engine claims all 32 logical NCs during `init_device()`, leaving none for subsequent sleeping engines.

**Root cause**: `init_device()` eagerly called `get_spike_singleton()` which does `nrt_init()` and allocates the NeuronCore specified by `NEURON_RT_VISIBLE_CORES`. This happened before `load_model()` could determine the engine has no checkpoint (standby mode).

**Fix**: Make NRT initialization conditional on having a checkpoint. Standby engines (no `NKIPY_CHECKPOINT`) skip `get_spike_singleton()` in `init_device()` — NRT will be lazily initialized during `nkipy_wake_up()` when the engine actually needs a NeuronCore. Kernel compilation (`_compile_kernels()`) uses NKI trace/compile which is CPU-only and doesn't require NRT.

---

## Appendix

### A.1 Comparison with Alternatives

| | P2P RDMA (this work) | Checkpoint from local disk | Checkpoint from S3/EFS | CPU Offload |
|---|---|---|---|---|
| **Prerequisite** | Active sender loaded | Compiled checkpoint on local NVMe | Compiled checkpoint in S3 | Weights in CPU RAM |
| **Switch latency (70B, TP=32)** | 5s | 44–86s | 90–180s | 30–60s |
| **Switch latency (30B MoE, TP=32)** | 5s | ~45s | ~90s | ~20s |
| **Switch latency (235B MoE, TP=32)** | 7.6s | ~120s | ~400s | N/A (448 GB RAM) |
| **Bandwidth** | 630 Gbps EFA (aggregate) | 3.4 GB/s NVMe | 1–2 GB/s network | 25 GB/s PCIe |
| **Sender impact** | 0 stalls | N/A | N/A | N/A |
| **Standby cost** | 3 GB RAM | Full instance | Full instance | Full instance + CPU RAM |
| **Multi-model switching** | ~12s | 60–120s | 120–240s | 40–80s |
| **Max standby per instance** | 100+ | 1 | 1 | 2–3 |
| **Requires local checkpoint** | No | Yes | No (but slow) | No |
| **Hardware requirement** | EFA NICs | Large NVMe | Network bandwidth | Large CPU RAM |
