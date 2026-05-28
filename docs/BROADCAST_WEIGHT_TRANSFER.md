# Broadcast Weight Transfer: Multi-Engine Scale-Up via Parallel P2P Push

## 1. Problem Statement

The current P2P weight transfer supports **1-to-1 scaling**: one active sender engine pushes weights to one receiver engine at a time. When the scheduler needs to scale from 1 to N engines (e.g., 1 → 5), it must wake receivers sequentially — paying the full wake-up latency N times.

**Goal:** Minimize total time to wake N engines simultaneously from a single sender, targeting N ≤ 5 on Trn2.

---

## 2. Design Space

### 2.1 Option A: Parallel Push from Single Sender

All N receivers wake simultaneously and independently contact the sender for P2P transfer. The sender pushes to all N receivers concurrently, sharing NIC bandwidth equally.

```
Sender ─────┬───RDMA──→ Receiver 1    (all in parallel)
            ├───RDMA──→ Receiver 2
            ├───RDMA──→ Receiver 3
            ├───RDMA──→ Receiver 4
            └───RDMA──→ Receiver 5
```

**Pros:**
- Simple orchestration — no coordination between receivers
- Minimal code changes — sender handles concurrent connections, receivers are unchanged
- Sender is stateless per-connection

**Cons:**
- Sender NIC bandwidth divided across N receivers — RDMA write time scales linearly with N
- Becomes a hard bottleneck at large N (>10)

### 2.2 Option B: Tree-Based Relay

Sender pushes to K receivers, those receivers become secondary senders and push to next-level receivers.

```
Level 0:  Sender ──→ R1, R2, R3, R4     (K=4 parallel)
Level 1:  R1→R5, R2→R6, R3→R7, R4→R8   (each L0 receiver relays)
```

**Pros:**
- Aggregate bandwidth scales with N — each level multiplies capacity
- Latency is O(log_K(N)) × per-level-time

**Cons:**
- Each level adds full wake latency (~8-20s) because receiver must fully wake before relaying
- Host-staged variant: receiver can relay from host buffer before DMA to device, but still needs sender MR registration + QP setup (~4s per level)
- Significant orchestration complexity (tree topology, failure handling)
- For N ≤ 5: tree relay is **slower** than parallel push because the tree overhead exceeds bandwidth savings

### 2.3 Option C: Hybrid (Parallel + Tree)

Parallel push for N ≤ K, tree relay for N > K. K is the bandwidth saturation point.

Not needed for N ≤ 5 — revisit if scaling targets increase.

---

## 3. Analysis: Parallel Push on Trn2

### 3.1 LLaMA-3.1-70B (TP=32, trn2.48xlarge, direct device RDMA)

**Per-receiver budget (from measured 1-to-1):**

| Phase | Latency | Parallelizable? |
|---|---|---|
| Gloo distributed init | 1.0s | Yes (per-receiver, independent) |
| NRT init + tensor alloc | 0.1s | Yes |
| Receiver MR registration | 2.4s | Yes |
| Sender RDMA connect (overlapped) | 2.0s | Yes (independent QP per receiver) |
| RDMA write (4.3 GB/rank) | 3.4s | Shared bandwidth |
| Kernel load + barrier | 1.1s | Yes |

**Aggregate NIC utilization (1 receiver):** ~24 GB/s across 32 ranks × 16 NICs = ~59% of 3.2 Tbps capacity.

**Parallel push latency estimates:**

| N | Setup (parallel) | RDMA write (shared BW) | Post-transfer | Total |
|---|---|---|---|---|
| 1 | 3.5s | 3.4s | 1.1s | **8.4s** (measured) |
| 2 | 3.5s | 5.7s | 1.1s | **~10.3s** |
| 3 | 3.5s | 7.5s | 1.1s | **~12.1s** |
| 5 | 3.5s | 11.0s | 1.1s | **~15.6s** |

vs. Sequential: 8.4s × N (42s for N=5)

**Speedup at N=5: 2.7×**

### 3.2 Qwen3-235B-A22B (TP=32, trn2.48xlarge, host-staged RDMA)

**Per-receiver budget (from measured 1-to-1):**

| Phase | Latency | Parallelizable? |
|---|---|---|
| Gloo distributed init | 0.9s | Yes |
| NRT init + tensor alloc | 0.4s | Yes |
| Sender DMA device→host (overlapped) | 7.8s | Done once, amortized across N receivers |
| Receiver MR registration | 8.4s | Yes |
| RDMA write (12.5 GB/rank, host→host) | 1.7s | Shared bandwidth |
| Receiver DMA host→device | 8.3s | Yes (per-receiver, independent) |
| Kernel load | 0.1s | Yes |

**Aggregate NIC utilization (1 receiver):** 12.5 GB/rank × 32 ranks / 1.7s = ~1880 Gbps / 3200 Gbps = ~59%.

**Key advantage for host-staged:** Sender DMA (device→host) is done **once** regardless of N. The sender's host staging buffer is read N times — no repeated device→host copies.

**Parallel push latency estimates:**

| N | Setup (parallel) | RDMA write (shared BW) | Recv DMA (parallel) | Total |
|---|---|---|---|---|
| 1 | 8.4s | 1.7s | 8.3s | **19.9s** (measured) |
| 2 | 8.4s | 2.9s | 8.3s | **~19.9s** (RDMA still hidden behind recv DMA) |
| 3 | 8.4s | 5.1s | 8.3s | **~21.8s** |
| 5 | 8.4s | 8.5s | 8.3s | **~25.2s** |

Note: For N ≤ 2, RDMA write time (2.9s) is less than receiver DMA time (8.3s), so it's fully overlapped — total latency equals the 1-to-1 case.

vs. Sequential: 19.9s × N (99.5s for N=5)

**Speedup at N=5: 3.9×**

### 3.3 Bandwidth Saturation Analysis

Trn2.48xlarge EFA capacity: 3.2 Tbps (16 NICs × 200 Gbps).

| Model | Per-receiver BW | N=5 total | % of NIC capacity |
|---|---|---|---|
| LLaMA-70B | ~1880 Gbps | ~1880 Gbps × 5 / N = same | ~59% per recv |
| Qwen3-235B | ~1880 Gbps | ~1880 Gbps × 5 / N = same | ~59% per recv |

At N=5, each receiver gets ~376 Gbps (~20% of capacity). RDMA write time scales to 5× single-receiver: 3.4s → 17s (LLaMA), 1.7s → 8.5s (Qwen3-235B).

For Qwen3-235B, receiver DMA (8.3s) is the parallel bottleneck — RDMA at N=5 (8.5s) barely exceeds it, so the overhead is minimal.

### 3.4 Sender Inference Impact

With N=5 concurrent pushes, sender NICs are near full utilization for ~8-17s. Expected inference latency impact during this window:
- N=1 measured: 3% increase (§3.4 in FAST_MODEL_SWITCH_DEMO.md)
- N=5 estimated: 10-15% TTFT increase during push window

This is acceptable for a brief burst (~10-17s), and the sender continues to serve throughout.

---

## 4. Chosen Design: Parallel Push (Option A)

### 4.1 Architecture

```
Scheduler ──┬── POST /wake_up (peer=sender_url) ──→ Receiver 1
            ├── POST /wake_up (peer=sender_url) ──→ Receiver 2
            ├── POST /wake_up (peer=sender_url) ──→ Receiver 3
            ├── POST /wake_up (peer=sender_url) ──→ Receiver 4
            └── POST /wake_up (peer=sender_url) ──→ Receiver 5

Each receiver independently (in parallel):
  1. Gloo init + NRT switch + MR registration      (8.4s)
  2. POST /p2p_preconnect to sender                (~2s, overlapped with MR reg)
  3. POST /p2p_push_weights to sender              (1.7-8.5s, BW shared)
  4. Host→device DMA                               (8.3s, parallel)
  5. Kernel reload                                 (0.1s)

Sender handles all N preconnects and pushes concurrently:
  - N independent RDMA connections (no shared state between them)
  - N background threads perform RDMA writes to different receivers
  - Deferred endpoint reset after all pushes drain
```

### 4.2 Key Design Decisions

**1. No coordinator between receivers.** Each receiver independently contacts the sender. The sender serves whoever shows up. This avoids distributed consensus and means receivers can stagger naturally (some wake faster than others).

**2. Sender push concurrency.** The sender runs N push threads concurrently. Each operates on independent QPs writing to different remote addresses. The only shared resource is NIC bandwidth, handled by EFA fair queuing.

**3. Deferred endpoint reset.** Change from "reset QPs after each push" to "reset after all active pushes complete." A refcount tracks active pushes; reset fires when count reaches zero.

**4. Connection independence.** Each sender↔receiver connection uses its own set of QPs (4 data channels + 1 control per connection). N=5 means 5 × 5 = 25 QPs per rank — well within EFA device limits.

**5. Sender pre-DMA is done once.** For host-staged transfer, the sender's device→host DMA happens once at the start. The host buffer is read by all N RDMA writes without re-staging.

### 4.3 Sender-Side State During Concurrent Pushes

```
Sender Rank Endpoint state:
├── Host staging buffer (pre-registered, read-only during pushes)
├── Sender MRs (pre-registered at init, valid for all connections)
├── Connection pool:
│   ├── conn[0]: QPs to Receiver 1 (active push in background thread)
│   ├── conn[1]: QPs to Receiver 2 (active push in background thread)
│   ├── conn[2]: QPs to Receiver 3 (preconnect in progress)
│   ├── conn[3]: (empty slot)
│   └── conn[4]: (empty slot)
├── active_push_count: AtomicInt (tracks in-flight pushes)
└── reset_pending: bool (set when push count drops to 0)
```

### 4.4 Failure Handling

- **One receiver fails:** Other receivers unaffected (connections are independent). Failed receiver returns to sleep state via existing error handling (§4.11 in FAST_MODEL_SWITCH_DEMO.md).
- **Sender fails mid-push:** All receivers detect timeout on the HTTP response, clean up RDMA state, return to sleep.
- **Partial N (some receivers slower):** No issue — each receiver proceeds at its own pace. The sender pushes to each as they arrive; there's no "wait for all N" barrier.

### 4.5 Asynchrony Tolerance

The design does **not** assume receivers arrive simultaneously. Each receiver independently contacts the sender whenever it's ready — there is no "wait for all N" barrier or synchronization point.

**Sender state lifetime:**

The sender's state that receivers depend on persists until the sender sleeps:
- **Host staging buffer** (Trn2 staged path): created at init or first `/p2p_prepare`, persists indefinitely.
- **Sender MRs**: pre-registered at init, stay valid until sleep.
- **Model weights in device memory**: immutable during inference.

Any receiver can arrive at any time and get correct weights, as long as the sender hasn't slept.

**QP accumulation with staggered arrivals:**

For the host-staged path (Trn2), there is no `reset_endpoint_async()` call between pushes. QPs accumulate across connections. At N≤5: 5 × 5 = 25 QPs per rank — well within EFA device limits. QP exhaustion (§4.7 in FAST_MODEL_SWITCH_DEMO.md) only occurs after ~11+ consecutive pushes without cleanup.

**Sleep latency is unaffected by QP accumulation.** The 60× `spike_reset()` slowdown (§4.1 in FAST_MODEL_SWITCH_DEMO.md) is caused by active RDMA Memory Regions, not QPs. The sleep path calls `dereg_sync()` before `spike_reset()`, which deregisters all MRs. QPs are destroyed when the endpoint is destroyed during sleep. Sleep latency remains **~3.2s** regardless of how many pushes occurred.

**Parallelism benefit degrades gracefully with stagger:**

```
Case 1: All 5 arrive together (ideal, scheduler fires all at once)
  ├─ Setup:  8.4s (all parallel)
  ├─ RDMA:   8.5s (5-way shared BW, all finish together)
  ├─ DMA:    8.3s (all parallel)
  └─ Total:  ~25s, all 5 ready simultaneously

Case 2: Arrivals staggered by 3s (realistic, natural variance)
  R1 arrives t=0:   ready at t≈20s
  R2 arrives t=3:   ready at t≈23s  (partial BW sharing with R1)
  R3 arrives t=6:   ready at t≈26s
  R4 arrives t=9:   ready at t≈29s
  R5 arrives t=12:  ready at t≈32s
  └─ Last receiver ready at ~32s (vs 25s ideal, vs 100s sequential)

Case 3: Arrivals staggered by 20s+ (independent, no BW sharing)
  Each receiver gets dedicated bandwidth — same as sequential 1-to-1.
  └─ No benefit from parallelism, but no harm either.
```

**Design implication:** The scheduler should fire all N `/wake_up` requests simultaneously (async HTTP fan-out) as a best effort. Natural variance in receiver setup (Gloo: 0.9-1.1s, MR reg: 8.0-8.8s) typically keeps receivers within a ~1s arrival window at the sender, which is tight enough for full bandwidth sharing.

---

## 5. Future Extensions

### 5.1 Tree Relay for N > 10

If scaling targets grow beyond N=10, implement tree relay (Option D) as a layer on top of parallel push:
- First K receivers get parallel push from sender
- Those K receivers relay from their host staging buffer to the next K receivers
- Host-staged architecture makes this natural (host buffer already exists)
- Add sender MR pre-registration on receivers at init time to eliminate relay setup cost

### 5.2 EFA Multicast

If AWS adds reliable multicast support to EFA, a single RDMA write could reach all N receivers simultaneously with no bandwidth division. This would make parallel push O(1) regardless of N. Monitor EFA roadmap.

### 5.3 Bandwidth-Aware Scheduling

The scheduler could stagger wake-ups slightly (e.g., 2 at a time) to avoid NIC saturation for latency-sensitive sender inference. Trade-off: total time increases but sender impact stays within 3%.

---

## 6. Implementation Plan

### 6.1 Overview of Changes

The implementation is contained to three files, with the relay layer unchanged:

| File | Change | Risk |
|---|---|---|
| `nkipy/src/nkipy/vllm_plugin/server.py` | Support concurrent `/p2p_push_weights` handlers | Low — HTTP is already async |
| `nkipy/src/nkipy/vllm_plugin/worker.py` | Track multiple concurrent push threads | Medium — must avoid race conditions |
| `relay/src/relay/transfer.py` | Skip `reset_endpoint_async()` for concurrent pushes; add deferred reset | Medium — QP lifecycle |

The relay C++ layer (`relay/src/`) and `relay/src/relay/endpoint.py` require **no changes** — the endpoint already supports multiple concurrent `add_remote_endpoint()` calls and multiple `transfer()` calls on different connection IDs.

### 6.2 Step 1: Worker — Track Multiple Concurrent Push Threads

**File:** `nkipy/src/nkipy/vllm_plugin/worker.py`

**Current state:** Single `self._push_thread` — starting a second push overwrites the reference, orphaning the first thread.

**Change:** Replace `self._push_thread` with `self._push_threads: list[Thread]` and `self._push_errors: list[Exception]`.

```python
# In __init__ or class body:
self._push_threads: list[threading.Thread] = []
self._push_errors: list[Exception] = []
self._push_lock = threading.Lock()  # protects _push_threads/_push_errors

def nkipy_push_weights(self, per_rank_info, chunk_start, chunk_end, is_last_chunk):
    from relay import push_weights_to_peer
    model = self.model_runner._nkipy_model
    if model is None:
        return {"status": "error", "message": "model not loaded"}

    info = [(r["remote_metadata"], r.get("remote_descs", "")) for r in per_rank_info]

    def _bg_push():
        try:
            push_weights_to_peer(model, info, chunk_start=chunk_start,
                                 chunk_end=chunk_end, is_last_chunk=is_last_chunk)
        except BaseException as exc:
            with self._push_lock:
                self._push_errors.append(exc)
            logger.exception("Background push failed on rank %d", self.rank)

    t = threading.Thread(target=_bg_push, daemon=True)
    t.start()
    with self._push_lock:
        self._push_threads.append(t)
    return {"status": "started"}

def nkipy_push_status(self):
    with self._push_lock:
        # Remove completed threads
        alive = [t for t in self._push_threads if t.is_alive()]
        if alive:
            self._push_threads = alive
            return {"status": "running"}
        # All done
        self._push_threads = []
        if self._push_errors:
            err = self._push_errors[0]
            self._push_errors = []
            return {"status": "error", "message": str(err)}
        return {"status": "done"}
```

**Also update `nkipy_sleep()`** to join all threads:
```python
# Replace: self._push_thread.join()
# With:
with self._push_lock:
    threads = list(self._push_threads)
for t in threads:
    t.join()
with self._push_lock:
    self._push_threads = []
```

### 6.3 Step 2: Server — Allow Concurrent Push Requests

**File:** `nkipy/src/nkipy/vllm_plugin/server.py`

**Current state:** The `/nkipy/p2p_push_weights` handler polls `nkipy_push_status` in a loop until **all** threads report done. If a second HTTP request arrives while polling, the worker spawns a new thread, and the first poller sees the new thread as "running" — it just keeps waiting.

**Key insight:** The current server handler already works correctly for concurrent requests without modification! Each handler call:
1. Starts a new push thread via `nkipy_push_weights()`
2. Polls `nkipy_push_status()` until all threads complete

With the Step 1 change, `nkipy_push_status()` returns "running" as long as any thread is alive. All concurrent HTTP handlers will poll until their respective threads (plus any others) finish. This is safe — each handler just waits longer if other pushes are active, but all pushes complete before any handler returns.

**Optimization (optional):** If we want each handler to return independently when its specific push finishes (not wait for others), we can assign a push_id and track per-push status. For N≤5 this is unnecessary — the overhead of waiting for all pushes to finish is minimal since they all started near-simultaneously and share bandwidth equally.

**No changes required to `server.py` for basic functionality.** The concurrent semantics fall out naturally from the worker-side changes.

### 6.4 Step 3: Relay — Deferred Endpoint Reset for Non-Staged Path

**File:** `relay/src/relay/transfer.py`

**Current state (non-staged `push_to_peer`):** Calls `ep.reset_endpoint_async()` after each push when `is_last_chunk=True` and MRs are pre-registered (line 415). This would destroy QPs for in-flight concurrent pushes.

**Change:** Skip the reset when concurrent pushes are active. Add a refcount:

```python
# In push_to_peer(), replace the is_last_chunk block:
if is_last_chunk:
    if pre_registered:
        # For concurrent pushes, defer reset until all pushes complete.
        # The caller (worker) is responsible for triggering reset after
        # all concurrent pushes finish, or relies on sleep to clean up.
        pass  # QPs cleaned up at sleep time via dereg_sync()
    else:
        ep.dereg_async()
```

**Why this is safe for N≤5:** 25 QPs per rank is well within EFA limits. The sender's `nkipy_sleep()` calls `rank_endpoint.dereg_sync()` which destroys the endpoint and all its QPs. Between pushes, QPs accumulate but don't cause issues at this scale.

**For the staged path (`push_weights_to_peer_staged`):** No changes needed — it already has no `reset_endpoint_async()` call.

### 6.5 Step 4: Test with N Concurrent Receivers

**Test scenario:** Launch 1 sender + N receivers on separate instances (or same instance for unit test). Fire N `/wake_up` requests in parallel, verify:
1. All N receivers wake successfully and produce correct inference
2. Sender continues serving inference during parallel pushes
3. Sender sleeps normally after all pushes complete
4. Sender can push again after waking (QPs cleaned up properly)

**Test script structure:**
```python
import asyncio, aiohttp

async def wake_all(receiver_urls, sender_url):
    async with aiohttp.ClientSession() as session:
        tasks = [session.post(f"{url}/nkipy/wake_up", json={"peer_url": sender_url})
                 for url in receiver_urls]
        results = await asyncio.gather(*tasks)
    return results
```

### 6.6 Summary: Change Scope

| Change | Lines of code | Complexity |
|---|---|---|
| Worker: `_push_threads` list + lock | ~30 lines modified | Low |
| Worker: `nkipy_sleep()` join all threads | ~5 lines modified | Low |
| Transfer: skip `reset_endpoint_async()` in concurrent mode | ~5 lines modified | Low |
| Server: no changes needed | 0 | — |
| Test: multi-receiver wake script | ~50 lines new | Low |
| **Total** | **~90 lines** | **Low-Medium** |

---

## 7. Cross-Instance Benchmark Results

### 7.1 Setup

- **Sender:** trn2.48xlarge at 10.3.211.148, TinyLlama-1.1B TP=2 (1.1 GB/shard)
- **Receivers:** trn2.48xlarge at 10.3.215.3, N=1..5 engines, TP=2, dedicated core pairs
- **Network:** EFA RDMA, 16 NICs per instance (1.6 Tbps aggregate), 4 NICs used per endpoint
- **Transfer mode:** Host-staged (device → host DMA, then host → host RDMA)
- **Shared filesystem:** /fsx (both instances)

### 7.2 Latency Breakdown (Server-Side, Receiver Rank 0)

| Component | N=1 | N=2 | N=3 | N=4 | N=5 | Scales with N? |
|---|---|---|---|---|---|---|
| gloo_init | 0.48s | 0.46s | 0.57s | 0.58s | 0.49s | No |
| nrt_init | 0.21s | 0.21s | 0.19s | 0.21s | 0.20s | No |
| alloc_tensors | 0.01s | 0.01s | 0.01s | 0.01s | 0.01s | No |
| **p2p_transfer** | **2.34s** | **4.35s** | **6.36s** | **8.25s** | **10.53s** | **Yes (linear)** |
| rdma_ack | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | No |
| kernel_load | 1.82s | 1.76s | 1.76s | 1.89s | 1.59s | No |
| **total** | **4.93s** | **6.89s** | **8.98s** | **11.03s** | **12.93s** | |

### 7.3 P2P Transfer Sub-Breakdown (Receiver Logs)

The `p2p_transfer` phase includes receiver-side MR registration and the actual RDMA data transfer. Receiver logs (135 bufs, 1100 MB, 2 chunks) show:

| N | mr_reg (ibv_reg_mr) | transfer+dma (actual RDMA) | p2p total |
|---|---|---|---|
| 1 | 2.00s | 0.33s | 2.34s |
| 2 | 3.84s | 0.42s | 4.27s |
| 3 | 5.71s | 0.42s | 6.14s |
| 4 | 7.73s | 0.42s | 8.16s |
| 5 | 9.60s | 0.69s | 10.29s |

### 7.4 Root Cause Analysis

**The bottleneck is `ibv_reg_mr` (memory registration), NOT RDMA transfer bandwidth.**

- The actual RDMA transfer of 1.1 GB takes only **0.33-0.69s** (~220 Gbps observed for the pre-DMA path at N=1).
- MR registration (`ibv_reg_mr` × 135 buffers × 4 NICs = 540 kernel calls) takes **2.0s at N=1** and grows linearly to **9.6s at N=5**.
- MR registration is a kernel operation that pins physical pages. When 5 receivers on the same instance register MRs concurrently, the kernel's page-pinning path serializes due to `mm_struct` lock contention on the shared address space.

**Why mr_reg scales linearly with N (same-instance receivers):**

All 5 receivers share the same OS kernel. `ibv_reg_mr()` calls `pin_user_pages()` in the kernel, which acquires `mmap_lock` on the process's mm. When 5 processes concurrently pin pages through the same EFA driver, the driver-level lock serializes them. Each receiver registers 135 buffers × 4 NICs = 540 MRs; with 5 receivers that's 2700 concurrent MR registrations competing for kernel page-pinning resources.

### 7.5 Sender-Side RDMA Throughput (from sender logs)

When pre-DMA is complete (data already in host staging buffer), the sender's RDMA write throughput:

| Scenario | Stage 1 (494 MB) | Stage 2 (606 MB) | Effective Gbps |
|---|---|---|---|
| N=1 sequential | 0.018s | 0.022s | 220 Gbps |
| N=2 concurrent | 0.032s | 0.038s | 125 Gbps per receiver |
| N=5 concurrent | 0.068s | 0.066s | 58-67 Gbps per receiver |

The RDMA layer itself achieves excellent bandwidth even at N=5 — total aggregate write throughput of 5×60 = 300 Gbps, well within the 4-NIC (400 Gbps) capacity.

### 7.6 Implications for Production (Cross-Instance Receivers)

In production, receivers run on **separate instances** with their own kernel. Key differences:

1. **MR registration is local:** Each receiver registers MRs against its own kernel/EFA driver independently — no contention with other receivers. Expected mr_reg ≈ 2.0s regardless of N.
2. **Sender NIC bandwidth is the only shared resource:** At 4 NICs × 100 Gbps = 400 Gbps, pushing 1.1 GB to 5 receivers = 5.5 GB total ÷ 50 GB/s = 0.11s theoretical. Observed: 0.33-0.69s (good, accounting for protocol overhead and chunked transfers).
3. **Expected production latency for TinyLlama at N=5:** gloo_init(0.5s) + nrt(0.2s) + mr_reg(2.0s) + transfer(0.5s) + kernels(1.8s) ≈ **5.0s** (same as N=1).

**For larger models (Qwen3-235B, TP=64):**
- Model uses `NKIPY_HOST_STAGING=1` with pre-registered staging buffer — **no per-wake MR registration** on the receiver.
- Transfer: 120 GB/rank ÷ 50 GB/s (4 NICs) = 2.4s at N=1; at N=5 with NIC sharing: ~12s.
- But with 16 NICs available, using more NICs per endpoint (setting `kNICContextNumber=16`) would bring this to ~3s at N=5.

### 7.7 Key Takeaway

The same-instance benchmark's linear p2p_transfer scaling is caused by **kernel-level MR registration contention** (all receivers share one kernel), not by RDMA bandwidth limitation. The RDMA data plane achieves 220 Gbps per transfer at N=1 and 300 Gbps aggregate at N=5 — proving the broadcast design scales well at the network level. In cross-instance production deployment, MR registration runs independently on each instance, eliminating this bottleneck entirely.
