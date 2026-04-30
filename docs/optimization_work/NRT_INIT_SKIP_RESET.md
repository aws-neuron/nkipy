# NRT Init: Skip NC Reset (`NEURON_RT_RESET_CORES=0`)

## Problem

`nrt_init()` dominates wake-up latency. On a trn1.32xlarge (32 NeuronCores), it takes 5–15s per cycle. NRT debug logs (`NEURON_RT_LOG_LEVEL=DEBUG`) reveal that 97% of that time is spent in `tdrv_init_mla_phase1`, which performs a hardware NC firmware reset:

```
nrt_init called
...
Reset triggered            ← firmware reboot begins
...  ~5.1s gap ...
Reset done                 ← firmware reboot complete
```

The NC reset reboots the firmware on every core. It is necessary on a cold start (cores never initialized), but redundant after a clean `nrt_close()` — the firmware is already in a known-good state.

## Solution

Use a shared marker file (`/tmp/nkipy_nrt_closed`) to coordinate across engine processes on the same machine. When any engine completes `nrt_close()` during sleep, it writes the marker. When any engine wakes up, it checks for the marker and sets `NEURON_RT_RESET_CORES=0` before `nrt_init()`.

```python
# In nkipy_sleep(), after spike_reset() calls nrt_close():
spike_reset()
open("/tmp/nkipy_nrt_closed", "w").close()

# In nkipy_wake_up(), before nrt_init():
if os.path.exists("/tmp/nkipy_nrt_closed"):
    os.environ["NEURON_RT_RESET_CORES"] = "0"
get_spike_singleton()  # calls nrt_init()
```

This design ensures that all standby engines benefit, not just the one that previously slept. Cold cores (no prior `nrt_close` on this machine) still get a full reset because the marker doesn't exist.

### What Still Runs with `reset_cores=0`

HBM scrub (zero-fill of 32 GB device memory) still executes — it takes ~97ms and is not skipped. All device memory is clean before the next model loads.

## Validation

### Single-Model Cycle (TinyLlama TP=8, same-node)

3-cycle wake/sleep test on a single trn1.32xlarge:

| Metric | Cycle 1 (cold) | Cycle 2 (`reset_cores=0`) | Cycle 3 (`reset_cores=0`) |
|--------|---------------|--------------------------|--------------------------|
| `nrt_init` | 5.237s | 0.148s | 0.147s |
| Total wake-up | 16.2s | 4.04s | 3.98s |

Inference output identical across all cycles.

### Cross-Model Interleaved (Qwen3-30B-A3B + LLaMA-3-70B, TP=32, 2× trn1.32xlarge)

Test: Qwen3 wake → infer → sleep → LLaMA wake → infer → sleep → Qwen3 wake → infer. Two sleeping engines on Instance B (receiver), sender engines cycled on Instance A. Sender engines are slept before termination.

| Step | Model | `nrt_init` | Total wake-up | Inference |
|------|-------|-----------|---------------|-----------|
| 1. First wake (cold, no marker) | Qwen3-30B-A3B | 6.32s | 30.74s | Correct |
| 2. First wake (marker from step 1) | LLaMA-3-70B | **0.15s** | **12.46s** | Correct |
| 3. Second wake (marker exists) | Qwen3-30B-A3B | **0.17s** | **9.54s** | Correct |

Step 1 is a cold start (no marker file exists). Step 1's sleep writes the marker file. Step 2 is LLaMA's **first-ever wake-up** on a separate engine process (port 8001) — it reads the marker written by Qwen3's sleep (port 8000) and skips NC reset. Step 3 confirms cross-model correctness (Qwen3 on cores that last ran LLaMA).

### Inference Correctness

Verified across all cycles for both models:

| Model | Prompt | Output |
|-------|--------|--------|
| Qwen3-30B-A3B | "The capital of France is" | " Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain" |
| LLaMA-3-70B | "The capital of France is" | " Paris. It is the largest city in France. It is the capital of the Île-de-F" |

Qwen3 output is byte-identical between step 1 and step 3, confirming no corruption from the cross-model interleave.

## Sender MR Pre-Registration

### Problem

After fixing `nrt_init`, P2P transfer became the dominant wake-up cost (~7.7s). Profiling the transfer pipeline (Qwen3-30B-A3B, TP=32):

| Phase | Receiver | Sender |
|-------|----------|--------|
| MR registration | 2.34s | 2.35s |
| RDMA connect | — | 2.00s |
| RDMA write (1977 MB) | — | 0.63s |
| HTTP + gather | 0.09s | — |

The sender's 2.35s MR registration happened on every push, even though the sender's weight buffers don't change between pushes. This was serialized on the critical path: receiver registers → HTTP post → sender registers → connect → write.

### Solution

Pre-register sender MRs at model load time and after wake-from-checkpoint. `push_to_peer()` detects pre-registered MRs and skips registration (`reg 0.000s`). The sleep path handles cleanup: `dereg_sync()` if MRs are still active, or `wait()` if async dereg is in progress.

### Results (Qwen3-30B-A3B, TP=32, cycle 2+)

| Phase | Before | After |
|-------|--------|-------|
| P2P transfer | 7.67s | **5.18s** |
| Sender reg | 2.35s | **0.00s** (pre-registered) |
| RDMA write | 0.63s | 0.63s |
| **Total wake-up** | **9.5s** | **7.1s** |

### Remaining P2P Breakdown (after sender pre-registration)

| Phase | Time | Notes |
|-------|------|-------|
| Receiver MR registration | ~2.3s | On critical path, cannot pre-register receiver |
| Sender RDMA connect | ~2.0s | QP establishment per wake-up |
| RDMA write | ~0.6s | 1977 MB at 25 Gbps |
| HTTP + serialization | ~0.3s | Metadata exchange |

## Overlapped Sender Connect with Receiver Registration

### Problem

After sender pre-registration, the remaining P2P latency was ~5.2s with the sender's 2.0s RDMA connect (`add_remote_endpoint`) running sequentially after receiver MR registration. The receiver registered MRs first, then sent metadata to the sender, who then connected and transferred — these steps were unnecessarily serialized.

### Solution

Split the receiver's metadata exchange into two HTTP calls:

1. **Early preconnect** (`/nkipy/p2p_preconnect`): Sends receiver endpoint metadata to sender BEFORE MR registration. Fires in a background thread so all ranks proceed to MR registration immediately.
2. **Push descriptors** (`/nkipy/p2p_push_weights`): Sends transfer descriptors after registration completes.

The sender's `add_remote_endpoint()` (2.0s) runs in parallel with receiver's `register_memory()` (2.3s). The sender caches the connection, so the subsequent push call's `add_remote_endpoint()` returns instantly.

```
Timeline (before):
  Receiver: [dereg+reg 2.3s] → [gather+POST] → ...wait...
  Sender:                                       [connect 2.0s] → [xfer 0.6s]
  Total: 2.3 + 2.0 + 0.6 = ~5.2s (measured: 5.18s)

Timeline (after):
  Receiver: [accept] → [POST /preconnect (async)] → [reg 2.3s] → [gather+POST]
  Sender:              [connect 2.0s .........]                    [xfer 0.6s]
  Total: max(2.3, 2.0) + 0.6 = ~2.9s (measured: 3.25s)
```

### Results (Qwen3-30B-A3B, TP=32, warm cycle)

| Phase | Before (sender prereg only) | After (+ overlap) |
|-------|------|-------|
| accept | — | 0.004s |
| preconnect (async) | — | 0.031s |
| reg | 2.34s | 2.36s |
| gather | 0.09s | 0.09s |
| post+xfer | 2.60s | 0.78s |
| **P2P total** | **5.18s** | **3.25s** |

The 0.78s post+xfer (vs 2.60s before) confirms the sender's connect is fully overlapped — it only needs to do the RDMA write (0.6s) plus HTTP round-trip.

## Combined Impact on Wake-Up Latency

For any wake-up after the first sleep cycle on the machine (all standby engines benefit):

| Phase | Original | + `reset_cores=0` | + sender pre-reg | + overlap | Speedup |
|-------|----------|-------------------|-----------------|-----------|---------|
| `nrt_init` | 5–15s | 0.15–0.17s | 0.17s | 0.16s | **35–100×** |
| P2P transfer | 7.7–10.4s | 7.7–10.4s | 5.2s | **3.25s** | **2.4–3.2×** |
| Total wake-up (TP=32) | 28–31s | 9.5–12.5s | 7.1s | **5.0s** | **~6×** |
| Total wake-up (TP=8, same-node) | 16.2s | 4.0s | — | — | **~4×** |

The remaining wake-up time is dominated by receiver MR registration (~2.3s, overlapped with sender connect), RDMA write (~0.6s), and Gloo init (~0.9s).

## 10-Engine Scalability Test (Qwen3-30B-A3B, TP=32)

10 standby engines started on Instance B (172.31.40.200, trn1.32xlarge), all sleeping. Single sender on Instance A (172.31.44.131). Each engine woken sequentially: wake → verify inference → sleep → next engine.

| Engine | nrt | p2p | gloo | total | server | Inference |
|--------|-----|-----|------|-------|--------|-----------|
| 1 (cold) | 5.79s | 3.22s | 0.86s | 21.63s | 21.83s | Correct |
| 2 | 0.17s | 3.25s | 0.58s | 4.75s | 4.81s | Correct |
| 3 | 0.15s | 3.23s | 0.91s | 5.03s | 5.21s | Correct |
| 4 | 0.16s | 3.23s | 0.83s | 4.99s | 5.01s | Correct |
| 5 | 0.16s | 3.23s | 0.95s | 5.07s | 5.21s | Correct |
| 6 | 0.17s | 3.21s | 0.95s | 5.01s | 5.11s | Correct |
| 7 | 0.18s | 3.23s | 0.92s | 5.10s | 5.21s | Correct |
| 8 | 0.17s | 3.21s | 0.89s | 5.04s | 5.11s | Correct |
| 9 | 0.17s | 3.22s | 0.94s | 5.06s | 5.21s | Correct |
| 10 | 0.17s | 3.24s | 0.93s | 5.11s | 5.21s | Correct |
| **Warm avg** | **0.17s** | **3.23s** | **0.88s** | **5.02s** | **5.12s** | **10/10** |

P2P transfer is extremely consistent at 3.21–3.25s across all 10 engines. No degradation as the sender serves more receivers.

### Sleep Latency

Sleep latency depends on whether background MR deregistration has completed:

| Scenario | dereg_wait | spike_reset | gloo_destroy | Total |
|----------|-----------|-------------|--------------|-------|
| Immediate (seconds after wake) | 61–63s | 0.12s | 0.8–1.4s | **63–65s** |
| Deferred (>60s after wake) | 0.0s | 0.12–0.14s | 0.8–1.4s | **~2s** |

MR deregistration (`ibv_dereg_mr`) runs asynchronously after each wake-up and takes ~60s for 435 MRs × 4 NIC contexts. If sleep is called before deregistration finishes, it must wait synchronously. In production, standby engines typically serve requests for much longer than 60s, so sleep is expected to hit the fast ~2s path.

## Safety

- `reset_cores=0` is only applied when the marker file exists, proving a prior `nrt_close()` ran on this machine.
- Cold starts (fresh machine, no marker) always use the default `reset_cores=1`.
- HBM scrub still runs, so device memory is zeroed before reuse.
- Cross-model interleaving is safe: cores that previously ran LLaMA can immediately run Qwen3 without firmware reset.
- Sender pre-registered MRs are cleaned up during sleep via `dereg_sync()` before `spike_reset()`, so no MR slowdown.

## Files Changed

- `nkipy/src/nkipy/vllm_plugin/worker.py` — Sleep: write `/tmp/nkipy_nrt_closed` marker after `spike_reset()`. Wake: read marker and set `NEURON_RT_RESET_CORES=0` before `get_spike_singleton()`. Load/wake: pre-register sender MRs when checkpoint is loaded. Sleep: `dereg_sync()` active MRs before `spike_reset()`. Added `nkipy_preconnect()` method for early RDMA connect.
- `nkipy/src/nkipy/vllm_plugin/server.py` — Added `/nkipy/p2p_preconnect` endpoint for early RDMA connect.
- `relay/src/relay/transfer.py` — `receive_from_peer()` sends endpoint metadata before MR registration via async HTTP POST. Added `preconnect_to_peer()` function.
- `relay/src/relay/__init__.py` — Added `preconnect_to_peer` to exports.

## Date

2026-04-29 (NRT skip reset), 2026-04-30 (sender pre-registration, overlapped connect)
