# test P2P weight transfer in NKIPy plugin

## Goal

Test the scalability of fast model switch in NKIPy based on P2P weight transfer.


## Test settings

- Use QWen3 as the base model with TP=32.
- There are two trn1 instances for tests: Instance A 172.31.44.131 (this instance) and Instance B 172.31.40.200
- the code path is: /home/ubuntu/vllm-nkipy/nkipy
- the python venv path is: /home/ubuntu/vllm-nkipy/nkipy/.venv/
- the example shell scripts are in: /home/ubuntu/vllm-nkipy/nkipy/examples/p2p/


## Two roles in the tests

There are two roles in the tests:
- server engine, which serves as the pusher of the model weights. The server engine is initialized with model checkpoints.
- receiver engine, which serves as the receiver of the model weights, The receiver engine is initialized without model checkpoints, and it waits for /wake_up endpoint to be activated. Multiple receiver engines are needed to test the scalability and they may or may not share the same set of Neuron cores.
- Due to Neuron contraints, if multiple engines are started on the same set of neuron cores, only one of them can be active and it must be asleep before another engine wakes up. We call the asleep receiver engines as `standby engines`.
- Server engine: `./run_engine.sh --model Qwen/Qwen3-30B-A3B --tp 32 --checkpoint ~/models/Qwen3-30b-a3b_TP32 --skip-cte`
- Receiver engine: `./run_engine.sh --model Qwen/Qwen3-30B-A3B --tp 32 --skip-cte --activate-venv`


## How to test

**Step 0:** Start the server engine on Instance A.
**Step 1:** Start a number of receiver engines on Instance B. Task: need to figure out if the system can support up to 50 receiver engines sharing the same set of hardware. If not, identify the bottleneck and figure out the solution. 
**Step 2:** Test the /sleep and /wake_up cycles on these standy engines. Given the same prompt, ensure the generated outputs from different receiver engines are identical since they share the same model weights.
**Step 3:** Dynamically start or kill receiver engines.
**Step 4:** Sleep the server engine and wake it up from the local model checkpoint.
**Step 5:** Sleep the server engine and wake it up from the active receiver engine with P2P weight transfer.
**Step 6:** After the first 5 steps pass, write an integration test with qwen3 for this scalablity feature

## Test results

### Step 0: Server engine on Instance A

Server engine started successfully on Instance A (172.31.44.131) port 8000 with Qwen3-30B-A3B TP=32.
Inference verified: `"The capital of France is"` -> `"Paris. The capital of Germany is Berlin..."`

### Step 1: Scalability test — standby receiver engines (same instance)

Tested on Instance A with TP=32 Qwen3-30B-A3B sleeping receiver engines started sequentially.
Each engine started with `HF_HUB_OFFLINE=1` and no checkpoint (standby mode).

**Result: 29 engines succeeded, engine 30 failed due to TCP ephemeral port exhaustion.**

Root cause of failure at engine 30:
```
RuntimeError: bind(2) for [172.31.44.131] failed with: Address already in use
```
Gloo distributed backend could not bind to ephemeral ports — the range (32768-60999 = 28,232 ports)
was saturated at ~14,000 ports used (each engine consumes ~482 TCP ports).

#### Per-engine resource cost (TP=32, measured average over 29 engines)

| Resource | Per Engine | Total at 29 |
|----------|-----------|-------------|
| Memory | 2.9 GB | 85 GB |
| Processes | 35 | 1,019 |
| TCP ports | 482 | 13,991 |
| File descriptors | 11,473 | 334,836 |
| SHM files | 34 | 986 |

#### Startup time growth (sequential starts)

| Engine # | Startup Time | Bottleneck |
|----------|-------------|------------|
| 1 | 15s | — |
| 5 | 30s | Python imports (torch/vllm) |
| 10 | 60s | Python imports + Gloo init |
| 15 | 85s | Python imports + Gloo init |
| 20 | 110s | Python imports + Gloo init |
| 25 | 155s | Python imports + Gloo init |
| 29 | 110s | Python imports + Gloo init |

Startup time increases because each new engine spawns 33 processes that must import
torch/vllm/transformers while competing for CPU with hundreds of existing processes.
The dominant phase is Python module import: from 4s (engine 1) to 35s (engine 20).

#### Scalability limits on trn1 (495 GB RAM, TP=32)

| Bottleneck | Limit | Notes |
|------------|-------|-------|
| **TCP ephemeral ports** | **~58 engines** | **Binding bottleneck** (28K ports / 482 per engine) |
| FDs (ulimit -n 1M) | ~91 engines | 11,473 FDs per engine |
| Memory (450 GB avail) | ~140 engines | 2.9 GB per engine |

#### Full resource consumption data

```
engines  mem_used_gb  mem_avail_gb  procs  shm_files  shm_kb   tcp_ports  fds
0        44           450           4      1472       95896    15         2158
1        47           448           40     1506       97000    497        13645
5        58           437           180    1642       101416   2425       59542
10       72           423           355    1812       106936   4835       116873
15       86           409           530    1982       112456   7242       174135
20       100          395           705    2152       117976   9652       231521
25       116          379           880    2322       123496   12062      288906
29       129          366           1019   2458       128704   13991      334836
30       FAIL — Gloo bind: Address already in use
```

### Root cause analysis: CPU busy-spinning in idle workers

Investigation revealed that each sleeping worker spins at **~99% CPU** via `sched_yield()` in
vLLM's SHM message queue polling loop (`shm_broadcast.py:SpinTimer.spin()`). By default,
`VLLM_SLEEP_WHEN_IDLE` is off, so workers never switch to the low-power `time.sleep(0.1)` path.

With 29 engines x 32 workers = 928 CPU-spinning threads:
- New engine startup competes for CPU time (startup grows from 15s to 185s)
- Gloo distributed init slows from 1s to 11s due to scheduling contention
- Any work at `/wake_up` time would face the same contention

### Fix 1: `VLLM_SLEEP_WHEN_IDLE=1` (zero code changes)

**What it does:** Enables `SpinSleepTimer` in vLLM's SHM message queue. After 3s of
inactivity, idle workers call `time.sleep(0.1)` instead of `sched_yield()`, reducing
per-worker CPU from ~99% to ~11%.

**Result: VALIDATED.** Constant 15s startup time through engine 29 (vs 15s→185s without).
Still limited by TCP ephemeral ports at ~58 engines.

### Fix 2: Destroy Gloo on sleep, recreate on wake_up (code change)

**What it does:** After Fix 1 eliminates CPU contention, the binding bottleneck is TCP
ephemeral ports (482 ports per engine from Gloo process groups). Fix 2 destroys the Gloo
distributed environment during `/sleep` and recreates it during `/wake_up`.

**Implementation:** Changes in `worker.py`:
1. `_destroy_gloo()`: new method calling `destroy_model_parallel()` + `destroy_distributed_environment()`
2. `nkipy_sleep()`: calls `_destroy_gloo()` after setting `_sleeping = True`
3. `nkipy_wake_up()`: calls `_init_distributed()` before `dist.broadcast_object_list()`
4. `load_model()`: standby engines (no checkpoint) call `_destroy_gloo()` after init

**Result: VALIDATED — 50 engines started successfully.**

#### Per-engine resource cost with Fix 1 + Fix 2 (TP=32)

| Resource | Per Engine (Fix 2) | Per Engine (Baseline) | Reduction |
|----------|-------------------|-----------------------|-----------|
| Memory | 2.8 GB | 2.9 GB | same |
| Processes | 35 | 35 | same |
| TCP ports | ~1 | ~482 | **99.8%** |
| File descriptors | 2,243 | 11,473 | **80%** |
| SHM files | 33 | 34 | same |

#### Startup time (Fix 1 + Fix 2, sequential starts)

Constant ~15s for all 50 engines (vs 15s→185s without fixes).

#### Full resource consumption data (Fix 1 + Fix 2)

```
engines  mem_used_gb  mem_avail_gb  procs  shm_files  shm_kb   tcp_ports  fds
0        49           446           33     2559       131096   13         4412
1        52           443           68     2592       132196   14         6673
5        64           431           208    2724       136596   18         15643
10       79           416           383    2889       142096   23         26853
15       92           403           558    3054       147596   28         38063
20       105          390           733    3219       153096   33         49318
25       118          377           908    3384       158596   38         60510
30       132          363           1083   3549       164096   43         71690
35       145          349           1258   3714       169596   48         82896
40       159          336           1433   3879       175096   53         94141
45       173          322           1608   4044       180596   58         105410
50       187          308           1783   4209       186096   63         116548
```

#### Updated scalability limits (Fix 1 + Fix 2)

| Bottleneck | Limit | Notes |
|------------|-------|-------|
| **Memory (308 GB avail)** | **~110 engines** | **Primary bottleneck** (2.8 GB per engine) |
| FDs (ulimit -n 1M) | ~400 engines | 2,243 FDs per engine |
| TCP ephemeral ports | ~28,000 engines | ~1 port per sleeping engine |

### Scalability projections

| Configuration | Max Engines | Startup Time | Wake_up Impact |
|---------------|------------|-------------|----------------|
| Baseline (no fixes) | ~58 (TCP limit) | 15s → 185s+ | N/A |
| Fix 1 only | ~58 (TCP limit) | constant ~15s | none |
| Fix 1 + Fix 2 | **~110 (memory limit)** | constant ~15s | +1-2s (Gloo reinit) |

### Step 2: Cross-instance /sleep and /wake_up latency (Fix 1 + Fix 2)

Tested with server engine on Instance A (172.31.44.131, with Qwen3-30B-A3B checkpoint)
and receiver engine on Instance B (172.31.40.200, no checkpoint, standby mode).
All inference outputs verified identical: `"The capital of France is"` → `"Paris. The capital of Germany is Berlin. The"`

#### /wake_up latency (P2P transfer from Instance A → Instance B)

**With non-blocking push (version-0.1.3+):**

| Cycle | gloo_init_s | nrt_init_s | nrt_barrier_s | p2p_transfer_s | tok_embedding_s | total_s |
|-------|------------|-----------|---------------|---------------|----------------|---------|
| 1 | 0.91 | 13.99 | 1.57 | 8.27 | 1.09 | 26.61 |
| 2 | 0.81 | 7.11 | 9.68 | 8.25 | 0.96 | 27.53 |
| 3 | 0.95 | 14.70 | 1.89 | 8.22 | 0.89 | 27.38 |

Gloo reinit adds a consistent **~0.9s** to wake_up latency. The dominant costs are
NRT init (7-15s, varies per cycle) and P2P transfer (~8.2s, consistent).

#### /wake_up latency — other modes

| Mode | gloo_init_s | total_s | Notes |
|------|------------|---------|-------|
| Checkpoint (Instance A, local) | 0.96 | 17.02 | No P2P transfer needed |
| Reverse P2P (B → A) | 0.91 | 24.70 | Bidirectional P2P verified |

#### /sleep latency

**With non-blocking push (version-0.1.3+):**

| Scenario | dereg_wait_s | gloo_destroy_s | spike_reset_s | total_s |
|----------|-------------|---------------|--------------|---------|
| After 60s wait (cycle 1) | 0.00 | 1.32 | 0.15 | 1.95 |
| After 60s wait (cycle 2) | 0.00 | 1.35 | 0.14 | 2.00 |
| After 60s wait (cycle 3) | 0.00 | 1.44 | 0.13 | 1.95 |
| Instance A (server, no RDMA) | 0.00 | 1.49 | N/A | 1.98 |

Gloo destroy adds **~1.3-1.5s** to sleep latency. When RDMA deregistration has completed
in the background (>30s after wake_up), total sleep latency is **~2s**.

### Step 3: Dynamically start or kill receiver engines

Tested with server engine on Instance A (172.31.44.131) and multiple receivers on
Instance B (172.31.40.200).

#### Test 3a: Sequential wake/sleep cycling across 3 receivers

Started 3 sleeping receivers on Instance B (ports 8000, 8001, 8002). Woke each one
sequentially from Instance A via P2P, verified inference, then slept it.

| Receiver | wake_up total_s | Inference output | sleep total_s |
|----------|----------------|-----------------|--------------|
| B:8000 | 27.01 | Paris. The capital of Germany is Berlin. The | 6.43 |
| B:8001 | 29.82 | Paris. The capital of Germany is Berlin. The | 6.55 |
| B:8002 | 31.06 | Paris. The capital of Germany is Berlin. The | 6.56 |

All 3 receivers produce identical outputs. Sleep latency ~6.5s (includes partial dereg wait).

#### Test 3b: Kill sleeping receiver and replace

1. Killed sleeping B:8001 with `kill` (no Neuron cores held — safe)
2. Started new receiver on same port 8001
3. Woke new receiver from A:8000 via P2P — total 28.0s
4. Inference verified: identical output

**Result: PASS.** Sleeping engines can be safely killed and replaced.

#### Test 3c: Kill awake receiver (negative test)

Killing an awake engine with `kill -9` does NOT release Neuron cores. New engines
starting on the same cores fail with `NRT Error NRT_FAILURE(1): Failed to initialize NRT runtime`.

**Mitigation:** Always call `/nkipy/sleep` before killing an engine. Sleep releases
Neuron cores via `spike_reset()`.

#### Test 3d: Full dynamic lifecycle

Sequential test covering the full dynamic lifecycle:
1. Wake B:8000 from A:8000 → 27.2s, inference correct
2. Kill sleeping B:8001 → dead (safe, no cores held)
3. Start replacement B:8001 → ready in ~15s
4. Sleep B:8000 (after 60s dereg wait) → 2.03s
5. Wake replacement B:8001 from A:8000 → 28.8s, inference correct
6. Kill sleeping B:8001, start replacement → wake 28.7s, inference correct

**Result: PASS.** All dynamic operations (start, kill sleeping, replace, cycle) work correctly.

#### Key findings for Step 3

- **Sleeping engines can be safely killed** (no Neuron cores held after Fix 2 Gloo destroy)
- **Awake engines must be slept before killing** (Neuron cores are not released by kill -9)
- **Dynamically started replacements work immediately** (~15s startup, then wake/infer/sleep cycle)
- **All inference outputs are identical** across all receivers regardless of lifecycle

### Step 3e: Non-blocking P2P push — concurrent inference + weight transfer

Tested whether Instance A can serve inference requests with zero stalls while
simultaneously pushing model weights to Instance B via RDMA.

**Setup:**
- A is awake and serving inference (Qwen3-30B-A3B, TP=32)
- B is sleeping (standby mode, no checkpoint)
- Background inference loop sends requests to A every ~1.3s (request + 0.5s gap)
- After 8s baseline, trigger B's `/nkipy/wake_up` with `peer_url=http://172.31.44.131:8000`

**Result: PASS — zero inference stalls during P2P transfer.**

| Phase | Requests | Min (ms) | Max (ms) | Avg (ms) |
|-------|----------|----------|----------|----------|
| Baseline (before push) | 6 | 764 | 787 | 775 |
| During push | 20 | 772 | 829 | 800 |
| After push | 7 | 786 | 808 | 800 |

- B wake_up: 27.2s total (gloo_init 0.9s, nrt_init 5.4s, p2p_transfer 8.2s)
- B inference verified: output identical to A ("Paris. The capital of Germany is Berlin. The")
- No stalls detected (threshold: 1574ms, all requests below 829ms)
- During-push latency increase: ~3% over baseline (within normal variance)

**Key insight:** The `nkipy_push_weights` RPC runs the RDMA push in a background thread
per worker. vLLM's engine core serializes RPCs, but the push RPC returns immediately
(just starts the thread), so inference RPCs (`execute_model`) are not blocked. The
server-side polling loop (`nkipy_push_status`) uses `asyncio.sleep(0.1)` which yields
to the asyncio event loop, allowing inference requests to interleave.

**Previous test failure:** An earlier test hung because the `peer_url` used `localhost`
instead of A's real IP. B's workers resolved `localhost` to B itself, causing B to
POST the push request to its own engine — which was blocked executing `nkipy_wake_up`.
Fix: always use the pusher's real IP in `peer_url`.
