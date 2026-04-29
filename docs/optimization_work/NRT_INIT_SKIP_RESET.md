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

## Impact on Wake-Up Latency

For any wake-up after the first sleep cycle on the machine (all standby engines benefit):

| Phase | Before | After | Speedup |
|-------|--------|-------|---------|
| `nrt_init` | 5–15s | 0.15–0.17s | **35–100×** |
| Total wake-up (TP=32, P2P) | 28–31s | 9.5–12.5s | **~2.5–3×** |
| Total wake-up (TP=8, same-node) | 16.2s | 4.0s | **~4×** |

The remaining wake-up time is dominated by P2P transfer (~7.7–10.4s for TP=32 cross-node) and Gloo init (~0.9s).

## Safety

- `reset_cores=0` is only applied when the marker file exists, proving a prior `nrt_close()` ran on this machine.
- Cold starts (fresh machine, no marker) always use the default `reset_cores=1`.
- HBM scrub still runs, so device memory is zeroed before reuse.
- Cross-model interleaving is safe: cores that previously ran LLaMA can immediately run Qwen3 without firmware reset.

## Files Changed

- `nkipy/src/nkipy/vllm_plugin/worker.py` — Sleep: write `/tmp/nkipy_nrt_closed` marker after `spike_reset()`. Wake: read marker and set `NEURON_RT_RESET_CORES=0` before `get_spike_singleton()`.

## Date

2026-04-29
