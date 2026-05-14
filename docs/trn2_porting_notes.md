# Trn2 Porting Notes

## Overview

This document tracks issues encountered and fixes applied while porting NKIPy
from Trn1 (trn1.32xlarge) to Trn2 (trn2.48xlarge).

**Cluster**: Trn2 cluster with head node (r7i.24xlarge) and compute nodes
(trn2.48xlarge), managed via Slurm.

**Trn2 hardware**: 16 Neuron devices, 4 cores each = 64 physical NeuronCores.
With `logical-neuroncore-config: 2`, there are 32 Logical NeuronCores (LNC 0-31).
16 EFA NICs for RDMA. 2 NUMA nodes (devices 4-11 on NUMA 0, devices 0-3 and
12-15 on NUMA 1).

---

## Issue 1: `_relay` C++ extension fails to build (missing glog/gflags)

**Symptom**: `_relay` is `None` causing `AttributeError: 'NoneType' object has
no attribute 'Endpoint'`

**Root cause**: The head node doesn't have `libgoogle-glog-dev` or
`libgflags-dev` installed, and no sudo access to install them.

**Fix**:
- Built glog v0.6.0 and gflags v2.2.2 from source into `relay/third_party/install/`
- Updated `relay/src/Makefile` to detect and use `third_party/install/` headers/libs
- Used `patchelf` to set rpath on `libglog.so` for transitive gflags dependency
- Added `relay/third_party/` to `.gitignore`

**Notes**: If glog/gflags are installed system-wide (e.g., on an AMI with the
packages), the Makefile automatically uses them and ignores third_party.

---

## Issue 2: No NIC-to-NeuronCore mapping for trn2.48xlarge

**Symptom**: `CHECK(!selected_dev_indices.empty())` assertion failure because
`load_gpu_nic_map()` returns empty vector for unknown instance types.

**Root cause**: `relay/src/include/util.h` only had hardcoded NIC mappings for
`p5.48xlarge`, `p5en.48xlarge`, and `trn1.32xlarge`.

**Fix**: Added `discover_nic_map_from_numa()` in `relay/src/include/rdma_device.h`
that dynamically discovers NIC-to-core affinity via NUMA topology in sysfs. Falls
back to all NICs if no NUMA match. Caps to `kNICContextNumber` (4) to satisfy
engine constraints.

---

## Issue 3: NRT tensor `copy_()` fails on Trn2

**Symptom**: `RuntimeError: nrt_tensor_read status=1` in `_make_buffer` when
copying from CPU tensor to NRT device tensor.

**Root cause**: On Trn2, `_relay.create_nrt_tensor()` allocates device memory
correctly, but the subsequent `buf_nrt.copy_(buf)` fails because the NRT tensor
allocated via the C API isn't properly integrated with torch_neuronx's device
context.

**Fix**: Modified `_make_buffer()` in `benchmark_relay_write_neuron.py` to skip
the `copy_()` call for `device="trn"` — the NRT tensor is used directly for
RDMA registration without needing to copy data into it (benchmark uses it as a
target buffer).

---

## Issue 4: `NEURON_RT_VISIBLE_CORES` must be set before `import torch_neuronx`

**Symptom**: torch_neuronx's PJRT layer overrides core assignment using
`NEURON_PJRT_PROCESS_INDEX` if `NEURON_RT_VISIBLE_CORES` isn't set before import.

**Fix**: Moved `os.environ["NEURON_RT_VISIBLE_CORES"] = ...` before
`import torch_neuronx` in both `benchmark_relay_write_neuron.py` and `worker.py`.

---

## Issue 5: Dead `UCCL_RCMODE` environment variable

**Symptom**: None (no functional impact). The env var was set but never read by
any library (EFA, NRT, or Python packages).

**Fix**: Removed entirely from `relay/src/relay/endpoint.py` and
`relay/tests/benchmark_relay_write_neuron.py`. Confirmed benchmark still works.

---

## Issue 6: NRT_TIMEOUT in vLLM engine startup with TP=32

**Symptom**: `spike._spike.NrtError: NRT Error NRT_TIMEOUT(5): Failed to
initialize NRT runtime` — workers time out during `nrt_init()`.

**Root cause**: `import torch_neuronx` (in `platform.py`) calls
`xla.set_rt_root_comm_id()` which sets `NEURON_RT_ROOT_COMM_ID`. This env var is
inherited by spawned worker processes, causing spike's `nrt_init()` to attempt a
collective initialization across all 32 workers. Since vLLM spawns workers via
multiprocessing (not torchrun), they don't start simultaneously, causing timeout.

**Fix** (in progress, untested due to cluster unavailability):
1. `platform.py`: Set `NEURON_PJRT_WORLD_SIZE=1` before `import torch_neuronx`,
   then clear `NEURON_RT_ROOT_COMM_ID` after import.
2. `worker.py`: Remove `import torch_neuronx` (not needed — platform.py handles
   registration). Clear `NEURON_RT_ROOT_COMM_ID` before spike's `nrt_init()`.

---

## RDMA Benchmark Results (Trn2, intra-node)

Tested with `benchmark_relay_write_neuron.py`, NCs 5 and 10, single node:

### CPU buffers (`--device cpu`)
| Size    | Throughput | Bandwidth | Latency  |
|---------|-----------|-----------|----------|
| 256 B   | 0.16 Gbps | 0.02 GB/s | 13 us   |
| 1 KB    | 0.71 Gbps | 0.09 GB/s | 12 us   |
| 4 KB    | 2.89 Gbps | 0.36 GB/s | 11 us   |
| 16 KB   | 11.33 Gbps| 1.42 GB/s | 12 us   |
| 64 KB   | 27.64 Gbps| 3.45 GB/s | 19 us   |
| 256 KB  | 50.39 Gbps| 6.30 GB/s | 42 us   |
| 1 MB    | 138.63 Gbps| 17.33 GB/s| 61 us   |
| 10 MB   | 108.38 Gbps| 13.55 GB/s| 774 us  |
| 64 MB   | 106.87 Gbps| 13.36 GB/s| 5.0 ms  |
| 100 MB  | 95.33 Gbps| 11.92 GB/s| 8.8 ms  |

### NRT device buffers (`--device trn`)
| Size    | Throughput | Bandwidth | Latency  |
|---------|-----------|-----------|----------|
| 256 B   | 0.14 Gbps | 0.02 GB/s | 15 us   |
| 1 KB    | 0.63 Gbps | 0.08 GB/s | 13 us   |
| 4 KB    | 2.58 Gbps | 0.32 GB/s | 13 us   |
| 16 KB   | 9.34 Gbps | 1.17 GB/s | 14 us   |
| 64 KB   | 17.00 Gbps| 2.12 GB/s | 31 us   |
| 256 KB  | 20.35 Gbps| 2.54 GB/s | 103 us  |
| 1 MB    | 12.61 Gbps| 1.58 GB/s | 665 us  |
| 10 MB   | 9.28 Gbps | 1.16 GB/s | 9.0 ms  |
| 64 MB   | 19.31 Gbps| 2.41 GB/s | 27.8 ms |
| 100 MB  | 19.56 Gbps| 2.44 GB/s | 42.9 ms |

**Note**: NRT device buffer throughput is lower because HBM is not mapped to
user-space on Trn2 (unlike Trn1 with `NEURON_RT_MAP_HBM=1`). RDMA reads/writes
to device memory go through DMA rather than direct MMIO.

---

## P2P Weight Transfer

### Status: Engine startup validated, push protocol has timing issue

Both engines start successfully on Trn2 (TP=32, exclusive node):
- Engine A (sender): loads sharded checkpoint, serves on port 8000
- Engine B (receiver): starts in sleep mode on port 8001
- RDMA preconnect between engines succeeds (200 OK)

**Remaining issue**: `remote_descs=0 vs xfer_descs=483` — the push protocol
has a timing race where Engine A attempts RDMA push before Engine B finishes
MR registration. This is NOT Trn2-specific — it's a protocol-level issue in
the chunked push flow that may be masked by different timing on Trn1.

### Issue 7: Endpoint URL mismatch in wake_up flow

**Symptom**: `404 Not Found for url: http://localhost:8000/p2p_push_weights`

**Root cause**: `receive_from_peer()` defaults to `/p2p_push_weights` and
`/p2p_preconnect` endpoints, but the server mounts them at
`/nkipy/p2p_push_weights` and `/nkipy/p2p_preconnect`.

**Fix**: Pass explicit endpoints in the `receive_from_peer` call in
`worker.py`'s `nkipy_wake_up`.

### Issue 8: ibv_reg_mr fails for NeuronCore HBM addresses

**Symptom**: `Error: ibv_reg_mr failed for data at 0x7056bc000000 size 8388608
context_id 0 errno=12 (Cannot allocate memory)`

**Root cause**: On Trn2, NeuronCore HBM requires explicit opt-in for RDMA MR
registration (unlike Trn1 where `NEURON_RT_MAP_HBM=1` handled this).

**Fix**: Set `NEURON_RT_REGISTER_HBM_MR=1` before NRT init. Added to
`platform.py` via `os.environ.setdefault("NEURON_RT_REGISTER_HBM_MR", "1")`.

Additionally, the number of simultaneous MR registrations must be reduced on
Trn2 — registering 128+ buffers per rank simultaneously exceeds EFA's
capacity. Set `NKIPY_MAX_RDMA_BUFS=32` to use smaller chunks.

### Issue 9: Chunked transfer protocol error on chunk 2+ (OPEN)

**Symptom**: After chunk 1 transfers successfully, chunk 2 fails with
`CQE error, status=9 (remote invalid request error)`.

**Root cause**: The sender attempts to write using MR keys from chunk 1 after
the receiver has deregistered them and registered new MRs for chunk 2. The
re-registration/key exchange between chunks isn't synchronized correctly in
the intra-node case.

**Status**: Chunk 1 works (295.86 MB at 6.83 Gbps per rank). The multi-chunk
protocol needs debugging — likely a race in the receiver's `reregister` +
sender's key refresh sequence.

## P2P Weight Transfer Results

**Model**: LLaMA-3.1-70B-Instruct, TP=32, bf16
**Setup**: 2 exclusive trn2.48xlarge nodes (cross-node RDMA)

### /wake_up latency (30.3s total, with host staging + batch DMA)

| Phase | Time |
|-------|------|
| Gloo init | 1.61s |
| NRT init | 7.98s |
| NRT barrier | 9.23s |
| Alloc tensors | 0.09s |
| **P2P transfer (staged)** | **11.02s** |
| Kernel load | 0.16s |
| Tok embedding | 0.12s |
| **Total** | **30.27s** |

P2P transfer breakdown (per rank 0):
- Sender DMA (device→host, 8 threads): 2.68s
- Sender MR registration: 0.0s (pre-registered at model load)
- RDMA (host→remote_host): **0.62s at 60.2 Gbps**
- Receiver MR registration: 2.68s
- Receiver DMA (host→device, 8 threads): 2.67s

### /sleep latency (2.4s total, with 60s post-wake delay)

| Phase | Time |
|-------|------|
| GC collect | 0.50s |
| Dereg wait | 0.0s |
| Spike reset | 0.44s |
| Gloo destroy | 1.49s |
| **Total** | **2.43s** |

**Note**: `/sleep` must be called at least ~40s after `/wake_up` to allow
background MR deregistration to complete. If called immediately, `dereg_wait`
adds ~41s.

### Key latency differences vs Trn1

| Phase | Trn2 | Trn1 | Cause |
|-------|------|------|-------|
| NRT init + barrier | 17.2s | <1s | Collective NRT handshake with `NEURON_RT_ROOT_COMM_ID` |
| P2P transfer | 11.0s | ~3s | DMA overhead (device↔host, ~5.4s) dominates; RDMA fast |
| RDMA throughput | 60.2 Gbps/rank | ~95 Gbps/rank | Host-staged, limited by NIC sharing |
| /sleep | 2.3s | ~2s | Comparable (with delay) |

### Host-Staged P2P Architecture

On Trn2, EFA NICs and NeuronCore HBM sit in separate PCIe domains. Direct
device-to-device RDMA only achieves ~3.5 Gbps per rank, while host memory RDMA
reaches ~30 Gbps per rank. The host-staged path (`NKIPY_HOST_STAGING=1`) routes
data through host memory:

```
Sender device (HBM)  ──DMA──▶  Sender host (RAM)  ──RDMA──▶  Receiver host (RAM)  ──DMA──▶  Receiver device (HBM)
       ~13 Gbps                         ~30 Gbps                          ~13 Gbps
```

### Optimizations applied

1. **Host-staged RDMA** (`NKIPY_HOST_STAGING=1`): Routes transfers through
   pinned host memory — RDMA achieves 60 Gbps/rank (vs 3.5 Gbps direct device)
2. **Batch parallel DMA** (`spike.batch_dma_read/write`): 8-thread parallel
   NRT DMA copies reduce device↔host from 5.5s to 2.7s (2x speedup)
3. **Sender MR pre-registration**: Host staging buffer registered at model
   load time, saving 1.0s per transfer
4. **NIC rotation**: All 16 EFA NICs utilized across 32 workers (vs 4 before)
5. **Early-prepare pre-DMA**: The receiver fires a lightweight
   `/nkipy/p2p_prepare` signal to the sender at the very start of the wake-up
   flow (before Gloo init, NRT init, and tensor allocation). This gives the
   sender ~4 seconds of head start to DMA all weights from device to host
   staging buffer. By the time the actual push request arrives, the data is
   already in host memory and RDMA can proceed at full speed with zero wait.
   Without early-prepare, the sender DMA and receiver setup run sequentially,
   adding 2+ seconds to the critical path.

**PCIe bus contention constraint**: RDMA and DMA cannot run concurrently on
the same host without severe PCIe bus contention (throughput drops from ~30
Gbps to ~9 Gbps). The implementation waits for all DMA to complete before
starting any RDMA writes.

### /wake_up latency with early-prepare pre-DMA (9.1–10.3s total)

Same setup as above, with early-prepare optimization enabled:

| Phase | Time |
|-------|------|
| Gloo init | 1.0s |
| NRT init + tensor alloc | 0.2s |
| MR registration (receiver) | 2.7s |
| Sender RDMA (2 chunks, 4.6 GB) | 1.3s |
| Receiver host→device DMA | 2.9s |
| Kernel load + barrier | 0.5–2.0s |
| **Total wake-up** | **9.1–10.3s** |

The NRT init time dropped from 17.2s to 0.2s by fixing the
`NEURON_RT_ROOT_COMM_ID` collective issue (see Issue 6). Combined with
early-prepare pre-DMA, total wake-up went from 30.3s to 9.1–10.3s.

### Remaining optimization opportunities

1. **NRT collective init (17.2s)**: `nrt_init()` with `NEURON_RT_ROOT_COMM_ID`
   requires all 32 workers to synchronize. On Trn1 this is fast (<1s). On Trn2
   it takes 7-14s. May be NRT version or CCOM topology discovery overhead.

2. **Receiver MR pre-registration (2.7s)**: Currently blocked by endpoint
   lifecycle conflicts. Needs careful integration with `dereg_sync`.

3. **DMA pipeline**: Overlap sender DMA with receiver DMA from previous chunk
   via double-buffered protocol.

### Configuration

```bash
NEURON_RT_REGISTER_HBM_MR=1  # Required for ibv_reg_mr on Trn2 HBM
NKIPY_HOST_STAGING=1          # Host-staged RDMA (default on Trn2)
```

### Commands

```bash
# Engine A (sender, cores 0-31, port 8000):
./examples/p2p/run_engine.sh --model /shared/zhuangw/models/llama-3.1-70b-instruct \
    --tp 32 --checkpoint /shared/zhuangw/models/llama-3.1-70b-instruct_TP32 \
    --skip-cte --hf-offline --activate-venv

# Engine B (receiver, cores 32-63, port 8001):
./examples/p2p/run_engine.sh --model /shared/zhuangw/models/llama-3.1-70b-instruct \
    --tp 32 --core-offset 32 --port 8001 \
    --skip-cte --hf-offline --activate-venv

# Test:
./examples/p2p/test_p2p_full.sh \
    --model /shared/zhuangw/models/llama-3.1-70b-instruct \
    --engine-a http://localhost:8000 --engine-b http://localhost:8001
```

Metrics to capture:
- `/nkipy/wake_up` total latency (P2P transfer + NRT init + gloo)
- `/nkipy/sleep` total latency
