# Peer-to-Peer Weight Transfer

This tutorial shows how to transfer model weights between two inference engines
over RDMA using the `nkipy.p2p` module.  One engine (Engine A) holds the
weights on device; the other (Engine B) starts *sleeping* with no checkpoint
and receives weights directly into device memory via one-sided RDMA writes.

## Overview

```
  Machine 1                          Machine 2
┌──────────────┐   RDMA write   ┌──────────────┐
│  Engine A    │ ─────────────► │  Engine B    │
│  (active)    │   per-rank     │  (sleeping)  │
│  port 8000   │                │  port 8000   │
│  NCs 0-31    │                │  NCs 0-31    │
└──────────────┘                └──────────────┘
```

Each rank on Engine A pushes its weight shard directly into the corresponding
rank's device memory on Engine B.  No CPU staging or filesystem I/O is
involved.

In production, the two engines run on **separate machines** using the **same
NeuronCores** (e.g., both use NCs 0–31).  Single-machine testing with
different core offsets is also supported (see [Single-Machine Testing](#single-machine-testing)).

## Prerequisites

- Two `trn1.32xlarge` instances on the same VPC / placement group
- Pre-sharded model weights on Machine 1 (see `examples/models/*/tensor_preparation.py`)
- UCCL installed (`uccl` Python package) on both machines
- Security group allowing all TCP traffic between the two instances

## Quick Start (Two Machines)

Both machines run the same script (`run_qwen_1.sh`) with the same core
assignment.  The only difference: Machine 1 has `--checkpoint`, Machine 2
does not.

### 1. Start Engine A on Machine 1 (active, with checkpoint)

```bash
cd nkipy/examples/p2p
bash run_qwen_1.sh
```

Which runs:

```bash
WEIGHTS=../models/qwen3/tmp_Qwen3-30b-a3b_TP32
TP=32
export NKIPY_MAX_RDMA_BUFS=1024

torchrun --nproc-per-node $TP --master-port 29501 \
    server.py \
    --checkpoint $WEIGHTS \
    --model Qwen/Qwen3-30B-A3B \
    --arch qwen3 \
    --port 8000 \
    --neuron-port 62239 \
    --core-offset 0 \
    --context-len 64 \
    --max-tokens 256
```

### 2. Start Engine B on Machine 2 (sleeping, no checkpoint)

```bash
cd nkipy/examples/p2p

WEIGHTS=../models/qwen3/tmp_Qwen3-30b-a3b_TP32
TP=32
export NKIPY_MAX_RDMA_BUFS=1024

torchrun --nproc-per-node $TP --master-port 29501 \
    server.py \
    --model Qwen/Qwen3-30B-A3B \
    --arch qwen3 \
    --port 8000 \
    --neuron-port 62239 \
    --core-offset 0 \
    --context-len 64 \
    --max-tokens 256
```

Note: identical to `run_qwen_1.sh` but without `--checkpoint`.  Engine B
compiles kernels to NEFF at startup and waits for `/wake_up`.

### 3. Transfer weights and run inference

From any machine that can reach both:

```bash
MACHINE_A="http://<machine-1-ip>:8000"
MACHINE_B="http://<machine-2-ip>:8000"

# Wake Engine B via P2P from Engine A
curl -s -X POST "$MACHINE_B/wake_up" \
  -H "Content-Type: application/json" \
  -d "{\"peer_url\": \"$MACHINE_A\"}"

# Run inference on Engine B
curl -s -X POST "$MACHINE_B/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

# Put Engine B back to sleep
curl -s -X POST "$MACHINE_B/sleep"
```

## Single-Machine Testing

For development, both engines can run on the same `trn1.32xlarge` by placing
them on different NeuronCore groups.  See `run_qwen_1.sh` / `run_qwen_2.sh`
and `test_p2p_qwen.sh`:

```bash
# Terminal 1 — Engine A (NCs 0-7, port 8000)
bash run_qwen_1.sh

# Terminal 2 — Engine B (NCs 16-23, port 8001, no checkpoint)
bash run_qwen_2.sh

# Terminal 3 — run the test
bash test_p2p_qwen.sh
```

Key differences from two-machine setup:
- Engines use **different `--core-offset`** values (0 vs 16)
- Engines use **different `--port`** and **`--neuron-port`** values
- Engines use **different `--master-port`** values for torchrun
- `NKIPY_MAX_RDMA_BUFS` may need to be lower (e.g., 64) since per-rank
  shards are larger at lower TP

## How It Works

### Transfer flow

1. **Engine B wakes up**: allocates empty device tensors, reloads compiled
   kernels from cache, registers RDMA memory regions.
2. **Engine B requests push**: rank 0 sends an HTTP POST to Engine A's
   `/p2p_push_weights` endpoint with per-rank RDMA metadata.
3. **Engine A pushes**: all ranks perform one-sided RDMA writes directly
   into Engine B's device memory.  No CPU copies on either side.
4. **Engine B receives token embedding**: fetched separately over HTTP
   (small enough that RDMA is unnecessary).

### Chunked registration

EFA has a limit on how much device memory can be registered for RDMA at once.
Dense models (e.g., TinyLlama with ~134 buffers) fit in a single registration.
MoE models (e.g., Qwen3-30B-A3B with 434 buffers) exceed this limit.

The `nkipy.p2p` module automatically chunks registration:

- Buffers are registered in groups of `NKIPY_MAX_RDMA_BUFS` (default: 64)
- The RDMA endpoint and connection are reused across chunks
- Only memory regions are swapped per chunk

### Pre-registration

To avoid registration overhead during transfer, call `preregister_weights()`
after model load or wake-up:

```python
from nkipy.p2p import preregister_weights

# After model tensors are allocated:
preregister_weights(model)
```

This registers all buffers in chunks at startup.  Subsequent transfers skip
registration entirely, reducing transfer latency to just the RDMA write time.

The example server (`examples/p2p/server.py`) calls `preregister_weights()`
automatically after model load and after each wake-up.

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NKIPY_MAX_RDMA_BUFS` | `1024` | Max buffers per RDMA registration chunk. At TP=32 the default fits all buffers in one shot (no chunking). Lower to `64` for TP=8 where per-rank shards are larger. |

### Server flags

| Flag | Description |
|------|-------------|
| `--checkpoint PATH` | Load weights from disk. Omit to start sleeping. |
| `--arch {llama3,qwen3}` | Model architecture. |
| `--core-offset N` | Rank *i* uses NeuronCore `N + i`. Default 0. For single-machine testing, use different offsets to place two engines on different core groups (e.g., 0 and 16). |
| `--port PORT` | HTTP API port. |
| `--neuron-port PORT` | Neuron runtime collective communication port. Must differ between engines. |

## NIC Topology (trn1.32xlarge)

The trn1.32xlarge has 32 NeuronCores and 8 EFA NICs.  Each group of 4
consecutive NCs shares one NIC:

| NeuronCores | NIC |
|-------------|-----|
| 0 – 3       | 0   |
| 4 – 7       | 6   |
| 8 – 11      | 2   |
| 12 – 15     | 4   |
| 16 – 19     | 3   |
| 20 – 23     | 5   |
| 24 – 27     | 1   |
| 28 – 31     | 7   |

With TP=8 on consecutive cores (e.g., NCs 0–7), you get **2 NICs** (NIC 0 and
NIC 6).  With TP=32 (NCs 0–31), all **8 NICs** are used.

The P2P module automatically selects the correct NIC for each rank based on
`NEURON_RT_VISIBLE_CORES`.

## Performance Tips

1. **Use `preregister_weights()`** — eliminates ~2-4s of MR registration
   from the transfer critical path.

2. **Tune `NKIPY_MAX_RDMA_BUFS`** — with smaller per-rank shards (higher TP),
   you can increase this to reduce the number of registration chunks or
   eliminate chunking entirely.

3. **Maximize NIC utilization** — with TP=8, only 2 of 8 NICs are used.
   TP=32 uses all 8 NICs for ~4× aggregate bandwidth.

4. **Use `dereg_async`** — the module automatically deregisters MRs in a
   background thread after transfer, keeping teardown off the critical path.

## API Reference

### High-level (recommended)

- `preregister_weights(model)` — pre-register all weight buffers for RDMA.
- `receive_weights(model, peer_url)` — receive weights from a peer engine.
- `push_weights_to_peer(model, per_rank_info)` — push weights to a peer.
- `fetch_tok_embedding(peer_url)` — fetch token embedding table over HTTP.
- `WeightServer(model)` — tracks buffer metadata for the `/weight_info` endpoint.

### Low-level

- `RankEndpoint` — per-rank RDMA endpoint with `register()`,
  `register_chunked()`, `reregister()`, `dereg_sync()`, `dereg_async()`.
- `push_to_peer(ep, buffers, per_rank_info)` — single push operation.
- `receive_from_peer(ep, buffers, peer_url)` — single receive operation.
- `collect_weight_buffers(model)` — extract `(name, va, size)` tuples from model.
