# NIXL Direct Device RDMA — Wake/Sleep Latency

Cross-instance P2P weight transfer using NIXL LIBFABRIC backend on trn2.48xlarge.

## Setup

- **Hardware**: 2× trn2.48xlarge (16 EFA rails each, 64 NeuronCores)
- **Backend**: NIXL 1.2.0, LIBFABRIC transport, direct device-to-device RDMA (no host staging)
- **Tensor parallelism**: TP=32
- **Transfer model**: Sender-initiated RDMA WRITE (push)
- **Registration**: Single contiguous span per rank, epoch-tagged agents for scalability

## Results

| Model | Per-Rank Shard | Wake (cold) | Wake (warm) | P2P (warm) | Sleep |
|-------|---------------|-------------|-------------|------------|-------|
| Qwen3-30B-A3B | 1.8 GB | 6.18s | 4.60s | 2.92s | 2.71s |
| LLaMA-3.1-70B | 4.2 GB | 6.22s | 5.03s | 3.26s | 2.69s |
| Qwen3-235B-A22B | 14.3 GB | 8.34s | 7.06s | 5.06s | 2.66s |

"Cold" = first iteration (includes NIXL registration + descriptor setup).
"Warm" = best of iterations 2-3 (registration reused within epoch).

## Average Wake Breakdown (3 iterations)

| Phase | Qwen3-30B | LLaMA-70B | Qwen3-235B |
|-------|-----------|-----------|------------|
| Gloo init | 0.93s | 0.98s | 0.96s |
| NRT init | 0.30s | 0.32s | 0.33s |
| Alloc tensors | 0.06s | 0.11s | 0.15s |
| **P2P transfer** | **3.43s** | **3.73s** | **5.49s** |
| Kernel load | 0.31s | 0.17s | 0.36s |
| Tok embedding | 0.03s | 0.12s | 0.07s |
| **Total** | **5.17s** | **5.52s** | **7.51s** |

## Effective Throughput

| Model | Per-Rank Gbps (warm) | Aggregate (32 ranks) |
|-------|---------------------|---------------------|
| Qwen3-30B-A3B | 4.9 Gbps | 157 Gbps |
| LLaMA-3.1-70B | 10.3 Gbps | 330 Gbps |
| Qwen3-235B-A22B | 22.6 Gbps | 723 Gbps |

Throughput increases with shard size due to amortization of per-transfer fixed costs
(agent metadata exchange, descriptor setup). The 14.3 GB Qwen3-235B shard saturates
more of the 16 EFA rails per rank.

## Sleep Breakdown

Sleep latency is model-size-independent (~2.5-2.7s) and dominated by:
- NIXL agent destroy + deregister: ~0.3s
- spike_reset (NRT close): ~1.5-2.0s
- Gloo destroy: ~0.2s

## Design: Epoch-Tagged Agents

Each sleep/wake cycle increments the receiver's epoch counter. The NIXL agent name
includes the epoch (e.g. `nkipy_host_r0_e3`), so the sender treats each wake as a
new remote agent. This avoids the `remove_remote_agent` limitation (NIXL_ERR_NOT_ALLOWED
during active progress thread) and ensures VRAM is fully released on sleep for
multi-tenant scalability.

Old remote agent entries on the sender side are metadata-only (~KB each) and do not
pin device memory.
