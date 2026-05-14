# Trn2 Porting Notes

## P2P Transfer: Host-Staged Path

On Trn2, EFA NICs and NeuronCore HBM sit in separate PCIe domains. Direct
device-to-device RDMA only achieves ~3.5 Gbps per rank, while host memory RDMA
reaches ~30 Gbps per rank. To exploit this, the host-staged path
(`NKIPY_HOST_STAGING=1`) routes data through host memory:

```
Sender device (HBM)  ──DMA──▶  Sender host (RAM)  ──RDMA──▶  Receiver host (RAM)  ──DMA──▶  Receiver device (HBM)
       ~13 Gbps                         ~30 Gbps                          ~13 Gbps
```

### Early-Prepare Pre-DMA Optimization

The critical optimization is **early-prepare pre-DMA**: the receiver fires a
lightweight `/nkipy/p2p_prepare` signal to the sender at the very start of the
wake-up flow (before Gloo init, NRT init, and tensor allocation). This gives the
sender ~4 seconds of head start to DMA all weights from device to host staging
buffer. By the time the actual push request arrives, the data is already in host
memory and RDMA can proceed at full speed with zero wait.

Without early-prepare, the sender DMA and receiver setup run sequentially,
adding 2+ seconds to the critical path.

### PCIe Bus Contention Constraint

RDMA and DMA cannot run concurrently on the same host without severe PCIe bus
contention (throughput drops from ~30 Gbps to ~9 Gbps). The implementation waits
for all DMA to complete before starting any RDMA writes.

### Profiled Latency

**Llama-3.1-70B, TP=32, trn2.48xlarge, same-node:**

| Phase | Time |
|-------|------|
| Gloo init | 1.0s |
| NRT init + tensor alloc | 0.2s |
| MR registration (receiver) | 2.7s |
| Sender RDMA (2 chunks, 4.6 GB) | 1.3s |
| Receiver host→device DMA | 2.9s |
| Kernel load + barrier | 0.5–2.0s |
| **Total wake-up** | **9.1–10.3s** |

### Remaining Bottlenecks

Receiver MR registration (2.7s) and receiver host→device DMA (2.9s) are both
hardware-limited and account for ~80% of the P2P transfer time.

## Running P2P on Trn2

Use `--host-staging` with `run_engine.sh`:

```bash
# Sender (Engine A) with checkpoint:
./examples/p2p/run_engine.sh --model /fsx/models/llama-3.1-70b --tp 32 \
    --checkpoint /fsx/models/llama_3.1_70b_TP32 \
    --host-staging --lnc 2 --skip-cte --hf-offline --activate-venv

# Receiver (Engine B) on same node with core offset:
./examples/p2p/run_engine.sh --model /fsx/models/llama-3.1-70b --tp 32 \
    --core-offset 32 --host-staging --lnc 2 \
    --skip-cte --hf-offline --activate-venv --port 8001

# Test the transfer:
./examples/p2p/test_p2p_cycle.sh --model /fsx/models/llama-3.1-70b
```
