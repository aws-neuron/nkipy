# Trn2 Cross-Instance Transfer: Architecture Analysis

## Conclusion

**All cross-instance data transfer on Trn2 goes through host memory.** This includes
CCOM AllReduce, our relay RDMA, and the experimental `nrt_async_sendrecv` API. There
is no functional path for EFA to directly access NeuronCore HBM on the current
system (driver 2.x.8963.0, collectives 2.x.53613.0, libfabric 2.4.0amzn3.0).

## Evidence

### 1. EFA cannot register Neuron device memory

libfabric's EFA provider probes FI_HMEM_NEURON support at initialization time.
On this system it fails:

```
core:neuron_get_dmabuf_fd():246  Failed to retrieve dmabuf_fd: 2
efa:mr:efa_hmem_info_check_p2p_support_neuron():308  Unable to retrieve dmabuf fd of Neuron device buffer, Fall back to ibv_reg_mr
efa:core:efa_hmem_info_check_p2p_support_neuron():320  Failed to register Neuron buffer with the EFA device
efa:core:efa_hmem_info_init_iface():389  FI_HMEM_NEURON P2P support is not available.
efa:core:efa_prov_info_direct_set_hmem_flags():424  HMEM support will be disabled.
```

This message appears during **both** `nrt_async_sendrecv_init` and `nrt_build_global_comm`
(the AllReduce setup path). Both paths see the same disabled HMEM.

### 2. CCOM AllReduce uses a host-memory network proxy

The NRT library implements inter-node collective operations via:
- `neuronStartNetworkProxy` — starts a CPU thread that drives EFA send/recv
- `enc_network_proxy_task` — orchestrates DMA ↔ EFA pipelining
- `alloc_net_connector(..., enc_host_mem_shared_t*)` — allocates host-pinned buffers
- `mesh->channel.net_send->dynamic_input_host_mem` — confirms data flows through host

The path is: device HBM → DMA → pinned host buffer → EFA → remote host → DMA → remote HBM.

CCOM achieves 1.6 Tbps through deep pipelining:
- Hardware semaphore-driven DMA (TOPSP orchestration)
- Multiple in-flight chunks per EFA rail
- All 16 EFA rails used simultaneously
- Pipeline orchestrated by firmware, not software polling

### 3. `nrt_async_sendrecv` is broken on this system

The experimental API requires FI_HMEM_NEURON for its flush buffer registration.
Even a 4-byte host-allocated buffer gets passed through the HMEM path (type=4,
device=10) which fails because HMEM is disabled:

```
ENC:async_sr_alloc_and_register_flush_mem  [host_lnc 0] failed to register flush buffer
```

Required env vars discovered:
- `NEURON_RT_ASYNC_SENDRECV_EXPERIMENTAL_ENABLED=1`
- `NEURON_RT_ASYNC_SENDRECV_BOOTSTRAP_PORT=<port>`
- `NEURON_RT_ROOT_COMM_ID=<ip>:<port>` (bootstrap rendezvous)
- `LD_LIBRARY_PATH` must include `/opt/aws/neuron/lib` (for dlopen of libnccom.so)

### 4. Direct device RDMA via ibv_reg_mr also fails

Our relay's `ep.reg()` for NeuronCore HBM pointers returns "relay_regmr called with
null data" — EFA's ibv_reg_mr cannot pin neuron device memory.

### 5. Root cause: neuron dmabuf → EFA integration not functional

The neuron kernel module (2.x.8963.0) has dmabuf support code:
- `neuron_p2p_register_va`
- `neuron_p2p_register_and_get_pa`
- `ncdev_get_dmabuf_fd`

But libfabric's `neuron_get_dmabuf_fd()` returns error code 2, suggesting the
ioctl or device file interface isn't compatible with this EFA driver version.

## What this means for optimization

Since host-staged transfer is the only viable path, and CCOM proves it can reach
1.6 Tbps on this hardware, the optimization target is clear: **improve our pipeline
efficiency to approach CCOM's pipelining**.

Current Qwen3-235B-A22B (TP=32, 14.9 GB/rank) transfer breakdown:
- Sender DMA (device→host): ~8s sequential
- RDMA (host→remote host): ~2-3s
- Receiver DMA (host→device): ~8s sequential
- Total: ~18-19s

With optimal pipelining (DMA and RDMA overlapped across chunks), the theoretical
minimum approaches max(DMA_time, RDMA_time) ≈ 8s, since DMA is the bottleneck.

## Optimization: batch_dma_read / batch_dma_write

**Change:** Replaced sequential per-tensor `spike.tensor_read_to_pybuffer()` /
`spike.tensor_write_from_pybuffer()` with `spike.batch_dma_read()` /
`spike.batch_dma_write()` in all three host-staged transfer functions:
- `start_pre_dma_to_staging` (sender background DMA)
- `push_weights_to_peer_staged` (sender pipelined fallback path)
- `receive_from_peer_staged` (receiver DMA)

**How it works:** `batch_dma_read`/`batch_dma_write` accept a list of
`(tensor_ref, buffer)` tuples, releases the GIL, and dispatches them across
8 C++ threads. Each thread handles a slice of the tensor list, calling
`nrt_tensor_read`/`nrt_tensor_write` concurrently.

**Expected improvement:** 8s sequential DMA → 2-3s parallel DMA per side.
Total wake-up: 19.9s → ~8-10s.

**Trn1 safety:** These changes only affect the `NKIPY_HOST_STAGING=1` code path
(default on Trn2). Trn1 uses direct device RDMA (`NKIPY_HOST_STAGING=0`) which
bypasses staging entirely — no DMA calls at all.

## System versions

| Component | Version |
|-----------|---------|
| Neuron driver (DKMS) | 2.x.8963.0 |
| Neuron runtime | 2.x.51731.0 |
| Neuron collectives | 2.x.53613.0 |
| libfabric | 2.4.0amzn3.0 |
| EFA driver | 3.0.0-1.amzn1 |
| Kernel | 6.8.0-1055-aws |
| Instance | trn2.48xlarge (16 devices, 64 NCs, 16 EFA NICs) |
