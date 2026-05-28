## Build (C++ backend)

```bash
cd relay/src
make all PYTHON=/path/to/.venv/bin/python3
```

## NIXL Backend

The NIXL backend provides direct device-to-device RDMA transfers via NIXL's
LIBFABRIC plugin. On Trn2, this enables 16-rail EFA striping at ~192 Gbps
aggregate without host staging.

### Requirements

| Component | Minimum Version | Notes |
|-----------|----------------|-------|
| NIXL | 1.2.0 | Built from `/fsx/zhuangw/nixl/` with LIBFABRIC backend |
| aws-neuronx-dkms | 2.x.9372.0+ | Kernel driver must support `NEURON_IOCTL_DMABUF_FD` |
| libnrt.so | 2.32.x | The dmabuf bug is patched at runtime — no manual fix needed |
| EFA driver | — | Must support dmabuf (standard on Trn2 instances) |
| Instance type | trn2.48xlarge | Trn1 not supported (no dmabuf in kernel driver) |

### Python dependencies

- `nixl` (the `nixl._api` module — provides `nixl_agent`, `nixl_agent_config`)
- `nkipy` (for `nkipy.vllm_plugin.nrt_dmabuf_fix`)
- `torch`, `torch.distributed` (for multi-rank coordination)
- `requests` (for HTTP metadata exchange with peer)

### Installation

NIXL must be built from source with the LIBFABRIC backend enabled:

```bash
cd /fsx/zhuangw/nixl
pip install -e .
```

Verify the LIBFABRIC plugin is available:

```bash
python -c "from nixl._api import nixl_agent, nixl_agent_config; \
  a = nixl_agent('test', nixl_agent_config(backends=['LIBFABRIC'])); \
  print('OK')"
```

If NIXL was installed from pre-built wheels (e.g. `nixl_cu12`), you may need
the plugin symlink:

```bash
# Only if plugin loading fails:
ln -sf ../plugins/libfabric/libplugin_LIBFABRIC.so \
  $(python -c "import nixl; print(nixl.__path__[0])")/lib/plugins/libplugin_LIBFABRIC.so
```

### Activation

Set the environment variable on both sender and receiver instances:

```bash
export NKIPY_USE_NIXL=1
```

Optional configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `NKIPY_USE_NIXL` | `0` | Set to `1` to use NIXL instead of C++ relay |
| `NKIPY_NIXL_PORT` | `21000` | Base listen port for NIXL agents (per-rank: base + rank) |
| `NEURON_RT_MAP_HBM` | — | Must be `1` for device memory to be mmap-accessible |

### How it works

1. **dmabuf fix**: On agent creation, `nrt_dmabuf_fix.py` hot-patches
   `nrt_get_dmabuf_fd()` in libnrt.so to call the kernel ioctl directly
   (bypasses a bug in libnrt 2.32.x that returns `NRT_INVALID`).

2. **VRAM registration**: Device memory is registered with NIXL as `VRAM`
   type via the LIBFABRIC backend. EFA's dmabuf support enables direct
   NIC access to NeuronCore HBM.

3. **Transfer**: Sender issues RDMA WRITE from its registered VRAM into
   the receiver's registered VRAM. All 16 EFA rails are used automatically
   via NIXL/libfabric striping.

4. **Coordination**: Receiver POSTs its NIXL agent metadata + buffer VAs
   to the sender's `/nkipy/nixl_push` HTTP endpoint. The sender performs
   the RDMA WRITE and returns when complete.

### Architecture (push model)

```
Receiver (waking up)                    Sender (active)
─────────────────────                   ───────────────────
1. register VRAM with NIXL
2. gather metadata + buffer VAs
3. POST /nkipy/nixl_push ──────────►  4. add_remote_agent()
                                        5. RDMA WRITE local→remote
                           ◄────────── 6. HTTP 200 (done)
7. barrier, deregister
```

## Testing

### NIXL backend integration tests

All NIXL tests require two Trn2 instances with shared FSX filesystem and
`NEURON_RT_MAP_HBM=1`. Tests use `torchrun` with Gloo backend for rank
coordination.

#### test_nixl_transfer.py — Basic correctness

Single-buffer push with pattern verification. Run on both nodes simultaneously:

```bash
# Node 0 (sender):
NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
    --node-rank=0 --master_addr=<SENDER_IP> --master_port=29500 \
    relay/tests/test_nixl_transfer.py --nc 0 --size-mb 64 --nixl-port 22000

# Node 1 (receiver):
NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
    --node-rank=1 --master_addr=<SENDER_IP> --master_port=29500 \
    relay/tests/test_nixl_transfer.py --nc 0 --size-mb 64 --nixl-port 22000
```

Options: `--size-mb` (transfer size), `--iterations` (repeat count), `--nc` (NeuronCore).

Expected: `verify=PASS` on all iterations. First iteration ~1.7s (agent setup),
subsequent ~4ms for 64 MB.

#### test_nixl_multi_buffer.py — Multi-layer correctness

Allocates N separate buffers with unique patterns (simulating per-layer model
weights), transfers all in one RDMA WRITE, verifies each independently:

```bash
# Node 0 (sender):
NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
    --node-rank=0 --master_addr=<SENDER_IP> --master_port=29500 \
    relay/tests/test_nixl_multi_buffer.py --nc 0 --num-buffers 32 --buf-size-mb 8

# Node 1 (receiver):
NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
    --node-rank=1 --master_addr=<SENDER_IP> --master_port=29500 \
    relay/tests/test_nixl_multi_buffer.py --nc 0 --num-buffers 32 --buf-size-mb 8
```

Expected: `VERIFY PASS: all 32 buffers correct`.

#### bench_nixl_write.py — Throughput benchmark

Sweeps multiple transfer sizes and reports avg/peak throughput:

```bash
# Node 0 (sender):
NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
    --node-rank=0 --master_addr=<SENDER_IP> --master_port=29500 \
    relay/tests/bench_nixl_write.py --nc 0 --sizes "64,256,1024,4096" --iters 3

# Node 1 (receiver):
NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
    --node-rank=1 --master_addr=<SENDER_IP> --master_port=29500 \
    relay/tests/bench_nixl_write.py --nc 0 --sizes "64,256,1024,4096" --iters 3
```

Reference results (trn2.48xlarge, 16 EFA rails):

| Size | RDMA xfer | Throughput |
|------|-----------|-----------|
| 64 MB | 4 ms | 175 Gbps |
| 256 MB | 13 ms | 21 Gbps |
| 1 GB | 47 ms | 127 Gbps |
| 4 GB | 181 ms | 127 Gbps |

Note: end-to-end latency includes NIXL agent creation (~600ms) per iteration
in the benchmark. In production the agent is persistent — only the RDMA xfer
time matters.

#### One-liner (SSH from sender node)

```bash
# From the sender instance, run both sides:
ssh <RECEIVER_IP> 'source /fsx/zhuangw/nkipy/.venv/bin/activate && \
    NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
    --node-rank=1 --master_addr=<SENDER_IP> --master_port=29500 \
    /fsx/zhuangw/nkipy/relay/tests/test_nixl_transfer.py --nc 0 --size-mb 256' &
sleep 3
source /fsx/zhuangw/nkipy/.venv/bin/activate && \
    NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
    --node-rank=0 --master_addr=<SENDER_IP> --master_port=29500 \
    /fsx/zhuangw/nkipy/relay/tests/test_nixl_transfer.py --nc 0 --size-mb 256
wait
```

### C++ backend tests

#### 2 nodes x 1 rank/node
```
# master node
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=0 --master_addr=<MASTER_IP> tests/benchmark_relay_write_neuron.py --local-nc-idx 5 --device <trn or cpu>

# worker node
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=<MASTER_IP> tests/benchmark_relay_write_neuron.py --local-nc-idx 5 --device <trn or cpu>
```

#### 1 node x 2 ranks/node
```
torchrun --nnodes=1 --nproc_per_node=2 tests/benchmark_relay_write_neuron.py --local-nc-idx-group 5,10 --device <trn or cpu>
```
