#!/usr/bin/env python3
"""Integration test: NIXL multi-buffer transfer (simulating model weights).

Allocates N buffers (simulating per-layer weight tensors) and transfers them
all in a single NIXL RDMA WRITE operation. Verifies data integrity on each
buffer independently.

This tests the production path where a model has hundreds of weight tensors
that are all registered and transferred together.

Usage (2 nodes x 1 rank/node):
    NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
        --node-rank=0 --master_addr=<IP> \
        relay/tests/test_nixl_multi_buffer.py --nc 0 --num-buffers 64

    NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
        --node-rank=1 --master_addr=<IP> \
        relay/tests/test_nixl_multi_buffer.py --nc 0 --num-buffers 64

Usage (1 node x 2 ranks):
    NEURON_RT_MAP_HBM=1 torchrun --nnodes=1 --nproc_per_node=2 \
        relay/tests/test_nixl_multi_buffer.py --nc-group 0,1 --num-buffers 32
"""

import argparse
import ctypes
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("NEURON_RT_MAP_HBM", "1")


def main():
    parser = argparse.ArgumentParser(
        description="NIXL multi-buffer transfer test")
    parser.add_argument("--nc", type=int, default=0)
    parser.add_argument("--nc-group", type=str, default="")
    parser.add_argument("--num-buffers", type=int, default=32,
                        help="Number of separate buffers (simulating layers)")
    parser.add_argument("--buf-size-mb", type=int, default=8,
                        help="Size of each buffer in MB")
    parser.add_argument("--nixl-port", type=int, default=24000)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.nc_group:
        nc_list = [int(x) for x in args.nc_group.split(",")]
        nc_idx = nc_list[local_rank]
    else:
        nc_idx = args.nc

    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{nc_idx}-{nc_idx + 1}"
    os.environ["NKIPY_NIXL_PORT"] = str(args.nixl_port)

    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2

    buf_size = args.buf_size_mb * 1024 * 1024
    total_size = buf_size * args.num_buffers
    total_mb = total_size / (1024 * 1024)

    print(f"[Rank {rank}] {args.num_buffers} buffers x {args.buf_size_mb} MB "
          f"= {total_mb:.0f} MB total, NC={nc_idx}")

    # Allocate individual buffers via spike
    from spike import get_spike_singleton
    spike = get_spike_singleton()

    tensors = []
    buffers = []
    for i in range(args.num_buffers):
        t = spike.allocate_tensor(buf_size, 0, f"layer{i}_rank{rank}")
        tensors.append(t)
        buffers.append((f"layer_{i}", t.va, buf_size))

    print(f"[Rank {rank}] Allocated {args.num_buffers} tensors "
          f"(first VA=0x{tensors[0].va:x})")

    # Sender writes a unique pattern per buffer for verification
    if rank == 0:
        for i, t in enumerate(tensors):
            pattern = (i + 1) & 0xFF  # 1, 2, 3, ..., wrap at 255
            ctypes.memset(t.va, pattern, min(4096, buf_size))
        print(f"[Rank {rank}] Wrote per-buffer verification patterns")

    dist.barrier()

    # --- Transfer ---
    from relay.nixl_endpoint import NixlEndpoint
    from relay.nixl_transfer import push_to_peer

    ep = NixlEndpoint(nc_idx=0)

    t0 = time.time()

    if rank == 0:
        # Sender
        ep.register(buffers)
        t_reg = time.time()

        obj_list = [None]
        dist.broadcast_object_list(obj_list, src=1)
        receiver_info = obj_list[0]

        per_rank_info = [receiver_info]
        push_to_peer(ep, buffers, per_rank_info)
        t_xfer = time.time()

    else:
        # Receiver: clear all buffers first
        for t in tensors:
            ctypes.memset(t.va, 0x00, min(4096, buf_size))

        ep.register(buffers)
        t_reg = time.time()

        local_info = {
            "agent_metadata": ep.get_metadata().hex(),
            "agent_name": f"nkipy_rank{rank}",
            "buffer_vas": [(t.va, buf_size) for t in tensors],
        }
        dist.broadcast_object_list([local_info], src=1)
        t_xfer = time.time()

    dist.barrier()
    t_done = time.time()

    elapsed = t_done - t0
    throughput_gbps = (total_size * 8) / elapsed / 1e9

    if rank == 0:
        print(f"[Rank {rank}] Push complete: reg {(t_reg-t0)*1000:.0f} ms, "
              f"xfer {(t_xfer-t_reg)*1000:.0f} ms, "
              f"total {elapsed*1000:.0f} ms ({throughput_gbps:.1f} Gbps)")

    # Verify data on receiver
    if rank == 1:
        errors = 0
        for i, t in enumerate(tensors):
            expected = (i + 1) & 0xFF
            buf = (ctypes.c_ubyte * 64).from_address(t.va)
            actual = bytes(buf[:64])
            if not all(b == expected for b in actual):
                print(f"  FAIL buffer {i}: expected 0x{expected:02x}, "
                      f"got {actual[:8].hex()}")
                errors += 1

        if errors == 0:
            print(f"[Rank {rank}] VERIFY PASS: all {args.num_buffers} buffers correct")
            print(f"[Rank {rank}] Total: {elapsed*1000:.0f} ms, "
                  f"{throughput_gbps:.1f} Gbps")
        else:
            print(f"[Rank {rank}] VERIFY FAIL: {errors}/{args.num_buffers} buffers wrong")
            sys.exit(1)

    # Cleanup
    ep.deregister_sync()
    for t in tensors:
        spike.free_tensor(t)
    dist.destroy_process_group()
    print(f"[Rank {rank}] DONE")


if __name__ == "__main__":
    main()
