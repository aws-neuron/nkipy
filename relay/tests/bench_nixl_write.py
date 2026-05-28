#!/usr/bin/env python3
"""Benchmark: NIXL RDMA WRITE throughput via relay's NIXL backend.

Measures throughput for direct device-to-device RDMA WRITE using the
NixlEndpoint / push_to_peer path (same as production weight transfer).

Tests multiple buffer sizes and reports per-size throughput. Uses torchrun
for multi-rank coordination (Gloo backend for metadata exchange).

Usage (2 nodes x 1 rank/node):
    # Node 0 (sender):
    NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
        --node-rank=0 --master_addr=<IP> \
        relay/tests/bench_nixl_write.py --nc 0

    # Node 1 (receiver):
    NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
        --node-rank=1 --master_addr=<IP> \
        relay/tests/bench_nixl_write.py --nc 0

Usage (1 node x 2 ranks):
    NEURON_RT_MAP_HBM=1 torchrun --nnodes=1 --nproc_per_node=2 \
        relay/tests/bench_nixl_write.py --nc-group 0,1
"""

import argparse
import ctypes
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("NEURON_RT_MAP_HBM", "1")


def _pretty(num: int) -> str:
    units, val = ["B", "KB", "MB", "GB"], float(num)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024


def main():
    parser = argparse.ArgumentParser(description="NIXL WRITE throughput benchmark")
    parser.add_argument("--nc", type=int, default=0)
    parser.add_argument("--nc-group", type=str, default="")
    parser.add_argument("--sizes", type=str,
                        default="1,4,16,64,256,1024,4096,14848",
                        help="Comma-separated transfer sizes in MB")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--nixl-port", type=int, default=23000)
    args = parser.parse_args()

    sizes_mb = [int(x) for x in args.sizes.split(",")]

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

    from spike import get_spike_singleton
    spike = get_spike_singleton()

    max_size = max(sizes_mb) * 1024 * 1024
    tensor = spike.allocate_tensor(max_size, 0, f"bench_rank{rank}")
    print(f"[Rank {rank}] Allocated {max(sizes_mb)} MB on NC {nc_idx}, "
          f"VA=0x{tensor.va:x}")

    from relay.nixl_endpoint import NixlEndpoint
    from relay.nixl_transfer import push_to_peer, _poll_xfer

    print(f"\n{'Size':>10} | {'Latency (ms)':>12} | {'Throughput':>12} | "
          f"{'Per-iter':>10}")
    print("-" * 60)

    for size_mb in sizes_mb:
        size = size_mb * 1024 * 1024
        buffers = [("weight", tensor.va, size)]

        latencies = []
        for i in range(args.iters):
            ep = NixlEndpoint(nc_idx=0)
            dist.barrier()

            t0 = time.time()

            if rank == 0:
                # Sender
                ep.register(buffers)
                obj_list = [None]
                dist.broadcast_object_list(obj_list, src=1)
                receiver_info = obj_list[0]

                per_rank_info = [receiver_info]
                push_to_peer(ep, buffers, per_rank_info)
            else:
                # Receiver
                ep.register(buffers)
                local_info = {
                    "agent_metadata": ep.get_metadata().hex(),
                    "agent_name": f"nkipy_rank{rank}",
                    "buffer_vas": [(tensor.va, size)],
                }
                dist.broadcast_object_list([local_info], src=1)

            dist.barrier()
            elapsed = time.time() - t0
            latencies.append(elapsed)

            ep.deregister_sync()

        avg = sum(latencies) / len(latencies)
        best = min(latencies)
        avg_gbps = (size * 8) / avg / 1e9
        peak_gbps = (size * 8) / best / 1e9

        if rank == 0:
            print(f"{_pretty(size):>10} | {avg*1000:>9.1f} ms | "
                  f"{avg_gbps:>8.1f} Gbps | {peak_gbps:.1f} Gbps peak")

    # Cleanup
    spike.free_tensor(tensor)
    dist.destroy_process_group()
    if rank == 0:
        print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
