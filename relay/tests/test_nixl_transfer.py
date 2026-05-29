#!/usr/bin/env python3
"""Integration test: NIXL backend P2P weight transfer via relay.

Tests the full push_to_peer flow using Endpoint with device memory on
two ranks (torchrun with 2 processes on one or two nodes).

Usage (single node, 2 NeuronCores):
    NEURON_RT_MAP_HBM=1 torchrun --nnodes=1 --nproc_per_node=2 \
        relay/tests/test_nixl_transfer.py --nc-group 0,32 --size-mb 64

Usage (two nodes, 1 rank each):
    # Node 0 (sender):
    NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
        --node-rank=0 --master_addr=<IP> \
        relay/tests/test_nixl_transfer.py --nc 0 --size-mb 256

    # Node 1 (receiver):
    NEURON_RT_MAP_HBM=1 torchrun --nnodes=2 --nproc_per_node=1 \
        --node-rank=1 --master_addr=<IP> \
        relay/tests/test_nixl_transfer.py --nc 0 --size-mb 256
"""

import argparse
import ctypes
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ.setdefault("NEURON_RT_MAP_HBM", "1")


def main():
    parser = argparse.ArgumentParser(description="NIXL backend integration test")
    parser.add_argument("--nc", type=int, default=0,
                        help="NeuronCore index (2-node mode)")
    parser.add_argument("--nc-group", type=str, default="",
                        help="Comma-separated NC indices (1-node mode, e.g. '0,32')")
    parser.add_argument("--size-mb", type=int, default=64,
                        help="Transfer size in MB per rank")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--nixl-port", type=int, default=22000)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.nc_group:
        nc_list = [int(x) for x in args.nc_group.split(",")]
        nc_idx = nc_list[local_rank]
    else:
        nc_idx = args.nc

    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{nc_idx}-{nc_idx + 1}"
    os.environ["NKIPY_NIXL_PORT"] = str(args.nixl_port)

    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This test requires exactly 2 ranks"

    size = args.size_mb * 1024 * 1024
    print(f"[Rank {rank}] NC={nc_idx}, size={args.size_mb} MB, "
          f"iterations={args.iterations}")

    from spike import get_spike_singleton
    spike = get_spike_singleton()
    tensor = spike.allocate_tensor(size, 0, f"test_rank{rank}")
    print(f"[Rank {rank}] Allocated tensor VA=0x{tensor.va:x}")

    pattern_size = min(4096, size)
    if rank == 0:
        ctypes.memset(tensor.va, 0xAB, pattern_size)
        print(f"[Rank {rank}] Wrote 0xAB pattern ({pattern_size} bytes)")

    dist.barrier()

    from relay.endpoint import Endpoint
    ep = Endpoint(nc_idx=0)

    buffers = [("test_weight", tensor.va, size)]

    latencies = []
    for iteration in range(args.iterations):
        if rank == 1:
            ctypes.memset(tensor.va, 0x00, pattern_size)

        dist.barrier()
        t0 = time.time()

        if rank == 0:
            # Sender: register and wait for receiver metadata
            ep.register(buffers)

            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=1)
            receiver_info = obj_list[0]

            # push_to_peer expects per_rank_info[rank] for this rank.
            # In this 2-process test, the sender is rank 0, so we put
            # receiver info at index 0.
            per_rank_info = [receiver_info]
            from relay.transfer import push_to_peer
            push_to_peer(ep, buffers, per_rank_info)

        else:
            # Receiver: register and send metadata to sender
            ep.register(buffers)

            local_info = {
                "agent_metadata": ep.get_metadata().hex(),
                "agent_name": ep.agent_name,
                "nc_idx": 0,
                "buffer_vas": [(tensor.va, size)],
            }
            dist.broadcast_object_list([local_info], src=1)

        dist.barrier()
        elapsed = time.time() - t0
        latencies.append(elapsed)

        if rank == 1:
            buf = (ctypes.c_char * pattern_size).from_address(tensor.va)
            first_bytes = bytes(buf[:64])
            correct = all(b == 0xAB for b in first_bytes)
            throughput = (size * 8) / elapsed / 1e9
            status = "PASS" if correct else "FAIL"
            print(f"[Rank {rank}] Iter {iteration+1}: {elapsed*1000:.1f} ms, "
                  f"{throughput:.1f} Gbps, verify={status}")
            if not correct:
                print(f"  ERROR: first 16 bytes = {first_bytes[:16].hex()}")
                sys.exit(1)
        else:
            throughput = (size * 8) / elapsed / 1e9
            print(f"[Rank {rank}] Iter {iteration+1}: {elapsed*1000:.1f} ms, "
                  f"{throughput:.1f} Gbps")

        ep.deregister_sync()

    if rank == 1:
        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        avg_gbps = (size * 8) / avg_lat / 1e9
        peak_gbps = (size * 8) / min_lat / 1e9
        print(f"\n=== Results ({args.size_mb} MB, {args.iterations} iters) ===")
        print(f"  Avg latency:    {avg_lat*1000:.1f} ms")
        print(f"  Min latency:    {min_lat*1000:.1f} ms")
        print(f"  Avg throughput: {avg_gbps:.1f} Gbps")
        print(f"  Peak throughput: {peak_gbps:.1f} Gbps")

    spike.free_tensor(tensor)
    dist.destroy_process_group()
    print(f"[Rank {rank}] DONE")


if __name__ == "__main__":
    main()
