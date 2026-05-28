"""Multi-endpoint P2P throughput benchmark for Trn2.

Uses N parallel Relay endpoints (each on different gpu_idx → different 4 NICs)
to saturate all 16 EFA NICs and achieve aggregate throughput close to 3.2 Tbps.

Usage (2 nodes, 1 rank/node):
  # Node 0 (client):
  torchrun --nnodes=2 --nproc_per_node=1 --node-rank=0 --master_addr=<MASTER_IP> \
      tests/benchmark_multi_ep_p2p.py --num-eps 4

  # Node 1 (server):
  torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=<MASTER_IP> \
      tests/benchmark_multi_ep_p2p.py --num-eps 4
"""
from __future__ import annotations

import argparse
import mmap
import os
import struct
import sys
import time
import threading
from typing import List

import numpy as np


def _pretty(num: int) -> str:
    units, val = ["B", "KB", "MB", "GB"], float(num)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024


def _alloc_host_buffer(size: int):
    """Allocate a page-aligned host buffer using mmap."""
    buf = mmap.mmap(-1, size)
    arr = np.frombuffer(buf, dtype=np.uint8)
    return buf, arr, arr.ctypes.data


fifo_blob_size = 64


def _run_server(args, eps, rank, world_size):
    """Server: advertise buffers for each endpoint and wait for writes."""
    import torch
    import torch.distributed as dist

    peer = 0
    num_eps = len(eps)

    for sz in args.sizes:
        size_per_ep = sz // num_eps

        # For each endpoint, register a buffer and advertise it
        all_fifo_blobs = []  # [ep_idx][iov_idx]
        for ep_idx, ep in enumerate(eps):
            buf_mmap, buf_arr, buf_ptr = _alloc_host_buffer(size_per_ep)
            ok, mr_id = ep.reg(buf_ptr, size_per_ep)
            assert ok, f"EP {ep_idx}: reg failed"
            ok, fifo_blob_v = ep.advertisev(
                ep.conn_id_of_rank(0), [mr_id], [buf_ptr], [size_per_ep], 1
            )
            assert ok
            all_fifo_blobs.append(fifo_blob_v)

        # Send all fifo blobs to client
        for ep_idx in range(num_eps):
            for blob in all_fifo_blobs[ep_idx]:
                dist.send(torch.ByteTensor(list(blob)), dst=peer)

        dist.barrier()

    print("[Server] Benchmark complete")


def _run_client(args, eps, rank, world_size):
    """Client: receive fifo blobs, perform parallel RDMA writes across all EPs."""
    import torch
    import torch.distributed as dist

    peer = 1
    num_eps = len(eps)

    print(f"[Client] Using {num_eps} endpoints (4 NICs each = {num_eps * 4} NICs)")

    for sz in args.sizes:
        size_per_ep = sz // num_eps

        # Register local buffers for each endpoint
        ep_bufs = []
        for ep_idx, ep in enumerate(eps):
            buf_mmap, buf_arr, buf_ptr = _alloc_host_buffer(size_per_ep)
            ok, mr_id = ep.reg(buf_ptr, size_per_ep)
            assert ok, f"EP {ep_idx}: reg failed"
            ep_bufs.append((mr_id, buf_ptr, size_per_ep))

        # Receive fifo blobs from server
        ep_fifo_blobs = []
        for ep_idx in range(num_eps):
            blobs = []
            fifo_blob = torch.zeros(fifo_blob_size, dtype=torch.uint8)
            dist.recv(fifo_blob, src=peer)
            blobs.append(bytes(fifo_blob.tolist()))
            ep_fifo_blobs.append(blobs)

        # Warmup: parallel writes across all endpoints
        threads = []
        for ep_idx in range(num_eps):
            ep = eps[ep_idx]
            mr_id, ptr, size = ep_bufs[ep_idx]
            blobs = ep_fifo_blobs[ep_idx]

            def _write(ep, mr_id, ptr, size, blobs):
                ep.writev(
                    ep.conn_id_of_rank(0),
                    [mr_id], [ptr], [size], blobs, 1
                )

            t = threading.Thread(target=_write, args=(ep, mr_id, ptr, size, blobs))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Timed iterations: parallel writes
        start = time.perf_counter()
        total = 0
        for _iter in range(args.iters):
            threads = []
            for ep_idx in range(num_eps):
                ep = eps[ep_idx]
                mr_id, ptr, size = ep_bufs[ep_idx]
                blobs = ep_fifo_blobs[ep_idx]

                def _write(ep, mr_id, ptr, size, blobs):
                    ep.writev(
                        ep.conn_id_of_rank(0),
                        [mr_id], [ptr], [size], blobs, 1
                    )

                t = threading.Thread(
                    target=_write, args=(ep, mr_id, ptr, size, blobs)
                )
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            total += sz

        elapsed = time.perf_counter() - start
        gbps = (total * 8) / elapsed / 1e9
        gbs = total / elapsed / 1e9
        per_iter = elapsed / args.iters
        print(
            f"[Client] {_pretty(sz):>8} ({num_eps} EPs): "
            f"{gbps:7.2f} Gbps | {gbs:6.2f} GB/s | {per_iter:.6f} s/iter"
        )
        dist.barrier()

    print("[Client] Benchmark complete")


def main():
    p = argparse.ArgumentParser("Multi-EP P2P benchmark (all-NIC saturation)")
    p.add_argument(
        "--num-eps", type=int, default=4,
        help="Number of parallel endpoints (each uses 4 NICs). "
             "4 eps = 16 NICs = full bandwidth."
    )
    p.add_argument(
        "--gpu-idx-list", type=str, default="",
        help="Comma-separated list of gpu_idx for each endpoint. "
             "Default: 0,1,2,...,num_eps-1"
    )
    p.add_argument(
        "--sizes",
        type=lambda v: [int(x) for x in v.split(",") if x],
        default=[1048576, 10485760, 67108864, 134217728, 268435456, 536870912],
    )
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    if args.gpu_idx_list:
        gpu_idxs = [int(x) for x in args.gpu_idx_list.split(",")]
    else:
        gpu_idxs = list(range(args.num_eps))

    assert len(gpu_idxs) == args.num_eps

    import torch
    import torch.distributed as dist

    # Don't actually need neuron cores for host-memory RDMA benchmark
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

    try:
        from relay import _relay
    except ImportError:
        sys.stderr.write("Failed to import relay._relay\n")
        raise

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    print(f"[Rank {rank}] Creating {args.num_eps} endpoints with gpu_idxs={gpu_idxs}")

    # Create multiple endpoints, each bound to different NICs
    eps = []
    for gidx in gpu_idxs:
        ep = _relay.Endpoint(gidx)
        eps.append(ep)

    # Exchange metadata for all endpoints
    # Format: for each EP, send its metadata to the peer
    all_local_meta = [ep.get_metadata() for ep in eps]

    # Exchange metadata
    if rank == 0:
        # Send our metadata to rank 1
        for meta in all_local_meta:
            dist.send(torch.ByteTensor(list(meta)), dst=1)
        # Receive rank 1's metadata
        all_remote_meta = []
        for _ in range(args.num_eps):
            remote_meta = torch.zeros(len(all_local_meta[0]), dtype=torch.uint8)
            dist.recv(remote_meta, src=1)
            all_remote_meta.append(bytes(remote_meta.tolist()))
    else:
        # Receive rank 0's metadata
        all_remote_meta = []
        for _ in range(args.num_eps):
            remote_meta = torch.zeros(len(all_local_meta[0]), dtype=torch.uint8)
            dist.recv(remote_meta, src=0)
            all_remote_meta.append(bytes(remote_meta.tolist()))
        # Send our metadata to rank 0
        for meta in all_local_meta:
            dist.send(torch.ByteTensor(list(meta)), dst=0)

    # Connect all endpoints to their corresponding peer
    print(f"[Rank {rank}] Connecting {args.num_eps} endpoints...")
    for ep_idx, ep in enumerate(eps):
        ip, port, r_gpu = _relay.Endpoint.parse_metadata(all_remote_meta[ep_idx])
        if rank == 0:
            ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
        else:
            ep.start_passive_accept()
            ok, r_ip, r_nc, conn_id = ep.accept()
        assert ok, f"EP {ep_idx}: connection failed"
        print(f"[Rank {rank}] EP {ep_idx} connected (conn_id={conn_id})")

    if rank == 0:
        _run_client(args, eps, rank, world_size)
    else:
        _run_server(args, eps, rank, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] Aborted.")
        sys.exit(1)
