#!/usr/bin/env python3
"""Multi-process P2P throughput benchmark for Trn2.

Launches N worker processes (one per endpoint/NIC-group) to avoid
the OOB connection issues when multiple endpoints share one process.

Usage:
  # Server (receiver):
  python bench_multi_proc.py --role server --peer-ip <CLIENT_IP> --num-eps 4

  # Client (sender):
  python bench_multi_proc.py --role client --peer-ip <SERVER_IP> --num-eps 4
"""
from __future__ import annotations

import argparse
import json
import mmap
import multiprocessing as mp
import os
import socket
import struct
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

COORD_BASE_PORT = 19900


def _pretty(num: int) -> str:
    units, val = ["B", "KB", "MB", "GB"], float(num)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024


def _alloc_host_buffer(size: int):
    buf = mmap.mmap(-1, size)
    arr = np.frombuffer(buf, dtype=np.uint8)
    return buf, arr, arr.ctypes.data


# --- TCP coordination helpers ---

def send_msg(sock, data: bytes):
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_msg(sock) -> bytes:
    raw = b""
    while len(raw) < 4:
        chunk = sock.recv(4 - len(raw))
        if not chunk:
            raise ConnectionError("connection closed")
        raw += chunk
    length = struct.unpack("!I", raw)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(65536, length - len(data)))
        if not chunk:
            raise ConnectionError("connection closed")
        data += chunk
    return data


def send_json(sock, obj):
    send_msg(sock, json.dumps(obj).encode())


def recv_json(sock):
    return json.loads(recv_msg(sock).decode())


# --- Worker processes ---

def server_worker(gpu_idx: int, ep_idx: int, coord_port: int,
                  sizes: List[int], num_iovs: int, iters: int,
                  peer_ip: str):
    """One server worker per endpoint. Handles its own TCP coordination."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    os.environ.setdefault("RELAY_LOG_LEVEL", "ERROR")
    from relay import _relay

    ep = _relay.Endpoint(gpu_idx)
    ep.start_passive_accept()

    meta_hex = ep.get_metadata().hex()

    # TCP coordination for this endpoint pair
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", coord_port))
    srv.listen(1)
    conn, addr = srv.accept()

    # Exchange metadata
    send_json(conn, {"metadata": meta_hex})
    recv_json(conn)

    for sz in sizes:
        iov_size = sz // num_iovs
        buf_mmap, buf_arr, buf_ptr = _alloc_host_buffer(sz)

        mr_ids, ptrs, ep_sizes = [], [], []
        for iov in range(num_iovs):
            offset = iov * iov_size
            ok, mr_id = ep.reg(buf_ptr + offset, iov_size)
            assert ok
            mr_ids.append(mr_id)
            ptrs.append(buf_ptr + offset)
            ep_sizes.append(iov_size)

        ok, fifo_blob_v = ep.advertisev(0, mr_ids, ptrs, ep_sizes, num_iovs)
        assert ok

        send_json(conn, {"fifo_blobs": [b.hex() for b in fifo_blob_v]})
        msg = recv_json(conn)
        assert msg["status"] == "done"

    conn.close()
    srv.close()


def client_worker(gpu_idx: int, ep_idx: int, coord_port: int,
                  sizes: List[int], num_iovs: int, iters: int,
                  peer_ip: str, result_queue, start_barrier):
    """One client worker per endpoint. Handles its own TCP coordination."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    os.environ.setdefault("RELAY_LOG_LEVEL", "ERROR")
    from relay import _relay

    ep = _relay.Endpoint(gpu_idx)

    # TCP coordination for this endpoint pair
    time.sleep(2)
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for retry in range(30):
        try:
            conn.connect((peer_ip, coord_port))
            break
        except ConnectionRefusedError:
            time.sleep(1)
    else:
        raise ConnectionError(f"EP {ep_idx}: Cannot connect to {peer_ip}:{coord_port}")

    # Exchange metadata and connect RDMA
    server_info = recv_json(conn)
    server_meta_hex = server_info["metadata"]
    meta_bytes = bytes.fromhex(server_meta_hex)
    ok, conn_id = ep.add_remote_endpoint(meta_bytes)
    assert ok, f"EP {ep_idx}: add_remote_endpoint failed"
    send_json(conn, {"status": "connected"})

    for sz in sizes:
        iov_size = sz // num_iovs
        buf_mmap, buf_arr, buf_ptr = _alloc_host_buffer(sz)

        mr_ids, ptrs, ep_sizes = [], [], []
        for iov in range(num_iovs):
            offset = iov * iov_size
            ok, mr_id = ep.reg(buf_ptr + offset, iov_size)
            assert ok
            mr_ids.append(mr_id)
            ptrs.append(buf_ptr + offset)
            ep_sizes.append(iov_size)

        # Get fifo blobs from server
        msg = recv_json(conn)
        blobs = [bytes.fromhex(h) for h in msg["fifo_blobs"]]

        # Warmup
        ep.writev(conn_id, mr_ids, ptrs, ep_sizes, blobs, num_iovs)

        # Synchronize start across all workers
        start_barrier.wait()

        # Timed iterations
        t0 = time.perf_counter()
        for _ in range(iters):
            ep.writev(conn_id, mr_ids, ptrs, ep_sizes, blobs, num_iovs)
        elapsed = time.perf_counter() - t0

        result_queue.put((ep_idx, sz, elapsed, sz * iters))
        send_json(conn, {"status": "done"})

    conn.close()


# --- Main orchestration ---

def run_server(args):
    num_eps = args.num_eps
    gpu_idxs = args.gpu_idxs

    workers = []
    for i, gidx in enumerate(gpu_idxs):
        port = COORD_BASE_PORT + i
        p = mp.Process(target=server_worker,
                       args=(gidx, i, port, args.sizes, args.num_iovs,
                             args.iters, args.peer_ip))
        p.start()
        workers.append(p)
        print(f"[Server] EP {i} (gpu_idx={gidx}) listening on port {port}")

    for p in workers:
        p.join()
    print("[Server] Benchmark complete")


def run_client(args):
    num_eps = args.num_eps
    gpu_idxs = args.gpu_idxs

    print(f"[Client] {num_eps} endpoints, {num_eps * 4} NICs, {num_iovs} IOVs/EP"
          if (num_iovs := args.num_iovs) else "")

    result_queue = mp.Queue()
    start_barrier = mp.Barrier(num_eps)

    workers = []
    for i, gidx in enumerate(gpu_idxs):
        port = COORD_BASE_PORT + i
        p = mp.Process(target=client_worker,
                       args=(gidx, i, port, args.sizes, args.num_iovs,
                             args.iters, args.peer_ip, result_queue,
                             start_barrier))
        p.start()
        workers.append(p)

    # Collect and display results per size
    results_by_size = {}
    total_results = num_eps * len(args.sizes)
    for _ in range(total_results):
        ep_idx, sz, elapsed, total_bytes = result_queue.get(timeout=300)
        if sz not in results_by_size:
            results_by_size[sz] = []
        results_by_size[sz].append((ep_idx, elapsed, total_bytes))

        # Print when all EPs for a size are done
        if len(results_by_size[sz]) == num_eps:
            max_elapsed = max(r[1] for r in results_by_size[sz])
            total_data = sum(r[2] for r in results_by_size[sz])
            gbps = (total_data * 8) / max_elapsed / 1e9
            gbs = total_data / max_elapsed / 1e9
            per_iter = max_elapsed / args.iters

            per_ep = []
            for idx, el, tb in sorted(results_by_size[sz]):
                ep_gbps = (tb * 8) / el / 1e9
                per_ep.append(f"EP{idx}={ep_gbps:.0f}")

            print(
                f"[Client] {_pretty(sz):>8} x {num_eps}EP x {args.num_iovs}IOV: "
                f"{gbps:7.2f} Gbps | {gbs:6.2f} GB/s | {per_iter:.6f} s/iter "
                f"[{', '.join(per_ep)}]"
            )

    for p in workers:
        p.join(timeout=10)
    print("\n[Client] Benchmark complete")


def main():
    p = argparse.ArgumentParser("Multi-process P2P benchmark")
    p.add_argument("--role", choices=["client", "server"], required=True)
    p.add_argument("--peer-ip", type=str, required=True)
    p.add_argument("--num-eps", type=int, default=4)
    p.add_argument("--num-iovs", type=int, default=16)
    p.add_argument(
        "--gpu-idx-list", type=str, default="",
        help="Comma-separated gpu_idx for each endpoint"
    )
    p.add_argument(
        "--sizes",
        type=lambda v: [int(x) for x in v.split(",") if x],
        default=[268435456, 536870912, 1073741824, 2147483648],
    )
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    if args.gpu_idx_list:
        args.gpu_idxs = [int(x) for x in args.gpu_idx_list.split(",")]
    else:
        args.gpu_idxs = list(range(args.num_eps))

    assert len(args.gpu_idxs) == args.num_eps

    if args.role == "server":
        run_server(args)
    else:
        run_client(args)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] Aborted.")
        sys.exit(1)
