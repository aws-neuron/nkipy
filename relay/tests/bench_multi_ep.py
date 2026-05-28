#!/usr/bin/env python3
"""Multi-endpoint P2P throughput benchmark for Trn2.

Standalone (no torchrun) — uses TCP sockets for coordination.

Usage:
  # Server (receiver, remote host):
  python bench_multi_ep.py --role server --peer-ip <CLIENT_IP>

  # Client (sender, this host):
  python bench_multi_ep.py --role client --peer-ip 10.3.215.3
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import socket
import struct
import sys
import time
import threading
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

COORD_PORT = 19876
fifo_blob_size = 64


def _pretty(num: int) -> str:
    units, val = ["B", "KB", "MB", "GB"], float(num)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024


def _alloc_host_buffer(size: int):
    """Allocate page-aligned host buffer."""
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


# --- Benchmark ---

def run_server(args):
    """Server = receiver. Registers buffers, lets client write into them."""
    from relay import _relay

    num_eps = args.num_eps
    gpu_idxs = args.gpu_idxs
    num_iovs = args.num_iovs

    # Create endpoints
    print(f"[Server] Creating {num_eps} endpoints with gpu_idxs={gpu_idxs}")
    eps = []
    for gidx in gpu_idxs:
        ep = _relay.Endpoint(gidx)
        ep.start_passive_accept()
        eps.append(ep)

    # Collect metadata
    all_meta = [ep.get_metadata().hex() for ep in eps]

    # Start TCP coordination server
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", COORD_PORT))
    srv.listen(1)
    print(f"[Server] Listening on port {COORD_PORT}...")
    conn, addr = srv.accept()
    print(f"[Server] Client connected from {addr}")

    # Exchange metadata
    send_json(conn, {"metadata": all_meta})
    client_info = recv_json(conn)

    # Wait for client to establish all RDMA connections
    msg = recv_json(conn)
    assert msg["status"] == "connected"
    time.sleep(0.5)
    print(f"[Server] All {num_eps} endpoints connected")

    # Run benchmark for each size
    for sz in args.sizes:
        size_per_ep = sz // num_eps
        iov_size = size_per_ep // num_iovs

        # Register + advertise buffers for each endpoint
        all_fifo_hex = []
        for ep_idx, ep in enumerate(eps):
            buf_mmap, buf_arr, buf_ptr = _alloc_host_buffer(size_per_ep)
            # Register multiple IOVs within the buffer
            mr_ids = []
            ptrs = []
            sizes = []
            for iov in range(num_iovs):
                offset = iov * iov_size
                ok, mr_id = ep.reg(buf_ptr + offset, iov_size)
                assert ok
                mr_ids.append(mr_id)
                ptrs.append(buf_ptr + offset)
                sizes.append(iov_size)

            ok, fifo_blob_v = ep.advertisev(
                0, mr_ids, ptrs, sizes, num_iovs
            )
            assert ok
            all_fifo_hex.append([b.hex() for b in fifo_blob_v])

        # Send fifo blobs to client
        send_json(conn, {"fifo_blobs": all_fifo_hex, "size": sz})

        # Wait for client to finish
        msg = recv_json(conn)
        assert msg["status"] == "done"

    print("[Server] Benchmark complete")
    conn.close()
    srv.close()


def run_client(args):
    """Client = sender. Connects to all endpoints, does parallel RDMA writes."""
    from relay import _relay

    num_eps = args.num_eps
    gpu_idxs = args.gpu_idxs
    num_iovs = args.num_iovs

    # Create endpoints
    print(f"[Client] Creating {num_eps} endpoints with gpu_idxs={gpu_idxs}")
    eps = []
    for gidx in gpu_idxs:
        ep = _relay.Endpoint(gidx)
        eps.append(ep)

    # Collect metadata
    all_meta = [ep.get_metadata().hex() for ep in eps]

    # Connect to server via TCP
    time.sleep(2)
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[Client] Connecting to {args.peer_ip}:{COORD_PORT}...")
    for retry in range(30):
        try:
            conn.connect((args.peer_ip, COORD_PORT))
            break
        except ConnectionRefusedError:
            time.sleep(1)
    else:
        raise ConnectionError(f"Cannot connect to {args.peer_ip}:{COORD_PORT}")
    print("[Client] TCP connected")

    # Exchange metadata
    server_info = recv_json(conn)
    server_meta = server_info["metadata"]
    send_json(conn, {"metadata": all_meta})

    # Connect all endpoints via add_remote_endpoint (sequential with delay)
    conn_ids = []
    for ep_idx, ep in enumerate(eps):
        meta_bytes = bytes.fromhex(server_meta[ep_idx])
        ok, conn_id = ep.add_remote_endpoint(meta_bytes)
        assert ok, f"EP {ep_idx}: add_remote_endpoint failed"
        conn_ids.append(conn_id)
        print(f"[Client] EP {ep_idx} connected (conn_id={conn_id})")
        if ep_idx < num_eps - 1:
            time.sleep(1)

    send_json(conn, {"status": "connected"})
    print(f"[Client] All {num_eps} endpoints connected "
          f"({num_eps * 4} NICs, {num_iovs} IOVs/EP)")
    print()

    # Run benchmark for each size
    for sz in args.sizes:
        size_per_ep = sz // num_eps
        iov_size = size_per_ep // num_iovs

        # Receive fifo blobs from server
        msg = recv_json(conn)
        all_fifo_hex = msg["fifo_blobs"]
        assert msg["size"] == sz

        # Register local send buffers
        ep_bufs = []  # [(mr_ids, ptrs, sizes)]
        for ep_idx, ep in enumerate(eps):
            buf_mmap, buf_arr, buf_ptr = _alloc_host_buffer(size_per_ep)
            mr_ids = []
            ptrs = []
            sizes = []
            for iov in range(num_iovs):
                offset = iov * iov_size
                ok, mr_id = ep.reg(buf_ptr + offset, iov_size)
                assert ok
                mr_ids.append(mr_id)
                ptrs.append(buf_ptr + offset)
                sizes.append(iov_size)
            ep_bufs.append((mr_ids, ptrs, sizes))

        # Parse fifo blobs
        ep_fifo_blobs = []
        for ep_idx in range(num_eps):
            blobs = [bytes.fromhex(h) for h in all_fifo_hex[ep_idx]]
            ep_fifo_blobs.append(blobs)

        # Warmup
        def _do_write(ep_idx):
            ep = eps[ep_idx]
            mr_ids, ptrs, sizes = ep_bufs[ep_idx]
            blobs = ep_fifo_blobs[ep_idx]
            ep.writev(conn_ids[ep_idx], mr_ids, ptrs, sizes, blobs, num_iovs)

        threads = [threading.Thread(target=_do_write, args=(i,))
                   for i in range(num_eps)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Timed iterations
        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            threads = [threading.Thread(target=_do_write, args=(i,))
                       for i in range(num_eps)]
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
            f"[Client] {_pretty(sz):>8} x {num_eps}EP x {num_iovs}IOV: "
            f"{gbps:7.2f} Gbps | {gbs:6.2f} GB/s | {per_iter:.6f} s/iter"
        )

        send_json(conn, {"status": "done"})

    print("\n[Client] Benchmark complete")
    conn.close()


def main():
    p = argparse.ArgumentParser("Multi-EP P2P benchmark")
    p.add_argument("--role", choices=["client", "server"], required=True)
    p.add_argument("--peer-ip", type=str, required=True)
    p.add_argument(
        "--num-eps", type=int, default=4,
        help="Number of parallel endpoints (4 eps = 16 NICs = full BW)"
    )
    p.add_argument(
        "--num-iovs", type=int, default=16,
        help="Number of IOVs per endpoint (more = better QP utilization)"
    )
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

    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
    os.environ.setdefault("RELAY_LOG_LEVEL", "ERROR")

    if args.role == "server":
        run_server(args)
    else:
        run_client(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] Aborted.")
        sys.exit(1)
