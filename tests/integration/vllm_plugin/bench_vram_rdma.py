#!/usr/bin/env python3
"""Benchmark: Cross-instance device-to-device RDMA via NIXL LIBFABRIC.

Measures RDMA throughput for direct NeuronCore HBM transfers between two
trn2 instances using NIXL with the dmabuf ioctl fix.

Architecture:
  Sender:   allocates device tensors on NeuronCore, fills with pattern
  Receiver: allocates device tensors on NeuronCore, performs RDMA READ

Usage (run on both instances simultaneously):
    # Sender (instance A):
    python bench_vram_rdma.py --role sender --peer-host 10.3.215.3

    # Receiver (instance B):
    python bench_vram_rdma.py --role receiver --peer-host 10.3.211.148
"""

import argparse
import ctypes
import os
import socket
import struct
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ.setdefault("NEURON_RT_MAP_HBM", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../nkipy/src"))


def apply_dmabuf_fix():
    """Apply the nrt_get_dmabuf_fd hotpatch."""
    from relay.nrt_dmabuf_fix import patch_nrt_dmabuf
    patch_nrt_dmabuf()


def exchange_metadata(role, peer_host, local_port, agent, local_va, local_size):
    """Exchange NIXL agent metadata and buffer VAs via TCP sideband."""
    meta_port = local_port + 5000
    local_meta = agent.get_agent_metadata()
    local_info = struct.pack("!QQ", local_va, local_size)

    if role == "sender":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", meta_port))
        srv.listen(1)
        print(f"  Waiting for receiver on port {meta_port}...")
        conn, addr = srv.accept()
        print(f"  Connected from {addr}")
        conn.sendall(struct.pack("!I", len(local_meta)) + local_meta + local_info)
        sz = struct.unpack("!I", _recv_exact(conn, 4))[0]
        peer_meta = _recv_exact(conn, sz)
        peer_info = _recv_exact(conn, 16)
        conn.close()
        srv.close()
    else:
        time.sleep(0.5)
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for attempt in range(30):
            try:
                conn.connect((peer_host, meta_port))
                break
            except ConnectionRefusedError:
                time.sleep(1)
        else:
            raise RuntimeError(f"Cannot connect to sender at {peer_host}:{meta_port}")
        sz = struct.unpack("!I", _recv_exact(conn, 4))[0]
        peer_meta = _recv_exact(conn, sz)
        peer_info = _recv_exact(conn, 16)
        conn.sendall(struct.pack("!I", len(local_meta)) + local_meta + local_info)
        conn.close()

    peer_va, peer_size = struct.unpack("!QQ", peer_info)
    return peer_meta, peer_va, peer_size


def _recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise RuntimeError("Connection closed")
        data += chunk
    return data


def sync_barrier(role, peer_host, port):
    """Simple TCP barrier to synchronize sender/receiver."""
    barrier_port = port + 6000
    if role == "sender":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", barrier_port))
        srv.listen(1)
        conn, _ = srv.accept()
        conn.recv(1)
        conn.sendall(b"K")
        conn.close()
        srv.close()
    else:
        time.sleep(0.2)
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for _ in range(30):
            try:
                conn.connect((peer_host, barrier_port))
                break
            except ConnectionRefusedError:
                time.sleep(1)
        conn.sendall(b"R")
        conn.recv(1)
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Cross-instance VRAM RDMA benchmark")
    parser.add_argument("--role", choices=["sender", "receiver"], required=True)
    parser.add_argument("--peer-host", required=True, help="IP of the other instance")
    parser.add_argument("--port", type=int, default=20000, help="NIXL listen port")
    parser.add_argument("--size-mb", type=int, default=256, help="Transfer size in MB")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--core", type=int, default=0, help="NeuronCore ID")
    args = parser.parse_args()

    size = args.size_mb * 1024 * 1024
    visible_cores = f"{args.core}-{args.core + 1}"
    os.environ["NEURON_RT_VISIBLE_CORES"] = visible_cores

    print(f"=== VRAM RDMA Benchmark ({args.role}) ===")
    print(f"  Transfer size: {args.size_mb} MB")
    print(f"  Iterations: {args.iterations}")
    print(f"  Peer: {args.peer_host}")
    print(f"  NeuronCore: {args.core}")
    print()

    # Step 1: Apply dmabuf fix
    print("[1] Applying nrt_dmabuf_fd fix...")
    apply_dmabuf_fix()
    print("    Done")

    # Step 2: Allocate device memory via spike
    print(f"[2] Allocating {args.size_mb} MB on NeuronCore {args.core}...")
    t0 = time.time()
    from spike import get_spike_singleton
    spike = get_spike_singleton()
    tensor = spike.allocate_tensor(size, args.core, f"bench_{args.role}")
    print(f"    VA: 0x{tensor.va:x}, took {time.time() - t0:.2f}s")

    if args.role == "sender":
        # Write small pattern for verification (mapped device memory writes are slow)
        verify_size = 4096
        print(f"[3] Writing test pattern (first {verify_size} bytes)...")
        ctypes.memset(tensor.va, 0xAB, verify_size)

    # Step 3: Create NIXL agent and register VRAM
    step = 3 if args.role == "receiver" else 4
    print(f"[{step}] Creating NIXL agent and registering VRAM...")
    t0 = time.time()
    from nixl._api import nixl_agent, nixl_agent_config
    config = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        listen_port=args.port,
        backends=["LIBFABRIC"],
    )
    agent = nixl_agent(f"bench_{args.role}", config)
    mem_desc = [(tensor.va, tensor.size, args.core, "")]
    reg = agent.register_memory(mem_desc, mem_type="VRAM", backends=["LIBFABRIC"])
    print(f"    Registered {args.size_mb} MB VRAM in {time.time() - t0:.3f}s")

    # Step 4: Exchange metadata
    step += 1
    print(f"[{step}] Exchanging metadata with peer...")
    t0 = time.time()
    peer_meta, peer_va, peer_size = exchange_metadata(
        args.role, args.peer_host, args.port, agent, tensor.va, tensor.size
    )
    peer_name = f"bench_{'receiver' if args.role == 'sender' else 'sender'}"
    agent.add_remote_agent(peer_meta)
    print(f"    Done ({time.time() - t0:.2f}s, peer VA=0x{peer_va:x})")

    # Step 5: Transfer
    step += 1
    if args.role == "receiver":
        print(f"[{step}] Running RDMA READ benchmark ({args.iterations} iterations)...")
        local_descs = agent.get_xfer_descs([(tensor.va, size, args.core)], "VRAM")
        remote_descs = agent.get_xfer_descs([(peer_va, peer_size, args.core)], "VRAM")

        latencies = []
        for i in range(args.iterations):
            t0 = time.time()
            xfer = agent.initialize_xfer("READ", local_descs, remote_descs, peer_name)
            status = agent.transfer(xfer)
            if status == "PROC":
                while True:
                    status = agent.check_xfer_state(xfer)
                    if status != "PROC":
                        break
                    time.sleep(0.0001)
            elapsed = time.time() - t0
            latencies.append(elapsed)
            throughput = (size * 8) / elapsed / 1e9
            print(f"    Iter {i+1}: {elapsed*1000:.1f} ms, {throughput:.1f} Gbps, status={status}")
            xfer.release()

        # Verify data (check first few bytes of the pattern region)
        buf = (ctypes.c_char * 64).from_address(tensor.va)
        first_bytes = bytes(buf[:8])
        correct = all(b == 0xAB for b in first_bytes)
        print(f"\n    Data verification: {'PASS' if correct else 'FAIL (expected 0xAB)'} "
              f"(first 8 bytes: {first_bytes.hex()})")

        # Summary
        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        avg_gbps = (size * 8) / avg_lat / 1e9
        peak_gbps = (size * 8) / min_lat / 1e9
        print(f"\n  === Results ({args.size_mb} MB, {args.iterations} iters) ===")
        print(f"  Avg latency: {avg_lat*1000:.1f} ms")
        print(f"  Min latency: {min_lat*1000:.1f} ms")
        print(f"  Avg throughput: {avg_gbps:.1f} Gbps")
        print(f"  Peak throughput: {peak_gbps:.1f} Gbps")

    else:
        print(f"[{step}] Sender ready. Waiting for receiver to complete transfers...")
        sync_barrier("sender", args.peer_host, args.port)
        print("    Receiver signaled completion.")

    # Notify peer we're done
    if args.role == "receiver":
        sync_barrier("receiver", args.peer_host, args.port)

    # Cleanup
    step += 1
    print(f"\n[{step}] Cleanup...")
    agent.deregister_memory(reg, backends=["LIBFABRIC"])
    spike.free_tensor(tensor)
    print("    Done")
    print("\n=== BENCHMARK COMPLETE ===")


if __name__ == "__main__":
    main()
