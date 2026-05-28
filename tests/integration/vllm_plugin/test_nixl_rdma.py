#!/usr/bin/env python3
"""Test: NIXL cross-instance RDMA transfer via LIBFABRIC (host-staged).

Verifies NIXL can perform RDMA transfers between two trn2 instances
using host memory buffers registered with the LIBFABRIC backend.

This validates that NIXL can replace our custom relay for the RDMA hop
in the host-staged weight transfer pipeline.

Architecture:
  Instance A (sender): registers host buffer, fills with data
  Instance B (receiver): registers host buffer, receives via RDMA READ

Usage (run on receiver instance):
    python test_nixl_rdma.py --role receiver --sender-host 10.3.211.148

Usage (run on sender instance):
    python test_nixl_rdma.py --role sender --receiver-host 10.3.215.3
"""

import argparse
import ctypes
import json
import socket
import struct
import time


def create_agent(name, port):
    from nixl._api import nixl_agent, nixl_agent_config
    config = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        listen_port=port,
        backends=["LIBFABRIC"],
    )
    return nixl_agent(name, config)


def register_host_buffer(agent, size_bytes):
    buf = (ctypes.c_char * size_bytes)()
    addr = ctypes.addressof(buf)
    mem_desc = [(addr, size_bytes, 0, "")]
    reg = agent.register_memory(mem_desc, mem_type="DRAM", backends=["LIBFABRIC"])
    return buf, addr, reg


def exchange_metadata(role, peer_host, local_port, peer_port, local_agent):
    """Exchange NIXL agent metadata via TCP sideband."""
    local_meta = local_agent.get_agent_metadata()

    if role == "sender":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", local_port + 1000))
        srv.listen(1)
        print(f"  Waiting for receiver to connect on port {local_port + 1000}...")
        conn, _ = srv.accept()
        data = local_meta
        conn.sendall(struct.pack("!I", len(data)) + data)
        sz = struct.unpack("!I", conn.recv(4))[0]
        peer_meta = b""
        while len(peer_meta) < sz:
            peer_meta += conn.recv(sz - len(peer_meta))
        conn.close()
        srv.close()
    else:
        time.sleep(1)
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for _ in range(10):
            try:
                conn.connect((peer_host, peer_port + 1000))
                break
            except ConnectionRefusedError:
                time.sleep(1)
        data = local_meta
        conn.sendall(struct.pack("!I", len(data)) + data)
        sz = struct.unpack("!I", conn.recv(4))[0]
        peer_meta = b""
        while len(peer_meta) < sz:
            peer_meta += conn.recv(sz - len(peer_meta))
        conn.close()

    return peer_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["sender", "receiver"], required=True)
    parser.add_argument("--sender-host", default="10.3.211.148")
    parser.add_argument("--receiver-host", default="10.3.215.3")
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--size-mb", type=int, default=256)
    args = parser.parse_args()

    size = args.size_mb * 1024 * 1024
    peer_host = args.receiver_host if args.role == "sender" else args.sender_host

    print(f"=== NIXL RDMA Test ({args.role}) ===")
    print(f"  Buffer size: {args.size_mb} MB")
    print(f"  Peer: {peer_host}")

    print(f"\n[1] Creating NIXL agent...")
    t0 = time.time()
    agent = create_agent(f"nixl_{args.role}", args.port)
    print(f"    Done ({time.time() - t0:.2f}s)")

    print(f"[2] Registering {args.size_mb} MB host buffer...")
    t0 = time.time()
    buf, addr, reg = register_host_buffer(agent, size)
    print(f"    Done ({time.time() - t0:.3f}s, addr=0x{addr:x})")

    if args.role == "sender":
        # Fill buffer with recognizable pattern
        print(f"[3] Filling buffer with test pattern...")
        ctypes.memset(addr, 0xAB, size)

    print(f"[{'3' if args.role == 'receiver' else '4'}] Exchanging metadata with peer...")
    t0 = time.time()
    peer_meta = exchange_metadata(args.role, peer_host, args.port, args.port, agent)
    elapsed = time.time() - t0
    print(f"    Done ({elapsed:.2f}s)")

    peer_name = f"nixl_{'receiver' if args.role == 'sender' else 'sender'}"
    agent.add_remote_agent(peer_meta)

    if args.role == "receiver":
        # Receiver does RDMA READ from sender
        print(f"[4] Performing RDMA READ ({args.size_mb} MB)...")
        t0 = time.time()

        local_descs = agent.get_xfer_descs([(addr, size, 0)], "DRAM")
        remote_descs = agent.get_xfer_descs([(addr, size, 0)], "DRAM")

        xfer = agent.initialize_xfer("READ", local_descs, remote_descs, peer_name)
        status = agent.transfer(xfer)
        if status == "PROC":
            while True:
                status = agent.check_xfer_state(xfer)
                if status != "PROC":
                    break
                time.sleep(0.001)

        elapsed = time.time() - t0
        throughput = (size * 8) / elapsed / 1e9

        print(f"    Status: {status}")
        print(f"    Time: {elapsed:.3f}s")
        print(f"    Throughput: {throughput:.1f} Gbps")

        # Verify
        first_byte = buf[0]
        last_byte = buf[size - 1]
        correct = (first_byte == b'\xab' and last_byte == b'\xab')
        print(f"    Data correct: {correct} (first=0x{first_byte[0]:02x}, last=0x{last_byte[0]:02x})")

        xfer.release()
    else:
        # Sender waits for receiver to complete
        print(f"[5] Sender waiting (receiver will READ from us)...")
        input("    Press Enter when receiver is done...")

    print(f"\n[6] Cleanup...")
    agent.deregister_memory(reg, backends=["LIBFABRIC"])
    print(f"    Done")
    print(f"\n=== TEST COMPLETE ===")


if __name__ == "__main__":
    main()
