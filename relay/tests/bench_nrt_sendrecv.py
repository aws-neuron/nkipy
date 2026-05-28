#!/usr/bin/env python3
"""Benchmark nrt_async_sendrecv for cross-instance HBM-to-HBM transfer on Trn2.

Uses the NRT async send/recv API which operates directly on device tensors
without host memory staging.

Usage:
  # Receiver (remote host):
  python bench_nrt_sendrecv.py --role recv --peer-ip <SENDER_IP> --lnc 0

  # Sender (this host):
  python bench_nrt_sendrecv.py --role send --peer-ip <RECEIVER_IP> --lnc 0
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import socket
import struct
import sys
import time

COORD_PORT = 19878

# NRT constants
NRT_SUCCESS = 0
NRT_TENSOR_PLACEMENT_DEVICE = 0
NRT_TENSOR_PLACEMENT_HOST = 1
NRT_FRAMEWORK_TYPE_NO_FW = 1

# Load libnrt
_libnrt = ctypes.CDLL("/opt/aws/neuron/lib/libnrt.so", mode=ctypes.RTLD_GLOBAL)

# --- NRT core API ---
_libnrt.nrt_init.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
_libnrt.nrt_init.restype = ctypes.c_int
_libnrt.nrt_close.argtypes = []
_libnrt.nrt_close.restype = None

# --- Tensor API ---
_libnrt.nrt_tensor_allocate.argtypes = [
    ctypes.c_int,       # placement
    ctypes.c_int,       # vnc
    ctypes.c_size_t,    # size
    ctypes.c_char_p,    # name
    ctypes.POINTER(ctypes.c_void_p),  # tensor**
]
_libnrt.nrt_tensor_allocate.restype = ctypes.c_int

_libnrt.nrt_tensor_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libnrt.nrt_tensor_free.restype = None

_libnrt.nrt_tensor_get_size.argtypes = [ctypes.c_void_p]
_libnrt.nrt_tensor_get_size.restype = ctypes.c_size_t

_libnrt.nrt_tensor_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
_libnrt.nrt_tensor_write.restype = ctypes.c_int

_libnrt.nrt_tensor_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
_libnrt.nrt_tensor_read.restype = ctypes.c_int

# --- Async sendrecv API ---
_libnrt.nrt_async_sendrecv_get_max_num_communicators_per_lnc.argtypes = [ctypes.POINTER(ctypes.c_int)]
_libnrt.nrt_async_sendrecv_get_max_num_communicators_per_lnc.restype = ctypes.c_int

_libnrt.nrt_async_sendrecv_get_max_num_pending_request.argtypes = [ctypes.POINTER(ctypes.c_int)]
_libnrt.nrt_async_sendrecv_get_max_num_pending_request.restype = ctypes.c_int

_libnrt.nrt_async_sendrecv_init.argtypes = [ctypes.c_int]
_libnrt.nrt_async_sendrecv_init.restype = ctypes.c_int

_libnrt.nrt_async_sendrecv_close.argtypes = [ctypes.c_int]
_libnrt.nrt_async_sendrecv_close.restype = ctypes.c_int

_libnrt.nrt_async_sendrecv_connect.argtypes = [
    ctypes.c_char_p,    # peer_ip
    ctypes.c_int,       # peer_lnc
    ctypes.c_int,       # lnc
    ctypes.POINTER(ctypes.c_void_p),  # send_comm**
]
_libnrt.nrt_async_sendrecv_connect.restype = ctypes.c_int

_libnrt.nrt_async_sendrecv_accept.argtypes = [
    ctypes.c_char_p,    # peer_ip
    ctypes.c_int,       # peer_lnc
    ctypes.c_int,       # lnc
    ctypes.POINTER(ctypes.c_void_p),  # recv_comm**
]
_libnrt.nrt_async_sendrecv_accept.restype = ctypes.c_int

_libnrt.nrt_async_sendrecv_test_comm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
_libnrt.nrt_async_sendrecv_test_comm.restype = ctypes.c_int

_libnrt.nrt_async_sendrecv_send_tensor.argtypes = [
    ctypes.c_void_p,    # tensor
    ctypes.c_size_t,    # offset
    ctypes.c_size_t,    # length
    ctypes.c_void_p,    # send_comm
    ctypes.POINTER(ctypes.c_void_p),  # request**
]
_libnrt.nrt_async_sendrecv_send_tensor.restype = ctypes.c_int

_libnrt.nrt_async_sendrecv_recv_tensor.argtypes = [
    ctypes.c_void_p,    # tensor
    ctypes.c_size_t,    # offset
    ctypes.c_size_t,    # length
    ctypes.c_void_p,    # recv_comm
    ctypes.POINTER(ctypes.c_void_p),  # request**
]
_libnrt.nrt_async_sendrecv_recv_tensor.restype = ctypes.c_int

_libnrt.nrt_async_sendrecv_test_request.argtypes = [
    ctypes.c_void_p,    # request
    ctypes.POINTER(ctypes.c_bool),    # done
    ctypes.POINTER(ctypes.c_size_t),  # size
]
_libnrt.nrt_async_sendrecv_test_request.restype = ctypes.c_int

_libnrt.nrt_async_sendrecv_flush.argtypes = [ctypes.c_int]
_libnrt.nrt_async_sendrecv_flush.restype = ctypes.c_int


def _pretty(num: int) -> str:
    units, val = ["B", "KB", "MB", "GB"], float(num)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024


def _check(status: int, msg: str):
    if status != NRT_SUCCESS:
        raise RuntimeError(f"NRT error {status}: {msg}")


# TCP coordination
def send_msg(sock, data: bytes):
    sock.sendall(struct.pack("!I", len(data)) + data)

def recv_msg(sock) -> bytes:
    raw = b""
    while len(raw) < 4:
        chunk = sock.recv(4 - len(raw))
        if not chunk:
            raise ConnectionError("closed")
        raw += chunk
    length = struct.unpack("!I", raw)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(65536, length - len(data)))
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return data

def send_json(sock, obj):
    send_msg(sock, json.dumps(obj).encode())

def recv_json(sock):
    return json.loads(recv_msg(sock).decode())


def wait_request(request, timeout=60.0):
    """Poll until request completes."""
    done = ctypes.c_bool(False)
    size = ctypes.c_size_t(0)
    t0 = time.time()
    while not done.value:
        status = _libnrt.nrt_async_sendrecv_test_request(request, ctypes.byref(done), ctypes.byref(size))
        _check(status, "test_request")
        if time.time() - t0 > timeout:
            raise TimeoutError("request timed out")
    return size.value


def wait_comm(comm, timeout=30.0):
    """Poll until communicator is connected."""
    done = ctypes.c_bool(False)
    t0 = time.time()
    while not done.value:
        status = _libnrt.nrt_async_sendrecv_test_comm(comm, ctypes.byref(done))
        _check(status, "test_comm")
        if time.time() - t0 > timeout:
            raise TimeoutError("comm connection timed out")


def run_sender(args):
    """Sender: allocates device tensor, sends to receiver."""
    lnc = args.lnc
    peer_ip = args.peer_ip.encode()

    # Init NRT
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(lnc)
    _check(_libnrt.nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, b"1.0", b"1.0"), "nrt_init")
    print(f"[Sender] NRT initialized, lnc={lnc}")

    # Query limits
    max_comms = ctypes.c_int(0)
    max_pending = ctypes.c_int(0)
    _check(_libnrt.nrt_async_sendrecv_get_max_num_communicators_per_lnc(ctypes.byref(max_comms)), "get_max_comms")
    _check(_libnrt.nrt_async_sendrecv_get_max_num_pending_request(ctypes.byref(max_pending)), "get_max_pending")
    print(f"[Sender] Max comms/lnc: {max_comms.value}, max pending: {max_pending.value}")

    # Init sendrecv on this lnc
    _check(_libnrt.nrt_async_sendrecv_init(lnc), "sendrecv_init")
    print("[Sender] async_sendrecv initialized")

    # TCP coordination - wait for receiver to be ready
    time.sleep(2)
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[Sender] Connecting to {args.peer_ip}:{COORD_PORT}...")
    for retry in range(30):
        try:
            conn.connect((args.peer_ip, COORD_PORT))
            break
        except ConnectionRefusedError:
            time.sleep(1)
    else:
        raise ConnectionError(f"Cannot connect to {args.peer_ip}:{COORD_PORT}")

    # Exchange info
    send_json(conn, {"lnc": lnc, "ip": args.local_ip})
    peer_info = recv_json(conn)
    peer_lnc = peer_info["lnc"]
    print(f"[Sender] Peer lnc={peer_lnc}")

    # Create send communicator
    send_comm = ctypes.c_void_p()
    _check(_libnrt.nrt_async_sendrecv_connect(peer_ip, peer_lnc, lnc, ctypes.byref(send_comm)),
           "sendrecv_connect")
    print("[Sender] Waiting for connection...")
    send_json(conn, {"status": "connecting"})
    wait_comm(send_comm)
    print("[Sender] Connected!")

    # Wait for receiver to be ready
    msg = recv_json(conn)
    assert msg["status"] == "ready"

    # Benchmark
    for sz in args.sizes:
        # Allocate device tensor
        tensor = ctypes.c_void_p()
        _check(_libnrt.nrt_tensor_allocate(
            NRT_TENSOR_PLACEMENT_DEVICE, lnc, sz, b"send_buf", ctypes.byref(tensor)
        ), f"tensor_allocate({sz})")

        # Warmup
        request = ctypes.c_void_p()
        _check(_libnrt.nrt_async_sendrecv_send_tensor(
            tensor, 0, sz, send_comm, ctypes.byref(request)
        ), "send_tensor warmup")
        wait_request(request)

        # Signal receiver warmup done
        send_json(conn, {"status": "warmup_done", "size": sz})
        msg = recv_json(conn)
        assert msg["status"] == "ready"

        # Timed iterations
        start = time.perf_counter()
        for _ in range(args.iters):
            request = ctypes.c_void_p()
            _check(_libnrt.nrt_async_sendrecv_send_tensor(
                tensor, 0, sz, send_comm, ctypes.byref(request)
            ), "send_tensor")
            wait_request(request)
        elapsed = time.perf_counter() - start

        total = sz * args.iters
        gbps = (total * 8) / elapsed / 1e9
        gbs = total / elapsed / 1e9
        per_iter = elapsed / args.iters
        print(
            f"[Sender] {_pretty(sz):>8}: "
            f"{gbps:7.2f} Gbps | {gbs:6.2f} GB/s | {per_iter:.6f} s/iter"
        )

        # Signal done
        send_json(conn, {"status": "done", "size": sz})
        msg = recv_json(conn)

        # Free tensor
        _libnrt.nrt_tensor_free(ctypes.byref(tensor))

    # Cleanup
    _check(_libnrt.nrt_async_sendrecv_close(lnc), "sendrecv_close")
    _libnrt.nrt_close()
    conn.close()
    print("[Sender] Done")


def run_receiver(args):
    """Receiver: allocates device tensor, receives from sender."""
    lnc = args.lnc
    peer_ip = args.peer_ip.encode()

    # Init NRT
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(lnc)
    _check(_libnrt.nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, b"1.0", b"1.0"), "nrt_init")
    print(f"[Receiver] NRT initialized, lnc={lnc}")

    # Init sendrecv
    _check(_libnrt.nrt_async_sendrecv_init(lnc), "sendrecv_init")
    print("[Receiver] async_sendrecv initialized")

    # TCP coordination
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", COORD_PORT))
    srv.listen(1)
    print(f"[Receiver] Listening on port {COORD_PORT}...")
    conn, addr = srv.accept()
    print(f"[Receiver] Sender connected from {addr}")

    peer_info = recv_json(conn)
    peer_lnc = peer_info["lnc"]
    sender_ip = peer_info["ip"]
    send_json(conn, {"lnc": lnc, "ip": args.local_ip})
    print(f"[Receiver] Peer lnc={peer_lnc}, ip={sender_ip}")

    # Create receive communicator
    recv_comm = ctypes.c_void_p()
    _check(_libnrt.nrt_async_sendrecv_accept(
        sender_ip.encode() if isinstance(sender_ip, str) else sender_ip,
        peer_lnc, lnc, ctypes.byref(recv_comm)
    ), "sendrecv_accept")

    # Wait for sender to initiate connection
    msg = recv_json(conn)
    assert msg["status"] == "connecting"
    print("[Receiver] Waiting for connection...")
    wait_comm(recv_comm)
    print("[Receiver] Connected!")

    # Signal ready
    send_json(conn, {"status": "ready"})

    # Benchmark
    for sz in args.sizes:
        # Allocate device tensor
        tensor = ctypes.c_void_p()
        _check(_libnrt.nrt_tensor_allocate(
            NRT_TENSOR_PLACEMENT_DEVICE, lnc, sz, b"recv_buf", ctypes.byref(tensor)
        ), f"tensor_allocate({sz})")

        # Warmup receive
        request = ctypes.c_void_p()
        _check(_libnrt.nrt_async_sendrecv_recv_tensor(
            tensor, 0, sz, recv_comm, ctypes.byref(request)
        ), "recv_tensor warmup")
        wait_request(request)
        _check(_libnrt.nrt_async_sendrecv_flush(lnc), "flush warmup")

        # Signal warmup done
        msg = recv_json(conn)
        assert msg["status"] == "warmup_done"
        send_json(conn, {"status": "ready"})

        # Timed iterations
        start = time.perf_counter()
        for _ in range(args.iters):
            request = ctypes.c_void_p()
            _check(_libnrt.nrt_async_sendrecv_recv_tensor(
                tensor, 0, sz, recv_comm, ctypes.byref(request)
            ), "recv_tensor")
            wait_request(request)
        _check(_libnrt.nrt_async_sendrecv_flush(lnc), "flush")
        elapsed = time.perf_counter() - start

        total = sz * args.iters
        gbps = (total * 8) / elapsed / 1e9
        gbs = total / elapsed / 1e9
        per_iter = elapsed / args.iters
        print(
            f"[Receiver] {_pretty(sz):>8}: "
            f"{gbps:7.2f} Gbps | {gbs:6.2f} GB/s | {per_iter:.6f} s/iter"
        )

        msg = recv_json(conn)
        assert msg["status"] == "done"
        send_json(conn, {"status": "ack"})

        # Free tensor
        _libnrt.nrt_tensor_free(ctypes.byref(tensor))

    # Cleanup
    _check(_libnrt.nrt_async_sendrecv_close(lnc), "sendrecv_close")
    _libnrt.nrt_close()
    conn.close()
    srv.close()
    print("[Receiver] Done")


def _get_local_ip():
    """Get the local IP reachable by the peer."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    p = argparse.ArgumentParser("NRT async sendrecv benchmark")
    p.add_argument("--role", choices=["send", "recv"], required=True)
    p.add_argument("--peer-ip", type=str, required=True)
    p.add_argument("--lnc", type=int, default=0, help="Logical neuron core index")
    p.add_argument("--local-ip", type=str, default=None, help="Local IP (auto-detected if omitted)")
    p.add_argument(
        "--sizes",
        type=lambda v: [int(x) for x in v.split(",") if x],
        default=[1048576, 16777216, 67108864, 268435456, 469762048],
    )
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--bootstrap-port", type=int, default=19879)
    args = p.parse_args()

    os.environ["NEURON_RT_ASYNC_SENDRECV_EXPERIMENTAL_ENABLED"] = "1"
    os.environ["NEURON_RT_ASYNC_SENDRECV_BOOTSTRAP_PORT"] = str(args.bootstrap_port)
    # ROOT_COMM_ID must point to the receiver (bootstrap root) — format is "ip:port"
    if args.role == "recv":
        root_ip = args.local_ip if args.local_ip else _get_local_ip()
    else:
        root_ip = args.peer_ip
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", f"{root_ip}:{args.bootstrap_port}")

    if args.local_ip is None:
        args.local_ip = _get_local_ip()
    print(f"Local IP: {args.local_ip}")
    print(f"NEURON_RT_ROOT_COMM_ID: {os.environ['NEURON_RT_ROOT_COMM_ID']}")

    if args.role == "send":
        run_sender(args)
    else:
        run_receiver(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] Aborted.")
        sys.exit(1)
