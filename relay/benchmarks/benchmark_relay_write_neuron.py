from __future__ import annotations
import argparse, sys, time, socket, struct
from typing import List
import os

fifo_blob_size = 64  # bytes

# parse_metadata is now provided by the C++ layer via _relay.Endpoint.parse_metadata()


def _check_ptr_valid(ptr: int) -> bool:
    if ptr == 0:
        return False
    with open("/proc/self/maps", "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            start, end = parts[0].split("-")
            if int(start, 16) <= ptr < int(end, 16):
                return parts[1].startswith("rw")
    return False


def _make_buffer(n_bytes: int, device: str, nc: int):
    n = n_bytes // 4
    if device == "trn":
        buf = torch.ones(n, dtype=torch.float32).to(f"privateuseone:{nc}")
    else:
        buf = torch.ones(n, dtype=torch.float32)
    ptr = buf.data_ptr()

    if _check_ptr_valid(ptr):
        # 1/ buf is created on CPU, or
        # 2/ buf is created on Trn with eager mode and NEURON_RT_MAP_HBM = 1
        return buf, ptr
    else:
        buf_nrt = _relay.create_nrt_tensor(nc_idx=nc, size_bytes=n_bytes)
        buf_nrt.copy_(buf)
        ptr_nrt = buf_nrt.data_ptr()
        return buf_nrt, ptr_nrt


def _pretty(num: int):
    units, val = ["B", "KB", "MB", "GB"], float(num)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024


def _run_server(args, ep, remote_metadata):
    peer = 0
    print("[Server] Waiting for connection …")
    ok, r_ip, r_nc, conn_id = ep.accept()
    assert ok
    print(f"[Server] Connected to {r_ip} (NC {r_nc}) id={conn_id}")

    for sz in args.sizes:
        size_per_block = sz // args.num_iovs
        buf_v = []
        ptr_v = []
        mr_id_v = []
        size_v = []
        for _ in range(args.num_iovs):
            # nc=0 because only 1 neuron core is visible
            buf, ptr = _make_buffer(size_per_block, args.device, nc=0)
            ok, mr_id = ep.reg(ptr, size_per_block)
            assert ok
            buf_v.append(buf)
            ptr_v.append(ptr)
            mr_id_v.append(mr_id)
            size_v.append(size_per_block)
        # Use advertisev to advertise all blocks at once
        ok, fifo_blob_v = ep.advertisev(conn_id, mr_id_v, ptr_v, size_v, args.num_iovs)
        assert ok
        assert all(len(fifo_blob) == fifo_blob_size for fifo_blob in fifo_blob_v)
        # Send all fifo_blobs to peer
        for fifo_blob in fifo_blob_v:
            dist.send(torch.ByteTensor(list(fifo_blob)), dst=peer)
        dist.barrier()
    print("[Server] Benchmark complete")


def _run_client(args, ep, remote_metadata):
    peer = 1
    ip, port, r_nc = _relay.Endpoint.parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(ip, r_nc, remote_port=port)
    assert ok
    print(f"[Client] Connected to {ip}:{port} id={conn_id}")

    for sz in args.sizes:
        size_per_block = sz // args.num_iovs
        buf_v = []
        ptr_v = []
        mr_id_v = []
        size_v = []
        for _ in range(args.num_iovs):
            # nc=0 because only 1 neuron core is visible
            buf, ptr = _make_buffer(size_per_block, args.device, nc=0)
            ok, mr_id = ep.reg(ptr, size_per_block)
            assert ok
            buf_v.append(buf)
            ptr_v.append(ptr)
            mr_id_v.append(mr_id)
            size_v.append(size_per_block)

        fifo_blob_v = []
        for _ in range(args.num_iovs):
            fifo_blob = torch.zeros(fifo_blob_size, dtype=torch.uint8)
            dist.recv(fifo_blob, src=peer)
            fifo_blob_v.append(bytes(fifo_blob.tolist()))

        # Warmup
        ep.writev(conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, args.num_iovs)

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            ep.writev(
                conn_id,
                mr_id_v,
                ptr_v,
                size_v,
                fifo_blob_v,
                args.num_iovs,
            )
            total += sum(size_v)
        elapsed = time.perf_counter() - start
        print(
            f"[Client] {_pretty(sz):>8} : "
            f"{(total*8)/elapsed/1e9:6.2f} Gbps | "
            f"{total/elapsed/1e9:6.2f} GB/s | "
            f"{elapsed/args.iters:6.6f} s"
        )
        dist.barrier()
    print("[Client] Benchmark complete")


def parse_sizes(v: str) -> List[int]:
    try:
        return [int(x) for x in v.split(",") if x]
    except ValueError:
        raise argparse.ArgumentTypeError("bad --sizes")


def main():
    p = argparse.ArgumentParser("Relay WRITE benchmark (one-sided)")
    p.add_argument(
        "--local-nc-idx", type=int, default=5, help="2 nodes x 1 rank/node, e.g. 5"
    )
    p.add_argument(
        "--local-nc-idx-group",
        type=parse_sizes,
        default="",
        help="1 node x 2 ranks/node, e.g. 4,5",
    )
    p.add_argument("--device", choices=["cpu", "trn"], default="trn")
    p.add_argument(
        "--sizes",
        type=parse_sizes,
        default=[
            256,
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            10485760,
            67108864,
            104857600,
        ],
    )
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--num-iovs",
        type=int,
        default=1,
        help="Number of iovs to write in a single call",
    )
    args = p.parse_args()

    # 1 node x 2 ranks/node or 2 nodes x 1 rank/node
    if len(args.local_nc_idx_group) > 0:
        local_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        nc_idx = args.local_nc_idx_group[local_rank]
    else:
        nc_idx = args.local_nc_idx

    global torch, dist, _relay
    import torch.distributed as dist
    import torch
    import torch_neuronx

    os.environ["NEURON_RT_VISIBLE_CORES"] = str(nc_idx)
    os.environ["UCCL_RCMODE"] = "1"
    try:
        from relay import _relay
    except ImportError:
        sys.stderr.write("Failed to import relay._relay\n")
        raise

    print("Sizes:", ", ".join(_pretty(s) for s in args.sizes))

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    ep = _relay.Endpoint(nc_idx)
    local_metadata = ep.get_metadata()

    if rank == 0:
        dist.send(torch.ByteTensor(list(local_metadata)), dst=1)
        remote_metadata_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_metadata_tensor, src=1)
        remote_metadata = bytes(remote_metadata_tensor.tolist())
    else:
        remote_metadata_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_metadata_tensor, src=0)
        dist.send(torch.ByteTensor(list(local_metadata)), dst=0)
        remote_metadata = bytes(remote_metadata_tensor.tolist())

    if rank == 0:
        _run_client(args, ep, remote_metadata)
    elif rank == 1:
        _run_server(args, ep, remote_metadata)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] Aborted.")
        sys.exit(1)
