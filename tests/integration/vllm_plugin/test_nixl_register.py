#!/usr/bin/env python3
"""Test: NIXL LIBFABRIC backend can register NeuronCore HBM on Trn2.

Verifies that NIXL's LIBFABRIC plugin can directly register device memory
(allocated via spike/NRT) with EFA, bypassing host-staged DMA+RDMA.

Prerequisites:
  - NIXL built with LIBFABRIC plugin (from /fsx/zhuangw/nixl)
  - Running on trn2 instance with NeuronCores available

Usage:
    python tests/integration/vllm_plugin/test_nixl_register.py [--core 0]
"""

import argparse
import time

import torch
import torch_neuronx  # noqa: F401 — registers privateuseone device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=int, default=0, help="NeuronCore ID")
    parser.add_argument("--size-mb", type=int, default=64, help="Tensor size in MB")
    parser.add_argument("--num-tensors", type=int, default=4,
                        help="Number of tensors to register in batch test")
    args = parser.parse_args()

    core_id = args.core
    size_bytes = args.size_mb * 1024 * 1024

    print(f"[1] Initializing Neuron runtime via spike...")
    t0 = time.time()
    from spike import get_spike_singleton
    spike = get_spike_singleton()
    print(f"    Done ({time.time() - t0:.2f}s)")

    print(f"[2] Allocating {args.size_mb} MB on NeuronCore {core_id} via spike...")
    t0 = time.time()
    nrt_tensor = spike.allocate_tensor(size_bytes, core_id, "nixl_test")
    print(f"    Done ({time.time() - t0:.2f}s)")
    print(f"    va = 0x{nrt_tensor.va:x}")
    print(f"    size = {nrt_tensor.size} bytes")

    print(f"[3] Creating NIXL agent with LIBFABRIC backend...")
    t0 = time.time()
    from nixl._api import nixl_agent, nixl_agent_config

    config = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        listen_port=18500,
        backends=["LIBFABRIC"],
    )
    agent = nixl_agent(f"test_core_{core_id}", config)
    print(f"    Done ({time.time() - t0:.2f}s)")
    print(f"    Plugins: {agent.get_plugin_list()}")
    print(f"    LIBFABRIC mem types: {agent.get_backend_mem_types('LIBFABRIC')}")

    print(f"[4] Registering device memory (spike VA) with LIBFABRIC...")
    t0 = time.time()
    mem_desc = [(nrt_tensor.va, nrt_tensor.size, core_id, "")]
    reg_descs = agent.register_memory(mem_desc, mem_type="VRAM", backends=["LIBFABRIC"])
    elapsed = time.time() - t0
    print(f"    SUCCESS! Registration took {elapsed:.3f}s")

    print(f"\n[5] Batch test: {args.num_tensors} x {args.size_mb} MB tensors...")
    t0 = time.time()
    tensors = []
    mem_descs = []
    for i in range(args.num_tensors):
        t = spike.allocate_tensor(size_bytes, core_id, f"nixl_batch_{i}")
        tensors.append(t)
        mem_descs.append((t.va, t.size, core_id, ""))
    reg_descs2 = agent.register_memory(mem_descs, mem_type="VRAM", backends=["LIBFABRIC"])
    elapsed = time.time() - t0
    print(f"    SUCCESS! {args.num_tensors} tensors registered in {elapsed:.3f}s")

    print(f"\n[6] Deregistering...")
    agent.deregister_memory(reg_descs, backends=["LIBFABRIC"])
    agent.deregister_memory(reg_descs2, backends=["LIBFABRIC"])
    for t in tensors:
        spike.free_tensor(t)
    spike.free_tensor(nrt_tensor)
    print(f"    Done")

    print(f"\n=== ALL TESTS PASSED ===")
    print(f"NIXL LIBFABRIC can register NeuronCore HBM (via spike VA) on Trn2.")


if __name__ == "__main__":
    main()
