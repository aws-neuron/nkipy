#!/usr/bin/env python3
"""
Test MR registration/deregistration cycle and sleep latency.

Simulates the full lifecycle:
1. Load model (may register MRs depending on NKIPY_PREREGISTER_MRS)
2. Trigger MR deregistration via dereg_async()
3. Wait for deregistration to complete
4. Call spike_reset() - should be fast if dereg completed
"""
import os
import sys
import time
import torch.distributed as dist

def test_mr_dereg_sleep_cycle():
    """Test that spike_reset is fast after MR deregistration completes."""

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"\n{'='*60}")
    print(f"MR Deregistration → Sleep Cycle Test (Rank {rank}/{world_size})")
    print(f"{'='*60}\n")

    # Step 1: Initialize NRT and load model (registers MRs if NKIPY_PREREGISTER_MRS=1)
    print(f"[Rank {rank}] Step 1: Initializing NRT and loading model...")
    t0 = time.time()

    from spike import get_spike_singleton
    get_spike_singleton()
    t_nrt = time.time()
    print(f"[Rank {rank}]   NRT init: {t_nrt - t0:.3f}s")

    # Load model to trigger MR registration if enabled
    from nkipy.vllm_plugin.worker import NkipyWorker
    from vllm import LLMConfig, VllmConfig, ModelConfig, ParallelConfig, SchedulerConfig, DecodingConfig

    # Minimal config
    model_config = ModelConfig(
        model="Qwen/Qwen3-30B-A3B",
        tokenizer="Qwen/Qwen3-30B-A3B",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
        max_model_len=128,
    )

    parallel_config = ParallelConfig(
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
    )

    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=128,
        max_num_seqs=1,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
    )

    worker = NkipyWorker(vllm_config=vllm_config, local_rank=rank, rank=rank, distributed_init_method="env://")
    worker.init_device()
    t_init = time.time()
    print(f"[Rank {rank}]   Worker init: {t_init - t_nrt:.3f}s")

    # Check if we're using pre-registration
    preregister = os.environ.get("NKIPY_PREREGISTER_MRS", "0") == "1"
    print(f"[Rank {rank}]   NKIPY_PREREGISTER_MRS: {preregister}")

    # Step 2: Check for active MRs and trigger deregistration if any
    print(f"\n[Rank {rank}] Step 2: Checking for active MRs...")
    from nkipy.p2p import rank_endpoint

    has_mrs = bool(rank_endpoint.xfer_descs)
    mr_count = len(rank_endpoint.xfer_descs) if rank_endpoint.xfer_descs else 0
    print(f"[Rank {rank}]   Active MRs: {mr_count}")

    if has_mrs:
        print(f"[Rank {rank}]   Triggering dereg_async()...")
        t_dereg_start = time.time()
        rank_endpoint.dereg_async()
        print(f"[Rank {rank}]   dereg_async() started at t={t_dereg_start - t0:.3f}s")

        # Step 3: Wait for deregistration to complete
        print(f"\n[Rank {rank}] Step 3: Waiting for deregistration to complete...")
        if rank_endpoint._dereg_thread:
            rank_endpoint.wait()
            t_dereg_done = time.time()
            dereg_duration = t_dereg_done - t_dereg_start
            print(f"[Rank {rank}]   Deregistration completed in {dereg_duration:.3f}s")
        else:
            print(f"[Rank {rank}]   No dereg thread (dereg was synchronous or no MRs)")
    else:
        print(f"[Rank {rank}]   No MRs to deregister")

    # Step 4: Test spike_reset latency
    print(f"\n[Rank {rank}] Step 4: Testing spike_reset latency...")

    # Clear references like in nkipy_sleep
    worker.model_runner._nkipy_model = None
    worker.model_runner.model = None
    import gc
    gc.collect()

    from spike import spike_reset
    t_reset_start = time.time()
    spike_reset()
    t_reset_done = time.time()
    reset_duration = t_reset_done - t_reset_start

    print(f"\n{'='*60}")
    print(f"Results (Rank {rank})")
    print(f"{'='*60}")
    print(f"  MR pre-registration: {preregister}")
    print(f"  MRs detected:        {mr_count}")
    print(f"  spike_reset time:    {reset_duration:.4f}s")
    print(f"{'='*60}\n")

    # Gather results from all ranks
    dist.barrier()
    if rank == 0:
        print("\n" + "="*60)
        print("Hypothesis Verification")
        print("="*60)
        if has_mrs:
            if reset_duration < 2.0:
                print("✓ PASS: spike_reset is FAST (<2s) after MR deregistration")
                print("  Hypothesis CONFIRMED: Proper dereg timing → fast sleep")
            else:
                print("✗ FAIL: spike_reset is SLOW (≥2s) despite deregistration")
                print(f"  Expected: <2s, Got: {reset_duration:.4f}s")
        else:
            if reset_duration < 2.0:
                print("✓ PASS: spike_reset is FAST (<2s) with no MRs")
                print("  Baseline confirmed (matches earlier test)")
            else:
                print("⚠ WARNING: spike_reset SLOW (≥2s) even without MRs")
                print("  This is unexpected - may indicate other issues")
        print("="*60 + "\n")

if __name__ == "__main__":
    test_mr_dereg_sleep_cycle()
