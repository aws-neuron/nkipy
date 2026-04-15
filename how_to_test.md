# test P2P weight transfer in NKIPy plugin

## Goal

Make the p2p weight transfer in NKIPy vllm plugin workable for qwen3.
<!-- test the code changes made for latency optimizations does crash the p2p weight transfer. -->


## Instances for testing

There are two trn1 instances for tests of p2p weight transfer:
Instance A: 172.31.44.131
Instance B: 172.31.40.200

## code path

the codes on the two instances have the same path: /home/ubuntu/vllm-nkipy/nkipy
Test shell script path: /home/ubuntu/vllm-nkipy/nkipy/examples/p2p/run_vllm_qwen_1.sh

## How to test

Start separates session for the test without blocking the current kiro-cli interaction window.

Step 0: activate the python venv in /home/ubuntu/vllm-nkipy/nkipy/.venv/
Step 1: start Engine A on Instance A by running run_vllm_qwen_1.sh, which will load model weights for serving as the p2p transfer server. 
Step 2: start Engine B on Instance B by running run_vllm_qwen_1.sh without loading model weights as the p2p transfer receiver.
Step 3: Instance B generates a /wake_up endpoint from another session for p2p transfer like: curl -sf -X POST "http://localhost:8000/nkipy/wake_up"   -H "Content-Type: application/json"   -d "{\"peer_url\": \"http://172.31.44.131:8000\"}".
Step 4: Catch the log messages and fix any encountered issues, especially the RDMA registration error.
Step 5: rsync the code changes on Engine A to Engine B.
Step 5: repeat the process until the test passes

Note: you may need to kill the online model serving engines if the test passes or any error messages are caught.


## Some experiences

Experience 1: the p2p weight transfer works for tinyLlama (run_vllm_tinyllama_1.sh)
Experience 2: you can revert to a simple approach if the code changes cannot fix issues after several attempts

Note: update this section with any of your debugging experiences here in any attempts to help the future debugging and development

## Debugging Log (Kiro)

### Key Finding: EFA MR limit exhaustion in vllm plugin

The P2P weight transfer works with the standalone `torchrun` server (`~/zhuangw/nkipy/examples/p2p/server.py`) but fails in the vllm plugin due to EFA Memory Region (MR) exhaustion.

**EFA device layout on trn1.32xlarge:**
- 8 EFA devices, each shared by 4 neuron cores (e.g. cores 0-3 share `rdmap16s25`)
- Check mapping in engine logs: `GPU 0 uses device 0 (rdmap16s25)`

**What was tried (all failed for Qwen3 TP=32):**

1. **MAX_RDMA_BUFS=64 (chunked transfer):** Chunk 1 succeeded, chunk 2+ got `CQE error, status=10 (remote access error)`. Cause: `reregister()` deregistered old MRs async in a background thread — Engine A's RDMA writes hit Engine B's stale rkeys.

2. **MAX_RDMA_BUFS=1024 (single-shot):** `register_memory()` succeeded but `add_remote_endpoint()` failed with `Failed to register memory with RDMA`. 32 ranks × 434 weight MRs exhausts the per-device MR budget before connection MRs can be allocated.

3. **Deregister weight MRs before connection, re-register after:** `add_remote_endpoint()` succeeded, but even `register_memory()` for 8 MRs per rank then failed with `ibv_reg_mr failed`. The vllm runtime leaves near-zero MR headroom.

**Why the standalone server works but vllm doesn't:**
- Both use gloo backend for torch.distributed (not EFA)
- Both create 32 processes (one per neuron core)
- The vllm plugin environment (Neuron runtime init, model loading, or vllm internals) consumes more EFA MR slots per device, leaving no headroom for P2P RDMA

**Useful commands:**
- Check per-device MR errors: `cat /sys/class/infiniband/*/hw_counters/reg_mr_err`
- Check EFA devices: `ls /sys/class/infiniband/`
- Sync only changed dirs: `rsync -az .../examples/p2p/ B:.../examples/p2p/ & rsync -az .../nkipy/src/nkipy/ B:.../nkipy/src/nkipy/ & wait`
- Kill engines: `fuser -k 8000/tcp; pkill -9 -f nkipy`
- Engine B receiver script (no weights, skip cte kernels): `examples/p2p/run_vllm_qwen_1_receiver.sh`

**Next steps:**
- Investigate what consumes extra EFA MRs in the vllm plugin vs standalone server
- Consider using standalone `server.py` from `~/zhuangw/nkipy/examples/p2p/` for P2P testing
- Check if EFA MR limits can be tuned via kernel/device parameters
- TinyLlama (TP=8) may work since fewer ranks per EFA device = more MR headroom
