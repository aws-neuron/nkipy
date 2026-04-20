# Code Sync Status - Instance A → Instance B

## Date: 2026-04-20 17:40 UTC

## ✅ Synced Files

### Core Source Code
- ✅ `nkipy/src/nkipy/vllm_plugin/worker.py` - Enhanced with RDMA cleanup on Ctrl+C
- ✅ `nkipy/src/nkipy/vllm_plugin/server.py` - Enhanced with RDMA cleanup on Ctrl+C  
- ✅ `nkipy/src/nkipy/p2p/transfer.py` - Latest P2P transfer logic with dereg_async
- ✅ `nkipy/src/nkipy/p2p/endpoint.py` - RDMA endpoint with background deregistration
- ✅ `nkipy/src/nkipy/runtime/device_tensor.py` - DeviceTensor implementation
- ✅ `nkipy/src/nkipy/vllm_plugin/models/qwen3.py` - Qwen3 model with tok_embedding opt
- ✅ `nkipy/src/nkipy/vllm_plugin/models/llama.py` - Llama model with tok_embedding opt

### Test Scripts
- ✅ `test_e2e_dereg_hypothesis.sh` - End-to-end hypothesis test (2 scenarios)
- ✅ `run_remote_hypothesis_test.sh` - Remote test runner
- ✅ `cleanup_instance_b.sh` - Cleanup script
- ✅ `examples/p2p/run_vllm_qwen_1_receiver.sh` - Receiver startup script
- ✅ `examples/p2p/test_p2p_vllm_qwen.sh` - P2P test script

### Documentation
- ✅ `CTRL_C_CLEANUP_FIX.md` - **NEW** - Documents Ctrl+C cleanup fix
- ✅ `HYPOTHESIS_VERIFICATION.md` - Baseline test results
- ✅ `SPIKE_RESET_MR_ANALYSIS.md` - Root cause analysis
- ✅ `P2P_SLEEP_OPTIMIZATION_RESULTS.md` - Previous test results
- ✅ `REMOTE_EXECUTION_GUIDE.md` - How to run commands on Instance B correctly
- ✅ All other *.md documentation files

## Key Changes Since Last Sync

### 1. Ctrl+C Cleanup Fix (CRITICAL)
**Problem**: Engines couldn't be restarted after Ctrl+C due to RDMA MRs not being cleaned up
**Solution**: Enhanced signal handlers to properly deregister RDMA MRs before spike_reset()
**Files**: `worker.py`, `server.py`
**Status**: ✅ FIXED - Both instances updated

### 2. Hypothesis Test Scripts
**Created**: Complete end-to-end test for MR deregistration hypothesis
**Files**: `test_e2e_dereg_hypothesis.sh`, `run_remote_hypothesis_test.sh`
**Status**: ✅ Ready to run

## Instance Configuration

### Instance A (172.31.44.131) - Server
- **Role**: P2P sender with checkpoint
- **Status**: Running (started earlier)
- **Endpoint**: http://172.31.44.131:8000
- **Code**: Latest (17:40 UTC)

### Instance B (172.31.40.200) - Receiver  
- **Role**: P2P receiver, starts sleeping
- **Status**: Ready to start
- **Endpoint**: http://172.31.40.200:8000
- **Code**: Latest (17:40 UTC) - **SYNCED**

## Ready to Test

Both instances now have:
- ✅ Latest code with Ctrl+C cleanup fix
- ✅ MR deregistration optimization (dereg_async)
- ✅ Conditional wait logic in /sleep
- ✅ Test scripts and documentation

## Next Steps

1. **Start receiver on Instance B**:
   ```bash
   ssh ubuntu@172.31.40.200
   cd /home/ubuntu/vllm-nkipy/nkipy
   source .venv/bin/activate
   bash examples/p2p/run_vllm_qwen_1_receiver.sh > receiver.log 2>&1 &
   ```

2. **Wait for receiver ready** (~2-3 min):
   ```bash
   tail -f receiver.log
   # Wait for "Application startup complete"
   ```

3. **Run hypothesis test**:
   ```bash
   bash test_e2e_dereg_hypothesis.sh
   ```

## Verification Commands

### Check sync status:
```bash
# From Instance A
ssh ubuntu@172.31.40.200 "stat -c '%y' /home/ubuntu/vllm-nkipy/nkipy/nkipy/src/nkipy/vllm_plugin/worker.py"
# Should show: 2026-04-20 17:38:...
```

### Verify Ctrl+C fix:
```bash
# On Instance B, check worker.py has cleanup
ssh ubuntu@172.31.40.200 "grep -A 5 'Clean up P2P RDMA resources' /home/ubuntu/vllm-nkipy/nkipy/nkipy/src/nkipy/vllm_plugin/worker.py"
# Should show the enhanced _release_neuron_cores method
```

## Sync Command Reference

For future syncs from Instance A to Instance B:

```bash
# Sync nkipy source
rsync -avz --delete nkipy/src/nkipy/ ubuntu@172.31.40.200:/home/ubuntu/vllm-nkipy/nkipy/nkipy/src/nkipy/

# Sync P2P scripts
rsync -avz examples/p2p/ ubuntu@172.31.40.200:/home/ubuntu/vllm-nkipy/nkipy/examples/p2p/

# Sync test scripts and docs
rsync -avz *.sh *.md ubuntu@172.31.40.200:/home/ubuntu/vllm-nkipy/nkipy/
```

## Status: ✅ READY FOR TESTING

Both instances are synchronized and ready for the end-to-end hypothesis test.
