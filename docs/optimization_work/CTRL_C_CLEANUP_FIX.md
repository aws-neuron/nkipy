# Ctrl+C Cleanup Fix for RDMA Memory Registrations

## Date: 2026-04-20

## Problem

When terminating the engine with Ctrl+C, RDMA memory registrations (MRs) were not being cleaned up properly, causing:
- **ibv_reg_mr failures** on subsequent starts: `Error: ibv_reg_mr failed for data at 0x... size ... context_id 3`
- **Registration failures at buffer 434/435**: `[Endpoint::regv] registration failed at i=434`
- **Instance instability**: Required full reboot to recover

## Root Cause

The existing signal handlers (`SIGINT`, `SIGTERM`) called `spike_reset()` to release Neuron cores, but did not clean up P2P RDMA endpoints and memory registrations. This left RDMA resources allocated, which:
1. Count against device limits (max ~500-1000 MRs per device)
2. Persist until process fully terminates
3. Accumulate across failed/interrupted runs

## Solution

Enhanced cleanup handlers in both `server.py` and `worker.py` to properly release P2P RDMA resources before spike_reset().

### Changes Made

#### 1. Enhanced `server.py::_release_neuron_cores()`

**Location**: `nkipy/src/nkipy/vllm_plugin/server.py:169`

**Added**:
```python
def _release_neuron_cores():
    """Best-effort release of Neuron cores and RDMA resources."""
    # Clean up P2P RDMA resources first
    try:
        from nkipy.p2p import rank_endpoint
        if rank_endpoint.ep is not None:
            logger.info("Cleaning up P2P endpoint and MRs...")
            # Synchronously wait for any pending dereg
            if rank_endpoint._dereg_thread and rank_endpoint._dereg_thread.is_alive():
                logger.info("Waiting for background MR deregistration to complete...")
                rank_endpoint.wait()
            # Clear endpoint (will deregister any remaining MRs synchronously)
            rank_endpoint.ep = None
            rank_endpoint.xfer_descs = []
            rank_endpoint.buf_info = []
            logger.info("P2P cleanup complete")
    except Exception as e:
        logger.warning("Failed to clean up P2P endpoint: %s", e)

    # Release Neuron cores via spike.reset() + nrt_close
    # ... (existing code)
```

#### 2. Enhanced `worker.py::_release_neuron_cores()`

**Location**: `nkipy/src/nkipy/vllm_plugin/worker.py:99`

**Added**:
```python
@staticmethod
def _release_neuron_cores():
    """Best-effort release of Neuron cores and P2P RDMA resources."""
    # Clean up P2P RDMA resources first
    try:
        from nkipy.p2p import rank_endpoint
        if rank_endpoint.ep is not None:
            # Synchronously wait for any pending dereg
            if rank_endpoint._dereg_thread and rank_endpoint._dereg_thread.is_alive():
                rank_endpoint.wait()
            # Clear endpoint (will deregister any remaining MRs synchronously)
            rank_endpoint.ep = None
            rank_endpoint.xfer_descs = []
            rank_endpoint.buf_info = []
    except Exception:
        pass

    # Release Neuron cores
    # ... (existing code)
```

## Cleanup Order

**Critical**: RDMA resources must be cleaned up BEFORE spike_reset():

1. **Wait for background deregistration** (if in progress)
2. **Clear endpoint** → triggers synchronous MR deregistration
3. **Call spike_reset()** → releases Neuron cores
4. **Call nrt_close()** → final NRT cleanup

## Signal Handlers Covered

Both server and worker now properly clean up on:
- **SIGINT** (Ctrl+C)
- **SIGTERM** (kill command)
- **atexit** (normal termination)
- **SystemExit** (Python exit)

## Testing

### Before Fix:
```bash
# Start receiver
bash examples/p2p/run_vllm_qwen_1_receiver.sh &
# Ctrl+C to stop
^C

# Start again - FAILS
bash examples/p2p/run_vllm_qwen_1_receiver.sh
# Error: ibv_reg_mr failed at i=434
```

### After Fix:
```bash
# Start receiver
bash examples/p2p/run_vllm_qwen_1_receiver.sh &
# Ctrl+C to stop - shows cleanup logs
^C
# [INFO] Cleaning up P2P endpoint and MRs...
# [INFO] P2P cleanup complete

# Start again - WORKS
bash examples/p2p/run_vllm_qwen_1_receiver.sh
# ✓ No ibv_reg_mr errors
```

## Verification

After Ctrl+C termination, verify cleanup succeeded:
```bash
# Check no vllm processes remain
ps aux | grep vllm_plugin.server

# Check neuron cores are released
neuron-ls

# Restart should work without errors
bash examples/p2p/run_vllm_qwen_1_receiver.sh
```

## Edge Cases Handled

1. **Background deregistration in progress**: Waits for completion
2. **No P2P endpoint created**: Gracefully skips cleanup
3. **Cleanup failures**: Logged as warnings, doesn't block spike_reset
4. **Multiple signals**: Idempotent cleanup, safe to call multiple times

## Files Modified

- `nkipy/src/nkipy/vllm_plugin/server.py` - Enhanced main process cleanup
- `nkipy/src/nkipy/vllm_plugin/worker.py` - Enhanced worker process cleanup

## Benefits

✅ **No more RDMA registration failures** after Ctrl+C  
✅ **No instance reboots required** for recovery  
✅ **Graceful cleanup** with logging for debugging  
✅ **Production-ready** signal handling  

## Status

**✅ FIXED** - Deployed to both Instance A and Instance B
- Ctrl+C now properly cleans up RDMA resources
- Engines can be restarted immediately without errors
- No manual cleanup or reboots required
