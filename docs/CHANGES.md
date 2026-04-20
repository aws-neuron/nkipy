# P2P Sleep/Wake Optimization & Fixes

This document summarizes the changes made to optimize `/sleep` latency and fix wake-up from checkpoint.

## Summary

- **Sleep latency**: Reduced from 46.5s to ~1-2s by fixing RDMA MR deregistration
- **Wake from checkpoint**: Fixed garbage output bug 
- **Ctrl+C cleanup**: Fixed RDMA resource leakage on termination

## Key Changes

### 1. RDMA MR Deregistration Fix (CRITICAL)

**Problem**: Receiver never called `dereg_async()` after P2P transfer, leaving 435 MRs active. This caused `spike_reset()` to be 60x slower (18.5s instead of 0.3s).

**Solution**: Added `dereg_async()` call in `receive_from_peer()` after transfer completes.

**Files changed**:
- `nkipy/src/nkipy/p2p/transfer.py`: Added `ep.dereg_async()` at end of `receive_from_peer()`

### 2. Wake from Checkpoint Fix

**Problem**: After sleep/wake cycle loading from checkpoint, model produced garbage output (`!!!!!!!!!!!!!!!!!!!` instead of correct text).

**Root cause**: `_prepare_tensors()` creates new DeviceTensor objects, so calling it after `_allocate_empty_tensors()` left old empty tensors orphaned while creating new ones with loaded weights. The model referenced the wrong set of tensors.

**Solution**: Added `_load_weights_into_existing_tensors()` method that uses `DeviceTensor.write_from_torch()` to populate existing tensors without recreating them.

**Files changed**:
- `nkipy/src/nkipy/vllm_plugin/models/qwen3.py`: Added `_load_weights_into_existing_tensors()` method
- `nkipy/src/nkipy/vllm_plugin/models/llama.py`: Added identical method
- `nkipy/src/nkipy/vllm_plugin/worker.py`: Changed wake_up to use new method

### 3. Ctrl+C RDMA Cleanup

**Problem**: Terminating engine with Ctrl+C left RDMA MRs registered, causing "ibv_reg_mr failed" errors on restart.

**Solution**: Enhanced `_release_neuron_cores()` to wait for dereg threads and clear RDMA resources before spike_reset.

**Files changed**:
- `nkipy/src/nkipy/vllm_plugin/server.py`: Enhanced `_release_neuron_cores()` with RDMA cleanup
- `nkipy/src/nkipy/vllm_plugin/worker.py`: Enhanced `_release_neuron_cores()` with RDMA cleanup

### 4. Sleep Latency Optimization

**Changes**:
- Wait for dereg completion before `spike_reset()` to ensure fast cleanup
- Removed pre-registered MR path from receiver (receivers always use on-demand registration)
- Made MR pre-registration configurable via `NKIPY_PREREGISTER_MRS` env var (default: off)

**Files changed**:
- `nkipy/src/nkipy/vllm_plugin/worker.py`: 
  - Added dereg wait before spike_reset
  - Made pre-registration configurable
  - Updated sleep latency breakdown logging
- `nkipy/src/nkipy/p2p/transfer.py`: Removed pre-registration path from receiver

## Architecture Notes

### MR Registration Strategy

**Server (sender)**:
- Can pre-register MRs at startup (`NKIPY_PREREGISTER_MRS=1`) OR
- Use on-demand registration during push (`NKIPY_PREREGISTER_MRS=0`, default)
- With pre-registration: fast wake_up, slower sleep (~20s if no wait)
- With on-demand: +2s wake_up, fast sleep (~2s)

**Receiver**:
- NEVER pre-registers (architectural constraint)
- Always uses on-demand chunked registration
- MUST call `dereg_async()` after receive

### Sleep/Wake Flow

**Sleep**:
1. Clear model references
2. GC collect
3. Wait for dereg completion (if in progress)
4. spike_reset() — now fast (~1s) because MRs are deregistered
5. Clear endpoint state

**Wake from P2P**:
1. Init NRT
2. Allocate empty tensors
3. Receive weights via P2P RDMA
4. Call dereg_async() to start background cleanup
5. Reload kernels from cache
6. Convert tok_embedding to CPU tensor

**Wake from Checkpoint**:
1. Init NRT
2. Allocate empty tensors
3. Load weights from safetensors
4. **Populate existing tensors** via `_load_weights_into_existing_tensors()`
5. Reload kernels from cache
6. Set tok_embedding

## Testing

All fixes verified on:
- Instance A (172.31.44.131): Server role
- Instance B (172.31.40.200): Receiver role
- Model: Qwen3-30B-A3B (MoE, 435 weight tensors)
- Configuration: TP=8 and TP=32

## Documentation

Key documentation files in `docs/optimization_work/`:
- `RECEIVER_DEREG_FIX.md` - The critical dereg_async() fix
- `WAKE_FROM_CHECKPOINT_FIX.md` - Detailed explanation of the wake bug
- `CTRL_C_CLEANUP_FIX.md` - RDMA resource cleanup on termination
- `ARCHITECTURE_CONSTRAINTS.md` - MR registration strategy
- `SPIKE_RESET_MR_ANALYSIS.md` - Why active MRs cause 60x slowdown

Test scripts preserved in `test_scripts/` for reference.
