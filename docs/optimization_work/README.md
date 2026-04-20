# P2P RDMA Sleep/Wake Optimization Work

This directory contains documentation from the optimization work to reduce `/sleep` latency from 46.5s to ~1s and fix the wake-up from checkpoint bug.

## Key Fixes

1. **RECEIVER_DEREG_FIX.md** - The critical fix: receiver must call `dereg_async()` after P2P transfer
2. **WAKE_FROM_CHECKPOINT_FIX.md** - Fixed garbage output after sleep/wake by properly loading weights into existing tensors
3. **CTRL_C_CLEANUP_FIX.md** - Fixed RDMA resource leakage on Ctrl+C termination
4. **DEREG_CONDITION_FIX.md** - Fixed condition logic bug in dereg_async() call

## Architecture Documentation

- **ARCHITECTURE_CONSTRAINTS.md** - MR registration strategy and server/receiver differences
- **SPIKE_RESET_MR_ANALYSIS.md** - Analysis of why spike_reset() is 60x slower with active MRs

## Historical Analysis

The other markdown files document the investigation process, hypothesis testing, and intermediate results. They are kept for reference but are not required reading.

## Final Results

- Server sleep: ~2s (was 1s, small regression due to cleanup additions)
- Receiver sleep: ~1-2s (was 46.5s with active MRs)
- Wake from checkpoint: Fixed - no longer produces garbage output
- Wake from P2P: Works correctly
- Ctrl+C cleanup: No longer leaks RDMA resources
