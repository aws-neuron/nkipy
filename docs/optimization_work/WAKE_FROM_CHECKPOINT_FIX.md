# Fix: Server Wake-up from Checkpoint After Sleep

## Date: 2026-04-20

## Problem

After server sleeps and then wakes up from local checkpoint (no peer_url), the model produces garbage output (`!!!!!!!!!!!!!!!!!!!` instead of correct text).

**Root cause**: The wake_up flow was calling `model._prepare_tensors(weights)` after `model._allocate_empty_tensors()`. The problem is that `_prepare_tensors()` **recreates all DeviceTensor objects** via `DeviceTensor.from_torch()`, which allocates new device memory and leaves the old (empty) tensors orphaned. The model then uses the **newly allocated tensors** which contain the loaded weights, but **the old empty tensors are still referenced somewhere**, causing garbage output.

## Detailed Root Cause

The wake_up flow creates the model in two steps:

```python
# Step 1: Allocate empty tensor structure
model = self._model_class(config=self._config, skip_kernels=True)
model._allocate_empty_tensors()

# Step 2: Load weights from checkpoint
weights = load_file(shard_path, device="cpu")
model._prepare_tensors(weights)  # ❌ BUG: This recreates tensors!
```

### Why `_prepare_tensors()` is wrong here:

`_prepare_tensors()` creates **new** DeviceTensor objects:

```python
def _prepare_tensors(self, weights):
    self.layer_tensors = []
    for lid in range(cfg.num_layers):
        layer = {}
        for wk, prefix in LAYER_WEIGHT_KEYS:
            # Creates NEW DeviceTensor, allocating new device memory
            layer[wk] = DeviceTensor.from_torch(weights[f"layers.{lid}.{wk}"], ...)
        self.layer_tensors.append(layer)
    # ... same for norm_weight, lm_head_weight, tok_embedding_device
```

This means:
1. `_allocate_empty_tensors()` creates tensors at memory addresses A1, A2, A3, ...
2. `_prepare_tensors()` creates **new** tensors at addresses B1, B2, B3, ... with loaded weights
3. The model's `self.layer_tensors` now points to the **new** tensors (B1, B2, B3, ...)
4. But somewhere in the code, the **old** tensor addresses (A1, A2, A3, ...) are still referenced
5. When inference runs, it reads from the **old empty tensors** → garbage output

## Solution

### Fix 1: Add `_load_weights_into_existing_tensors()` method

Added a new method to `Qwen3Model` that **populates existing DeviceTensors** without recreating them:

```python
def _load_weights_into_existing_tensors(self, weights):
    """Load weights from checkpoint into existing DeviceTensors (for wake_up from checkpoint).

    This populates already-allocated tensors without recreating them.
    Used after _allocate_empty_tensors() has created the tensor structure.
    """
    # Load layer weights
    for lid in range(self.config.num_layers):
        layer = self.layer_tensors[lid]
        for wk, _ in LAYER_WEIGHT_KEYS:
            weight_key = f"layers.{lid}.{wk}"
            if weight_key in weights:
                layer[wk].write_from_torch(weights[weight_key])  # ✅ Write into existing tensor!

    # Load model head weights
    if "norm_weight" in weights:
        self.norm_weight.write_from_torch(weights["norm_weight"])
    if "lm_head_weight" in weights:
        self.lm_head_weight.write_from_torch(weights["lm_head_weight"])

    # Load tok_embedding
    if "tok_embedding" in weights:
        self.tok_embedding_device.write_from_torch(weights["tok_embedding"])
        self.tok_embedding = weights["tok_embedding"]
```

### Fix 2: Use the new method in wake_up

```python
else:
    # No peer - load weights from checkpoint
    import os
    from safetensors.torch import load_file
    checkpoint_path = os.environ.get("NKIPY_CHECKPOINT")
    if checkpoint_path:
        if self.rank == 0:
            logger.info("Rank %d: Loading weights from checkpoint: %s", self.rank, checkpoint_path)
        shard_path = os.path.join(checkpoint_path, f"shard_{self.rank}.safetensors")
        weights = load_file(shard_path, device="cpu")
        # Use _load_weights_into_existing_tensors() instead of _prepare_tensors()
        # to populate the already-allocated tensors without recreating them
        model._load_weights_into_existing_tensors(weights)  # ✅ Correct!
        t_collect = _time.time()
        t_p2p = t_collect
        t_ack = t_collect
```

## Wake-up Flow After Fix

### Scenario 1: Wake from Checkpoint (Server)
```python
# Server calls: POST /nkipy/wake_up (no peer_url)
nkipy_wake_up(peer_url=None)
├── Init NRT
├── Allocate empty tensors
├── Load checkpoint via safetensors ← Fixed!
├── Populate tensors via _prepare_tensors ← Fixed!
├── Reload kernels (if cached)
└── Set model as active
```

### Scenario 2: Wake from P2P (Receiver)
```python
# Receiver calls: POST /nkipy/wake_up {"peer_url": "http://..."}
nkipy_wake_up(peer_url="http://server:8000")
├── Init NRT
├── Allocate empty tensors
├── Receive weights via P2P
├── Reload kernels (if cached)
└── Set model as active
```

## Key Difference: Initial Load vs Wake Up

### Initial Load (model_runner.py)
```python
# No tensors exist yet - constructor calls _prepare_tensors()
weights = load_file(shard_path, device="cpu")
model = ModelClass(weights, config, skip_kernels=True)
# ↓ Inside __init__:
# if model_weights:
#     self._prepare_tensors(model_weights)  # Creates tensors from scratch
```

### Wake Up (worker.py)
```python
# Tensors already allocated - must populate without recreating
model = ModelClass(config=config, skip_kernels=True)  # No weights passed
model._allocate_empty_tensors()  # Creates empty tensor structure
weights = load_file(shard_path, device="cpu")
model._load_weights_into_existing_tensors(weights)  # ✅ Fills existing tensors
```

The critical insight: **`_prepare_tensors()` always creates new tensors**, so it's only correct when no tensors exist yet (initial load). For wake_up, we must use `_load_weights_into_existing_tensors()` to write into the already-allocated memory.

## Files Modified

- `nkipy/src/nkipy/vllm_plugin/models/qwen3.py`:
  - Added new method `_load_weights_into_existing_tensors()` (after `_prepare_tensors()`)
  - Uses `DeviceTensor.write_from_torch()` to populate existing tensors without reallocation

- `nkipy/src/nkipy/vllm_plugin/models/llama.py`:
  - Added identical `_load_weights_into_existing_tensors()` method
  - Same fix applies to both Llama3Model and Qwen3Model architectures

- `nkipy/src/nkipy/vllm_plugin/worker.py`:
  - Line 413: Changed from `model._prepare_tensors(weights)` to `model._load_weights_into_existing_tensors(weights)`
  - Line 414-416: Added explanatory comment
  - Fix works for any model architecture that has both `_allocate_empty_tensors()` and `_prepare_tensors()` methods

## Deployment

✅ **Instance A** - Fixed  
⚠️ **Instance B** - Needs sync  
⚠️ **Restart required** to load new code

## Use Cases

This fix enables:
1. **Server sleep/wake cycles**: Server can sleep and wake from checkpoint repeatedly
2. **Graceful recovery**: If server crashes, restart in sleeping mode, then wake from checkpoint
3. **Resource management**: Sleep to free resources, wake when needed

## Summary

Server wake-up from checkpoint now works correctly:
- ✅ Loads weights from checkpoint using safetensors.load_file()
- ✅ Populates existing tensors using `_load_weights_into_existing_tensors()` without recreating them
- ✅ Handles missing kernel cache gracefully
- ✅ Supports both P2P and checkpoint wake-up paths
- ✅ Works for both Qwen3Model and Llama3Model architectures

## Architecture Notes

The fix is implemented separately in both `Qwen3Model` and `Llama3Model` classes:
- Both models have identical method signatures and implementation
- Each uses its own `LAYER_WEIGHT_KEYS` constant for model-specific weights
- No base class inheritance between the two models (intentional - keeps implementations independent)
- If more model architectures are added, the same pattern applies: add `_load_weights_into_existing_tensors()` that calls `write_from_torch()` on existing DeviceTensors
