# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""Debug utilities for collecting intermediate tensor results.

This module provides tools for debugging tensor outputs during model execution
on Neuron hardware. It's designed to be non-intrusive: when disabled (default),
all debugging calls are no-ops with zero overhead. When enabled, it prints
tensor statistics and saves tensors to disk for analysis.

Environment Variables
---------------------
The debugger is controlled by three environment variables:

VLLM_NKIPY_COMPILE_DEBUG : str
    Set to "layer" to enable debugging, "no" to disable (default: "no").

VLLM_NKIPY_COMPILE_DEBUG_DIR : str
    Directory where debug tensors are saved (default: "./debug_tensors").
    Tensors are organized in rank-specific subdirectories.

VLLM_NKIPY_COMPILE_DEBUG_RANK : str
    Which distributed rank(s) to debug (default: "0").
    - "0", "1", etc.: Debug only the specified rank
    - "all": Debug all ranks

Quick Start
-----------
1. Enable debugging via environment variables::

    export VLLM_NKIPY_COMPILE_DEBUG=layer
    export VLLM_NKIPY_COMPILE_DEBUG_DIR=./my_debug_output
    export VLLM_NKIPY_COMPILE_DEBUG_RANK=0
    python your_script.py

2. Add debug calls in your model code::

    from vllm_nkipy.model_executor.models.debug_utils import log_tensor, increment_step

    class MyModel:
        def forward(self, hidden_states):
            # Call at the start of each forward pass to organize output by step
            increment_step()

            # Log tensors at various points
            log_tensor(hidden_states, "input_hidden_states")

            attn_output = self.attention(hidden_states)
            log_tensor(attn_output, "attn_output", layer_idx=0)

            return attn_output

API Reference
-------------
log_tensor(tensor, name, layer_idx=None)
    Log a tensor's statistics and save it to disk.

    Parameters:
        tensor : torch.Tensor
            The tensor to log.
        name : str
            Descriptive name for the tensor (e.g., "attn_output", "hidden_states").
            This name is used in console output and file names.
        layer_idx : int, optional
            Layer index for layer-specific tensors. When provided, the layer
            index is included in the output name (e.g., "layer_00_attn_output").

    Returns:
        torch.Tensor
            The original tensor unchanged (pass-through for non-intrusive integration).

    Notes:
        - No-op when debugging is disabled (safe for production code)
        - Detects and warns about NaN/Inf values
        - Computes statistics (min, max, mean, std) for float tensors

increment_step()
    Increment the step counter to organize debug output by forward pass.

    Call this at the start of each forward pass. It:
    - Increments the internal step counter
    - Resets duplicate tensor name tracking
    - Prints a visual separator in console output

    Notes:
        - No-op when debugging is disabled
        - Steps are numbered starting from 1

Output Format
-------------
When debugging is enabled, each log_tensor call prints::

    [DEBUG] Value: step_0001_layer_00_attn_output
      Type: Tensor
      Shape: (1, 128, 4096)
      Dtype: torch.bfloat16
      Device: xla:0
      Min: -0.123456
      Max: 0.654321
      Mean: 0.000123
      Std: 0.012345
      Saved to: ./debug_tensors/rank_0/step_0001_layer_00_attn_output.pt

Tensors are saved as `.pt` files that can be loaded with `torch.load()`.

Directory Structure
-------------------
Debug output is organized as follows::

    {VLLM_NKIPY_COMPILE_DEBUG_DIR}/
    ├── rank_0/
    │   ├── step_0001_input_hidden_states.pt
    │   ├── step_0001_layer_00_attn_output.pt
    │   ├── step_0001_layer_01_attn_output.pt
    │   └── ...
    ├── rank_1/
    │   └── ...
    └── ...

Example: Debugging a Transformer Layer
--------------------------------------
::

    from vllm_nkipy.model_executor.models.debug_utils import log_tensor, increment_step

    class TransformerBlock:
        def forward(self, hidden_states, positions, kv_cache):
            # Log input
            log_tensor(hidden_states, "input", layer_idx=self.layer_idx)

            # Attention
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            log_tensor(hidden_states, "post_ln1", layer_idx=self.layer_idx)

            hidden_states = self.attention(hidden_states, positions, kv_cache)
            log_tensor(hidden_states, "post_attn", layer_idx=self.layer_idx)

            hidden_states = residual + hidden_states
            log_tensor(hidden_states, "post_residual1", layer_idx=self.layer_idx)

            # MLP
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            log_tensor(hidden_states, "post_ln2", layer_idx=self.layer_idx)

            hidden_states = self.mlp(hidden_states)
            log_tensor(hidden_states, "post_mlp", layer_idx=self.layer_idx)

            hidden_states = residual + hidden_states
            log_tensor(hidden_states, "output", layer_idx=self.layer_idx)

            return hidden_states

Example: Loading Saved Tensors
------------------------------
::

    import torch

    # Load a saved tensor
    tensor = torch.load("./debug_tensors/rank_0/step_0001_layer_00_attn_output.pt")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")

    # Compare tensors from different runs
    tensor_run1 = torch.load("./run1/rank_0/step_0001_layer_00_attn_output.pt")
    tensor_run2 = torch.load("./run2/rank_0/step_0001_layer_00_attn_output.pt")
    diff = (tensor_run1.float() - tensor_run2.float()).abs()
    print(f"Max diff: {diff.max().item()}")

See Also
--------
- `log_value`: Log arbitrary values (tensors, scalars, None, etc.)
- `TensorDebugger`: The underlying debugger class for advanced usage
"""

import os
from pathlib import Path
from typing import Optional

import torch

import vllm_nkipy.envs as envs


class TensorDebugger:
    """Utility class for debugging tensor outputs in model execution.

    This debugger is controlled by environment variables:
    - VLLM_NKIPY_COMPILE_DEBUG: "layer" to enable,
      "no" to disable (default)
    - VLLM_NKIPY_COMPILE_DEBUG_DIR: Directory to save tensors
      (default: ./debug_tensors)
    - VLLM_NKIPY_COMPILE_DEBUG_RANK: Rank to debug
      ("0", "1", ... or "all")
    """

    def __init__(self):
        self.enabled = envs.VLLM_NKIPY_COMPILE_DEBUG.lower() == "layer"
        self.debug_dir = envs.VLLM_NKIPY_COMPILE_DEBUG_DIR
        self.debug_rank_str = envs.VLLM_NKIPY_COMPILE_DEBUG_RANK
        self.step_counter = 0

        # Track tensor names to handle duplicates
        self.tensor_name_counts = {}

        # Get current rank from distributed environment
        self.current_rank = self._get_current_rank()

        # Determine if this rank should log
        self.should_log = self._should_log_rank()

        if self.enabled and self.should_log:
            # Create rank-specific debug directory
            self.rank_dir = Path(self.debug_dir) / f"rank_{self.current_rank}"
            self.rank_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[DEBUG] TensorDebugger enabled for "
                f"rank {self.current_rank}. "
                f"Saving to: {self.rank_dir}"
            )

    def _get_current_rank(self) -> int:
        """Get the current process rank from distributed environment."""
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass

        # Fallback to environment variables
        rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
        try:
            return int(rank)
        except ValueError:
            return 0

    def _should_log_rank(self) -> bool:
        """Determine if the current rank should log."""
        if not self.enabled:
            return False

        if self.debug_rank_str.lower() == "all":
            return True

        target_rank = int(self.debug_rank_str)
        return self.current_rank == target_rank

    def log_value(self, value, name: str, layer_idx: Optional[int] = None):
        """
        Log value information and save to disk if debugging is enabled.
        Supports tensors, scalars (int, float, bool), None, and other types.

        Args:
            value: The value to log (tensor, int, float, bool, None, etc.)
            name: Name/description of the value
            layer_idx: Optional layer index for layer-specific values

        Returns:
            The original value (pass-through for non-intrusive integration)
        """
        if not self.enabled or not self.should_log:
            return value

        # Build descriptive name
        if layer_idx is not None:
            full_name = f"step_{self.step_counter:04d}_layer_{layer_idx:02d}_{name}"
        else:
            full_name = f"step_{self.step_counter:04d}_{name}"

        # Handle duplicate names by adding a counter
        if full_name in self.tensor_name_counts:
            self.tensor_name_counts[full_name] += 1
            full_name_with_counter = f"{full_name}_{self.tensor_name_counts[full_name]}"
        else:
            self.tensor_name_counts[full_name] = 0
            full_name_with_counter = full_name

        # Print value summary
        self._print_value_summary(value, full_name_with_counter)

        # Save value to disk
        self._save_value(value, full_name_with_counter)

        return value

    def log_tensor(
        self, tensor: torch.Tensor, name: str, layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Log tensor information and save to disk if debugging is enabled.
        This is a convenience wrapper around log_value for backward compatibility.

        Args:
            tensor: The tensor to log
            name: Name/description of the tensor
            layer_idx: Optional layer index for layer-specific tensors

        Returns:
            The original tensor (pass-through for non-intrusive integration)
        """
        return self.log_value(tensor, name, layer_idx)

    def _print_value_summary(self, value, name: str):
        """Print a summary of the value (tensor or scalar)."""
        print(f"\n[DEBUG] Value: {name}")

        if isinstance(value, torch.Tensor):
            print("  Type: Tensor")
            print(f"  Shape: {tuple(value.shape)}")
            print(f"  Dtype: {value.dtype}")
            print(f"  Device: {value.device}")

            # Convert to float for statistics if needed
            if value.dtype in [
                torch.bfloat16,
                torch.float16,
                torch.float32,
                torch.float64,
            ]:
                tensor_float = value.float()
                print(f"  Min: {tensor_float.min().item():.6f}")
                print(f"  Max: {tensor_float.max().item():.6f}")
                print(f"  Mean: {tensor_float.mean().item():.6f}")
                print(f"  Std: {tensor_float.std().item():.6f}")

                # Check for NaN or Inf
                if torch.isnan(tensor_float).any():
                    print("  WARNING: Contains NaN values!")
                if torch.isinf(tensor_float).any():
                    print("  WARNING: Contains Inf values!")
            else:
                print(f"  (Statistics not computed for dtype {value.dtype})")
        elif value is None:
            print("  Type: NoneType")
            print("  Value: None")
        elif isinstance(value, bool):
            print("  Type: bool")
            print(f"  Value: {value}")
        elif isinstance(value, int):
            print("  Type: int")
            print(f"  Value: {value}")
        elif isinstance(value, float):
            print("  Type: float")
            print(f"  Value: {value}")
        else:
            print(f"  Type: {type(value).__name__}")
            print(f"  Value: {value}")

    def _print_tensor_summary(self, tensor: torch.Tensor, name: str):
        """Print a summary of the tensor. Kept for backward compatibility."""
        self._print_value_summary(tensor, name)

    def _save_value(self, value, name: str):
        """Save value to disk (tensor or scalar)."""
        try:
            save_path = self.rank_dir / f"{name}.pt"

            if isinstance(value, torch.Tensor):
                # Save tensor as before
                torch.save(value.detach().cpu(), save_path)
            else:
                # Save scalar/other types
                torch.save(value, save_path)

            print(f"  Saved to: {save_path}")
        except Exception as e:
            print(f"  ERROR saving value: {e}")

    def _save_tensor(self, tensor: torch.Tensor, name: str):
        """Save tensor to disk. Kept for backward compatibility."""
        self._save_value(tensor, name)

    def increment_step(self):
        """Increment the step counter (call this at the start of each forward pass)."""
        if self.enabled and self.should_log:
            self.step_counter += 1
            # Reset tensor name counts for the new step
            self.tensor_name_counts.clear()
            print(f"\n{'=' * 60}")
            print(f"[DEBUG] Starting step {self.step_counter}")
            print(f"{'=' * 60}")


# Global singleton instance
_debugger = None


def get_debugger() -> TensorDebugger:
    """Get the global TensorDebugger instance."""
    global _debugger
    if _debugger is None:
        _debugger = TensorDebugger()
    return _debugger


def log_value(value, name: str, layer_idx: Optional[int] = None):
    """
    Convenience function to log a value using the global debugger.
    Supports tensors, scalars (int, float, bool), None, and other types.

    Args:
        value: The value to log
        name: Name/description of the value
        layer_idx: Optional layer index

    Returns:
        The original value (pass-through)
    """
    return get_debugger().log_value(value, name, layer_idx)


def log_tensor(
    tensor: torch.Tensor, name: str, layer_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Convenience function to log a tensor using the global debugger.
    This is a wrapper around log_value for backward compatibility.

    Args:
        tensor: The tensor to log
        name: Name/description of the tensor
        layer_idx: Optional layer index

    Returns:
        The original tensor (pass-through)
    """
    return get_debugger().log_tensor(tensor, name, layer_idx)


def increment_step():
    """Convenience function to increment the step counter."""
    get_debugger().increment_step()
