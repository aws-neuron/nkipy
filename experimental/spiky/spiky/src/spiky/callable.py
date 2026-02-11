# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NKIPyCallable - Dynamic shape callable with bucket selection and JIT compilation.

This module provides the bridge between torch-to-nkipy and spiky runtime:
- torch-to-nkipy provides: compiler_fn callback for bucket compilation
- spiky provides: bundle registration, bucket selection, padded execution
"""

import atexit
import logging
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import spiky
from spiky.utils.dynamic_shapes import DynamicSpec
from spiky.utils.dynamic_shapes import select_bucket as _select_bucket
from spiky.utils.ntff_meta import NtffMeta

logger = logging.getLogger(__name__)

_KEEP_ON_DEVICE_WARNED = False

# dtype string (from C++ DeviceTensor) -> numpy dtype
_DTYPE_STR_TO_NUMPY = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
}


def _device_tensor_to_torch(device_tensor, device: torch.device) -> torch.Tensor:
    """Convert a C++ DeviceTensor to a torch.Tensor via host-side copy."""
    raw_bytes = device_tensor.read_to_bytes()
    np_dtype = _DTYPE_STR_TO_NUMPY.get(device_tensor.dtype)

    if np_dtype is None and device_tensor.dtype == "bfloat16":
        arr = np.frombuffer(bytes(raw_bytes), dtype=np.uint16).reshape(
            device_tensor.shape
        )
        return torch.from_numpy(arr.copy()).view(torch.bfloat16).to(device)

    if np_dtype is None:
        raise ValueError(f"Unsupported DeviceTensor dtype: {device_tensor.dtype}")

    arr = np.frombuffer(bytes(raw_bytes), dtype=np_dtype).reshape(device_tensor.shape)
    return torch.from_numpy(arr.copy()).to(device)


# Track live callables for cleanup at interpreter shutdown.
# Using weak refs so callables can still be garbage-collected normally.
_live_callables: Dict[int, weakref.ref] = {}


def _cleanup_all_callables():
    """Flush and unregister all live callables before interpreter shutdown."""
    for ref in list(_live_callables.values()):
        obj = ref()
        if obj is not None:
            try:
                obj.close()
            except Exception:
                pass
    _live_callables.clear()


atexit.register(_cleanup_all_callables)


@dataclass
class CallableConfig:
    """Configuration for NKIPyCallable.

    Attributes:
        cache_dir: Directory for caching compiled NEFFs
        buckets: List of bucket sizes to use
        dynamic_specs: Dictionary mapping arg_idx to DynamicSpec
        jit_enabled: Whether to JIT compile new buckets on demand
        pipelined: Whether to use pipelined execution
        unpad_outputs: Whether to unpad outputs to original length
        cc_enabled: Whether collective communication is enabled
        rank_id: Process rank for collectives
        world_size: Total number of processes
        ntff_meta: Optional profiling metadata
    """

    cache_dir: Path
    buckets: List[int]
    dynamic_specs: Dict[int, DynamicSpec]
    symint_indices: List[int] = field(default_factory=list)
    jit_enabled: bool = True
    pipelined: bool = True
    unpad_outputs: bool = True
    pad_on_device: bool = True
    keep_outputs_on_device: bool = False
    input_layout: str = "auto"
    output_layout: str = "unpad"
    # Distributed
    cc_enabled: bool = False
    rank_id: int = 0
    world_size: int = 1
    # Profiling
    ntff_meta: Optional[NtffMeta] = None


class NKIPyCallable:
    """Callable wrapper handling bucket selection, JIT compilation, and padding.

    This class orchestrates dynamic shape execution:
    1. Analyzes input shapes to determine actual sequence length
    2. Selects appropriate bucket (or JIT compiles if needed)
    3. Registers bundle with spiky runtime
    4. Executes via spiky with device-side padding
    5. Returns outputs (optionally unpadded)

    Usage (from torch-to-nkipy):
        callable = NKIPyCallable(
            config=CallableConfig(...),
            compiler_fn=lambda bucket_size: compile_for_bucket(gm, bucket_size),
        )
        outputs = callable(*inputs)
    """

    def __init__(
        self,
        config: CallableConfig,
        compiler_fn: Callable[[int], Tuple[str, Dict[int, int], List[int]]],
    ):
        """Initialize NKIPyCallable.

        Args:
            config: CallableConfig with bucket and execution settings
            compiler_fn: Callback to compile for a specific bucket size.
                         Signature: (bucket_size: int) ->
                         (neff_path, alias_map, none_idx_list)
        """
        self._config = config
        self._compiler_fn = compiler_fn
        self._bundle_id: Optional[int] = None
        self._compiled_buckets: Dict[int, str] = {}  # bucket_size -> neff_path
        self._jit_lock = threading.Lock()
        self._buckets = list(config.buckets)  # Mutable copy
        self._alias_map: Optional[Dict[int, int]] = None
        self._none_idx_list: Optional[List[int]] = None

        # Register for atexit cleanup
        _live_callables[id(self)] = weakref.ref(
            self, lambda ref, k=id(self): _live_callables.pop(k, None)
        )

    def _build_adjusted_dynamic_specs(self) -> Dict[int, int]:
        """Build dynamic_specs dict adjusted for SymInt arg removal.

        Returns arg_idx -> dim_idx mapping where arg_idx accounts for
        SymInt arguments that are filtered out before execute_bundle.
        """
        symint_set = set(self._config.symint_indices)
        dynamic_specs_dict = {}
        for spec in self._config.dynamic_specs.values():
            offset = sum(1 for si in symint_set if si < spec.arg_idx)
            dynamic_specs_dict[spec.arg_idx - offset] = spec.dim_idx
        return dynamic_specs_dict

    def _ensure_bundle_registered(self) -> None:
        """Register bundle with spiky if not already done."""
        if self._bundle_id is not None:
            return

        # Register with current compiled buckets (may be empty for JIT)
        self._bundle_id = spiky.register_bundle(
            bucket_to_neff=self._compiled_buckets.copy(),
            dynamic_specs=self._build_adjusted_dynamic_specs(),
            cc_enabled=self._config.cc_enabled,
            rank_id=self._config.rank_id,
            world_size=self._config.world_size,
        )

    def _ensure_bucket_compiled(self, bucket_size: int) -> None:
        """JIT compile a bucket if not already compiled."""
        if bucket_size in self._compiled_buckets:
            return

        with self._jit_lock:
            # Double-check after acquiring lock
            if bucket_size in self._compiled_buckets:
                return

            logger.info(f"JIT compiling bucket size {bucket_size}")
            neff_path, alias_map, none_idx_list = self._compiler_fn(bucket_size)
            self._compiled_buckets[bucket_size] = neff_path
            if self._alias_map is None:
                self._alias_map = alias_map
                self._none_idx_list = none_idx_list

            # Re-register bundle with updated buckets
            self._reregister_bundle()

    def _reregister_bundle(self) -> None:
        """Re-register bundle with current compiled buckets."""
        # Flush and unregister existing bundle if any
        if self._bundle_id is not None:
            try:
                spiky.flush_pipeline(self._bundle_id)
            except Exception:
                pass
            try:
                spiky.unregister_bundle(self._bundle_id)
            except Exception:
                pass  # Bundle may already be unregistered
            self._bundle_id = None

        self._bundle_id = spiky.register_bundle(
            bucket_to_neff=self._compiled_buckets.copy(),
            dynamic_specs=self._build_adjusted_dynamic_specs(),
            cc_enabled=self._config.cc_enabled,
            rank_id=self._config.rank_id,
            world_size=self._config.world_size,
        )

    def _determine_padding_strategy(self, args) -> Tuple[int, int, bool]:
        """Determine actual_len, bucket_size, and whether to skip padding.

        Implements input_layout logic:
        - "padded": requires pre-padded input with PaddingMetadata
        - "auto": detects pre-padded inputs, falls through to normal flow otherwise

        Args:
            args: Input tensors

        Returns:
            (actual_len, bucket_size, skip_padding)
        """
        from spiky.utils.tensor_metadata import get_metadata

        primary_spec = list(self._config.dynamic_specs.values())[0]
        dyn_arg = args[primary_spec.arg_idx]
        meta = get_metadata(dyn_arg)

        if self._config.input_layout == "padded":
            # Require pre-padded input with metadata
            if meta is None:
                raise ValueError(
                    "input_layout='padded' requires input with PaddingMetadata"
                )
            if meta.pad_dim != primary_spec.dim_idx:
                raise ValueError(
                    f"Input padded on dim {meta.pad_dim}, "
                    f"expected {primary_spec.dim_idx}"
                )
            if meta.padded_size not in self._buckets:
                raise ValueError(
                    f"Input bucket {meta.padded_size} not in {self._buckets}"
                )
            return meta.original_size, meta.padded_size, True

        # input_layout == "auto"
        if meta is not None and meta.pad_dim == primary_spec.dim_idx:
            if meta.padded_size in self._buckets:
                return meta.original_size, meta.padded_size, True

        # Fall through to normal flow
        actual_len = dyn_arg.shape[primary_spec.dim_idx]
        bucket_size = _select_bucket(actual_len, self._buckets)
        return actual_len, bucket_size, False

    def __call__(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Execute with automatic bucket selection and padding.

        Args:
            *args: Input tensors

        Returns:
            Tuple of output tensors
        """
        self._ensure_bundle_registered()

        # Handle static path (no dynamic specs) vs dynamic path
        if not self._config.dynamic_specs:
            bucket_size = self._buckets[0]
            actual_len = 0
            skip_padding = True
        else:
            # Determine padding strategy based on input_layout config
            actual_len, bucket_size, skip_padding = self._determine_padding_strategy(
                args
            )

        # Check if we need to JIT compile
        if bucket_size not in self._compiled_buckets:
            if self._config.jit_enabled:
                # Round to power of 2 for new bucket
                new_bucket = 1 << (actual_len - 1).bit_length()
                if new_bucket not in self._buckets:
                    self._buckets = sorted(self._buckets + [new_bucket])
                bucket_size = new_bucket
            # else: will use the selected bucket even if not compiled (may fail)

        self._ensure_bucket_compiled(bucket_size)

        # Convert inputs to numpy for spiky.
        # Skip SymInt args (they were inlined as constants during compilation).
        symint_set = set(self._config.symint_indices)
        inputs_np = []
        tensor_device = None
        for i, t in enumerate(args):
            if i in symint_set:
                continue
            if not isinstance(t, torch.Tensor):
                inputs_np.append(np.array(t))
                continue
            if tensor_device is None:
                tensor_device = t.device
            if t.device.type != "cpu":
                inputs_np.append(t.detach().cpu().contiguous().numpy())
            else:
                inputs_np.append(t.detach().contiguous().numpy())

        # When inputs are already padded, skip device-side padding
        pad_on_device = self._config.pad_on_device and not skip_padding

        # Wrap execution in optional profiling context
        ntff_meta = self._config.ntff_meta
        if (
            ntff_meta is not None
            and ntff_meta.save_ntff
            and bucket_size in self._compiled_buckets
        ):
            from spiky.device.profiling import nkipy_profile

            profile_ctx = nkipy_profile(ntff_meta, self._compiled_buckets[bucket_size])
        else:
            from contextlib import nullcontext

            profile_ctx = nullcontext((False, None))

        with profile_ctx as (save_trace, ntff_file):
            # Execute via spiky (handles padding internally on device)
            if self._config.pipelined:
                outputs = spiky.execute_pipelined(
                    bundle_id=self._bundle_id,
                    bucket_size=bucket_size,
                    inputs=inputs_np,
                    # No self-prefetch: content-blind hit check causes
                    # stale data.
                    next_inputs=[],
                    pad_on_device=pad_on_device,
                    keep_outputs_on_device=self._config.keep_outputs_on_device,
                    unpad_outputs=self._config.unpad_outputs,
                    actual_len=actual_len if self._config.unpad_outputs else 0,
                    save_trace=save_trace,
                    ntff_name=ntff_file or "",
                )
            else:
                outputs = spiky.execute_bundle(
                    bundle_id=self._bundle_id,
                    bucket_size=bucket_size,
                    inputs=inputs_np,
                    pad_on_device=pad_on_device,
                    keep_outputs_on_device=self._config.keep_outputs_on_device,
                    unpad_outputs=self._config.unpad_outputs,
                    actual_len=actual_len if self._config.unpad_outputs else 0,
                    save_trace=save_trace,
                    ntff_name=ntff_file or "",
                )

        # Convert back to torch tensors
        device = tensor_device or torch.device("cpu")
        if self._config.keep_outputs_on_device:
            global _KEEP_ON_DEVICE_WARNED
            if not _KEEP_ON_DEVICE_WARNED:
                logger.warning(
                    "keep_outputs_on_device currently performs a host roundtrip "
                    "for tensor conversion. True zero-copy device tensors require "
                    "spike-torch integration (not yet implemented)."
                )
                _KEEP_ON_DEVICE_WARNED = True
            result = list(_device_tensor_to_torch(out, device) for out in outputs)
        else:
            result = list(torch.from_numpy(out).to(device) for out in outputs)

        # Attach metadata when output_layout="padded"
        if self._config.output_layout == "padded" and self._config.dynamic_specs:
            from spiky.utils.tensor_metadata import PaddingMetadata, attach_metadata

            primary_spec = list(self._config.dynamic_specs.values())[0]
            for out in result:
                attach_metadata(
                    out,
                    PaddingMetadata(
                        original_size=actual_len,
                        padded_size=bucket_size,
                        pad_dim=primary_spec.dim_idx,
                        arg_indices=tuple(self._config.dynamic_specs.keys()),
                    ),
                )

        # Handle alias_map: copy aliased outputs back to input tensors
        if self._alias_map:
            for output_idx, input_idx in self._alias_map.items():
                if output_idx < len(result):
                    args[input_idx].copy_(result[output_idx].to(args[input_idx].device))
            result = [r for i, r in enumerate(result) if i not in self._alias_map]

        # Handle none_idx_list: insert None at specified positions
        if self._none_idx_list:
            for idx in self._none_idx_list:
                result.insert(idx, None)

        return tuple(result)

    def flush(self) -> None:
        """Flush any pending pipelined execution."""
        if self._bundle_id is not None:
            spiky.flush_pipeline(self._bundle_id)

    def close(self) -> None:
        """Unregister bundle and clean up resources."""
        _live_callables.pop(id(self), None)
        if self._bundle_id is not None:
            try:
                self.flush()
            except Exception:
                pass
            try:
                spiky.unregister_bundle(self._bundle_id)
            except Exception:
                pass
            self._bundle_id = None
        self._compiled_buckets.clear()

    @property
    def buckets(self) -> List[int]:
        """Get current list of bucket sizes."""
        return self._buckets.copy()

    @property
    def compiled_buckets(self) -> Dict[int, str]:
        """Get dictionary of compiled buckets (bucket_size -> neff_path)."""
        return self._compiled_buckets.copy()

    def __del__(self):
        """Auto-flush on garbage collection."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during interpreter shutdown
