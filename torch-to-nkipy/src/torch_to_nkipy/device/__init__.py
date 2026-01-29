# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch

# Import spike_torch for device registration (automatic on import)
import spike_torch
from spike import SpikeModel, get_spike_singleton
from spike._spike import NrtTensor
from torch_to_nkipy.backend.nkipy_backend_config import get_nkipy_backend_config
from torch_to_nkipy.device import device_module, distributed_backend  # noqa: F401
from torch_to_nkipy.utils.ntff_meta import NtffMeta

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# spike_torch._ensure_initialized() is called on import, which:
#   - Registers "nkipy" device name
#   - Registers hooks interface
#   - Registers device module
#   - Generates backend methods

_nkipy_initialized = False


def nkipy_init(visible_core: int):
    """Initialize Neuron Runtime for the given core.

    Args:
        visible_core: The core index to use (rank + core_offset from backend init)

    Sets up visible cores and root comm ID for collectives.
    """
    global _nkipy_initialized

    # Set root comm ID for collectives
    if os.environ.get("NEURON_RT_ROOT_COMM_ID", None) is None:
        root_addr = os.environ.get("MASTER_ADDR", "localhost")
        root_port = os.environ.get("NEURON_RT_PORT", "61234")
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{root_addr}:{root_port}"

    # Reset and reconfigure spike with visible cores for this core.
    # spike_torch auto-initializes on import with default cores, so we need
    # to reset and reconfigure for the specific core.
    # Suppress the expected warning about invalidated objects since this is
    # initialization and no user objects exist yet.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="spike.reset\\(\\) called")
        spike_torch.reset()
    spike_torch.configure(visible_cores=[visible_core])

    # Re-initialize the spike singleton with the new configuration.
    # This is needed because spike_torch.reset() closed the previous runtime.
    get_spike_singleton()
    _nkipy_initialized = True


def is_nkipy_device_initialized() -> bool:
    """Check if nkipy device is properly initialized."""
    return _nkipy_initialized


def nkipy_close():
    """Close Neuron Runtime and release resources."""
    global _nkipy_initialized
    spike_torch.reset()
    _nkipy_initialized = False


def nkipy_load_model(neff_file, cc_enabled, device_id, device_count) -> SpikeModel:
    """Load NEFF and return a SpikeModel instance."""
    return SpikeModel.load_from_neff(
        neff_path=neff_file,
        core_id=spike_torch.current_device(),
        cc_enabled=cc_enabled,
        rank_id=device_id,
        world_size=device_count,
    )


def _torch_to_nkipy_tensor(tensor: torch.Tensor, name: str) -> NrtTensor:
    """Convert PyTorch tensor (on nkipy device) to non-owning NrtTensor."""
    nrt_ptr, size, core_id = spike_torch.get_tensor_info(tensor.data_ptr())
    return NrtTensor.wrap(nrt_ptr, core_id, size, name)


def nkipy_execute_model(
    model: SpikeModel,
    inputs: dict,
    outputs: dict,
    save_trace: bool = False,
    ntff_name: str = None,
):
    """Execute model with PyTorch tensors on nkipy device.

    Args:
        model: SpikeModel instance from nkipy_load_model
        inputs: Dict mapping input names to PyTorch tensors on nkipy device
        outputs: Dict mapping output names to PyTorch tensors on nkipy device
        save_trace: Whether to save execution trace
        ntff_name: Optional name for the trace file
    """
    # Convert to NrtTensor (non-owning wrappers)
    input_tensors = {
        name: _torch_to_nkipy_tensor(t, name) for name, t in inputs.items()
    }
    output_tensors = {
        name: _torch_to_nkipy_tensor(t, name) for name, t in outputs.items()
    }

    # Execute using spike's existing API
    get_spike_singleton().execute(
        model.model_ref,
        inputs=input_tensors,
        outputs=output_tensors,
        save_trace=save_trace,
        ntff_name=ntff_name,
    )


@contextmanager
def nkipy_profile(ntff_meta: NtffMeta, neff_path: str):
    """Context manager determining profiling settings.

    Yields (save_trace, ntff_name) tuple for use in nkipy_execute_model.
    """
    logger.debug(
        f"Profiling NEFF with hash {ntff_meta.kernel_hash}, "
        f"save_ntff_exe_idx {ntff_meta.save_ntff_exe_idx}, and "
        f"curr_exe_idx {ntff_meta.curr_exe_idx}"
    )

    exe_idx_check = (
        not ntff_meta.save_ntff_exe_idx
        or ntff_meta.curr_exe_idx in ntff_meta.save_ntff_exe_idx
    )
    rank_check = get_nkipy_backend_config().rank == 0
    should_profile = ntff_meta.save_ntff and exe_idx_check and rank_check
    ntff_file = None

    if should_profile:
        save_dir = Path(
            f"{ntff_meta.save_ntff_dir}/kernel_{ntff_meta.kernel_hash}"
        ).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        neff_filename = os.path.basename(neff_path)
        target_neff_path = save_dir / neff_filename
        if not target_neff_path.exists() and os.path.exists(neff_path):
            shutil.copy2(neff_path, target_neff_path)

        ntff_file = str(save_dir / f"{ntff_meta.curr_exe_idx}.ntff")
        if Path(ntff_file).exists():
            Path(ntff_file).unlink()

        logger.debug(f"Saving NTFF profile to {ntff_file}")

    yield should_profile, ntff_file
    ntff_meta.curr_exe_idx += 1
