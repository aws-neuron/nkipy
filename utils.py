import numpy as np
import torch
import torch.distributed as dist
import ml_dtypes
import os
import mmap
import ctypes

bfloat16 = np.dtype(ml_dtypes.bfloat16)

def create_shared_pinned_tensor(source_tensor, name="shared_tensor", verbose=0):
    """
    Create a shared tensor across ranks, and pin in memory
    Assume all ranks see the same tensor, but only Rank 0 will create the pinned tensor
    """
    rank = dist.get_rank()
    filepath = f"/tmp/{name}.dat"

    if rank == 0:
        # Rank 0: Write tensor data directly
        with open(filepath, 'wb') as f:
            # Get raw bytes from tensor
            data_ptr = source_tensor.data_ptr()
            size = source_tensor.numel() * source_tensor.element_size()

            # Copy raw memory to file
            raw_data = ctypes.string_at(data_ptr, size)
            f.write(raw_data)

        if verbose:
            print(f"Rank 0: Created shared file {filepath}")

    dist.barrier()

    # All ranks: Memory map the file
    with open(filepath, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)

    # Create tensor from raw pointer
    shared_tensor = torch.frombuffer(
        mm,
        dtype=source_tensor.dtype,
        count=source_tensor.numel()
    ).reshape(source_tensor.shape)

    # Pin memory using mlock
    libc = ctypes.CDLL("libc.so.6")
    data_ptr = shared_tensor.data_ptr()
    size = shared_tensor.numel() * shared_tensor.element_size()

    result = libc.mlock(ctypes.c_void_p(data_ptr), ctypes.c_size_t(size))
    if result == 0:
        if verbose:
            print(f"Rank {rank}: Pinned tensor {name} with {size} bytes")
    else:
        raise RuntimeError(f"Rank {rank}: Failed to pin {name} inmemory")

    return shared_tensor


def load_parameters(model_path):
    """
    Convert different kinds of models into numpy format
    """
    if model_path.endswith("pt"):
        import torch

        ckpt = torch.load(model_path)
        weight = {}

        for key, value in ckpt["model"].items():
            new_key = f"model.{key}"
            weight[new_key] = value.cpu().numpy()

        return weight
    elif model_path.endswith("safetensors"):
        import torch
        from safetensors.torch import safe_open

        weight = {}
        with safe_open(model_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # FIXME: only f32 for now
                # If the tensor is in BF16, convert it to float32
                if tensor.dtype == torch.bfloat16:
                    tensor_f32 = tensor.to(dtype=torch.float32)
                else:
                    tensor_f32 = tensor

                if "embed_tokens" in key:
                    weight[key] = (
                        tensor_f32.numpy()
                    )  # keep fp32 for on host sampling performance
                else:
                    weight[key] = np.asarray(tensor_f32.numpy(), dtype=bfloat16)
        return weight

    else:
        return np.load(model_path)


def assert_allclose(a, b, rtol=None, atol=None, use_matrix_rel_err=True):
    if a.dtype == bfloat16 and b.dtype == bfloat16:
        if atol is None:
            atol = 1e-3
        if rtol is None:
            if use_matrix_rel_err:
                # machine epsilon
                rtol = 2**-7
            else:
                rtol = 5e-2
    elif a.dtype == np.float32 and b.dtype == np.float32:
        if atol is None:
            atol = 1e-4
        if rtol is None:
            if use_matrix_rel_err:
                # TODO: ideally, usemachine epsilon 2**-23
                rtol = 1e-5
            else:
                rtol = 1e-2
    else:
        raise ValueError(f"Unsupported dtype: {a.dtype} and {b.dtype}")

    if a.dtype == bfloat16:
        a = a.astype(np.float32)
    if b.dtype == bfloat16:
        b = b.astype(np.float32)

    # if the matrix error check not pass, then go check the elementwise mismatch
    try:
        if use_matrix_rel_err:
            diff = a - b
            assert np.linalg.norm(diff) <= rtol * np.linalg.norm(a), (
                f"{np.linalg.norm(diff)=} > {rtol * np.linalg.norm(a)=}"
            )
        else:
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    except AssertionError as e:
        _raise_with_mismatch_report(e, a, b, atol, rtol)

def _raise_with_mismatch_report(e, a, b, atol, rtol):
    """
    """
    # Capture the original error message
    original_msg = str(e)

    # Find all elements that fail the tolerance check
    abs_diff = np.abs(a - b)
    tolerance = atol + rtol * np.abs(b)
    failed_mask = abs_diff > tolerance

    # Get indices of all failed elements
    failed_indices = np.where(failed_mask)
    failed_positions = list(zip(*failed_indices))

    # Build detailed analysis message
    num_failed = len(failed_positions)
    detailed_msg = f"\nDetailed analysis - Total failed elements: {num_failed}\n"
    detailed_msg += "First 10 failed elements:\n"

    for i, pos in enumerate(failed_positions[:10]):
        abs_diff_val = abs_diff[pos]
        tolerance_val = tolerance[pos]
        rel_diff = (
            abs_diff_val / np.abs(b[pos]) if np.abs(b[pos]) != 0 else abs_diff_val
        )
        detailed_msg += f"  {i + 1}. Index: {pos}, a={a[pos]}, b={b[pos]}, abs_diff={abs_diff_val:.6e}, rel_diff={rel_diff:.6e}, tolerance={tolerance_val:.6e}\n"

    if num_failed > 10:
        detailed_msg += f"  ... and {num_failed - 10} more failed elements"

    # Combine original message with detailed analysis
    combined_msg = original_msg + detailed_msg

    # Raise new AssertionError with combined message
    raise AssertionError(combined_msg)


def save_device_tensor(folder: str, filename:str, tensor):
    """"""
    os.makedirs(folder, exist_ok=True)
    np_tensor = tensor.numpy().astype(float)
    np.save(os.path.join(folder, filename), np_tensor)
