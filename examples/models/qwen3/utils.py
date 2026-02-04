import sys

import ml_dtypes
import numpy as np
import torch.distributed as dist

bfloat16 = np.dtype(ml_dtypes.bfloat16)


def print_log(msg, rank_list=[0], verbose=0):
    if not dist.is_initialized():
        print(msg)
    elif dist.get_rank() in rank_list:
        print(f"[RANK {dist.get_rank()}] {msg}")
        sys.stdout.flush()


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

                if tensor.dtype == torch.bfloat16:
                    tensor_f32 = tensor.to(dtype=torch.float32)
                else:
                    tensor_f32 = tensor

                if "embed_tokens" in key:
                    weight[key] = tensor_f32.numpy()
                else:
                    weight[key] = np.asarray(tensor_f32.numpy(), dtype=bfloat16)
        return weight

    else:
        return np.load(model_path)


def assert_allclose_with_maxdiff(a, b, rtol=None, atol=None):
    if rtol is None:
        rtol = 1e-2
    if atol is None:
        atol = 1e-4
    if a.dtype == bfloat16:
        a = a.astype(np.float32)
    if b.dtype == bfloat16:
        b = b.astype(np.float32)
    try:
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    except AssertionError as e:
        # Compute relative difference: |a - b| / |b|
        abs_diff = np.abs(a - b)
        rel_diff = abs_diff / np.abs(
            b, where=np.abs(b) != 0, out=np.full_like(b, np.inf)
        )
        rel_diff = np.where(np.abs(b) == 0, abs_diff, rel_diff)  # Handle b=0 case

        # Find index of maximum relative difference
        max_diff_idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
        max_rel_diff = rel_diff[max_diff_idx]
        elements = (a[max_diff_idx], b[max_diff_idx])
        print(
            f"Max diff index: {max_diff_idx}, Max relative diff value: {max_rel_diff}, Elements: {elements}"
        )
        raise e
