import torch
import numpy as np
import ml_dtypes

bfloat16 = np.dtype(ml_dtypes.bfloat16)
float8_e5m2 = np.dtype(ml_dtypes.float8_e5m2)

# Lookup table between torch and numpy dtype
# only support ml standard types

torch_to_numpy_type = {
    torch.float32: np.float32,
    torch.bfloat16: bfloat16,
    torch.float8_e5m2: float8_e5m2,
    torch.int32: np.int32,
    torch.uint32: np.uint32,
}

numpy_to_torch_type = {
    np.dtype("float32"): torch.float32,
    np.dtype("bfloat16"): torch.bfloat16,
    np.dtype("float8_e5m2"): torch.float8_e5m2,
    np.dtype("int32"): torch.int32,
    np.dtype("uint32"): torch.uint32,
    # Also support numpy types directly (not just dtype instances)
    np.float32: torch.float32,
    bfloat16: torch.bfloat16,
    float8_e5m2: torch.float8_e5m2,
    np.int32: torch.int32,
    np.uint32: torch.uint32,
}
