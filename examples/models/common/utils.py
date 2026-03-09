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
