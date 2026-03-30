import numpy as np

from config import Config

from .rmsnorm import rmsnorm_kernel


def compute_logits(h, norm_weight, lm_head_weight, configs: Config):
    """Compute logits on device (argmax done on CPU to avoid topk bug on large vocab).

    Returns partial logits for this rank's vocab shard.
    """
    h = rmsnorm_kernel(h, norm_weight, configs.norm_eps)
    logits = h[:, -1, :] @ lm_head_weight
    return logits.astype(np.float32)
