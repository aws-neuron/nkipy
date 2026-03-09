from common.kernels.rope import apply_rotary_emb_kernel as rope_qwen3  # noqa: F401
from common.kernels.rope import compute_cos_sin_cache


def compute_qwen3_cos_sin(max_model_len, head_dim, theta=1000000.0):
    return compute_cos_sin_cache(head_dim, max_model_len, base=theta)
