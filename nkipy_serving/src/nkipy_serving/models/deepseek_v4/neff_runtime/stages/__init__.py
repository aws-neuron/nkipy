"""Model-owned NEFF execution stages for DeepSeek-V4.

This package is intentionally separate from ``nkipy_serving.attention.deepseek_v4``.
The attention package owns the reusable sparse-attention backend and NKI
kernels; these stage modules own model-specific NEFF compilation, buckets, and
executor runtime orchestration.
"""
