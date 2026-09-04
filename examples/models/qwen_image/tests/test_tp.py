"""TP weight-sharding math (no kernel trace, no hardware).

Validates that the Megatron-style shard slicers partition the weights cleanly:
column-parallel q shards must concatenate back to the full weight with no
overlap or gap. End-to-end TP equivalence is validated on device by
``tests/test_tp_device.py`` (torchrun; skips without hardware).

    cd examples/models/qwen_image
    python -m pytest tests/test_tp.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_diff = pytest.importorskip("diffusers", reason="diffusers with Qwen-Image required")
from diffusers import QwenImageTransformer2DModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weight_extract import extract_flat_weights, shard_flat_weights  # noqa: E402


HF = dict(
    num_layers=2, num_attention_heads=8, attention_head_dim=24,
    joint_attention_dim=40, in_channels=16, out_channels=4, patch_size=2,
    axes_dims_rope=(8, 8, 8),
)


def test_column_parallel_shards_reconstruct():
    """Column-parallel q shards must concat back to the full weight (no overlap)."""
    torch.manual_seed(0)
    model = QwenImageTransformer2DModel(**HF).eval()
    flat = extract_flat_weights(model, HF["num_layers"], dtype=np.float32)
    for tp_size in (2, 4):
        shards = [
            shard_flat_weights(flat, r, tp_size, HF["num_layers"],
                               HF["num_attention_heads"], HF["attention_head_dim"])
            for r in range(tp_size)
        ]
        recon = np.concatenate([shards[r]["b0_iq_w"] for r in range(tp_size)], axis=1)
        assert recon.shape == flat["b0_iq_w"].shape
        assert np.allclose(recon, flat["b0_iq_w"])
