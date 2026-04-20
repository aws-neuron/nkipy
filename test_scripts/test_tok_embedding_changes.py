#!/usr/bin/env python3
"""
Verify tok_embedding optimization changes are correct.
This test checks:
1. tok_embedding_device is allocated in _allocate_empty_tensors()
2. tok_embedding_device is included in weight_buffers()
3. tok_embedding_device is converted from CPU tensor in _prepare_tensors()
"""
import sys
import numpy as np
import torch

# Mock imports before loading modules
class MockDeviceTensor:
    """Mock DeviceTensor for testing"""
    def __init__(self, shape, dtype, name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.tensor_ref = MockTensorRef()

    class MockTensorRef:
        def __init__(self):
            self.va = 0x1000  # Mock virtual address

    @classmethod
    def from_torch(cls, tensor, name):
        return cls(tuple(tensor.shape), str(tensor.dtype).replace('torch.', ''), name)

    @classmethod
    def allocate_uninitialized(cls, shape, dtype, name=None):
        return cls(shape, dtype, name)

    @classmethod
    def from_numpy(cls, array, name):
        return cls(array.shape, str(array.dtype), name)

    def to_torch(self):
        """Mock conversion back to PyTorch"""
        return torch.zeros(self.shape, dtype=getattr(torch, self.dtype.replace('float16', 'bfloat16')))

# Patch nkipy.runtime before imports
sys.modules['nkipy.runtime'] = type(sys)('nkipy.runtime')
sys.modules['nkipy.runtime'].DeviceTensor = MockDeviceTensor
sys.modules['nkipy.runtime'].DeviceKernel = object

# Now we can import the model
from nkipy.vllm_plugin.models.config import Config
from nkipy.vllm_plugin.models.qwen3 import Qwen3Model

def test_allocate_empty_tensors():
    """Test that tok_embedding_device is allocated in _allocate_empty_tensors"""
    print("Test 1: _allocate_empty_tensors includes tok_embedding_device")

    cfg = Config(
        num_layers=2,
        hidden_size=512,
        vocab_size=1000,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        intermediate_size=1024,
        num_experts=8,
        max_batch_size=1,
        max_seq_len=128,
        dtype='bfloat16'
    )

    model = Qwen3Model(config=cfg, skip_kernels=True)

    # Mock distributed
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='tcp://localhost:29500', rank=0, world_size=1)

    model._allocate_empty_tensors()

    # Check tok_embedding_device was allocated
    assert hasattr(model, 'tok_embedding_device'), "tok_embedding_device not found"
    assert model.tok_embedding_device is not None, "tok_embedding_device is None"
    assert model.tok_embedding_device.shape == (cfg.vocab_size, cfg.hidden_size), \
        f"Wrong shape: {model.tok_embedding_device.shape}"

    print("  ✓ tok_embedding_device allocated correctly")
    print(f"    Shape: {model.tok_embedding_device.shape}")
    print(f"    Dtype: {model.tok_embedding_device.dtype}")

def test_weight_buffers():
    """Test that tok_embedding_device is included in weight_buffers()"""
    print("\nTest 2: weight_buffers includes tok_embedding")

    cfg = Config(
        num_layers=2,
        hidden_size=512,
        vocab_size=1000,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        intermediate_size=1024,
        num_experts=8,
        max_batch_size=1,
        max_seq_len=128,
        dtype='bfloat16'
    )

    model = Qwen3Model(config=cfg, skip_kernels=True)

    # Mock distributed
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='tcp://localhost:29500', rank=0, world_size=1)

    model._allocate_empty_tensors()

    # Collect weight buffers
    buffers = list(model.weight_buffers())
    buffer_names = [name for name, va, size in buffers]

    # Check tok_embedding is in the list
    assert "tok_embedding" in buffer_names, "tok_embedding not found in weight_buffers"

    # Find tok_embedding buffer
    tok_buf = [buf for buf in buffers if buf[0] == "tok_embedding"][0]
    name, va, size = tok_buf

    expected_size = cfg.vocab_size * cfg.hidden_size * 2  # bfloat16 = 2 bytes
    assert size == expected_size, f"Wrong size: {size} vs {expected_size}"

    print("  ✓ tok_embedding found in weight_buffers")
    print(f"    Name: {name}")
    print(f"    Size: {size / 1e6:.1f} MB")
    print(f"    Total buffers: {len(buffers)}")

def test_prepare_tensors():
    """Test that tok_embedding is converted to DeviceTensor in _prepare_tensors"""
    print("\nTest 3: _prepare_tensors converts tok_embedding to DeviceTensor")

    cfg = Config(
        num_layers=2,
        hidden_size=512,
        vocab_size=1000,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        intermediate_size=1024,
        num_experts=8,
        max_batch_size=1,
        max_seq_len=128,
        dtype='bfloat16'
    )

    # Create mock weights
    weights = {
        "tok_embedding": torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16),
        "norm_weight": torch.randn(cfg.hidden_size, dtype=torch.bfloat16),
        "lm_head_weight": torch.randn(cfg.hidden_size, cfg.vocab_size, dtype=torch.bfloat16),
    }

    # Add layer weights
    for lid in range(cfg.num_layers):
        weights[f"layers.{lid}.qkv_weight"] = torch.randn(cfg.hidden_size, 512, dtype=torch.bfloat16)
        weights[f"layers.{lid}.o_weight"] = torch.randn(512, cfg.hidden_size, dtype=torch.bfloat16)
        weights[f"layers.{lid}.gate_up_weight"] = torch.randn(cfg.num_experts, cfg.hidden_size,
                                                               2 * cfg.intermediate_size, dtype=torch.bfloat16)
        weights[f"layers.{lid}.down_weight"] = torch.randn(cfg.num_experts, cfg.intermediate_size,
                                                            cfg.hidden_size, dtype=torch.bfloat16)
        weights[f"layers.{lid}.input_weight"] = torch.randn(cfg.hidden_size, dtype=torch.bfloat16)
        weights[f"layers.{lid}.post_attention_weight"] = torch.randn(cfg.hidden_size, dtype=torch.bfloat16)
        weights[f"layers.{lid}.router_weight"] = torch.randn(cfg.hidden_size, cfg.num_experts, dtype=torch.bfloat16)
        weights[f"layers.{lid}.q_norm_weight"] = torch.randn(cfg.head_dim, dtype=torch.bfloat16)
        weights[f"layers.{lid}.k_norm_weight"] = torch.randn(cfg.head_dim, dtype=torch.bfloat16)

    model = Qwen3Model(model_weights=weights, config=cfg, skip_kernels=True)

    # Check tok_embedding and tok_embedding_device are both set
    assert model.tok_embedding is not None, "tok_embedding is None"
    assert model.tok_embedding_device is not None, "tok_embedding_device is None after _prepare_tensors"
    assert model.tok_embedding_device.shape == (cfg.vocab_size, cfg.hidden_size), \
        f"Wrong tok_embedding_device shape: {model.tok_embedding_device.shape}"

    print("  ✓ tok_embedding_device created from CPU tensor")
    print(f"    CPU tensor shape: {model.tok_embedding.shape}")
    print(f"    Device tensor shape: {model.tok_embedding_device.shape}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing tok_embedding P2P optimization changes")
    print("=" * 60)

    try:
        test_allocate_empty_tensors()
        test_weight_buffers()
        test_prepare_tensors()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nChanges verified:")
        print("  1. tok_embedding_device allocated in _allocate_empty_tensors()")
        print("  2. tok_embedding included in weight_buffers() for P2P")
        print("  3. tok_embedding_device created from CPU tensor")
        print("\nExpected performance improvement:")
        print("  - tok_embedding_s: 2.5s → 0.2s")
        print("  - Total wake_up: 40s → 38s")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
