import numpy as np
import pytest
import torch
import torch.nn as nn
from config import Config
from nkipy.runtime import baremetal_jit
from kernels.token_embedding import token_embedding
from utils import assert_allclose


def setup():
    """Setup test data for token embedding tests"""
    config = Config(max_batch_size_per_dp=2, max_model_len=8)

    # Create test parameters
    vocab_size = 100
    hidden_size = config.hidden_size
    batch_size = config.max_batch_size_per_dp
    seq_len = config.max_model_len

    # Generate random embedding weights
    tok_embedding = np.random.randn(vocab_size, hidden_size).astype(np.float32)

    # Generate random token IDs (ensure they're within vocab range)
    token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)

    # Create reference using PyTorch embedding
    embedding_layer = nn.Embedding(vocab_size, hidden_size)
    embedding_layer.weight.data = torch.from_numpy(tok_embedding)

    with torch.no_grad():
        ref_output = embedding_layer(torch.from_numpy(token_ids))

    return tok_embedding, token_ids, ref_output.numpy()

def test_cpu():
    """Test CPU implementation of token embedding"""
    tok_embedding, token_ids, ref_output = setup()

    # Test our implementation
    output = token_embedding(tok_embedding, token_ids)

    # Verify output shape
    expected_shape = (*token_ids.shape, tok_embedding.shape[1])
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Verify correctness against reference
    assert_allclose(ref_output, output)

def test_device():
    """Test device implementation of token embedding"""
    tok_embedding, token_ids, ref_output = setup()

    output = baremetal_jit(token_embedding)(tok_embedding=tok_embedding, token_ids=token_ids)

    assert_allclose(ref_output, output)

def test_edge_cases():
    """Test edge cases for token embedding"""
    # Test with single token
    vocab_size = 50
    hidden_size = 64
    tok_embedding = np.random.randn(vocab_size, hidden_size).astype(np.float32)

    # Single token case
    token_ids = np.array([[5]], dtype=np.int32)  # Shape: [1, 1]
    output = token_embedding(tok_embedding, token_ids)
    expected = tok_embedding[5:6, :].reshape(1, 1, hidden_size)
    assert_allclose(expected, output)

    # Test with different batch and sequence sizes
    token_ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # Shape: [2, 3]
    output = token_embedding(tok_embedding, token_ids)

    # Verify shape
    assert output.shape == (2, 3, hidden_size)

    # Verify specific values
    assert_allclose(tok_embedding[1], output[0, 0])
    assert_allclose(tok_embedding[2], output[0, 1])
    assert_allclose(tok_embedding[3], output[0, 2])
    assert_allclose(tok_embedding[4], output[1, 0])
    assert_allclose(tok_embedding[5], output[1, 1])
    assert_allclose(tok_embedding[6], output[1, 2])

if __name__ == "__main__":
    pytest.main(["-s", __file__])
