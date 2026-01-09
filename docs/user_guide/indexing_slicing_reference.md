# Tensor Indexing and Slicing

NKIPy tensors support a subset of NumPy's indexing and slicing operations, optimized for machine learning workloads.

## Supported Operations

| Operation | Example | Description |
|-----------|---------|-------------|
| Basic slicing | `tensor[0:5, 2:8]` | Slice with start:stop |
| Step slicing | `tensor[::2, 1::3]` | Every nth element |
| Negative slicing | `tensor[:-1, 1:-1]` | Negative bounds |
| Negative indexing | `tensor[-1, :]` | Single negative indices |
| Array indexing | `tensor[np.array([0,2,4])]` | Index with array (single axis) |
| List indexing | `tensor[[0, 2, 4]]` | Index with Python list |
| Mixed indexing | `tensor[0:5, indices]` | Static slice + dynamic index |
| Scalar indexing | `tensor[scalar_tensor]` | Use 0D tensor as index |
| Assignment | `tensor[0:3, :] = value` | Slice assignment |
| Chained slicing | `tensor[0:10, :][2:5, 3:8]` | Multiple slice operations |

## Unsupported Operations

| Operation | Example | Workaround |
|-----------|---------|------------|
| Multiple dynamic indices | `tensor[idx1, idx2, :]` | Use separate operations |
| Boolean indexing | `tensor[tensor > 0]` | Use `np.where()` |
| Ellipsis | `tensor[..., 0:3]` | Use explicit slicing |
| newaxis/None | `tensor[:, None, :]` | Use `np.expand_dims()` |

## Limitations

**Single dynamic index**: Only one tensor/array index per operation is supported. For multiple dynamic indices, use separate operations:

```python
# ❌ Not supported
tensor[batch_idx, seq_idx, :]

# ✅ Use separate operations
batch_selected = tensor[batch_idx, :, :]
result = batch_selected[:, seq_idx, :]
```

## Examples

### Batch Selection
```python
def select_batches(tensor, batch_indices):
    return tensor[batch_indices, :, :]

# Usage
batch_tensor = np.random.randn(8, 32, 64)  # [batch, seq, hidden]
indices = np.array([0, 2, 4, 6])
result = select_batches(batch_tensor, indices)  # Shape: (4, 32, 64)
```

### Token Selection
```python
def select_tokens(sequences, token_positions):
    return sequences[:, token_positions, :]

# Usage
sequences = np.random.randn(4, 128, 768)  # [batch, seq_len, hidden]
positions = np.array([0, 10, 20, 30])
result = select_tokens(sequences, positions)  # Shape: (4, 4, 768)
```

### Embedding Lookup
```python
def embedding_lookup(embeddings, token_ids):
    flat_tokens = np.reshape(token_ids, (-1,))
    selected = embeddings[flat_tokens]
    batch_size, seq_len = token_ids.shape
    return np.reshape(selected, (batch_size, seq_len, embeddings.shape[1]))

# Usage
embeddings = np.random.randn(10000, 512)  # Vocabulary embeddings
token_ids = np.random.randint(0, 10000, (8, 64))
result = embedding_lookup(embeddings, token_ids)  # Shape: (8, 64, 512)
```

### Scalar Tensor Indexing
```python
def select_expert(top_k_indices, expert_weights):
    expert_idx = top_k_indices[0]  # 0D tensor (scalar)
    return expert_weights[expert_idx]

# Usage - Mixture of Experts routing
top_k_indices = np.array([1, 0, 2], dtype=np.int32)
expert_weights = np.array([[0.8, 0.6], [0.4, 0.7], [0.5, 0.9]])
result = select_expert(top_k_indices, expert_weights)  # Shape: (2,)
