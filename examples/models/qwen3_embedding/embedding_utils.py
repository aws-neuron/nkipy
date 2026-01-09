"""Utility functions for Qwen3 Embedding"""

import numpy as np


def last_token_pool(
    last_hidden_states: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    """
    Extract embeddings from the last valid token position for each sequence.

    Args:
        last_hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]

    Returns:
        embeddings: [batch_size, hidden_size]
    """
    batch_size = last_hidden_states.shape[0]
    left_padding = attention_mask[:, -1].sum() == batch_size

    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(axis=1) - 1
        sequence_lengths = sequence_lengths.astype(np.int32)
        embeddings = np.zeros(
            (batch_size, last_hidden_states.shape[2]), dtype=last_hidden_states.dtype
        )
        for i in range(batch_size):
            embeddings[i] = last_hidden_states[i, sequence_lengths[i]]
        return embeddings


def normalize_embeddings(
    embeddings: np.ndarray, p: int = 2, axis: int = 1
) -> np.ndarray:
    """
    Normalize embeddings using Lp norm.

    Args:
        embeddings: [batch_size, hidden_size]
        p: Norm order (default: 2 for L2 norm)
        axis: Axis along which to normalize

    Returns:
        normalized_embeddings: [batch_size, hidden_size]
    """
    norms = np.linalg.norm(embeddings, ord=p, axis=axis, keepdims=True)
    return embeddings / norms


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Format query with task instruction for retrieval.

    Args:
        task_description: Description of the retrieval task
        query: The actual query text

    Returns:
        Formatted instruction + query string
    """
    return f"Instruct: {task_description}\nQuery: {query}"
