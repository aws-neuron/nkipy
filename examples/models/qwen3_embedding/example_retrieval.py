#!/usr/bin/env python3
"""
Reproduce the official Qwen3-Embedding example with Trainium implementation.
This demonstrates query-document similarity scoring with instruction-based queries.
"""

import logging

import numpy as np
from config import Qwen3Config  # For 0.6B model
from config_8b import Qwen3Config_8B  # For 8B model
from embedding_utils import get_detailed_instruct, normalize_embeddings
from model import Qwen3EmbeddingModel
from prepare_weights import load_qwen3_weights
from transformers import AutoTokenizer


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)
    # set the verbosity of root to avoid double printing
    for handler in root_logger.handlers:
        handler.setLevel(logging.ERROR)

    print("=" * 80)
    print("Qwen3-Embedding Retrieval Example on Trainium")
    print("=" * 80)

    # Task description for queries
    task = "Given a web search query, retrieve relevant passages that answer the query"

    # Queries with instructions
    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]

    # Documents (no instruction needed)
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    input_texts = queries + documents

    print("\nInput texts:")
    for i, text in enumerate(input_texts):
        print(f"{i + 1}. {text[:80]}{'...' if len(text) > 80 else ''}")

    # Load tokenizer with default (right) padding
    # Since we process one at a time, right padding works perfectly
    print("\nLoading tokenizer...")
    config = Qwen3Config()  # Qwen3Config_8B()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load Trainium model
    print("Loading Trainium model...")
    weights = load_qwen3_weights(config.weights_path)
    model = Qwen3EmbeddingModel(weights, config)

    # Tokenize input texts
    print(f"\nTokenizing with max_length={config.max_model_len}...")
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=config.max_model_len,
        return_tensors="np",
    )

    input_ids = batch_dict["input_ids"].astype(np.uint32)
    attention_mask = batch_dict["attention_mask"].astype(np.float32)

    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")

    # Get embeddings from Trainium model
    print("\nRunning inference on Trainium...")

    # Process in batches if needed (current config supports batch_size=1)
    all_embeddings = []
    for i in range(len(input_texts)):
        single_input_ids = input_ids[i : i + 1]
        single_attention_mask = attention_mask[i : i + 1]

        print(
            f"  Processing input {i + 1}: shape={single_input_ids.shape}, mask_sum={single_attention_mask.sum()}"
        )

        # Get embeddings - keep left padding intact
        embeddings = model.forward(single_input_ids, single_attention_mask)
        all_embeddings.append(embeddings[0])

    # Stack all embeddings
    embeddings = np.stack(all_embeddings, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")

    # Normalize embeddings
    print("\nNormalizing embeddings...")
    embeddings = normalize_embeddings(embeddings, p=2, axis=1)

    # Compute similarity scores
    print("\nComputing similarity scores...")
    query_embeddings = embeddings[:2]  # First 2 are queries
    doc_embeddings = embeddings[2:]  # Last 2 are documents

    # Compute cosine similarity: queries @ documents.T
    scores = query_embeddings @ doc_embeddings.T

    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print("\nSimilarity Scores (queries × documents):")
    print(f"Query 1 vs Doc 1: {scores[0, 0]:.4f}")
    print(f"Query 1 vs Doc 2: {scores[0, 1]:.4f}")
    print(f"Query 2 vs Doc 1: {scores[1, 0]:.4f}")
    print(f"Query 2 vs Doc 2: {scores[1, 1]:.4f}")

    print("\nScore matrix:")
    print(scores.tolist())

    print("\nExpected scores (from HuggingFace):")
    print("[[0.7646, 0.1414], [0.1355, 0.5999]]")

    print("\n" + "=" * 80)
    print("Interpretation:")
    print("=" * 80)
    print("Query 1 ('What is the capital of China?') matches best with:")
    best_doc_idx = np.argmax(scores[0])
    print(f"  → Document {best_doc_idx + 1}: {documents[best_doc_idx]}")
    print(f"  → Score: {scores[0, best_doc_idx]:.4f}")

    print("\nQuery 2 ('Explain gravity') matches best with:")
    best_doc_idx = np.argmax(scores[1])
    print(f"  → Document {best_doc_idx + 1}: {documents[best_doc_idx]}")
    print(f"  → Score: {scores[1, best_doc_idx]:.4f}")


if __name__ == "__main__":
    main()
