#!/usr/bin/env python3
"""
Compare Qwen3 embeddings between Trainium and HuggingFace implementations
"""

import numpy as np
import torch
from config import Qwen3Config
from embedding_utils import last_token_pool
from model import Qwen3EmbeddingModel
from prepare_weights import load_qwen3_weights
from transformers import AutoModel, AutoTokenizer


def get_hf_embeddings(text: str, model_name: str):
    """Get embeddings from HuggingFace model"""
    print("\n" + "=" * 80)
    print("HuggingFace Embeddings")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    ).eval()

    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state.numpy()
        attention_mask_np = inputs["attention_mask"].numpy()
        print("HF", last_hidden_state[:, :10])

        # Last token pooling using shared utility
        embeddings_np = last_token_pool(last_hidden_state, attention_mask_np)[0]

    print(f"Shape: {embeddings_np.shape}")
    print(f"First 10 values: {embeddings_np[:10]}")
    print(f"L2 norm: {np.linalg.norm(embeddings_np):.6f}")

    return embeddings_np


def get_trainium_embeddings(text: str, config: Qwen3Config):
    """Get embeddings from Trainium model"""
    print("\n" + "=" * 80)
    print("Trainium Embeddings")
    print("=" * 80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    # Load model
    weights = load_qwen3_weights(config.weights_path)
    model = Qwen3EmbeddingModel(weights, config)

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=config.max_model_len,
    )
    input_ids = inputs["input_ids"].astype(np.uint32)
    attention_mask = inputs["attention_mask"].astype(np.float32)
    print(attention_mask.shape)
    print(attention_mask)

    # Get embeddings
    embeddings = model.forward(input_ids, attention_mask)
    embeddings_np = embeddings[0].astype(np.float32)

    print("NKIPy", embeddings_np[:10])
    print(f"Shape: {embeddings_np.shape}")
    print(f"First 10 values: {embeddings_np[:10]}")
    print(f"L2 norm: {np.linalg.norm(embeddings_np):.6f}")

    return embeddings_np


def compare_embeddings(hf_emb, trainium_emb):
    """Compare two embedding vectors"""
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)

    # Compute metrics
    diff = np.abs(hf_emb - trainium_emb)
    cosine_sim = np.dot(hf_emb, trainium_emb) / (
        np.linalg.norm(hf_emb) * np.linalg.norm(trainium_emb)
    )

    print(f"Max absolute difference: {diff.max():.6f}")
    print(f"Mean absolute difference: {diff.mean():.6f}")
    print(f"Cosine similarity: {cosine_sim:.6f}")

    if cosine_sim > 0.99:
        print("\n✅ PASS: Embeddings match well (cosine similarity > 0.99)")
    elif cosine_sim > 0.95:
        print(
            "\n⚠️  WARNING: Embeddings are similar but not identical (0.95 < cosine similarity < 0.99)"
        )
    else:
        print("\n❌ FAIL: Embeddings differ significantly (cosine similarity < 0.95)")

    return cosine_sim


def main():
    text = "Hello, how are you?"

    print(f"Input text: '{text}'")

    # Load config
    config = Qwen3Config()

    # Get embeddings from both implementations
    hf_emb = get_hf_embeddings(text, config.model_name)
    trainium_emb = get_trainium_embeddings(text, config)

    # Compare
    compare_embeddings(hf_emb, trainium_emb)


if __name__ == "__main__":
    main()
