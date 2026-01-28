#!/usr/bin/env python3
"""
Qwen3-Embedding example for Trainium.

Supports:
- Retrieval example (default): Query-document similarity scoring
- Benchmark mode: Performance measurement with warmup
- Compare mode: Verify Trainium vs HuggingFace embeddings

Usage:
    # Basic retrieval example (0.6B model)
    python example_retrieval.py

    # Use 8B model
    python example_retrieval.py --model-size 8b

    # Run benchmark
    python example_retrieval.py --benchmark --num-warmup 3 --num-iterations 10

    # Compare with HuggingFace
    python example_retrieval.py --compare
"""

import argparse
import logging
import time
from typing import List, Tuple

import numpy as np
import torch
from config import get_config
from embedding_utils import get_detailed_instruct, last_token_pool, normalize_embeddings
from model import Qwen3EmbeddingModel
from prepare_weights import load_qwen3_weights
from transformers import AutoModel, AutoTokenizer


def setup_logging():
    """Suppress verbose logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)
    for handler in root_logger.handlers:
        handler.setLevel(logging.ERROR)


def run_retrieval_example(model, tokenizer, config):
    """Run the standard retrieval example with query-document similarity."""
    print("=" * 70)
    print("Qwen3-Embedding Retrieval Example on Trainium")
    print("=" * 70)

    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the "
        "movement of planets around the sun.",
    ]

    input_texts = queries + documents

    print("\nInput texts:")
    for i, text in enumerate(input_texts):
        print(f"  {i + 1}. {text[:70]}{'...' if len(text) > 70 else ''}")

    # Tokenize
    print(f"\nTokenizing (max_length={config.max_model_len})...")
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=config.max_model_len,
        return_tensors="np",
    )
    input_ids = batch_dict["input_ids"].astype(np.uint32)
    attention_mask = batch_dict["attention_mask"].astype(np.float32)

    # Get embeddings (process one at a time for batch_size=1)
    print("\nRunning inference...")
    all_embeddings = []
    for i in range(len(input_texts)):
        embeddings = model.forward(input_ids[i : i + 1], attention_mask[i : i + 1])
        all_embeddings.append(embeddings[0])

    embeddings = np.stack(all_embeddings, axis=0)
    embeddings = normalize_embeddings(embeddings, p=2, axis=1)

    # Compute similarity
    query_emb = embeddings[:2]
    doc_emb = embeddings[2:]
    scores = query_emb @ doc_emb.T

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print("\nSimilarity Scores:")
    print(f"  Query 1 vs Doc 1: {scores[0, 0]:.4f}")
    print(f"  Query 1 vs Doc 2: {scores[0, 1]:.4f}")
    print(f"  Query 2 vs Doc 1: {scores[1, 0]:.4f}")
    print(f"  Query 2 vs Doc 2: {scores[1, 1]:.4f}")

    print("\nScore matrix:", scores.tolist())

    print("\nInterpretation:")
    for i, query in enumerate(["What is the capital of China?", "Explain gravity"]):
        best_idx = np.argmax(scores[i])
        print(
            f"  Query '{query}' → Doc {best_idx + 1} (score: {scores[i, best_idx]:.4f})"
        )


def run_benchmark(
    model, tokenizer, config, num_warmup: int, num_iterations: int
) -> List[float]:
    """Run performance benchmark with warmup."""
    print("=" * 70)
    print("Performance Benchmark")
    print("=" * 70)

    task = "Given a web search query, retrieve relevant passages that answer the query"
    sample_text = get_detailed_instruct(task, "What is the capital of China?")

    batch_dict = tokenizer(
        [sample_text],
        padding="max_length",
        truncation=True,
        max_length=config.max_model_len,
        return_tensors="np",
    )
    input_ids = batch_dict["input_ids"].astype(np.uint32)
    attention_mask = batch_dict["attention_mask"].astype(np.float32)

    print(f"\nInput shape: {input_ids.shape}")

    # Warmup
    print(f"\nWarmup ({num_warmup} iterations):")
    for i in range(num_warmup):
        start = time.perf_counter()
        _ = model.forward(input_ids, attention_mask)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  {i + 1}/{num_warmup}: {elapsed:.2f} ms")

    # Benchmark
    print(f"\nBenchmark ({num_iterations} iterations):")
    latencies = []
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = model.forward(input_ids, attention_mask)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        print(f"  {i + 1}/{num_iterations}: {elapsed:.2f} ms")

    # Statistics
    latencies_np = np.array(latencies)
    print(f"\nStatistics:")
    print(f"  Mean:   {latencies_np.mean():.2f} ms")
    print(f"  Std:    {latencies_np.std():.2f} ms")
    print(f"  Min:    {latencies_np.min():.2f} ms")
    print(f"  Max:    {latencies_np.max():.2f} ms")
    print(f"  Throughput: {1000 / latencies_np.mean():.2f} inferences/sec")

    return latencies


def get_hf_embeddings(texts: List[str], model_name: str, max_length: int) -> np.ndarray:
    """
    Get embeddings from HuggingFace model.

    Args:
        texts: List of input texts
        model_name: HuggingFace model name
        max_length: Maximum sequence length

    Returns:
        embeddings: [num_texts, hidden_size]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    ).eval()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state.numpy()
        attention_mask_np = inputs["attention_mask"].numpy()
        embeddings = last_token_pool(last_hidden_state, attention_mask_np)

    return embeddings.astype(np.float32)


def run_compare(model, tokenizer, config) -> Tuple[float, bool]:
    """Compare Trainium vs HuggingFace embeddings on retrieval task."""
    print("=" * 70)
    print("Trainium vs HuggingFace Comparison")
    print("=" * 70)

    # Use the same retrieval task for comparison
    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the "
        "movement of planets around the sun.",
    ]
    input_texts = queries + documents

    print("\nInput texts:")
    for i, text in enumerate(input_texts):
        print(f"  {i + 1}. {text[:70]}{'...' if len(text) > 70 else ''}")

    # Get Trainium embeddings
    print("\nGetting Trainium embeddings...")
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=config.max_model_len,
        return_tensors="np",
    )
    input_ids = batch_dict["input_ids"].astype(np.uint32)
    attention_mask = batch_dict["attention_mask"].astype(np.float32)

    trainium_embeddings = []
    for i in range(len(input_texts)):
        emb = model.forward(input_ids[i : i + 1], attention_mask[i : i + 1])
        trainium_embeddings.append(emb[0])
    trainium_embeddings = np.stack(trainium_embeddings, axis=0)
    trainium_embeddings = normalize_embeddings(trainium_embeddings, p=2, axis=1)

    # Get HuggingFace embeddings
    print("Getting HuggingFace embeddings...")
    hf_embeddings = get_hf_embeddings(
        input_texts, config.model_name, config.max_model_len
    )
    hf_embeddings = normalize_embeddings(hf_embeddings, p=2, axis=1)

    # Compute similarity scores
    trainium_scores = trainium_embeddings[:2] @ trainium_embeddings[2:].T
    hf_scores = hf_embeddings[:2] @ hf_embeddings[2:].T

    print("\n" + "=" * 70)
    print("Similarity Scores Comparison")
    print("=" * 70)
    print("\nTrainium scores:")
    print(f"  Query 1 vs Doc 1: {trainium_scores[0, 0]:.4f}")
    print(f"  Query 1 vs Doc 2: {trainium_scores[0, 1]:.4f}")
    print(f"  Query 2 vs Doc 1: {trainium_scores[1, 0]:.4f}")
    print(f"  Query 2 vs Doc 2: {trainium_scores[1, 1]:.4f}")

    print("\nHuggingFace scores:")
    print(f"  Query 1 vs Doc 1: {hf_scores[0, 0]:.4f}")
    print(f"  Query 1 vs Doc 2: {hf_scores[0, 1]:.4f}")
    print(f"  Query 2 vs Doc 1: {hf_scores[1, 0]:.4f}")
    print(f"  Query 2 vs Doc 2: {hf_scores[1, 1]:.4f}")

    # Compare embeddings directly
    print("\n" + "=" * 70)
    print("Embedding Comparison (per input)")
    print("=" * 70)
    cosine_sims = []
    for i in range(len(input_texts)):
        t_emb = trainium_embeddings[i]
        h_emb = hf_embeddings[i]
        cos_sim = np.dot(t_emb, h_emb) / (np.linalg.norm(t_emb) * np.linalg.norm(h_emb))
        cosine_sims.append(cos_sim)
        print(f"  Input {i + 1}: cosine similarity = {cos_sim:.6f}")

    avg_cosine = np.mean(cosine_sims)
    min_cosine = np.min(cosine_sims)

    print(f"\nAverage cosine similarity: {avg_cosine:.6f}")
    print(f"Minimum cosine similarity: {min_cosine:.6f}")

    passed = min_cosine > 0.95
    if min_cosine > 0.99:
        print("\n✅ PASS: All embeddings match well (min cosine > 0.99)")
    elif min_cosine > 0.95:
        print("\n⚠️  WARNING: Similar but not identical (0.95 < min cosine < 0.99)")
    else:
        print("\n❌ FAIL: Some embeddings differ significantly (min cosine < 0.95)")

    return avg_cosine, passed


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Embedding on Trainium",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python example_retrieval.py                    # Basic retrieval (0.6B)
  python example_retrieval.py --model-size 8b   # Use 8B model
  python example_retrieval.py --benchmark       # Run benchmark
  python example_retrieval.py --compare         # Compare with HuggingFace
        """,
    )
    parser.add_argument(
        "--model-size",
        choices=["0.6b", "8b"],
        default="0.6b",
        help="Model size (default: 0.6b)",
    )
    parser.add_argument(
        "--lnc",
        type=int,
        choices=[1, 2],
        default=2,
        help="Logical NeuronCore count for compilation (default: 2)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override max sequence length",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Trainium vs HuggingFace embeddings",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Warmup iterations for benchmark (default: 3)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="Benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--no-fused",
        action="store_true",
        help="Use separate kernels instead of fused transformer layer (for debugging)",
    )
    args = parser.parse_args()

    setup_logging()

    # Build config with overrides
    overrides = {}
    if args.seq_len is not None:
        overrides["max_model_len"] = args.seq_len

    config = get_config(args.model_size, **overrides)

    use_fused = not args.no_fused

    print(f"Model: {config.model_name}")
    print(f"Sequence length: {config.max_model_len}")
    print(f"LNC: {args.lnc}")
    print(f"Fused layer: {use_fused}")

    # Load tokenizer and model
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    print("Loading model and compiling kernels...")
    weights = load_qwen3_weights(config.weights_path)
    model = Qwen3EmbeddingModel(
        weights, config, lnc=args.lnc, use_fused_layer=use_fused
    )

    # Run requested mode
    if args.benchmark:
        run_benchmark(model, tokenizer, config, args.num_warmup, args.num_iterations)
    elif args.compare:
        run_compare(model, tokenizer, config)
    else:
        run_retrieval_example(model, tokenizer, config)


if __name__ == "__main__":
    main()
