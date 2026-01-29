"""Shared generation utilities for inference scripts."""

from typing import Any, Callable, Dict


def prepare_generation_kwargs(
    max_length: int, do_sample: bool = False
) -> Dict[str, Any]:
    """Prepare generation keyword arguments.

    Args:
        max_length: Maximum length for generation
        do_sample: Whether to use sampling (default: False for greedy)

    Returns:
        Dictionary of generation kwargs
    """
    return {
        "do_sample": do_sample,
        "max_length": max_length,
    }


def run_warmup_generation(
    model: Any,
    input_ids: Dict[str, Any],
    generation_kwargs: Dict[str, Any],
    print_fn: Callable[..., None] = print,
) -> Any:
    """Run warmup generation to compile the model.

    Args:
        model: The model to warm up
        input_ids: Tokenized input
        generation_kwargs: Generation configuration
        print_fn: Print function to use (default: print)

    Returns:
        Generated outputs from warmup
    """
    print_fn("Warming up the model...", flush=True)
    outputs = model.generate(**input_ids, **generation_kwargs)
    print_fn("Model warmed up!", flush=True)
    return outputs


def run_generation(
    model: Any,
    input_ids: Dict[str, Any],
    tokenizer: Any,
    generation_kwargs: Dict[str, Any],
    print_fn: Callable[..., None] = print,
) -> str:
    """Run generation and decode the output.

    Args:
        model: The model to run generation with
        input_ids: Tokenized input
        tokenizer: Tokenizer for decoding
        generation_kwargs: Generation configuration
        print_fn: Print function to use (default: print)

    Returns:
        Decoded output string
    """
    print_fn("Running generation...", flush=True)
    outputs = model.generate(**input_ids, **generation_kwargs)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print_fn(f"Output (device):\n{result[0]}\n", flush=True)
    return result[0]
