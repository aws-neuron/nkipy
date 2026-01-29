"""Shared argument parsing utilities for inference scripts."""

import argparse
from typing import List


def create_base_parser(
    description: str,
    supported_models: List[str],
    default_model: str,
) -> argparse.ArgumentParser:
    """Create base argument parser with common arguments.

    Args:
        description: Description for the argument parser
        supported_models: List of supported model IDs
        default_model: Default model ID to use

    Returns:
        ArgumentParser with common arguments configured
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        choices=supported_models,
        help=f"Name of the model to use. Valid options: {', '.join(supported_models)}",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Model dtype (float32, bfloat16, or float16)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is Paris.",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum length for generation",
    )

    return parser


def add_single_device_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments specific to single-device inference.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        The same parser with additional arguments
    """
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./nkipy_cache",
        help="Directory for nkipy cache",
    )
    return parser


def add_distributed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments specific to distributed inference.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        The same parser with additional arguments
    """
    # Placeholder for future distributed-specific arguments
    # (e.g., custom TP degree, checkpoint path, etc.)
    return parser
