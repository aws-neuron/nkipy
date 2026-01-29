import logging
from typing import Any, Callable, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StaticCache,
)

from torch_to_nkipy import init_nkipy_backend


def move_to_device(arg: Any, device: Optional[Union[str, torch.device]] = None) -> Any:
    if device is None:
        device = "cpu"

    if isinstance(arg, torch.Tensor):
        return arg.to(device)
    elif isinstance(arg, StaticCache):
        # Handle both old API (key_cache/value_cache) and new API (layers)
        if hasattr(arg, "layers"):
            # New transformers API (4.56+): layers is a list of StaticLayer objects
            for layer in arg.layers:
                if hasattr(layer, "keys") and layer.keys is not None:
                    layer.keys = layer.keys.to(device)
                if hasattr(layer, "values") and layer.values is not None:
                    layer.values = layer.values.to(device)
        elif hasattr(arg, "key_cache"):
            # Old transformers API
            arg.key_cache = move_to_device(arg.key_cache, device)
            arg.value_cache = move_to_device(arg.value_cache, device)
        return arg
    elif isinstance(arg, (list, tuple)):
        return type(arg)(move_to_device(item, device) for item in arg)
    elif isinstance(arg, dict):
        return {k: move_to_device(v, device) for k, v in arg.items()}
    else:
        return arg


def forward_wrap(compiled_forward: Callable, *args: Any, **kwargs: Any) -> Any:
    # Set target device based on model type
    target_device = "nkipy"

    # Move inputs to target device
    args = [move_to_device(arg, target_device) for arg in args]
    kwargs = {k: move_to_device(v, target_device) for k, v in kwargs.items()}

    # Run the compiled forward pass
    out = compiled_forward(*args, **kwargs)

    # Move outputs back to CPU
    if hasattr(out, "logits"):
        out.logits = out.logits.cpu()

    return out


def setup_neuron_environment(
    rank: int = None,
    world_size: int = None,
    cache_dir: str = None,
    log_level: int = None,
) -> None:
    """Initialize the Neuron backend.

    Args:
        rank: Process rank. Auto-detected if None
            (from torch.distributed or env vars).
        world_size: Total processes. Auto-detected if None
            (from torch.distributed or env vars).
        cache_dir: Directory for NKIPy cache. Defaults to "./nkipy_cache".
        log_level: Logging level. If None, uses INFO for rank 0 and
            ERROR for others. Pass logging.INFO to see logs from all ranks.
    """
    import os

    import torch.distributed as dist

    cache_path = "./nkipy_cache" if cache_dir is None else cache_dir

    # Determine effective rank for log level
    # (using same detection logic as init_nkipy_backend)
    if rank is not None:
        effective_rank = rank
    elif dist.is_initialized():
        effective_rank = dist.get_rank()
    elif "LOCAL_RANK" in os.environ:
        effective_rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        effective_rank = int(os.environ["RANK"])
    else:
        effective_rank = 0

    # Use provided log_level, or default to INFO for rank 0, ERROR for others
    if log_level is None:
        log_level = logging.INFO if effective_rank == 0 else logging.ERROR

    # init_nkipy_backend will auto-detect rank/world_size if None
    init_nkipy_backend(
        nkipy_cache=cache_path,
        rank=rank,
        world_size=world_size,
        log_level=log_level,
    )


def compile_and_wrap_model(
    model: Any,
) -> Any:
    # Wrap the forward method
    compiled_forward = torch.compile(
        model.forward, backend="nkipy", dynamic=False, fullgraph=True
    )

    # Create a wrapper function that uses the compiled forward
    def model_forward_wrap(*args: Any, **kwargs: Any) -> Any:
        return forward_wrap(compiled_forward, *args, **kwargs)

    # Replace the model's forward method
    model.forward = model_forward_wrap

    # Move the model to the target device
    model.to("nkipy")

    return model


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "float32",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.generation_config.cache_implementation = "static"
    model.config._attn_implementation = "eager"

    return model, tokenizer
