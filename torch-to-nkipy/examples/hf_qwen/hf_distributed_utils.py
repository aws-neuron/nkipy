"""Utilities for distributed tensor parallelism with Hugging Face models."""

import math
from typing import Any, Dict, List, Union

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    PretrainedConfig,
    StaticCache,
)
from transformers.modeling_utils import _get_resolved_checkpoint_files
from transformers.utils import ContextManagers

# Patch for StaticCache to work with distributed tensors
original_static_cache_init = StaticCache.__init__


def static_cache_init_dist(*args: Any, **kwargs: Any):
    config = kwargs["config"]
    assert isinstance(config, PretrainedConfig)

    # Adjust heads for distributed execution
    num_key_value_heads = (
        config.num_attention_heads
        if getattr(config, "num_key_value_heads", None) is None
        else config.num_key_value_heads
    )
    num_key_value_heads_dist = int(num_key_value_heads / dist.get_world_size())
    config.num_key_value_heads = num_key_value_heads_dist

    # Call original init
    res = original_static_cache_init(*args, **kwargs)

    # Restore original config
    config.num_key_value_heads = num_key_value_heads

    return res


def initialize_distributed(backend: str = "nkipy") -> tuple:
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = init_device_mesh("nkipy", (world_size,))

    # Set current device to 0 since each process only sees its own core
    # (NEURON_RT_VISIBLE_CORES is set per-rank before import)
    torch.nkipy.set_device(0)

    # Apply StaticCache patch
    StaticCache.__init__ = static_cache_init_dist

    return rank, world_size, mesh


def create_tp_plan_from_config(model: Any, mesh: Any) -> Dict:
    if hasattr(model.config, "base_model_tp_plan"):
        base_model_tp_plan = model.config.base_model_tp_plan
    else:
        # Fallback for Qwen3 and other models without base_model_tp_plan
        base_model_tp_plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.mlp.gate_proj": "colwise",
            "layers.*.mlp.up_proj": "colwise",
            "layers.*.mlp.down_proj": "rowwise",
        }

    num_hidden_layers = model.config.num_hidden_layers
    concrete_tp_plan = {}

    # Convert pattern-based plan to concrete layer plan
    for pattern, parallelism_type in base_model_tp_plan.items():
        for layer_idx in range(num_hidden_layers):
            concrete_key = pattern.replace("*", str(layer_idx))
            if hasattr(model, "model"):
                concrete_key = "model." + concrete_key
            else:
                raise KeyError(f"Base model cannot be located in {model}")

            # Convert string parallelism type to actual class
            if parallelism_type == "colwise":
                parallel_op = ColwiseParallel()
            elif parallelism_type == "rowwise":
                parallel_op = RowwiseParallel()
            else:
                continue

            concrete_tp_plan[concrete_key] = parallel_op

    return concrete_tp_plan


def init_meta_model(model_name: str, dtype: torch.dtype, world_size: int) -> Any:
    config = AutoConfig.from_pretrained(model_name)

    # Handle KV heads compatibility with tensor parallelism
    if hasattr(config, "num_key_value_heads"):
        if config.num_key_value_heads >= world_size:
            if config.num_key_value_heads % world_size != 0:
                raise ValueError(
                    f"Number of KV heads ({config.num_key_value_heads}) must be "
                    f"divisible by world_size ({world_size}) for efficient "
                    f"tensor parallelism"
                )
        else:
            if world_size % config.num_key_value_heads != 0:
                raise ValueError(
                    f"When world_size ({world_size}) is larger than the number "
                    f"of KV heads ({config.num_key_value_heads}), world_size "
                    f"must be divisible by num_key_value_heads"
                )
            config.num_key_value_heads = world_size

    # Create and initialize the model
    model_class = MODEL_FOR_CAUSAL_LM_MAPPING[type(config)]
    model_init_context = model_class.get_init_context(
        is_quantized=False, _is_ds_init_called=False
    )

    with ContextManagers(model_init_context):
        model = model_class(config).to(dtype)

    # Configure generation settings
    model.generation_config.cache_implementation = "static"
    model.config._attn_implementation = "eager"

    return model


def get_checkpoint_files(model_name: str) -> List[str]:
    checkpoint_files, _ = _get_resolved_checkpoint_files(
        pretrained_model_name_or_path=model_name,
        subfolder="",
        variant=None,
        gguf_file=None,
        from_tf=False,
        from_flax=False,
        use_safetensors=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        local_files_only=False,
        token=None,
        user_agent={
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": False,
        },
        revision="main",
        commit_hash=None,
        is_remote_code=False,
    )

    if not checkpoint_files:
        raise RuntimeError(f"No checkpoint files found for model {model_name}")

    return checkpoint_files


def shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    tp_plan: Dict[str, Union[ColwiseParallel, RowwiseParallel]],
    tp_rank: int,
    tp_degree: int,
    head_dim: int,
) -> Dict[str, torch.Tensor]:
    def param_to_module_name(param_name: str) -> str:
        return ".".join(param_name.split(".")[:-1])

    def shard(
        tensor: torch.Tensor, dim: int, tp_rank: int, tp_degree: int
    ) -> torch.Tensor:
        if tensor.shape[dim] % tp_degree != 0:
            raise ValueError(
                f"Tensor of shape {tensor.shape} cannot be evenly sharded along "
                f"dimension {dim} with degree {tp_degree}."
            )
        chunks = torch.chunk(tensor, tp_degree, dim=dim)
        return chunks[tp_rank].contiguous().clone()

    def shard_kv(
        tensor: torch.Tensor, tp_rank: int, tp_degree: int, head_dim: int
    ) -> torch.Tensor:
        num_key_value_heads = tensor.shape[0] // head_dim

        if num_key_value_heads >= tp_degree:
            # When we have more KV heads than TP degree, we shard normally
            return shard(tensor, dim=0, tp_rank=tp_rank, tp_degree=tp_degree)
        else:
            # When we have fewer KV heads than TP degree, we need special handling
            if tensor.ndim == 2:  # weight
                tensor = tensor.reshape(num_key_value_heads, head_dim, -1)
            elif tensor.ndim == 1:  # bias
                tensor = tensor.reshape(num_key_value_heads, head_dim)
            else:
                raise ValueError(
                    f"Expected tensor dim == 2 (weight) or 1 (bias), "
                    f"but got ndim = = {tensor.ndim}"
                )
            # Calculate which head goes to which rank
            head_index = math.floor(num_key_value_heads * tp_rank / tp_degree)

            if head_index >= num_key_value_heads:
                raise ValueError(
                    f"Calculated head_index {head_index} exceeds available heads "
                    f"{num_key_value_heads} for rank {tp_rank}/{tp_degree}"
                )

            ret = tensor[head_index].contiguous().clone()
            return ret

    # Identify KV projection patterns
    KV_PROJECTION_PATTERNS = [".k_proj", ".v_proj", ".key_proj", ".value_proj"]

    def is_kv_projection(name: str) -> bool:
        """Identify if a parameter belongs to key or value projection layers."""
        return any(pattern in name for pattern in KV_PROJECTION_PATTERNS)

    # Create a copy of state_dict to avoid modifying the input
    sharded_dict = {}

    # Process all parameters
    for param_name, tensor in state_dict.items():
        module_name = param_to_module_name(param_name)

        if module_name not in tp_plan:
            # Parameter doesn't need sharding - copy directly
            sharded_dict[param_name] = tensor
            continue

        # Handle KV projections with special sharding
        if is_kv_projection(module_name):
            if not isinstance(tp_plan[module_name], ColwiseParallel):
                raise TypeError(
                    f"KV projection {module_name} must use ColwiseParallel sharding, "
                    f"but found {type(tp_plan[module_name]).__name__}"
                )

            sharded_dict[param_name] = shard_kv(
                tensor, tp_rank=tp_rank, tp_degree=tp_degree, head_dim=head_dim
            )

        # Handle standard column-wise parallelism
        elif isinstance(tp_plan[module_name], ColwiseParallel):
            sharded_dict[param_name] = shard(
                tensor, dim=0, tp_rank=tp_rank, tp_degree=tp_degree
            )

        # Handle standard row-wise parallelism
        elif isinstance(tp_plan[module_name], RowwiseParallel):
            # Do not shard if the tensor is a bias weight
            if tensor.ndim == 1:
                sharded_dict[param_name] = tensor
            else:
                sharded_dict[param_name] = shard(
                    tensor, dim=1, tp_rank=tp_rank, tp_degree=tp_degree
                )

        else:
            raise TypeError(
                f"Unknown sharding type for '{module_name}': "
                f"{type(tp_plan[module_name]).__name__}"
            )

    return sharded_dict


def load_state_dict_to_nkipy(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor], dtype: torch.dtype
) -> torch.nn.Module:
    def update_tensor(name: str, target: torch.Tensor, source: torch.Tensor):
        src = source.to(device="nkipy", dtype=dtype)

        if isinstance(target, DTensor):
            dtensor = DTensor.from_local(src, target.device_mesh, target.placements)
            torch.utils.swap_tensors(target, dtensor)

        elif isinstance(target, torch.nn.Parameter):
            param = torch.nn.Parameter(src, requires_grad=target.requires_grad)
            torch.utils.swap_tensors(target, param)

        else:  # buffer
            torch.utils.swap_tensors(target, src)

    # Collect all parameters/buffers
    param_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())
    all_names = set(param_map) | set(buffer_map)

    # Warn on unexpected keys
    unexpected_keys = [k for k in state_dict if k not in all_names]
    if unexpected_keys:
        preview = ", ".join(unexpected_keys[:10])
        if len(unexpected_keys) > 10:
            preview += ", ..."
        print(
            f"[Warning] {len(unexpected_keys)} unexpected keys in state_dict: {preview}"
        )

    # Update params/buffers
    for name, tensor in state_dict.items():
        if name in param_map:
            update_tensor(name, param_map[name], tensor)
        elif name in buffer_map:
            update_tensor(name, buffer_map[name], tensor)

    return model


def load_parallelized_model(
    model_name: str, dtype: str, rank: int, world_size: int, mesh: Any
) -> Any:
    import json
    import time

    try:
        # Convert string dtype to torch.dtype
        torch_dtype = getattr(torch, dtype)
    except AttributeError:
        raise ValueError(f"Invalid dtype: {dtype}. Must be a valid torch.dtype name.")

    # Initialize the model
    model = init_meta_model(model_name, torch_dtype, world_size)

    # Create parallelism plan and parallelize the model
    tp_plan = create_tp_plan_from_config(model, mesh)
    parallelized_model = parallelize_module(model, mesh, tp_plan)

    # Load checkpoints - rank 0 resolves files first, writes to temp file,
    # other ranks wait and retry until file is available
    cache_file = "/tmp/nkipy_checkpoint_files.json"

    if rank == 0:
        checkpoint_files = get_checkpoint_files(model_name)
        with open(cache_file, "w") as f:
            json.dump(checkpoint_files, f)
    else:
        # Wait for rank 0 to write the file, retry up to 60 seconds
        for _ in range(120):
            try:
                with open(cache_file, "r") as f:
                    checkpoint_files = json.load(f)
                break
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.5)
        else:
            raise RuntimeError(f"Rank {rank}: Timeout waiting for checkpoint file list")

    config = model.config
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    for checkpoint_file in checkpoint_files:
        # Safetensors load_file may have issues for certain models
        # Fall back to torch.load if needed
        try:
            state_dict = load_file(checkpoint_file)
        except Exception:
            state_dict = torch.load(checkpoint_file, map_location="cpu")
        sharded_state_dict = shard_state_dict(
            state_dict, tp_plan, rank, world_size, head_dim
        )
        load_state_dict_to_nkipy(parallelized_model, sharded_state_dict, torch_dtype)

        # Free up memory
        del state_dict
        del sharded_state_dict

    # Tie weights if needed
    parallelized_model.tie_weights()

    # set the model to eval mode
    parallelized_model.eval()

    return parallelized_model


def master_print(msg, flush=False):
    if dist.get_rank() == 0:
        print(msg, flush=flush)
