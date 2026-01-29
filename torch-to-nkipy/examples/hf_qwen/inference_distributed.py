"""Distributed tensor-parallel inference script for Qwen3 models on Neuron devices."""

import warnings

from args import add_distributed_args, create_base_parser
from generation import prepare_generation_kwargs, run_generation, run_warmup_generation
from hf_distributed_utils import (
    initialize_distributed,
    load_parallelized_model,
    master_print,
)
from hf_utils import compile_and_wrap_model, setup_neuron_environment
from model_config import get_distributed_models
from transformers import AutoTokenizer, logging

warnings.filterwarnings("ignore", module="transformers")
logging.set_verbosity_error()

SUPPORTED_MODELS = get_distributed_models()
DEFAULT_MODEL = "Qwen/Qwen3-8B"


def main():
    """Main entry point for distributed inference."""
    parser = create_base_parser(
        description="Run distributed inference with Qwen3 on AWS Neuron",
        supported_models=SUPPORTED_MODELS,
        default_model=DEFAULT_MODEL,
    )
    add_distributed_args(parser)
    args = parser.parse_args()

    # Initialize distributed environment
    rank, world_size, mesh = initialize_distributed()

    # Setup neuron environment
    setup_neuron_environment(rank=rank, world_size=world_size)

    # Load tokenizer and parallelized model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_parallelized_model(args.model, args.dtype, rank, world_size, mesh)

    # Prepare input and generation config
    input_ids = tokenizer(args.prompt, return_tensors="pt")
    generation_kwargs = prepare_generation_kwargs(args.max_length)

    # Compile, warm up, and run generation
    model_compiled = compile_and_wrap_model(model)
    run_warmup_generation(
        model_compiled, input_ids, generation_kwargs, print_fn=master_print
    )
    run_generation(
        model_compiled, input_ids, tokenizer, generation_kwargs, print_fn=master_print
    )


if __name__ == "__main__":
    main()
