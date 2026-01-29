"""Single-device inference script for Qwen3 models on Neuron devices."""

import warnings

from args import add_single_device_args, create_base_parser
from generation import prepare_generation_kwargs, run_generation, run_warmup_generation
from hf_utils import (
    compile_and_wrap_model,
    load_model_and_tokenizer,
    setup_neuron_environment,
)
from model_config import get_single_device_models
from transformers import logging

warnings.filterwarnings("ignore", module="transformers")
logging.set_verbosity_error()

SUPPORTED_MODELS = get_single_device_models()
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def main():
    """Main entry point for single-device inference."""
    parser = create_base_parser(
        description="Run single-device inference with Qwen3 on AWS Neuron",
        supported_models=SUPPORTED_MODELS,
        default_model=DEFAULT_MODEL,
    )
    add_single_device_args(parser)
    args = parser.parse_args()

    # Setup environment and load model
    setup_neuron_environment(cache_dir=args.cache_dir)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=args.dtype)

    # Prepare input and generation config
    input_ids = tokenizer(args.prompt, return_tensors="pt")
    generation_kwargs = prepare_generation_kwargs(args.max_length)

    # Run host generation for comparison
    outputs = model.generate(**input_ids, **generation_kwargs)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"Output (host):\n{result[0]}\n", flush=True)

    # Compile, warm up, and run generation
    model_compiled = compile_and_wrap_model(model)
    run_warmup_generation(model_compiled, input_ids, generation_kwargs)
    run_generation(model_compiled, input_ids, tokenizer, generation_kwargs)


if __name__ == "__main__":
    main()
