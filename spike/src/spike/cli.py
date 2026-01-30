"""CLI for running NEFFs on Trainium hardware.

This module provides the `spike-run` command-line interface for executing
compiled NEFF files on AWS Trainium NeuronCores.

Execution Modes
---------------
Single-core (default):
    Run on one NeuronCore. Use --core-id to select which core.

Multi-core (--num-cores N):
    Run on N NeuronCores with per-core input data. Each core executes
    independently with its own inputs. Use {rank} placeholder in input
    paths for per-core files.

Collective (--num-cores N --collective):
    Run on N NeuronCores with collective compute enabled. Cores can
    communicate via all-reduce, all-gather, etc. Use {rank} placeholder
    for per-core inputs.

Examples
--------
Single-core execution::

    spike-run kernel.neff -i "input=data.npy" -o ./output

Multi-core independent execution::

    spike-run kernel.neff -i "input=data_{rank}.npy" --num-cores 2 -o ./output
    # Loads data_0.npy for core 0, data_1.npy for core 1

Multi-core with collective compute::

    spike-run cc_kernel.neff -i "input=data_{rank}.npy" --num-cores 2 --collective

Benchmarking::

    spike-run kernel.neff -i "input=data.npy" --benchmark --warmup 5 --iterations 100

Save execution trace::

    spike-run kernel.neff -i "input=data.npy" --save-trace

Core Configuration
------------------
The --num-cores option specifies the number of **logical** NeuronCores to use.

Logical vs Physical Cores (Trn2 only):
    The NEURON_LOGICAL_NC_CONFIG env var controls how physical cores map to logical cores.
    - NEURON_LOGICAL_NC_CONFIG=2 (default): trn2.48xlarge has 64 logical cores (0-63)
    - NEURON_LOGICAL_NC_CONFIG=1: trn2.48xlarge has 128 logical cores (0-127)

Environment variables:
    NEURON_RT_VISIBLE_CORES: Specifies which logical cores to use (e.g., "0-3", "4-7")
    NEURON_RT_NUM_CORES: Specifies how many logical cores to use (runtime picks which)

If neither NEURON_RT_VISIBLE_CORES nor NEURON_RT_NUM_CORES is set, spike-run sets
NEURON_RT_VISIBLE_CORES=0-(N-1) based on --num-cores.
If an env var is set, spike-run validates it matches --num-cores and uses the user's setting.

Examples::

    # Use logical cores 0-3 (default behavior)
    spike-run kernel.neff --num-cores 4

    # Use specific logical cores 4-7 (user controls which cores)
    NEURON_RT_VISIBLE_CORES=4-7 spike-run kernel.neff --num-cores 4
"""

import argparse
import os
import threading

import numpy as np

from .spike_model import SpikeModel
from .spike_singleton import configure
from .spike_tensor import SpikeTensor


def parse_visible_cores(env_value: str) -> list:
    """Parse NEURON_RT_VISIBLE_CORES env var into list of core IDs.

    Supports formats: "0-3", "0,1,2,3", "0-3,8-11"
    """
    cores = []
    for part in env_value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            cores.extend(range(int(start), int(end) + 1))
        else:
            cores.append(int(part))
    return cores


def validate_core_config(
    num_cores: int,
    visible_cores_env: str | None,
    num_cores_env: str | None,
    verbose: bool = False,
) -> None:
    """Validate user's core env vars against --num-cores, raise on mismatch.

    Args:
        num_cores: The --num-cores CLI argument value.
        visible_cores_env: Value of NEURON_RT_VISIBLE_CORES env var (or None).
        num_cores_env: Value of NEURON_RT_NUM_CORES env var (or None).
        verbose: If True, print which env var is being used.

    Raises:
        ValueError: If env var core count doesn't match num_cores.
    """
    if visible_cores_env:
        env_core_count = len(parse_visible_cores(visible_cores_env))
        if env_core_count != num_cores:
            raise ValueError(
                f"--num-cores={num_cores} conflicts with "
                f"NEURON_RT_VISIBLE_CORES={visible_cores_env} ({env_core_count} cores). "
                f"Either unset the env var or use --num-cores={env_core_count}."
            )
        if verbose:
            print(f"Using NEURON_RT_VISIBLE_CORES={visible_cores_env}")

    if num_cores_env:
        env_num = int(num_cores_env)
        if env_num != 0 and env_num != num_cores:
            raise ValueError(
                f"--num-cores={num_cores} conflicts with "
                f"NEURON_RT_NUM_CORES={num_cores_env}. "
                f"Either unset the env var or use --num-cores={env_num}."
            )
        if verbose:
            print(f"Using NEURON_RT_NUM_CORES={num_cores_env}")


def load_inputs_for_core(input_specs, core_id):
    """Load input tensors for a specific core, expanding {rank} placeholders.

    Args:
        input_specs: List of "name=path" strings. Path may contain {rank}
            placeholder which gets replaced with core_id.
        core_id: The core ID to substitute for {rank}.

    Returns:
        Dict mapping tensor names to numpy arrays.

    Example:
        >>> load_inputs_for_core(["x=data_{rank}.npy"], core_id=1)
        # Loads data_1.npy and returns {"x": <numpy array>}
    """
    inputs = {}
    for spec in input_specs or []:
        name, path_pattern = spec.split("=", 1)
        path = path_pattern.replace("{rank}", str(core_id))
        arr = np.load(path)
        inputs[name] = arr
    return inputs


def run_single_core(args):
    """Execute NEFF on a single NeuronCore.

    Loads the model, creates input tensors, executes, and saves outputs.
    Supports benchmarking and trace capture modes.
    """
    model = SpikeModel.load_from_neff(
        args.neff_path,
        name=args.name,
        core_id=args.core_id,
        cc_enabled=args.cc_enabled,
        rank_id=args.rank_id,
        world_size=args.world_size,
    )

    if args.verbose:
        print(f"Model: {model.name}")
        print(f"Inputs: {model.input_tensors_info}")
        print(f"Outputs: {model.output_tensors_info}")

    # Load inputs
    spike_inputs = {}
    for inp in args.inputs or []:
        name, path = inp.split("=", 1)
        arr = np.load(path)
        spike_inputs[name] = SpikeTensor.from_numpy(arr, name, core_id=args.core_id)
        if args.verbose:
            print(f"Loaded input '{name}': shape={arr.shape}, dtype={arr.dtype}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Execute
    if args.benchmark:
        result = model.benchmark(
            spike_inputs,
            warmup_iter=args.warmup,
            benchmark_iter=args.iterations,
        )
        if args.verbose:
            print(f"Benchmark: {result}")
        spike_outputs = model(spike_inputs)
    else:
        spike_outputs = model(
            spike_inputs,
            save_trace=args.save_trace,
            ntff_name=args.ntff_name,
        )
        if args.save_trace and args.verbose:
            ntff = args.ntff_name or f"{os.path.splitext(args.neff_path)[0]}.ntff"
            print(f"Trace saved: {ntff}")

    # Save outputs
    for name, tensor in spike_outputs.items():
        out_path = os.path.join(args.output_dir, f"{name}.npy")
        np.save(out_path, tensor.numpy())
        if args.verbose:
            print(f"Saved '{name}' to {out_path}")

    if args.verbose:
        print(f"Outputs: {args.output_dir}")


def run_multi_core(args):
    """Execute NEFF on multiple NeuronCores in parallel.

    Each core gets its own input data (via {rank} placeholder expansion)
    and produces separate outputs in core-specific subdirectories.

    When --collective is enabled, cores can communicate via collective
    operations (all-reduce, all-gather, etc.).
    """
    num_cores = args.num_cores
    collective = args.collective

    # Check if user has already configured core allocation
    visible_cores_env = os.environ.get("NEURON_RT_VISIBLE_CORES")
    num_cores_env = os.environ.get("NEURON_RT_NUM_CORES")

    if visible_cores_env is None and num_cores_env is None:
        # User hasn't configured cores - set based on --num-cores
        configure(visible_cores=list(range(num_cores)))
        if args.verbose:
            print(f"Set NEURON_RT_VISIBLE_CORES=0-{num_cores-1}")
    else:
        # User has configured cores - validate and respect their setting
        validate_core_config(
            num_cores, visible_cores_env, num_cores_env, verbose=args.verbose
        )

    if args.verbose:
        print(f"Multi-core mode: {num_cores} cores, collective={collective}")

    # Load inputs per core (expanding {rank} placeholders)
    inputs_per_core = [
        load_inputs_for_core(args.inputs, i)
        for i in range(num_cores)
    ]

    # Load model on each core
    models = [
        SpikeModel.load_from_neff(
            args.neff_path,
            name=args.name,
            core_id=i,
            cc_enabled=collective,
            rank_id=i if collective else 0,
            world_size=num_cores if collective else 1,
        )
        for i in range(num_cores)
    ]

    if args.verbose:
        print(f"Model: {models[0].name}")
        print(f"Inputs: {models[0].input_tensors_info}")
        print(f"Outputs: {models[0].output_tensors_info}")

    # Create SpikeTensors for each core
    spike_inputs_per_core = [
        {name: SpikeTensor.from_numpy(arr, name, core_id=i)
         for name, arr in inputs_per_core[i].items()}
        for i in range(num_cores)
    ]

    if args.verbose:
        for i in range(num_cores):
            for name, arr in inputs_per_core[i].items():
                print(f"Core {i}: Loaded input '{name}': shape={arr.shape}, dtype={arr.dtype}")

    # Execute in parallel using threads (spike releases GIL during execute)
    results = [None] * num_cores
    errors = [None] * num_cores

    def run_on_core(core_id):
        try:
            if args.benchmark:
                bench_result = models[core_id].benchmark(
                    spike_inputs_per_core[core_id],
                    warmup_iter=args.warmup,
                    benchmark_iter=args.iterations,
                )
                if args.verbose:
                    print(f"Core {core_id} benchmark: {bench_result}")
                results[core_id] = models[core_id](spike_inputs_per_core[core_id])
            else:
                results[core_id] = models[core_id](spike_inputs_per_core[core_id])
        except Exception as e:
            errors[core_id] = e

    threads = [threading.Thread(target=run_on_core, args=(i,)) for i in range(num_cores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Check for errors
    for i, err in enumerate(errors):
        if err is not None:
            raise RuntimeError(f"Core {i} failed: {err}") from err

    # Save outputs per core in separate subdirectories
    for i in range(num_cores):
        core_output_dir = os.path.join(args.output_dir, f"core_{i}")
        os.makedirs(core_output_dir, exist_ok=True)
        for name, tensor in results[i].items():
            out_path = os.path.join(core_output_dir, f"{name}.npy")
            np.save(out_path, tensor.numpy())
            if args.verbose:
                print(f"Core {i}: Saved '{name}' to {out_path}")

    if args.verbose:
        print(f"Outputs: {args.output_dir}/core_{{0..{num_cores-1}}}")


def main():
    """Entry point for spike-run command."""
    parser = argparse.ArgumentParser(
        prog="spike-run",
        description="Run a compiled NEFF on Trainium hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-core execution
  spike-run kernel.neff -i "input=data.npy" -o ./output

  # Multi-core with per-core inputs ({rank} expands to 0, 1, ...)
  spike-run kernel.neff -i "input=data_{rank}.npy" --num-cores 2

  # Multi-core with collective compute
  spike-run cc_kernel.neff -i "input=data_{rank}.npy" --num-cores 2 --collective

  # Benchmarking
  spike-run kernel.neff -i "input=data.npy" --benchmark --iterations 100
""",
    )
    parser.add_argument("neff_path", help="Path to .neff file")
    parser.add_argument(
        "--input", "-i", action="append", dest="inputs", metavar="NAME=PATH",
        help="Input tensor as name=path.npy (repeatable). Use {rank} for per-core files.",
    )
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print tensor info")

    # Trace options
    parser.add_argument("--save-trace", action="store_true", help="Save execution trace")
    parser.add_argument("--ntff-name", help="Path for trace file (default: <neff>.ntff)")

    # Benchmarking options
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")

    # Single-core options
    parser.add_argument("--name", help="Model name (default: neff filename)")
    parser.add_argument("--core-id", type=int, default=0, help="NeuronCore ID (default: 0)")
    parser.add_argument("--cc-enabled", action="store_true", help="Enable collective compute (single-core)")
    parser.add_argument("--rank-id", type=int, default=0, help="Rank ID for distributed (default: 0)")
    parser.add_argument("--world-size", type=int, default=1, help="World size for distributed (default: 1)")

    # Multi-core options
    parser.add_argument(
        "--num-cores", type=int, default=1,
        help="Number of logical NeuronCores. Respects NEURON_RT_VISIBLE_CORES if set."
    )
    parser.add_argument("--collective", action="store_true", help="Enable collective compute (multi-core)")

    args = parser.parse_args()

    if args.num_cores > 1:
        run_multi_core(args)
    else:
        run_single_core(args)


if __name__ == "__main__":
    main()
