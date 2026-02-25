import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ml_dtypes import bfloat16, float8_e4m3, float8_e5m2

from ._spike import NrtModel, SystemTraceSession
from .logger import get_logger
from .spike_singleton import get_spike_singleton
from .spike_tensor import SpikeTensor

logger = get_logger()


@dataclass
class BenchmarkResult:
    """Result of a model benchmark run."""

    mean_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float
    iterations: int
    warmup_iterations: int
    durations_ms: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(mean={self.mean_ms:.4f}ms, "
            f"min={self.min_ms:.4f}ms, "
            f"max={self.max_ms:.4f}ms, "
            f"std_dev={self.std_dev_ms:.4f}ms, "
            f"iterations={self.iterations}, "
            f"warmup_iterations={self.warmup_iterations})"
        )


def _parse_trace_durations(events_json: str) -> list[float]:
    """Parse device execution durations from NRT sys trace JSON.

    Returns:
        List of execution durations in milliseconds.
    """
    if not events_json:
        return []

    root = json.loads(events_json)
    events = root.get("events", [])

    DEVICE_EXECUTE_TYPE = "nc_exec_running"

    # nc_exec_running start timestamps keyed by tracking_id
    starts: dict[int, int] = {}
    durations_ms: list[float] = []

    for event in events:
        if event.get("event_type") != DEVICE_EXECUTE_TYPE:
            continue

        phase = event.get("phase")
        tracking_id = event.get("tracking_id")
        data = event.get("data", {})
        nc_ts = data.get("nc_timestamp_ns")

        if nc_ts is None or tracking_id is None:
            continue

        if phase == "start":
            starts[tracking_id] = nc_ts
        elif phase == "stop":
            start_ts = starts.pop(tracking_id, None)
            if start_ts is not None:
                durations_ms.append((nc_ts - start_ts) / 1_000_000.0)

    return durations_ms


def _make_benchmark_result(
    durations_ms: list[float], warmup_iterations: int
) -> "BenchmarkResult":
    """Compute statistics from a list of durations in milliseconds."""
    n = len(durations_ms)
    mean = sum(durations_ms) / n
    min_val = min(durations_ms)
    max_val = max(durations_ms)
    variance = sum((d - mean) ** 2 for d in durations_ms) / n
    return BenchmarkResult(
        mean_ms=mean,
        min_ms=min_val,
        max_ms=max_val,
        std_dev_ms=math.sqrt(variance),
        iterations=n,
        warmup_iterations=warmup_iterations,
        durations_ms=list(durations_ms),
    )


class SpikeModel:
    """A wrapper class for executing compiled kernels from NEFF files."""

    def __init__(self, model_ref: NrtModel, name: str, neff_path: Path):
        """Initialize a SpikeModel.

        Args:
            model_ref: The loaded model object from load_from_neff
            name: Name for the model (for debugging/profiling)
        """
        self.model_ref = model_ref
        self.name = name
        self.neff_path = neff_path
        tensors_info = get_spike_singleton().get_tensor_info(self.model_ref)
        self.input_tensors_info = tensors_info.inputs
        self.output_tensors_info = tensors_info.outputs

    @classmethod
    def load_from_neff(
        cls,
        neff_path: Path | str,
        name: Optional[str] = None,
        core_id=0,
        cc_enabled=False,
        rank_id=0,
        world_size=1,
    ):
        """Load a NEFF file and return a SpikeModel instance.

        Args:
            neff_path: Path to the NEFF file to load
            name: Optional name for the model. If None, uses the NEFF filename

        Returns:
            SpikeModel: A SpikeModel instance with the loaded model
        """

        neff_path = Path(neff_path)
        if name is None:
            name = neff_path.stem

        logger.info(f"Loading model from: {neff_path}")
        model_ref = get_spike_singleton().load_model(
            str(neff_path), core_id, cc_enabled, rank_id, world_size
        )

        return cls(model_ref, name, neff_path)

    def allocate_output_tensors(self) -> List[SpikeTensor]:
        """Allocate output tensors based on model tensor info.

        Returns:
            List[DeviceTensor]: List of allocated output tensors
        """
        core_id = self.model_ref.core_id
        output_tensors = []
        for k, v in self.output_tensors_info.items():
            # Convert string dtype to numpy dtype
            if isinstance(v.dtype, str):
                dtype = v.dtype

                if dtype == "bfloat16":
                    dtype = bfloat16
                dtype = np.dtype(dtype)
                # FIXME: need to handle fp8 --
                # fine right now because they are currently as int8 in NEFF

            expected_size = v.size

            tensor = np.zeros(v.shape, dtype=dtype)
            assert tensor.nbytes == expected_size, (
                f"Size mismatch: tensor.nbytes={tensor.nbytes}, "
                f"expected={expected_size}"
            )
            output_tensors.append(
                SpikeTensor.from_numpy(tensor, k, core_id=core_id)
            )  # allocate to the same core
        return output_tensors

    def _check_dtype_compatibility(
        self, actual_dtype, expected_dtype, tensor_name: str, is_input: bool
    ):
        tensor_type = "Input" if is_input else "Output"
        # FIXME: Pending NRT proper handling of FP8 Dtypes
        if actual_dtype in {np.dtype(float8_e4m3), np.dtype(float8_e5m2)}:
            assert expected_dtype == "int8", (
                f"{tensor_type} {tensor_name}: expected dtype int8 for fp8 types, "
                f"got {expected_dtype}"
            )
        else:
            # Strict dtype checking
            assert actual_dtype == expected_dtype, (
                f"{tensor_type} {tensor_name}: expected dtype {expected_dtype}, "
                f"got {actual_dtype}"
            )

    def _validate_io(self, inputs, outputs):
        model_core_id = self.model_ref.core_id
        for k, v in inputs.items():
            tensor_core_id = v.tensor_ref.core_id
            assert tensor_core_id == model_core_id, (
                f"Input {k}: expected model and tensor on the same core, "
                f"got model core_id {model_core_id} and tensor core_id {tensor_core_id}"
            )
            assert v.shape == tuple(self.input_tensors_info[k].shape), (
                f"Input {k}: expected shape {self.input_tensors_info[k].shape}, "
                f"got {v.shape}"
            )
            self._check_dtype_compatibility(
                v.dtype, self.input_tensors_info[k].dtype, k, is_input=True
            )

        for k, v in outputs.items():
            tensor_core_id = v.tensor_ref.core_id
            assert tensor_core_id == model_core_id, (
                f"Output {k}: expected model and tensor on the same core, "
                f"got model core_id {model_core_id} and tensor core_id {tensor_core_id}"
            )
            assert v.shape == tuple(self.output_tensors_info[k].shape), (
                f"Output {k}: expected shape {self.output_tensors_info[k].shape}, "
                f"got {v.shape}"
            )
            self._check_dtype_compatibility(
                v.dtype, self.output_tensors_info[k].dtype, k, is_input=False
            )

    def __call__(
        self,
        inputs: Dict[str, SpikeTensor],
        outputs: Dict[str, SpikeTensor] = None,
        save_trace: bool = False,
        ntff_name: Optional[str] = None,
    ) -> None:
        """Execute the model forward pass.

        Args:
            inputs: Dict[str, SpikeTensor]. Key needs to match neff input name.
            outputs: Dict[str, SpikeTensor]. Key needs to match neff output name.
            save_trace: Whether to save execution trace
            ntff_name: Optional name for the trace file
        """
        auto_allocated = False
        if outputs is None:
            output_tensors = self.allocate_output_tensors()
            outputs = {tensor.name: tensor for tensor in output_tensors}
            auto_allocated = True

        self._validate_io(inputs, outputs)

        input_refs = {k: v.tensor_ref for k, v in inputs.items()}
        output_refs = {k: v.tensor_ref for k, v in outputs.items()}

        # if ntff_name not specified, always keep it around the same dir as neff
        if ntff_name is None:
            ntff_name = str(self.neff_path.with_suffix("")) + ".ntff"

        logger.info(f"Executing model: {self.model_ref}")
        get_spike_singleton().execute(
            self.model_ref,
            inputs=input_refs,
            outputs=output_refs,
            save_trace=save_trace,
            ntff_name=ntff_name,
        )

        if auto_allocated:
            return outputs

    def benchmark(
        self,
        inputs: Dict[str, SpikeTensor],
        outputs: Dict[str, SpikeTensor] = None,
        warmup_iter=5,
        benchmark_iter=5,
        mode: str = "device",
    ) -> BenchmarkResult:
        """Benchmark the model execution.

        Args:
            inputs: Dict[str, SpikeTensor]. Key needs to match neff input name.
            outputs: Dict[str, SpikeTensor]. Key needs to match neff output name.
            warmup_iter: Number of warmup iterations
            benchmark_iter: Number of benchmark iterations
            mode: Timing mode.
                'device' — NeuronCore execution time via device-side trace
                           (nc_exec_running, device clock). Most accurate for
                           kernel timing.
                'host'   — Host wall-clock time. Includes all host-device
                           overhead. No tracing overhead.

        Returns:
            BenchmarkResult with mean_ms, min_ms, max_ms, std_dev_ms,
            iterations, warmup_iterations, durations_ms.
        """
        if mode not in ("device", "host"):
            raise ValueError(
                f"Invalid benchmark mode '{mode}'. Use 'device' or 'host'."
            )

        if outputs is None:
            output_tensors = self.allocate_output_tensors()
            outputs = {tensor.name: tensor for tensor in output_tensors}

        self._validate_io(inputs, outputs)

        input_refs = {k: v.tensor_ref for k, v in inputs.items()}
        output_refs = {k: v.tensor_ref for k, v in outputs.items()}
        spike = get_spike_singleton()

        logger.info(f"Benchmarking model: {self.model_ref}")

        if mode == "device":
            with SystemTraceSession(self.model_ref.core_id) as trace:
                for _ in range(warmup_iter):
                    spike.execute(self.model_ref, input_refs, output_refs)
                trace.drain_events()

                for _ in range(benchmark_iter):
                    spike.execute(self.model_ref, input_refs, output_refs)

                events_json = trace.fetch_events_json()

            durations_ms = _parse_trace_durations(events_json)
            if not durations_ms:
                raise RuntimeError(
                    "No nc_exec_running events captured during benchmark. "
                    "Cannot compute device-side timing."
                )
            return _make_benchmark_result(durations_ms, warmup_iter)

        # Host mode: wall-clock timing
        for _ in range(warmup_iter):
            spike.execute(self.model_ref, input_refs, output_refs)

        durations_ms = []
        for _ in range(benchmark_iter):
            start = time.perf_counter()
            spike.execute(self.model_ref, input_refs, output_refs)
            elapsed = time.perf_counter() - start
            durations_ms.append(elapsed * 1000.0)

        return _make_benchmark_result(durations_ms, warmup_iter)
