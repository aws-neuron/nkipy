# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ml_dtypes import bfloat16, float8_e4m3, float8_e5m2

from ._spike import NrtModel
from .logger import get_logger
from .spike_singleton import get_spike_singleton
from .spike_tensor import SpikeTensor

logger = get_logger()


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
                "expected={expected_size}"
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
                "got {expected_dtype}"
            )
        else:
            # Strict dtype checking
            assert actual_dtype == expected_dtype, (
                f"{tensor_type} {tensor_name}: expected dtype {expected_dtype}, "
                "got {actual_dtype}"
            )

    def _validate_io(self, inputs, outputs):
        model_core_id = self.model_ref.core_id
        for k, v in inputs.items():
            tensor_core_id = v.tensor_ref.core_id
            assert tensor_core_id == model_core_id, (
                f"Input {k}: expected model and tensor on the same core, "
                "got model core_id {model_core_id} and tensor core_id {tensor_core_id}"
            )
            assert v.shape == tuple(self.input_tensors_info[k].shape), (
                f"Input {k}: expected shape {self.input_tensors_info[k].shape}, "
                "got {v.shape}"
            )
            self._check_dtype_compatibility(
                v.dtype, self.input_tensors_info[k].dtype, k, is_input=True
            )

        for k, v in outputs.items():
            tensor_core_id = v.tensor_ref.core_id
            assert tensor_core_id == model_core_id, (
                f"Output {k}: expected model and tensor on the same core, "
                "got model core_id {model_core_id} and tensor core_id {tensor_core_id}"
            )
            assert v.shape == tuple(self.output_tensors_info[k].shape), (
                f"Output {k}: expected shape {self.output_tensors_info[k].shape}, "
                "got {v.shape}"
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
    ):
        """Benchmark the model execution.

        Args:
            inputs: Dict[str, SpikeTensor]. Key needs to match neff input name.
            outputs: Dict[str, SpikeTensor]. Key needs to match neff output name.
            warmup_iter: Number of warmup iterations
            benchmark_iter: Number of benchmark iterations

        Returns:
            Benchmark results from spike
        """
        if outputs is None:
            output_tensors = self.allocate_output_tensors()
            outputs = {tensor.name: tensor for tensor in output_tensors}

        self._validate_io(inputs, outputs)

        inputs = {k: v.tensor_ref for k, v in inputs.items()}
        outputs = {k: v.tensor_ref for k, v in outputs.items()}

        logger.info(f"Benchmarking model: {self.model_ref}")
        return get_spike_singleton().benchmark(
            self.model_ref,
            inputs=inputs,
            outputs=outputs,
            warmup_iterations=warmup_iter,
            benchmark_iterations=benchmark_iter,
        )
