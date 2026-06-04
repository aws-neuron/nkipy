# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiler wrappers to lower NKIPy kernels"""

import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from neuronxcc.driver.CommandDriver import main as neuronx_cc_main

from nkipy.core.backend.hlo import HLOModule
from nkipy.core.trace import NKIPyKernel


def _lite_dtype_to_kb(lite_dtype):
    """Convert a nkigen_lite DType to a kernel_builder dtype."""
    import nki.compiler.kernel_builder as nb
    from nkigen_lite.core import DType
    _map = {
        DType.F32: nb.float32,
        DType.F16: nb.float16,
        DType.BF16: nb.bfloat16,
        DType.TF32: nb.tfloat32,
        DType.FP8_E4M3: nb.float8_e4m3fn,
        DType.FP8_E4M3_IEEE: nb.float8_e4m3,
        DType.FP8_E5M2: nb.float8_e5m2,
        DType.FP8_E3M4: nb.float8_e3m4,
        DType.I32: nb.int32,
        DType.I16: nb.int16,
        DType.I8: nb.int8,
        DType.U32: nb.uint32,
        DType.U16: nb.uint16,
        DType.U8: nb.uint8,
        DType.BOOL: nb.uint8,
    }
    return _map[lite_dtype]

trace = NKIPyKernel.trace

# Build directory for compiled kernels
_DEFAULT_BUILD_DIR = "/tmp/build"


def _get_build_dir():
    return _DEFAULT_BUILD_DIR


def _set_build_dir(build_dir):
    global _DEFAULT_BUILD_DIR
    _DEFAULT_BUILD_DIR = build_dir


# Compiler arguments
DEFAULT_ADDITIONAL_COMPILER_ARGS = "--lnc 1"
NKIPY_KERNEL_ADDITIONAL_COMPILER_ARGS = "--internal-tensorizer-opt-level=2"

nkipy_compiler_args = (
    DEFAULT_ADDITIONAL_COMPILER_ARGS + " " + NKIPY_KERNEL_ADDITIONAL_COMPILER_ARGS
)


class CompilationTarget(Enum):
    TRN1 = "trn1"
    TRN2 = "trn2"
    TRN3 = "trn3"
    DEFAULT = "default"


def get_platform_target() -> str:
    _TRN_1 = "trn1"
    _TRN_2 = "trn2"
    _TRN_3 = "trn3"

    fpath = "/sys/devices/virtual/dmi/id/product_name"
    try:
        with open(fpath, "r") as f:
            fc = f.readline()
    except IOError:
        raise RuntimeError(
            'Unable to read platform target. If running on CPU, please supply \
        compiler argument target, with one of supported platform options. Ex: \
        "--target trn1"'
        )

    instance_type = fc.split(".")[0]
    if _TRN_1 in instance_type:
        return CompilationTarget.TRN1
    elif _TRN_2 in instance_type:
        return CompilationTarget.TRN2
    elif _TRN_3 in instance_type:
        return CompilationTarget.TRN3
    else:
        raise RuntimeError(f"Unsupported Platform - {fc}.")


@dataclass
class CompilationConfig:
    """Configuration for compilation process"""

    pipeline: tuple[str] = ("compile", "SaveTemps")
    target: CompilationTarget = CompilationTarget.DEFAULT
    additional_args: str = ""
    neff_name: str = "file.neff"


class CompilationResult:
    """Results from compilation process"""

    def __init__(self, work_dir: Path, neff_path: Path):
        self.work_dir = work_dir
        self.neff_path = neff_path

    def save_artifacts(self, output_dir: Path) -> None:
        """
        Save all compilation artifacts by copying the entire working directory
        """
        output_dir = Path(output_dir)
        if self.work_dir.exists():
            # Create a fresh directory
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(self.work_dir, output_dir)


class Compiler:
    """Handles compilation of traced kernels"""

    def __init__(self, config: CompilationConfig):
        self.config = config

    def _resolve_target(self) -> CompilationTarget:
        if self.config.target != CompilationTarget.DEFAULT:
            return self.config.target
        try:
            return get_platform_target()
        except Exception:
            logging.warning(
                "Failed to detect platform target, falling back to trn2..."
            )
            return CompilationTarget.TRN2

    def _build_hlo_compile_command(self, work_dir: Path) -> List[str]:
        """Build the neuronx-cc command line for HLO compilation."""
        target = self._resolve_target()
        self.config.target = target

        cmd = [
            "neuronx-cc", "compile",
            "--framework", "XLA",
            str(work_dir / "hlo_module.pb"),
            "--pipeline", *self.config.pipeline,
            "--target", target.value,
            f"--output={self.config.neff_name}",
        ]

        if self.config.additional_args:
            cmd.extend(shlex.split(self.config.additional_args))

        return cmd

    @staticmethod
    def _compilation_error(message, cmd=None, result=None):
        """Build a RuntimeError with compiler output when available."""
        parts = [message]
        if cmd is not None:
            parts.append(f"Command: {' '.join(cmd)}")
        if result is not None:
            def decode(b):
                return b.decode("utf-8", errors="replace") if b else ""
            parts.append(f"stderr:\n{decode(result.stderr)}")
            parts.append(f"stdout:\n{decode(result.stdout)}")
        return RuntimeError("\n".join(parts))

    def _compile_hlo(
        self,
        ir,
        work_dir: Path,
        output_file: str,
        use_neuronx_cc_python_interface: bool = False,
    ) -> Path:
        """Compile an HLOModule to NEFF via neuronx-cc."""
        hlo_pb_path = work_dir / "hlo_module.pb"
        proto = ir.to_proto()
        with open(hlo_pb_path, "wb") as f:
            f.write(proto.SerializeToString())

        cmd = self._build_hlo_compile_command(work_dir)

        current_dir = os.getcwd()
        try:
            os.chdir(work_dir)
            if use_neuronx_cc_python_interface:
                original_argv = sys.argv.copy()
                sys.argv = cmd
                neuronx_cc_main()
            else:
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    raise self._compilation_error(
                        f"Compilation failed (exit code {result.returncode}).",
                        cmd, result,
                    )
        finally:
            if use_neuronx_cc_python_interface:
                sys.argv = original_argv
            os.chdir(current_dir)

        output_path = work_dir / output_file
        if not output_path.exists():
            raise self._compilation_error(
                f"Compilation failed: {output_file} expected but not generated.",
                cmd,
                result if not use_neuronx_cc_python_interface else None,
            )
        return output_path

    def _compile_nkigen(self, ir, work_dir: Path, output_file: str) -> Path:
        """Compile a NkiGenIR module to NEFF via nkigen."""
        from nkigen.compile import compile_to_neff

        target_str = self._resolve_target().value

        cc_args = tuple(shlex.split(self.config.additional_args)) if self.config.additional_args else ()

        compile_to_neff(
            ir._mlir_text,
            ir._func_name,
            input_specs=[(s.name, s.shape, s.dtype) for s in ir.inputs],
            output_specs=[(s.name, s.shape, s.dtype) for s in ir.outputs],
            target=target_str,
            output_path=str(work_dir / output_file),
            artifacts_dir=str(work_dir),
            neuronx_cc_args=cc_args,
        )

        output_path = work_dir / output_file
        if not output_path.exists():
            raise self._compilation_error(
                f"NkiGen compilation failed: {output_file} not generated."
            )
        return output_path

    def _compile_nkigen_lite(self, ir, work_dir: Path, output_file: str) -> Path:
        """Compile a NkiGenLiteIR module to NEFF via nkigen_lite lowering + kernel_builder."""
        from nkigen_lite.tensor_ir.passes import lower_to_nki
        from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel
        from nkigen_lite.core import to_np_dtype
        import nki.compiler.kernel_builder as nb
        from nki.compiler.kernel_builder import Tensor

        target_str = self._resolve_target().value

        # Lower tensor_ir → nki_ir (canonicalize/decompose mutate ir._graph)
        nki_graph = lower_to_nki(ir._graph)

        # Update output specs to reflect shape changes from lowering
        # (e.g. scalar () → (1,) for NKI compatibility)
        ir._sync_output_specs_from_nki_graph(nki_graph)

        # Build kernel function from nki_ir
        kernel_fn = build_kb_kernel(nki_graph)

        # Prepare input/output specs for kernel_builder
        input_specs = {}
        for v in nki_graph.inputs:
            kb_dtype = _lite_dtype_to_kb(v.type.dtype)
            input_specs[v.name] = Tensor(v.type.shape, kb_dtype, nb.hbm)

        output_specs = {}
        for name, v in nki_graph.outputs.items():
            kb_dtype = _lite_dtype_to_kb(v.type.dtype)
            output_specs[name] = Tensor(v.type.shape, kb_dtype, nb.hbm)

        # Build the kernel module
        module = nb.build_kernel(
            kernel_fn,
            input_specs=input_specs,
            output_specs=output_specs,
            target=target_str,
        )

        # Compile to NEFF
        cc_args = tuple(shlex.split(self.config.additional_args)) if self.config.additional_args else ()
        neff_path = work_dir / output_file

        from nki.compiler.kernel_builder import CompileOptions
        compile_opts = CompileOptions(
            target=target_str,
            output_path=str(neff_path),
            artifacts_dir=str(work_dir),
            neuronx_cc_args=cc_args,
        )

        # compile_kernel expects numpy input/output arrays for shape/dtype info.
        # The nki_ir graph includes output buffers as graph inputs (suffixed _out).
        # We split them into inputs (user params) and outputs (result buffers).
        import numpy as _np

        # Identify which graph inputs are output buffers (they end with _out
        # and correspond to a graph output name)
        output_names = set(nki_graph.outputs.keys())
        np_inputs = {}
        np_outputs = {}
        for v in nki_graph.inputs:
            # Output buffers are named "<output_name>_out"
            candidate_out_name = v.name[:-4] if v.name.endswith("_out") else None
            if candidate_out_name and candidate_out_name in output_names:
                np_outputs[v.name] = _np.empty(v.type.shape, dtype=to_np_dtype(v.type.dtype))
            else:
                np_inputs[v.name] = _np.empty(v.type.shape, dtype=to_np_dtype(v.type.dtype))

        nb.compile_kernel(
            kernel_fn,
            inputs=np_inputs,
            outputs=np_outputs,
            compile_opts=compile_opts,
        )

        if not neff_path.exists():
            raise self._compilation_error(
                f"NkiGen-Lite compilation failed: {output_file} not generated."
            )
        return neff_path

    def compile(
        self,
        ir,
        work_dir: Path,
        output_file: str,
        use_neuronx_cc_python_interface: bool = False,
    ) -> Path:
        """Compile an IR module to a NEFF file.

        Dispatches to ``_compile_hlo``, ``_compile_nkigen``, or
        ``_compile_nkigen_lite`` based on the IR type.
        """
        if isinstance(ir, HLOModule):
            return self._compile_hlo(
                ir, work_dir, output_file, use_neuronx_cc_python_interface
            )

        from nkipy.core.backend.nkigen import NkiGenIR

        if isinstance(ir, NkiGenIR):
            return self._compile_nkigen(ir, work_dir, output_file)

        from nkipy.core.backend.nkigen_lite import NkiGenLiteIR

        if isinstance(ir, NkiGenLiteIR):
            return self._compile_nkigen_lite(ir, work_dir, output_file)

        raise RuntimeError(
            f"Unknown IR type: {type(ir).__name__}. "
            "Expected HLOModule, NkiGenIR, or NkiGenLiteIR."
        )

    def compile_in_directory(
        self,
        ir,
        output_file: str,
        output_dir: Optional[str] = None,
        save_artifacts: bool = False,
        use_neuronx_cc_python_interface: bool = False,
    ) -> Path:
        """
        Compile in either a temporary directory or the specified output directory

        Args:
            ir: The IR to compile
            output_file: Name of the output file to check for
            output_dir: Directory to save outputs
            save_artifacts: If True, saves all compilation artifacts

        Returns:
            Path to the output file
        """
        if save_artifacts:
            if not output_dir:
                raise ValueError(
                    "output_dir must be specified when save_artifacts is True"
                )
            # Work directly in output directory
            work_dir = Path(output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            return self.compile(
                ir, work_dir, output_file, use_neuronx_cc_python_interface
            )

        # Work in temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            output_path = self.compile(
                ir, work_dir, output_file, use_neuronx_cc_python_interface
            )

            if output_dir:
                # Copy the output file
                dest_path = Path(output_dir) / output_file
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_path, dest_path)
                output_path = dest_path

            return output_path


# Helper function to get IR from traced kernel
def get_ir(traced_kernel):
    assert traced_kernel._code is not None, "Kernel must be traced before compilation"
    return traced_kernel._code


def compile_to_neff(
    trace_kernel,
    output_dir: str,
    neff_name: str = "file.neff",
    target: CompilationTarget = CompilationTarget.DEFAULT,
    additional_compiler_args: str = "",
    save_artifacts: bool = False,
    use_neuronx_cc_python_interface: bool = False,
) -> str:
    """
    Compile traced kernel to NEFF file

    Args:
        trace_kernel: The kernel to compile
        output_dir: Directory to save outputs
        target: Target platform for compilation
        save_artifacts: If True, saves all compilation artifacts

    Returns:
        Path to the generated NEFF file as str
    """
    config = CompilationConfig(
        target=target, additional_args=additional_compiler_args, neff_name=neff_name
    )
    compiler = Compiler(config)
    ir = get_ir(trace_kernel)

    posix_path = compiler.compile_in_directory(
        ir, neff_name, output_dir, save_artifacts, use_neuronx_cc_python_interface
    )

    return str(posix_path)


def lower_to_nki(
    trace_kernel,
    output_dir: Optional[str] = None,
    target: CompilationTarget = CompilationTarget.DEFAULT,
    additional_compiler_args: str = "",
    save_artifacts: bool = False,
) -> str:
    """
    Lower traced kernel to NKI representation

    Args:
        trace_kernel: The kernel to lower
        output_dir: Directory to save outputs
        target: Target platform for compilation
        save_artifacts: If True, saves all compilation artifacts

    Returns:
        The NKI representation as string
    """
    config = CompilationConfig(
        pipeline=("tensorize", "SaveTemps"),
        target=target,
        additional_args='--tensorizer-options="--print-nki" '
        + additional_compiler_args,
    )
    compiler = Compiler(config)

    try:
        is_temp = output_dir is None
        if is_temp:
            output_dir = tempfile.mkdtemp()
        nki_path = compiler.compile_in_directory(
            get_ir(trace_kernel), "nki.py", output_dir, save_artifacts
        )
        return nki_path.read_text()
    finally:
        if is_temp:
            shutil.rmtree(output_dir)
