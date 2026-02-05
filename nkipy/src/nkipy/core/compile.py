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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from neuronxcc.driver.CommandDriver import main as neuronx_cc_main

from nkipy.core.backend.hlo import HLOModule
from nkipy.core.trace import NKIPyKernel

trace = NKIPyKernel.trace

# Build directory for compiled kernels
_DEFAULT_BUILD_DIR = "/tmp/build"


def _get_build_dir():
    return _DEFAULT_BUILD_DIR


def _set_build_dir(build_dir):
    global _DEFAULT_BUILD_DIR
    _DEFAULT_BUILD_DIR = build_dir


@dataclass
class CompilerConfig:
    """Structured compiler configuration for neuronx-cc.

    Provides type-safe configuration with sensible defaults for NKIPy and NKI kernels.
    Use factory methods `for_nkipy()` and `for_nki()` for preset configurations.

    See neuronx-cc documentation for full option details:
    https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html

    Example:
        # Use preset with overrides
        config = CompilerConfig.for_nkipy(model_type="transformer")

        # Or create custom config
        config = CompilerConfig(lnc=2, model_type="transformer")

        # View the resulting args
        print(config.to_args())
    """

    # Core options
    lnc: int = 1  # Logical NeuronCore config (1 or 2)
    model_type: Optional[str] = None  # "generic", "transformer", "unet-inference"

    # Precision options
    auto_cast: Optional[str] = None  # "none", "matmult", "all"
    auto_cast_type: Optional[str] = None  # "bf16", "fp16", "tf32", "fp8_e4m3"
    enable_mixed_precision_accumulation: Optional[bool] = None  # None = default
    enable_saturate_infinity: bool = False

    # Performance options
    optlevel: Optional[int] = None  # 1 (fast), 2 (balanced), 3 (maximum)
    enable_fast_context_switch: bool = False
    enable_fast_loading_neuron_binaries: bool = False

    # Arbitrary extra arguments (for options not covered above)
    extra_args: List[str] = field(default_factory=list)

    def to_args(self) -> str:
        """Convert configuration to compiler argument string."""
        args = []

        # Core options
        args.append(f"--lnc {self.lnc}")
        if self.model_type:
            args.append(f"--model-type {self.model_type}")

        # Precision options
        if self.auto_cast:
            args.append(f"--auto-cast {self.auto_cast}")
        if self.auto_cast_type:
            args.append(f"--auto-cast-type {self.auto_cast_type}")
        if self.enable_mixed_precision_accumulation is True:
            args.append("--enable-mixed-precision-accumulation")
        elif self.enable_mixed_precision_accumulation is False:
            args.append("--disable-mixed-precision-accumulation")
        if self.enable_saturate_infinity:
            args.append("--enable-saturate-infinity")

        # Performance options
        if self.optlevel is not None:
            args.append(f"-O{self.optlevel}")
        if self.enable_fast_context_switch:
            args.append("--enable-fast-context-switch")
        if self.enable_fast_loading_neuron_binaries:
            args.append("--enable-fast-loading-neuron-binaries")

        # Extra args
        args.extend(self.extra_args)

        return " ".join(args)

    @classmethod
    def for_nkipy(
        cls,
        lnc: int = 1,
        model_type: Optional[str] = None,
        auto_cast: Optional[str] = None,
        auto_cast_type: Optional[str] = None,
        enable_mixed_precision_accumulation: Optional[bool] = None,
        enable_saturate_infinity: bool = False,
        optlevel: Optional[int] = None,
        enable_fast_context_switch: bool = False,
        enable_fast_loading_neuron_binaries: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> "CompilerConfig":
        """Create configuration preset for NKIPy kernels.

        Default settings:
        - lnc=1
        """
        return cls(
            lnc=lnc,
            model_type=model_type,
            auto_cast=auto_cast,
            auto_cast_type=auto_cast_type,
            enable_mixed_precision_accumulation=enable_mixed_precision_accumulation,
            enable_saturate_infinity=enable_saturate_infinity,
            optlevel=optlevel,
            enable_fast_context_switch=enable_fast_context_switch,
            enable_fast_loading_neuron_binaries=enable_fast_loading_neuron_binaries,
            extra_args=extra_args or [],
        )

    @classmethod
    def for_nki(
        cls,
        lnc: int = 1,
        model_type: Optional[str] = None,
        auto_cast: Optional[str] = None,
        auto_cast_type: Optional[str] = None,
        enable_mixed_precision_accumulation: Optional[bool] = None,
        enable_saturate_infinity: bool = False,
        optlevel: Optional[int] = None,
        enable_fast_context_switch: bool = False,
        enable_fast_loading_neuron_binaries: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> "CompilerConfig":
        """Create configuration preset for NKI kernels.

        Default settings:
        - lnc=1
        """
        return cls(
            lnc=lnc,
            model_type=model_type,
            auto_cast=auto_cast,
            auto_cast_type=auto_cast_type,
            enable_mixed_precision_accumulation=enable_mixed_precision_accumulation,
            enable_saturate_infinity=enable_saturate_infinity,
            optlevel=optlevel,
            enable_fast_context_switch=enable_fast_context_switch,
            enable_fast_loading_neuron_binaries=enable_fast_loading_neuron_binaries,
            extra_args=extra_args or [],
        )


def get_default_compiler_args() -> str:
    """Return the default compiler arguments string for NKIPy kernels.

    Useful for debugging and understanding what args will be passed to neuronx-cc.

    Returns:
        The default compiler arguments as a string.
    """
    return CompilerConfig.for_nkipy().to_args()


# Legacy compatibility - computed from CompilerConfig
nkipy_compiler_args = CompilerConfig.for_nkipy().to_args()
nki_compiler_args = CompilerConfig.for_nki().to_args()


class CompilationTarget(Enum):
    TRN1 = "trn1"
    TRN2 = "trn2"
    DEFAULT = "default"


def get_platform_target() -> str:
    _TRN_1 = "trn1"
    _TRN_2 = "trn2"

    fpath = "/sys/devices/virtual/dmi/id/product_name"
    try:
        with open(fpath, "r") as f:
            fc = f.readline()
    except IOError:
        raise RuntimeError(
            'Unable to read platform target. If running on CPU, please supply \
        compiler argument target, with one of options trn1, trn1n, or trn2. Ex: \
        "--target trn1"'
        )

    instance_type = fc.split(".")[0]
    if _TRN_1 in instance_type:
        return CompilationTarget.TRN1
    elif _TRN_2 in instance_type:
        return CompilationTarget.TRN2
    else:
        raise RuntimeError(
            f'Unsupported Platform - {fc}. If you want to compile on CPU, \
        please supply compiler argument target, with one of options trn1, \
        trn1n, or trn2. Ex: "--target trn1"'
        )


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

    def _build_compile_command(self, mode="hlo") -> List[str]:
        cmd = [
            "neuronx-cc",
            "compile",
            "--framework",
            "XLA",
        ]
        if mode == "hlo":
            cmd.extend(["hlo_module.pb"])
        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        cmd.append("--pipeline")
        cmd.extend(self.config.pipeline)

        # When using default target, detect platform target
        if self.config.target == CompilationTarget.DEFAULT:
            try:
                self.config.target = get_platform_target()
            except Exception:
                logging.warning(
                    "Failed to detect platform target, falling back to trn1..."
                )
                self.config.target = CompilationTarget.TRN1

        cmd.extend(
            ["--target", self.config.target.value, f"--output={self.config.neff_name}"]
        )

        if self.config.additional_args:
            cmd.extend(shlex.split(self.config.additional_args))

        return cmd

    def compile(
        self,
        ir,
        work_dir: Path,
        output_file: str,
        use_neuronx_cc_python_interface: bool = False,
    ) -> Path:
        """
        Run compilation in specified directory

        Args:
            ir: The IR to compile
            work_dir: Directory to compile in
            output_file: Name of the output file to check for ("file.neff" or "nki.py")

        Returns:
            Path to the output file
        """

        mode = "hlo" if isinstance(ir, HLOModule) else "unknown"
        cmd = self._build_compile_command(mode)

        current_dir = os.getcwd()
        try:
            os.chdir(work_dir)
            if mode == "hlo":
                hlo_pb_path = "hlo_module.pb"
                proto = ir.to_proto()
                with open(hlo_pb_path, "wb") as f:
                    f.write(proto.SerializeToString())
            else:
                raise RuntimeError(
                    f"Unknown mode: {mode}. "
                    "Note: For NKI kernels, You can either embed a NKI kernel as an op"
                    " in NKIPy kernel or implement your own helper function to get the"
                    " NEFF from a NKI kernel."
                )
            if use_neuronx_cc_python_interface:
                original_argv = sys.argv.copy()
                sys.argv = cmd
                neuronx_cc_main()
            else:
                subprocess.run(cmd, check=True, capture_output=True)
        finally:
            if use_neuronx_cc_python_interface:
                sys.argv = original_argv
            os.chdir(current_dir)

        output_path = work_dir / output_file
        if not output_path.exists():
            raise RuntimeError(
                f"Compilation failed: {output_file} expected but not generated"
            )

        return output_path

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
