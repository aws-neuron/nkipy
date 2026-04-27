# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile MLIR to NEFF via the nkipy-opt pass pipeline and nki.compiler.

This module encapsulates all NKI compiler internals so that callers only
need to provide MLIR text, tensor metadata, and a target string.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compile_to_neff(
    mlir_text: str,
    func_name: str,
    input_specs: List[Tuple[str, Tuple[int, ...], np.dtype]],
    output_specs: List[Tuple[str, Tuple[int, ...], np.dtype]],
    *,
    target: str,
    output_path: str,
    artifacts_dir: str | None = None,
    neuronx_cc_args: Tuple[str, ...] = (),
) -> None:
    """Compile MLIR text to a NEFF file.

    Two-stage pipeline:
    1. Run the nkipy-opt MLIR pass pipeline to produce NISA IR.
    2. Compile NISA IR to NEFF via ``nki.compiler``.

    Args:
        mlir_text: MLIR module text (linalg-on-tensor level).
        func_name: Entry-point function name in the MLIR module.
        input_specs: ``[(name, shape, dtype), ...]`` for each input tensor.
        output_specs: ``[(name, shape, dtype), ...]`` for each output tensor.
        target: Hardware target string (e.g. ``"trn2"``).
        output_path: Where to write the NEFF file.
        artifacts_dir: Optional directory for intermediate compilation artifacts.
        neuronx_cc_args: Extra arguments forwarded to ``neuronx-cc``.
    """
    from nkipy_kernelgen.transforms.nkipy_opt import apply_complete_knob_pipeline

    dump_dir = f"{artifacts_dir}/mlir_passes" if artifacts_dir else None
    nisa_mlir = apply_complete_knob_pipeline(
        mlir_text,
        target=target,
        dump_dir=dump_dir,
    )

    from nki.compiler._internal import ir as nki_ir
    from nki.compiler._internal import register_all_dialects
    from nki.compiler.ncc_driver import CompileOptions, compile_mlir_to_neff

    nki_ctx = nki_ir.Context()
    register_all_dialects(nki_ctx)

    with nki_ctx:
        mlir_module = nki_ir.Module.parse(nisa_mlir, nki_ctx)

        all_arrays = [
            np.zeros(shape, dtype=dtype) for _, shape, dtype in input_specs
        ] + [
            np.zeros(shape, dtype=dtype) for _, shape, dtype in output_specs
        ]

        argument_names = [
            name for name, _, _ in input_specs
        ] + [
            name for name, _, _ in output_specs
        ]
        output_arg_names = [name for name, _, _ in output_specs]

        opts = CompileOptions(
            target=target,
            verbose=False,
            output_path=output_path,
            neuronx_cc_args=neuronx_cc_args,
            artifacts_dir=artifacts_dir,
        )

        compile_mlir_to_neff(
            mlir_module,
            func_name,
            all_arrays,
            argument_names,
            output_arg_names,
            opts,
        )
