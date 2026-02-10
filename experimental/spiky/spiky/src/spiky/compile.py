# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct compile API for spiky.

Provides spiky.compile() as a convenience wrapper around
torch.compile(backend="nkipy") with option validation and
automatic backend initialization.
"""

import logging
from typing import Callable

import torch
import torch._dynamo

logger = logging.getLogger(__name__)


def compile(model, *, options=None) -> Callable:
    """Compile a model for Neuron hardware via spiky.

    Convenience wrapper around torch.compile(backend="nkipy"). Handles
    backend initialization, option validation, and wraps execution in
    torch.no_grad() (required to avoid aot_autograd input aliasing that
    triggers neuronx-cc ICE).

    Args:
        model: A PyTorch nn.Module or callable to compile.
        options: Optional dictionary of compilation options passed to
            the nkipy backend. Common options:
            - buckets: List[int] — explicit bucket sizes for dynamic shapes
            - jit: bool — JIT compile unseen buckets (default True)
            - pipelined: bool — pipelined execution (default True)
            - pad_on_device: bool — pad inputs on device (default True)
            - keep_outputs_on_device: bool (default False)
            - input_layout: "auto" | "padded" (default "auto")
            - output_layout: "unpad" | "padded" (default "unpad")

    Returns:
        A callable that compiles and executes the model on Neuron hardware.

    Raises:
        ValueError: If options contain invalid values.
    """
    opts = options or {}

    # Validate buckets early
    if "buckets" in opts:
        buckets = opts["buckets"]
        if not buckets:
            raise ValueError("options['buckets'] cannot be an empty list")
        if any(b <= 0 for b in buckets):
            raise ValueError(
                f"All bucket sizes must be positive, got: {sorted(buckets)}"
            )

    return _SpikyCompiled(model, opts)


class _SpikyCompiled:
    """Traces model once via torch.compile, then executes CompiledWrapper directly."""

    def __init__(self, model, options: dict):
        self._model = model
        self._options = options
        self._wrapper = None  # CompiledWrapper — captured from fw_compiler
        self._frozen_params = None  # model params in aot_module_simplified order

    def _trace(self, first_args):
        """Run torch.compile once to trace. Capture CompiledWrapper + frozen params."""
        import torch._functorch._aot_autograd.runtime_wrappers as rw
        from torch._decomp import core_aten_decompositions
        from torch._functorch._aot_autograd.utils import make_boxed_func
        from torch._functorch.aot_autograd import aot_module_simplified

        from spiky.torch.backend import (
            CompiledWrapper,
            init_nkipy_backend,
            is_nkipy_backend_initialized,
        )

        rw.AliasOfInputHandler.__call__ = rw.NoopAliasHandler.__call__

        if not is_nkipy_backend_initialized():
            init_nkipy_backend()

        captured = {}

        def _capture_backend(gm, example_inputs, **kwargs):
            num_user_inputs = len(example_inputs)

            def _fw_compiler(decomposed_gm, flat_inputs):
                num_params = len(flat_inputs) - num_user_inputs
                captured["params"] = [
                    x.detach().clone() for x in flat_inputs[:num_params]
                ]
                wrapper = CompiledWrapper(decomposed_gm, self._options)
                captured["wrapper"] = wrapper
                return make_boxed_func(wrapper)

            return aot_module_simplified(
                gm,
                example_inputs,
                fw_compiler=_fw_compiler,
                decompositions=core_aten_decompositions(),
                keep_inference_input_mutations=True,
            )

        torch._dynamo.reset()
        compiled = torch.compile(self._model, backend=_capture_backend)
        with torch.no_grad():
            first_output = compiled(*first_args)

        self._wrapper = captured["wrapper"]
        self._frozen_params = captured["params"]
        return first_output

    @torch._dynamo.disable
    def __call__(self, *args, **kwargs):
        if kwargs:
            raise ValueError(
                "spiky.compile() does not support keyword arguments in forward calls"
            )

        if self._wrapper is None:
            return self._trace(args)

        # Direct call — no dynamo, no guard checks
        with torch.no_grad():
            outputs = self._wrapper(*tuple(self._frozen_params) + args)

        if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
            return outputs[0]
        return outputs

    def flush(self) -> None:
        """Flush any pending pipelined execution."""
        if (
            self._wrapper is not None
            and hasattr(self._wrapper, "_callable")
            and self._wrapper._callable is not None
        ):
            self._wrapper._callable.flush()

    def close(self) -> None:
        """Clean up compiled resources."""
        self._wrapper = None
        self._frozen_params = None

    def __del__(self):
        self.close()
