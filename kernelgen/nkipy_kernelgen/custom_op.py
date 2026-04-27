"""
CustomOp: Wrap kernel_builder NISA functions for use in @trace-decorated kernels.

A CustomOp represents a pre-compiled NISA function (from kernel_builder) that can
be called during tracing. It emits a func.call in the traced IR, and the NISA body
is stashed as a string attribute for late resolution by the resolve-custom-ops pass.
"""

from typing import Optional, Callable, List, Tuple
from mlir import ir
from mlir.dialects import func

from .traced_array import TracedArray
from .mlir_utils import to_mlir_type, ranked_tensor_of

# Module-level registry for custom ops used during tracing.
# No thread safety needed -- tracing is always single-threaded.
_custom_op_registry: list = []


def _get_registry() -> list:
    return _custom_op_registry


def _clear_registry():
    _custom_op_registry.clear()


def _nb_dtype_to_str(nb_dtype) -> str:
    """Convert kernel_builder dtype to string (e.g., nb.float32 -> 'f32')."""
    import nki.compiler.kernel_builder as nb
    mapping = {
        nb.float32: "f32",
        nb.float16: "f16",
        nb.bfloat16: "bf16",
    }
    if nb_dtype in mapping:
        return mapping[nb_dtype]
    raise ValueError(f"Unsupported kernel_builder dtype: {nb_dtype}")


class CustomOp:
    """A kernel_builder function compiled to NISA, usable during KernelGen tracing.

    Wraps the output of nb.build_kernel() and provides a callable interface
    that emits func.call during tracing. Falls back to reference_fn for
    NumPy execution (testing).
    """

    def __init__(
        self,
        nisa_mlir: str,
        func_name: str,
        input_names: List[str],
        output_names: List[str],
        input_shapes: List[Tuple[int, ...]],
        output_shapes: List[Tuple[int, ...]],
        input_dtypes: List[str],
        output_dtypes: List[str],
        reference_fn: Optional[Callable] = None,
    ):
        self.nisa_mlir = nisa_mlir
        self.func_name = f"__custom_op__{func_name}"
        self.input_names = input_names
        self.output_names = output_names
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.input_dtypes = input_dtypes
        self.output_dtypes = output_dtypes
        self.reference_fn = reference_fn

    @classmethod
    def from_kernel_builder(
        cls,
        kernel_func: Callable,
        input_specs: dict,
        output_specs: dict,
        reference_fn: Optional[Callable] = None,
        **hyperparams,
    ) -> "CustomOp":
        """Compile a kernel_builder function to a CustomOp.

        Args:
            kernel_func: Function using nki.compiler.kernel_builder APIs.
            input_specs: Dict of name -> nb.Tensor specs for inputs.
            output_specs: Dict of name -> nb.Tensor specs for outputs.
            reference_fn: NumPy reference implementation for testing.
            **hyperparams: Compile-time constants passed to kernel_func.

        Returns:
            CustomOp ready to use inside @trace-decorated functions.
        """
        import nki.compiler.kernel_builder as nb

        module = nb.build_kernel(
            kernel_func,
            input_specs=input_specs,
            output_specs=output_specs,
            **hyperparams,
        )
        # Use generic MLIR form -- NISA custom assembly isn't round-trippable
        nisa_mlir = module.operation.get_asm(print_generic_op_form=True)

        input_names = list(input_specs.keys())
        output_names = list(output_specs.keys())
        input_shapes = [spec.shape for spec in input_specs.values()]
        output_shapes = [spec.shape for spec in output_specs.values()]
        input_dtypes = [_nb_dtype_to_str(spec.dtype) for spec in input_specs.values()]
        output_dtypes = [_nb_dtype_to_str(spec.dtype) for spec in output_specs.values()]

        # Include shape signature and hyperparams in func_name to deduplicate
        # same function compiled with different shapes or hyperparams
        shape_sig = "_".join(
            "x".join(str(d) for d in s)
            for s in list(input_shapes) + list(output_shapes)
        )
        if hyperparams:
            import hashlib
            hp_hash = hashlib.md5(
                repr(sorted(hyperparams.items())).encode()
            ).hexdigest()[:8]
            func_name = f"{kernel_func.__name__}_{shape_sig}_{hp_hash}"
        else:
            func_name = f"{kernel_func.__name__}_{shape_sig}"

        return cls(
            nisa_mlir=nisa_mlir,
            func_name=func_name,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            reference_fn=reference_fn,
        )

    def __call__(self, *args):
        """Call during tracing or NumPy execution.

        During tracing: emits func.call and returns TracedArray(s).
        Outside tracing: calls reference_fn with numpy inputs.
        """
        # Not tracing -- use NumPy reference
        if not any(isinstance(a, TracedArray) for a in args):
            if self.reference_fn is None:
                raise RuntimeError(
                    f"CustomOp '{self.func_name}' called outside tracing "
                    f"without a reference_fn."
                )
            return self.reference_fn(*args)

        # --- Tracing mode ---
        if len(args) != len(self.input_shapes):
            raise ValueError(
                f"CustomOp '{self.func_name}' expects {len(self.input_shapes)} "
                f"inputs, got {len(args)}."
            )

        for i, (arg, expected_shape) in enumerate(zip(args, self.input_shapes)):
            if not isinstance(arg, TracedArray):
                raise TypeError(
                    f"Argument {i} ('{self.input_names[i]}') must be TracedArray, "
                    f"got {type(arg)}"
                )
            if tuple(arg.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Shape mismatch for '{self.input_names[i]}': "
                    f"expected {expected_shape}, got {arg.shape}"
                )

        # Register for module-level emission (deduplicate by func_name)
        registry = _get_registry()
        if not any(op.func_name == self.func_name for op in registry):
            registry.append(self)

        loc = args[0]._get_caller_location()
        input_values = [a.value for a in args]

        # Build result types: return-value style (tensor types)
        result_types = [
            ranked_tensor_of(shape, to_mlir_type(dtype))
            for shape, dtype in zip(self.output_shapes, self.output_dtypes)
        ]

        # Emit: %result = func.call @name(%in0, %in1) -> tensor<...>
        call_op = func.CallOp(result_types, self.func_name, input_values, loc=loc)

        # Return output TracedArray(s)
        if len(self.output_shapes) == 1:
            return TracedArray(
                call_op.results[0],
                self.output_shapes[0],
                to_mlir_type(self.output_dtypes[0]),
                source_file=args[0].source_file,
            )
        return tuple(
            TracedArray(
                call_op.results[i], shape,
                to_mlir_type(dtype),
                source_file=args[0].source_file,
            )
            for i, (shape, dtype) in enumerate(
                zip(self.output_shapes, self.output_dtypes)
            )
        )


def emit_custom_op_declaration(custom: CustomOp):
    """Emit func.func private @name(tensor<...>) -> tensor<...>
    attributes {nkipy.custom_op}

    Return-value style: inputs only as arguments, outputs as return values.
    resolve-custom-ops will later convert to output-as-argument.
    """
    input_types = [
        ranked_tensor_of(s, to_mlir_type(d))
        for s, d in zip(custom.input_shapes, custom.input_dtypes)
    ]
    result_types = [
        ranked_tensor_of(s, to_mlir_type(d))
        for s, d in zip(custom.output_shapes, custom.output_dtypes)
    ]
    fn_type = ir.FunctionType.get(input_types, result_types)
    fn = func.FuncOp(name=custom.func_name, type=fn_type)
    fn.attributes["sym_visibility"] = ir.StringAttr.get("private")
    fn.attributes["nkipy.custom_op"] = ir.UnitAttr.get()
    return fn
