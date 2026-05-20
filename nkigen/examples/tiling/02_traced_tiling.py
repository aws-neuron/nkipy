"""
Example demonstrating how to apply tiling transformations to traced NumPy functions.

This example shows how to:
1. Trace a NumPy function to MLIR using @trace decorator
2. Apply transform dialect tiling passes to the traced IR
3. See the tiled result with nested scf.for loops

Based on examples/01_basic_usage.py and examples/tiling/01_basic_tiling.py
"""

import numpy as np
from nkigen import trace

from mlir.passmanager import PassManager
from mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from mlir.dialects import func, transform
from mlir.dialects.transform import any_op_t
from mlir.dialects.transform.structured import (
    structured_match,
    TileUsingForOp,
)
from mlir.dialects.transform.extras import named_sequence
from mlir.dialects.builtin import module


def apply_tiling_to_traced_function(traced_func, tile_specs):
    """
    Apply tiling transformations to a traced function.

    Args:
        traced_func: A traced function (decorated with @trace)
        tile_specs: List of (op_name, tile_sizes) tuples
                   e.g., [("linalg.matmul", [128, 128, 128]),
                          ("linalg.add", [64, 64])]

    Returns:
        Tiled MLIR module as string
    """
    with Context() as ctx, Location.unknown():
        # Get the traced function's MLIR as string and parse it
        mlir_module = traced_func.to_mlir()
        mlir_str = str(mlir_module)
        module_op = Module.parse(mlir_str, ctx)

        # Add transform module with tiling instructions
        with InsertionPoint(module_op.body):
            @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
            def transform_mod():
                @named_sequence("__transform_main", [any_op_t()], [])
                def main(target):
                    # Apply tiling for each specified operation
                    for op_name, tile_sizes in tile_specs:
                        matched_op = structured_match(
                            any_op_t(), target, ops=[op_name]
                        )
                        TileUsingForOp(matched_op, sizes=tile_sizes)

        # Apply transform-interpreter pass
        print("\n>>> Applying transform-interpreter pass...")
        pm = PassManager.parse("builtin.module(transform-interpreter)")
        pm.run(module_op.operation)

        return str(module_op)


def example_matmul_tiling():
    """Example 1: Tile a simple matmul operation"""
    print("\n" + "=" * 80)
    print("Example 1: Tiling Matrix Multiplication")
    print("=" * 80)

    @trace(input_specs=[((256, 512), "f32"), ((512, 128), "f32")])
    def simple_matmul(A, B):
        """Matrix multiply: C = A @ B"""
        return np.matmul(A, B)

    print("\n>>> Original traced MLIR:")
    print(simple_matmul.to_mlir())

    # Tile matmul into 128x128x256 tiles
    tiled_mlir = apply_tiling_to_traced_function(
        simple_matmul,
        [("linalg.matmul", [128, 128, 256])]
    )

    print("\n>>> After tiling with [128, 128, 256]:")
    print(tiled_mlir)


def example_elementwise_tiling():
    """Example 2: Tile element-wise operations"""
    print("\n" + "=" * 80)
    print("Example 2: Tiling Element-wise Addition")
    print("=" * 80)

    @trace(input_specs=[((512, 256), "f32"), ((512, 256), "f32")])
    def elementwise_add(A, B):
        """Element-wise addition: C = A + B"""
        return np.add(A, B)

    print("\n>>> Original traced MLIR:")
    print(elementwise_add.to_mlir())

    # Tile add into 128x64 tiles
    tiled_mlir = apply_tiling_to_traced_function(
        elementwise_add,
        [("linalg.add", [128, 64])]
    )

    print("\n>>> After tiling with [128, 64]:")
    print(tiled_mlir)


def example_fused_tiling():
    """Example 3: Tile fused matmul + add operations"""
    print("\n" + "=" * 80)
    print("Example 3: Tiling Fused Matmul + Add")
    print("=" * 80)

    @trace(input_specs=[((256, 512), "f32"), ((512, 128), "f32")])
    def matmul_add_bias(A, B):
        """Matrix multiply with bias add: C = (A @ B) + (A @ B)"""
        C = np.matmul(A, B)
        return np.add(C, C)

    print("\n>>> Original traced MLIR:")
    print(matmul_add_bias.to_mlir())

    # Tile both operations
    tiled_mlir = apply_tiling_to_traced_function(
        matmul_add_bias,
        [
            ("linalg.matmul", [128, 128, 128]),
            ("linalg.add", [64, 64])
        ]
    )

    print("\n>>> After tiling matmul[128,128,128] and add[64,64]:")
    print(tiled_mlir)


def example_complex_tiling():
    """Example 4: Tile a more complex computation"""
    print("\n" + "=" * 80)
    print("Example 4: Tiling Complex Computation")
    print("=" * 80)

    @trace(input_specs=[((128, 256), "f32"), ((256, 128), "f32"), ((128, 128), "f32")])
    def complex_ops(A, B, C):
        """Complex operation: result = (A @ B) * C"""
        matmul_result = np.matmul(A, B)
        return np.multiply(matmul_result, C)

    print("\n>>> Original traced MLIR:")
    print(complex_ops.to_mlir())

    # Tile all operations
    tiled_mlir = apply_tiling_to_traced_function(
        complex_ops,
        [
            ("linalg.matmul", [64, 64, 128]),
            ("linalg.mul", [64, 64])
        ]
    )

    print("\n>>> After tiling matmul[64,64,128] and mul[64,64]:")
    print(tiled_mlir)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  Transform Dialect Tiling on Traced Programs")
    print("=" * 80)

    # Run all examples
    example_matmul_tiling()
    example_elementwise_tiling()
    example_fused_tiling()
    example_complex_tiling()

    print("\n" + "=" * 80)
    print("  All Examples Complete!")
    print("=" * 80)
    print("""
KEY POINTS:
-----------
1. @trace decorator: Converts NumPy functions to MLIR
2. to_mlir(): Gets MLIR representation from traced function
3. Transform dialect: Applies tiling transformations to traced IR
4. structured_match: Finds operations by name in the IR
5. TileUsingForOp: Creates tiled loops with specified tile sizes
6. transform-interpreter pass: Executes the transform dialect operations

WORKFLOW:
---------
NumPy function → @trace → MLIR (linalg ops) → Transform dialect → Tiled MLIR (scf.for loops)

This approach combines the convenience of NumPy tracing with powerful
MLIR transform dialect capabilities for hardware-specific optimizations.
""")
