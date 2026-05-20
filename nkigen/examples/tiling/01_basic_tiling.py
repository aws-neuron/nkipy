"""
Example demonstrating how to tile linalg.generic (elementwise) and linalg.matmul
using the transform dialect in Python.

Based on examples/04_transform.py pattern.
"""

from mlir.passmanager import PassManager
from mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from mlir.dialects import func, linalg, tensor, transform
from mlir.dialects.transform import any_op_t
from mlir.dialects.transform.structured import (
    structured_match,
    TileUsingForOp,
)
from mlir.dialects.transform.extras import named_sequence
from mlir.extras import types as T
from mlir.dialects.builtin import module, ModuleOp


def construct_and_print_in_module(f):
    print("\n" + "=" * 80)
    print("TEST:", f.__name__)
    print("=" * 80)
    with Context(), Location.unknown():
        mod = Module.create()
        with InsertionPoint(mod.body):
            f(mod)
        print(mod)
    return f


@construct_and_print_in_module
def test_tile_matmul(module_):
    """Example: Tile linalg.matmul using transform dialect"""

    # Step 1: Create a function with linalg.matmul
    @func.func()
    def matmul_func():
        # Define input tensors: A[256x512], B[512x128] -> C[256x128]
        A = tensor.empty([256, 512], T.f32())
        B = tensor.empty([512, 128], T.f32())
        C = tensor.empty([256, 128], T.f32())

        # Perform matmul: C = A @ B
        result = linalg.matmul(A, B, outs=[C])
        return result

    # Step 2: Create transform module with tiling instructions
    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def transform_mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def main(target):
            # Match the linalg.matmul operation
            matmul_op = structured_match(any_op_t(), target, ops=["linalg.matmul"])

            # Tile matmul with sizes [128, 128, 256] for dimensions [M, N, K]
            # This creates 3 nested scf.for loops
            tiled_ops = TileUsingForOp(matmul_op, sizes=[128, 128, 256])

    print("\n>>> Applying transform-interpreter pass...")
    pm = PassManager.parse("builtin.module(transform-interpreter)")
    pm.run(module_.operation)

    print("\n>>> After tiling:")
    print(module_)


@construct_and_print_in_module
def test_tile_elementwise(module_):
    """Example: Tile linalg.generic (elementwise) using transform dialect"""

    # Step 1: Create function with elementwise operation using linalg.add
    @func.func()
    def add_func():
        # Define input tensors
        A = tensor.empty([512, 256], T.f32())
        B = tensor.empty([512, 256], T.f32())
        C = tensor.empty([512, 256], T.f32())

        # Elementwise add: C = A + B
        # linalg.add is a named op that becomes linalg.generic
        result = linalg.add(A, B, outs=[C])
        return result

    # Step 2: Create transform module
    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def transform_mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def main(target):
            # Match linalg.add (which is a linalg.generic)
            add_op = structured_match(any_op_t(), target, ops=["linalg.add"])

            # Tile with sizes [128, 64] for dimensions [0, 1]
            tiled_ops = TileUsingForOp(add_op, sizes=[128, 64])

    print("\n>>> Applying transform-interpreter pass...")
    pm = PassManager.parse("builtin.module(transform-interpreter)")
    pm.run(module_.operation)

    print("\n>>> After tiling:")
    print(module_)


@construct_and_print_in_module
def test_tile_both(module_):
    """Example: Tile both matmul and elementwise in same function"""

    # Step 1: Create function with both operations
    @func.func()
    def fused_matmul_add():
        # Matrix multiply
        A = tensor.empty([256, 512], T.f32())
        B = tensor.empty([512, 128], T.f32())
        C = tensor.empty([256, 128], T.f32())
        matmul_result = linalg.matmul(A, B, outs=[C])

        # Add bias
        bias = tensor.empty([256, 128], T.f32())
        output = tensor.empty([256, 128], T.f32())
        add_result = linalg.add(matmul_result, bias, outs=[output])

        return add_result

    # Step 2: Transform both operations
    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def transform_mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def main(target):
            # Tile matmul
            matmul_op = structured_match(any_op_t(), target, ops=["linalg.matmul"])
            tiled_matmul = TileUsingForOp(matmul_op, sizes=[128, 128, 128])

            # Tile add
            add_op = structured_match(any_op_t(), target, ops=["linalg.add"])
            tiled_add = TileUsingForOp(add_op, sizes=[64, 64])

    print("\n>>> Applying transform-interpreter pass...")
    pm = PassManager.parse("builtin.module(transform-interpreter)")
    pm.run(module_.operation)

    print("\n>>> After tiling both operations:")
    print(module_)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  Transform Dialect Tiling Examples Complete!")
    print("=" * 80)
    print("""
KEY POINTS:
-----------
1. structured_match: Find operations by name (e.g., "linalg.matmul", "linalg.add")
2. TileUsingForOp: Tile operations into nested scf.for loops
3. sizes parameter: List of tile sizes for each dimension
   - For matmul: [M_tile, N_tile, K_tile]
   - For elementwise: [dim0_tile, dim1_tile, ...]
4. Transform interpreter pass: Applies transforms to generate tiled code
5. Tiling creates:
   - Nested scf.for loops for iteration
   - tensor.extract_slice for input slicing
   - tensor.insert_slice for output assembly
""")
