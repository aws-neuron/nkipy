"""
Example demonstrating the knob() builder API.

knob(t).layout(...) and knob(t).tile_op(...) inject nkipy.layout and
nkipy.tile_op ops into the IR — side-effecting, not returning the tensor.
"""

import numpy as np
from nkigen import trace
from nkigen.apis import knob
from nkigen.transforms import apply_passes


# Example 1: Only partition_dim - emits a nkipy.layout with partition_dim
@trace(input_specs=[((4, 4), "f32"), ((4, 4), "f32")])
def partition_only(A, B):
    temp = np.add(A, B)
    knob(temp).layout(partition_dim=0)
    result = np.multiply(temp, 2.0)
    return result


# Example 2: Only mem_space - emits a nkipy.layout with mem_space
@trace(input_specs=[((64, 64), "f32"), ((64, 64), "f32")])
def mem_space_only(A, B):
    C = np.matmul(A, B)
    knob(C).layout(mem_space="Hbm")
    return C


# Example 3: Both partition_dim and mem_space, with multiple knob annotations
@trace(input_specs=[((8, 8), "f32"), ((8, 8), "f32")])
def both_params(A, B):
    temp0 = np.add(A, B)
    temp1 = np.multiply(temp0, 2.0)
    temp2 = np.square(temp1)
    # temp2 = knob(temp2).layout(partition_dim=0, mem_space="Sbuf")
    return temp2


if __name__ == "__main__":
    print("=" * 80)
    print("Example 1: Only partition_dim (injects nkipy.annotate with partition_dim)")
    print("=" * 80)
    print(partition_only.to_mlir())
    print()
    
    print("=" * 80)
    print("Example 2: Only mem_space (injects nkipy.annotate with mem_space)")
    print("=" * 80)
    print(mem_space_only.to_mlir())
    print()
    
    print("=" * 80)
    print("Example 3: Both partition_dim and mem_space (before fusion)")
    print("=" * 80)
    mlir_module = both_params.to_mlir()
    print(mlir_module)
    print()
    
    print("=" * 80)
    print("Example 3: After applying linalg-generalize-named-ops + fuse-elementwise-ops")
    print("=" * 80)
    # Apply passes using a list of pass names (recommended approach)
    passes = ["linalg-generalize-named-ops", "linalg-fuse-elementwise-ops"]
    transformed_module = apply_passes(both_params.to_mlir(), passes)
    print(transformed_module)
    print()
    
    print("=" * 80)
    print("Example 4: Complete pipeline with bufferization, loop conversion, and NISA")
    print("=" * 80)

    complete_pipeline = [
        "linalg-generalize-named-ops",
        "linalg-fuse-elementwise-ops",
        "one-shot-bufferize{bufferize-function-boundaries}",
        "convert-linalg-to-loops",
    ]
    fully_transformed = apply_passes(both_params.to_mlir(), complete_pipeline)
    print(fully_transformed)
    print()
