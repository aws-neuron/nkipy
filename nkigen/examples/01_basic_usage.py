"""
Example usage of the @trace decorator for tracing NumPy functions to MLIR.
"""

import numpy as np
from nkigen import trace


# Example 1: Simple matmul with addition
@trace(input_specs=[((4, 3), "f32"), ((3, 5), "f32")])
def matmul_add(A, B):
    """Matrix multiply and add: returns A @ B + A @ B"""
    C = np.matmul(A, B)
    return C + C


# Example 2: Element-wise operations
@trace(input_specs=[((10,), "f32"), ((10,), "f32")])
def elementwise_ops(x, y):
    """Perform element-wise add and multiply."""
    temp = np.add(x, y)
    result = np.multiply(temp, 2.0)
    return result


# Example 3: Complex computation with non-ufunc apis
@trace(input_specs=[((4, 4), "f32"), ((4, 4), "f32"), ((4, 4), "f32")])
def complex_computation(a, b, c):
    """Perform a series of operations."""
    temp1 = np.add(a, b)
    temp2 = np.multiply(temp1, c)
    temp3 = np.ones(temp2.shape)
    result = np.subtract(temp2, temp3)
    return result



if __name__ == "__main__":
    print("=" * 60)
    print("NKIPyKernelGen - MLIR Generation Examples")
    print("=" * 60)
    
    # Example 1: Matmul with addition
    print("\n1. Matrix Multiplication with Addition:")
    print("   Function: C = A @ B + A @ B")
    print("   Input shapes: A(4,3), B(3,5)")
    print("\n   Generated MLIR:")
    print("-" * 60)
    module1 = matmul_add.to_mlir()
    print(module1)
    print("-" * 60)
    
    # Example 2: Element-wise operations
    print("\n2. Element-wise Operations:")
    print("   Function: result = (x + y) * 2.0")
    print("   Input shapes: x(10,), y(10,)")
    print("\n   Generated MLIR:")
    print("-" * 60)
    module2 = elementwise_ops.to_mlir()
    print(module2)
    print("-" * 60)
    
    # Example 3: Complex computation
    print("\n3. Complex Computation:")
    print("   Function: result = (a + b) * c - a")
    print("   Input shapes: a(4,4), b(4,4), c(4,4)")
    print("\n   Generated MLIR:")
    print("-" * 60)
    module3 = complex_computation.to_mlir()
    print(module3)
    print("-" * 60)
    
    print("\n" + "=" * 60)
    print("All MLIR modules generated successfully!")
    print("=" * 60)
    
