import numpy as np
from nkipy.core.trace import NKIPyKernel
from nkipy.runtime.execute import baremetal_run_traced_kernel, simulate_traced_kernel

print("=" * 80)
print("EINSUM OPERATION TESTS")
print("=" * 80)


def run_test(test_func, *test_args):
    """Helper to trace, simulate, and run on baremetal."""
    # Run numpy version to get expected output
    expected = test_func(*test_args)
    print(f"Input shapes: {[a.shape for a in test_args if hasattr(a, 'shape')]}")
    if hasattr(expected, "shape"):
        print(f"Output shape: {expected.shape}")
    else:
        print(f"Output: {expected}")

    traced_kernel = NKIPyKernel.trace(test_func)

    # Simulation
    out_nkipy = simulate_traced_kernel(traced_kernel, *test_args)
    sim_match = np.allclose(out_nkipy, expected)
    print(f"Simulation matches NumPy? {sim_match}")

    # Baremetal
    try:
        out_baremetal = baremetal_run_traced_kernel(traced_kernel, *test_args)
        bm_match = np.allclose(out_baremetal, expected)
        print(f"Baremetal matches NumPy? {bm_match}")
    except Exception as e:
        print(f"Baremetal test skipped/failed: {type(e).__name__} - {e}")


# =============================================================================
# 1. Matrix Multiplication
# =============================================================================
print("\n1. Matrix Multiplication (ik,kj->ij)")
print("-" * 80)


def einsum_matmul(A, B):
    """Standard matrix multiply: (i, k) x (k, j) -> (i, j)"""
    return np.einsum("ik,kj->ij", A, B)


A = np.random.rand(2, 3).astype(np.float32)
B = np.random.rand(3, 4).astype(np.float32)
run_test(einsum_matmul, A, B)


# =============================================================================
# 2. Batch Matrix Multiplication
# =============================================================================
print("\n2. Batch Matrix Multiplication (bik,bkj->bij)")
print("-" * 80)


def einsum_batch_matmul(A, B):
    """Batch matrix multiply: (batch, i, k) x (batch, k, j) -> (batch, i, j)"""
    return np.einsum("bik,bkj->bij", A, B)


A = np.random.rand(5, 2, 3).astype(np.float32)
B = np.random.rand(5, 3, 4).astype(np.float32)
run_test(einsum_batch_matmul, A, B)


# =============================================================================
# 3. Dot Product (Inner Product)
# =============================================================================
print("\n3. Dot Product (i,i->)")
print("-" * 80)


def einsum_dot(a, b):
    """Dot product of two vectors: sum(a * b)"""
    return np.einsum("i,i->", a, b)


a = np.array([1, 2, 3], dtype=np.float32)
b = np.array([4, 5, 6], dtype=np.float32)
run_test(einsum_dot, a, b)


# =============================================================================
# 4. Outer Product
# =============================================================================
print("\n4. Outer Product (i,j->ij)")
print("-" * 80)


def einsum_outer(a, b):
    """Outer product: (i,) x (j,) -> (i, j)"""
    return np.einsum("i,j->ij", a, b)


a = np.array([1, 2, 3], dtype=np.float32)
b = np.array([4, 5], dtype=np.float32)
run_test(einsum_outer, a, b)


# =============================================================================
# 5. Element-wise Multiply and Sum (Frobenius inner product)
# =============================================================================
print("\n5. Element-wise Multiply and Sum (ij,ij->)")
print("-" * 80)


def einsum_hadamard_sum(A, B):
    """Element-wise multiply then sum all: sum(A * B)"""
    return np.einsum("ij,ij->", A, B)


A = np.array([[1, 2], [3, 4]], dtype=np.float32)
B = np.array([[5, 6], [7, 8]], dtype=np.float32)
run_test(einsum_hadamard_sum, A, B)


# =============================================================================
# 6. Transpose
# =============================================================================
print("\n6. Transpose (ij->ji)")
print("-" * 80)


def einsum_transpose(A):
    """Matrix transpose: (i, j) -> (j, i)"""
    return np.einsum("ij->ji", A)


A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
run_test(einsum_transpose, A)


# =============================================================================
# 7. Sum Along Axis
# =============================================================================
print("\n7. Sum Along Axis (ij->i)")
print("-" * 80)


def einsum_sum_axis(A):
    """Sum along last axis: (i, j) -> (i,)"""
    return np.einsum("ij->i", A)


A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
run_test(einsum_sum_axis, A)


# =============================================================================
# 8. Batched Dot Product
# =============================================================================
print("\n8. Batched Dot Product (bi,bi->b)")
print("-" * 80)


def einsum_batch_dot(A, B):
    """Dot product for each pair in batch: (batch, i) x (batch, i) -> (batch,)"""
    return np.einsum("bi,bi->b", A, B)


A = np.random.rand(5, 10).astype(np.float32)
B = np.random.rand(5, 10).astype(np.float32)
run_test(einsum_batch_dot, A, B)


# =============================================================================
# 9. Tensor Contraction
# =============================================================================
print("\n9. Tensor Contraction (ijk,jkl->il)")
print("-" * 80)


def einsum_tensor_contract(A, B):
    """Contract on middle dimensions: (i,j,k) x (j,k,l) -> (i,l)"""
    return np.einsum("ijk,jkl->il", A, B)


A = np.random.rand(2, 3, 4).astype(np.float32)
B = np.random.rand(3, 4, 5).astype(np.float32)
run_test(einsum_tensor_contract, A, B)


print("\n" + "=" * 80)
print("TESTS COMPLETE")
print("=" * 80)
