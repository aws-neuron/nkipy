import numpy as np
from nkigen import trace, verify_against_numpy
from nkigen.apis import fori_loop

# Example 1: Unrolled tile loops
@trace(input_specs=[((4, 8), "f16"), ((8, 5), "f16")])
def tiled_rmsnorm_matmul_unroll(input_tensor, weight_matrix, TILING_FACTOR=2):
    M, K = input_tensor.shape
    K_, N = weight_matrix.shape
    assert K == K_, "Input and weight matrix dimensions do not match."
    assert K % TILING_FACTOR == 0
    TILED_CHUNK = K // TILING_FACTOR
    
    square_sum_buffer = np.zeros((M, TILED_CHUNK), dtype=np.float16)
    matmul_buffer = np.zeros((M, N), dtype=np.float16)

    for i in range(TILING_FACTOR): # TILING+FUSION LOOP
        input_chunk = input_tensor[:, i * TILED_CHUNK : (i + 1) * TILED_CHUNK]
        weight_chunk = weight_matrix[i * TILED_CHUNK : (i + 1) * TILED_CHUNK, :]

        squared_input = np.square(input_chunk)
        scaled_square = np.divide(squared_input, K)
        square_sum_buffer = np.add(scaled_square, square_sum_buffer)

        matmul_result = np.matmul(input_chunk, weight_chunk)
        matmul_buffer = np.add(matmul_result, matmul_buffer)

    rms_sum = np.sum(square_sum_buffer, axis=1, keepdims=True)
    rms_norm = np.sqrt(rms_sum)
    normalized_output = np.divide(matmul_buffer, rms_norm)

    return normalized_output



# Example 2: Same logic with fori_loop (SCF loop structure in IR)
@trace(input_specs=[((4, 8), "f16"), ((8, 5), "f16")])
def tiled_rmsnorm_matmul_fori(input_tensor, weight_matrix, TILING_FACTOR=2):
    M, K = input_tensor.shape
    K_, N = weight_matrix.shape
    assert K == K_, "Input and weight matrix dimensions do not match."
    assert K % TILING_FACTOR == 0
    TILED_CHUNK = K // TILING_FACTOR
    
    square_sum_buffer = np.zeros((M, TILED_CHUNK), dtype=np.float16)
    matmul_buffer = np.zeros((M, N), dtype=np.float16)

    def body(i, accs):
        square_sum_buffer, matmul_buffer = accs
        input_chunk = input_tensor[:, i * TILED_CHUNK : (i + 1) * TILED_CHUNK]
        weight_chunk = weight_matrix[i * TILED_CHUNK : (i + 1) * TILED_CHUNK, :]

        squared_input = np.square(input_chunk)
        scaled_square = np.divide(squared_input, K)
        square_sum_buffer = np.add(scaled_square, square_sum_buffer)

        matmul_result = np.matmul(input_chunk, weight_chunk)
        matmul_buffer = np.add(matmul_result, matmul_buffer)
        
        return (square_sum_buffer, matmul_buffer)
    
    square_sum_buffer, matmul_buffer = fori_loop(
        0, TILING_FACTOR, body, (square_sum_buffer, matmul_buffer)
    )

    rms_sum = np.sum(square_sum_buffer, axis=1, keepdims=True)
    rms_norm = np.sqrt(rms_sum)
    normalized_output = np.divide(matmul_buffer, rms_norm)

    return normalized_output

if __name__ == "__main__":
    # Test Example 1 (unrolled)
    print("=" * 60)
    print("Example 1: Unrolled tile loops")
    print("=" * 60)
    module1 = tiled_rmsnorm_matmul_unroll.to_mlir()
    print(module1)

    A1 = np.random.randn(4, 8).astype(np.float16)
    B1 = np.random.randn(8, 5).astype(np.float16)

    print(f"\nInput A shape: {A1.shape}")
    print(f"Input B shape: {B1.shape}")

    matches, mlir_result, numpy_result = verify_against_numpy(
        tiled_rmsnorm_matmul_unroll, tiled_rmsnorm_matmul_unroll.__wrapped__, [A1, B1],
        rtol=0.01, atol=0.01
    )

    if matches is not None:
        print(f"\n✓ Results match: {matches}")
        print(f"Max difference: {np.max(np.abs(mlir_result - numpy_result))}")
    
    # Test Example 2 (fori_loop)
    print("\n" + "=" * 60)
    print("Example 2: Same logic with fori_loop (SCF loop in IR)")
    print("=" * 60)
    module2 = tiled_rmsnorm_matmul_fori.to_mlir()
    print(module2)

    print(f"\nInput A shape: {A1.shape}")
    print(f"Input B shape: {B1.shape}")

    matches2, mlir_result2, numpy_result2 = verify_against_numpy(
        tiled_rmsnorm_matmul_fori, tiled_rmsnorm_matmul_fori.__wrapped__, [A1, B1],
        rtol=0.01, atol=0.01
    )

    if matches2 is not None:
        print(f"\n✓ Results match: {matches2}")
        print(f"Max difference: {np.max(np.abs(mlir_result2 - numpy_result2))}")
