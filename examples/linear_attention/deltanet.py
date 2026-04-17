#!/usr/bin/env python3
"""
DeltaNet Linear Attention Example using NKIPy

DeltaNet applies the "delta rule" to linear attention, replacing the softmax
with a recurrent state update that achieves O(N*D^2) complexity instead of
O(N^2*D). For each timestep t:

    S_t = S_{t-1} + beta_t * (v_t - S_{t-1} @ k_t) outer k_t   # state update
    o_t = S_t @ q_t                                               # output

This example provides:
1. A PyTorch reference implementation for correctness validation
2. An NKIPy kernel using pure NumPy ops (the timestep loop is unrolled at trace time)
3. Optional on-device compilation and benchmarking
"""

import time

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from nkipy.core import tensor_apis
from nkipy.runtime import DeviceKernel, DeviceTensor, is_neuron_compatible


def deltanet_pytorch(q, k, v, beta):
    """
    PyTorch reference for DeltaNet recurrent linear attention.

    Args:
        q: queries  [B, H, L, D]
        k: keys     [B, H, L, D] (will be L2-normalized)
        v: values   [B, H, L, D]
        beta: gates [B, H, L] in (0, 1), post-sigmoid

    Returns:
        output [B, H, L, D]
    """
    B, H, L, D = q.shape

    # L2-normalize keys
    k = k / torch.clamp(torch.norm(k, dim=-1, keepdim=True), min=1e-6)

    S = torch.zeros(B, H, D, D, dtype=q.dtype, device=q.device)
    outputs = []

    for t in range(L):
        q_t = q[:, :, t, :]  # [B, H, D]
        k_t = k[:, :, t, :]  # [B, H, D]
        v_t = v[:, :, t, :]  # [B, H, D]
        beta_t = beta[:, :, t]  # [B, H]

        # delta = beta_t * (v_t - S @ k_t)
        Sk = torch.einsum("bhde,bhe->bhd", S, k_t)  # [B, H, D]
        delta = beta_t.unsqueeze(-1) * (v_t - Sk)  # [B, H, D]

        # S += delta outer k_t
        S = S + torch.einsum("bhd,bhe->bhde", delta, k_t)  # [B, H, D, D]

        # o_t = S @ q_t
        o_t = torch.einsum("bhde,bhe->bhd", S, q_t)  # [B, H, D]
        outputs.append(o_t.unsqueeze(2))

    return torch.cat(outputs, dim=2)  # [B, H, L, D]


def deltanet_nkipy(q, k, v, beta_logits):
    """
    NKIPy kernel for DeltaNet recurrent linear attention.

    Args:
        q: queries      [B, H, L, D] float32
        k: keys         [B, H, L, D] float32 (will be L2-normalized)
        v: values       [B, H, L, D] float32
        beta_logits: gate logits [B, H, L] float32 (pre-sigmoid)

    Returns:
        output [B, H, L, D] float32
    """
    B, H, L, D = q.shape

    # Sigmoid activation: beta = 1 / (1 + exp(-x))
    beta = 1.0 / (1.0 + np.exp(-beta_logits))

    # L2-normalize keys
    k_norm = np.linalg.norm(k, axis=-1, keepdims=True)
    k = k / np.maximum(k_norm, 1e-6)

    # Initialize state [B, H, D, D]
    # Use tensor_apis.zeros so this works during both CPU and HLO tracing
    S = tensor_apis.zeros((B, H, D, D), dtype=q.dtype)

    outputs = []
    for t in range(L):
        q_t = q[:, :, t, :]  # [B, H, D]
        k_t = k[:, :, t, :]  # [B, H, D]
        v_t = v[:, :, t, :]  # [B, H, D]
        beta_t = beta[:, :, t]  # [B, H]

        # S @ k_t: matmul state with key vector
        # [B, H, D, D] @ [B, H, D, 1] -> [B, H, D, 1] -> [B, H, D]
        k_col = np.expand_dims(k_t, axis=-1)  # [B, H, D, 1]
        Sk = np.matmul(S, k_col)[:, :, :, 0]  # [B, H, D]

        # delta = beta_t * (v_t - Sk)
        beta_2d = np.expand_dims(beta_t, axis=-1)  # [B, H, 1]
        delta = beta_2d * (v_t - Sk)  # [B, H, D]

        # Batched outer product: delta outer k_t -> [B, H, D, D]
        outer = np.expand_dims(delta, axis=-1) * np.expand_dims(k_t, axis=-2)

        # State update
        S = S + outer

        # Output: S @ q_t -> [B, H, D]
        q_col = np.expand_dims(q_t, axis=-1)  # [B, H, D, 1]
        o_t = np.matmul(S, q_col)[:, :, :, 0]  # [B, H, D]

        outputs.append(np.expand_dims(o_t, axis=2))  # [B, H, 1, D]

    return np.concatenate(outputs, axis=2)  # [B, H, L, D]


def main():
    print("=" * 80)
    print("DeltaNet Linear Attention Example")
    print("=" * 80)

    # Configuration
    B, H, L, D = 1, 4, 64, 32
    dtype = np.float32

    print(f"\nConfiguration: B={B}, H={H}, L={L}, D={D}, dtype={dtype.__name__}")

    # Create random inputs
    print("\n[1/5] Creating test data...")
    np.random.seed(42)
    q = np.random.randn(B, H, L, D).astype(dtype) * 0.1
    k = np.random.randn(B, H, L, D).astype(dtype) * 0.1
    v = np.random.randn(B, H, L, D).astype(dtype) * 0.1
    beta_logits = np.random.randn(B, H, L).astype(dtype)  # pre-sigmoid

    # PyTorch reference
    if TORCH_AVAILABLE:
        print("\n[2/5] Running PyTorch reference...")
        q_pt = torch.from_numpy(q)
        k_pt = torch.from_numpy(k)
        v_pt = torch.from_numpy(v)
        beta_pt = torch.sigmoid(torch.from_numpy(beta_logits))
        ref_output = deltanet_pytorch(q_pt, k_pt, v_pt, beta_pt).numpy()
        print(f"  PyTorch output shape: {ref_output.shape}")
    else:
        print("\n[2/5] PyTorch not available, skipping reference...")
        ref_output = None

    # NKIPy CPU execution (pure numpy)
    print("\n[3/5] Running NKIPy kernel (CPU mode)...")
    cpu_output = deltanet_nkipy(q, k, v, beta_logits)
    print(f"  NKIPy CPU output shape: {cpu_output.shape}")

    # Compare CPU vs PyTorch
    if ref_output is not None:
        print("\n[4/5] Validating CPU correctness against PyTorch...")
        try:
            np.testing.assert_allclose(cpu_output, ref_output, rtol=1e-4, atol=1e-4)
            max_err = np.max(np.abs(cpu_output - ref_output))
            print(f"  PASSED - max absolute error: {max_err:.2e}")
        except AssertionError as e:
            print(f"  FAILED: {e}")
            return
    else:
        print("\n[4/5] Skipping validation (no PyTorch reference)...")

    # On-device execution
    if is_neuron_compatible():
        print("\n[5/5] Compiling and running on Neuron hardware...")
        compile_start = time.time()
        kernel = DeviceKernel.compile_and_load(
            deltanet_nkipy,
            q,
            k,
            v,
            beta_logits,
            name="deltanet_kernel",
            use_cached_if_exists=False,
        )
        compile_time = time.time() - compile_start
        print(f"  Compiled in {compile_time:.2f}s")

        # Create device tensors
        d_q = DeviceTensor.from_numpy(q)
        d_k = DeviceTensor.from_numpy(k)
        d_v = DeviceTensor.from_numpy(v)
        d_beta = DeviceTensor.from_numpy(beta_logits)
        d_out = DeviceTensor.from_numpy(np.zeros_like(cpu_output))

        kernel(
            inputs={"q": d_q, "k": d_k, "v": d_v, "beta_logits": d_beta},
            outputs={"output0": d_out},
        )
        device_output = d_out.numpy()

        try:
            np.testing.assert_allclose(device_output, cpu_output, rtol=1e-2, atol=1e-2)
            max_err = np.max(np.abs(device_output - cpu_output))
            print(f"  Device output matches CPU - max error: {max_err:.2e}")
        except AssertionError as e:
            print(f"  Device validation failed: {e}")
            return

        # Benchmark
        stats = kernel.benchmark(
            inputs={"q": d_q, "k": d_k, "v": d_v, "beta_logits": d_beta},
            outputs={"output0": d_out},
            warmup_iter=5,
            benchmark_iter=10,
        )
        print(
            f"\n  Performance: mean={stats.mean_ms:.3f}ms, "
            f"min={stats.min_ms:.3f}ms, max={stats.max_ms:.3f}ms"
        )
    else:
        print("\n[5/5] No Neuron hardware detected, skipping on-device execution.")

    print(f"\n{'=' * 80}")
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
