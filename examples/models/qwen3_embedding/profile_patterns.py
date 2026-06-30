#!/usr/bin/env python3
"""Per-pattern device benchmark: nkigen-lite vs HLO.

Compiles + loads each building-block kernel (rmsnorm, softmax, feedforward,
plus a layernorm and a standalone matmul) on the selected backend and times it
in isolation on device. For nkigen-lite it also reports the lowered nki-op
count so the wall-clock can be tied back to instruction issue.

Run once per backend (clean build between them):

    cd examples/models/qwen3_embedding && rm -rf build_pat/
    QWEN3_BACKEND=hlo         uv run python profile_patterns.py
    rm -rf build_pat/
    QWEN3_BACKEND=nkigen-lite uv run python profile_patterns.py
"""

import os
import time

import numpy as np

from config import Qwen3Config
from kernels.rmsnorm import rmsnorm
from kernels.softmax import softmax
from kernels.ffn import feedforward_kernel

from nkipy.core.trace import NKIPyKernel
from nkipy.runtime import DeviceKernel, DeviceTensor

BACKEND = os.environ.get("QWEN3_BACKEND", "nkigen-lite")
BUILD_DIR = os.path.join(os.path.dirname(__file__), "build_pat")
CC = " --lnc 2" + (" --enable-mixed-precision-accumulation" if BACKEND == "hlo" else "")

DT = np.dtype("float32")  # use f32 so both backends share one numerics path

# Representative shapes from Qwen3-Embedding-0.6B, batch=1 seq=128.
H = 1024            # hidden
INTER = 3072        # intermediate
HEADS = 16
SEQ = 128
SCORE = (1, HEADS, SEQ, SEQ)   # attention scores for softmax
HID = (1, SEQ, H)              # hidden states for norms / ffn


def _nki_op_count(fn, **arrays):
    """Lower the kernel to nki_ir and count ops (nkigen-lite only)."""
    if BACKEND != "nkigen-lite":
        return None
    from nkigen_lite.tensor_ir.passes import lower_to_nki
    k = NKIPyKernel.trace(fn, backend="nkigen-lite")
    ir = k.specialize(**arrays)
    return len(lower_to_nki(ir._graph).ops)


def bench(name, fn, arrays, n_warmup=5, n_iters=30):
    """Compile+load one pattern, time it on device, return (mean_ms, nki_ops)."""
    nki_ops = _nki_op_count(fn, **arrays)

    kernel = DeviceKernel.compile_and_load(
        kernel=NKIPyKernel.trace(fn, backend=BACKEND),
        name=f"pat_{name}",
        build_dir=BUILD_DIR,
        additional_compiler_args=CC,
        **arrays,
    )
    # Only ndarray args are device inputs; scalars (e.g. eps) are baked in.
    dev_inputs = {
        k: DeviceTensor.from_numpy(v, k)
        for k, v in arrays.items()
        if isinstance(v, np.ndarray)
    }
    out_shape = _infer_out_shape(name, arrays)
    out = DeviceTensor.from_numpy(np.zeros(out_shape, dtype=DT), "out")
    (out_name,) = kernel.output_tensors_info.keys()

    for _ in range(n_warmup):
        kernel(inputs=dev_inputs, outputs={out_name: out})
    lat = []
    for _ in range(n_iters):
        t = time.perf_counter()
        kernel(inputs=dev_inputs, outputs={out_name: out})
        lat.append((time.perf_counter() - t) * 1000)
    return float(np.mean(lat)), float(np.min(lat)), nki_ops


def _infer_out_shape(name, arrays):
    if name == "matmul_gup":
        return (*arrays["x"].shape[:-1], arrays["w"].shape[-1])
    if name == "swiglu_act":
        return arrays["g"].shape
    # softmax / rmsnorm / layernorm / feedforward all return the x shape
    return arrays["x"].shape


def main():
    cfg = Qwen3Config()
    eps = cfg.rms_norm_eps

    patterns = []

    # rmsnorm over hidden states (reduce over last dim)
    patterns.append((
        "rmsnorm",
        rmsnorm,
        {"x": np.random.randn(*HID).astype(DT),
         "weight": np.random.randn(H).astype(DT),
         "eps": eps},
    ))

    # softmax over attention scores (reduce over last dim, 16 heads)
    patterns.append((
        "softmax",
        softmax,
        {"x": np.random.randn(*SCORE).astype(DT)},
    ))

    # layernorm (mean-subtract + var-normalize + affine), separate from rmsnorm
    def layernorm(x, weight, bias, eps: float):
        cd = np.float32
        x = x.astype(cd)
        mu = np.mean(x, axis=-1, keepdims=True)
        var = np.mean(np.square(x - mu), axis=-1, keepdims=True)
        return ((x - mu) / np.sqrt(var + eps) * weight + bias)

    patterns.append((
        "layernorm",
        layernorm,
        {"x": np.random.randn(*HID).astype(DT),
         "weight": np.random.randn(H).astype(DT),
         "bias": np.random.randn(H).astype(DT),
         "eps": eps},
    ))

    # feedforward (SwiGLU): two matmuls + silu + elementwise
    patterns.append((
        "feedforward",
        feedforward_kernel,
        {"x": np.random.randn(*HID).astype(DT),
         "gate_up_weight": np.random.randn(H, 2 * INTER).astype(DT),
         "down_weight": np.random.randn(INTER, H).astype(DT),
         "gate_up_bias": np.zeros(2 * INTER, dtype=DT),
         "down_bias": np.zeros(H, dtype=DT)},
    ))

    # standalone matmul (the FFN's gate/up projection) to separate compute
    # from the surrounding silu/elementwise/split data movement
    def matmul_gup(x, w):
        return np.matmul(x, w)

    patterns.append((
        "matmul_gup",
        matmul_gup,
        {"x": np.random.randn(*HID).astype(DT),
         "w": np.random.randn(H, 2 * INTER).astype(DT)},
    ))

    # silu-gated elementwise on the FFN intermediate (no matmul): isolates the
    # split + silu + mul data-movement cost
    def swiglu_act(g, u):
        return (g * (1 / (1 + np.exp(-g)))) * u

    patterns.append((
        "swiglu_act",
        swiglu_act,
        {"g": np.random.randn(1, SEQ, INTER).astype(DT),
         "u": np.random.randn(1, SEQ, INTER).astype(DT)},
    ))

    print(f"\nBackend: {BACKEND}")
    print("=" * 60)
    hdr = f"  {'pattern':14s} {'mean ms':>9s} {'min ms':>9s}"
    if BACKEND == "nkigen-lite":
        hdr += f" {'nki_ops':>9s}"
    print(hdr)
    for name, fn, arrays in patterns:
        mean_ms, min_ms, nki_ops = bench(name, fn, arrays)
        line = f"  {name:14s} {mean_ms:9.3f} {min_ms:9.3f}"
        if BACKEND == "nkigen-lite":
            line += f" {nki_ops:9d}"
        print(line)


if __name__ == "__main__":
    main()
