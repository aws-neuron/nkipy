"""Microbenchmark: baseline (per-token/expert loop) MoE vs batched MoE.

Compiles each MoE variant as an isolated device kernel over N = B*L tokens,
verifies the two produce the same output on identical random weights, then times
both. No real checkpoint needed -- random weights exercise the same op graph.

Run (from examples/models/gpt_oss/):
    NEURON_RT_VISIBLE_CORES=0 python _moe_bench.py --L 4
"""

import argparse
import os
import time

import numpy as np
from nkipy.core import tensor_apis
from nkipy.runtime import DeviceKernel, DeviceTensor

from kernels.feedforward import clamped_swiglu, moe_batched
from kernels.softmax import softmax_kernel

# gpt-oss-20b MoE dims, per TP4 rank (intermediate 2880 -> 720).
D = 2880           # hidden
I = 720            # intermediate per rank
E = 32             # experts
TOP_K = 4
ALPHA = 1.702
LIMIT = 7.0
DT = np.float32    # overridden by --dtype


def moe_baseline(
    norm_z, router_weight, router_bias, gate_up_weight, gate_up_bias,
    down_weight, down_bias, top_k, alpha, limit,
):
    """Current implementation: Python loop over (batch, token, expert)."""
    B, L, Dh = norm_z.shape
    router_logits = np.matmul(norm_z, router_weight) + router_bias
    output = np.empty_like(norm_z)
    for b in range(B):
        for t in range(L):
            token_input = norm_z[b, t, :]
            token_logits = router_logits[b, t]
            tvals, tidx = tensor_apis.topk(token_logits, k=top_k)
            tw = softmax_kernel(tvals)
            acc = tensor_apis.zeros((Dh), dtype=output.dtype)
            for e in range(top_k):
                ei = tidx[e]
                mm = np.matmul(token_input, gate_up_weight[ei]) + gate_up_bias[ei]
                xg, x_up = np.split(mm, 2, axis=-1)
                gated = clamped_swiglu(xg, x_up, alpha, limit)
                eo = np.matmul(gated, down_weight[ei]) + down_bias[ei]
                acc = acc + tw[e] * eo
            output[b, t] = acc
    return output


def _make_kernel(fn, x, w):
    return DeviceKernel.compile_and_load(
        fn,
        name=fn.__name__,
        norm_z=x,
        router_weight=w["router_weight"],
        router_bias=w["router_bias"],
        gate_up_weight=w["gate_up_weight"],
        gate_up_bias=w["gate_up_bias"],
        down_weight=w["down_weight"],
        down_bias=w["down_bias"],
        top_k=TOP_K,
        alpha=ALPHA,
        limit=LIMIT,
        build_dir=os.path.abspath("./build"),
        additional_compiler_args="--lnc 1",
    )


def main():
    global DT
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4, help="tokens (K+1 for verify)")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = p.parse_args()
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
    if args.dtype == "bf16":
        import ml_dtypes
        DT = np.dtype(ml_dtypes.bfloat16)

    B, L = 1, args.L
    rng = np.random.default_rng(0)

    def r(shape, scale):
        # Scale BEFORE the final astype: bf16_array * python_float promotes back
        # to float32, so the cast must come last or the kernel compiles as fp32.
        return (rng.standard_normal(shape) * scale).astype(DT)

    x_np = r((B, L, D), 0.1)
    w_np = {
        "router_weight": r((D, E), 0.05),
        "router_bias": r((E,), 0.05),
        "gate_up_weight": r((E, D, 2 * I), 0.02),
        "gate_up_bias": r((E, 2 * I), 0.02),
        "down_weight": r((E, I, D), 0.02),
        "down_bias": r((E, D), 0.02),
    }
    x = DeviceTensor.from_numpy(x_np, "x")
    w = {k: DeviceTensor.from_numpy(v, k) for k, v in w_np.items()}
    assert x.numpy().dtype == DT, f"input dtype {x.numpy().dtype} != requested {DT}"

    print(f"compiling kernels (B={B}, L={L}, E={E}, top_k={TOP_K}, "
          f"dtype={np.dtype(DT).name}) ...")
    k_base = _make_kernel(moe_baseline, x, w)
    k_batch = _make_kernel(moe_batched, x, w)

    # Kernel output dtype follows the input dtype (the matmuls preserve it).
    out_base = DeviceTensor.from_numpy(np.zeros((B, L, D), dtype=DT), "ob")
    out_batch = DeviceTensor.from_numpy(np.zeros((B, L, D), dtype=DT), "obt")

    def run(kernel, out):
        kernel(inputs={"norm_z": x, **w}, outputs={"output0": out})

    # correctness
    run(k_base, out_base)
    run(k_batch, out_batch)
    a = out_base.torch().float().numpy()
    b = out_batch.torch().float().numpy()
    max_abs = np.max(np.abs(a - b))
    rel = max_abs / (np.max(np.abs(a)) + 1e-9)
    tol = 2e-2 if args.dtype == "bf16" else 1e-4
    print(f"\nequivalence: max|Δ|={max_abs:.3e}  rel={rel:.3e}  (tol={tol})")
    ok = rel < tol
    print("  -> MATCH" if ok else "  -> MISMATCH")

    def timeit(kernel, out):
        for _ in range(5):
            run(kernel, out)
        out.torch()
        t0 = time.time()
        for _ in range(args.iters):
            run(kernel, out)
        out.torch()
        return (time.time() - t0) / args.iters

    tb = timeit(k_base, out_base)
    tt = timeit(k_batch, out_batch)

    # Device-time ground truth via .ntff trace (host wall-clock is dispatch-bound).
    def trace(kernel, out, name):
        ntff = os.path.abspath(f"{name}.ntff")
        kernel(inputs={"norm_z": x, **w}, outputs={"output0": out},
               save_trace=True, ntff_name=ntff)
        out.torch()
        return ntff

    nb = trace(k_base, out_base, "moe_base")
    nt = trace(k_batch, out_batch, "moe_batch")

    print(f"\n=========== MoE timing (L={L}, {args.iters} iters) ===========")
    print(f"  [host wall-clock, dispatch-bound]")
    print(f"    baseline (loop) : {tb*1e6:8.1f} us/call")
    print(f"    batched         : {tt*1e6:8.1f} us/call   ({tb/tt:.2f}x)")
    print(f"  [device exec time -- see extract below]")
    print(f"    baseline ntff   : {nb}")
    print(f"    batched  ntff   : {nt}")
    print("==============================================")
    return ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
