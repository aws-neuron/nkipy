"""Microbenchmark: sweep MoE implementations across batch regimes.

Compiles each MoE variant as an isolated device kernel over N = B*L tokens,
verifies every variant matches the baseline on identical random weights, then
reports device exec time (from .ntff traces) for each. No real checkpoint needed
-- random weights exercise the same op graph.

Regimes of interest:
    L=1     decode (batch=1)
    L=4/8   speculative-decode verify (K+1 tokens)

Run (from examples/models/gpt_oss/):
    NEURON_RT_VISIBLE_CORES=0 python _moe_bench.py --dtype bf16 --L 1 4 8
"""

import argparse
import os
import time

import numpy as np
from nkipy.core import tensor_apis
from nkipy.runtime import DeviceKernel, DeviceTensor

from kernels.feedforward import (
    clamped_swiglu,
    moe_batched,
    moe_concat,
    moe_dense_masked,
)
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


# Variants under test. moe_baseline is the reference for correctness.
VARIANTS = {
    "baseline": moe_baseline,
    "batched": moe_batched,
    "concat": moe_concat,
    "dense": moe_dense_masked,
}


def _device_us(ntff, fn_name):
    """Device wall-clock (us) + matmul count from a trace, auto-matching NEFF."""
    import glob
    import json
    import subprocess

    for neff in sorted(glob.glob(f"build/{fn_name}_*/*.neff"),
                       key=os.path.getmtime, reverse=True):
        r = subprocess.run(
            ["neuron-profile", "view", "-n", neff, "-s", ntff,
             "--output-format", "json", "--output-file", "/tmp/_moe.json"],
            capture_output=True, text=True)
        if os.path.exists("/tmp/_moe.json") and os.path.getsize("/tmp/_moe.json") > 1000 \
           and "Unable to process" not in (r.stderr + r.stdout):
            d = json.load(open("/tmp/_moe.json"))
            os.remove("/tmp/_moe.json")
            ins = [i for i in d["instruction"] if i.get("duration", 0)]
            t0 = min(i["timestamp"] for i in ins)
            t1 = max(i["timestamp"] + i["duration"] for i in ins)
            mm = sum(1 for i in d["instruction"] if i.get("opcode") == "MATMUL")
            return (t1 - t0) / 1000.0, mm
        if os.path.exists("/tmp/_moe.json"):
            os.remove("/tmp/_moe.json")
    return None, None


def run_L(L, args):
    B = 1
    rng = np.random.default_rng(0)

    def r(shape, scale):
        # Scale BEFORE the final astype: bf16_array * python_float promotes back
        # to float32, so the cast must come last or the kernel compiles as fp32.
        return (rng.standard_normal(shape) * scale).astype(DT)

    x = DeviceTensor.from_numpy(r((B, L, D), 0.1), "x")
    w = {
        "router_weight": DeviceTensor.from_numpy(r((D, E), 0.05), "router_weight"),
        "router_bias": DeviceTensor.from_numpy(r((E,), 0.05), "router_bias"),
        "gate_up_weight": DeviceTensor.from_numpy(r((E, D, 2 * I), 0.02), "gate_up_weight"),
        "gate_up_bias": DeviceTensor.from_numpy(r((E, 2 * I), 0.02), "gate_up_bias"),
        "down_weight": DeviceTensor.from_numpy(r((E, I, D), 0.02), "down_weight"),
        "down_bias": DeviceTensor.from_numpy(r((E, D), 0.02), "down_bias"),
    }
    assert x.numpy().dtype == DT

    def run(kernel, out):
        kernel(inputs={"norm_z": x, **w}, outputs={"output0": out})

    tol = 2e-2 if args.dtype == "bf16" else 1e-4
    ref = None
    rows = []
    all_ok = True
    for tag, fn in VARIANTS.items():
        k = _make_kernel(fn, x, w)
        out = DeviceTensor.from_numpy(np.zeros((B, L, D), dtype=DT), f"o_{tag}")
        run(k, out)
        val = out.torch().float().numpy()
        if ref is None:
            ref, rel = val, 0.0
        else:
            rel = np.max(np.abs(val - ref)) / (np.max(np.abs(ref)) + 1e-9)
        ok = rel < tol
        all_ok = all_ok and ok
        ntff = os.path.abspath(f"moe_{tag}.ntff")
        run(k, out)  # warm
        k(inputs={"norm_z": x, **w}, outputs={"output0": out},
          save_trace=True, ntff_name=ntff)
        out.torch()
        dev_us, mm = _device_us(ntff, fn.__name__)
        os.remove(ntff)
        rows.append((tag, dev_us, mm, rel, ok))

    base_us = rows[0][1]
    print(f"\n===== L={L} (N={L}), dtype={np.dtype(DT).name}, top_k={TOP_K} =====")
    print(f"  {'variant':10s} {'device_us':>10s} {'matmuls':>8s} {'speedup':>8s} "
          f"{'rel_err':>9s}  ok")
    for tag, dev_us, mm, rel, ok in rows:
        sp = base_us / dev_us if dev_us else float("nan")
        print(f"  {tag:10s} {dev_us:10.1f} {mm:8d} {sp:7.2f}x {rel:9.2e}  "
              f"{'Y' if ok else 'N'}")
    return all_ok


def main():
    global DT
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, nargs="+", default=[1, 4, 8],
                   help="token counts to sweep (1=decode, K+1=verify)")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    args = p.parse_args()
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
    if args.dtype == "bf16":
        import ml_dtypes
        DT = np.dtype(ml_dtypes.bfloat16)

    ok = True
    for L in args.L:
        ok = run_L(L, args) and ok
    return ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
