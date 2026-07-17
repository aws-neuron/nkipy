"""Sweep P-EAGLE over (K, MoE kernel) and tabulate end-to-end throughput.

Runs peagle/speculate.py once per (K, kernel) combo via torchrun, parses the
metrics it prints, and reports tok/s + acceptance + verify ms/step so the best
combination is obvious.

Run (from examples/models/gpt_oss/):
    python peagle/_sweep.py -n 128 --prompt "..." \
        --k 1 3 5 7 --kernels loop batched dense
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # gpt_oss/

PATTERNS = {
    "tok_s": re.compile(r"Decode tokens/sec:\s*([\d.]+)"),
    "accept": re.compile(r"Mean acceptance length:\s*([\d.]+)"),
    "verify_ms": re.compile(r"verify:\s*([\d.]+)\s*ms/step"),
    "draft_ms": re.compile(r"draft\s*:\s*([\d.]+)\s*ms/step"),
    "n_steps": re.compile(r"in (\d+) verify steps"),
}


def run_one(k, kernel, args):
    env = dict(os.environ)
    env["GPT_OSS_MOE_KERNEL"] = kernel
    env["SPEC_PROFILE"] = "1"
    cmd = [
        "torchrun", "--nproc-per-node", str(args.tp),
        os.path.join(HERE, "speculate.py"),
        "--target-checkpoint", args.target_checkpoint,
        "--draft-checkpoint", args.draft_checkpoint,
        "--model", args.model,
        "--draft-model", args.draft_model,
        "-n", str(args.n), "-k", str(k),
        args.prompt,
    ]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    res = {}
    for key, pat in PATTERNS.items():
        m = pat.search(out)
        res[key] = float(m.group(1)) if m else None
    if res["tok_s"] is None:
        # surface the tail so failures aren't silent
        print(f"  !! parse failed for K={k} {kernel}; tail:\n"
              + "\n".join(out.strip().splitlines()[-8:]))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=128)
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 7])
    p.add_argument("--kernels", nargs="+", default=["loop", "batched", "dense"])
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--prompt", default="Write a Python function for binary search.")
    p.add_argument("--target-checkpoint", default="./tmp_gpt-oss-20b")
    p.add_argument("--draft-checkpoint", default="./peagle/tmp_p-eagle")
    p.add_argument("--model", default="/home/ubuntu/models/gpt-oss-20b")
    p.add_argument("--draft-model", default="/home/ubuntu/models/GPT-OSS-20B-P-EAGLE")
    p.add_argument("--results", default="/tmp/peagle_sweep_results.jsonl",
                   help="append each combo's metrics here; completed combos are "
                   "skipped on re-run so a killed sweep resumes cheaply")
    args = p.parse_args()

    # Resume: load already-completed (K, kernel) results.
    import json
    done = {}
    if os.path.exists(args.results):
        for line in open(args.results):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            done[(d["k"], d["kernel"])] = d["res"]

    rows = []
    for k in args.k:
        for kernel in args.kernels:
            if (k, kernel) in done:
                print(f"skip  K={k} kernel={kernel} (cached)", flush=True)
                rows.append((k, kernel, done[(k, kernel)]))
                continue
            print(f"running K={k} kernel={kernel} ...", flush=True)
            r = run_one(k, kernel, args)
            rows.append((k, kernel, r))
            with open(args.results, "a") as f:  # persist immediately
                f.write(json.dumps({"k": k, "kernel": kernel, "res": r}) + "\n")

    print("\n================ P-EAGLE sweep ================")
    print(f"  n={args.n}  prompt={args.prompt!r}")
    print(f"  {'K':>2} {'kernel':8s} {'tok/s':>7s} {'accept':>7s} "
          f"{'verify_ms':>10s} {'draft_ms':>9s}")
    best = None
    for k, kernel, r in rows:
        ts = r["tok_s"]
        flag = ""
        if ts is not None and (best is None or ts > best[0]):
            best = (ts, k, kernel)
        print(f"  {k:>2} {kernel:8s} "
              f"{_f(r['tok_s'],7,1)} {_f(r['accept'],7,2)} "
              f"{_f(r['verify_ms'],10,1)} {_f(r['draft_ms'],9,1)}{flag}")
    if best:
        print(f"\n  BEST: {best[0]:.1f} tok/s at K={best[1]}, kernel={best[2]}")
    print("===============================================")


def _f(v, w, p):
    return f"{v:{w}.{p}f}" if v is not None else " " * (w - 1) + "?"


if __name__ == "__main__":
    sys.exit(main())
