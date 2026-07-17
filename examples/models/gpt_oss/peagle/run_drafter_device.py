"""Standalone harness: run the on-device P-EAGLE drafter kernel on Trainium.

Loads the prepared (replicated) drafter weights, compiles the parallel-draft
NKI kernel, and runs a single draft() call. This exercises the device path
(`DrafterModel` + `kernels/drafter.py`), which speculate.py does not currently
use. Run with: NEURON_PLATFORM_TARGET_OVERRIDE=trn2 torchrun --nproc-per-node 1 \
    peagle/run_drafter_device.py --draft-checkpoint ./peagle/tmp_p-eagle \
    --draft-model amazon/GPT-OSS-20B-P-EAGLE
"""

import argparse
import os
import sys

# Same sys.path dance as speculate.py: the base gpt_oss/ dir must win over the
# peagle/ dir for the flat `config.py`, and peagle code is reached via `peagle.*`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from config import get_config  # noqa: E402  (base target config for hidden size)
from peagle.config import get_eagle_config  # noqa: E402
from peagle import drafter_model as dm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-checkpoint", default="./peagle/tmp_p-eagle")
    ap.add_argument("--draft-model", default="amazon/GPT-OSS-20B-P-EAGLE")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("-k", "--num-draft-tokens", type=int, default=7)
    args = ap.parse_args()

    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "localhost:61240")
    dist.init_process_group()
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank())

    # Target hidden size (gpt-oss-20b). get_config wants prompt/gen lengths; the
    # drafter only reads hidden_size from it.
    tgt_cfg = get_config(args.model, 16, 16)
    target_hidden = tgt_cfg.hidden_size

    cfg = get_eagle_config(
        args.draft_model, target_hidden, num_draft_tokens=args.num_draft_tokens
    )
    print(f"[rank {dist.get_rank()}] EagleConfig: H={cfg.hidden_size} "
          f"layers={cfg.num_layers} K={cfg.num_draft_tokens} "
          f"target_hidden={cfg.target_hidden_size}")

    weights = load_file(
        os.path.join(args.draft_checkpoint, "drafter.safetensors"), device="cpu"
    )

    build_dir = os.path.abspath(os.path.join(args.draft_checkpoint, "build_device"))
    os.makedirs(build_dir, exist_ok=True)

    print(f"[rank {dist.get_rank()}] Compiling drafter kernels...")
    drafter = dm.DrafterModel(weights, cfg, build_dir)
    print(f"[rank {dist.get_rank()}] Compiled in {drafter._compile_time:.1f}s")

    from peagle.drafter_cpu import DrafterCPU

    cpu = DrafterCPU(
        args.draft_model, target_hidden, num_draft_tokens=cfg.num_draft_tokens
    )

    # Synthetic context: a P-token prompt with tap-hiddens, then several commit +
    # draft steps with evolving positions (mirrors the speculation loop). At each
    # step compare the FULL logit rows (not just argmax) between device and CPU.
    rng = np.random.default_rng(0)
    P = 24
    K = cfg.num_draft_tokens
    prompt_tokens = torch.from_numpy(
        rng.integers(0, 50000, size=(P,)).astype(np.int64)
    )
    prompt_aux = torch.from_numpy(
        (rng.standard_normal((1, P, 3 * target_hidden)) * 0.02).astype(np.float32)
    ).to(torch.bfloat16)

    drafter.prefill(prompt_tokens, prompt_aux)
    cpu.reset()
    cpu.prefill(prompt_tokens, prompt_aux)

    def _cos(a, b):
        a, b = a.reshape(-1).double(), b.reshape(-1).double()
        return float((a @ b) / (a.norm() * b.norm() + 1e-12))

    base_pos = P
    all_ok = True
    for step in range(10):
        # Commit a variable number of tokens (1..3), like real acceptance.
        C = int(rng.integers(1, 4))
        commit_tok = rng.integers(0, 50000, size=(C,)).astype(int).tolist()
        commit_aux = torch.from_numpy(
            (rng.standard_normal((1, C, 3 * target_hidden)) * 0.02).astype(np.float32)
        ).to(torch.bfloat16)

        dev_drafts = drafter.draft(commit_tok, commit_aux, base_pos)
        dev_logits = drafter.last_logits.clone()
        cpu_drafts = cpu.draft(commit_tok, commit_aux, base_pos)
        cpu_logits = cpu.last_logits.clone()

        tok_match = sum(int(a == b) for a, b in zip(dev_drafts, cpu_drafts))
        cos = _cos(dev_logits, cpu_logits)
        max_abs = float((dev_logits - cpu_logits).abs().max())
        rel = float(
            (dev_logits - cpu_logits).abs().max()
            / (cpu_logits.abs().max() + 1e-6)
        )

        # For any mismatching draft position, check whether it's a near-tie: the
        # CPU logit gap between the device's pick and the CPU's pick. A tiny gap
        # (< bf16 resolution) means the two just rounded a tie differently, not a
        # real divergence.
        # A mismatch is "real" only if the CPU logit gap between the two picks
        # exceeds the step's numerical noise floor (max_abs_diff). A gap below the
        # noise is a bf16-resolution tie: within precision either side can win.
        tie_notes = []
        real_mismatch = False
        for i in range(K):
            if dev_drafts[i] != cpu_drafts[i]:
                gap = float(
                    cpu_logits[i, cpu_drafts[i]] - cpu_logits[i, dev_drafts[i]]
                )
                kind = "tie" if gap <= max_abs else "REAL"
                tie_notes.append(f"pos{i} gap={gap:.4f}({kind})")
                if gap > max_abs:
                    real_mismatch = True

        ok = cos > 0.999 and not real_mismatch
        all_ok = all_ok and ok
        note = ("  ties[" + ", ".join(tie_notes) + "]") if tie_notes else ""
        print(
            f"[rank {dist.get_rank()}] step {step} C={C} pos={base_pos}: "
            f"tokens {tok_match}/{K}  logit_cos={cos:.5f}  "
            f"max_abs_diff={max_abs:.4f}  rel={rel:.4f}  "
            f"{'OK' if ok else 'REAL MISMATCH'}{note}"
        )
        base_pos += C

    print(f"[rank {dist.get_rank()}] "
          f"{'ALL STEPS OK' if all_ok else 'MISMATCH DETECTED'}")
    print(f"[rank {dist.get_rank()}] OK: device drafter ran on trn2.")


if __name__ == "__main__":
    main()
