"""Device tensor-parallel validation for the Qwen-Image MMDiT.

Shards a reduced-config denoiser across ``TP`` cores, compiles and runs
``qwenimage_forward`` with the real Neuron all-reduce collective, and (on rank 0)
compares against the single-core diffusers baseline. This exercises the actual
multi-core path — sharded weights + collectives — without the 20B download.

    cd examples/models/qwen_image
    torchrun --nproc-per-node 4 tests/test_tp_device.py --num-layers 2 --num-heads 8

The reduced config keeps head_dim=128 (real value) but shrinks heads / text dim /
blocks so it fits and compiles fast; num_heads must be divisible by TP.

Needs Neuron hardware + multiple cores, so the collected ``pytest`` entry
(`test_tp_device`) **skips by default** on a normal CPU test run. Opt in with
``QWEN_IMAGE_TP_DEVICE_TEST=1`` (optionally ``QWEN_IMAGE_TP_SIZE=<n>``) and it
shells out to the ``torchrun`` invocation above.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

# runnable from tests/: put the example root (config, kernels, weight_extract) on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--joint-dim", type=int, default=64)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--txt-len", type=int, default=16)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # must be set before importing nkipy: keep the NRT per-execution barrier on
    # for correct multi-rank collectives (SPMD-with-collectives).
    os.environ["NEURON_RT_DISABLE_EXECUTION_BARRIER"] = "0"
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "localhost:61375")
    dist.init_process_group()
    rank = dist.get_rank()
    tp_size = dist.get_world_size()
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)
    torch.set_num_threads(1)

    assert args.num_heads % tp_size == 0, "num_heads must be divisible by TP size"

    from config import Config
    from diffusers import QwenImageTransformer2DModel
    from kernels.tp import make_all_reduce
    from kernels.transformer import qwenimage_forward
    from kernels.weight_layout import BLOCK_KEYS, SHARED_KEYS, block_key
    from nkipy.runtime import DeviceKernel, DeviceTensor
    from nkipy.runtime.device_tensor import bfloat16
    from weight_extract import extract_flat_weights, shard_flat_weights

    d = args.head_dim
    a0 = (d // 3) & ~1
    a1 = (d // 3) & ~1
    a2 = d - a0 - a1
    hf = dict(
        num_layers=args.num_layers, num_attention_heads=args.num_heads,
        attention_head_dim=args.head_dim, joint_attention_dim=args.joint_dim,
        in_channels=16, out_channels=4, patch_size=2, axes_dims_rope=(a0, a1, a2),
    )

    torch.manual_seed(args.seed)
    model = QwenImageTransformer2DModel(**hf).eval()

    frame, gh, gw = 1, args.grid, args.grid
    B, Limg = 2, frame * gh * gw
    rng = np.random.default_rng(args.seed)
    latent = rng.standard_normal((B, Limg, hf["in_channels"])).astype(np.float32)
    text = rng.standard_normal((B, args.txt_len, hf["joint_attention_dim"])).astype(np.float32)
    timestep = rng.uniform(0, 1000, size=(B,)).astype(np.float32)

    # single-core reference (rank 0 only)
    ref = None
    if rank == 0:
        with torch.no_grad():
            ref = model(
                hidden_states=torch.from_numpy(latent),
                encoder_hidden_states=torch.from_numpy(text),
                timestep=torch.from_numpy(timestep),
                img_shapes=[(frame, gh, gw)] * B, return_dict=True,
            ).sample.cpu().numpy()

    # this rank's shard
    full = extract_flat_weights(model, hf["num_layers"], dtype=bfloat16)
    shard = shard_flat_weights(full, rank, tp_size, hf["num_layers"],
                               hf["num_attention_heads"], hf["attention_head_dim"])

    cfg = Config(
        num_layers=hf["num_layers"], num_heads=hf["num_attention_heads"],
        head_dim=hf["attention_head_dim"], joint_attention_dim=hf["joint_attention_dim"],
        in_channels=hf["in_channels"], out_channels=hf["out_channels"],
        patch_size=hf["patch_size"], axes_dims_rope=hf["axes_dims_rope"],
        dtype=bfloat16, tp_size=tp_size, all_reduce_fn=make_all_reduce(tp_size),
    )

    present = [k for k in SHARED_KEYS if k in shard]
    for i in range(cfg.num_layers):
        present += [block_key(i, s) for s in BLOCK_KEYS if block_key(i, s) in shard]
    device_weights = {k: DeviceTensor.from_numpy(shard[k], name=k) for k in present}

    latent_d = DeviceTensor.from_numpy(latent.astype(bfloat16), "latent")
    text_d = DeviceTensor.from_numpy(text.astype(bfloat16), "text")
    timestep_d = DeviceTensor.from_numpy(timestep.astype(np.float32), "timestep")

    if rank == 0:
        print(f"[qwen-image-tp] compiling (TP={tp_size}, {cfg.num_layers} blocks, "
              f"{cfg.num_heads}x{cfg.head_dim} heads, grid {gh}x{gw})")
    kernel = DeviceKernel.compile_and_load(
        qwenimage_forward, name="qwenimage_forward_tp",
        latent=latent_d, text=text_d, timestep=timestep_d,
        img_shape=(frame, gh, gw), configs=cfg, build_dir="./build_tp",
        additional_compiler_args=cfg.additional_compiler_args_nkipy,
        **device_weights,
    )
    out_dim = cfg.patch_size * cfg.patch_size * cfg.out_channels
    out_d = DeviceTensor.from_numpy(np.empty((B, Limg, out_dim), dtype=bfloat16), "out")
    expected = set(kernel.input_tensors_info)
    inputs = {"latent": latent_d, "text": text_d, "timestep": timestep_d}
    inputs.update({k: v for k, v in device_weights.items() if k in expected})
    kernel(inputs=inputs, outputs={"output0": out_d})
    dev = out_d.torch().to(torch.float32).numpy()

    dist.barrier()
    if rank == 0:
        rel = _rel_l2(dev, ref)
        print(f"[qwen-image-tp] TP={tp_size} device vs diffusers: rel_l2={rel:.4e}")
        print("[qwen-image-tp] PASS" if rel < args.tol else "[qwen-image-tp] FAIL")
    dist.destroy_process_group()


def test_tp_device():
    """Collected pytest entry: skips unless opted in (needs Neuron hardware +
    multiple cores), then drives the torchrun harness above and checks it PASSes.

    A single pytest process can't be the N ranks, so we shell out to torchrun on
    this same file — its ``__main__`` block runs ``main()`` per rank.
    """
    import subprocess

    import pytest

    if os.environ.get("QWEN_IMAGE_TP_DEVICE_TEST") != "1":
        pytest.skip("device TP test needs Neuron hardware; opt in with "
                    "QWEN_IMAGE_TP_DEVICE_TEST=1")
    root = Path(__file__).resolve().parents[1]
    nproc = os.environ.get("QWEN_IMAGE_TP_SIZE", "4")
    proc = subprocess.run(
        ["torchrun", "--nproc-per-node", nproc, str(Path(__file__)),
         "--num-layers", "2", "--num-heads", "8"],
        cwd=str(root), capture_output=True, text=True)
    print(proc.stdout)
    print(proc.stderr)
    assert "[qwen-image-tp] PASS" in proc.stdout, proc.stderr or proc.stdout


if __name__ == "__main__":
    main()
