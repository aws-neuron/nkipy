"""Layer-by-layer validation: compare HF (CPU) vs NKIPy (Trainium) outputs.

Usage:
  uv run torchrun --nproc_per_node 8 validate_layers.py \
      --checkpoint ./qwen3_5_shards --model Qwen/Qwen3.5-35B-A3B

Finds the first layer where NKIPy diverges from the HF reference.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from config import FULL_ATTENTION, LINEAR_ATTENTION, Config, get_config
from nkipy.runtime import DeviceKernel, DeviceTensor
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import print_log


def compare_tensors(name, hf_tensor, nkipy_tensor, atol=0.05, rtol=0.05):
    """Compare two tensors, return (passed, stats_str)."""
    hf = hf_tensor.float().cpu()
    nk = torch.from_numpy(nkipy_tensor).float().cpu() if isinstance(nkipy_tensor, np.ndarray) else nkipy_tensor.float().cpu()

    # Align shapes
    if hf.shape != nk.shape:
        return False, f"SHAPE MISMATCH: HF {hf.shape} vs NKIPy {nk.shape}"

    abs_diff = (hf - nk).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # Relative error (avoid div by zero)
    denom = hf.abs().clamp(min=1e-8)
    rel_diff = (abs_diff / denom)
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        hf.flatten().unsqueeze(0), nk.flatten().unsqueeze(0)
    ).item()

    passed = cos_sim > 0.99 and max_abs < 5.0
    status = "PASS" if passed else "FAIL"

    stats = (
        f"[{status}] {name}: "
        f"cos_sim={cos_sim:.6f}  "
        f"max_abs={max_abs:.4f}  mean_abs={mean_abs:.4f}  "
        f"max_rel={max_rel:.4f}  mean_rel={mean_rel:.4f}"
    )
    return passed, stats


def run_hf_reference(model_name, prompt, dtype=torch.bfloat16):
    """Run HF model on CPU, capture outputs after each layer."""
    print_log("Loading HF reference model on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Disable grouped_mm which has alignment issues on CPU
    import transformers.integrations.moe as _moe_mod
    _moe_mod.is_grouped_mm_available = lambda: False

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="cpu", low_cpu_mem_usage=True
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Register hooks to capture layer outputs
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # DecoderLayer returns hidden_states (possibly as tensor directly)
            if isinstance(out, tuple):
                layer_outputs[layer_idx] = out[0].detach().clone()
            else:
                layer_outputs[layer_idx] = out.detach().clone()
        return hook

    # Hook embedding
    def embed_hook(module, inp, out):
        layer_outputs["embedding"] = out.detach().clone()

    model.model.embed_tokens.register_forward_hook(embed_hook)

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(make_hook(i))

    # Hook final norm
    def norm_hook(module, inp, out):
        layer_outputs["final_norm"] = out.detach().clone()
    model.model.norm.register_forward_hook(norm_hook)

    print_log("Running HF forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)

    # Get logits for the last token
    logits = outputs.logits[:, -1, :]  # (1, vocab_size)
    layer_outputs["logits"] = logits.detach().clone()

    # Get top-5 predictions
    top5_vals, top5_ids = logits.topk(5, dim=-1)
    print_log(f"HF top-5 token IDs: {top5_ids[0].tolist()}")
    print_log(f"HF top-5 logit values: {[f'{v:.2f}' for v in top5_vals[0].tolist()]}")
    decoded = [tokenizer.decode([tid]) for tid in top5_ids[0].tolist()]
    print_log(f"HF top-5 tokens: {decoded}")

    # Clean up HF model to free memory
    del model
    import gc
    gc.collect()

    return layer_outputs, input_ids


def run_nkipy_prefill_with_capture(args, input_ids_np):
    """Run NKIPy prefill, capture hidden_states after each layer."""
    from qwen3_5 import Qwen35Model

    config = get_config(args.model, input_ids_np.shape[1], 1)

    shard_path = os.path.join(args.checkpoint, f"shard_{dist.get_rank()}.safetensors")
    weights = load_file(shard_path, device="cpu")

    model = Qwen35Model(weights, config)

    # Capture embedding output
    nkipy_outputs = {}
    hidden_states = DeviceTensor.from_torch(
        model.tok_embedding[input_ids_np], "hidden_states"
    )
    nkipy_outputs["embedding"] = hidden_states.torch().clone()

    # Run prefill layer by layer, capturing outputs
    for i in range(config.num_layers):
        model._run_layer(
            model.kernel_cte_full_attn,
            model.kernel_cte_linear_attn,
            i,
            hidden_states,
            None,  # prefill: start_pos=None
        )
        # Read back hidden_states from device
        nkipy_outputs[i] = hidden_states.torch().clone()

    # Run sampling to get logits (we can compare top token)
    next_id = DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "next_id")
    model.kernel_cte_greedy_sampling(
        inputs={
            "h": hidden_states,
            "norm_weight": model.norm_weight,
            "lm_head_weight": model.lm_head_weight,
        },
        outputs={"output0": next_id},
    )
    next_id_val = next_id.torch().item()
    nkipy_outputs["next_token_id"] = next_id_val

    return nkipy_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./qwen3_5_shards")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    args = parser.parse_args()

    # Distributed setup
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61239"
    dist.init_process_group()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank())

    rank = dist.get_rank()
    is_rank0 = rank == 0

    # --- Step 1: Run HF reference (only on rank 0) ---
    hf_outputs = None
    input_ids_pt = None
    if is_rank0:
        hf_outputs, input_ids_pt = run_hf_reference(args.model, args.prompt)
        print_log(f"HF captured {len(hf_outputs)} outputs")

    dist.barrier()

    # --- Step 2: Run NKIPy on Trainium ---
    # All ranks need to tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_inputs = tokenizer(args.prompt, return_tensors="np")
    input_ids_np = model_inputs["input_ids"]

    print_log("Running NKIPy prefill with layer capture...")
    nkipy_outputs = run_nkipy_prefill_with_capture(args, input_ids_np)

    dist.barrier()

    # --- Step 3: Compare (rank 0 only) ---
    if is_rank0:
        config = get_config(args.model, input_ids_np.shape[1], 1)

        print("\n" + "=" * 70)
        print("LAYER-BY-LAYER COMPARISON: HF (CPU) vs NKIPy (Trainium)")
        print("=" * 70)

        # Compare embedding
        passed, stats = compare_tensors(
            "embedding", hf_outputs["embedding"], nkipy_outputs["embedding"]
        )
        print(stats)
        if not passed:
            print(">>> DIVERGENCE at embedding! Stopping.")
            return

        # Compare each layer
        first_fail_layer = None
        for layer_idx in range(config.num_layers):
            layer_type = config.layer_types[layer_idx]
            label = f"layer {layer_idx:2d} ({layer_type[:6]})"

            if layer_idx not in hf_outputs:
                print(f"[SKIP] {label}: no HF output captured")
                continue

            passed, stats = compare_tensors(
                label, hf_outputs[layer_idx], nkipy_outputs[layer_idx]
            )
            print(stats)

            if not passed and first_fail_layer is None:
                first_fail_layer = layer_idx

                # Print detailed info about the failing layer
                hf_t = hf_outputs[layer_idx].float()
                nk_t = nkipy_outputs[layer_idx].float()
                print(f"    HF  range: [{hf_t.min():.4f}, {hf_t.max():.4f}], mean={hf_t.mean():.4f}, std={hf_t.std():.4f}")
                print(f"    NKI range: [{nk_t.min():.4f}, {nk_t.max():.4f}], mean={nk_t.mean():.4f}, std={nk_t.std():.4f}")

                # Show where the biggest differences are
                diff = (hf_t - nk_t).abs()
                flat_idx = diff.flatten().topk(5).indices
                print(f"    Top-5 abs diff locations (flat idx): {flat_idx.tolist()}")

                # Also check if previous layer was OK
                if layer_idx > 0 and (layer_idx - 1) in hf_outputs:
                    prev_passed, _ = compare_tensors(
                        "prev", hf_outputs[layer_idx - 1], nkipy_outputs[layer_idx - 1]
                    )
                    if prev_passed:
                        print(f"    >>> Layer {layer_idx - 1} was OK. Bug is IN layer {layer_idx} ({layer_type}).")

        # Compare next token
        if "next_token_id" in nkipy_outputs:
            hf_next = hf_outputs["logits"].argmax(dim=-1).item()
            nk_next = nkipy_outputs["next_token_id"]
            match = "MATCH" if hf_next == nk_next else "MISMATCH"
            hf_tok = tokenizer.decode([hf_next])
            nk_tok = tokenizer.decode([nk_next])
            print(f"\nNext token: HF={hf_next} ('{hf_tok}') vs NKIPy={nk_next} ('{nk_tok}') [{match}]")

        print("\n" + "=" * 70)
        if first_fail_layer is not None:
            print(f"FIRST DIVERGENCE: Layer {first_fail_layer} ({config.layer_types[first_fail_layer]})")
        else:
            print("ALL LAYERS MATCH!")
        print("=" * 70)


if __name__ == "__main__":
    main()
