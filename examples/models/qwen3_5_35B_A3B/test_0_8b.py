"""End-to-end test: Qwen3.5-0.8B on Trainium with TP=1 (no sharding).

Tests GDN correctness without any TP complexity.
The 0.8B model is dense (no MoE), same GDN architecture as 35B-A3B.

Usage:
  # Step 1: Prepare weights (only needed once)
  uv run python test_0_8b.py --prepare

  # Step 2: Run inference + validate
  uv run torchrun --nproc_per_node 1 test_0_8b.py --run "The capital of France is"
"""

import argparse
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import ml_dtypes
import numpy as np
import torch

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
WEIGHTS_DIR = "./qwen3_5_0_8b_shards"
BUILD_DIR = "./build_0_8b"

bf16 = ml_dtypes.bfloat16


# =========================================================================
# Weight preparation (no TP, just rename/transpose)
# =========================================================================
def prepare_weights():
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(MODEL_NAME)
    text_cfg = config.text_config if hasattr(config, "text_config") else config

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=False
    )

    sd = model.state_dict()
    processed = {}
    n_layers = text_cfg.num_hidden_layers
    layer_types = list(text_cfg.layer_types)

    # Global
    processed["tok_embedding"] = sd["model.embed_tokens.weight"]
    processed["norm_weight"] = sd["model.norm.weight"]
    processed["lm_head_weight"] = sd["lm_head.weight"].T

    for i in range(n_layers):
        p = f"model.layers.{i}"
        lt = layer_types[i]

        # Norms
        processed[f"layers.{i}.input_weight"] = sd[f"{p}.input_layernorm.weight"]
        processed[f"layers.{i}.post_attention_weight"] = sd[f"{p}.post_attention_layernorm.weight"]

        if lt == "linear_attention":
            # GDN weights - just transpose projections
            processed[f"layers.{i}.linear_qkv_weight"] = sd[f"{p}.linear_attn.in_proj_qkv.weight"].T
            processed[f"layers.{i}.linear_z_weight"] = sd[f"{p}.linear_attn.in_proj_z.weight"].T
            processed[f"layers.{i}.linear_b_weight"] = sd[f"{p}.linear_attn.in_proj_b.weight"].T
            processed[f"layers.{i}.linear_a_weight"] = sd[f"{p}.linear_attn.in_proj_a.weight"].T
            processed[f"layers.{i}.linear_conv_weight"] = sd[f"{p}.linear_attn.conv1d.weight"].squeeze(1)
            processed[f"layers.{i}.linear_dt_bias"] = sd[f"{p}.linear_attn.dt_bias"]
            processed[f"layers.{i}.linear_A_log"] = sd[f"{p}.linear_attn.A_log"]
            processed[f"layers.{i}.linear_norm_weight"] = sd[f"{p}.linear_attn.norm.weight"]
            processed[f"layers.{i}.linear_out_weight"] = sd[f"{p}.linear_attn.out_proj.weight"].T
        else:
            # Full attention
            q_w = sd[f"{p}.self_attn.q_proj.weight"]
            k_w = sd[f"{p}.self_attn.k_proj.weight"]
            v_w = sd[f"{p}.self_attn.v_proj.weight"]
            processed[f"layers.{i}.qkv_weight"] = torch.cat([q_w.T, k_w.T, v_w.T], dim=-1)
            processed[f"layers.{i}.o_weight"] = sd[f"{p}.self_attn.o_proj.weight"].T
            processed[f"layers.{i}.q_norm_weight"] = sd[f"{p}.self_attn.q_norm.weight"]
            processed[f"layers.{i}.k_norm_weight"] = sd[f"{p}.self_attn.k_norm.weight"]

        # Dense FFN (not MoE)
        processed[f"layers.{i}.gate_proj_weight"] = sd[f"{p}.mlp.gate_proj.weight"].T
        processed[f"layers.{i}.up_proj_weight"] = sd[f"{p}.mlp.up_proj.weight"].T
        processed[f"layers.{i}.down_proj_weight"] = sd[f"{p}.mlp.down_proj.weight"].T

    for k, v in processed.items():
        processed[k] = v.contiguous()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_file(processed, os.path.join(WEIGHTS_DIR, "shard_0.safetensors"))
    print(f"Saved {len(processed)} tensors to {WEIGHTS_DIR}/shard_0.safetensors")
    del model


# =========================================================================
# Model (TP=1, dense FFN)
# =========================================================================
def run_inference(prompt, max_new_tokens=64):
    import torch.distributed as dist
    from nkipy.runtime import DeviceKernel, DeviceTensor
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer

    from kernels.rmsnorm import rmsnorm_kernel
    from kernels.attention import attention_kernel
    from kernels.linear_attention import gated_delta_net_kernel
    from kernels.softmax import softmax_kernel
    from kernels.feedforward import silu_kernel_
    import nkipy.distributed.collectives as cc
    from nkipy.core import tensor_apis
    import nkipy.core.typing as nt
    from typing import Optional

    os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61239"
    os.environ["OMP_NUM_THREADS"] = "1"
    dist.init_process_group()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"

    config = AutoConfig.from_pretrained(MODEL_NAME)
    tc = config.text_config

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    input_ids_np = tokenizer(prompt, return_tensors="np")["input_ids"]
    context_len = input_ids_np.shape[1]

    weights = load_file(os.path.join(WEIGHTS_DIR, "shard_0.safetensors"), device="cpu")
    tok_embedding = weights["tok_embedding"]

    # --- Define kernels ---
    def dense_ffn(x, gate_w, up_w, down_w):
        gate = silu_kernel_(np.matmul(x, gate_w))
        up = np.matmul(x, up_w)
        return np.matmul(gate * up, down_w)

    def linear_attn_layer(
        x, input_weight, qkv_w, z_w, b_w, a_w, conv_w, dt_bias, A_log,
        linear_norm_w, out_w, post_attention_weight, gate_w, up_w, down_w,
        conv_state, recurrent_state, start_pos: Optional[nt.tensor],
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel_size, norm_eps,
    ):
        norm_x = rmsnorm_kernel(x, input_weight, norm_eps)
        h1 = gated_delta_net_kernel(
            norm_x, qkv_w, z_w, b_w, a_w, conv_w, dt_bias, A_log, linear_norm_w,
            out_w, norm_eps, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_kernel_size, conv_state, recurrent_state, start_pos=start_pos,
        )
        z = x + h1
        norm_z = rmsnorm_kernel(z, post_attention_weight, norm_eps)
        ffn_out = dense_ffn(norm_z, gate_w, up_w, down_w)
        # No all-reduce needed for TP=1, but the GDN kernel still has it (identity for ws=1)
        return (z + ffn_out).astype(x.dtype)

    def full_attn_layer(
        x, input_weight, qkv_weight, o_weight, q_norm_weight, k_norm_weight,
        post_attention_weight, gate_w, up_w, down_w,
        cache_k, cache_v,
        start_pos: Optional[nt.tensor],
        num_heads, head_dim, num_kv_heads, norm_eps,
    ):
        norm_x = rmsnorm_kernel(x, input_weight, norm_eps)
        h1 = attention_kernel(
            norm_x, qkv_weight, q_norm_weight, k_norm_weight, norm_eps,
            num_heads, head_dim, num_kv_heads, 0.25, 10000000.0,
            cache_k, cache_v,
            start_pos=start_pos, o_weight=o_weight,
        )
        z = x + h1
        norm_z = rmsnorm_kernel(z, post_attention_weight, norm_eps)
        ffn_out = dense_ffn(norm_z, gate_w, up_w, down_w)
        return (z + ffn_out).astype(x.dtype)

    def compute_logits(h, norm_weight, lm_head_weight, norm_eps):
        """Compute logits (argmax done on CPU to avoid NKIPy argmax bug on large dims)."""
        h = rmsnorm_kernel(h, norm_weight, norm_eps)
        logits = h[:, -1, :] @ lm_head_weight
        return logits.astype(np.float32)

    # --- Prepare device tensors ---
    print("Preparing tensors...")
    layer_types = list(tc.layer_types)
    n_layers = tc.num_hidden_layers
    norm_eps = tc.rms_norm_eps

    n_kv = tc.num_key_value_heads
    n_v = tc.linear_num_value_heads
    n_k = tc.linear_num_key_heads
    hk = tc.linear_key_head_dim
    hv = tc.linear_value_head_dim
    head_dim = tc.head_dim
    max_seq = 4096

    layers = []
    for i in range(n_layers):
        lt = layer_types[i]
        d = {"type": lt}
        for key in ["input_weight", "post_attention_weight", "gate_proj_weight",
                     "up_proj_weight", "down_proj_weight"]:
            d[key] = DeviceTensor.from_torch(weights[f"layers.{i}.{key}"], f"{key}_L{i}")

        if lt == "linear_attention":
            for key in ["linear_qkv_weight", "linear_z_weight", "linear_b_weight",
                         "linear_a_weight", "linear_conv_weight", "linear_dt_bias",
                         "linear_A_log", "linear_norm_weight", "linear_out_weight"]:
                d[key] = DeviceTensor.from_torch(weights[f"layers.{i}.{key}"], f"{key}_L{i}")
            conv_dim = n_k * hk * 2 + n_v * hv
            d["conv_state"] = DeviceTensor.from_numpy(
                np.zeros((1, conv_dim, tc.linear_conv_kernel_dim), dtype=bf16), f"cs_L{i}")
            d["recurrent_state"] = DeviceTensor.from_numpy(
                np.zeros((1, n_v, hk, hv), dtype=bf16), f"rs_L{i}")
        else:
            for key in ["qkv_weight", "o_weight", "q_norm_weight", "k_norm_weight"]:
                d[key] = DeviceTensor.from_torch(weights[f"layers.{i}.{key}"], f"{key}_L{i}")
            d["cache_k"] = DeviceTensor.from_numpy(
                np.zeros((1, max_seq, n_kv, head_dim), dtype=bf16), f"ck_L{i}")
            d["cache_v"] = DeviceTensor.from_numpy(
                np.zeros((1, max_seq, n_kv, head_dim), dtype=bf16), f"cv_L{i}")
        layers.append(d)

    d_norm = DeviceTensor.from_torch(weights["norm_weight"], "norm_w")
    d_lm = DeviceTensor.from_torch(weights["lm_head_weight"], "lm_head")

    # --- Compile kernels ---
    print("Compiling kernels...")

    x_ctx = DeviceTensor.from_numpy(np.empty((1, context_len, tc.hidden_size), dtype=bf16), "x_ctx")
    x_tok = DeviceTensor.from_numpy(np.empty((1, 1, tc.hidden_size), dtype=bf16), "x_tok")
    d_sp = DeviceTensor.from_numpy(np.empty((1,), dtype=np.int32), "sp")

    # Find first of each type
    la_idx = next(i for i, l in enumerate(layers) if l["type"] == "linear_attention")
    fa_idx = next(i for i, l in enumerate(layers) if l["type"] == "full_attention")
    la, fa = layers[la_idx], layers[fa_idx]

    common_la = dict(
        num_k_heads=n_k, num_v_heads=n_v, head_k_dim=hk, head_v_dim=hv,
        conv_kernel_size=tc.linear_conv_kernel_dim, norm_eps=norm_eps,
    )
    common_fa = dict(
        num_heads=tc.num_attention_heads, head_dim=head_dim,
        num_kv_heads=n_kv, norm_eps=norm_eps,
    )

    def make_la_args(x, sp, la):
        return dict(
            x=x, input_weight=la["input_weight"],
            qkv_w=la["linear_qkv_weight"], z_w=la["linear_z_weight"],
            b_w=la["linear_b_weight"], a_w=la["linear_a_weight"],
            conv_w=la["linear_conv_weight"], dt_bias=la["linear_dt_bias"],
            A_log=la["linear_A_log"], linear_norm_w=la["linear_norm_weight"],
            out_w=la["linear_out_weight"], post_attention_weight=la["post_attention_weight"],
            gate_w=la["gate_proj_weight"], up_w=la["up_proj_weight"],
            down_w=la["down_proj_weight"],
            conv_state=la["conv_state"], recurrent_state=la["recurrent_state"],
            start_pos=sp, **common_la,
        )

    def make_fa_args(x, sp, fa):
        return dict(
            x=x, input_weight=fa["input_weight"],
            qkv_weight=fa["qkv_weight"], o_weight=fa["o_weight"],
            q_norm_weight=fa["q_norm_weight"], k_norm_weight=fa["k_norm_weight"],
            post_attention_weight=fa["post_attention_weight"],
            gate_w=fa["gate_proj_weight"], up_w=fa["up_proj_weight"],
            down_w=fa["down_proj_weight"],
            cache_k=fa["cache_k"], cache_v=fa["cache_v"],
            start_pos=sp, **common_fa,
        )

    k_cte_la = DeviceKernel.compile_and_load(linear_attn_layer, name="cte_la",
        build_dir=BUILD_DIR, **make_la_args(x_ctx, None, la))
    k_tkg_la = DeviceKernel.compile_and_load(linear_attn_layer, name="tkg_la",
        build_dir=BUILD_DIR, **make_la_args(x_tok, d_sp, la))
    k_cte_fa = DeviceKernel.compile_and_load(full_attn_layer, name="cte_fa",
        build_dir=BUILD_DIR, **make_fa_args(x_ctx, None, fa))
    k_tkg_fa = DeviceKernel.compile_and_load(full_attn_layer, name="tkg_fa",
        build_dir=BUILD_DIR, **make_fa_args(x_tok, d_sp, fa))
    vocab_size = tc.vocab_size
    d_logits_ctx = DeviceTensor.from_numpy(np.empty((1, vocab_size), dtype=np.float32), "logits_ctx")
    d_logits_tok = DeviceTensor.from_numpy(np.empty((1, vocab_size), dtype=np.float32), "logits_tok")
    k_cte_sample = DeviceKernel.compile_and_load(compute_logits, name="cte_samp",
        h=x_ctx, norm_weight=d_norm, lm_head_weight=d_lm, norm_eps=norm_eps,
        build_dir=BUILD_DIR)
    k_tkg_sample = DeviceKernel.compile_and_load(compute_logits, name="tkg_samp",
        h=x_tok, norm_weight=d_norm, lm_head_weight=d_lm, norm_eps=norm_eps,
        build_dir=BUILD_DIR)

    print(f"Compilation done. Generating {max_new_tokens} tokens...\n")

    # --- Generate ---
    def run_layer(kernel_la, kernel_fa, idx, h, sp_tensor):
        l = layers[idx]
        if l["type"] == "linear_attention":
            inp = {
                "x": h, "input_weight": l["input_weight"],
                "qkv_w": l["linear_qkv_weight"], "z_w": l["linear_z_weight"],
                "b_w": l["linear_b_weight"], "a_w": l["linear_a_weight"],
                "conv_w": l["linear_conv_weight"], "dt_bias": l["linear_dt_bias"],
                "A_log": l["linear_A_log"], "linear_norm_w": l["linear_norm_weight"],
                "out_w": l["linear_out_weight"], "post_attention_weight": l["post_attention_weight"],
                "gate_w": l["gate_proj_weight"], "up_w": l["up_proj_weight"],
                "down_w": l["down_proj_weight"],
                "conv_state.must_alias_input": l["conv_state"],
                "recurrent_state.must_alias_input": l["recurrent_state"],
            }
            if sp_tensor is not None:
                inp["start_pos"] = sp_tensor
            out = {"output0": h, "conv_state": l["conv_state"], "recurrent_state": l["recurrent_state"]}
            kernel_la(inputs=inp, outputs=out)
        else:
            inp = {
                "x": h, "input_weight": l["input_weight"],
                "qkv_weight": l["qkv_weight"], "o_weight": l["o_weight"],
                "q_norm_weight": l["q_norm_weight"], "k_norm_weight": l["k_norm_weight"],
                "post_attention_weight": l["post_attention_weight"],
                "gate_w": l["gate_proj_weight"], "up_w": l["up_proj_weight"],
                "down_w": l["down_proj_weight"],
                "cache_k.must_alias_input": l["cache_k"], "cache_v.must_alias_input": l["cache_v"],
            }
            if sp_tensor is not None:
                inp["start_pos"] = sp_tensor
            out = {"output0": h, "cache_k": l["cache_k"], "cache_v": l["cache_v"]}
            kernel_fa(inputs=inp, outputs=out)

    # Reset GDN states
    for l in layers:
        if l["type"] == "linear_attention":
            l["conv_state"].write_from_numpy(np.zeros(l["conv_state"].numpy().shape, dtype=bf16))
            l["recurrent_state"].write_from_numpy(np.zeros(l["recurrent_state"].numpy().shape, dtype=bf16))

    h = DeviceTensor.from_torch(tok_embedding[input_ids_np], "h")
    next_id = DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "nid")

    # Prefill
    for i in range(n_layers):
        run_layer(k_cte_la, k_cte_fa, i, h, None)
    k_cte_sample(inputs={"h": h, "norm_weight": d_norm, "lm_head_weight": d_lm},
                  outputs={"output0": d_logits_ctx})

    # CPU argmax (NKIPy argmax has bug on large vocab dims)
    logits = d_logits_ctx.numpy().flatten().astype(np.float32)
    first_tid = int(np.argmax(logits))
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"First token: {first_tid} = '{tokenizer.decode([first_tid])}'")
    print(f"Top 5: {[(t, tokenizer.decode([t]), f'{logits[t]:.2f}') for t in top5]}")

    generated = [first_tid]

    # Decode
    start = time.time()
    for pos in range(context_len, context_len + max_new_tokens - 1):
        sp = DeviceTensor.from_numpy(np.array([pos], dtype=np.int32))
        tid = generated[-1]
        if tid >= vocab_size:
            print(f"Warning: token {tid} >= vocab_size {vocab_size}, clamping")
            tid = vocab_size - 1
        nid_torch = torch.tensor([[tid]], dtype=torch.int)
        h = DeviceTensor.from_torch(tok_embedding[nid_torch.numpy()], "h_dec")
        for i in range(n_layers):
            run_layer(k_tkg_la, k_tkg_fa, i, h, sp)
        k_tkg_sample(inputs={"h": h, "norm_weight": d_norm, "lm_head_weight": d_lm},
                      outputs={"output0": d_logits_tok})
        tid = int(np.argmax(d_logits_tok.numpy().flatten().astype(np.float32)))
        generated.append(tid)
        if tid in {248044}:  # EOS
            break

    elapsed = time.time() - start
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"{prompt}{text}")
    print(f"\nTokens: {len(generated)}, Time: {elapsed:.2f}s, Speed: {len(generated)/elapsed:.1f} tok/s")

    # --- Validate against HF ---
    print("\n--- Validating against HF reference ---")
    import transformers.integrations.moe as _m
    _m.is_grouped_mm_available = lambda: False
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cpu")
    hf_model.eval()
    hf_inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        hf_out = hf_model.generate(hf_inputs["input_ids"], max_new_tokens=max_new_tokens, do_sample=False)
    hf_text = tokenizer.decode(hf_out[0][hf_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"HF:    {prompt}{hf_text}")
    print(f"NKIPy: {prompt}{text}")

    # Token-level comparison
    hf_ids = hf_out[0][hf_inputs["input_ids"].shape[1]:].tolist()
    n_match = sum(1 for a, b in zip(generated, hf_ids) if a == b)
    n_total = min(len(generated), len(hf_ids))
    print(f"\nToken match: {n_match}/{n_total} ({100*n_match/max(n_total,1):.0f}%)")
    if n_match == n_total:
        print("PERFECT MATCH!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare weights")
    parser.add_argument("--run", action="store_true", help="Run inference")
    parser.add_argument("prompt", nargs="?", default="who are you? ")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if args.prepare:
        prepare_weights()
    elif args.run:
        run_inference(args.prompt, args.max_new_tokens)
    else:
        print("Use --prepare or --run")
