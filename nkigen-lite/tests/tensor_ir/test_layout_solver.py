"""Baseline tests for layout solver assignments on ML patterns.

Each pattern specifies a "default" layout that most values should have, plus
explicit overrides for values that differ. Format:

    {"default": "ipf", "w": "f", "x": "ifp"}

means: every value should be @ipf, except %w which should be @f and %x which
should be @ifp.

When the solver improves, update the baseline to reflect the new assignment.
"""
from __future__ import annotations


import pytest

from nkigen_lite.tensor_ir.passes.canonicalize import canonicalize
from nkigen_lite.tensor_ir.passes.decompose import decompose
from nkigen_lite.tensor_ir.passes.layout_solver import Layout, solve_graph, _value_shape
from nkigen_lite.tensor_ir.patterns import (
    build_rmsnorm,
    build_softmax,
    build_ffn,
    build_attention,
    build_full_attention,
    build_layernorm,
    build_gqa_attention,
    build_rope,
    build_residual_add,
    build_kv_cache_update,
    build_swiglu_gate,
    build_multi_head_projection,
    build_output_projection,
    build_cross_entropy_loss,
    build_linear_attention_deltanet,
    build_cross_lane_reduce,
    build_fused_scale_bias_activation,
    build_matmul_with_epilogue,
    build_elementwise_rank_change,
    build_elementwise_merge_for_utilization,
    build_elementwise_split_for_batched_mm,
    build_qk_norm,
)


def _layout_str(layout: Layout, shape: tuple[int, ...]) -> str:
    chars = []
    for i in range(len(shape)):
        if i in layout.i_dims:
            chars.append("i")
        elif i in layout.p_dims:
            chars.append("p")
        elif i in layout.f_dims:
            chars.append("f")
        else:
            chars.append("?")
    return "".join(chars)


def _solve_pattern(build_fn) -> dict[str, str]:
    graph = build_fn()
    canonicalize(graph)
    decompose(graph)
    layouts = solve_graph(graph)
    result = {}
    for v in graph.inputs:
        if v.name in layouts:
            result[v.name] = _layout_str(layouts[v.name], _value_shape(v))
    for op in graph.ops:
        for r in op.results:
            if r.name in layouts:
                result[r.name] = _layout_str(layouts[r.name], _value_shape(r))
    return result


# ---------------------------------------------------------------------------
# Golden baselines: {"default": "...", "value_name": "override", ...}
#
# Graph dumps (after canonicalize + decompose) are shown as comments above
# each group so value names/shapes are visible for verifying layout overrides.
# ---------------------------------------------------------------------------

# graph @rmsnorm_(4, 128, 512)(
#   %x: <4x128x512xf32>,
#   %w: <512xf32>,
# ) -> (output: <4x128x512xf32>) {
#   %v1: <4x128x512xf32> = mul(%x, %x)
#   %v8: <4x128x1xf32> = reduce(%v1) {axis=(2,), keepdims=True, kind=sum}
#   %v9: <4x128x1xf32> = constant() {value=0.001953125}
#   %v10: <4x128x1xf32> = mul(%v8, %v9)
#   %v3: <4x128x1xf32> = constant() {value=1e-05}
#   %v4: <4x128x1xf32> = add(%v10, %v3)
#   %v5: <4x128x1xf32> = rsqrt(%v4)
#   %v6: <4x128x512xf32> = mul(%x, %v5)
#   %v7: <4x128x512xf32> = mul(%v6, %w)
#   return output=%v7
# }
RMSNORM_BASELINES = [
    ("rmsnorm_rank2", lambda: build_rmsnorm((128, 512)), {
        "default": "pf",
        "w": "f",
    }),
    ("rmsnorm_rank3", lambda: build_rmsnorm((4, 128, 512)), {
        "default": "ppf",
        "w": "f",
    }),
    ("rmsnorm_rank4", lambda: build_rmsnorm((2, 4, 128, 512)), {
        "default": "pppf",
        "w": "f",
    }),
    ("rmsnorm_rank4_small_p", lambda: build_rmsnorm((2, 4, 8, 512)), {
        "default": "pppf",
        "w": "f",
    }),
]

# graph @layernorm_(4, 128, 512)(
#   %x: <4x128x512xf32>,
#   %gamma: <512xf32>,
#   %beta: <512xf32>,
# ) -> (output: <4x128x512xf32>) {
#   %v11: <4x128x1xf32> = reduce(%x) {axis=(2,), keepdims=True, kind=sum}
#   %v12: <4x128x1xf32> = constant() {value=0.001953125}
#   %v13: <4x128x1xf32> = mul(%v11, %v12)
#   %v2: <4x128x512xf32> = sub(%x, %v13)
#   %v3: <4x128x512xf32> = mul(%v2, %v2)
#   %v14: <4x128x1xf32> = reduce(%v3) {axis=(2,), keepdims=True, kind=sum}
#   %v15: <4x128x1xf32> = constant() {value=0.001953125}
#   %v16: <4x128x1xf32> = mul(%v14, %v15)
#   %v5: <4x128x1xf32> = constant() {value=1e-05}
#   %v6: <4x128x1xf32> = add(%v16, %v5)
#   %v7: <4x128x1xf32> = rsqrt(%v6)
#   %v8: <4x128x512xf32> = mul(%v2, %v7)
#   %v9: <4x128x512xf32> = mul(%v8, %gamma)
#   %output_out: <4x128x512xf32> = add(%v9, %beta)
#   return output=%output_out
# }
LAYERNORM_BASELINES = [
    ("layernorm_rank2", lambda: build_layernorm((128, 512)), {
        "default": "pf",
        "gamma": "f",
        "beta": "f",
    }),
    ("layernorm_rank3", lambda: build_layernorm((4, 128, 512)), {
        "default": "ppf",
        "gamma": "f",
        "beta": "f",
    }),
    ("layernorm_rank4", lambda: build_layernorm((2, 4, 128, 512)), {
        "default": "pppf",
        "gamma": "f",
        "beta": "f",
    }),
    ("layernorm_rank4_small_p", lambda: build_layernorm((2, 4, 8, 512)), {
        "default": "pppf",
        "gamma": "f",
        "beta": "f",
    }),
]

# graph @softmax_(4, 128, 512)(
#   %x: <4x128x512xf32>,
# ) -> (probs: <4x128x512xf32>) {
#   %v1: <4x128x1xf32> = reduce(%x) {axis=(2,), keepdims=True, kind=max}
#   %v2: <4x128x512xf32> = sub(%x, %v1)
#   %v3: <4x128x512xf32> = exp(%v2)
#   %v4: <4x128x1xf32> = reduce(%v3) {axis=(2,), keepdims=True, kind=sum}
#   %v5: <4x128x1xf32> = reciprocal(%v4)
#   %v6: <4x128x512xf32> = mul(%v3, %v5)
#   return probs=%v6
# }
SOFTMAX_BASELINES = [
    ("softmax_rank2", lambda: build_softmax((128, 512)), {
        "default": "pf",
    }),
    ("softmax_rank3", lambda: build_softmax((4, 128, 512)), {
        "default": "ppf",
    }),
    ("softmax_rank4", lambda: build_softmax((2, 4, 128, 512)), {
        "default": "pppf",
    }),
]

# graph @ce_loss_B2_S64_V1024(
#   %logits: <2x64x1024xf32>,
# ) -> (log_softmax: <2x64x1024xf32>) {
#   %v1: <2x64x1xf32> = reduce(%logits) {axis=(2,), keepdims=True, kind=max}
#   %v2: <2x64x1024xf32> = sub(%logits, %v1)
#   %v3: <2x64x1024xf32> = exp(%v2)
#   %v4: <2x64x1xf32> = reduce(%v3) {axis=(2,), keepdims=True, kind=sum}
#   %v5: <2x64x1xf32> = log(%v4)
#   %log_softmax_out: <2x64x1024xf32> = sub(%v2, %v5)
#   return log_softmax=%log_softmax_out
# }
CROSS_ENTROPY_BASELINES = [
    ("cross_entropy_loss", lambda: build_cross_entropy_loss(2, 64, 1024), {
        "default": "ppf",
    }),
]

# graph @ffn_(2, 64, 256)(
#   %x: <2x64x256xf32>,
#   %gate_up_w: <256x1024xf32>,
#   %down_w: <512x256xf32>,
# ) -> (output: <2x64x256xf32>) {
#   %mm_gate_up_out: <2x64x1024xf32> = matmul(%x, %gate_up_w)
#   %v2: <2x64x512xf32> = slice(%mm_gate_up_out) {starts=(0, 0, 0), stops=(2, 64, 512)}
#   %v3: <2x64x512xf32> = slice(%mm_gate_up_out) {starts=(0, 0, 512), stops=(2, 64, 1024)}
#   %v4: <2x64x512xf32> = sigmoid(%v2)
#   %v5: <2x64x512xf32> = mul(%v2, %v4)
#   %v6: <2x64x512xf32> = mul(%v5, %v3)
#   %v7: <2x64x256xf32> = matmul(%v6, %down_w)
#   return output=%v7
# }
FFN_BASELINES = [
    ("ffn_rank2", lambda: build_ffn((64, 256), intermediate=512), {
        "default": "pf",
        "x": "fp",
        "v6": "fp",
    }),
    ("ffn_rank3", lambda: build_ffn((2, 64, 256), intermediate=512), {
        "default": "ipf",
        "x": "ifp",
        "gate_up_w": "pf",
        "down_w": "pf",
        "v6": "ifp",
    }),
]

# graph @swiglu_(2, 64, 256)_I512(
#   %x: <2x64x256xf32>,
#   %W_gate: <256x512xf32>,
#   %W_up: <256x512xf32>,
# ) -> (gated: <2x64x512xf32>) {
#   %gate_proj_out: <2x64x512xf32> = matmul(%x, %W_gate)
#   %v2: <2x64x512xf32> = matmul(%x, %W_up)
#   %v3: <2x64x512xf32> = sigmoid(%gate_proj_out)
#   %v4: <2x64x512xf32> = mul(%gate_proj_out, %v3)
#   %v5: <2x64x512xf32> = mul(%v4, %v2)
#   return gated=%v5
# }
SWIGLU_BASELINES = [
    ("swiglu_rank2", lambda: build_swiglu_gate((64, 256), intermediate=512), {
        "default": "pf",
        "x": "fp",
    }),
    ("swiglu_rank3", lambda: build_swiglu_gate((2, 64, 256), intermediate=512), {
        "default": "ipf",
        "x": "ifp",
        "W_gate": "pf",
        "W_up": "pf",
    }),
]

# graph @attention_(2, 8, 64, 64)(
#   %q: <2x8x64x64xf32>,
#   %k: <2x8x64x64xf32>,
#   %v: <2x8x64x64xf32>,
# ) -> (output: <2x8x64x64xf32>) {
#   %v1: <2x8x64x64xf32> = transpose(%k) {perm=(0, 1, 3, 2)}
#   %scores_out: <2x8x64x64xf32> = matmul(%q, %v1)
#   %v3: <2x8x64x64xf32> = mul(%scores_out, %scores_out)
#   %v4: <2x8x64x1xf32> = reduce(%v3) {axis=(3,), keepdims=True, kind=max}
#   %v5: <2x8x64x64xf32> = sub(%v3, %v4)
#   %v6: <2x8x64x64xf32> = exp(%v5)
#   %v7: <2x8x64x1xf32> = reduce(%v6) {axis=(3,), keepdims=True, kind=sum}
#   %v8: <2x8x64x1xf32> = reciprocal(%v7)
#   %v9: <2x8x64x64xf32> = mul(%v6, %v8)
#   %output_out: <2x8x64x64xf32> = matmul(%v9, %v)
#   return output=%output_out
# }
ATTENTION_BASELINES = [
    ("attention_rank3", lambda: build_attention((4, 32, 64)), {
        "default": "ipf",
        "q": "ifp",
        "k": "ifp",
        "v9": "ifp",
    }),
    ("attention_rank4", lambda: build_attention((2, 8, 64, 64)), {
        "default": "iipf",
        "q": "iifp",
        "k": "iifp",
        "v9": "iifp",
    }),
]

# graph @full_mha_B2_S64_D256_H8(
#   %x: <2x64x256xf32>,
#   %W_qkv: <256x768xf32>,
#   %W_o: <256x256xf32>,
# ) -> (output: <2x64x256xf32>) {
#   %qkv_proj: <2x64x768xf32> = matmul(%x, %W_qkv)
#   %v2: <2x64x256xf32> = slice(%qkv_proj) {starts=(0, 0, 0), stops=(2, 64, 256)}
#   %v3: <2x64x256xf32> = slice(%qkv_proj) {starts=(0, 0, 256), stops=(2, 64, 512)}
#   %v4: <2x64x256xf32> = slice(%qkv_proj) {starts=(0, 0, 512), stops=(2, 64, 768)}
#   %v5: <2x64x8x32xf32> = reshape(%v2) {shape=(2, 64, 8, 32)}
#   %q_heads: <2x8x64x32xf32> = transpose(%v5) {perm=(0, 2, 1, 3)}
#   %v7: <2x64x8x32xf32> = reshape(%v3) {shape=(2, 64, 8, 32)}
#   %k_heads: <2x8x64x32xf32> = transpose(%v7) {perm=(0, 2, 1, 3)}
#   %v9: <2x64x8x32xf32> = reshape(%v4) {shape=(2, 64, 8, 32)}
#   %v_heads: <2x8x64x32xf32> = transpose(%v9) {perm=(0, 2, 1, 3)}
#   %v11: <2x8x32x64xf32> = transpose(%k_heads) {perm=(0, 1, 3, 2)}
#   %attn_scores: <2x8x64x64xf32> = matmul(%q_heads, %v11)
#   %v13: <2x8x64x1xf32> = reduce(%attn_scores) {axis=(3,), keepdims=True, kind=max}
#   %v14: <2x8x64x64xf32> = sub(%attn_scores, %v13)
#   %v15: <2x8x64x64xf32> = exp(%v14)
#   %v16: <2x8x64x1xf32> = reduce(%v15) {axis=(3,), keepdims=True, kind=sum}
#   %v17: <2x8x64x1xf32> = reciprocal(%v16)
#   %attn_probs: <2x8x64x64xf32> = mul(%v15, %v17)
#   %attn_out: <2x8x64x32xf32> = matmul(%attn_probs, %v_heads)
#   %v20: <2x64x8x32xf32> = transpose(%attn_out) {perm=(0, 2, 1, 3)}
#   %attn_flat: <2x64x256xf32> = reshape(%v20) {shape=(2, 64, 256)}
#   %out_proj: <2x64x256xf32> = matmul(%attn_flat, %W_o)
#   return output=%out_proj
# }
FULL_ATTENTION_BASELINES = [
    ("full_attention", lambda: build_full_attention(B=2, S=64, D=256, H=8), {
        "default": "iipf",
        "x": "ifp",
        "W_qkv": "pf",
        "W_o": "pf",
        "qkv_proj": "ipf",
        "v2": "ipf",
        "v3": "ipf",
        "v4": "ipf",
        "v5": "ipff",
        "v7": "ipff",
        "v9": "ipff",
        "q_heads": "iifp",
        "k_heads": "iifp",
        "attn_probs": "iifp",
        "v20": "?",
        "attn_flat": "ifp",
        "out_proj": "ipf",
    }),
]

# graph @gqa_B1_Hq8_Hkv2_S64_D64(
#   %q: <1x8x64x64xf32>,
#   %k: <1x2x64x64xf32>,
#   %v: <1x2x64x64xf32>,
# ) -> (output: <1x8x64x64xf32>) {
#   %v1: <1x2x1x64x64xf32> = reshape(%k) {shape=(1, 2, 1, 64, 64)}
#   %v2: <1x2x4x64x64xf32> = broadcast_to(%v1) {shape=(1, 2, 4, 64, 64)}
#   %v3: <1x8x64x64xf32> = reshape(%v2) {shape=(1, 8, 64, 64)}
#   %v4: <1x2x1x64x64xf32> = reshape(%v) {shape=(1, 2, 1, 64, 64)}
#   %v5: <1x2x4x64x64xf32> = broadcast_to(%v4) {shape=(1, 2, 4, 64, 64)}
#   %v6: <1x8x64x64xf32> = reshape(%v5) {shape=(1, 8, 64, 64)}
#   %v7: <1x8x64x64xf32> = transpose(%v3) {perm=(0, 1, 3, 2)}
#   %scores_out: <1x8x64x64xf32> = matmul(%q, %v7)
#   %v9: <1x8x64x1xf32> = reduce(%scores_out) {axis=(3,), keepdims=True, kind=max}
#   %v10: <1x8x64x64xf32> = sub(%scores_out, %v9)
#   %v11: <1x8x64x64xf32> = exp(%v10)
#   %v12: <1x8x64x1xf32> = reduce(%v11) {axis=(3,), keepdims=True, kind=sum}
#   %v13: <1x8x64x1xf32> = reciprocal(%v12)
#   %v14: <1x8x64x64xf32> = mul(%v11, %v13)
#   %output_out: <1x8x64x64xf32> = matmul(%v14, %v6)
#   return output=%output_out
# }
GQA_BASELINES = [
    ("gqa_attention", lambda: build_gqa_attention(B=1, H_q=8, H_kv=2, S=64, D=64), {
        "default": "iipf",
        "q": "iifp",
        "k": "iifp",
        "v1": "iiifp",
        "v2": "iiifp",
        "v3": "iifp",
        "v4": "iiipf",
        "v5": "iiipf",
        "v14": "iifp",
    }),
]

# graph @rope_(2, 8, 64, 64)(
#   %x: <2x8x64x64xf32>,
#   %cos: <2x8x64x32xf32>,
#   %sin: <2x8x64x32xf32>,
# ) -> (rope: <2x8x64x64xf32>) {
#   %v1: <2x8x64x32xf32> = slice(%x) {starts=(0, 0, 0, 0), stops=(2, 8, 64, 32)}
#   %v2: <2x8x64x32xf32> = slice(%x) {starts=(0, 0, 0, 32), stops=(2, 8, 64, 64)}
#   %v3: <2x8x64x32xf32> = mul(%v1, %cos)
#   %v4: <2x8x64x32xf32> = mul(%v2, %sin)
#   %v5: <2x8x64x32xf32> = sub(%v3, %v4)
#   %v6: <2x8x64x32xf32> = mul(%v2, %cos)
#   %v7: <2x8x64x32xf32> = mul(%v1, %sin)
#   %v8: <2x8x64x32xf32> = add(%v6, %v7)
#   %concat_rope_out: <2x8x64x64xf32> = concat(%v5, %v8) {axis=-1}
#   return rope=%concat_rope_out
# }
ROPE_BASELINES = [
    ("rope_rank3", lambda: build_rope((4, 64, 64)), {
        "default": "ppf",
    }),
    ("rope_rank4", lambda: build_rope((2, 8, 64, 64)), {
        "default": "ppff",
    }),
]

# graph @residual_(2, 64, 256)(
#   %x: <2x64x256xf32>,
#   %W: <256x256xf32>,
# ) -> (residual: <2x64x256xf32>) {
#   %v1: <2x64x256xf32> = matmul(%x, %W)
#   %v2: <2x64x256xf32> = gelu(%v1)
#   %residual_add_out: <2x64x256xf32> = add(%x, %v2)
#   return residual=%residual_add_out
# }
RESIDUAL_BASELINES = [
    ("residual_rank2", lambda: build_residual_add((64, 256)), {
        "default": "pf",
        "x": "fp",
    }),
    ("residual_rank3", lambda: build_residual_add((2, 64, 256)), {
        "default": "ipf",
        "x": "ifp",
        "W": "pf",
    }),
]

# graph @mhp_B2_S64_D256_H8(
#   %x: <2x64x256xf32>,
#   %W_qkv: <256x768xf32>,
# ) -> (q: <2x8x64x32xf32>, k: <2x8x64x32xf32>, v: <2x8x64x32xf32>) {
#   %qkv_proj_out: <2x64x768xf32> = matmul(%x, %W_qkv)
#   %v2: <2x64x256xf32> = slice(%qkv_proj_out) {starts=(0, 0, 0), stops=(2, 64, 256)}
#   %v3: <2x64x256xf32> = slice(%qkv_proj_out) {starts=(0, 0, 256), stops=(2, 64, 512)}
#   %v4: <2x64x256xf32> = slice(%qkv_proj_out) {starts=(0, 0, 512), stops=(2, 64, 768)}
#   %v5: <2x64x8x32xf32> = reshape(%v2) {shape=(2, 64, 8, 32)}
#   %q_reshape_out: <2x8x64x32xf32> = transpose(%v5) {perm=(0, 2, 1, 3)}
#   %v7: <2x64x8x32xf32> = reshape(%v3) {shape=(2, 64, 8, 32)}
#   %v8: <2x8x64x32xf32> = transpose(%v7) {perm=(0, 2, 1, 3)}
#   %v9: <2x64x8x32xf32> = reshape(%v4) {shape=(2, 64, 8, 32)}
#   %v10: <2x8x64x32xf32> = transpose(%v9) {perm=(0, 2, 1, 3)}
#   return q=%q_reshape_out, k=%v8, v=%v10
# }
MULTI_HEAD_PROJECTION_BASELINES = [
    ("multi_head_projection", lambda: build_multi_head_projection(B=2, S=64, D=256, H=8), {
        "default": "ipff",
        "x": "ifp",
        "W_qkv": "pf",
        "qkv_proj_out": "ipf",
        "v2": "ipf",
        "v3": "ipf",
        "v4": "ipf",
        "q_reshape_out": "ppff",
        "v8": "ppff",
        "v10": "ppff",
    }),
]

# graph @out_proj_B2_H8_S64_Dh32_D256(
#   %attn_out: <2x8x64x32xf32>,
#   %W_o: <256x256xf32>,
# ) -> (output: <2x64x256xf32>) {
#   %v1: <2x64x256xf32> = reshape(%attn_out) {shape=(2, 64, 256)}
#   %out_proj_out: <2x64x256xf32> = matmul(%v1, %W_o)
#   return output=%out_proj_out
# }
OUTPUT_PROJECTION_BASELINES = [
    ("output_projection", lambda: build_output_projection(B=2, H=8, S=64, D_h=32, D=256), {
        "default": "ipf",
        "attn_out": "ppff",
        "W_o": "pf",
        "v1": "ifp",
    }),
]

# graph @kv_cache_B1_H8_S128+16_D64(
#   %cached_k: <1x8x128x64xf32>,
#   %new_k: <1x8x16x64xf32>,
# ) -> (kv_concat: <1x8x144x64xf32>) {
#   %v1: <1x8x144x64xf32> = concat(%cached_k, %new_k) {axis=2}
#   return kv_concat=%v1
# }
KV_CACHE_BASELINES = [
    ("kv_cache_update", lambda: build_kv_cache_update(B=1, H=8, S_cached=128, S_new=16, D=64), {
        "default": "pppf",
    }),
]

# graph @fused_scale_bias_act_(4, 128, 256)(
#   %x: <4x128x256xf32>,
#   %scale: <256xf32>,
#   %bias: <256xf32>,
# ) -> (activated: <4x128x256xf32>) {
#   %v1: <4x128x256xf32> = mul(%x, %scale)
#   %v2: <4x128x256xf32> = add(%v1, %bias)
#   %v3: <4x128x256xf32> = gelu(%v2)
#   return activated=%v3
# }
FUSED_SCALE_BIAS_BASELINES = [
    ("fused_scale_bias_rank2", lambda: build_fused_scale_bias_activation((128, 256)), {
        "default": "pf",
        "scale": "f",
        "bias": "f",
    }),
    ("fused_scale_bias_rank3", lambda: build_fused_scale_bias_activation((4, 128, 256)), {
        "default": "ppf",
        "scale": "f",
        "bias": "f",
    }),
]

# graph @matmul_epilogue_(4, 128, 256)(
#   %x: <4x128x256xf32>,
#   %W: <256x256xf32>,
#   %bias: <256xf32>,
# ) -> (output: <4x128x256xf32>) {
#   %linear_out: <4x128x256xf32> = matmul(%x, %W)
#   %v2: <4x128x256xf32> = add(%linear_out, %bias)
#   %relu_out: <4x128x256xf32> = relu(%v2)
#   return output=%relu_out
# }
MATMUL_EPILOGUE_BASELINES = [
    ("matmul_epilogue_rank2", lambda: build_matmul_with_epilogue((128, 256), N=512), {
        "default": "pf",
        "x": "fp",
        "bias": "f",
    }),
    ("matmul_epilogue_rank3", lambda: build_matmul_with_epilogue((4, 128, 256), N=256), {
        "default": "ipf",
        "x": "ifp",
        "W": "pf",
        "bias": "f",
    }),
]

# graph @cross_lane_reduce_(8, 128, 512)(
#   %x: <8x128x512xf32>,
# ) -> (p_reduce: <1x128x512xf32>) {
#   %v1: <1x128x512xf32> = reduce(%x) {axis=(0,), keepdims=True, kind=sum}
#   return p_reduce=%v1
# }
CROSS_LANE_REDUCE_BASELINES = [
    ("cross_lane_reduce_rank2", lambda: build_cross_lane_reduce((128, 512)), {
        "default": "pf",
    }),
    ("cross_lane_reduce_rank3", lambda: build_cross_lane_reduce((8, 128, 512)), {
        "default": "ppf",
    }),
]

# graph @deltanet_B1_H4_L64_D32(
#   %q: <1x4x64x32xf32>,
#   %k: <1x4x64x32xf32>,
#   %v: <1x4x64x32xf32>,
#   %beta_logits: <1x4x64xf32>,
# ) -> (qkv_interact: <1x4x64x32xf32>) {
#   %v5: <1x4x64x1xf32> = reshape(%beta_logits) {shape=(1, 4, 64, 1)}
#   %v6: <1x4x64x1xf32> = sigmoid(%v5)
#   %v7: <1x4x64x32xf32> = mul(%v, %v6)
#   %v8: <1x4x64x32xf32> = mul(%q, %v7)
#   return qkv_interact=%v8
# }
DELTANET_BASELINES = [
    ("deltanet", lambda: build_linear_attention_deltanet(), {
        "default": "pppf",
        "beta_logits": "ppf",
    }),
    ("deltanet_alt", lambda: build_linear_attention_deltanet(B=2, H=8, L=32, D=64), {
        "default": "ppff",
        "beta_logits": "ppf",
    }),
]

# graph @elementwise_rank_change(
#   %x: <2x64x128xf32>,
#   %W: <128x256xf32>,
#   %V: <2x256x32xf32>,
# ) -> (output: <2x64x32xf32>) {
#   %v1: <2x64x256xf32> = matmul(%x, %W)
#   %v2: <2x64x256xf32> = relu(%v1)
#   %v3: <2x64x256xf32> = mul(%v2, %v2)
#   %v4: <2x64x32xf32> = matmul(%v3, %V)
#   return output=%v4
# }
ELEMENTWISE_RANK_CHANGE_BASELINES = [
    ("elementwise_rank_change", lambda: build_elementwise_rank_change(), {
        "default": "ipf",
        "x": "ifp",
        "W": "pf",
        "v2": "ipf",
        "v3": "ifp",
    }),
]

# graph @elementwise_merge_utilization(
#   %x: <4x32x64xf32>,
#   %W: <64x128xf32>,
#   %bias: <128xf32>,
#   %scale: <128xf32>,
#   %W2: <128x64xf32>,
# ) -> (output: <4x32x64xf32>) {
#   %v1: <4x32x128xf32> = matmul(%x, %W)
#   %v2: <4x32x128xf32> = gelu(%v1)
#   %v3: <4x32x128xf32> = add(%v2, %bias)
#   %v4: <4x32x128xf32> = mul(%v3, %scale)
#   %v5: <4x32x128xf32> = relu(%v4)
#   %v6: <4x32x64xf32> = matmul(%v5, %W2)
#   return output=%v6
# }
ELEMENTWISE_MERGE_BASELINES = [
    ("elementwise_merge", lambda: build_elementwise_merge_for_utilization(), {
        "default": "ipf",
        "x": "ifp",
        "W": "pf",
        "bias": "f",
        "scale": "f",
        "W2": "pf",
        "v5": "ifp",
    }),
]

# graph @elementwise_split_for_batched(
#   %x: <128x128xf32>,
#   %W: <128x64xf32>,
#   %K: <2x64x32xf32>,
# ) -> (output: <2x64x32xf32>) {
#   %v1: <128x64xf32> = matmul(%x, %W)
#   %v2: <2x64x64xf32> = reshape(%v1) {shape=(2, 64, 64)}
#   %v3: <2x64x64xf32> = relu(%v2)
#   %v4: <2x64x64xf32> = gelu(%v3)
#   %v5: <2x64x32xf32> = matmul(%v4, %K)
#   return output=%v5
# }
ELEMENTWISE_SPLIT_BASELINES = [
    ("elementwise_split", lambda: build_elementwise_split_for_batched_mm(), {
        "x": "fp",
        "W": "pf",
        "K": "ipf",
        "v1": "pf",
        "v2": "ppf",
        "v3": "ppf",
        "v4": "ifp",
        "v5": "ipf",
    }),
    ("elementwise_split_alt", lambda: build_elementwise_split_for_batched_mm(S=64, D=64, N=32, B_out=2, O=16), {
        "default": "pf",
        "x": "fp",
        "K": "ipf",
        "v2": "ppf",
        "v3": "ppf",
        "v4": "ifp",
        "v5": "ipf",
    }),
]

# graph @qk_norm_B1_S32_H4_D64(
#   %q: <1x32x4x64xf32>,
#   %k: <1x32x4x64xf32>,
#   %q_norm_w: <64xf32>,
#   %k_norm_w: <64xf32>,
# ) -> (q_normed: <1x32x4x64xf32>, k_normed: <1x32x4x64xf32>) {
#   %v1: <1x32x4x64xf32> = mul(%q, %q)
#   %v15: <1x32x4x1xf32> = reduce(%v1) {axis=(3,), keepdims=True, kind=sum}
#   %v16: <1x32x4x1xf32> = constant() {value=0.015625}
#   %v17: <1x32x4x1xf32> = mul(%v15, %v16)
#   %v3: <1x32x4x1xf32> = constant() {value=1e-05}
#   %v4: <1x32x4x1xf32> = add(%v17, %v3)
#   %v5: <1x32x4x1xf32> = rsqrt(%v4)
#   %v6: <1x32x4x64xf32> = mul(%q, %v5)
#   %v7: <1x32x4x64xf32> = mul(%v6, %q_norm_w)
#   %v8: <1x32x4x64xf32> = mul(%k, %k)
#   %v18: <1x32x4x1xf32> = reduce(%v8) {axis=(3,), keepdims=True, kind=sum}
#   %v19: <1x32x4x1xf32> = constant() {value=0.015625}
#   %v20: <1x32x4x1xf32> = mul(%v18, %v19)
#   %v10: <1x32x4x1xf32> = constant() {value=1e-05}
#   %v11: <1x32x4x1xf32> = add(%v20, %v10)
#   %v12: <1x32x4x1xf32> = rsqrt(%v11)
#   %v13: <1x32x4x64xf32> = mul(%k, %v12)
#   %v14: <1x32x4x64xf32> = mul(%v13, %k_norm_w)
#   return q_normed=%v7, k_normed=%v14
# }
QK_NORM_BASELINES = [
    ("qk_norm", lambda: build_qk_norm(1, 32, 4, 64), {
        "default": "pppf",
        "q_norm_w": "f",
        "k_norm_w": "f",
    }),
    ("qk_norm_alt", lambda: build_qk_norm(2, 64, 8, 128), {
        "default": "ppff",
        "q_norm_w": "f",
        "k_norm_w": "f",
    }),
]

# ---------------------------------------------------------------------------
# Aggregated baseline list
# ---------------------------------------------------------------------------

BASELINES = (
    RMSNORM_BASELINES
    + LAYERNORM_BASELINES
    + SOFTMAX_BASELINES
    + CROSS_ENTROPY_BASELINES
    + FFN_BASELINES
    + SWIGLU_BASELINES
    + ATTENTION_BASELINES
    + FULL_ATTENTION_BASELINES
    + GQA_BASELINES
    + ROPE_BASELINES
    + RESIDUAL_BASELINES
    + MULTI_HEAD_PROJECTION_BASELINES
    + OUTPUT_PROJECTION_BASELINES
    + KV_CACHE_BASELINES
    + FUSED_SCALE_BIAS_BASELINES
    + MATMUL_EPILOGUE_BASELINES
    + CROSS_LANE_REDUCE_BASELINES
    + DELTANET_BASELINES
    + ELEMENTWISE_RANK_CHANGE_BASELINES
    + ELEMENTWISE_MERGE_BASELINES
    + ELEMENTWISE_SPLIT_BASELINES
    + QK_NORM_BASELINES
)


class TestLayoutBaseline:
    """Verify layout solver produces expected assignments for each pattern.

    When the solver improves, update the baseline dicts above.
    """

    @pytest.mark.parametrize(
        "name,build_fn,expected",
        BASELINES,
        ids=[b[0] for b in BASELINES],
    )
    def test_layout_assignment(self, name, build_fn, expected):
        actual = _solve_pattern(build_fn)
        default = expected.get("default")
        overrides = {k: v for k, v in expected.items() if k != "default"}

        # Values marked "?" should be opaque (no layout assigned)
        opaque_values = {k for k, v in overrides.items() if v == "?"}

        for value_name, actual_layout in actual.items():
            if value_name in opaque_values:
                pytest.fail(
                    f"{name}: '{value_name}' should be opaque (no layout), "
                    f"but got @{actual_layout}"
                )
            elif value_name in overrides:
                assert actual_layout == overrides[value_name], (
                    f"{name}: '{value_name}' expected @{overrides[value_name]}, "
                    f"got @{actual_layout}"
                )
            elif default is not None:
                assert actual_layout == default, (
                    f"{name}: '{value_name}' expected default @{default}, "
                    f"got @{actual_layout}"
                )

        # Verify all non-opaque overrides reference real values
        for value_name in overrides:
            if value_name in opaque_values:
                assert value_name not in actual, (
                    f"{name}: '{value_name}' should be opaque but got layout "
                    f"@{actual.get(value_name)}"
                )
            else:
                assert value_name in actual, (
                    f"{name}: override '{value_name}' not found in solver output. "
                    f"Available: {sorted(actual.keys())}"
                )
