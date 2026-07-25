"""Runtime surface assembly for DSV4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nkipy_serving.models.deepseek_v4.assembly.topology import (
    _default_v4_tp_groups,
)
from nkipy_serving.models.deepseek_v4.eager_ops import precompute_freqs_cis_yarn


def _required(obj: Any, name: str, *, ctx: str) -> Any:
    value = getattr(obj, name, None) if obj is not None else None
    if value is None:
        raise RuntimeError(f"{ctx} missing required field {name!r}")
    return value


@dataclass(frozen=True)
class Dsv4RuntimeArgs:
    dim: int
    n_heads: int
    n_hash_layers: int
    n_routed_experts: int
    n_activated_experts: int
    n_shared_experts: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    head_dim: int
    rope_head_dim: int
    window_size: int
    compress_ratios: tuple[int, ...]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    norm_eps: float
    swiglu_limit: float
    routed_scaling_factor: float
    scoring_func: str
    topk_method: str
    rope_theta: float
    compress_rope_theta: float
    rope_scaling_factor: float
    original_seq_len: int
    beta_fast: int
    beta_slow: int
    vocab_size: int
    moe_inter_dim: int
    tp_degree: int
    tp_rank: int
    tp_replica_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Dsv4CompressorSurface:
    dim: int
    head_dim: int
    rope_head_dim: int
    compress_ratio: int
    overlap: bool
    rotate: bool
    eps: float
    wkv: Any
    wgate: Any
    ape: Any
    norm_weight: Any
    freqs_cis: np.ndarray
    freqs_cos: np.ndarray
    freqs_sin: np.ndarray


@dataclass(frozen=True)
class Dsv4IndexerSurface:
    n_heads: int
    head_dim: int
    rope_head_dim: int
    compress_ratio: int
    index_topk: int
    wq_b: Any
    weights_proj: Any
    compressor: Dsv4CompressorSurface
    softmax_scale: float


@dataclass(frozen=True)
class Dsv4AttentionSurface:
    args: Dsv4RuntimeArgs
    layer_id: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    n_groups: int
    window_size: int
    compress_ratio: int
    eps: float
    softmax_scale: float
    tp_degree: int
    tp_rank: int
    tp_replica_groups: tuple[tuple[int, ...], ...]
    attn_sink: Any
    wq_a: Any
    q_norm: Any
    wq_b: Any
    wkv: Any
    kv_norm: Any
    wo_a: Any
    wo_b: Any
    compressor: Dsv4CompressorSurface | None
    indexer: Dsv4IndexerSurface | None
    freqs_cis: np.ndarray
    freqs_cos: np.ndarray
    freqs_sin: np.ndarray


@dataclass(frozen=True)
class Dsv4GateSurface:
    is_hash: bool
    topk: int
    route_scale: float
    score_func: str
    weight: Any
    bias: Any | None
    tid2eid: Any | None


@dataclass(frozen=True)
class Dsv4FfnSurface:
    layer_id: int
    dim: int
    n_routed_experts: int
    gate: Dsv4GateSurface
    experts: tuple[Any, ...]
    shared: Any


@dataclass(frozen=True)
class Dsv4LayerSurface:
    args: Dsv4RuntimeArgs
    layer_id: int
    attn: Dsv4AttentionSurface
    ffn: Dsv4FfnSurface
    attn_norm: Any
    ffn_norm: Any
    hc_attn_fn: Any
    hc_attn_scale: Any
    hc_attn_base: Any
    hc_ffn_fn: Any
    hc_ffn_scale: Any
    hc_ffn_base: Any


@dataclass(frozen=True)
class Dsv4HeadSurface:
    lm_head: Any
    final_norm: Any
    hc_head_fn: Any
    hc_head_scale: Any
    hc_head_base: Any
    norm_eps: float
    hc_eps: float


@dataclass(frozen=True)
class Dsv4WeightsSurface:
    embed: Any
    final_norm: Any
    lm_head: Any


@dataclass(frozen=True)
class Dsv4RuntimeSurface:
    model_config: Any
    v4: Any
    w: Dsv4WeightsSurface
    max_batch_size: int
    max_seq_len: int
    args: Dsv4RuntimeArgs
    blocks: tuple[Dsv4LayerSurface, ...]
    head: Dsv4HeadSurface
    _dsv4_source_runtime: Any | None
    _dsv4_device_weights: Any
    _dsv4_freqs_cis_full: np.ndarray
    _dsv4_freqs_cis_compressed: np.ndarray


class _SampledSharedExpertSurface:
    """Shared-expert handle for the pure sampled surface.

    Product blockwise serving carries BF16 ``shared_w{1,2,3}`` tensors and
    uses the normal ``shared_expert_add`` trace function.
    """

    def __init__(self, layer_weights: Any) -> None:
        ctx = "DSV4 sampled shared expert"
        self._w1 = getattr(layer_weights, "shared_w1", None)
        self._w2 = getattr(layer_weights, "shared_w2", None)
        self._w3 = getattr(layer_weights, "shared_w3", None)
        self.tp_sharded = bool(getattr(layer_weights, "shared_tp_sharded", False))
        has_bf16 = all(v is not None for v in (self._w1, self._w2, self._w3))
        if not has_bf16:
            raise RuntimeError(
                f"{ctx} missing required BF16 shared_w1/shared_w2/shared_w3"
            )
        # Upstream DSV4 applies swiglu_limit only to routed experts; the shared
        # expert is constructed without a limit.
        self.swiglu_limit = 0.0

    @property
    def w1(self) -> Any:
        if self._w1 is None:
            raise RuntimeError("DSV4 sampled shared expert has no BF16 w1")
        return self._w1

    @property
    def w2(self) -> Any:
        if self._w2 is None:
            raise RuntimeError("DSV4 sampled shared expert has no BF16 w2")
        return self._w2

    @property
    def w3(self) -> Any:
        if self._w3 is None:
            raise RuntimeError("DSV4 sampled shared expert has no BF16 w3")
        return self._w3


def _build_runtime_args(v4: Any) -> Dsv4RuntimeArgs:
    n_heads = int(getattr(v4, "local_num_attention_heads", v4.num_attention_heads))
    tp_degree = int(getattr(v4, "tp_degree", 1))
    tp_rank = int(getattr(v4, "tp_rank", 0))
    o_groups = int(v4.o_groups)
    if tp_degree > 1:
        if o_groups % tp_degree != 0:
            raise RuntimeError(
                "Pure-TP DSV4 attention requires whole output groups per TP rank: "
                f"o_groups={o_groups}, tp_degree={tp_degree}"
            )
        o_groups = o_groups // tp_degree
    index_n_heads = int(getattr(v4, "index_n_heads"))
    if index_n_heads == int(
        getattr(v4, "num_attention_heads", index_n_heads)
    ) and hasattr(v4, "local_num_attention_heads"):
        index_n_heads = int(v4.local_num_attention_heads)
    return Dsv4RuntimeArgs(
        dim=int(v4.hidden_size),
        n_heads=n_heads,
        n_hash_layers=int(v4.num_hash_layers),
        n_routed_experts=int(v4.num_routed_experts),
        n_activated_experts=int(v4.experts_per_token),
        n_shared_experts=int(v4.num_shared_experts),
        q_lora_rank=int(v4.q_lora_rank),
        o_lora_rank=int(v4.o_lora_rank),
        o_groups=o_groups,
        head_dim=int(v4.head_dim),
        rope_head_dim=int(v4.qk_rope_head_dim),
        window_size=int(v4.sliding_window),
        compress_ratios=tuple(int(r) for r in v4.compress_ratios),
        index_n_heads=index_n_heads,
        index_head_dim=int(v4.index_head_dim),
        index_topk=int(v4.index_topk),
        hc_mult=int(v4.hc_mult),
        hc_sinkhorn_iters=int(v4.hc_sinkhorn_iters),
        hc_eps=float(v4.hc_eps),
        norm_eps=float(v4.rms_norm_eps),
        swiglu_limit=float(v4.swiglu_limit),
        routed_scaling_factor=float(v4.routed_scaling_factor),
        scoring_func=str(v4.scoring_func),
        topk_method=str(v4.topk_method),
        rope_theta=float(v4.rope_theta),
        compress_rope_theta=float(v4.compress_rope_theta),
        rope_scaling_factor=float(v4.rope_scaling_factor),
        original_seq_len=int(v4.rope_original_max_position),
        beta_fast=int(v4.rope_beta_fast),
        beta_slow=int(v4.rope_beta_slow),
        vocab_size=int(v4.vocab_size),
        moe_inter_dim=int(v4.moe_intermediate_size),
        tp_degree=tp_degree,
        tp_rank=tp_rank,
        tp_replica_groups=_default_v4_tp_groups(v4, tp_degree=tp_degree),
    )


def _build_sampled_freqs(args: Any, max_seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    full = precompute_freqs_cis_yarn(
        dim=int(args.rope_head_dim),
        seqlen=int(max_seq_len),
        original_seq_len=0,
        base=float(args.rope_theta),
        factor=float(args.rope_scaling_factor),
        beta_fast=int(args.beta_fast),
        beta_slow=int(args.beta_slow),
    )
    compressed = precompute_freqs_cis_yarn(
        dim=int(args.rope_head_dim),
        seqlen=int(max_seq_len),
        original_seq_len=int(args.original_seq_len),
        base=float(args.compress_rope_theta),
        factor=float(args.rope_scaling_factor),
        beta_fast=int(args.beta_fast),
        beta_slow=int(args.beta_slow),
    )
    return full, compressed


def _sampled_compressor_surface(
    layer_weights: Any,
    args: Dsv4RuntimeArgs,
    freqs_cis: np.ndarray,
    *,
    ratio: int,
    indexer: bool,
    layer_id: int,
) -> Dsv4CompressorSurface:
    prefix = "idx_comp_" if indexer else "comp_"
    ctx = f"DSV4 layer {layer_id} {'indexer ' if indexer else ''}compressor"
    return Dsv4CompressorSurface(
        dim=int(args.dim),
        head_dim=int(args.index_head_dim if indexer else args.head_dim),
        rope_head_dim=int(args.rope_head_dim),
        compress_ratio=int(ratio),
        overlap=int(ratio) == 4,
        rotate=bool(indexer),
        eps=float(args.norm_eps),
        wkv=_required(layer_weights, f"{prefix}wkv", ctx=ctx),
        wgate=_required(layer_weights, f"{prefix}wgate", ctx=ctx),
        ape=_required(layer_weights, f"{prefix}ape", ctx=ctx),
        norm_weight=_required(layer_weights, f"{prefix}norm", ctx=ctx),
        freqs_cis=freqs_cis,
        freqs_cos=np.ascontiguousarray(freqs_cis.real.astype(np.float32)),
        freqs_sin=np.ascontiguousarray(freqs_cis.imag.astype(np.float32)),
    )


def _sampled_indexer_surface(
    layer_weights: Any,
    args: Dsv4RuntimeArgs,
    freqs_cis: np.ndarray,
    *,
    ratio: int,
    layer_id: int,
) -> Dsv4IndexerSurface:
    compressor = _sampled_compressor_surface(
        layer_weights,
        args,
        freqs_cis,
        ratio=ratio,
        indexer=True,
        layer_id=layer_id,
    )
    ctx = f"DSV4 layer {layer_id} indexer"
    return Dsv4IndexerSurface(
        n_heads=int(args.index_n_heads),
        head_dim=int(args.index_head_dim),
        rope_head_dim=int(args.rope_head_dim),
        compress_ratio=int(ratio),
        index_topk=int(args.index_topk),
        wq_b=_required(layer_weights, "idx_wq_b", ctx=ctx),
        weights_proj=_required(layer_weights, "idx_weights_proj", ctx=ctx),
        compressor=compressor,
        softmax_scale=float(args.index_head_dim**-0.5),
    )


def _sampled_attention_surface(
    layer_weights: Any,
    args: Dsv4RuntimeArgs,
    freqs_cis: np.ndarray,
    comp_freqs_cis: np.ndarray,
    *,
    layer_id: int,
) -> Dsv4AttentionSurface:
    ratio = int(args.compress_ratios[layer_id])
    compressor = None
    indexer = None
    if ratio:
        compressor = _sampled_compressor_surface(
            layer_weights,
            args,
            comp_freqs_cis,
            ratio=ratio,
            indexer=False,
            layer_id=layer_id,
        )
        if ratio == 4:
            indexer = _sampled_indexer_surface(
                layer_weights,
                args,
                comp_freqs_cis,
                ratio=ratio,
                layer_id=layer_id,
            )
    ctx = f"DSV4 layer {layer_id} attention"
    return Dsv4AttentionSurface(
        args=args,
        layer_id=int(layer_id),
        n_heads=int(args.n_heads),
        head_dim=int(args.head_dim),
        rope_head_dim=int(args.rope_head_dim),
        n_groups=int(args.o_groups),
        window_size=int(args.window_size),
        compress_ratio=ratio,
        eps=float(args.norm_eps),
        softmax_scale=float(args.head_dim**-0.5),
        tp_degree=int(getattr(args, "tp_degree", 1)),
        tp_rank=int(getattr(args, "tp_rank", 0)),
        tp_replica_groups=tuple(getattr(args, "tp_replica_groups", ())),
        attn_sink=_required(layer_weights, "attn_sink", ctx=ctx),
        wq_a=_required(layer_weights, "wq_a", ctx=ctx),
        q_norm=_required(layer_weights, "q_norm", ctx=ctx),
        wq_b=_required(layer_weights, "wq_b", ctx=ctx),
        wkv=_required(layer_weights, "wkv", ctx=ctx),
        kv_norm=_required(layer_weights, "kv_norm", ctx=ctx),
        wo_a=_required(layer_weights, "wo_a", ctx=ctx),
        wo_b=_required(layer_weights, "wo_b", ctx=ctx),
        compressor=compressor,
        indexer=indexer,
        freqs_cis=freqs_cis,
        freqs_cos=np.ascontiguousarray(freqs_cis.real.astype(np.float32)),
        freqs_sin=np.ascontiguousarray(freqs_cis.imag.astype(np.float32)),
    )


def _sampled_gate_surface(
    layer_weights: Any,
    args: Dsv4RuntimeArgs,
    *,
    layer_id: int,
) -> Dsv4GateSurface:
    is_hash = int(layer_id) < int(args.n_hash_layers)
    ctx = f"DSV4 layer {layer_id} gate"
    return Dsv4GateSurface(
        is_hash=is_hash,
        topk=int(args.n_activated_experts),
        route_scale=float(args.routed_scaling_factor),
        score_func=str(args.scoring_func),
        weight=_required(layer_weights, "gate_weight", ctx=ctx),
        bias=None if is_hash else getattr(layer_weights, "gate_bias", None),
        tid2eid=(
            _required(layer_weights, "gate_tid2eid", ctx=ctx)
            if is_hash
            else getattr(layer_weights, "gate_tid2eid", None)
        ),
    )


def build_dsv4_runtime_surface_from_weights(
    *,
    model_config: Any,
    v4_weights: Any,
    device_weights: Any,
    max_batch_size: int,
    max_seq_len: int,
) -> Dsv4RuntimeSurface:
    """Build a runtime surface without constructing an eager executor.

    This owns static metadata, RoPE tables, layer/head shape fields, and the
    device tensor handles from ``V4DeviceWeights``. Routed expert execution
    uses the explicit blockwise-MoE state bridge built from device-resident
    weights; shared-expert BF16 tensors are consumed by the sampled graph
    fragment. Legacy FP8 shared tensors are still accepted for reference
    snapshots.
    """
    args = _build_runtime_args(v4_weights)
    freqs_full, freqs_compressed = _build_sampled_freqs(args, int(max_seq_len))
    layers = tuple(getattr(device_weights, "layers", ()))
    n_layers = int(v4_weights.num_hidden_layers)
    if len(layers) < n_layers:
        raise RuntimeError(
            "V4DeviceWeights does not cover all DSV4 layers: "
            f"{len(layers)} < {n_layers}"
        )

    blocks = []
    for layer_id in range(n_layers):
        lw = layers[layer_id]
        ratio = int(args.compress_ratios[layer_id])
        layer_freqs = freqs_compressed if ratio else freqs_full
        ffn = Dsv4FfnSurface(
            layer_id=int(layer_id),
            dim=int(args.dim),
            n_routed_experts=int(args.n_routed_experts),
            gate=_sampled_gate_surface(lw, args, layer_id=layer_id),
            experts=(),
            shared=_SampledSharedExpertSurface(lw),
        )
        ctx = f"DSV4 layer {layer_id}"
        blocks.append(
            Dsv4LayerSurface(
                args=args,
                layer_id=int(layer_id),
                attn=_sampled_attention_surface(
                    lw,
                    args,
                    layer_freqs,
                    freqs_compressed,
                    layer_id=layer_id,
                ),
                ffn=ffn,
                attn_norm=_required(lw, "attn_norm", ctx=ctx),
                ffn_norm=_required(lw, "ffn_norm", ctx=ctx),
                hc_attn_fn=_required(lw, "hc_attn_fn", ctx=ctx),
                hc_attn_scale=_required(lw, "hc_attn_scale", ctx=ctx),
                hc_attn_base=_required(lw, "hc_attn_base", ctx=ctx),
                hc_ffn_fn=_required(lw, "hc_ffn_fn", ctx=ctx),
                hc_ffn_scale=_required(lw, "hc_ffn_scale", ctx=ctx),
                hc_ffn_base=_required(lw, "hc_ffn_base", ctx=ctx),
            )
        )

    head = Dsv4HeadSurface(
        lm_head=_required(device_weights, "lm_head", ctx="DSV4 head"),
        final_norm=_required(device_weights, "final_norm", ctx="DSV4 head"),
        hc_head_fn=_required(device_weights, "hc_head_fn", ctx="DSV4 head"),
        hc_head_scale=_required(device_weights, "hc_head_scale", ctx="DSV4 head"),
        hc_head_base=_required(device_weights, "hc_head_base", ctx="DSV4 head"),
        norm_eps=float(args.norm_eps),
        hc_eps=float(args.hc_eps),
    )
    weights = Dsv4WeightsSurface(
        embed=_required(device_weights, "embed", ctx="DSV4 weights"),
        final_norm=head.final_norm,
        lm_head=head.lm_head,
    )
    return Dsv4RuntimeSurface(
        model_config=model_config,
        v4=v4_weights,
        w=weights,
        max_batch_size=int(max_batch_size),
        max_seq_len=int(max_seq_len),
        args=args,
        blocks=tuple(blocks),
        head=head,
        _dsv4_source_runtime=None,
        _dsv4_device_weights=device_weights,
        _dsv4_freqs_cis_full=freqs_full,
        _dsv4_freqs_cis_compressed=freqs_compressed,
    )
