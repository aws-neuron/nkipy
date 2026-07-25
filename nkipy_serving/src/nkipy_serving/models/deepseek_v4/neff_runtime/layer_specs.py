"""Layer-level graph variant specs for DSV4 NEFF runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(value, "shape", ()) or ())


def _router_kind(block: Any) -> str:
    moe = getattr(block, "ffn", None)
    gate = getattr(moe, "gate", None)
    if moe is None or gate is None:
        return "none"
    weight = getattr(gate, "weight", None)
    if weight is None:
        return "none"
    if bool(getattr(gate, "is_hash", False)):
        return (
            "hash_with_bias"
            if getattr(gate, "bias", None) is not None
            else "hash_no_bias"
        )
    return (
        "learned_with_bias"
        if getattr(gate, "bias", None) is not None
        else "learned_no_bias"
    )


def _gate_topk(block: Any) -> int:
    gate = getattr(getattr(block, "ffn", None), "gate", None)
    return int(getattr(gate, "topk", 0) or 0) if gate is not None else 0


def _gate_num_experts(block: Any) -> int:
    gate = getattr(getattr(block, "ffn", None), "gate", None)
    weight = getattr(gate, "weight", None) if gate is not None else None
    shape = _shape_tuple(weight)
    return int(shape[0]) if shape else 0


def _safe_key_part(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


@dataclass(frozen=True)
class Dsv4ProductLayerGraphSpec:
    """Canonical per-layer variant key for future layer-level NEFF fusion.

    ``layer_key`` includes ``layer_id`` for diagnostics. ``variant_key`` omits it
    so we can see which layers could share the same compiled graph shape/control
    variant once weights remain tensor inputs.
    """

    layer_id: int
    num_layers: int
    token_bucket: int
    ids_shape: tuple[int, ...]
    batch_size: int
    seqlen: int
    is_decode: bool
    has_dp_attention: bool
    has_dp_superstep: bool
    input_boundary: str
    output_boundary: str
    router_kind: str
    gate_topk: int
    gate_num_experts: int
    blockwise_moe_enabled: bool

    @property
    def phase(self) -> str:
        return "decode" if self.is_decode else "prefill"

    @property
    def final(self) -> bool:
        return int(self.layer_id) == int(self.num_layers) - 1

    @property
    def variant_key(self) -> tuple[Any, ...]:
        return (
            "dsv4_product_layer_graph",
            int(self.token_bucket),
            tuple(int(dim) for dim in self.ids_shape),
            int(self.batch_size),
            int(self.seqlen),
            self.phase,
            bool(self.has_dp_attention),
            bool(self.has_dp_superstep),
            str(self.input_boundary),
            str(self.output_boundary),
            str(self.router_kind),
            int(self.gate_topk),
            int(self.gate_num_experts),
            bool(self.blockwise_moe_enabled),
        )

    @property
    def layer_key(self) -> tuple[Any, ...]:
        return (int(self.layer_id),) + self.variant_key

    @property
    def variant_name(self) -> str:
        parts = (
            "dsv4_layer",
            f"t{int(self.token_bucket)}",
            self.phase,
            f"s{int(self.seqlen)}",
            "dp" if self.has_dp_attention else "nodp",
            "super" if self.has_dp_superstep else "nosuper",
            _safe_key_part(self.input_boundary),
            _safe_key_part(self.output_boundary),
            _safe_key_part(self.router_kind),
            f"k{int(self.gate_topk)}",
            f"e{int(self.gate_num_experts)}",
            "blockwise" if self.blockwise_moe_enabled else "dispatch",
        )
        return "_".join(parts)

    @property
    def layer_name(self) -> str:
        return f"layer{int(self.layer_id)}_{self.variant_name}"

    def profile_fields(self) -> dict[str, Any]:
        return {
            "layer_graph_key": self.layer_name,
            "layer_variant_key": self.variant_name,
            "layer_graph_phase": self.phase,
            "layer_graph_compile_seqlen": int(self.seqlen),
            "layer_graph_input": str(self.input_boundary),
            "layer_graph_output": str(self.output_boundary),
            "layer_graph_router": str(self.router_kind),
            "layer_graph_gate_topk": int(self.gate_topk),
            "layer_graph_gate_num_experts": int(self.gate_num_experts),
            "layer_graph_blockwise_moe": bool(self.blockwise_moe_enabled),
            "layer_graph_dp_attention": bool(self.has_dp_attention),
            "layer_graph_dp_superstep": bool(self.has_dp_superstep),
        }


def product_layer_graph_spec(
    block: Any,
    *,
    layer_id: int,
    num_layers: int,
    token_bucket: int,
    ids_shape: tuple[int, ...],
    batch_size: int,
    seqlen: int,
    is_decode: bool,
    has_dp_attention: bool,
    has_dp_superstep: bool,
    input_boundary: str,
    output_boundary: str,
    blockwise_moe_enabled: bool,
) -> Dsv4ProductLayerGraphSpec:
    return Dsv4ProductLayerGraphSpec(
        layer_id=int(layer_id),
        num_layers=int(num_layers),
        token_bucket=int(token_bucket),
        ids_shape=tuple(int(dim) for dim in ids_shape),
        batch_size=int(batch_size),
        seqlen=int(seqlen),
        is_decode=bool(is_decode),
        has_dp_attention=bool(has_dp_attention),
        has_dp_superstep=bool(has_dp_superstep),
        input_boundary=str(input_boundary),
        output_boundary=str(output_boundary),
        router_kind=_router_kind(block),
        gate_topk=_gate_topk(block),
        gate_num_experts=_gate_num_experts(block),
        blockwise_moe_enabled=bool(blockwise_moe_enabled),
    )
