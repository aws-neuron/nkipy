"""DSV4 MoE via the local ``blockwise_nki`` kernel.

Replaces the per-expert Python loop in ``sampled_forward._run_moe`` with one
``blockwise_nki_static`` (prefill) or ``blockwise_nki_decode`` (decode)
kernel call per MoE layer. Same shape GPT-OSS and Qwen3-MoE production
paths use.

Constraints are summarized in ``docs/models/deepseek_v4.md``:

- **No-scale FP8 routed weights.** Routed experts are preprocessed offline
  from HF MXFP4 into Neuron E4M3 and uploaded without runtime scales.
- **Router-agnostic.** Learned and hash-MoE layers both feed the same
  ``weights`` / ``indices`` scheduling contract; hash routing only changes
  how those arrays are produced.
- **Shared expert stays outside the blockwise kernel.** The sampled forward
  fuses shared-expert add in a trace function after routed expert compute.
- **No in-kernel residual.** V4's ``mhc_post`` is not a plain add;
  ``blockwise_add_residual`` doesn't fit. We call the bare kernel and
  return ``[T, H]`` to the caller.
"""

from __future__ import annotations

import fcntl
import hashlib
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import ml_dtypes
import numpy as np

from nkipy_serving.ops.moe.blockwise_index import (
    BLOCK_SIZE as _MOE_BLOCK_SIZE,
)
from nkipy_serving.ops.moe.blockwise_index import (
    get_n_blocks,
)
from nkipy_serving.ops.moe.blockwise_nki import (
    TILE_SIZE as _MOE_TILE_SIZE,
)
from nkipy_serving.ops.moe.blockwise_nki import (
    blockwise_nki_decode,
)
from nkipy_serving.ops.moe.blockwise_nki_beta2 import (
    blockwise_nki_prefill_dsv4_beta2,
)
from nkipy_serving.runtime.collective_load import collective_load_barrier
from nkipy_serving.runtime.device_tensor import alias_device_value_shape
from nkipy_serving.runtime.device_tensor import is_device_tensor as _is_device_tensor
from nkipy_serving.runtime.device_tensor import sample_like as sample_device_like
from nkipy_serving.runtime.kernel_compile import compile_and_load_with_lock

MOE_BLOCK_SIZE = int(_MOE_BLOCK_SIZE)
MOE_TILE_SIZE = int(_MOE_TILE_SIZE)


# ---------------------------------------------------------------------------
# Stacked per-layer weights
# ---------------------------------------------------------------------------


@dataclass
class LayerStackedExperts:
    """Per-layer bf16 tensors the kernel consumes.

    Shapes match ``blockwise_nki_static`` / ``blockwise_nki_decode``:

      - ``gate_up_proj_weight`` ``[E_local, H, 2, I_local]``
      - ``down_proj_weight``     ``[E_local, I_local, H]``
      - ``gate_up_bias_plus1_T`` ``[E_local, I_local, 2]`` (plus-1 convention)
      - ``down_bias_broadcasted`` ``[E_local, TILE_SIZE, H]`` when the model
        has down bias. V4 is bias-free and leaves this as ``None``.
    """

    gate_up_w: Any
    down_w: Any
    gate_up_bias: Any
    down_bias_bc: Any = None

    @property
    def n_local_experts(self) -> int:
        return int(self.gate_up_w.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.gate_up_w.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.gate_up_w.shape[3])


@dataclass
class BlockwiseMoEState:
    """Per-executor state: stacked weights + compiled kernels + scratch."""

    layers: list[LayerStackedExperts] = field(default_factory=list)
    hidden_size: int = 0
    intermediate_size: int = 0
    n_local_experts: int = 0
    experts_per_token: int = 0
    ep_degree: int = 1
    ep_rank: int = 0
    ep_replica_groups: tuple[tuple[int, ...], ...] = ()
    tp_degree: int = 1
    tp_rank: int = 0
    tp_replica_groups: tuple[tuple[int, ...], ...] = ()
    collective_rank: int = 0
    collective_world_size: int = 0
    # SwiGLU clamp bounds that match V4's ``swiglu_with_limit(limit)``.
    # Kernel defaults target GPT-OSS's swiglu_oai; V4 passes its own.
    swiglu_limit: float = 10.0
    # Prefill kernel cache. Device-router and schedule-input paths have
    # different graph signatures, so keys include a path discriminator.
    prefill_kernel_cache: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    # Device top-k → blockwise schedule kernels.
    prefill_schedule_kernel_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict
    )
    # Decode MoE kernel cache. Device-router and affinity-input paths use
    # different tuple keys because their graph signatures differ.
    decode_kernel_cache: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    # Device top-k → decode affinities/static block metadata kernels.
    decode_schedule_kernel_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict
    )
    # output shape/dtype → EP all-reduce kernel
    ep_all_reduce_kernel_cache: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    # output shape/dtype → TP all-reduce kernel
    tp_all_reduce_kernel_cache: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    # output shape/dtype → combined EP×TP all-reduce kernel
    ep_tp_all_reduce_kernel_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict
    )
    precompiled_kernel_caches_sealed: bool = False

    def _v4_clamp_kwargs(self) -> dict[str, Any]:
        """Clamp kwargs matching ``swiglu_with_limit(gate, up, limit)``.

        V4: ``up = clip(up, -limit, limit); gate = min(gate, limit)`` - no
        lower bound on gate. With ``swiglu_limit == 0`` the reference path skips
        clamping entirely; pass a large sentinel so the device kernel
        effectively no-ops.
        """
        lim = float(self.swiglu_limit)
        if lim <= 0:
            lim = 1e6  # effectively no clamp
        return dict(
            gate_clamp_upper=lim,
            gate_clamp_lower=None,
            up_clamp_upper=lim,
            up_clamp_lower=-lim,
        )

    def seal_precompiled_kernels(self) -> None:
        """Reject new blockwise-MoE DeviceKernel shapes after warmup."""
        self.precompiled_kernel_caches_sealed = True


@dataclass
class _BlockwisePrecompiledCollectiveKernel:
    neff_path: str
    name: str
    build_dir: str | None
    rank_id: int
    world_size: int
    load_barrier_name: str
    loaded: Any | None = None

    def load(self) -> Any:
        if self.loaded is not None:
            return self.loaded
        from nkipy_serving.models._device_utils import _get_device_kernel_cls

        collective_load_barrier(
            build_dir=self.build_dir,
            name=self.load_barrier_name,
            rank_id=int(self.rank_id),
            world_size=int(self.world_size),
        )
        self.loaded = _get_device_kernel_cls().load_from_neff(
            self.neff_path,
            name=self.name,
            cc_enabled=True,
            rank_id=int(self.rank_id),
            world_size=int(self.world_size),
        )
        return self.loaded

    def unload(self) -> None:
        if self.loaded is None:
            return
        try:
            from spike.spike_singleton import get_spike_singleton

            get_spike_singleton().unload_model(self.loaded.model_ref)
        finally:
            self.loaded = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kernel = self.load()
        try:
            return kernel(*args, **kwargs)
        finally:
            self.unload()


def _require_unsealed_blockwise_cache(
    state: BlockwiseMoEState,
    *,
    kind: str,
    cache_key: tuple[Any, ...],
) -> None:
    if not bool(getattr(state, "precompiled_kernel_caches_sealed", False)):
        return
    raise RuntimeError(
        "DSV4 blockwise MoE late DeviceKernel compile blocked after warmup "
        f"seal: kind={kind} key={cache_key}"
    )


# ---------------------------------------------------------------------------
# Weight stacking
# ---------------------------------------------------------------------------


def _cast_bf16(x: np.ndarray) -> np.ndarray:
    if x.dtype == ml_dtypes.bfloat16:
        return x
    return x.astype(ml_dtypes.bfloat16)


def _sample_like(x: Any, dtype: Any | None = None) -> np.ndarray:
    """Return a numpy compile sample for numpy or DeviceTensor weights."""
    if _is_device_tensor(x):
        return sample_device_like(x, dtype, fill="zeros")
    arr = np.asarray(x)
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def _down_bias_sample(x: Any) -> np.ndarray:
    if x is None:
        return np.zeros((1, 1, 1), dtype=ml_dtypes.bfloat16)
    return _sample_like(x)


def _to_device_tensor(x: Any, *, name: str, tensor_cls: Any) -> Any:
    if _is_device_tensor(x):
        return x
    return tensor_cls.from_numpy(np.asarray(x), name=name)


def _down_bias_device_tensor(x: Any, *, tensor_cls: Any) -> Any:
    if x is None:
        return tensor_cls.from_numpy(
            np.zeros((1, 1, 1), dtype=ml_dtypes.bfloat16),
            name="dn_b_zero",
        )
    return _to_device_tensor(x, name="dn_b", tensor_cls=tensor_cls)


def _normalize_replica_groups(groups: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(r) for r in group) for group in tuple(groups or ()))


def _replica_groups_tag(groups: tuple[tuple[int, ...], ...]) -> str:
    return hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]


def _blockwise_shared_build_dir(build_dir: str | Path | None) -> str | None:
    if build_dir is None:
        return None
    path = Path(str(build_dir))
    parts = path.parts
    for idx, part in enumerate(parts):
        if re.fullmatch(r"rank_\d+", part):
            return str(Path(*parts[:idx]) / "blockwise_moe")
    return str(path)


@contextmanager
def _blockwise_compile_lock(*, build_dir: str | None, name: str) -> Any:
    root = Path(build_dir or "/tmp/nkipy_serving_dsv4_blockwise_compile")
    lock_dir = root / ".dsv4_blockwise_compile_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:180] or "kernel"
    digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:12]
    with (lock_dir / f"{safe}_{digest}.lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _compile_blockwise_collective_kernel(
    fn: Any,
    *sample_args: Any,
    name: str,
    build_dir: str | Path | None,
    rank_id: int,
    world_size: int,
    additional_compiler_args: str | None = None,
    load_barrier_name: str | None = None,
    **kwargs: Any,
) -> Any:
    """Compile a CC kernel once, then collectively load it outside the lock."""
    from nkipy_serving.models._device_utils import _get_device_kernel_cls

    device_kernel_cls = _get_device_kernel_cls()
    shared_build_dir = _blockwise_shared_build_dir(build_dir)
    target = kwargs.pop("target", None)
    if target is None:
        from nkipy.core.compile import CompilationTarget

        target = CompilationTarget.DEFAULT
    with _blockwise_compile_lock(build_dir=shared_build_dir, name=name):
        neff_path, _ = device_kernel_cls._trace_and_compile(
            fn,
            name,
            sample_args,
            kwargs,
            additional_compiler_args=additional_compiler_args,
            use_cached_if_exists=True,
            build_dir=shared_build_dir,
            target=target,
        )
    if load_barrier_name:
        return _BlockwisePrecompiledCollectiveKernel(
            neff_path=str(neff_path),
            name=str(name),
            build_dir=shared_build_dir,
            rank_id=int(rank_id),
            world_size=int(world_size),
            load_barrier_name=str(load_barrier_name),
        )
    return device_kernel_cls.load_from_neff(
        neff_path,
        name=name,
        cc_enabled=True,
        rank_id=int(rank_id),
        world_size=int(world_size),
    )


def _fill_state_shape_metadata(state: BlockwiseMoEState) -> None:
    if state.layers:
        for lx in state.layers:
            if lx.n_local_experts > 0:
                state.hidden_size = lx.hidden_size
                state.intermediate_size = lx.intermediate_size
                state.n_local_experts = lx.n_local_experts
                break


def build_blockwise_state_from_device_weights(
    v4_meta: Any,
    device_weights: Any,
    *,
    experts_per_token: int | None = None,
    ep_degree: int = 1,
    ep_rank: int = 0,
    ep_replica_groups: tuple[tuple[int, ...], ...] = (),
    tp_degree: int = 1,
    tp_rank: int = 0,
    tp_replica_groups: tuple[tuple[int, ...], ...] = (),
    collective_rank: int | None = None,
    collective_world_size: int | None = None,
    swiglu_limit: float | None = None,
) -> BlockwiseMoEState:
    """Build blockwise MoE state from `V4DeviceWeights`.

    Requires the product no-scale FP8 stacked fields loaded by
    ``V4LoadPlan.blockwise_moe_fp8``. This keeps routed MoE weights in their
    final serving dtype.
    """
    state = BlockwiseMoEState(
        experts_per_token=int(
            experts_per_token
            if experts_per_token is not None
            else v4_meta.experts_per_token
        ),
        ep_degree=int(ep_degree),
        ep_rank=int(ep_rank),
        ep_replica_groups=_normalize_replica_groups(ep_replica_groups),
        tp_degree=int(tp_degree),
        tp_rank=int(tp_rank),
        tp_replica_groups=_normalize_replica_groups(tp_replica_groups),
        collective_rank=(
            int(collective_rank) if collective_rank is not None else int(tp_rank)
        ),
        collective_world_size=(
            int(collective_world_size)
            if collective_world_size is not None
            else int(tp_degree)
        ),
        swiglu_limit=float(
            swiglu_limit if swiglu_limit is not None else v4_meta.swiglu_limit
        ),
    )
    layers = tuple(getattr(device_weights, "layers", ()))
    n_layers = int(v4_meta.num_hidden_layers)
    if len(layers) < n_layers:
        raise RuntimeError(
            "V4DeviceWeights does not cover all DSV4 MoE layers: "
            f"{len(layers)} < {n_layers}"
        )
    required = (
        "blockwise_gate_up_w",
        "blockwise_down_w",
        "blockwise_gate_up_bias",
    )
    for layer_id in range(n_layers):
        lw = layers[layer_id]
        missing = [name for name in required if getattr(lw, name, None) is None]
        if missing:
            raise RuntimeError(
                "V4DeviceWeights missing blockwise FP8 MoE tensors for "
                f"layer {layer_id}: {missing}. Load with "
                "V4LoadPlan(..., blockwise_moe_fp8=True)."
            )
        state.layers.append(
            LayerStackedExperts(
                gate_up_w=lw.blockwise_gate_up_w,
                down_w=lw.blockwise_down_w,
                gate_up_bias=lw.blockwise_gate_up_bias,
                down_bias_bc=getattr(lw, "blockwise_down_bias_bc", None),
            )
        )

    _fill_state_shape_metadata(state)
    return state


def _tp_replica_groups_for_collective(state: BlockwiseMoEState) -> list[list[int]]:
    if state.tp_replica_groups:
        return [list(group) for group in state.tp_replica_groups]
    return [list(range(int(state.tp_degree)))]


def _ep_replica_groups_for_collective(state: BlockwiseMoEState) -> list[list[int]]:
    if state.ep_replica_groups:
        return [list(group) for group in state.ep_replica_groups]
    return [
        [
            row * int(state.tp_degree) + int(state.tp_rank)
            for row in range(int(state.ep_degree))
        ]
    ]


def _ep_all_reduce_output_fn(
    output: np.ndarray,
    *,
    ep_degree: int,
    ep_replica_groups: tuple = (),
) -> np.ndarray:
    if int(ep_degree) <= 1:
        return output
    import nkipy.distributed.collectives as cc

    groups = (
        [list(group) for group in ep_replica_groups]
        if ep_replica_groups
        else [list(range(int(ep_degree)))]
    )
    return cc.all_reduce(output, replica_groups=groups, reduce_op=np.add)


def _tp_all_reduce_output_fn(
    output: np.ndarray,
    *,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    if int(tp_degree) <= 1:
        return output
    import nkipy.distributed.collectives as cc

    groups = (
        [list(group) for group in tp_replica_groups]
        if tp_replica_groups
        else [list(range(int(tp_degree)))]
    )
    return cc.all_reduce(output, replica_groups=groups, reduce_op=np.add)


def _ep_tp_all_reduce_output_fn(
    output: np.ndarray,
    *,
    replica_groups: tuple = (),
) -> np.ndarray:
    if not replica_groups:
        return output
    import nkipy.distributed.collectives as cc

    groups = [list(group) for group in replica_groups]
    return cc.all_reduce(output, replica_groups=groups, reduce_op=np.add)


def _moe_ep_tp_replica_groups_for_collective(
    state: BlockwiseMoEState,
) -> tuple[tuple[int, ...], ...] | None:
    """Return full-replica groups when EP and TP partitions form a grid."""

    tp_groups = tuple(
        tuple(group) for group in _tp_replica_groups_for_collective(state)
    )
    ep_groups = tuple(
        tuple(group) for group in _ep_replica_groups_for_collective(state)
    )
    if not tp_groups or not ep_groups:
        return None

    def _rank_partition(groups: tuple[tuple[int, ...], ...]) -> dict[int, int] | None:
        seen: dict[int, int] = {}
        for group_idx, group in enumerate(groups):
            for rank in group:
                rank_i = int(rank)
                if rank_i in seen:
                    return None
                seen[rank_i] = int(group_idx)
        return seen

    tp_rank_to_group = _rank_partition(tp_groups)
    ep_rank_to_group = _rank_partition(ep_groups)
    if tp_rank_to_group is None or ep_rank_to_group is None:
        return None
    if set(tp_rank_to_group) != set(ep_rank_to_group):
        return None

    for tp_group in tp_groups:
        tp_set = set(tp_group)
        for ep_group in ep_groups:
            if len(tp_set & set(ep_group)) > 1:
                return None

    parent = {rank: rank for rank in tp_rank_to_group}

    def _find(rank: int) -> int:
        root = rank
        while parent[root] != root:
            root = parent[root]
        while parent[rank] != rank:
            nxt = parent[rank]
            parent[rank] = root
            rank = nxt
        return root

    def _union(a: int, b: int) -> None:
        ra = _find(int(a))
        rb = _find(int(b))
        if ra != rb:
            parent[rb] = ra

    for group in (*tp_groups, *ep_groups):
        if not group:
            return None
        head = int(group[0])
        for rank in group[1:]:
            _union(head, int(rank))

    components: dict[int, set[int]] = {}
    for rank in parent:
        components.setdefault(_find(rank), set()).add(int(rank))

    out: list[tuple[int, ...]] = []
    for ranks in components.values():
        comp_tp = [tuple(group) for group in tp_groups if ranks & set(group)]
        comp_ep = [tuple(group) for group in ep_groups if ranks & set(group)]
        if not comp_tp or not comp_ep:
            return None
        if any(set(group) - ranks for group in (*comp_tp, *comp_ep)):
            return None
        for tp_group in comp_tp:
            if len(tp_group) != len(comp_ep):
                return None
            tp_set = set(tp_group)
            if any(len(tp_set & set(ep_group)) != 1 for ep_group in comp_ep):
                return None
        for ep_group in comp_ep:
            if len(ep_group) != len(comp_tp):
                return None
        out.append(tuple(sorted(ranks)))
    return tuple(sorted(out))


def _state_collective_rank_world(state: BlockwiseMoEState) -> tuple[int, int]:
    groups = (
        _tp_replica_groups_for_collective(state)
        if int(state.tp_degree) > 1
        else _ep_replica_groups_for_collective(state)
    )
    rank = int(state.collective_rank)
    world = int(state.collective_world_size)
    if world > 0:
        return rank, world
    if groups:
        if 0 <= int(state.tp_rank) < len(groups[0]):
            rank = int(groups[0][int(state.tp_rank)])
        world = max(max(group) for group in groups) + 1
    else:
        world = int(state.tp_degree)
    return rank, world


def _canonical_moe_output_shape(
    output: Any, state: BlockwiseMoEState
) -> tuple[int, int]:
    """Normalize singleton decode DeviceTensor metadata to the runtime shape."""
    shape = tuple(int(v) for v in output.shape)
    hidden = int(state.hidden_size)
    if len(shape) == 1 and shape[0] == hidden:
        return (1, hidden)
    if len(shape) != 2:
        raise RuntimeError(f"blockwise MoE output must be [T, H], got {shape}")
    return shape


def _alias_device_tensor_shape(output: Any, shape: tuple[int, ...]) -> Any:
    alias = alias_device_value_shape(output, shape, default_name="moe_output")
    if alias is None:
        raise RuntimeError(
            "blockwise MoE output requires a DeviceTensor-compatible tensor_ref "
            "to normalize singleton decode output metadata"
        )
    return alias


def _ep_all_reduce_cache_key(
    shape: tuple[int, int],
    dtype: np.dtype,
    state: BlockwiseMoEState,
) -> tuple[Any, ...]:
    groups = tuple(tuple(group) for group in _ep_replica_groups_for_collective(state))
    return (shape, str(dtype), int(state.ep_degree), groups)


def _tp_all_reduce_cache_key(
    shape: tuple[int, int],
    dtype: np.dtype,
    state: BlockwiseMoEState,
) -> tuple[Any, ...]:
    groups = tuple(tuple(group) for group in _tp_replica_groups_for_collective(state))
    return (shape, str(dtype), int(state.tp_degree), groups)


def _ep_tp_all_reduce_cache_key(
    shape: tuple[int, int],
    dtype: np.dtype,
    state: BlockwiseMoEState,
) -> tuple[Any, ...] | None:
    groups = _moe_ep_tp_replica_groups_for_collective(state)
    if not groups:
        return None
    return (
        shape,
        str(dtype),
        int(state.ep_degree),
        int(state.tp_degree),
        groups,
    )


def _compile_ep_all_reduce_kernel(
    shape: tuple[int, int],
    dtype: np.dtype,
    state: BlockwiseMoEState,
    *,
    artifacts_dir: str | Path | None = None,
) -> Any:
    groups = tuple(tuple(group) for group in _ep_replica_groups_for_collective(state))
    rank, world = _state_collective_rank_world(state)
    sample = np.zeros(shape, dtype=dtype)
    group_tag = _replica_groups_tag(groups)
    barrier_name = f"dsv4_moe_ep_ar_t{shape[0]}_h{shape[1]}_ep{state.ep_degree}"
    return _compile_blockwise_collective_kernel(
        _ep_all_reduce_output_fn,
        sample,
        ep_degree=int(state.ep_degree),
        ep_replica_groups=groups,
        name=(
            f"dsv4_moe_ep_ar_t{shape[0]}_h{shape[1]}_ep{state.ep_degree}_{group_tag}"
        ),
        build_dir=str(artifacts_dir) if artifacts_dir else None,
        rank_id=rank,
        world_size=world,
        load_barrier_name=barrier_name,
    )


def _compile_tp_all_reduce_kernel(
    shape: tuple[int, int],
    dtype: np.dtype,
    state: BlockwiseMoEState,
    *,
    artifacts_dir: str | Path | None = None,
) -> Any:
    groups = tuple(tuple(group) for group in _tp_replica_groups_for_collective(state))
    rank, world = _state_collective_rank_world(state)
    sample = np.zeros(shape, dtype=dtype)
    group_tag = _replica_groups_tag(groups)
    barrier_name = f"dsv4_moe_tp_ar_t{shape[0]}_h{shape[1]}_tp{state.tp_degree}"
    return _compile_blockwise_collective_kernel(
        _tp_all_reduce_output_fn,
        sample,
        tp_degree=int(state.tp_degree),
        tp_replica_groups=groups,
        name=(
            f"dsv4_moe_tp_ar_t{shape[0]}_h{shape[1]}_tp{state.tp_degree}_{group_tag}"
        ),
        build_dir=str(artifacts_dir) if artifacts_dir else None,
        rank_id=rank,
        world_size=world,
        load_barrier_name=barrier_name,
    )


def _compile_ep_tp_all_reduce_kernel(
    shape: tuple[int, int],
    dtype: np.dtype,
    state: BlockwiseMoEState,
    *,
    artifacts_dir: str | Path | None = None,
) -> Any:
    groups = _moe_ep_tp_replica_groups_for_collective(state)
    rank, world = _state_collective_rank_world(state)
    sample = np.zeros(shape, dtype=dtype)
    group_tag = _replica_groups_tag(groups)
    barrier_name = (
        f"dsv4_moe_ep_tp_ar_t{shape[0]}_h{shape[1]}"
        f"_ep{state.ep_degree}_tp{state.tp_degree}"
    )
    return _compile_blockwise_collective_kernel(
        _ep_tp_all_reduce_output_fn,
        sample,
        replica_groups=groups,
        name=(
            f"dsv4_moe_ep_tp_ar_t{shape[0]}_h{shape[1]}"
            f"_ep{state.ep_degree}_tp{state.tp_degree}_{group_tag}"
        ),
        build_dir=str(artifacts_dir) if artifacts_dir else None,
        rank_id=rank,
        world_size=world,
        load_barrier_name=barrier_name,
    )


def precompile_blockwise_moe_all_reduce(
    *,
    rows: int,
    hidden_size: int | None = None,
    dtype: Any = ml_dtypes.bfloat16,
    state: BlockwiseMoEState,
    artifacts_dir: str | Path | None = None,
) -> None:
    """Compile blockwise-MoE collective kernels for a routed-output shape."""
    T = int(rows)
    H = int(hidden_size if hidden_size is not None else state.hidden_size)
    if T <= 0 or H <= 0:
        return
    shape = (T, H)
    np_dtype = np.dtype(dtype)
    if int(state.ep_degree) > 1 and int(state.tp_degree) > 1:
        cache_key = _ep_tp_all_reduce_cache_key(shape, np_dtype, state)
        if cache_key is not None:
            if cache_key not in state.ep_tp_all_reduce_kernel_cache:
                _require_unsealed_blockwise_cache(
                    state,
                    kind="ep_tp_all_reduce",
                    cache_key=cache_key,
                )
                state.ep_tp_all_reduce_kernel_cache[cache_key] = (
                    _compile_ep_tp_all_reduce_kernel(
                        shape,
                        np_dtype,
                        state,
                        artifacts_dir=artifacts_dir,
                    )
                )
            return
    if int(state.ep_degree) > 1:
        cache_key = _ep_all_reduce_cache_key(shape, np_dtype, state)
        if cache_key not in state.ep_all_reduce_kernel_cache:
            _require_unsealed_blockwise_cache(
                state,
                kind="ep_all_reduce",
                cache_key=cache_key,
            )
            state.ep_all_reduce_kernel_cache[cache_key] = _compile_ep_all_reduce_kernel(
                shape,
                np_dtype,
                state,
                artifacts_dir=artifacts_dir,
            )
    if int(state.tp_degree) > 1:
        cache_key = _tp_all_reduce_cache_key(shape, np_dtype, state)
        if cache_key not in state.tp_all_reduce_kernel_cache:
            _require_unsealed_blockwise_cache(
                state,
                kind="tp_all_reduce",
                cache_key=cache_key,
            )
            state.tp_all_reduce_kernel_cache[cache_key] = _compile_tp_all_reduce_kernel(
                shape,
                np_dtype,
                state,
                artifacts_dir=artifacts_dir,
            )


def _ep_all_reduce_output(
    output: Any,
    state: BlockwiseMoEState,
    *,
    artifacts_dir: str | Path | None = None,
    out: Any | None = None,
) -> Any:
    """Sum local-expert routed partials across the EP column."""
    if int(state.ep_degree) <= 1:
        return output
    if _is_device_tensor(output):
        from nkipy_serving.models._device_utils import _get_device_tensor_cls

        shape = _canonical_moe_output_shape(output, state)
        output = _alias_device_tensor_shape(output, shape)
        dtype = np.dtype(output.dtype)
        cache_key = _ep_all_reduce_cache_key(shape, dtype, state)
        kernel = state.ep_all_reduce_kernel_cache.get(cache_key)
        if kernel is None:
            _require_unsealed_blockwise_cache(
                state,
                kind="ep_all_reduce",
                cache_key=cache_key,
            )
            kernel = _compile_ep_all_reduce_kernel(
                shape,
                dtype,
                state,
                artifacts_dir=artifacts_dir,
            )
            state.ep_all_reduce_kernel_cache[cache_key] = kernel
        if out is None:
            out = _get_device_tensor_cls().from_numpy(
                np.zeros(shape, dtype=dtype),
                name="moe_ep_all_reduce",
            )
        else:
            out_shape = tuple(int(v) for v in getattr(out, "shape", ()))
            if out_shape != shape:
                raise RuntimeError(
                    f"MoE EP all-reduce output shape {out_shape} != {shape}"
                )
        kernel(inputs={"output": output}, outputs={"output0": out})
        return out

    import nkipy.distributed.collectives as cc

    return cc.all_reduce(
        output,
        replica_groups=_ep_replica_groups_for_collective(state),
        reduce_op=np.add,
    )


def _tp_all_reduce_output(
    output: Any,
    state: BlockwiseMoEState,
    *,
    artifacts_dir: str | Path | None = None,
    out: Any | None = None,
) -> Any:
    """Sum local-I routed expert partials across the TP row."""
    if int(state.tp_degree) <= 1:
        return output
    if _is_device_tensor(output):
        from nkipy_serving.models._device_utils import _get_device_tensor_cls

        shape = _canonical_moe_output_shape(output, state)
        output = _alias_device_tensor_shape(output, shape)
        dtype = np.dtype(output.dtype)
        cache_key = _tp_all_reduce_cache_key(shape, dtype, state)
        kernel = state.tp_all_reduce_kernel_cache.get(cache_key)
        if kernel is None:
            _require_unsealed_blockwise_cache(
                state,
                kind="tp_all_reduce",
                cache_key=cache_key,
            )
            kernel = _compile_tp_all_reduce_kernel(
                shape,
                dtype,
                state,
                artifacts_dir=artifacts_dir,
            )
            state.tp_all_reduce_kernel_cache[cache_key] = kernel
        if out is None:
            out = _get_device_tensor_cls().from_numpy(
                np.zeros(shape, dtype=dtype),
                name="moe_tp_all_reduce",
            )
        else:
            out_shape = tuple(int(v) for v in getattr(out, "shape", ()))
            if out_shape != shape:
                raise RuntimeError(
                    f"MoE TP all-reduce output shape {out_shape} != {shape}"
                )
        kernel(inputs={"output": output}, outputs={"output0": out})
        return out

    import nkipy.distributed.collectives as cc

    return cc.all_reduce(
        output,
        replica_groups=_tp_replica_groups_for_collective(state),
        reduce_op=np.add,
    )


def _ep_tp_all_reduce_output(
    output: Any,
    state: BlockwiseMoEState,
    *,
    artifacts_dir: str | Path | None = None,
    out: Any | None = None,
) -> Any | None:
    """Sum routed partials across the full EP×TP replica in one collective."""
    if int(state.ep_degree) <= 1 or int(state.tp_degree) <= 1:
        return None
    groups = _moe_ep_tp_replica_groups_for_collective(state)
    if not groups:
        return None
    if not _is_device_tensor(output):
        return None
    from nkipy_serving.models._device_utils import _get_device_tensor_cls

    shape = _canonical_moe_output_shape(output, state)
    output = _alias_device_tensor_shape(output, shape)
    dtype = np.dtype(output.dtype)
    cache_key = _ep_tp_all_reduce_cache_key(shape, dtype, state)
    if cache_key is None:
        return None
    kernel = state.ep_tp_all_reduce_kernel_cache.get(cache_key)
    if kernel is None:
        _require_unsealed_blockwise_cache(
            state,
            kind="ep_tp_all_reduce",
            cache_key=cache_key,
        )
        kernel = _compile_ep_tp_all_reduce_kernel(
            shape,
            dtype,
            state,
            artifacts_dir=artifacts_dir,
        )
        state.ep_tp_all_reduce_kernel_cache[cache_key] = kernel
    if out is None:
        out = _get_device_tensor_cls().from_numpy(
            np.zeros(shape, dtype=dtype),
            name="moe_ep_tp_all_reduce",
        )
    else:
        out_shape = tuple(int(v) for v in getattr(out, "shape", ()))
        if out_shape != shape:
            raise RuntimeError(
                f"MoE EPxTP all-reduce output shape {out_shape} != {shape}"
            )
    kernel(inputs={"output": output}, outputs={"output0": out})
    return out


def _moe_all_reduce_output(
    output: Any,
    state: BlockwiseMoEState,
    *,
    artifacts_dir: str | Path | None = None,
    ep_out: Any | None = None,
    tp_out: Any | None = None,
) -> Any:
    combined = _ep_tp_all_reduce_output(
        output,
        state,
        artifacts_dir=artifacts_dir,
        out=tp_out,
    )
    if combined is not None:
        return combined
    output = _ep_all_reduce_output(
        output,
        state,
        artifacts_dir=artifacts_dir,
        out=ep_out,
    )
    return _tp_all_reduce_output(
        output,
        state,
        artifacts_dir=artifacts_dir,
        out=tp_out,
    )


# ---------------------------------------------------------------------------
# Kernel wrappers (shape: emulate blockwise_add_residual without RS/residual)
# ---------------------------------------------------------------------------


def _beta2_prefill_compiler_args(logical_nc_config: int) -> str:
    """Keep the outer NKIPy NEFF compile aligned with the beta2 LNC kernel."""

    lnc = int(logical_nc_config)
    return "" if lnc == 1 else f"--lnc {lnc}"


def _make_prefill_router_wrapper_beta2(
    *,
    token_bucket: int,
    local_num_experts: int,
    experts_per_token: int,
    num_blocks: int,
    f_len: int,
    output_len: int,
    logical_nc_config: int,
    clamps: dict[str, Any],
) -> Callable[..., Any]:
    """Device-router prefill wrapper using full block-to-expert metadata."""

    from nkipy_serving.ops.moe.device_schedule import (
        make_prefill_fused_entry,
        wrap_nki_framework_kernel,
    )

    schedule_entry = make_prefill_fused_entry(
        token_bucket=int(token_bucket),
        local_num_experts=int(local_num_experts),
        experts_per_token=int(experts_per_token),
        num_blocks=int(num_blocks),
        f_len=int(f_len),
        output_len=int(output_len),
        logical_nc_config=int(logical_nc_config),
        compress_block_to_expert=False,
    )
    lnc = int(logical_nc_config)
    n_blocks = int(num_blocks)
    block = int(MOE_BLOCK_SIZE)
    gcu = float(clamps.get("gate_clamp_upper", 10.0))
    gcl = clamps.get("gate_clamp_lower", None)
    ucu = float(clamps.get("up_clamp_upper", 10.0))
    ucl = float(clamps.get("up_clamp_lower", -10.0))

    def _prefill_router_wrapper(
        hidden_states,
        router_weights_hbm,
        router_indices_hbm,
        ep_start,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
    ):
        del down_bias_broadcasted_hbm
        (
            expert_affinities_masked_hbm,
            token_position_to_id,
            block_to_expert,
        ) = schedule_entry(router_weights_hbm, router_indices_hbm, ep_start)
        return wrap_nki_framework_kernel(
            blockwise_nki_prefill_dsv4_beta2,
            lnc=lnc,
            args=(
                hidden_states,
                expert_affinities_masked_hbm,
                gate_up_proj_weight,
                gate_up_bias_plus1_T_hbm,
                down_proj_weight,
                token_position_to_id.reshape((n_blocks * block,)),
                block_to_expert,
            ),
            kwargs={
                "gate_clamp_upper": gcu,
                "gate_clamp_lower": gcl,
                "up_clamp_upper": ucu,
                "up_clamp_lower": ucl,
            },
        )

    return _prefill_router_wrapper


def _make_decode_router_wrapper(
    *,
    activation: Any,
    clamps: dict[str, Any],
    has_down_bias: bool,
) -> Callable[..., Any]:
    from nkipy.core.nki_op import wrap_nki_kernel

    gcu = clamps.get("gate_clamp_upper", 7.0)
    gcl = clamps.get("gate_clamp_lower", None)
    ucu = clamps.get("up_clamp_upper", 8.0)
    ucl = clamps.get("up_clamp_lower", -6.0)
    hdb = bool(has_down_bias)

    def _decode_router_wrapper(
        hidden_states,
        router_weights_hbm,
        router_indices_hbm,
        ep_start,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
    ):
        from nkipy.core import tensor_apis

        from nkipy_serving.ops.moe.device_schedule import (
            local_expert_affinities_dynamic_ep_fn,
        )

        T = int(hidden_states.shape[0])
        E = int(gate_up_proj_weight.shape[0])
        affinities_T = np.transpose(
            local_expert_affinities_dynamic_ep_fn(
                router_weights_hbm,
                router_indices_hbm,
                ep_start,
                local_num_experts=E,
            )
        )
        token_position_to_id = tensor_apis.full(
            (1, MOE_TILE_SIZE),
            -1,
            dtype=np.int32,
        )
        token_position_to_id[0, :T] = np.arange(T, dtype=np.int32) + tensor_apis.zeros(
            (T,), dtype=np.int32
        )
        token_position_to_id = np.broadcast_to(
            token_position_to_id,
            (E, MOE_TILE_SIZE),
        )
        block_to_expert = np.arange(E, dtype=np.int8) + tensor_apis.zeros(
            (E,),
            dtype=np.int8,
        )
        nki_op = wrap_nki_kernel(
            blockwise_nki_decode,
            [
                hidden_states,
                affinities_T,
                gate_up_proj_weight,
                gate_up_bias_plus1_T_hbm,
                down_proj_weight,
                down_bias_broadcasted_hbm,
                token_position_to_id,
                block_to_expert,
                activation,
                ml_dtypes.bfloat16,  # compute_dtype
                True,  # is_tensor_update_accumulating
                3,  # BUFFER_DEGREE (decode default)
                hdb,
                gcu,
                gcl,
                ucu,
                ucl,
            ],
        )
        return nki_op(
            hidden_states,
            affinities_T,
            gate_up_proj_weight,
            gate_up_bias_plus1_T_hbm,
            down_proj_weight,
            down_bias_broadcasted_hbm,
            token_position_to_id,
            block_to_expert,
        )

    return _decode_router_wrapper


def precompile_blockwise_moe_prefill_router(
    layer: LayerStackedExperts,
    *,
    rows: int,
    topk: int,
    router_weights_dtype: Any,
    state: BlockwiseMoEState,
    artifacts_dir: str | Path | None = None,
) -> None:
    """Compile the device-router prefill blockwise MoE kernel without running it."""
    from nkipy_serving.models._device_utils import _get_device_kernel_cls
    from nkipy_serving.ops.moe.device_schedule import (
        choose_indexed_flatten_f_len,
    )
    from nkipy_serving.ops.moe.device_schedule import (
        logical_nc_config as read_lnc_config,
    )

    T = int(rows)
    K = int(topk)
    E = int(layer.n_local_experts)
    H = int(layer.hidden_size)
    if T <= 0 or K <= 0 or E <= 0 or H <= 0:
        return
    num_blocks, _ = get_n_blocks(
        T,
        int(state.experts_per_token),
        E,
    )
    f_len = choose_indexed_flatten_f_len(T)
    output_len = int(num_blocks) * int(MOE_BLOCK_SIZE) + T
    logical_nc_config = read_lnc_config()
    has_down_bias = layer.down_bias_bc is not None
    cache_key = (
        "router" if has_down_bias else "router_beta2",
        T,
        K,
        E,
        int(state.experts_per_token),
        int(num_blocks),
        int(f_len),
        int(output_len),
        int(logical_nc_config),
        str(router_weights_dtype),
        bool(has_down_bias),
    )
    if cache_key in state.prefill_kernel_cache:
        return
    _require_unsealed_blockwise_cache(
        state,
        kind="prefill_router",
        cache_key=cache_key,
    )

    # DSV4 routed experts are bias-free (down_bias_bc is always None), so the
    # blockwise prefill router always takes the beta2 (no-down-bias) path. The
    # NEFF name/cache key still encode "_beta2"/"_db0"/"router_beta2" to stay
    # byte-identical to the historical warm cache.
    wrapper = _make_prefill_router_wrapper_beta2(
        token_bucket=T,
        local_num_experts=E,
        experts_per_token=int(state.experts_per_token),
        num_blocks=int(num_blocks),
        f_len=int(f_len),
        output_len=int(output_len),
        logical_nc_config=int(logical_nc_config),
        clamps=state._v4_clamp_kwargs(),
    )
    compile_kwargs = {
        "hidden_states": np.zeros((T, H), dtype=ml_dtypes.bfloat16),
        "router_weights_hbm": np.zeros((T, K), dtype=router_weights_dtype),
        "router_indices_hbm": np.zeros((T, K), dtype=np.int32),
        "ep_start": np.zeros((1,), dtype=np.int32),
        "gate_up_proj_weight": _sample_like(layer.gate_up_w),
        "gate_up_bias_plus1_T_hbm": _sample_like(layer.gate_up_bias),
        "down_proj_weight": _sample_like(layer.down_w),
        "down_bias_broadcasted_hbm": _down_bias_sample(layer.down_bias_bc),
        "name": (
            f"dsv4_moe_prefill_router_blockwise"
            f"{'_beta2' if not has_down_bias else ''}_t{T}_k{K}_e{E}"
            f"_n{int(num_blocks)}_f{int(f_len)}_o{int(output_len)}"
            f"_lnc{int(logical_nc_config)}_db{int(has_down_bias)}"
        ),
        "build_dir": artifacts_dir,
        "namespace": "blockwise_moe",
    }
    beta2_args = _beta2_prefill_compiler_args(logical_nc_config)
    if beta2_args:
        compile_kwargs["additional_compiler_args"] = beta2_args
    state.prefill_kernel_cache[cache_key] = compile_and_load_with_lock(
        _get_device_kernel_cls(),
        wrapper,
        **compile_kwargs,
    )


def precompile_blockwise_moe_decode_router(
    layer: LayerStackedExperts,
    *,
    rows: int,
    topk: int,
    router_weights_dtype: Any,
    state: BlockwiseMoEState,
    artifacts_dir: str | Path | None = None,
) -> None:
    """Compile the device-router decode blockwise MoE kernel without running it."""
    from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType

    from nkipy_serving.models._device_utils import _get_device_kernel_cls

    T = int(rows)
    K = int(topk)
    E = int(layer.n_local_experts)
    H = int(layer.hidden_size)
    if T <= 0 or K <= 0 or E <= 0 or H <= 0:
        return
    if T > MOE_BLOCK_SIZE:
        raise ValueError(f"decode path requires T<={MOE_BLOCK_SIZE}, got {T}")
    has_down_bias = layer.down_bias_bc is not None
    cache_key = ("router", T, K, bool(has_down_bias))
    if cache_key in state.decode_kernel_cache:
        return
    _require_unsealed_blockwise_cache(
        state,
        kind="decode_router",
        cache_key=cache_key,
    )

    wrapper = _make_decode_router_wrapper(
        activation=ActFnType.SiLU,
        clamps=state._v4_clamp_kwargs(),
        has_down_bias=has_down_bias,
    )
    state.decode_kernel_cache[cache_key] = compile_and_load_with_lock(
        _get_device_kernel_cls(),
        wrapper,
        hidden_states=np.zeros((T, H), dtype=ml_dtypes.bfloat16),
        router_weights_hbm=np.zeros((T, K), dtype=router_weights_dtype),
        router_indices_hbm=np.zeros((T, K), dtype=np.int32),
        ep_start=np.zeros((1,), dtype=np.int32),
        gate_up_proj_weight=_sample_like(layer.gate_up_w),
        gate_up_bias_plus1_T_hbm=_sample_like(layer.gate_up_bias),
        down_proj_weight=_sample_like(layer.down_w),
        down_bias_broadcasted_hbm=_down_bias_sample(layer.down_bias_bc),
        name=f"dsv4_moe_decode_router_blockwise_t{T}_k{K}_e{E}_db{int(has_down_bias)}",
        build_dir=artifacts_dir,
        namespace="blockwise_moe",
    )


# ---------------------------------------------------------------------------
# Prefill / decode entry points
# ---------------------------------------------------------------------------


def run_blockwise_moe_prefill(
    layer: LayerStackedExperts,
    *,
    hidden_states: Any,  # [T, H] — numpy bf16/fp32 or SpikeTensor bf16
    weights: Any,  # [T, K] fp32/bf16 DeviceTensor
    indices: Any,  # [T, K] int32 DeviceTensor, global expert ids
    state: BlockwiseMoEState,
    artifacts_dir: str | Path | None = None,
    return_device: bool = False,
    output: Any | None = None,
    ep_output: Any | None = None,
    tp_output: Any | None = None,
    skip_all_reduce: bool = False,
) -> "np.ndarray | Any":
    """Run one prefill MoE layer.

    ``weights`` and ``indices`` must be device-resident router top-k tensors.
    ``hidden_states`` may be a numpy array or an already-resident
    DeviceTensor/SpikeTensor whose dtype matches the kernel's bf16 contract.
    By default returns ``[T, H]`` fp32 numpy (kernel output is bf16).
    Pass ``return_device=True`` to get the raw bf16 DeviceTensor instead,
    so callers can chain device-side ops (e.g. shared-expert residual add)
    without a round-trip.
    """
    from nkipy_serving.models._device_utils import (
        _get_device_kernel_cls,
        _get_device_tensor_cls,
    )

    T, H = hidden_states.shape
    T = int(T)
    H = int(H)
    E = layer.n_local_experts
    router_on_device = _is_device_tensor(weights) and _is_device_tensor(indices)
    if not router_on_device:
        raise RuntimeError(
            "DSV4 blockwise MoE prefill requires device-resident router "
            "weights and indices"
        )
    from nkipy_serving.ops.moe.device_schedule import (
        choose_indexed_flatten_f_len,
    )
    from nkipy_serving.ops.moe.device_schedule import (
        logical_nc_config as read_lnc_config,
    )

    K = int(weights.shape[1])
    num_blocks, _ = get_n_blocks(
        T,
        int(state.experts_per_token),
        E,
    )
    f_len = choose_indexed_flatten_f_len(T)
    output_len = int(num_blocks) * int(MOE_BLOCK_SIZE) + T
    logical_nc_config = read_lnc_config()

    DK = _get_device_kernel_cls()
    DT = _get_device_tensor_cls()

    is_dev_hidden = _is_device_tensor(hidden_states)
    if is_dev_hidden:
        hidden_dev = hidden_states
        # Kernel-compile sample needs only shape and dtype.
        hidden_sample = np.zeros((T, H), dtype=ml_dtypes.bfloat16)
    else:
        hidden_bf = _cast_bf16(hidden_states)
        hidden_sample = hidden_bf
        hidden_dev = DT.from_numpy(hidden_bf, name="hidden")
    output_sample = np.zeros((T, H), dtype=ml_dtypes.bfloat16)
    gate_up_sample = _sample_like(layer.gate_up_w)
    down_sample = _sample_like(layer.down_w)
    gate_up_bias_sample = _sample_like(layer.gate_up_bias)
    has_down_bias = layer.down_bias_bc is not None
    use_beta2_prefill = not bool(has_down_bias)
    down_bias_sample = _down_bias_sample(layer.down_bias_bc)

    cache_key = (
        "router_beta2" if use_beta2_prefill else "router",
        int(T),
        int(K),
        int(E),
        int(state.experts_per_token),
        int(num_blocks),
        int(f_len),
        int(output_len),
        int(logical_nc_config),
        str(getattr(weights, "dtype", "unknown")),
        bool(has_down_bias),
    )
    moe_k = state.prefill_kernel_cache.get(cache_key)
    if moe_k is None:
        _require_unsealed_blockwise_cache(
            state,
            kind="prefill_router",
            cache_key=cache_key,
        )
        # DSV4 is bias-free, so use_beta2_prefill is always True (see the
        # precompile path); name/cache key still emit "_beta2"/"_db0".
        wrapper = _make_prefill_router_wrapper_beta2(
            token_bucket=T,
            local_num_experts=E,
            experts_per_token=int(state.experts_per_token),
            num_blocks=int(num_blocks),
            f_len=int(f_len),
            output_len=int(output_len),
            logical_nc_config=int(logical_nc_config),
            clamps=state._v4_clamp_kwargs(),
        )
        weights_sample_dtype = getattr(weights, "dtype", np.float32)
        compile_kwargs = {
            "hidden_states": hidden_sample,
            "router_weights_hbm": np.zeros((T, K), dtype=weights_sample_dtype),
            "router_indices_hbm": np.zeros((T, K), dtype=np.int32),
            "ep_start": np.zeros((1,), dtype=np.int32),
            "gate_up_proj_weight": gate_up_sample,
            "gate_up_bias_plus1_T_hbm": gate_up_bias_sample,
            "down_proj_weight": down_sample,
            "down_bias_broadcasted_hbm": down_bias_sample,
            "name": (
                f"dsv4_moe_prefill_router_blockwise"
                f"{'_beta2' if use_beta2_prefill else ''}_t{T}_k{K}_e{E}"
                f"_n{num_blocks}_f{f_len}_o{output_len}"
                f"_lnc{logical_nc_config}_db{int(has_down_bias)}"
            ),
            "build_dir": artifacts_dir,
            "namespace": "blockwise_moe",
        }
        beta2_args = _beta2_prefill_compiler_args(logical_nc_config)
        if beta2_args:
            compile_kwargs["additional_compiler_args"] = beta2_args
        moe_k = compile_and_load_with_lock(DK, wrapper, **compile_kwargs)
        state.prefill_kernel_cache[cache_key] = moe_k

    if output is None:
        output_dev = DT.from_numpy(output_sample, name="moe_out")
    else:
        out_shape = tuple(int(v) for v in getattr(output, "shape", ()))
        if out_shape != (T, H):
            raise RuntimeError(f"MoE prefill output shape {out_shape} != {(T, H)}")
        output_dev = output
    ep_start_dev = DT.from_numpy(
        np.asarray([int(state.ep_rank) * E], dtype=np.int32),
        name="moe_ep_start",
    )
    gu_w_dev = _to_device_tensor(layer.gate_up_w, name="gu_w", tensor_cls=DT)
    dn_w_dev = _to_device_tensor(layer.down_w, name="dn_w", tensor_cls=DT)
    gu_b_dev = _to_device_tensor(layer.gate_up_bias, name="gu_b", tensor_cls=DT)
    dn_b_dev = _down_bias_device_tensor(layer.down_bias_bc, tensor_cls=DT)

    inputs = {
        "hidden_states": hidden_dev,
        "router_weights_hbm": weights,
        "router_indices_hbm": indices,
        "ep_start": ep_start_dev,
        "gate_up_proj_weight": gu_w_dev,
        "gate_up_bias_plus1_T_hbm": gu_b_dev,
        "down_proj_weight": dn_w_dev,
        "down_bias_broadcasted_hbm": dn_b_dev,
    }
    if use_beta2_prefill:
        moe_k(inputs=inputs, outputs={"output0": output_dev})
    else:
        inputs["output.must_alias_input"] = output_dev
        moe_k(inputs=inputs, outputs={"output": output_dev})
    if not bool(skip_all_reduce):
        output_dev = _moe_all_reduce_output(
            output_dev,
            state,
            artifacts_dir=artifacts_dir,
            ep_out=ep_output,
            tp_out=tp_output,
        )
    if return_device:
        return output_dev
    return output_dev.numpy().astype(np.float32)


def run_blockwise_moe_decode(
    layer: LayerStackedExperts,
    *,
    hidden_states: Any,  # [T, H] — numpy or SpikeTensor bf16
    weights: Any,
    indices: Any,
    state: BlockwiseMoEState,
    artifacts_dir: str | Path | None = None,
    return_device: bool = False,
    output: Any | None = None,
    ep_output: Any | None = None,
    tp_output: Any | None = None,
    skip_all_reduce: bool = False,
) -> "np.ndarray | Any":
    """Run one decode MoE layer. Requires ``T <= MOE_BLOCK_SIZE``.

    ``weights`` and ``indices`` must be device-resident router top-k tensors.
    ``hidden_states`` may be a SpikeTensor — see
    ``run_blockwise_moe_prefill`` for the contract. Pass
    ``return_device=True`` to get the raw bf16 DeviceTensor output.
    """
    from nkipy_serving.models._device_utils import (
        _get_device_kernel_cls,
        _get_device_tensor_cls,
    )

    T, H = hidden_states.shape
    T = int(T)
    H = int(H)
    if T > MOE_BLOCK_SIZE:
        raise ValueError(f"decode path requires T<={MOE_BLOCK_SIZE}, got {T}")
    E = layer.n_local_experts
    router_on_device = _is_device_tensor(weights) and _is_device_tensor(indices)
    if not router_on_device:
        raise RuntimeError(
            "DSV4 blockwise MoE decode requires device-resident router "
            "weights and indices"
        )

    DK = _get_device_kernel_cls()
    DT = _get_device_tensor_cls()

    is_dev_hidden = _is_device_tensor(hidden_states)
    if is_dev_hidden:
        hidden_dev = hidden_states
        hidden_sample = np.zeros((T, H), dtype=ml_dtypes.bfloat16)
    else:
        hidden_bf = _cast_bf16(hidden_states)
        hidden_sample = hidden_bf
        hidden_dev = DT.from_numpy(hidden_bf, name="hidden")
    gate_up_sample = _sample_like(layer.gate_up_w)
    down_sample = _sample_like(layer.down_w)
    gate_up_bias_sample = _sample_like(layer.gate_up_bias)
    has_down_bias = layer.down_bias_bc is not None
    down_bias_sample = _down_bias_sample(layer.down_bias_bc)

    K = int(weights.shape[1])
    cache_key = ("router", int(T), int(K), bool(has_down_bias))
    moe_k = state.decode_kernel_cache.get(cache_key)
    if moe_k is None:
        _require_unsealed_blockwise_cache(
            state,
            kind="decode_router",
            cache_key=cache_key,
        )
        from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType

        wrapper = _make_decode_router_wrapper(
            activation=ActFnType.SiLU,
            clamps=state._v4_clamp_kwargs(),
            has_down_bias=has_down_bias,
        )
        weights_sample_dtype = getattr(weights, "dtype", np.float32)
        moe_k = compile_and_load_with_lock(
            DK,
            wrapper,
            hidden_states=hidden_sample,
            router_weights_hbm=np.zeros((T, K), dtype=weights_sample_dtype),
            router_indices_hbm=np.zeros((T, K), dtype=np.int32),
            ep_start=np.zeros((1,), dtype=np.int32),
            gate_up_proj_weight=gate_up_sample,
            gate_up_bias_plus1_T_hbm=gate_up_bias_sample,
            down_proj_weight=down_sample,
            down_bias_broadcasted_hbm=down_bias_sample,
            name=(
                f"dsv4_moe_decode_router_blockwise_t{T}_k{K}_e{E}"
                f"_db{int(has_down_bias)}"
            ),
            build_dir=artifacts_dir,
            namespace="blockwise_moe",
        )
        state.decode_kernel_cache[cache_key] = moe_k

    ep_start_dev = DT.from_numpy(
        np.asarray([int(state.ep_rank) * E], dtype=np.int32),
        name="ep_start",
    )
    gu_w_dev = _to_device_tensor(layer.gate_up_w, name="gu_w", tensor_cls=DT)
    dn_w_dev = _to_device_tensor(layer.down_w, name="dn_w", tensor_cls=DT)
    gu_b_dev = _to_device_tensor(layer.gate_up_bias, name="gu_b", tensor_cls=DT)
    dn_b_dev = _down_bias_device_tensor(layer.down_bias_bc, tensor_cls=DT)
    if output is None:
        out_dev = DT.from_numpy(
            np.zeros((T, H), dtype=ml_dtypes.bfloat16),
            name="out",
        )
    else:
        out_shape = tuple(int(v) for v in getattr(output, "shape", ()))
        if out_shape != (T, H):
            raise RuntimeError(f"MoE decode output shape {out_shape} != {(T, H)}")
        out_dev = output

    moe_k(
        inputs={
            "hidden_states": hidden_dev,
            "router_weights_hbm": weights,
            "router_indices_hbm": indices,
            "ep_start": ep_start_dev,
            "gate_up_proj_weight": gu_w_dev,
            "gate_up_bias_plus1_T_hbm": gu_b_dev,
            "down_proj_weight": dn_w_dev,
            "down_bias_broadcasted_hbm": dn_b_dev,
        },
        outputs={"output0": out_dev},
    )
    if not bool(skip_all_reduce):
        out_dev = _moe_all_reduce_output(
            out_dev,
            state,
            artifacts_dir=artifacts_dir,
            ep_out=ep_output,
            tp_out=tp_output,
        )
    if return_device:
        return out_dev
    return out_dev.numpy().astype(np.float32)
