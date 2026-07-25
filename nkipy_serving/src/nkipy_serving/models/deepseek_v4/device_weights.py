"""Device weight upload for DeepSeek-V4-Flash.

The product sampled path consumes preprocessed MoE tensors: routed experts are
Neuron-range FP8 E4M3 with no runtime scales, and shared experts are BF16. The
loader fails closed unless the checkpoint already provides this layout, so
serving never JIT-converts or keeps duplicate MoE formats in HBM.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, fields, is_dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable

import ml_dtypes
import numpy as np

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

from nkipy_serving.models._device_utils import _get_device_tensor_cls
from nkipy_serving.models.deepseek_v4.config import (
    DeepseekV4ModelConfig,
    DeepseekV4Weights,
)
from nkipy_serving.models.deepseek_v4.weight_reader import (
    V4WeightReader,
    _ShardReader,
    dequant_mxfp4_block,
)
from nkipy_serving.models.reload_utils import resolve_model_snapshot_path
from nkipy_serving.ops.nn import select_head_rows as _select_head_rows
from nkipy_serving.profiling import StartupProfiler

logger = logging.getLogger(__name__)


def _as_neuron_fp8_for_upload(arr: np.ndarray) -> np.ndarray:
    """View rescaled OCP FP8 bytes as Neuron-native E4M3 for HBM upload.

    NKIPy's NEFF metadata exposes FP8 inputs as ``int8``. The runtime accepts
    ``ml_dtypes.float8_e4m3`` as the compatible FP8 tensor dtype, but not the
    OCP ``float8_e4m3fn`` tag. After the loader clamps/rescales to Neuron's
    ±240 range, the byte layout is safe to reinterpret under the native tag.
    """
    if arr.dtype == ml_dtypes.float8_e4m3fn:
        return arr.view(ml_dtypes.float8_e4m3)
    return arr


def _upload(arr: np.ndarray, *, name: str):
    dt_cls = _get_device_tensor_cls()
    return dt_cls.from_numpy(
        np.ascontiguousarray(_as_neuron_fp8_for_upload(arr)),
        name=name,
    )


@dataclass(frozen=True)
class _StagedUpload:
    """Host-side tensor prepared for later HBM upload."""

    array: np.ndarray
    name: str

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype


def _stage_upload(arr: np.ndarray, *, name: str) -> _StagedUpload:
    return _StagedUpload(array=arr, name=name)


def _materialize_staged_uploads(value: Any) -> Any:
    if isinstance(value, _StagedUpload):
        return _upload(value.array, name=value.name)
    if isinstance(value, list):
        return [_materialize_staged_uploads(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            setattr(
                value, item.name, _materialize_staged_uploads(getattr(value, item.name))
            )
        return value
    return value


def _bf16_view_or_cast(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == ml_dtypes.bfloat16:
        return arr
    return arr.astype(ml_dtypes.bfloat16, copy=False)


def _weight_load_rank_label(v4_meta: DeepseekV4Weights) -> str:
    return (
        f"tp={int(v4_meta.tp_rank)}/{int(v4_meta.tp_degree)} "
        f"ep={int(v4_meta.ep_rank)}/{int(v4_meta.ep_degree)} "
        f"replica={int(v4_meta.replica_rank)}/{int(v4_meta.replica_degree)}"
    )


def _weight_load_global_rank(v4_meta: DeepseekV4Weights) -> int:
    return int(v4_meta.attention_lane) * int(v4_meta.tp_degree) + int(v4_meta.tp_rank)


def _weight_load_progress_enabled(v4_meta: DeepseekV4Weights) -> bool:
    flag = os.getenv("NKIPY_SERVING_WEIGHT_LOAD_PROGRESS", "1").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    all_ranks = (
        os.getenv(
            "NKIPY_SERVING_WEIGHT_LOAD_PROGRESS_ALL_RANKS",
            "0",
        )
        .strip()
        .lower()
    )
    if all_ranks in {"1", "true", "on", "yes"}:
        return True
    return (
        int(v4_meta.tp_rank) == 0
        and int(v4_meta.ep_rank) == 0
        and int(v4_meta.replica_rank) == 0
    )


def _weight_load_prefetch_enabled() -> bool:
    flag = os.getenv("NKIPY_SERVING_WEIGHT_LOAD_PREFETCH", "1").strip().lower()
    return flag not in {"0", "false", "off", "no"}


def _weight_load_log(
    v4_meta: DeepseekV4Weights,
    message: str,
    *,
    force: bool = False,
) -> None:
    if not force and not _weight_load_progress_enabled(v4_meta):
        return
    logger.info(
        "DSV4 weight load %s %s",
        _weight_load_rank_label(v4_meta),
        message,
    )


def _load_plan_summary(plan: "V4LoadPlan") -> str:
    enabled = []
    for name in (
        "dense",
        "hc",
        "moe",
        "attention",
        "compressor",
        "indexer",
        "blockwise_moe_fp8",
    ):
        if bool(getattr(plan, name)):
            enabled.append(name)
    return ",".join(enabled) if enabled else "none"


class _WeightLayerProgress:
    def __init__(self, total: int, v4_meta: DeepseekV4Weights) -> None:
        self._total = int(total)
        self._v4_meta = v4_meta
        self._enabled = _weight_load_progress_enabled(v4_meta)
        self._start = time.monotonic()
        self._last_log = self._start
        self._bar = None
        if self._enabled and _tqdm is not None:
            self._bar = _tqdm(
                total=self._total,
                desc=f"DSV4 weight layers {_weight_load_rank_label(v4_meta)}",
                unit="layer",
                dynamic_ncols=True,
                mininterval=5.0,
                file=sys.stderr,
            )
        elif self._enabled:
            _weight_load_log(v4_meta, f"layers 0/{self._total}")

    def update(self, layer_id: int) -> None:
        done = int(layer_id) + 1
        if self._bar is not None:
            self._bar.set_postfix_str(f"L{int(layer_id)}", refresh=False)
            self._bar.update(1)
            return
        if not self._enabled:
            return
        now = time.monotonic()
        if done == 1 or done == self._total or now - self._last_log >= 10.0:
            _weight_load_log(
                self._v4_meta,
                f"layers {done}/{self._total} elapsed={now - self._start:.1f}s",
            )
            self._last_log = now

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


# ---------------------------------------------------------------------------
# OCP ±448 → Neuron ±240 FP8 rescale
# ---------------------------------------------------------------------------


_DSV4_CONVERSION_FP8_NOSCALE = "neuron_fp8_noscale"


def _snapshot_conversion(snapshot_path: Path) -> str | None:
    index_path = Path(snapshot_path) / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    try:
        with index_path.open("r", encoding="utf-8") as f:
            idx = json.load(f)
        meta = idx.get("metadata") or {}
        value = meta.get("dsv4_conversion")
    except (OSError, JSONDecodeError, AttributeError, TypeError):
        return None
    return str(value) if value is not None else None


def _snapshot_is_fp8_noscale(snapshot_path: Path) -> bool:
    """Detect the product FP8-no-scale checkpoint layout."""
    return _snapshot_conversion(snapshot_path) == _DSV4_CONVERSION_FP8_NOSCALE


# ---------------------------------------------------------------------------
# Load plan (opt-in module set)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V4LoadPlan:
    """Which weight families to load."""

    dense: bool = True  # embed, final_norm, lm_head.
    hc: bool = True  # block hc_{attn,ffn}_{fn,base,scale}, final hc_head.
    moe: bool = True  # gate, routed experts, shared experts.
    attention: bool = False  # wq_a, wq_b, wkv, wo_a, wo_b, q_norm, kv_norm.
    compressor: bool = False
    indexer: bool = False
    blockwise_moe_fp8: bool = True  # Product no-scale FP8 blockwise MoE.

    @classmethod
    def sampled(cls) -> "V4LoadPlan":
        """Target-model sampled serving weights."""
        return cls(
            dense=True,
            hc=True,
            moe=True,
            attention=True,
            compressor=True,
            indexer=True,
            blockwise_moe_fp8=True,
        )

    @classmethod
    def sampled_blockwise_fp8(cls) -> "V4LoadPlan":
        """Target-model product path with no-scale FP8 blockwise MoE."""
        return cls(
            dense=True,
            hc=True,
            moe=True,
            attention=True,
            compressor=True,
            indexer=True,
            blockwise_moe_fp8=True,
        )


# ---------------------------------------------------------------------------
# Device weight containers
# ---------------------------------------------------------------------------


@dataclass
class V4LayerDeviceWeights:
    """Per-layer device tensors."""

    layer_id: int

    # RMS norms.
    attn_norm: Any = None  # [hidden] BF16
    ffn_norm: Any = None  # [hidden] BF16

    # HC block parameters.
    hc_attn_fn: Any = None  # [mix_hc, hc*hidden] FP32
    hc_attn_base: Any = None  # [mix_hc] FP32
    hc_attn_scale: Any = None  # [3] FP32
    hc_ffn_fn: Any = None
    hc_ffn_base: Any = None
    hc_ffn_scale: Any = None

    # MoE gate. For hash layers (layer_id < num_hash_layers), `tid2eid`
    # replaces `gate_weight` and `gate_bias` is None.
    gate_weight: Any = None  # [num_routed_experts, hidden] BF16
    gate_bias: Any = None  # [num_routed_experts] FP32 — None for hash layers
    gate_tid2eid: Any = None  # [vocab_size, topk] int32 — None for learned layers

    # Attention projection path. Linear weights are BF16 uploads from either
    # BF16/F32 tensors or FP8+scale dequant. FP8-retained dense attention is a
    # later performance pass.
    attn_sink: Any = None  # [local_n_heads] FP32
    wq_a: Any = None  # [q_lora_rank, hidden] BF16
    q_norm: Any = None  # [q_lora_rank] BF16
    wq_b: Any = None  # [local_n_heads*head_dim, q_lora_rank] BF16
    wkv: Any = None  # [head_dim, hidden] BF16
    kv_norm: Any = None  # [head_dim] BF16
    wo_a: Any = None  # [local_o_groups*o_lora, heads_per_group*head_dim] BF16
    wo_b: Any = None  # [hidden, local_o_groups*o_lora] BF16

    # Compressor / indexer projection weights.
    comp_wkv: Any = None
    comp_wgate: Any = None
    comp_ape: Any = None
    comp_norm: Any = None
    idx_wq_b: Any = None
    idx_weights_proj: Any = None
    idx_comp_wkv: Any = None
    idx_comp_wgate: Any = None
    idx_comp_ape: Any = None
    idx_comp_norm: Any = None

    # Shared expert — BF16 product path. This is small relative to routed MoE,
    # so keep it BF16 while routed experts carry the HBM savings.
    shared_w1: Any = None
    shared_w2: Any = None
    shared_w3: Any = None
    shared_tp_sharded: bool = False

    # Product blockwise MoE tensors. Routed experts are preprocessed offline
    # from HF MXFP4 into Neuron-range FP8 values with no runtime scales.
    blockwise_gate_up_w: Any = None  # [E_local, H, 2, I_local] FP8
    blockwise_down_w: Any = None  # [E_local, I_local, H] FP8
    blockwise_gate_up_bias: Any = None  # [E_local, I_local, 2] BF16
    blockwise_down_bias_bc: Any = None  # Optional [E_local, TILE_SIZE, H] BF16


@dataclass
class V4DeviceWeights:
    """Device-resident DSV4 weights."""

    embed: Any = None  # [local_embed_vocab, hidden] BF16
    embed_vocab_offset: int = 0
    embed_vocab_end: int = 0
    embed_tp_sharded: bool = False
    final_norm: Any = None  # [hidden] BF16
    lm_head: Any = None  # [local_vocab_size, hidden] BF16

    hc_head_fn: Any = None  # [hc_mult, hc_mult*hidden] FP32
    hc_head_base: Any = None  # [hc_mult] FP32
    hc_head_scale: Any = None  # [1] FP32

    layers: list[V4LayerDeviceWeights] = field(default_factory=list)

    # True when the snapshot was produced by the product pre-conversion
    # script. Informational only; main-layer MoE loading fails closed without
    # this metadata.
    fp8_noscale_snapshot: bool = False


_DSV4_PREPARED_WEIGHT_CACHE_VERSION = 1
_DENSE_PREPARED_KEYS = (
    "embed",
    "final_norm",
    "lm_head",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
)
_LAYER_PREPARED_KEYS = tuple(
    item.name
    for item in fields(V4LayerDeviceWeights)
    if item.name not in {"layer_id", "shared_tp_sharded", "blockwise_down_bias_bc"}
)


def _prepared_weight_rank_dir(root: Path, v4_meta: DeepseekV4Weights) -> Path:
    """Return the rank-local cache directory under a shared prepared root."""
    direct_meta = root / "metadata.json"
    if direct_meta.exists():
        return root
    return (
        root
        / f"tp{int(v4_meta.tp_degree)}_ep{int(v4_meta.ep_degree)}_rep{int(v4_meta.replica_degree)}"
        / f"lane{int(v4_meta.attention_lane):02d}_tp{int(v4_meta.tp_rank):02d}"
    )


def _prepared_weight_rank_dir_for(
    root: Path,
    *,
    tp_degree: int,
    ep_degree: int,
    replica_degree: int,
    lane: int,
    tp_rank: int,
) -> Path:
    return (
        root
        / f"tp{int(tp_degree)}_ep{int(ep_degree)}_rep{int(replica_degree)}"
        / f"lane{int(lane):02d}_tp{int(tp_rank):02d}"
    )


def prepared_weight_rank_dir(root: Path, v4_meta: DeepseekV4Weights) -> Path:
    """Public helper used by the offline prepared-weight script."""
    return _prepared_weight_rank_dir(Path(root), v4_meta)


def prepared_weight_rank_dir_for(
    root: Path,
    *,
    tp_degree: int,
    ep_degree: int,
    replica_degree: int,
    lane: int,
    tp_rank: int,
) -> Path:
    """Public helper for resolving a rank-local prepared-weight directory."""
    return _prepared_weight_rank_dir_for(
        Path(root),
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        replica_degree=replica_degree,
        lane=lane,
        tp_rank=tp_rank,
    )


def prepared_weight_local_rank_dir(
    source_root: Path,
    rank_dir: Path,
    local_root: Path,
) -> Path:
    """Return the local-cache mirror path for a prepared rank directory."""
    source_key = hashlib.sha1(
        str(Path(source_root).resolve()).encode("utf-8"),
    ).hexdigest()[:16]
    return (
        Path(local_root)
        / source_key
        / _rank_dir_relative_to_root(Path(source_root), Path(rank_dir))
    )


def _rank_dir_relative_to_root(root: Path, rank_dir: Path) -> Path:
    try:
        return rank_dir.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(rank_dir.name)


def _copytree_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(
        tempfile.mkdtemp(
            prefix=f".{dst.name}.",
            dir=str(dst.parent),
        )
    )
    try:
        shutil.copytree(src, tmp, dirs_exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        os.replace(tmp, dst)
    except BaseException:
        # Remove partial local copies even if staging is interrupted.
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def stage_prepared_weight_rank_dir_local(
    source_root: Path,
    rank_dir: Path,
    local_root: Path,
    *,
    log_fn: Callable[[str], None] | None = None,
) -> Path:
    """Copy one prepared rank dir to local storage if missing or stale."""
    source_root = Path(source_root)
    rank_dir = Path(rank_dir)
    local_root = Path(local_root)
    if rank_dir.resolve().is_relative_to(local_root.resolve()):
        return rank_dir

    source_meta = rank_dir / "metadata.json"
    if not source_meta.exists():
        return rank_dir
    source_meta_bytes = source_meta.read_bytes()
    local_rank_dir = prepared_weight_local_rank_dir(
        source_root,
        rank_dir,
        local_root,
    )
    local_meta = local_rank_dir / "metadata.json"
    if local_meta.exists() and local_meta.read_bytes() == source_meta_bytes:
        return local_rank_dir

    local_rank_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = local_rank_dir.parent / f".{local_rank_dir.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if local_meta.exists() and local_meta.read_bytes() == source_meta_bytes:
                return local_rank_dir
            if log_fn is not None:
                log_fn(
                    "prepared-cache local-stage start "
                    f"src={rank_dir} dst={local_rank_dir}"
                )
            copy_t0 = time.monotonic()
            _copytree_atomic(rank_dir, local_rank_dir)
            if not local_meta.exists() or local_meta.read_bytes() != source_meta_bytes:
                raise RuntimeError(
                    "Prepared DSV4 local-stage metadata mismatch after copy: "
                    f"src={rank_dir}, dst={local_rank_dir}"
                )
            if log_fn is not None:
                log_fn(
                    "prepared-cache local-stage done "
                    f"dst={local_rank_dir} elapsed={time.monotonic() - copy_t0:.1f}s"
                )
            return local_rank_dir
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _stage_prepared_weight_rank_dir_local(
    source_root: Path,
    rank_dir: Path,
    v4_meta: DeepseekV4Weights,
) -> Path:
    local_raw = (v4_meta.dsv4_prepared_weight_local_dir or "").strip()
    if not local_raw:
        return rank_dir
    return stage_prepared_weight_rank_dir_local(
        source_root,
        rank_dir,
        Path(local_raw),
        log_fn=lambda message: _weight_load_log(v4_meta, message),
    )


def _prepared_weight_cache_dir(v4_meta: DeepseekV4Weights) -> Path | None:
    raw = (v4_meta.dsv4_prepared_weight_dir or "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser()
    if not root.exists():
        raise RuntimeError(f"Prepared DSV4 weight cache does not exist: {root}")
    if (root / "metadata.json").exists():
        return _stage_prepared_weight_rank_dir_local(root, root, v4_meta)
    rank_dir = _prepared_weight_rank_dir(root, v4_meta)
    if not (rank_dir / "metadata.json").exists():
        # Replica ranks share the same expert row weights. Map absolute
        # attention lanes (for example 8..15 at R2) back to replica-zero lanes.
        replica_zero_lane = int(v4_meta.attention_lane) % int(v4_meta.ep_degree)
        replica_zero_dir = _prepared_weight_rank_dir_for(
            root,
            tp_degree=int(v4_meta.tp_degree),
            ep_degree=int(v4_meta.ep_degree),
            replica_degree=int(v4_meta.replica_degree),
            lane=replica_zero_lane,
            tp_rank=int(v4_meta.tp_rank),
        )
        if (replica_zero_dir / "metadata.json").exists():
            return _stage_prepared_weight_rank_dir_local(
                root,
                replica_zero_dir,
                v4_meta,
            )
        raise RuntimeError(
            "Prepared DSV4 weight cache is configured but rank metadata is "
            f"missing: expected {rank_dir / 'metadata.json'}"
            f" or {replica_zero_dir / 'metadata.json'}"
        )
    return _stage_prepared_weight_rank_dir_local(root, rank_dir, v4_meta)


def _copy_prepared_tensor(arr: np.ndarray) -> np.ndarray:
    """Materialize a prepared mmap view into host DRAM before HBM upload."""
    return np.array(arr, copy=True, order="C")


def _load_prepared_safetensors(
    path: Path,
    *,
    copy_to_host: bool = False,
) -> dict[str, np.ndarray]:
    shard = _ShardReader(path)
    try:
        tensors: dict[str, np.ndarray] = {}
        for key in shard.specs():
            arr = shard.raw(str(key))
            if copy_to_host:
                arr = _copy_prepared_tensor(arr)
            else:
                arr = np.ascontiguousarray(arr)
            tensors[str(key)] = arr
        return tensors
    finally:
        shard.close()


def _upload_prepared_tensor(
    tensors: dict[str, np.ndarray],
    key: str,
    *,
    upload_fn: Callable[..., Any],
) -> Any:
    arr = tensors.get(key)
    if arr is None:
        return None
    return upload_fn(arr, name=key)


def _prepared_meta_int(meta: dict[str, Any], key: str, cache_dir: Path) -> int:
    if key not in meta:
        raise RuntimeError(
            f"DSV4 prepared-weight cache metadata missing {key!r}: path={cache_dir}"
        )
    try:
        return int(meta[key])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "DSV4 prepared-weight cache metadata has invalid integer field: "
            f"field={key}, value={meta[key]!r}, path={cache_dir}"
        ) from exc


def _validate_prepared_weight_cache_metadata(
    meta: dict[str, Any],
    v4_meta: DeepseekV4Weights,
    cache_dir: Path,
) -> None:
    """Ensure prepared cache shards match this runtime rank before loading."""
    expected_ints = {
        "tp_degree": int(v4_meta.tp_degree),
        "tp_rank": int(v4_meta.tp_rank),
        "ep_degree": int(v4_meta.ep_degree),
    }
    for key, expected in expected_ints.items():
        cached = _prepared_meta_int(meta, key, cache_dir)
        if cached != expected:
            raise RuntimeError(
                "DSV4 prepared-weight cache topology mismatch: "
                f"field={key}, cache={cached}, expected={expected}, path={cache_dir}"
            )

    ep_degree = int(v4_meta.ep_degree)
    cached_lane = _prepared_meta_int(meta, "attention_lane", cache_dir)
    expected_lane_mod = int(v4_meta.attention_lane) % ep_degree
    cached_lane_mod = cached_lane % ep_degree
    if cached_lane_mod != expected_lane_mod:
        raise RuntimeError(
            "DSV4 prepared-weight cache topology mismatch: "
            "field=attention_lane, "
            f"cache={cached_lane}, expected={int(v4_meta.attention_lane)} "
            f"(mod {ep_degree}), path={cache_dir}"
        )

    if "local_expert_ids" not in meta:
        raise RuntimeError(
            "DSV4 prepared-weight cache metadata missing 'local_expert_ids': "
            f"path={cache_dir}"
        )
    cached_experts = tuple(int(x) for x in meta.get("local_expert_ids", ()))
    expected_experts = tuple(int(x) for x in v4_meta.local_expert_ids)
    if cached_experts != expected_experts:
        raise RuntimeError(
            "DSV4 prepared-weight cache expert mapping mismatch: "
            "field=local_expert_ids, "
            f"cache={cached_experts}, expected={expected_experts}, path={cache_dir}"
        )


def _build_prepared_layer_device_weights(
    layer_id: int,
    tensors: dict[str, np.ndarray],
    v4_meta: DeepseekV4Weights,
    *,
    upload_fn: Callable[..., Any],
) -> V4LayerDeviceWeights:
    lw = V4LayerDeviceWeights(layer_id=layer_id)
    lw.shared_tp_sharded = int(v4_meta.tp_degree) > 1
    for key in _LAYER_PREPARED_KEYS:
        if key not in tensors:
            continue
        setattr(
            lw,
            key,
            upload_fn(tensors[key], name=f"L{layer_id}_{key}"),
        )
    return lw


def _load_v4_prepared_device_weights(
    cache_dir: Path,
    v4_meta: DeepseekV4Weights,
    *,
    plan: V4LoadPlan,
    upload_fn: Callable[..., Any],
) -> V4DeviceWeights:
    meta_path = cache_dir / "metadata.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    version = int(meta.get("version", -1))
    if version != _DSV4_PREPARED_WEIGHT_CACHE_VERSION:
        raise RuntimeError(
            f"Unsupported DSV4 prepared-weight cache version {version}; "
            f"expected {_DSV4_PREPARED_WEIGHT_CACHE_VERSION}"
        )
    n_layers = int(v4_meta.num_hidden_layers)
    cached_layers = int(meta.get("num_hidden_layers", -1))
    if cached_layers < n_layers:
        raise RuntimeError(
            "DSV4 prepared-weight cache has too few layers: "
            f"cache={cached_layers}, requested={n_layers}, path={cache_dir}"
        )
    _validate_prepared_weight_cache_metadata(meta, v4_meta, cache_dir)

    out = V4DeviceWeights(fp8_noscale_snapshot=True)
    load_start = time.monotonic()
    copy_to_host = upload_fn is _upload and _weight_load_prefetch_enabled()
    load_profiler = StartupProfiler(
        "dsv4_weight_load",
        rank=_weight_load_global_rank(v4_meta),
        source="prepared",
        path=str(cache_dir),
        layers=n_layers,
        tp_rank=int(v4_meta.tp_rank),
        ep_rank=int(v4_meta.ep_rank),
        replica_rank=int(v4_meta.replica_rank),
    )
    load_profiler.record("start", copy_to_host=copy_to_host)
    prepared_read_elapsed = 0.0
    prepared_upload_elapsed = 0.0
    _weight_load_log(
        v4_meta,
        f"prepared-cache start path={cache_dir} layers={n_layers} "
        f"copy_to_host={copy_to_host}",
    )

    if plan.dense or plan.hc:
        dense_read_t0 = time.monotonic()
        dense = _load_prepared_safetensors(
            cache_dir / "dense.safetensors",
            copy_to_host=copy_to_host,
        )
        prepared_read_elapsed += time.monotonic() - dense_read_t0
        dense_upload_t0 = time.monotonic()
        if plan.dense:
            out.embed_vocab_offset = int(v4_meta.lm_head_vocab_offset)
            out.embed_vocab_end = out.embed_vocab_offset + int(v4_meta.local_vocab_size)
            out.embed_tp_sharded = int(v4_meta.tp_degree) > 1
            out.embed = _upload_prepared_tensor(dense, "embed", upload_fn=upload_fn)
            out.final_norm = _upload_prepared_tensor(
                dense, "final_norm", upload_fn=upload_fn
            )
            out.lm_head = _upload_prepared_tensor(dense, "lm_head", upload_fn=upload_fn)
        if plan.hc:
            out.hc_head_fn = _upload_prepared_tensor(
                dense, "hc_head_fn", upload_fn=upload_fn
            )
            out.hc_head_base = _upload_prepared_tensor(
                dense, "hc_head_base", upload_fn=upload_fn
            )
            out.hc_head_scale = _upload_prepared_tensor(
                dense, "hc_head_scale", upload_fn=upload_fn
            )
        prepared_upload_elapsed += time.monotonic() - dense_upload_t0
        load_profiler.record(
            "dense uploaded",
            read_elapsed_s=round(float(prepared_read_elapsed), 6),
            upload_elapsed_s=round(float(prepared_upload_elapsed), 6),
        )
        del dense

    def _read_prepared_layer(
        layer_id: int,
    ) -> tuple[int, dict[str, np.ndarray], float]:
        layer_file = cache_dir / f"layer_{layer_id:03d}.safetensors"
        if not layer_file.exists():
            raise RuntimeError(f"Missing prepared DSV4 layer file: {layer_file}")
        layer_read_t0 = time.monotonic()
        tensors = _load_prepared_safetensors(
            layer_file,
            copy_to_host=copy_to_host,
        )
        return layer_id, tensors, time.monotonic() - layer_read_t0

    layer_progress = _WeightLayerProgress(n_layers, v4_meta)
    try:
        if n_layers > 1 and copy_to_host:
            _weight_load_log(
                v4_meta,
                "prepared-cache layer host-prefetch enabled",
            )
            with ThreadPoolExecutor(max_workers=1) as pool:
                next_future: Future[tuple[int, dict[str, np.ndarray], float]] = (
                    pool.submit(_read_prepared_layer, 0)
                )
                for layer_id in range(n_layers):
                    loaded_layer_id, tensors, read_elapsed = next_future.result()
                    prepared_read_elapsed += read_elapsed
                    if loaded_layer_id != layer_id:
                        raise RuntimeError(
                            "Prepared DSV4 layer prefetch order mismatch: "
                            f"expected={layer_id}, got={loaded_layer_id}"
                        )
                    if layer_id + 1 < n_layers:
                        next_future = pool.submit(
                            _read_prepared_layer,
                            layer_id + 1,
                        )
                    layer_upload_t0 = time.monotonic()
                    out.layers.append(
                        _build_prepared_layer_device_weights(
                            layer_id,
                            tensors,
                            v4_meta,
                            upload_fn=upload_fn,
                        )
                    )
                    prepared_upload_elapsed += time.monotonic() - layer_upload_t0
                    del tensors
                    layer_progress.update(layer_id)
                    load_profiler.record(
                        "layer uploaded",
                        layer_id=int(layer_id),
                        layer_read_elapsed_s=round(float(read_elapsed), 6),
                        read_elapsed_s=round(float(prepared_read_elapsed), 6),
                        upload_elapsed_s=round(float(prepared_upload_elapsed), 6),
                    )
        else:
            for layer_id in range(n_layers):
                loaded_layer_id, tensors, read_elapsed = _read_prepared_layer(layer_id)
                prepared_read_elapsed += read_elapsed
                layer_upload_t0 = time.monotonic()
                out.layers.append(
                    _build_prepared_layer_device_weights(
                        loaded_layer_id,
                        tensors,
                        v4_meta,
                        upload_fn=upload_fn,
                    )
                )
                prepared_upload_elapsed += time.monotonic() - layer_upload_t0
                del tensors
                layer_progress.update(layer_id)
                load_profiler.record(
                    "layer uploaded",
                    layer_id=int(loaded_layer_id),
                    layer_read_elapsed_s=round(float(read_elapsed), 6),
                    read_elapsed_s=round(float(prepared_read_elapsed), 6),
                    upload_elapsed_s=round(float(prepared_upload_elapsed), 6),
                )
    finally:
        layer_progress.close()

    _weight_load_log(
        v4_meta,
        "prepared-cache done "
        f"layers={len(out.layers)} "
        f"read_elapsed={prepared_read_elapsed:.1f}s "
        f"upload_elapsed={prepared_upload_elapsed:.1f}s "
        f"elapsed={time.monotonic() - load_start:.1f}s",
    )
    load_profiler.record(
        "done",
        layers=len(out.layers),
        read_elapsed_s=round(float(prepared_read_elapsed), 6),
        upload_elapsed_s=round(float(prepared_upload_elapsed), 6),
    )
    return out


# ---------------------------------------------------------------------------
# MXFP4 → no-scale Neuron FP8 E4M3 conversion
# ---------------------------------------------------------------------------


def _cast_mxfp4_to_neuron_fp8_noscale(
    w_i8: np.ndarray,
    scale_e8m0: np.ndarray,
    *,
    fp4_block: int = 32,
) -> np.ndarray:
    """MXFP4 (I8 packed) + per-32 E8M0 → Neuron-range FP8 E4M3 ``[out, in]``.

    Product serving stores only FP8 weights and carries no runtime scale
    tensor. This is a normal FP8 quantization step: values must be finite and
    inside Neuron's E4M3 range, but they need not be bit-exact after the cast.
    """
    if w_i8.ndim != 2 or (w_i8.dtype != np.int8 and w_i8.dtype != np.uint8):
        raise RuntimeError(
            f"MXFP4 weight must be I8/U8 2D, got {w_i8.shape} {w_i8.dtype}"
        )

    w_f32 = dequant_mxfp4_block(w_i8, scale_e8m0, fp4_block=fp4_block)
    max_abs = float(np.max(np.abs(w_f32))) if w_f32.size else 0.0
    if not np.isfinite(w_f32).all() or max_abs > 240.0:
        raise RuntimeError(
            "MXFP4 routed expert value is outside Neuron E4M3 no-scale "
            f"range: max_abs={max_abs}"
        )

    # Store using the safetensors-compatible FN tag, then reinterpret the same
    # bytes as Neuron E4M3 on upload.
    fp8 = w_f32.astype(ml_dtypes.float8_e4m3fn)
    return fp8


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _stage_v4_dense_hc_weights(
    reader: V4WeightReader,
    out: V4DeviceWeights,
    v4_meta: DeepseekV4Weights,
    *,
    plan: V4LoadPlan,
    upload_fn: Callable[..., Any],
) -> None:
    """Stage or upload non-layer dense and mHC-head tensors into ``out``."""
    if plan.dense:
        emb = reader.read_bf16("embed.weight")
        ev0 = int(v4_meta.lm_head_vocab_offset)
        ev1 = ev0 + int(v4_meta.local_vocab_size)
        out.embed_vocab_offset = ev0
        out.embed_vocab_end = ev1
        out.embed_tp_sharded = int(v4_meta.tp_degree) > 1
        if out.embed_tp_sharded:
            emb = emb[ev0:ev1, :].copy()
        else:
            out.embed_vocab_offset = 0
            out.embed_vocab_end = int(emb.shape[0])
        out.embed = upload_fn(emb, name="embed")
        out.final_norm = upload_fn(reader.read_bf16("norm.weight"), name="final_norm")
        v0 = int(v4_meta.lm_head_vocab_offset)
        v1 = v0 + int(v4_meta.local_vocab_size)
        lm = reader.read_bf16("head.weight")
        if (v0, v1) != (0, lm.shape[0]):
            lm = lm[v0:v1, :].copy()
        out.lm_head = upload_fn(lm, name="lm_head")

    if plan.hc:
        out.hc_head_fn = upload_fn(reader.read_fp32("hc_head_fn"), name="hc_head_fn")
        out.hc_head_base = upload_fn(
            reader.read_fp32("hc_head_base"), name="hc_head_base"
        )
        out.hc_head_scale = upload_fn(
            reader.read_fp32("hc_head_scale"), name="hc_head_scale"
        )


def _stage_v4_layer_weights(
    reader: V4WeightReader,
    v4_meta: DeepseekV4Weights,
    *,
    plan: V4LoadPlan,
    layer_id: int,
    fp8_noscale: bool,
    upload_fn: Callable[..., Any],
) -> V4LayerDeviceWeights:
    """Stage or upload one DSV4 layer according to ``plan``."""
    lw = V4LayerDeviceWeights(layer_id=layer_id)
    prefix = f"layers.{layer_id}"

    if plan.hc:
        for part in ("attn", "ffn"):
            for sub in ("fn", "base", "scale"):
                key = f"{prefix}.hc_{part}_{sub}"
                setattr(
                    lw,
                    f"hc_{part}_{sub}",
                    upload_fn(
                        reader.read_fp32(key),
                        name=f"L{layer_id}_hc_{part}_{sub}",
                    ),
                )
        lw.attn_norm = upload_fn(
            reader.read_bf16(f"{prefix}.attn_norm.weight"),
            name=f"L{layer_id}_attn_norm",
        )
        lw.ffn_norm = upload_fn(
            reader.read_bf16(f"{prefix}.ffn_norm.weight"),
            name=f"L{layer_id}_ffn_norm",
        )

    if plan.moe:
        _load_layer_moe(
            reader,
            lw,
            prefix,
            v4_meta,
            layer_id,
            fp8_noscale=fp8_noscale,
            blockwise_moe_fp8=plan.blockwise_moe_fp8,
            upload_fn=upload_fn,
        )
    if plan.attention:
        _load_layer_attention(
            reader,
            lw,
            prefix,
            v4_meta,
            layer_id,
            upload_fn=upload_fn,
        )
    if plan.compressor and int(v4_meta.compress_ratios[layer_id]) > 0:
        _load_layer_compressor(
            reader,
            lw,
            prefix,
            layer_id,
            upload_fn=upload_fn,
        )
    if plan.indexer and int(v4_meta.compress_ratios[layer_id]) == 4:
        _load_layer_indexer(
            reader,
            lw,
            prefix,
            v4_meta,
            layer_id,
            upload_fn=upload_fn,
        )
    return lw


def load_v4_device_weights(
    model_config: DeepseekV4ModelConfig,
    v4_meta: DeepseekV4Weights,
    *,
    plan: V4LoadPlan | None = None,
    upload_fn: Callable[..., Any] | None = None,
) -> V4DeviceWeights:
    """Upload the `plan`-selected subset of V4 weights to device.

    Product serving requires ``scripts/convert_dsv4_checkpoint.py`` output:
    routed experts are Neuron-range FP8 with no runtime scales, and shared
    experts are BF16.
    """
    plan = plan or V4LoadPlan.sampled()
    upload_fn = upload_fn or _upload
    if upload_fn is _upload:
        cache_profiler = StartupProfiler(
            "dsv4_weight_load",
            rank=_weight_load_global_rank(v4_meta),
            source="resolve",
            tp_rank=int(v4_meta.tp_rank),
            ep_rank=int(v4_meta.ep_rank),
            replica_rank=int(v4_meta.replica_rank),
        )
        cache_profiler.record("resolve prepared-cache start")
        prepared_dir = _prepared_weight_cache_dir(v4_meta)
        cache_profiler.record(
            "resolve prepared-cache done",
            prepared_dir=str(prepared_dir) if prepared_dir is not None else "",
        )
        if prepared_dir is not None:
            return _load_v4_prepared_device_weights(
                prepared_dir,
                v4_meta,
                plan=plan,
                upload_fn=upload_fn,
            )

    snapshot = resolve_model_snapshot_path(
        model_config.hf_model_id,
        revision=model_config.hf_revision,
        local_files_only=model_config.hf_local_files_only,
    )
    fp8_noscale = _snapshot_is_fp8_noscale(snapshot)
    reader = V4WeightReader(snapshot)
    out = V4DeviceWeights()
    out.fp8_noscale_snapshot = fp8_noscale
    n_layers = int(v4_meta.num_hidden_layers)
    load_start = time.monotonic()
    load_profiler = StartupProfiler(
        "dsv4_weight_load",
        rank=_weight_load_global_rank(v4_meta),
        source="snapshot",
        snapshot=str(snapshot),
        layers=n_layers,
        tp_rank=int(v4_meta.tp_rank),
        ep_rank=int(v4_meta.ep_rank),
        replica_rank=int(v4_meta.replica_rank),
    )
    load_profiler.record("start", fp8_noscale=bool(fp8_noscale))
    _weight_load_log(
        v4_meta,
        "start "
        f"snapshot={snapshot} plan={_load_plan_summary(plan)} "
        f"layers={n_layers} fp8_noscale={bool(fp8_noscale)}",
    )

    _stage_v4_dense_hc_weights(
        reader,
        out,
        v4_meta,
        plan=plan,
        upload_fn=upload_fn,
    )
    if plan.dense:
        _weight_load_log(
            v4_meta,
            f"dense uploaded elapsed={time.monotonic() - load_start:.1f}s",
        )
        load_profiler.record("dense uploaded")
    if plan.hc:
        _weight_load_log(
            v4_meta,
            f"hc head uploaded elapsed={time.monotonic() - load_start:.1f}s",
        )
        load_profiler.record("hc head uploaded")

    def _stage_layer(
        layer_id: int, upload_fn: Callable[..., Any]
    ) -> V4LayerDeviceWeights:
        return _stage_v4_layer_weights(
            reader,
            v4_meta,
            plan=plan,
            layer_id=layer_id,
            fp8_noscale=fp8_noscale,
            upload_fn=upload_fn,
        )

    layer_progress = _WeightLayerProgress(n_layers, v4_meta)
    try:
        if n_layers > 1 and upload_fn is _upload and _weight_load_prefetch_enabled():
            _weight_load_log(v4_meta, "layer host-prefetch enabled")
            with ThreadPoolExecutor(max_workers=1) as pool:
                next_future: Future[V4LayerDeviceWeights] = pool.submit(
                    _stage_layer,
                    0,
                    _stage_upload,
                )
                for layer_id in range(n_layers):
                    staged = next_future.result()
                    if layer_id + 1 < n_layers:
                        next_future = pool.submit(
                            _stage_layer,
                            layer_id + 1,
                            _stage_upload,
                        )
                    lw = _materialize_staged_uploads(staged)
                    out.layers.append(lw)
                    layer_progress.update(layer_id)
                    load_profiler.record("layer uploaded", layer_id=int(layer_id))
        else:
            for layer_id in range(n_layers):
                lw = _stage_layer(layer_id, upload_fn)
                out.layers.append(lw)
                layer_progress.update(layer_id)
                load_profiler.record("layer uploaded", layer_id=int(layer_id))
    finally:
        layer_progress.close()

    reader.close()
    _weight_load_log(
        v4_meta,
        f"done layers={len(out.layers)} elapsed={time.monotonic() - load_start:.1f}s",
    )
    load_profiler.record(
        "done",
        layers=len(out.layers),
    )
    return out


def _read_linear_bf16(reader: V4WeightReader, key: str) -> np.ndarray:
    """Read a linear weight into BF16 serving layout."""
    tag = reader.spec(key).dtype_tag
    if tag in ("BF16", "F32"):
        return reader.read_bf16(key)
    if tag == "F8_E4M3":
        return reader.read_fp8_block_dequant(key).astype(ml_dtypes.bfloat16)
    raise RuntimeError(f"Unsupported linear weight dtype for {key}: {tag}")


def _read_shared_moe_weight_bf16(reader: V4WeightReader, key: str) -> np.ndarray:
    """Read one preprocessed shared-expert weight as BF16."""
    tag = reader.spec(key).dtype_tag
    if tag in ("BF16", "F32"):
        return reader.read_bf16(key)
    raise RuntimeError(
        f"Shared MoE weight {key} must be BF16 in the preprocessed product "
        f"snapshot, got {tag}"
    )


def _read_routed_moe_weight_fp8_noscale(
    reader: V4WeightReader,
    key: str,
) -> np.ndarray:
    """Read one preprocessed routed expert weight as no-scale Neuron FP8."""
    tag = reader.spec(key).dtype_tag
    if tag == "F8_E4M3":
        return reader.raw(key)
    raise RuntimeError(
        f"Routed MoE weight {key} must be preprocessed no-scale FP8 "
        f"(F8_E4M3), got {tag}"
    )


def _tp_slice_expert_intermediate(
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    v4_meta: DeepseekV4Weights,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the TP-local intermediate slice for one expert.

    Checkpoint expert tensors are out-in:
      - w1/w3: ``[I_global_or_local, H]``
      - w2:    ``[H, I_global_or_local]``

    The blockwise kernel consumes the local intermediate shard. If the reader
    already produced local shapes, keep them; otherwise slice by ``tp_rank``.
    """
    local_i = int(v4_meta.local_moe_intermediate_size)
    if local_i <= 0:
        local_i = int(w1.shape[0])
    if int(w1.shape[0]) == local_i and int(w2.shape[1]) == local_i:
        return w1, w2, w3

    i0 = int(v4_meta.tp_rank) * local_i
    i1 = i0 + local_i
    if w1.shape[0] < i1 or w3.shape[0] < i1 or w2.shape[1] < i1:
        raise RuntimeError(
            "Cannot TP-slice DSV4 expert weights for blockwise MoE: "
            f"w1={w1.shape}, w2={w2.shape}, w3={w3.shape}, "
            f"slice=[{i0}, {i1})"
        )
    return w1[i0:i1], w2[:, i0:i1], w3[i0:i1]


def _install_blockwise_fp8_layer(
    lw: V4LayerDeviceWeights,
    *,
    w1s: list[np.ndarray],
    w2s: list[np.ndarray],
    w3s: list[np.ndarray],
    v4_meta: DeepseekV4Weights,
    layer_id: int,
    upload_fn: Callable[..., Any] = _upload,
) -> None:
    """Stack TP-local no-scale FP8 expert arrays for the blockwise kernel."""
    hidden = int(v4_meta.hidden_size)
    intermediate = int(v4_meta.local_moe_intermediate_size)
    if intermediate <= 0 and w1s:
        intermediate = int(w1s[0].shape[0])
    E = len(w1s)
    if E == 0:
        raise RuntimeError(f"Layer {layer_id} has no local routed experts")
    fp8_dtype = w1s[0].dtype
    gate_up = np.empty((E, hidden, 2, intermediate), dtype=fp8_dtype)
    down = np.empty((E, intermediate, hidden), dtype=w2s[0].dtype)
    for e_id, (w1, w2, w3) in enumerate(zip(w1s, w2s, w3s, strict=True)):
        w1_l, w2_l, w3_l = _tp_slice_expert_intermediate(w1, w2, w3, v4_meta)
        gate_up[e_id, :, 0, :] = np.ascontiguousarray(
            w1_l.T,
        )
        gate_up[e_id, :, 1, :] = np.ascontiguousarray(
            w3_l.T,
        )
        down[e_id] = np.ascontiguousarray(
            w2_l.T,
        )

    gu_bias = np.zeros((E, intermediate, 2), dtype=ml_dtypes.bfloat16)
    lw.blockwise_gate_up_w = upload_fn(
        gate_up,
        name=f"L{layer_id}_blockwise_gate_up_w",
    )
    lw.blockwise_down_w = upload_fn(
        down,
        name=f"L{layer_id}_blockwise_down_w",
    )
    lw.blockwise_gate_up_bias = upload_fn(
        gu_bias,
        name=f"L{layer_id}_blockwise_gate_up_bias",
    )
    # V4 routed experts are bias-free. Do not upload a per-layer zero
    # broadcast tensor; the blockwise kernel has a no-down-bias specialization.
    lw.blockwise_down_bias_bc = None


def _attention_tp_partition(
    v4_meta: DeepseekV4Weights,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Return (local head ids, local output-group ids, heads_per_group)."""
    total_heads = int(v4_meta.num_attention_heads)
    local_heads = int(v4_meta.local_num_attention_heads)
    tp_degree = int(v4_meta.tp_degree)
    tp_rank = int(v4_meta.tp_rank)
    o_groups = int(v4_meta.o_groups)
    if total_heads % o_groups != 0:
        raise RuntimeError(
            "DSV4 attention output groups must evenly split heads: "
            f"num_heads={total_heads}, o_groups={o_groups}"
        )
    heads_per_group = total_heads // o_groups
    if tp_degree <= 1:
        return tuple(range(total_heads)), tuple(range(o_groups)), heads_per_group
    if o_groups % tp_degree != 0:
        raise RuntimeError(
            "Pure-TP DSV4 attention requires whole output groups per TP rank: "
            f"o_groups={o_groups}, tp_degree={tp_degree}"
        )
    local_groups = o_groups // tp_degree
    expected_local_heads = local_groups * heads_per_group
    if local_heads != expected_local_heads:
        raise RuntimeError(
            "DSV4 attention TP metadata mismatch: "
            f"local_heads={local_heads}, expected={expected_local_heads}, "
            f"num_heads={total_heads}, o_groups={o_groups}, tp={tp_degree}"
        )
    head_start = tp_rank * local_heads
    group_start = tp_rank * local_groups
    return (
        tuple(range(head_start, head_start + local_heads)),
        tuple(range(group_start, group_start + local_groups)),
        heads_per_group,
    )


def _slice_attention_wo_a(
    wo_a: np.ndarray,
    group_indices: tuple[int, ...],
    *,
    o_groups: int,
    o_lora_rank: int,
) -> np.ndarray:
    rank = int(o_lora_rank)
    grouped = np.asarray(wo_a).reshape(int(o_groups), rank, wo_a.shape[1])
    out = grouped[list(group_indices), :, :].reshape(
        len(group_indices) * rank, wo_a.shape[1]
    )
    return np.ascontiguousarray(out)


def _slice_attention_wo_b(
    wo_b: np.ndarray,
    group_indices: tuple[int, ...],
    *,
    o_groups: int,
    o_lora_rank: int,
) -> np.ndarray:
    rank = int(o_lora_rank)
    grouped = np.asarray(wo_b).reshape(wo_b.shape[0], int(o_groups), rank)
    out = grouped[:, list(group_indices), :].reshape(
        wo_b.shape[0], len(group_indices) * rank
    )
    return np.ascontiguousarray(out)


def _slice_indexer_rows(
    matrix: np.ndarray,
    v4_meta: DeepseekV4Weights,
    *,
    head_dim: int,
) -> np.ndarray:
    n_heads = int(v4_meta.index_n_heads)
    tp_degree = int(v4_meta.tp_degree)
    if tp_degree <= 1:
        return matrix
    if n_heads % tp_degree != 0:
        raise RuntimeError(
            "DSV4 indexer heads must be divisible by TP degree: "
            f"index_n_heads={n_heads}, tp_degree={tp_degree}"
        )
    local_heads = n_heads // tp_degree
    h0 = int(v4_meta.tp_rank) * local_heads
    head_indices = tuple(range(h0, h0 + local_heads))
    if matrix.ndim == 2 and int(matrix.shape[0]) == n_heads:
        return np.ascontiguousarray(matrix[list(head_indices), :])
    return _select_head_rows(matrix, head_indices, int(head_dim))


def _load_layer_attention(
    reader: V4WeightReader,
    lw: V4LayerDeviceWeights,
    prefix: str,
    v4_meta: DeepseekV4Weights,
    layer_id: int,
    *,
    upload_fn: Callable[..., Any] = _upload,
) -> None:
    attn_prefix = f"{prefix}.attn"
    head_indices, group_indices, _heads_per_group = _attention_tp_partition(v4_meta)
    o_groups = int(v4_meta.o_groups)
    o_lora_rank = int(v4_meta.o_lora_rank)
    lw.attn_sink = upload_fn(
        np.ascontiguousarray(
            reader.read_fp32(f"{attn_prefix}.attn_sink")[list(head_indices)],
        ),
        name=f"L{layer_id}_attn_sink",
    )
    lw.wq_a = upload_fn(
        _read_linear_bf16(reader, f"{attn_prefix}.wq_a.weight"),
        name=f"L{layer_id}_wq_a",
    )
    lw.q_norm = upload_fn(
        reader.read_bf16(f"{attn_prefix}.q_norm.weight"),
        name=f"L{layer_id}_q_norm",
    )
    lw.wq_b = upload_fn(
        _select_head_rows(
            _read_linear_bf16(reader, f"{attn_prefix}.wq_b.weight"),
            head_indices,
            int(v4_meta.head_dim),
        ),
        name=f"L{layer_id}_wq_b",
    )
    lw.wkv = upload_fn(
        _read_linear_bf16(reader, f"{attn_prefix}.wkv.weight"),
        name=f"L{layer_id}_wkv",
    )
    lw.kv_norm = upload_fn(
        reader.read_bf16(f"{attn_prefix}.kv_norm.weight"),
        name=f"L{layer_id}_kv_norm",
    )
    lw.wo_a = upload_fn(
        _slice_attention_wo_a(
            _read_linear_bf16(reader, f"{attn_prefix}.wo_a.weight"),
            group_indices,
            o_groups=o_groups,
            o_lora_rank=o_lora_rank,
        ),
        name=f"L{layer_id}_wo_a",
    )
    lw.wo_b = upload_fn(
        _slice_attention_wo_b(
            _read_linear_bf16(reader, f"{attn_prefix}.wo_b.weight"),
            group_indices,
            o_groups=o_groups,
            o_lora_rank=o_lora_rank,
        ),
        name=f"L{layer_id}_wo_b",
    )


def _load_layer_compressor(
    reader: V4WeightReader,
    lw: V4LayerDeviceWeights,
    prefix: str,
    layer_id: int,
    *,
    upload_fn: Callable[..., Any] = _upload,
) -> None:
    comp_prefix = f"{prefix}.attn.compressor"
    lw.comp_wkv = upload_fn(
        _read_linear_bf16(reader, f"{comp_prefix}.wkv.weight"),
        name=f"L{layer_id}_comp_wkv",
    )
    lw.comp_wgate = upload_fn(
        _read_linear_bf16(reader, f"{comp_prefix}.wgate.weight"),
        name=f"L{layer_id}_comp_wgate",
    )
    lw.comp_ape = upload_fn(
        reader.read_fp32(f"{comp_prefix}.ape"),
        name=f"L{layer_id}_comp_ape",
    )
    lw.comp_norm = upload_fn(
        reader.read_bf16(f"{comp_prefix}.norm.weight"),
        name=f"L{layer_id}_comp_norm",
    )


def _load_layer_indexer(
    reader: V4WeightReader,
    lw: V4LayerDeviceWeights,
    prefix: str,
    v4_meta: DeepseekV4Weights,
    layer_id: int,
    *,
    upload_fn: Callable[..., Any] = _upload,
) -> None:
    idx_prefix = f"{prefix}.attn.indexer"
    lw.idx_wq_b = upload_fn(
        _slice_indexer_rows(
            _read_linear_bf16(reader, f"{idx_prefix}.wq_b.weight"),
            v4_meta,
            head_dim=int(v4_meta.index_head_dim),
        ),
        name=f"L{layer_id}_idx_wq_b",
    )
    lw.idx_weights_proj = upload_fn(
        _slice_indexer_rows(
            _read_linear_bf16(reader, f"{idx_prefix}.weights_proj.weight"),
            v4_meta,
            head_dim=1,
        ),
        name=f"L{layer_id}_idx_weights_proj",
    )
    idx_comp_prefix = f"{idx_prefix}.compressor"
    lw.idx_comp_wkv = upload_fn(
        _read_linear_bf16(reader, f"{idx_comp_prefix}.wkv.weight"),
        name=f"L{layer_id}_idx_comp_wkv",
    )
    lw.idx_comp_wgate = upload_fn(
        _read_linear_bf16(reader, f"{idx_comp_prefix}.wgate.weight"),
        name=f"L{layer_id}_idx_comp_wgate",
    )
    lw.idx_comp_ape = upload_fn(
        reader.read_fp32(f"{idx_comp_prefix}.ape"),
        name=f"L{layer_id}_idx_comp_ape",
    )
    lw.idx_comp_norm = upload_fn(
        reader.read_bf16(f"{idx_comp_prefix}.norm.weight"),
        name=f"L{layer_id}_idx_comp_norm",
    )


def _load_layer_moe(
    reader: V4WeightReader,
    lw: V4LayerDeviceWeights,
    prefix: str,
    v4_meta: DeepseekV4Weights,
    layer_id: int,
    *,
    fp8_noscale: bool = False,
    blockwise_moe_fp8: bool = True,
    upload_fn: Callable[..., Any] = _upload,
) -> None:
    is_hash = layer_id < int(v4_meta.num_hash_layers)
    if not blockwise_moe_fp8:
        raise RuntimeError(
            "DSV4 product serving requires blockwise_moe_fp8=True; "
            "per-expert MoE load paths are not supported."
        )
    if not fp8_noscale:
        raise RuntimeError(
            "DSV4 product MoE requires a preprocessed no-scale FP8 snapshot "
            f"(metadata dsv4_conversion={_DSV4_CONVERSION_FP8_NOSCALE!r})."
        )

    # Gate: hash layers have BOTH gate.weight (for score computation, H0e)
    # and tid2eid (for expert-index lookup); the `bias` field is absent on
    # hash layers. Learned layers have gate.weight and (optionally) bias.
    lw.gate_weight = upload_fn(
        reader.read_bf16(f"{prefix}.ffn.gate.weight"),
        name=f"L{layer_id}_gate_w",
    )
    if is_hash:
        tid2eid_key = f"{prefix}.ffn.gate.tid2eid"
        if reader.has(tid2eid_key):
            lw.gate_tid2eid = upload_fn(
                np.asarray(reader.raw(tid2eid_key), dtype=np.int32),
                name=f"L{layer_id}_gate_tid2eid",
            )
    else:
        bias_key = f"{prefix}.ffn.gate.bias"
        if reader.has(bias_key):
            lw.gate_bias = upload_fn(
                reader.read_fp32(bias_key),
                name=f"L{layer_id}_gate_b",
            )

    shared_w1 = _read_shared_moe_weight_bf16(
        reader,
        f"{prefix}.ffn.shared_experts.w1.weight",
    )
    shared_w2 = _read_shared_moe_weight_bf16(
        reader,
        f"{prefix}.ffn.shared_experts.w2.weight",
    )
    shared_w3 = _read_shared_moe_weight_bf16(
        reader,
        f"{prefix}.ffn.shared_experts.w3.weight",
    )
    if int(v4_meta.tp_degree) > 1:
        shared_w1, shared_w2, shared_w3 = _tp_slice_expert_intermediate(
            shared_w1,
            shared_w2,
            shared_w3,
            v4_meta,
        )
        lw.shared_tp_sharded = True
    lw.shared_w1 = upload_fn(shared_w1, name=f"L{layer_id}_shared_w1")
    lw.shared_w2 = upload_fn(shared_w2, name=f"L{layer_id}_shared_w2")
    lw.shared_w3 = upload_fn(shared_w3, name=f"L{layer_id}_shared_w3")

    # Routed experts: upload only stacked no-scale FP8 tensors.
    blockwise_w1s: list[np.ndarray] = []
    blockwise_w2s: list[np.ndarray] = []
    blockwise_w3s: list[np.ndarray] = []
    for expert_id in v4_meta.local_expert_ids:
        blockwise_w1s.append(
            _read_routed_moe_weight_fp8_noscale(
                reader,
                f"{prefix}.ffn.experts.{expert_id}.w1.weight",
            )
        )
        blockwise_w2s.append(
            _read_routed_moe_weight_fp8_noscale(
                reader,
                f"{prefix}.ffn.experts.{expert_id}.w2.weight",
            )
        )
        blockwise_w3s.append(
            _read_routed_moe_weight_fp8_noscale(
                reader,
                f"{prefix}.ffn.experts.{expert_id}.w3.weight",
            )
        )

    _install_blockwise_fp8_layer(
        lw,
        w1s=blockwise_w1s,
        w2s=blockwise_w2s,
        w3s=blockwise_w3s,
        v4_meta=v4_meta,
        layer_id=layer_id,
        upload_fn=upload_fn,
    )
