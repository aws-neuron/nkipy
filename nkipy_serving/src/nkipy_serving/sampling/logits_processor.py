"""LogitsProcessor: unified LM-head → sampling → logprobs pipeline.

Owns all sampling-related kernel compilation, warmup, and forward dispatch.
Each model executor creates one at init and delegates its final stage to it.

Four forward paths:

  greedy, no logprobs  → rank-local top-k candidates for CPU TP merge
  greedy, logprobs     → all-gather logits, argmax, log_softmax + top-k
  sample, no logprobs  → all-gather logits, NKI CDF sampler
  sample, logprobs     → all-gather logits, NKI CDF sampler, log_softmax + top-k
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from nkipy_serving.runtime.collective_load import collective_load_barrier
from nkipy_serving.runtime.device_tensor import get_device_tensor_cls
from nkipy_serving.runtime.kernel_compile import (
    compile_and_load_neff_with_lock,
    compile_neff_path_with_lock,
    shared_kernel_build_dir,
)
from nkipy_serving.sampling.constants import LOGPROBS_K_MAX
from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
from nkipy_serving.sampling.lm_head_sampling import (
    lm_head_local_topk,
    lm_head_sample_tokens,
    lm_head_sample_tokens_with_logprobs,
)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogitsProcessorOutput:
    """Result of the logits processing pipeline."""

    # Always present (one of the two):
    next_token_ids: np.ndarray | None = None  # [bs] int32 — global vocab ids
    # OR rank-local top-k candidates (greedy, no logprobs):
    top1_values: np.ndarray | None = None  # [bs] float32
    top1_indices: np.ndarray | None = None  # [bs] int32 (local vocab)
    topk_values: np.ndarray | None = None  # [bs, k] float32
    topk_indices: np.ndarray | None = None  # [bs, k] int32 (local vocab)

    # Present only when logprobs requested:
    chosen_logprobs: np.ndarray | None = None  # [bs] float32
    topk_logprob_vals: np.ndarray | None = None  # [bs, k] float32
    topk_logprob_ids: np.ndarray | None = None  # [bs, k] int32 (global vocab)

    def to_shm_dict(self, *, vocab_offset: int = 0) -> dict[str, np.ndarray]:
        """Convert to dict for SHM output slot writing."""
        if self.next_token_ids is not None:
            out: dict[str, np.ndarray] = {
                "next_token_ids": self.next_token_ids,
            }
            if self.chosen_logprobs is not None:
                out["chosen_logprobs"] = self.chosen_logprobs
                out["topk_logprob_vals"] = self.topk_logprob_vals
                out["topk_logprob_ids"] = self.topk_logprob_ids
            return out

        # Greedy path: rank-local candidates for CPU merge.
        if self.top1_values is not None:
            return {
                "top1_values": self.top1_values,
                "top1_indices": self.top1_indices,
                "vocab_offset": np.asarray([int(vocab_offset)], dtype=np.int32),
            }
        return {
            "topk_values": self.topk_values,
            "topk_indices": self.topk_indices,
            "vocab_offset": np.asarray([int(vocab_offset)], dtype=np.int32),
        }


# ---------------------------------------------------------------------------
# NKIPy runtime
# ---------------------------------------------------------------------------

_DeviceKernel = None
_DeviceTensor = None
_NKIPY_AVAILABLE = False


def _ensure_nkipy_runtime():
    global _DeviceKernel, _DeviceTensor, _NKIPY_AVAILABLE
    if _NKIPY_AVAILABLE:
        return
    try:
        from nkipy.runtime import DeviceKernel

        _DeviceKernel = DeviceKernel
        _DeviceTensor = get_device_tensor_cls()
        _NKIPY_AVAILABLE = True
    except ImportError as exc:
        raise RuntimeError("nkipy runtime is required for LogitsProcessor") from exc


def _alloc_device_scratch(shape: tuple[int, ...], dtype, *, name: str):
    return _DeviceTensor.from_numpy(np.empty(shape, dtype=dtype), name=name)


@dataclass
class _PrecompiledLogitsKernel:
    """Lazy loader for logits kernels compiled during warmup."""

    neff_path: str
    name: str
    cc_enabled: bool
    shared_build_dir: str | None = None
    rank_id: int | None = None
    world_size: int | None = None
    loaded: Any | None = None

    def load(self) -> Any:
        if self.loaded is not None:
            return self.loaded
        if bool(self.cc_enabled):
            if self.rank_id is None or self.world_size is None:
                raise RuntimeError(
                    f"LogitsProcessor collective kernel {self.name} is missing "
                    "rank/world metadata"
                )
            collective_load_barrier(
                build_dir=self.shared_build_dir,
                name=self.name,
                rank_id=int(self.rank_id),
                world_size=int(self.world_size),
            )
            self.loaded = _DeviceKernel.load_from_neff(
                self.neff_path,
                name=self.name,
                cc_enabled=True,
                rank_id=int(self.rank_id),
                world_size=int(self.world_size),
            )
        else:
            self.loaded = _DeviceKernel.load_from_neff(
                self.neff_path,
                name=self.name,
            )
        return self.loaded

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.load()(*args, **kwargs)


def _compile_logits_kernel(
    kernel,
    *args,
    cc_enabled: bool,
    name: str,
    additional_compiler_args: str,
    build_dir: str,
    rank_id: int,
    world_size: int,
    defer_load: bool = False,
    **kwargs,
):
    """Compile logits kernels to NEFF, then load the resolved artifact."""

    if not (
        hasattr(_DeviceKernel, "_trace_and_compile")
        and hasattr(_DeviceKernel, "load_from_neff")
    ):
        raise RuntimeError(
            "LogitsProcessor kernels require DeviceKernel "
            "_trace_and_compile/load_from_neff so every warmup kernel has "
            "a precompiled NEFF."
        )
    if not cc_enabled:
        if defer_load:
            neff_path = compile_neff_path_with_lock(
                _DeviceKernel,
                kernel,
                *args,
                name=name,
                additional_compiler_args=additional_compiler_args,
                build_dir=build_dir,
                namespace="logits_processor",
                **kwargs,
            )
            return _PrecompiledLogitsKernel(
                neff_path=str(neff_path),
                name=str(name),
                cc_enabled=False,
            )
        return compile_and_load_neff_with_lock(
            _DeviceKernel,
            kernel,
            *args,
            name=name,
            additional_compiler_args=additional_compiler_args,
            build_dir=build_dir,
            namespace="logits_processor",
            **kwargs,
        )
    use_cached_if_exists = bool(kwargs.pop("use_cached_if_exists", True))
    shared_build_dir = shared_kernel_build_dir(
        build_dir,
        namespace="logits_processor",
    )
    neff_path = compile_neff_path_with_lock(
        _DeviceKernel,
        kernel,
        *args,
        name=name,
        build_dir=build_dir,
        namespace="logits_processor",
        additional_compiler_args=additional_compiler_args,
        use_cached_if_exists=use_cached_if_exists,
        **kwargs,
    )
    if defer_load:
        return _PrecompiledLogitsKernel(
            neff_path=str(neff_path),
            name=str(name),
            cc_enabled=True,
            shared_build_dir=shared_build_dir,
            rank_id=int(rank_id),
            world_size=int(world_size),
        )
    collective_load_barrier(
        build_dir=shared_build_dir,
        name=name,
        rank_id=int(rank_id),
        world_size=int(world_size),
    )
    return _DeviceKernel.load_from_neff(
        neff_path,
        name=name,
        cc_enabled=True,
        rank_id=int(rank_id),
        world_size=int(world_size),
    )


# ---------------------------------------------------------------------------
# Traceable wrapper functions for kernel compilation
# ---------------------------------------------------------------------------


def _lm_head_top1_fn(
    hidden: np.ndarray,
    final_norm: np.ndarray,
    lm_head: np.ndarray,
    last_token_indices: np.ndarray,
    *,
    rms_norm_eps: float = 1e-6,
    gather_hidden: bool = False,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> tuple[np.ndarray, np.ndarray]:
    top1_vals, top1_idx = lm_head_local_topk(
        hidden,
        final_norm,
        lm_head,
        last_token_indices,
        rms_norm_eps=rms_norm_eps,
        topk=1,
        gather_hidden=gather_hidden,
        tp_degree=tp_degree,
        tp_replica_groups=tp_replica_groups,
    )
    return top1_vals.reshape((-1,)), top1_idx.reshape((-1,)).astype(np.int32)


def _lm_head_topk_fn(
    hidden: np.ndarray,
    final_norm: np.ndarray,
    lm_head: np.ndarray,
    last_token_indices: np.ndarray,
    *,
    rms_norm_eps: float = 1e-6,
    topk: int = 1,
    gather_hidden: bool = False,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> tuple[np.ndarray, np.ndarray]:
    return lm_head_local_topk(
        hidden,
        final_norm,
        lm_head,
        last_token_indices,
        rms_norm_eps=rms_norm_eps,
        topk=topk,
        gather_hidden=gather_hidden,
        tp_degree=tp_degree,
        tp_replica_groups=tp_replica_groups,
    )


# ---------------------------------------------------------------------------
# Compiled kernel storage (per bucket)
# ---------------------------------------------------------------------------


@dataclass
class _CompiledSamplingKernels:
    """All sampling kernels for one bucket size."""

    greedy_kernel: object  # top-1 or top-k
    device_sample_kernel: object | None = None  # filtered NKI sampler
    device_sample_kernel_unfiltered: object | None = None  # unfiltered NKI sampler
    logprobs_sample_kernel: object | None = None  # sampler + logprobs (filtered)
    logprobs_sample_kernel_unfiltered: object | None = (
        None  # sampler + logprobs (unfiltered)
    )

    # Scratch buffers (allocated once per bucket).
    top1_values_scratch: object | None = None
    top1_indices_scratch: object | None = None
    topk_values_scratch: object | None = None
    topk_indices_scratch: object | None = None


# ---------------------------------------------------------------------------
# LogitsProcessor
# ---------------------------------------------------------------------------


class LogitsProcessor:
    """Unified LM-head → sampling → logprobs pipeline.

    Owns kernel compilation, warmup, scratch buffers, and forward dispatch.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        local_vocab_size: int,
        vocab_offset: int,
        hidden_size: int,
        dtype: np.dtype,
        tp_degree: int = 1,
        tp_rank: int = 0,
        tp_replica_groups: tuple = (),
        collective_rank: int | None = None,
        collective_world_size: int | None = None,
        rms_norm_eps: float = 1e-6,
        dense_local_topk: int = 1,
        gather_hidden: bool = False,
        nkipy_compiler_args: str = "",
        build_dir: str = "/tmp/build",
        max_requests_per_step: int = 32,
    ):
        _ensure_nkipy_runtime()

        self._vocab_size = int(vocab_size)
        self._local_vocab_size = int(local_vocab_size)
        self._vocab_offset = int(vocab_offset)
        self._hidden_size = int(hidden_size)
        self._dtype = np.dtype(dtype)
        self._tp_degree = int(tp_degree)
        self._tp_rank = int(tp_rank)
        self._tp_replica_groups = tp_replica_groups
        self._collective_rank = (
            int(tp_rank) if collective_rank is None else int(collective_rank)
        )
        self._collective_world_size = (
            int(tp_degree)
            if collective_world_size is None
            else int(collective_world_size)
        )
        self._rms_norm_eps = float(rms_norm_eps)
        self._dense_local_topk = int(dense_local_topk)
        self._gather_hidden = bool(gather_hidden)
        self._compiler_args = str(nkipy_compiler_args)
        self._build_dir = str(build_dir)
        self._max_bs = int(max_requests_per_step)
        # Compile logprobs kernels with a fixed max-k.  Requests with smaller k
        # get the full top-max and the scheduler slices to the requested k.
        # This avoids compiling a separate NEFF per requested logprobs_k.
        self._logprobs_k_max = LOGPROBS_K_MAX

        # Compiled kernels keyed by bucket size.
        self._kernels: dict[int, _CompiledSamplingKernels] = {}
        self._precompiled_kernel_buckets_sealed = False

    # -- Properties ------------------------------------------------------------

    @property
    def vocab_offset(self) -> int:
        return self._vocab_offset

    def _use_top1_fast_path(self) -> bool:
        return self._dense_local_topk == 1

    @staticmethod
    def _needs_full_sampler(sampling_batch: DeviceSamplingBatch | None) -> bool:
        return sampling_batch is not None and sampling_batch.enabled

    def _collective_name_suffix(self) -> str:
        if (
            int(self._collective_world_size) == int(self._tp_degree)
            and not self._tp_replica_groups
        ):
            return ""
        group_repr = repr(
            tuple(
                tuple(int(rank) for rank in group)
                for group in tuple(self._tp_replica_groups or ())
            )
        )
        digest = hashlib.sha1(
            f"{int(self._collective_world_size)}:{group_repr}".encode("utf-8")
        ).hexdigest()[:8]
        return f"_cw{int(self._collective_world_size)}_g{digest}"

    # -- Kernel compilation ----------------------------------------------------

    def compile_kernels(
        self,
        token_buckets: Sequence[int],
        bs_buckets: Sequence[int],
    ) -> None:
        """Compile all sampling kernels for the given buckets."""
        all_buckets = sorted(
            set(int(b) for b in token_buckets) | set(int(b) for b in bs_buckets)
        )
        for bucket in all_buckets:
            self._ensure_kernels(bucket)

    def seal_precompiled_kernels(self) -> None:
        """Reject new logits DeviceKernel shapes after warmup."""
        self._precompiled_kernel_buckets_sealed = True

    def _ensure_kernels(
        self,
        bucket: int,
        *,
        include_sampler: bool = True,
        include_logprobs: bool = True,
        deferred_sampler_load: bool = False,
    ) -> _CompiledSamplingKernels:
        bucket = int(bucket)
        cached = self._kernels.get(bucket)
        if (
            cached is not None
            and (not include_sampler or cached.device_sample_kernel is not None)
            and (not include_logprobs or cached.logprobs_sample_kernel is not None)
        ):
            return cached

        if self._precompiled_kernel_buckets_sealed:
            missing = []
            if cached is None:
                missing.append("base")
            if include_sampler and (
                cached is None or cached.device_sample_kernel is None
            ):
                missing.append("sampler")
            if include_logprobs and (
                cached is None or cached.logprobs_sample_kernel is None
            ):
                missing.append("logprobs")
            known = ", ".join(str(b) for b in sorted(self._kernels)) or "<none>"
            raise RuntimeError(
                "LogitsProcessor kernels were sealed after warmup, but request "
                f"needs uncompiled bucket={bucket} missing={tuple(missing)}. "
                f"precompiled_buckets=[{known}]"
            )

        # Shape-specialization tensors.
        sample_hidden = np.zeros((bucket, self._hidden_size), dtype=self._dtype)
        sample_norm = np.zeros((self._hidden_size,), dtype=self._dtype)
        sample_lm_head = np.zeros(
            (self._local_vocab_size, self._hidden_size),
            dtype=self._dtype,
        )
        sample_last_idx = np.zeros((self._max_bs,), dtype=np.int32)
        sample_temps = np.ones((self._max_bs,), dtype=np.float32)
        sample_top_ks = np.ones((self._max_bs,), dtype=np.int32)
        sample_top_ps = np.ones((self._max_bs,), dtype=np.float32)
        sample_min_ps = np.zeros((self._max_bs,), dtype=np.float32)
        sample_uniform_u = np.zeros((self._max_bs,), dtype=np.float32)

        cc_enabled = self._tp_degree > 1
        greedy_gather = self._gather_hidden
        collective_suffix = self._collective_name_suffix()

        if cached is None:
            # -- Greedy top-k kernel --
            if self._use_top1_fast_path():
                greedy_kernel = _compile_logits_kernel(
                    _lm_head_top1_fn,
                    sample_hidden,
                    sample_norm,
                    sample_lm_head,
                    sample_last_idx,
                    rms_norm_eps=self._rms_norm_eps,
                    gather_hidden=greedy_gather,
                    tp_degree=self._tp_degree,
                    tp_replica_groups=self._tp_replica_groups,
                    name=(
                        f"lp_top1_tp{self._tp_degree}_t{bucket}"
                        f"_bsmax{self._max_bs}"
                        f"{collective_suffix if greedy_gather else ''}"
                    ),
                    additional_compiler_args=self._compiler_args,
                    build_dir=self._build_dir,
                    cc_enabled=cc_enabled and greedy_gather,
                    rank_id=self._collective_rank,
                    world_size=self._collective_world_size,
                    defer_load=bool(deferred_sampler_load),
                )
            else:
                greedy_kernel = _compile_logits_kernel(
                    _lm_head_topk_fn,
                    sample_hidden,
                    sample_norm,
                    sample_lm_head,
                    sample_last_idx,
                    rms_norm_eps=self._rms_norm_eps,
                    topk=self._dense_local_topk,
                    gather_hidden=greedy_gather,
                    tp_degree=self._tp_degree,
                    tp_replica_groups=self._tp_replica_groups,
                    name=(
                        f"lp_topk_tp{self._tp_degree}_t{bucket}"
                        f"_k{self._dense_local_topk}_bsmax{self._max_bs}"
                        f"{collective_suffix if greedy_gather else ''}"
                    ),
                    additional_compiler_args=self._compiler_args,
                    build_dir=self._build_dir,
                    cc_enabled=cc_enabled and greedy_gather,
                    rank_id=self._collective_rank,
                    world_size=self._collective_world_size,
                    defer_load=bool(deferred_sampler_load),
                )

            # -- Scratch buffers --
            if self._use_top1_fast_path():
                top1_values_scratch = _alloc_device_scratch(
                    (self._max_bs,),
                    np.float32,
                    name=f"lp_top1_vals_t{bucket}",
                )
                top1_indices_scratch = _alloc_device_scratch(
                    (self._max_bs,),
                    np.int32,
                    name=f"lp_top1_idx_t{bucket}",
                )
                topk_values_scratch = None
                topk_indices_scratch = None
            else:
                top1_values_scratch = None
                top1_indices_scratch = None
                topk_values_scratch = _alloc_device_scratch(
                    (self._max_bs, self._dense_local_topk),
                    np.float32,
                    name=f"lp_topk_vals_t{bucket}",
                )
                topk_indices_scratch = _alloc_device_scratch(
                    (self._max_bs, self._dense_local_topk),
                    np.int32,
                    name=f"lp_topk_idx_t{bucket}",
                )

            cached = _CompiledSamplingKernels(
                greedy_kernel=greedy_kernel,
                top1_values_scratch=top1_values_scratch,
                top1_indices_scratch=top1_indices_scratch,
                topk_values_scratch=topk_values_scratch,
                topk_indices_scratch=topk_indices_scratch,
            )
            self._kernels[bucket] = cached

        if include_sampler and cached.device_sample_kernel is None:
            # -- Device sampler kernel (filtered) --
            cached.device_sample_kernel = _compile_logits_kernel(
                lm_head_sample_tokens,
                sample_hidden,
                sample_norm,
                sample_lm_head,
                sample_last_idx,
                sample_temps,
                sample_top_ks,
                sample_top_ps,
                sample_min_ps,
                sample_uniform_u,
                rms_norm_eps=self._rms_norm_eps,
                gather_hidden=self._gather_hidden,
                tp_degree=self._tp_degree,
                tp_replica_groups=self._tp_replica_groups,
                name=(
                    f"lp_sample_tp{self._tp_degree}_t{bucket}"
                    f"_bsmax{self._max_bs}"
                    f"{collective_suffix}"
                ),
                additional_compiler_args=self._compiler_args,
                use_cached_if_exists=True,
                build_dir=self._build_dir,
                cc_enabled=cc_enabled,
                rank_id=self._collective_rank,
                world_size=self._collective_world_size,
                defer_load=bool(deferred_sampler_load),
            )

            # -- Device sampler kernel (unfiltered) --
            cached.device_sample_kernel_unfiltered = _compile_logits_kernel(
                lm_head_sample_tokens,
                sample_hidden,
                sample_norm,
                sample_lm_head,
                sample_last_idx,
                sample_temps,
                sample_top_ks,
                sample_top_ps,
                sample_min_ps,
                sample_uniform_u,
                unfiltered=True,
                rms_norm_eps=self._rms_norm_eps,
                gather_hidden=self._gather_hidden,
                tp_degree=self._tp_degree,
                tp_replica_groups=self._tp_replica_groups,
                name=(
                    f"lp_sample_unf_tp{self._tp_degree}_t{bucket}"
                    f"_bsmax{self._max_bs}"
                    f"{collective_suffix}"
                ),
                additional_compiler_args=self._compiler_args,
                use_cached_if_exists=True,
                build_dir=self._build_dir,
                cc_enabled=cc_enabled,
                rank_id=self._collective_rank,
                world_size=self._collective_world_size,
                defer_load=bool(deferred_sampler_load),
            )

        if include_logprobs and cached.logprobs_sample_kernel is None:
            # -- Logprobs sampler kernels --
            logprobs_k = self._logprobs_k_max
            cached.logprobs_sample_kernel = _compile_logits_kernel(
                lm_head_sample_tokens_with_logprobs,
                sample_hidden,
                sample_norm,
                sample_lm_head,
                sample_last_idx,
                sample_temps,
                sample_top_ks,
                sample_top_ps,
                sample_min_ps,
                sample_uniform_u,
                rms_norm_eps=self._rms_norm_eps,
                logprobs_k=logprobs_k,
                gather_hidden=self._gather_hidden,
                tp_degree=self._tp_degree,
                tp_replica_groups=self._tp_replica_groups,
                name=(
                    f"lp_sample_logprobs_tp{self._tp_degree}_t{bucket}"
                    f"_bsmax{self._max_bs}_k{logprobs_k}"
                    f"{collective_suffix}"
                ),
                additional_compiler_args=self._compiler_args,
                use_cached_if_exists=True,
                build_dir=self._build_dir,
                cc_enabled=cc_enabled,
                rank_id=self._collective_rank,
                world_size=self._collective_world_size,
                defer_load=bool(deferred_sampler_load),
            )
            cached.logprobs_sample_kernel_unfiltered = _compile_logits_kernel(
                lm_head_sample_tokens_with_logprobs,
                sample_hidden,
                sample_norm,
                sample_lm_head,
                sample_last_idx,
                sample_temps,
                sample_top_ks,
                sample_top_ps,
                sample_min_ps,
                sample_uniform_u,
                unfiltered=True,
                logprobs_k=logprobs_k,
                rms_norm_eps=self._rms_norm_eps,
                gather_hidden=self._gather_hidden,
                tp_degree=self._tp_degree,
                tp_replica_groups=self._tp_replica_groups,
                name=(
                    f"lp_sample_logprobs_unf_tp{self._tp_degree}_t{bucket}"
                    f"_bsmax{self._max_bs}_k{logprobs_k}"
                    f"{collective_suffix}"
                ),
                additional_compiler_args=self._compiler_args,
                use_cached_if_exists=True,
                build_dir=self._build_dir,
                cc_enabled=cc_enabled,
                rank_id=self._collective_rank,
                world_size=self._collective_world_size,
                defer_load=bool(deferred_sampler_load),
            )
        return cached

    # -- Forward ---------------------------------------------------------------

    def forward(
        self,
        hidden_dev: object,
        final_norm_dev: object,
        lm_head_dev: object,
        last_token_indices_dev: object,
        batch_size: int,
        *,
        token_bucket: int,
        sampling_batch: DeviceSamplingBatch | None = None,
        needs_logprobs: bool = False,
        logprobs_k: int = 0,
    ) -> LogitsProcessorOutput:
        """Run the LM-head → sampling → logprobs pipeline.

        Args:
            hidden_dev: Device tensor [hidden_bucket, H].
            final_norm_dev: Device tensor [H] RMS norm weights.
            lm_head_dev: Device tensor [local_vocab, H] LM head weights.
            last_token_indices_dev: Device tensor [bs_bucket] int32.
            batch_size: Real (unpadded) batch size.
            token_bucket: First dimension of hidden_dev. For gather_hidden=True
                this is the TP shard size (token_bucket / tp_degree), not the
                full token count.
            sampling_batch: Device sampling parameters (None = greedy).
            needs_logprobs: Whether to compute logprobs.
            logprobs_k: Top-k for logprobs extraction.

        Returns:
            LogitsProcessorOutput with token IDs and optional logprobs.
        """
        use_full_sampler = self._needs_full_sampler(sampling_batch)

        # Force device sampler when logprobs are requested (need all-gather).
        if needs_logprobs and not use_full_sampler:
            use_full_sampler = True

        bs = int(batch_size)
        kernels = self._ensure_kernels(
            int(token_bucket),
            include_sampler=bool(use_full_sampler),
            include_logprobs=bool(needs_logprobs and logprobs_k > 0),
        )

        if use_full_sampler:
            return self._forward_full_sampler(
                kernels,
                hidden_dev,
                final_norm_dev,
                lm_head_dev,
                last_token_indices_dev,
                bs,
                token_bucket=int(token_bucket),
                sampling_batch=sampling_batch,
                needs_logprobs=needs_logprobs,
                logprobs_k=logprobs_k,
            )
        return self._forward_greedy(
            kernels,
            hidden_dev,
            final_norm_dev,
            lm_head_dev,
            last_token_indices_dev,
            bs,
        )

    # -- Greedy path -----------------------------------------------------------

    def _forward_greedy(
        self,
        kernels: _CompiledSamplingKernels,
        hidden_dev: object,
        final_norm_dev: object,
        lm_head_dev: object,
        last_token_indices_dev: object,
        bs: int,
    ) -> LogitsProcessorOutput:
        inputs = {
            "hidden": hidden_dev,
            "final_norm": final_norm_dev,
            "lm_head": lm_head_dev,
            "last_token_indices": last_token_indices_dev,
        }
        if self._use_top1_fast_path():
            kernels.greedy_kernel(
                inputs=inputs,
                outputs={
                    "output0": kernels.top1_values_scratch,
                    "output1": kernels.top1_indices_scratch,
                },
            )
            return LogitsProcessorOutput(
                top1_values=kernels.top1_values_scratch.numpy()[:bs].astype(
                    np.float32,
                    copy=False,
                ),
                top1_indices=kernels.top1_indices_scratch.numpy()[:bs].astype(
                    np.int32,
                    copy=False,
                ),
            )
        kernels.greedy_kernel(
            inputs=inputs,
            outputs={
                "output0": kernels.topk_values_scratch,
                "output1": kernels.topk_indices_scratch,
            },
        )
        return LogitsProcessorOutput(
            topk_values=kernels.topk_values_scratch.numpy()[:bs].astype(
                np.float32,
                copy=False,
            ),
            topk_indices=kernels.topk_indices_scratch.numpy()[:bs].astype(
                np.int32,
                copy=False,
            ),
        )

    # -- Device sampler path ---------------------------------------------------

    def _forward_full_sampler(
        self,
        kernels: _CompiledSamplingKernels,
        hidden_dev: object,
        final_norm_dev: object,
        lm_head_dev: object,
        last_token_indices_dev: object,
        bs: int,
        *,
        token_bucket: int,
        sampling_batch: DeviceSamplingBatch | None,
        needs_logprobs: bool,
        logprobs_k: int,
    ) -> LogitsProcessorOutput:
        # Build padded sampling parameter tensors.
        if sampling_batch is not None:
            sampling_inputs = sampling_batch.padded_inputs(self._max_bs)
        else:
            # Greedy forced into device sampler path (for logprobs).
            sampling_inputs = DeviceSamplingBatch(
                use_full_sampler=True,
                temperatures=np.ones((bs,), dtype=np.float32),
                top_ks=np.ones((bs,), dtype=np.int32),
                top_ps=np.ones((bs,), dtype=np.float32),
                min_ps=np.zeros((bs,), dtype=np.float32),
                uniform_u=np.zeros((bs,), dtype=np.float32),
            ).padded_inputs(self._max_bs)

        temperatures_dev = _DeviceTensor.from_numpy(
            sampling_inputs["temperatures"],
            name="lp_temperatures",
        )
        top_ks_dev = _DeviceTensor.from_numpy(
            sampling_inputs["top_ks"],
            name="lp_top_ks",
        )
        top_ps_dev = _DeviceTensor.from_numpy(
            sampling_inputs["top_ps"],
            name="lp_top_ps",
        )
        min_ps_dev = _DeviceTensor.from_numpy(
            sampling_inputs["min_ps"],
            name="lp_min_ps",
        )
        uniform_u_dev = _DeviceTensor.from_numpy(
            sampling_inputs["uniform_u"],
            name="lp_uniform_u",
        )
        # Select filtered vs unfiltered kernel.
        use_unfiltered = (
            kernels.device_sample_kernel_unfiltered is not None
            and sampling_batch is not None
            and not sampling_batch.needs_filtering
        )

        if needs_logprobs and logprobs_k > 0:
            # Use the logprobs-enabled kernel that returns sampled ids + logprobs.
            next_token_ids_dev = _DeviceTensor.from_numpy(
                np.zeros((self._max_bs,), dtype=np.int32),
                name="lp_next_token_ids",
            )
            chosen_logprobs_dev = _DeviceTensor.from_numpy(
                np.zeros((self._max_bs,), dtype=np.float32),
                name="lp_chosen_logprobs",
            )
            k_max = self._logprobs_k_max
            topk_logprob_vals_dev = _DeviceTensor.from_numpy(
                np.zeros((self._max_bs, k_max), dtype=np.float32),
                name="lp_topk_logprob_vals",
            )
            topk_logprob_ids_dev = _DeviceTensor.from_numpy(
                np.zeros((self._max_bs, k_max), dtype=np.int32),
                name="lp_topk_logprob_ids",
            )

            lp_kernel = (
                kernels.logprobs_sample_kernel_unfiltered
                if use_unfiltered
                and kernels.logprobs_sample_kernel_unfiltered is not None
                else kernels.logprobs_sample_kernel
            )
            if lp_kernel is None:
                raise RuntimeError("Logprobs kernel not compiled for this bucket")

            lp_kernel(
                inputs={
                    "hidden": hidden_dev,
                    "final_norm": final_norm_dev,
                    "lm_head": lm_head_dev,
                    "last_token_indices": last_token_indices_dev,
                    "temperatures": temperatures_dev,
                    "top_ks": top_ks_dev,
                    "top_ps": top_ps_dev,
                    "min_ps": min_ps_dev,
                    "uniform_u": uniform_u_dev,
                },
                outputs={
                    "output0": next_token_ids_dev,
                    "output1": chosen_logprobs_dev,
                    "output2": topk_logprob_vals_dev,
                    "output3": topk_logprob_ids_dev,
                },
            )

            return LogitsProcessorOutput(
                next_token_ids=next_token_ids_dev.numpy()[:bs].astype(
                    np.int32, copy=False
                ),
                chosen_logprobs=chosen_logprobs_dev.numpy()[:bs].astype(
                    np.float32, copy=False
                ),
                topk_logprob_vals=topk_logprob_vals_dev.numpy()[:bs].astype(
                    np.float32, copy=False
                ),
                topk_logprob_ids=topk_logprob_ids_dev.numpy()[:bs].astype(
                    np.int32, copy=False
                ),
            )

        # No logprobs — use standard sampler kernel.
        kernel = (
            kernels.device_sample_kernel_unfiltered
            if use_unfiltered
            else kernels.device_sample_kernel
        )
        next_token_ids_dev = _DeviceTensor.from_numpy(
            np.zeros((self._max_bs,), dtype=np.int32),
            name="lp_next_token_ids",
        )
        kernel(
            inputs={
                "hidden": hidden_dev,
                "final_norm": final_norm_dev,
                "lm_head": lm_head_dev,
                "last_token_indices": last_token_indices_dev,
                "temperatures": temperatures_dev,
                "top_ks": top_ks_dev,
                "top_ps": top_ps_dev,
                "min_ps": min_ps_dev,
                "uniform_u": uniform_u_dev,
            },
            outputs={"output0": next_token_ids_dev},
        )

        next_token_ids = next_token_ids_dev.numpy()[:bs].astype(np.int32, copy=False)
        return LogitsProcessorOutput(next_token_ids=next_token_ids)
