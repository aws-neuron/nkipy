"""Numpy-side safetensors reader + FP8/MXFP4 dequant for DeepSeek-V4.

Mirrors the HF `inference/kernel.py` dequant semantics exactly:

- **FP8 E4M3 weight** `[out, in]` with UE8M0 `[ceil(out/128), ceil(in/128)]`
  block scales. Dequant: `w_fp32[i,j] = e4m3(w[i,j]) * e8m0(scale[i//128, j//128])`.

- **MXFP4 expert weight** packed as I8 `[out, in // 2]` (two FP4 values per
  byte, packed along the input axis). Scale is UE8M0 `[out, in // 32]`.
  Dequant: `w_fp32[i,j] = fp4_table(w[i, j//2], j%2 == 1) * e8m0(scale[i, j//32])`.

Used by checkpoint conversion and CPU-side reference utilities. Serving uses
preprocessed device weights; this path stays CPU-only.

`ml_dtypes` handles `float8_e4m3fn` and `float8_e8m0fnu` casts; numpy gives
us packed byte views via `np.frombuffer`. safetensors' own numpy loader
chokes on `F8_E8M0`, so we parse the header and read the bytes ourselves.
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

# Mapping from safetensors dtype tag → (numpy/ml_dtypes dtype, element bytes).
# F8_E4M3 is the non-FN variant on some safetensors producers, but the HF V4
# checkpoint uses the same 8-bit payload as `float8_e4m3fn` (saturating, no
# inf, -0 handling differs only at bit patterns we don't hit for weights).
_DTYPE_MAP: dict[str, Any] = {
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": ml_dtypes.bfloat16,
    "F8_E4M3": ml_dtypes.float8_e4m3fn,
    "F8_E8M0": ml_dtypes.float8_e8m0fnu,
    "I8": np.int8,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
}


# FP4 E2M1 lookup: bit pattern (low nibble first) → float value.
# Matches `FP4_TABLE` in HF `inference/convert.py`.
_FP4_TABLE = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class TensorSpec:
    """Describes one tensor inside a safetensors shard."""

    dtype_tag: str
    shape: tuple[int, ...]
    byte_offset: int  # offset within the data region (NOT into file).
    byte_end: int


class _ShardReader:
    """mmap-based reader for a single `.safetensors` file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with open(path, "rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(hdr_len)
        self._header_len = hdr_len
        self._data_offset = 8 + hdr_len
        header = json.loads(header_bytes.decode("utf-8"))
        self._specs: dict[str, TensorSpec] = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            offsets = entry["data_offsets"]
            self._specs[name] = TensorSpec(
                dtype_tag=entry["dtype"],
                shape=tuple(int(x) for x in entry["shape"]),
                byte_offset=int(offsets[0]),
                byte_end=int(offsets[1]),
            )
        self._mm: mmap.mmap | None = None

    def _mmap(self) -> mmap.mmap:
        if self._mm is None:
            with open(self.path, "rb") as fd:
                self._mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm

    def close(self) -> None:
        # Best effort: if numpy arrays still reference the mmap (common when
        # upload is followed by close), keep the handle so a later close can
        # succeed after those exported pointers are released.
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:
                return
            else:
                self._mm = None

    def specs(self) -> dict[str, TensorSpec]:
        return dict(self._specs)

    def raw(self, name: str) -> np.ndarray:
        spec = self._specs[name]
        dtype = _DTYPE_MAP.get(spec.dtype_tag)
        if dtype is None:
            raise RuntimeError(
                f"Unsupported safetensors dtype tag {spec.dtype_tag!r} for {name!r}"
            )
        mm = self._mmap()
        start = self._data_offset + spec.byte_offset
        end = self._data_offset + spec.byte_end
        buf = np.frombuffer(mm, dtype=dtype, count=-1, offset=start)[
            : (end - start) // np.dtype(dtype).itemsize
        ]
        return buf.reshape(spec.shape)


class V4WeightReader:
    """Top-level reader indexed by `model.safetensors.index.json`.

    Opens shard files lazily on first access. Holds mmap views — caller must
    `close()` when done (or let the process exit; mmaps are released).
    """

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path
        index_path = snapshot_path / "model.safetensors.index.json"
        if not index_path.exists():
            raise RuntimeError(f"Missing safetensors index under {snapshot_path}")
        with index_path.open("r", encoding="utf-8") as f:
            idx = json.load(f)
        wm = idx.get("weight_map")
        if not isinstance(wm, dict):
            raise RuntimeError(f"Invalid weight_map in {index_path}")
        self._weight_map: dict[str, str] = {str(k): str(v) for k, v in wm.items()}
        self._shards: dict[str, _ShardReader] = {}

    def close(self) -> None:
        for s in self._shards.values():
            s.close()
        self._shards.clear()

    def keys(self) -> list[str]:
        return list(self._weight_map.keys())

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def spec(self, name: str) -> TensorSpec:
        return self._shard_for(name).specs()[name]

    # -- core raw access -------------------------------------------------

    def _shard_for(self, name: str) -> _ShardReader:
        fn = self._weight_map.get(name)
        if fn is None:
            raise KeyError(name)
        reader = self._shards.get(fn)
        if reader is None:
            reader = _ShardReader(self.snapshot_path / fn)
            self._shards[fn] = reader
        return reader

    def raw(self, name: str) -> np.ndarray:
        """Return the tensor in its on-disk dtype (possibly FP8 / E8M0 / I8)."""
        return self._shard_for(name).raw(name)

    # -- typed helpers ---------------------------------------------------

    def read_bf16(self, name: str) -> np.ndarray:
        """Read a tensor whose on-disk dtype is BF16 or FP32. Returns ml_dtypes.bfloat16."""
        arr = self.raw(name)
        tag = self.spec(name).dtype_tag
        if tag == "BF16":
            return arr
        if tag == "F32":
            return arr.astype(ml_dtypes.bfloat16)
        raise RuntimeError(f"Expected BF16/F32 for {name}, got {tag}")

    def read_fp32(self, name: str) -> np.ndarray:
        arr = self.raw(name)
        return arr.astype(np.float32)

    def read_fp8_block_dequant(
        self, name: str, block_m: int = 128, block_n: int = 128
    ) -> np.ndarray:
        """Dequant a FP8 E4M3 weight with UE8M0 block scale. Returns float32.

        Scale sibling key is `{name[:-len('.weight')]}.scale`. Weight shape
        must be `[out, in]`. Scale shape must be `[ceil(out/block_m), ceil(in/block_n)]`.
        """
        if not name.endswith(".weight"):
            raise RuntimeError(f"FP8 weight key must end in .weight: {name!r}")
        w = self.raw(name)
        scale_name = name[: -len(".weight")] + ".scale"
        s = self.raw(scale_name)
        return dequant_fp8_block(w, s, block_m=block_m, block_n=block_n)

    def read_mxfp4_block_dequant(self, name: str, fp4_block: int = 32) -> np.ndarray:
        """Dequant an MXFP4 expert weight. Returns float32 `[out, in]`.

        Weight is stored as I8 `[out, in // 2]` (low nibble = col 2j, high
        nibble = col 2j+1). Scale is UE8M0 `[out, in // fp4_block]`.
        """
        if not name.endswith(".weight"):
            raise RuntimeError(f"MXFP4 weight key must end in .weight: {name!r}")
        w = self.raw(name)
        scale_name = name[: -len(".weight")] + ".scale"
        s = self.raw(scale_name)
        return dequant_mxfp4_block(w, s, fp4_block=fp4_block)


# -- dequant kernels (numpy) ----------------------------------------------


def _block_scale_to_fp32(scale: np.ndarray) -> np.ndarray:
    """Normalize a block scale to fp32.

    Accepts ``ml_dtypes.float8_e8m0fnu`` (stock HF layout) or ``float32`` /
    ``float64`` (post-H1 rescaled layout emitted by the pre-convert CLI).
    """
    if scale.dtype == ml_dtypes.float8_e8m0fnu:
        return scale.astype(np.float32)
    if scale.dtype in (np.float32, np.float64):
        return scale.astype(np.float32)
    raise RuntimeError(
        f"Unsupported scale dtype {scale.dtype!r}; expected float8_e8m0fnu "
        f"or float32 (H1 rescaled scale)."
    )


def dequant_fp8_block(
    w_fp8: np.ndarray,
    scale: np.ndarray,
    *,
    block_m: int = 128,
    block_n: int = 128,
) -> np.ndarray:
    """FP8 E4M3 × block scale → fp32 ``[M, N]``.

    Scale ``[ceil(M/block_m), ceil(N/block_n)]`` is broadcast to ``[M, N]``
    by repeating each scale value ``block_m`` rows × ``block_n`` cols. The
    scale may be either UE8M0 (stock HF layout) or fp32 (post-H1 rescaled
    layout emitted by ``scripts/convert_dsv4_checkpoint.py``).
    """
    if w_fp8.ndim != 2:
        raise RuntimeError(f"FP8 weight must be 2D, got shape {w_fp8.shape}")
    if w_fp8.dtype != ml_dtypes.float8_e4m3fn:
        raise RuntimeError(f"Expected FP8 E4M3 weight, got {w_fp8.dtype}")
    out_dim, in_dim = w_fp8.shape
    exp_s0 = (out_dim + block_m - 1) // block_m
    exp_s1 = (in_dim + block_n - 1) // block_n
    if scale.shape != (exp_s0, exp_s1):
        raise RuntimeError(
            f"Scale shape {scale.shape} does not match expected "
            f"({exp_s0}, {exp_s1}) for weight {w_fp8.shape} with "
            f"blocks {(block_m, block_n)}."
        )
    w = w_fp8.astype(np.float32)
    s = _block_scale_to_fp32(scale)
    # Expand scale to full weight shape via kron-style tiling.
    s_full = np.repeat(np.repeat(s, block_m, axis=0), block_n, axis=1)
    s_full = s_full[:out_dim, :in_dim]
    return w * s_full


def dequant_mxfp4_block(
    w_i8: np.ndarray,
    scale_e8m0: np.ndarray,
    *,
    fp4_block: int = 32,
) -> np.ndarray:
    """MXFP4 × UE8M0 per-32 → fp32 `[out, in]`.

    Input `w_i8` shape `[out, in // 2]`. Unpacks two FP4 values per byte:
    low nibble → col 2j, high nibble → col 2j+1. Matches the packing in
    HF `inference/convert.py::cast_e2m1fn_to_e4m3fn`.
    """
    if w_i8.ndim != 2:
        raise RuntimeError(f"MXFP4 weight must be 2D, got shape {w_i8.shape}")
    if w_i8.dtype != np.int8 and w_i8.dtype != np.uint8:
        raise RuntimeError(f"Expected I8/U8 packed MXFP4, got {w_i8.dtype}")
    out_dim, packed_in = w_i8.shape
    in_dim = packed_in * 2
    if in_dim % fp4_block != 0:
        raise RuntimeError(f"in_dim {in_dim} not divisible by fp4_block {fp4_block}")
    u = w_i8.view(np.uint8)
    low = (u & 0x0F).astype(np.int32)
    high = ((u >> 4) & 0x0F).astype(np.int32)
    # Interleave: col index c = 2*j + {0,1}.
    vals = np.empty((out_dim, in_dim), dtype=np.float32)
    vals[:, 0::2] = _FP4_TABLE[low]
    vals[:, 1::2] = _FP4_TABLE[high]
    # Apply per-32 E8M0 scale.
    s = _block_scale_to_fp32(scale_e8m0)
    exp_s0, exp_s1 = s.shape
    if exp_s0 != out_dim or exp_s1 != in_dim // fp4_block:
        raise RuntimeError(
            f"MXFP4 scale shape {s.shape} does not match expected "
            f"({out_dim}, {in_dim // fp4_block})."
        )
    s_full = np.repeat(s, fp4_block, axis=1)
    return vals * s_full
