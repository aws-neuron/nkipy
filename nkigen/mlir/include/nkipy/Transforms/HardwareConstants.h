//===- HardwareConstants.h - NeuronCore target-info accessors ---*- C++ -*-===//
//
// Per-target hardware information used by the nkipy passes.
//
// This is the single source of truth used by the nkipy C++ passes; the
// numbers below mirror `nki.backends.mlir_tracer.target_info` in the public
// `nki` Python wheel.  When a new Trainium generation is added, update both
// places.  The accessors return `std::nullopt` for unknown targets so callers
// can surface a pass-level error.
//
//===----------------------------------------------------------------------===//

#ifndef NKIPY_TRANSFORMS_HARDWARECONSTANTS_H
#define NKIPY_TRANSFORMS_HARDWARECONSTANTS_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace nkipy {

namespace detail {

struct TargetInfo {
  int64_t sbufNumPartitions;
  int64_t sbufPartitionUsableSize; // bytes
  int64_t psumNumPartitions;
  int64_t psumPartitionUsableSize; // bytes
  int64_t psumNumBanks;
  int64_t psumBankSize; // bytes per partition per bank
};

inline std::optional<TargetInfo> lookupTarget(llvm::StringRef target) {
  if (target == "trn1")
    return TargetInfo{128, 180224, 128, 16384, 8, 2048};
  if (target == "trn2")
    return TargetInfo{128, 212984, 128, 16384, 8, 2048};
  if (target == "trn3")
    return TargetInfo{128, 245752, 128, 16384, 8, 2048};
  return std::nullopt;
}

} // namespace detail

/// Number of SBUF partitions on `target` (the partition-dim limit).
inline std::optional<int64_t> getSbufNumPartitions(llvm::StringRef target) {
  if (auto info = detail::lookupTarget(target))
    return info->sbufNumPartitions;
  return std::nullopt;
}

/// Per-partition usable SBUF size in bytes for `target`.
inline std::optional<int64_t>
getSbufPartitionUsableSize(llvm::StringRef target) {
  if (auto info = detail::lookupTarget(target))
    return info->sbufPartitionUsableSize;
  return std::nullopt;
}

/// Per-partition usable PSUM size in bytes for `target`.  This is the
/// effective ceiling for a matmul tile's free dimension on the PSUM side.
inline std::optional<int64_t>
getPsumPartitionUsableSize(llvm::StringRef target) {
  if (auto info = detail::lookupTarget(target))
    return info->psumPartitionUsableSize;
  return std::nullopt;
}

/// Per-partition PSUM bank size in bytes for `target`.  Each matmul writes
/// into a single PSUM bank, so dividing by the element size gives a
/// reasonable per-tile cap on the matmul output's free dimension.
inline std::optional<int64_t> getPsumBankSize(llvm::StringRef target) {
  if (auto info = detail::lookupTarget(target))
    return info->psumBankSize;
  return std::nullopt;
}

/// Tile-shape heuristic: per-tile cap on the matmul output's free dimension
/// in *f32 elements*.  Matches `psum_bank_size_bytes / sizeof(f32)`.  Tiles
/// in higher-precision element types still fit within the PSUM bank.
inline std::optional<int64_t>
getMatmulFreeDimTileCap(llvm::StringRef target) {
  if (auto bankBytes = getPsumBankSize(target))
    return *bankBytes / 4;
  return std::nullopt;
}

} // namespace nkipy
} // namespace mlir

#endif // NKIPY_TRANSFORMS_HARDWARECONSTANTS_H
