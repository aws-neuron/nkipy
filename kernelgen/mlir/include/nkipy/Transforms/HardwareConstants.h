//===- HardwareConstants.h - NeuronCore hardware limits ---------*- C++ -*-===//

#ifndef NKIPY_TRANSFORMS_HARDWARECONSTANTS_H
#define NKIPY_TRANSFORMS_HARDWARECONSTANTS_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace nkipy {

/// Maximum partition dimension size for NeuronCore hardware.
static constexpr int64_t MAX_PARTITION_DIM = 128;

/// Maximum free dimension size for matmul operands.
static constexpr int64_t MAX_FREE_DIM_MATMUL = 512;

/// Usable SBUF partition size in bytes for `target`. Values mirror
/// build-tools/j2gen/target_info.py in private-nki-staging. Kept local so
/// nkipy-opt does not need to link against NISA-internal target-info libs.
inline std::optional<int64_t>
getSbufPartitionUsableSize(llvm::StringRef target) {
  if (target == "trn1")
    return static_cast<int64_t>(192 * 1024 - 16384);
  if (target == "trn2")
    return static_cast<int64_t>(224 * 1024 - 16384 - 8);
  if (target == "trn3")
    return static_cast<int64_t>(256 * 1024 - 16384 - 8);
  return std::nullopt;
}

} // namespace nkipy
} // namespace mlir

#endif // NKIPY_TRANSFORMS_HARDWARECONSTANTS_H
