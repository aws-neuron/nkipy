//===- IRHelpers.h - Shared IR utility functions ----------------*- C++ -*-===//
//
// Small helpers for querying MLIR values, shared across multiple passes.
//
//===----------------------------------------------------------------------===//

#ifndef NKIPY_TRANSFORMS_IRHELPERS_H
#define NKIPY_TRANSFORMS_IRHELPERS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include <optional>

namespace mlir {
namespace nkipy {

/// Return true if `op` is nested inside an nkipy dialect region (e.g., the
/// reference_impl body of nkipy.gather).  These regions exist only for CPU
/// simulation and should be skipped by NISA-path passes.
inline bool isInsideNkipyRegion(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getDialect() &&
        parent->getDialect()->getNamespace() == "nkipy")
      return true;
  }
  return false;
}

/// Get the constant integer value from a Value, or std::nullopt if not constant.
/// Works for arith.constant with IntegerAttr and arith.constant_index.
std::optional<int64_t> getConstantInt(Value v);

/// Walk through view chains (subview, collapse_shape, expand_shape, etc.)
/// to find the base memref allocation. Uses ViewLikeOpInterface.
Value getBaseMemRef(Value v);

/// Extract the nkipy memory space kind from a memref type, if present.
/// Returns std::nullopt if the type is not a memref or has no nkipy mem space.
std::optional<nkipy::MemSpaceEnum> getNkipyMemSpace(Type type);

/// Convenience predicates on memref memory space.  Return false if the type
/// is not a memref or has no nkipy memory space attribute.
inline bool isHbm(Type type) {
  auto ms = getNkipyMemSpace(type);
  return ms && *ms == nkipy::MemSpaceEnum::Hbm;
}
inline bool isSharedHbm(Type type) {
  auto ms = getNkipyMemSpace(type);
  return ms && *ms == nkipy::MemSpaceEnum::SharedHbm;
}
inline bool isSbuf(Type type) {
  auto ms = getNkipyMemSpace(type);
  return ms && *ms == nkipy::MemSpaceEnum::Sbuf;
}
inline bool isPsum(Type type) {
  auto ms = getNkipyMemSpace(type);
  return ms && *ms == nkipy::MemSpaceEnum::Psum;
}
inline bool isAnyHbm(Type type) {
  auto ms = getNkipyMemSpace(type);
  return ms &&
         (*ms == nkipy::MemSpaceEnum::Hbm || *ms == nkipy::MemSpaceEnum::SharedHbm);
}

/// Walk up the parent chain from `op` until finding an ancestor that lives
/// directly in `block`.  Returns nullptr if `op` is not nested under `block`.
Operation *getAncestorInBlock(Operation *op, Block *block);

} // namespace nkipy
} // namespace mlir

#endif // NKIPY_TRANSFORMS_IRHELPERS_H
