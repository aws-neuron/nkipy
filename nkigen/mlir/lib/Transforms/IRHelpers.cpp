//===- IRHelpers.cpp - Shared IR utility functions -------------------------===//

#include "nkipy/Transforms/IRHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace nkipy {

std::optional<int64_t> getConstantInt(Value v) {
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  if (auto constOp = v.getDefiningOp<arith::ConstantIndexOp>())
    return constOp.value();
  return std::nullopt;
}

Value getBaseMemRef(Value v) {
  while (auto *def = v.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(def)) {
      v = view.getViewSource();
      continue;
    }
    break;
  }
  return v;
}

std::optional<nkipy::MemSpaceEnum> getNkipyMemSpace(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType)
    return std::nullopt;
  auto memSpaceAttr = memrefType.getMemorySpace();
  if (!memSpaceAttr)
    return std::nullopt;
  if (auto ms = dyn_cast<nkipy::MemSpaceEnumAttr>(memSpaceAttr))
    return ms.getValue();
  return std::nullopt;
}

Operation *getAncestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

} // namespace nkipy
} // namespace mlir
