//===- OpClassification.cpp - Shared op classification helpers -------------===//

#include "nkipy/Transforms/OpClassification.h"

using namespace mlir;

namespace mlir {
namespace nkipy {

bool isNamedUnaryElementwiseOp(StringRef opName) {
  return opName == "linalg.exp" || opName == "linalg.log" ||
         opName == "linalg.tanh" || opName == "linalg.negf" ||
         opName == "linalg.abs" || opName == "linalg.ceil" ||
         opName == "linalg.floor" || opName == "linalg.sqrt" ||
         opName == "linalg.reciprocal" || opName == "linalg.square" ||
         opName == "linalg.copy";
}

bool isNamedBinaryElementwiseOp(StringRef opName) {
  return opName == "linalg.add" || opName == "linalg.sub" ||
         opName == "linalg.mul" || opName == "linalg.div" ||
         opName == "linalg.max" || opName == "linalg.min";
}

bool isNamedElementwiseOp(StringRef opName) {
  return isNamedUnaryElementwiseOp(opName) ||
         isNamedBinaryElementwiseOp(opName);
}

bool isElementwiseGeneric(linalg::LinalgOp linalgOp) {
  auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
  if (!genericOp)
    return false;
  return llvm::all_of(genericOp.getIteratorTypesArray(),
      [](utils::IteratorType t) {
        return t == utils::IteratorType::parallel;
      });
}

bool isElementwiseOp(linalg::LinalgOp linalgOp) {
  return isNamedElementwiseOp(linalgOp->getName().getStringRef()) ||
         isElementwiseGeneric(linalgOp);
}

bool isReductionGeneric(linalg::LinalgOp linalgOp) {
  auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
  if (!genericOp)
    return false;
  return llvm::any_of(genericOp.getIteratorTypesArray(),
      [](utils::IteratorType t) {
        return t == utils::IteratorType::reduction;
      });
}

bool isMatmulOp(StringRef opName) {
  return opName == "linalg.matmul" || opName == "linalg.batch_matmul";
}

bool isMatmulOp(linalg::LinalgOp linalgOp) {
  return isMatmulOp(linalgOp->getName().getStringRef());
}

bool isAnnotatableOp(linalg::LinalgOp linalgOp) {
  return isElementwiseOp(linalgOp) || isReductionGeneric(linalgOp) ||
         isMatmulOp(linalgOp);
}

} // namespace nkipy
} // namespace mlir
