//===- OpClassification.h - Shared op classification helpers ----*- C++ -*-===//
//
// Utility functions for classifying linalg ops (elementwise, reduction, matmul)
// shared across multiple passes.
//
//===----------------------------------------------------------------------===//

#ifndef NKIPY_TRANSFORMS_OPCLASSIFICATION_H
#define NKIPY_TRANSFORMS_OPCLASSIFICATION_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace nkipy {

/// Named unary elementwise ops (exp, log, tanh, negf, abs, ceil, floor,
/// sqrt, reciprocal, square, copy).
bool isNamedUnaryElementwiseOp(StringRef opName);

/// Named binary elementwise ops (add, sub, mul, div).
bool isNamedBinaryElementwiseOp(StringRef opName);

/// Any named elementwise op (unary or binary).
bool isNamedElementwiseOp(StringRef opName);

/// linalg.generic with all-parallel iterator types.
bool isElementwiseGeneric(linalg::LinalgOp linalgOp);

/// Named elementwise op OR all-parallel generic.
bool isElementwiseOp(linalg::LinalgOp linalgOp);

/// linalg.generic with at least one reduction iterator type.
bool isReductionGeneric(linalg::LinalgOp linalgOp);

/// linalg.matmul or linalg.batch_matmul (by op name).
bool isMatmulOp(StringRef opName);

/// linalg.matmul or linalg.batch_matmul (from a LinalgOp).
bool isMatmulOp(linalg::LinalgOp linalgOp);

/// Elementwise, reduction, or matmul — ops that receive layout annotations.
bool isAnnotatableOp(linalg::LinalgOp linalgOp);

} // namespace nkipy
} // namespace mlir

#endif // NKIPY_TRANSFORMS_OPCLASSIFICATION_H
