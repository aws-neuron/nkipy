//===- NkipyTransformOps.h - Nkipy Transform Operations ---------*- C++ -*-===//
//
// Custom transform dialect operations for NKIPyKernelGen.
//
//===----------------------------------------------------------------------===//

#ifndef NKIPY_TRANSFORMOPS_NKIPYTRANSFORMOPS_H
#define NKIPY_TRANSFORMOPS_NKIPYTRANSFORMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace nkipy {

/// Registers the Nkipy transform ops extension with the transform dialect.
void registerTransformDialectExtension(DialectRegistry &registry);

} // namespace nkipy
} // namespace mlir

#define GET_OP_CLASSES
#include "nkipy/TransformOps/NkipyTransformOps.h.inc"

#endif // NKIPY_TRANSFORMOPS_NKIPYTRANSFORMOPS_H
