

#ifndef NKIPY_MLIR_PASSDETAIL_H
#define NKIPY_MLIR_PASSDETAIL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace nkipy {

#define GEN_PASS_CLASSES
#include "nkipy/Transforms/Passes.h.inc"

} // namespace nkipy
} // end namespace mlir

#endif // Allo_MLIR_PASSDETAIL_H
