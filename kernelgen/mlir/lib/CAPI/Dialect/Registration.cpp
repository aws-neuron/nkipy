
#include "nkipy-c/Dialect/Registration.h"
// #include "nkipy/Transforms/Passes.h"  // COMMENTED: Not using passes in CAPI
#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/TransformOps/NkipyTransformOps.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"

void nkipyMlirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::nkipy::NkipyDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::pdl::PDLDialect,
                  mlir::transform::TransformDialect>();

  // Register Transform dialect extensions (including nkipy transform ops)
  mlir::nkipy::registerTransformDialectExtension(registry);

  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void nkipyMlirRegisterAllPasses() {
  // General passes
  mlir::registerTransformsPasses();

  // Conversion passes
  // Note: registerConversionPasses() registers ALL conversion passes,
  // many of which require additional libraries we don't need.
  // Comment out for now - add specific conversion passes as needed.
  // mlir::registerConversionPasses();

  // Dialect passes
  mlir::affine::registerAffinePasses();
  mlir::arith::registerArithPasses();
  // mlir::LLVM::registerLLVMPasses();  // Requires NVVM and other GPU libraries
  mlir::memref::registerMemRefPasses();
  mlir::registerLinalgPasses();

  // mlir::nkipy::registerNkipyPasses();
}
