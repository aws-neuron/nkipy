//===- nkipy-opt.cpp - MLIR Optimizer Driver ------------------------------===//
//
// This file implements the 'nkipy-opt' tool, which is the nkipy analog of
// mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/TransformOps/NkipyTransformOps.h"

// Include Transform dialect for knob-driven-tiling pass
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::nkipy::registerNkipyPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::nkipy::NkipyDialect>();
  registry.insert<mlir::transform::TransformDialect>();  // Required for knob-driven-tiling
  
  // Note: bufferization dialect is included in registerAllDialects(), but
  // we explicitly register the transform extension below

  mlir::scf::registerTransformDialectExtension(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::bufferization::registerTransformDialectExtension(registry);
  mlir::nkipy::registerTransformDialectExtension(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Nkipy optimizer driver\n", registry));
}
