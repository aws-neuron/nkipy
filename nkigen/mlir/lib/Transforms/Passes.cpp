#include "nkipy/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
namespace {
#define GEN_PASS_REGISTRATION
#include "nkipy/Transforms/Passes.h.inc"
} // end namespace

void mlir::nkipy::registerNkipyPasses() { ::registerPasses(); }
