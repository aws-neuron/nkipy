//===- AssignLinalgOpIds.cpp - Assign unique IDs to linalg ops ------------===//
//
// This pass assigns unique nkipy.op_id attributes to ALL linalg operations.
//
// The op_id enables per-instance matching during transform dialect application,
// allowing different tile sizes to be applied to different instances of the
// same linalg operation type.
//
//===----------------------------------------------------------------------===//

#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {

namespace {

struct NkipyAssignLinalgOpIdsPass
    : public AssignLinalgOpIdsBase<NkipyAssignLinalgOpIdsPass> {
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    
    llvm::errs() << "[AssignLinalgOpIds] Processing function: " 
                 << func.getName() << "\n";
    
    // Counter for unique op_id
    int64_t opIdCounter = 0;
    
    // Walk all linalg ops and assign unique IDs.
    // Skip ops inside nkipy regions (e.g., reference_impl bodies) — these
    // exist only for CPU simulation and must not participate in tiling.
    func.walk([&](linalg::LinalgOp linalgOp) {
      if (isInsideNkipyRegion(linalgOp))
        return;
      // Only add op_id if it doesn't already have one
      if (!linalgOp->hasAttr("nkipy.op_id")) {
        OpBuilder builder(linalgOp);
        linalgOp->setAttr("nkipy.op_id", 
                          builder.getI64IntegerAttr(opIdCounter++));
        llvm::errs() << "[AssignLinalgOpIds] Added op_id=" 
                     << (opIdCounter - 1) << " to " 
                     << linalgOp->getName() << "\n";
      }
    });
    
    llvm::errs() << "[AssignLinalgOpIds] Assigned " << opIdCounter 
                 << " unique op_ids\n";
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createAssignLinalgOpIdsPass() {
  return std::make_unique<NkipyAssignLinalgOpIdsPass>();
}

} // namespace nkipy
} // namespace mlir
