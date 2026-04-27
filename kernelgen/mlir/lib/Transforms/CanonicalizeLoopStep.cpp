//===- CanonicalizeLoopStep.cpp - Canonicalize scf.for steps to 1 ---------===//
//
// This pass transforms scf.for loops to have step=1, which simplifies index
// computation for subsequent passes and is required for NISA lowering.
//
// The transformation:
//   scf.for %i = %lb to %ub step %step { ... uses %i ... }
//   =>
//   scf.for %i_idx = 0 to (%ub - %lb) / %step step 1 {
//     %i = %lb + %i_idx * %step
//     ... uses %i ...
//   }
//
// Loops are processed in post-order (innermost first) to handle nesting.
//
//===----------------------------------------------------------------------===//

#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {

namespace {

/// Canonicalize a single scf.for loop to have step=1.
/// Returns true if the loop was modified.
static bool canonicalizeForOp(scf::ForOp forOp) {
  auto stepConst = getConstantInt(forOp.getStep());

  // Skip if already step=1 or step is dynamic (can't canonicalize).
  if (!stepConst || *stepConst == 1)
    return false;

  auto lbConst = getConstantInt(forOp.getLowerBound());
  auto ubConst = getConstantInt(forOp.getUpperBound());

  // Check divisibility for all-constant case.
  if (lbConst && ubConst) {
    int64_t range = *ubConst - *lbConst;
    if (range % *stepConst != 0) {
      llvm::errs() << "[CanonicalizeLoopStep] Skipping: range " << range
                   << " not divisible by step " << *stepConst << "\n";
      return false;
    }
  }

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  Value step = forOp.getStep();

  // Compute new bounds: trip count = (ub - lb) / step.
  // Canonicalize runs after this pass and will fold constants.
  Value range = builder.create<arith::SubIOp>(loc, ub, lb);
  Value tripCount = builder.create<arith::DivUIOp>(loc, range, step);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

  // Reconstruct original IV at top of body: i = lb + idx * step.
  builder.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  Value scaled = builder.create<arith::MulIOp>(loc, iv, step);
  Value originalIV = builder.create<arith::AddIOp>(loc, lb, scaled);
  SmallPtrSet<Operation *, 2> exceptions;
  exceptions.insert(scaled.getDefiningOp());
  exceptions.insert(originalIV.getDefiningOp());
  iv.replaceAllUsesExcept(originalIV, exceptions);

  // Update loop bounds in place.
  forOp.setLowerBound(zero);
  forOp.setUpperBound(tripCount);
  forOp.setStep(one);

  llvm::errs() << "[CanonicalizeLoopStep] Transformed loop (step="
               << *stepConst << ") to step=1\n";
  return true;
}

struct NkipyCanonicalizeLoopStepPass
    : public CanonicalizeLoopStepBase<NkipyCanonicalizeLoopStepPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // PostOrder walk visits children before parents, so inner loops
    // are canonicalized before their enclosing outer loops.
    bool changed = false;
    func.walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
      if (canonicalizeForOp(forOp))
        changed = true;
    });

    if (changed)
      llvm::errs() << "[CanonicalizeLoopStep] Pass completed with modifications\n";
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeLoopStepPass() {
  return std::make_unique<NkipyCanonicalizeLoopStepPass>();
}

} // namespace nkipy
} // namespace mlir
