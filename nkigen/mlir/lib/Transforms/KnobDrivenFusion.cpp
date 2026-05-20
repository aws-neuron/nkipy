//===- KnobDrivenFusion.cpp - Fuse sibling scf.for loops via fuse_op -------===//
//
// Walks each `nkipy.fuse_op(%a, %b, ...)` in the function and fuses the
// `scf.for` loops producing the listed tensors into a single loop using
// upstream MLIR's fuseIndependentSiblingForLoops helper.
//
// Runs after apply-and-strip-transforms (so the per-op scf.for loops exist)
// and before canonicalize-loop-step (so loop bounds are still in their
// original step=tile form).  The helper itself does no legality check —
// we verify matching lower/upper/step bounds before calling it.
//
//===----------------------------------------------------------------------===//

#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {
namespace {

/// Walk back through aliasing ops to find the scf.for whose result is
/// `val`.  Returns null if `val`'s defining chain doesn't lead to one.
static scf::ForOp findProducingForLoop(Value val) {
  Operation *def = val.getDefiningOp();
  if (!def)
    return nullptr;
  return dyn_cast<scf::ForOp>(def);
}

/// Returns true if two `scf.for` loops have matching lower bound, upper
/// bound, and step (the precondition for fuseIndependentSiblingForLoops).
/// Constants emitted by independent tilings are distinct SSA values even
/// when their numeric value is identical, so prefer constant-value equality.
static bool sameBound(Value a, Value b) {
  if (a == b)
    return true;
  std::optional<int64_t> ca = getConstantIntValue(a);
  std::optional<int64_t> cb = getConstantIntValue(b);
  return ca.has_value() && cb.has_value() && *ca == *cb;
}

static bool sameBounds(scf::ForOp a, scf::ForOp b) {
  return sameBound(a.getLowerBound(), b.getLowerBound()) &&
         sameBound(a.getUpperBound(), b.getUpperBound()) &&
         sameBound(a.getStep(), b.getStep());
}

/// Hoist any pure setup ops textually between `first` and `second` above
/// `first`, so their defs dominate the fused loop's position after sibling
/// fusion merges `second` into `first`.
static void hoistSetupOpsBetween(Operation *first, Operation *second) {
  SmallVector<Operation *> toHoist;
  for (Operation *op = first->getNextNode(); op && op != second;
       op = op->getNextNode()) {
    if (isMemoryEffectFree(op))
      toHoist.push_back(op);
  }
  for (Operation *op : toHoist)
    op->moveBefore(first);
}

/// Recursively fuse pairs of consecutive same-bounds sibling scf.for loops
/// inside `parent`'s body.  Returns the (possibly rewritten) outer loop.
/// Used after outer-level fusion to merge the inner loops that came along
/// for the ride.
static void fuseInnerSiblings(scf::ForOp parent, IRRewriter &rewriter) {
  bool changed = true;
  while (changed) {
    changed = false;
    Block *body = parent.getBody();
    scf::ForOp prev;
    for (Operation &op : *body) {
      auto curr = dyn_cast<scf::ForOp>(&op);
      if (!curr) {
        prev = nullptr;
        continue;
      }
      if (prev && sameBounds(prev, curr)) {
        hoistSetupOpsBetween(prev, curr);
        scf::ForOp fused =
            ::mlir::fuseIndependentSiblingForLoops(curr, prev, rewriter);
        fuseInnerSiblings(fused, rewriter);
        changed = true;
        break;
      }
      prev = curr;
    }
  }
}

struct NkipyKnobDrivenFusionPass
    : public KnobDrivenFusionBase<NkipyKnobDrivenFusionPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<nkipy::FuseOp> fuseOps;
    func.walk([&](nkipy::FuseOp op) { fuseOps.push_back(op); });

    if (fuseOps.empty())
      return;

    IRRewriter rewriter(&getContext());

    for (nkipy::FuseOp fuseOp : fuseOps) {
      // Resolve each target to its producing scf.for.
      SmallVector<scf::ForOp> loops;
      for (Value target : fuseOp.getTargets()) {
        scf::ForOp loop = findProducingForLoop(target);
        if (!loop) {
          fuseOp.emitError() << "fuse_op target is not produced by an "
                                "scf.for (was it tiled?)";
          return signalPassFailure();
        }
        loops.push_back(loop);
      }

      // All loops must share lower/upper/step.  Check against loops[0].
      for (size_t i = 1; i < loops.size(); i++) {
        if (!sameBounds(loops[0], loops[i])) {
          fuseOp.emitError()
              << "fuse_op targets have mismatched loop bounds (target #0 "
              << "vs #" << i << "); ensure their tile_op tile_size matches";
          return signalPassFailure();
        }
      }

      // Fuse outer loops into loops[0] left-to-right.  Each independently
      // tiled loop and its setup ops (bound constants, tensor.empty inits)
      // sit textually after loops[0]; hoist them before fusing so their
      // defs dominate the fused loop's position.
      // fuseIndependentSiblingForLoops takes (target, source): target is
      // merged INTO source, which becomes the surviving fused loop.
      scf::ForOp fused = loops[0];
      for (size_t i = 1; i < loops.size(); i++) {
        hoistSetupOpsBetween(fused, loops[i]);
        fused = ::mlir::fuseIndependentSiblingForLoops(loops[i], fused,
                                                       rewriter);
      }

      // Outer-level fusion brought along each loop's inner nest as a
      // sibling inside the merged outer loop.  Recursively fuse any
      // consecutive same-bounds inner siblings so the merged body has a
      // single nest at every level (not a loop-of-loops at level 1
      // followed by another loop-of-loops at level 2).
      fuseInnerSiblings(fused, rewriter);

      llvm::errs() << "[KnobDrivenFusion] Fused " << loops.size()
                   << " scf.for nest(s)\n";

      rewriter.eraseOp(fuseOp);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createKnobDrivenFusionPass() {
  return std::make_unique<NkipyKnobDrivenFusionPass>();
}

} // namespace nkipy
} // namespace mlir
