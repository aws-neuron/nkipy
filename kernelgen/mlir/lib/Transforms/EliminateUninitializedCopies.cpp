//===- EliminateUninitializedCopies.cpp - Remove copies from uninit allocs ===//
//
// This pass eliminates memref.copy operations where the source is a freshly
// allocated buffer that has never been written to (contains undefined values).
// Such copies are effectively no-ops and can be safely eliminated.
//
// This commonly occurs after buffer promotion when the original tensor was
// freshly allocated (e.g., for accumulator initialization).
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using nkipy::getBaseMemRef;

namespace {

/// Check if `a` executes before `b` in program order. Walks up to the nearest
/// common ancestor block to compare positions. Conservative (returns true) if
/// no common ancestor is found.
static bool executesBefore(Operation *a, Operation *b) {
  if (a->getBlock() == b->getBlock())
    return a->isBeforeInBlock(b);

  // Collect a's ancestor chain (block pointers)
  SmallPtrSet<Block *, 8> aBlocks;
  for (Operation *cur = a; cur; cur = cur->getParentOp())
    aBlocks.insert(cur->getBlock());

  // Walk b up to find a common ancestor block
  for (Operation *bAnc = b; bAnc; bAnc = bAnc->getParentOp()) {
    if (aBlocks.count(bAnc->getBlock())) {
      Block *common = bAnc->getBlock();
      Operation *aAnc = a;
      while (aAnc->getBlock() != common)
        aAnc = aAnc->getParentOp();
      return aAnc->isBeforeInBlock(bAnc);
    }
  }
  return true; // conservative
}

/// Check if `val` (or any view derived from it) is written to by any user
/// that executes before `beforeOp`. Recurses through view-like ops to catch
/// writes to subviews of the allocation.
static bool hasAnyWriteBefore(Value val, Operation *beforeOp) {
  for (Operation *user : val.getUsers()) {
    if (user == beforeOp)
      continue;
    // View-like ops: recurse into their users
    if (isa<memref::SubViewOp, memref::CastOp, memref::CollapseShapeOp,
            memref::ExpandShapeOp>(user)) {
      if (hasAnyWriteBefore(user->getResult(0), beforeOp))
        return true;
      continue;
    }
    bool isWrite = false;
    if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      isWrite = (copy.getTarget() == val);
    } else if (isa<memref::StoreOp>(user)) {
      isWrite = true;
    } else if (auto dps = dyn_cast<DestinationStyleOpInterface>(user)) {
      isWrite = llvm::is_contained(dps.getDpsInits(), val);
    }
    if (isWrite && executesBefore(user, beforeOp))
      return true;
  }
  return false;
}

struct EliminateUninitializedCopiesPass
    : public PassWrapper<EliminateUninitializedCopiesPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateUninitializedCopiesPass)

  StringRef getArgument() const final {
    return "eliminate-uninitialized-copies";
  }
  
  StringRef getDescription() const final {
    return "Eliminate memref.copy operations where the source is an "
           "uninitialized allocation";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    
    SmallVector<memref::CopyOp> toErase;
    
    func.walk([&](memref::CopyOp copyOp) {
      Value srcBase = getBaseMemRef(copyOp.getSource());
      if (auto alloc = srcBase.getDefiningOp<memref::AllocOp>()) {
        if (!hasAnyWriteBefore(alloc.getResult(), copyOp)) {
          llvm::errs() << "[EliminateUninitializedCopies] Eliminating copy from "
                          "uninitialized alloc: "
                       << *copyOp << "\n";
          toErase.push_back(copyOp);
        }
      }
    });
    
    // Erase the identified copy operations
    for (auto copyOp : toErase) {
      copyOp.erase();
    }
    
    if (!toErase.empty()) {
      llvm::errs() << "[EliminateUninitializedCopies] Eliminated "
                   << toErase.size() << " copy operation(s)\n";
    }
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<func::FuncOp>>
createEliminateUninitializedCopiesPass() {
  return std::make_unique<EliminateUninitializedCopiesPass>();
}

} // namespace nkipy
} // namespace mlir
