//===- InsertMemRefDealloc.cpp - Insert dealloc after last use -----------===//
//
// This pass analyzes the lifetime of memref.alloc operations with NISA memory
// space attributes and inserts memref.dealloc after each allocation's last use.
// These are later lowered to nisa.release by the linalg-to-nisa pass.
//
// Uses last-use deallocation: traces through view chains (subview,
// reinterpret_cast, collapse_shape, expand_shape, cast) via BFS to find the
// last consumer, then inserts dealloc immediately after. Falls back to
// scope-based dealloc if the alloc has no uses. Uses inside nested regions
// (e.g., scf.for loop bodies) are mapped to their ancestor op in the
// allocation's block.
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "insert-memref-dealloc"

using namespace mlir;
using nkipy::getAncestorInBlock;
using nkipy::getBaseMemRef;
using nkipy::getNkipyMemSpace;

namespace {

/// Collect all allocations that escape via function return values.
/// For each return operand, trace back through view chains to find the base
/// allocation and add it to the set of escaped allocations.
static void collectEscapedAllocations(func::FuncOp func,
                                      llvm::DenseSet<Operation *> &escaped) {
  func.walk([&](func::ReturnOp returnOp) {
    for (Value operand : returnOp.getOperands()) {
      // Only care about memref types
      if (!isa<MemRefType>(operand.getType()))
        continue;

      // Trace back to base allocation
      Value base = getBaseMemRef(operand);
      if (auto allocOp = base.getDefiningOp<memref::AllocOp>()) {
        escaped.insert(allocOp);
      }
    }
  });
}

struct InsertMemRefDeallocPass
    : public PassWrapper<InsertMemRefDeallocPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertMemRefDeallocPass)

  // Track if any errors occurred during the pass
  bool hasError = false;

  StringRef getArgument() const final { return "insert-memref-dealloc"; }

  StringRef getDescription() const final {
    return "Insert memref.dealloc operations to mark allocation lifetime ends";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    hasError = false;

    // First, collect all allocations that escape via return values
    llvm::DenseSet<Operation *> escapedAllocs;
    collectEscapedAllocations(func, escapedAllocs);

    // Collect allocations to process and validate they have NISA memspace.
    // Skip ops inside nkipy regions (reference_impl bodies are for CPU
    // simulation only and don't have NISA memory spaces).
    SmallVector<memref::AllocOp> allocOps;
    func.walk([&](memref::AllocOp op) {
      if (nkipy::isInsideNkipyRegion(op))
        return;
      auto memSpace = getNkipyMemSpace(op.getType());
      if (memSpace) {
        // Skip SharedHbm, Hbm (externally managed) and Constant (scalar
        // broadcasts — small and not real SBUF allocations).
        if (*memSpace == nkipy::MemSpaceEnum::SharedHbm ||
            *memSpace == nkipy::MemSpaceEnum::Hbm ||
            *memSpace == nkipy::MemSpaceEnum::Constant)
          return;
        allocOps.push_back(op);
        return;
      }

      // Error: memref.alloc without memory space annotation.
      op.emitError() << "memref.alloc must have an nkipy memory space "
                     << "annotation (SBUF, PSUM, HBM, or SHAREDHBM)";
      hasError = true;
    });

    if (hasError) {
      signalPassFailure();
      return;
    }

    if (allocOps.empty())
      return;

    int numInserted = 0;
    for (auto allocOp : allocOps) {
      // Skip if allocation escapes (returned from function)
      if (escapedAllocs.contains(allocOp))
        continue;

      Block *block = allocOp->getBlock();
      Value allocVal = allocOp.getResult();

      // Find the last use of this alloc (or any derived view) in its block.
      // We trace through the entire view chain (subview, reinterpret_cast,
      // collapse_shape, expand_shape, cast) because the alloc may be
      // accessed only through derived views, not directly.
      // Uses inside nested regions (e.g., scf.for loop bodies) are mapped
      // to their ancestor op that lives directly in `block`.
      Operation *lastUseOp = nullptr;

      // BFS through all transitively derived values from the alloc.
      SmallVector<Value> worklist;
      llvm::DenseSet<Value> visited;
      worklist.push_back(allocVal);
      visited.insert(allocVal);

      while (!worklist.empty()) {
        Value val = worklist.pop_back_val();
        for (Operation *user : val.getUsers()) {
          // Walk up to find the ancestor in allocOp's block.
          Operation *ancestor = getAncestorInBlock(user, block);
          if (!ancestor)
            continue;
          if (!lastUseOp || lastUseOp->isBeforeInBlock(ancestor))
            lastUseOp = ancestor;

          // If this user produces a view-like result, trace through it.
          if (isa<ViewLikeOpInterface>(user)) {
            for (Value result : user->getResults()) {
              if (isa<MemRefType>(result.getType()) && !visited.contains(result)) {
                visited.insert(result);
                worklist.push_back(result);
              }
            }
          }
        }
      }

      if (lastUseOp) {
        // Insert dealloc right after the last use.  This allows the backend
        // allocator to reuse the SBUF space earlier, reducing peak memory
        // pressure after loop unrolling.
        OpBuilder builder(lastUseOp->getBlock(), ++Block::iterator(lastUseOp));
        builder.create<memref::DeallocOp>(allocOp.getLoc(), allocVal);
      } else {
        // No uses found — fall back to scope-based dealloc.
        Operation *terminator = block->getTerminator();
        OpBuilder builder(terminator);
        builder.create<memref::DeallocOp>(allocOp.getLoc(), allocVal);
      }
      ++numInserted;
    }

    LLVM_DEBUG(llvm::dbgs() << "Inserted " << numInserted
                           << " dealloc operation(s)\n");
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<func::FuncOp>> createInsertMemRefDeallocPass() {
  return std::make_unique<InsertMemRefDeallocPass>();
}

} // namespace nkipy
} // namespace mlir
