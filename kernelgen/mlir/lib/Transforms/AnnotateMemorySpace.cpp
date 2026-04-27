#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

#define DEBUG_TYPE "annotate-memory-space"

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {

namespace {

/// Verify that a memref annotated with CONSTANT was filled with a scalar constant.
/// Returns true if valid (is a scalar broadcast), false otherwise.
/// The pattern we look for is:
///   %buffer = memref.alloc()
///   %cst = arith.constant <scalar>
///   linalg.fill ins(%cst) outs(%buffer)
///   nkipy.annotate(%buffer, CONSTANT)
static bool isScalarBroadcast(Value memref) {
  Value base = getBaseMemRef(memref);
  for (Operation *user : base.getUsers()) {
    if (auto fillOp = dyn_cast<linalg::FillOp>(user)) {
      if (fillOp.getOutputs()[0] == base &&
          fillOp.getInputs()[0].getDefiningOp<arith::ConstantOp>())
        return true;
    }
  }
  return false;
}

struct NkipyAnnotateMemorySpacePass
    : public AnnotateMemorySpaceBase<NkipyAnnotateMemorySpacePass> {

  /// Rewrite a list of types, adding memSpaceAttr to any bare MemRefType.
  /// Returns true if any type was changed.
  static bool addMemSpaceToTypes(ArrayRef<Type> types, Attribute memSpaceAttr,
                                 SmallVectorImpl<Type> &out) {
    bool changed = false;
    for (Type ty : types) {
      auto memrefType = dyn_cast<MemRefType>(ty);
      if (!memrefType) {
        out.push_back(ty);
        continue;
      }
      assert(!memrefType.getMemorySpace() &&
             "memrefs should not have memory space before this pass");
      out.push_back(MemRefType::get(memrefType.getShape(),
                                    memrefType.getElementType(),
                                    memrefType.getLayout(), memSpaceAttr));
      changed = true;
    }
    return changed;
  }

  /// Annotate function inputs and outputs with SharedHbm memory space.
  void annotateInputOutput(func::FuncOp func) {
    MLIRContext *ctx = func.getContext();
    FunctionType oldType = func.getFunctionType();
    auto sharedHbm =
        nkipy::MemSpaceEnumAttr::get(ctx, nkipy::MemSpaceEnum::SharedHbm);

    SmallVector<Type> newInputs, newResults;
    bool changed = addMemSpaceToTypes(oldType.getInputs(), sharedHbm, newInputs);
    changed |= addMemSpaceToTypes(oldType.getResults(), sharedHbm, newResults);
    if (!changed)
      return;

    func.setType(FunctionType::get(ctx, newInputs, newResults));
    if (func.isDeclaration())
      return;

    // Update block arg types and return operand types.
    Block &entry = func.getBody().front();
    for (auto [i, arg] : llvm::enumerate(entry.getArguments()))
      arg.setType(newInputs[i]);
    auto returnOp = cast<func::ReturnOp>(entry.getTerminator());
    for (auto [i, operand] : llvm::enumerate(returnOp.getOperands()))
      if (i < newResults.size())
        operand.setType(newResults[i]);
  }

  /// Apply nkipy.annotate mem_space to memref types, then erase annotations.
  void applyAnnotations(func::FuncOp func) {
    MLIRContext *ctx = func.getContext();

    SmallVector<nkipy::AnnotateOp> annotateOps;
    func.walk([&](nkipy::AnnotateOp op) { annotateOps.push_back(op); });

    for (auto annotateOp : annotateOps) {
      Value target = annotateOp.getTarget();
      auto memSpace = annotateOp.getMemSpace();

      if (memSpace) {
        auto memrefType = dyn_cast<MemRefType>(target.getType());
        if (!memrefType) {
          LLVM_DEBUG(llvm::dbgs() << "Warning: annotate target is not a memref\n");
          annotateOp.erase();
          continue;
        }

        // CONSTANT is a marker; verify it matches a scalar broadcast pattern.
        if (*memSpace == nkipy::MemSpaceEnum::Constant &&
            !isScalarBroadcast(target)) {
          annotateOp.emitError()
              << "CONSTANT memory space requires a scalar broadcast "
              << "(linalg.fill with arith.constant)";
          signalPassFailure();
          return;
        }
        Attribute memSpaceAttr =
            nkipy::MemSpaceEnumAttr::get(ctx, *memSpace);

        target.setType(MemRefType::get(memrefType.getShape(),
                                       memrefType.getElementType(),
                                       memrefType.getLayout(), memSpaceAttr));
      }
      annotateOp.erase();
    }
  }

  /// Propagate memory space from one value to another.
  /// Returns true if the target type was changed.
  static bool propagateMemSpace(Value from, Value to) {
    auto fromType = cast<MemRefType>(from.getType());
    auto toType = cast<MemRefType>(to.getType());
    if (fromType.getMemorySpace() && !toType.getMemorySpace()) {
      to.setType(MemRefType::get(toType.getShape(), toType.getElementType(),
                                 toType.getLayout(), fromType.getMemorySpace()));
      return true;
    }
    return false;
  }

  /// Resolve memory space conflicts on SubView ops by inserting copies.
  /// When a subview's source and result have different memory spaces
  /// (e.g., SBUF source but SharedHbm result from return type annotation),
  /// replace uses of the subview with a new alloc + copy.
  void resolveSubViewConflicts(func::FuncOp func) {
    MLIRContext *ctx = func.getContext();
    SmallVector<memref::SubViewOp> conflictOps;

    func.walk([&](memref::SubViewOp op) {
      auto srcType = cast<MemRefType>(op.getSource().getType());
      auto dstType = cast<MemRefType>(op.getResult().getType());
      if (srcType.getMemorySpace() && dstType.getMemorySpace() &&
          srcType.getMemorySpace() != dstType.getMemorySpace()) {
        conflictOps.push_back(op);
      }
    });

    if (conflictOps.empty())
      return;

    for (auto op : conflictOps) {
      auto srcType = cast<MemRefType>(op.getSource().getType());
      auto dstType = cast<MemRefType>(op.getResult().getType());

      LLVM_DEBUG(llvm::dbgs()
                 << "Resolving SubView memory space conflict: source "
                 << srcType.getMemorySpace() << " vs result "
                 << dstType.getMemorySpace() << "\n");

      // Fix the subview to match its source memory space.
      auto fixedSubviewType = MemRefType::get(
          dstType.getShape(), dstType.getElementType(),
          dstType.getLayout(), srcType.getMemorySpace());
      op.getResult().setType(fixedSubviewType);

      // Insert alloc + copy after the subview to materialize in target space.
      OpBuilder builder(op->getBlock(), std::next(op->getIterator()));
      auto allocType = MemRefType::get(
          dstType.getShape(), dstType.getElementType(),
          MemRefLayoutAttrInterface(), dstType.getMemorySpace());
      Value alloc = builder.create<memref::AllocOp>(op.getLoc(), allocType);
      builder.create<memref::CopyOp>(op.getLoc(), op.getResult(), alloc);

      // Replace all uses of the subview with the alloc, except the copy.
      op.getResult().replaceAllUsesExcept(alloc,
          alloc.getDefiningOp()->getNextNode());
    }

    // Update function signature to match actual return operand types.
    auto returnOp = cast<func::ReturnOp>(
        func.getBody().front().getTerminator());
    FunctionType funcType = func.getFunctionType();
    SmallVector<Type> newResultTypes;
    for (Value operand : returnOp.getOperands())
      newResultTypes.push_back(operand.getType());
    func.setType(FunctionType::get(ctx, funcType.getInputs(), newResultTypes));
  }

  /// Infer HBM memory space for unannotated allocs that feed into copies
  /// to on-chip memory (e.g., gather output allocs that are DMA-copied
  /// into SBUF). These are internal intermediates, not user-facing.
  /// Returns true if any type was changed.
  bool inferHbmForCopySources(func::FuncOp func) {
    bool changed = false;
    func.walk([&](memref::CopyOp op) {
      auto srcType = cast<MemRefType>(op.getSource().getType());
      auto dstType = cast<MemRefType>(op.getTarget().getType());
      if (srcType.getMemorySpace() || !dstType.getMemorySpace())
        return;
      // Walk backward through subview chain to find the root alloc.
      Value root = op.getSource();
      while (auto sv = root.getDefiningOp<memref::SubViewOp>())
        root = sv.getSource();
      auto rootType = cast<MemRefType>(root.getType());
      if (rootType.getMemorySpace())
        return;
      auto hbm = nkipy::MemSpaceEnumAttr::get(
          func.getContext(), nkipy::MemSpaceEnum::Hbm);
      root.setType(MemRefType::get(rootType.getShape(),
                                   rootType.getElementType(),
                                   rootType.getLayout(), hbm));
      changed = true;
    });
    return changed;
  }

  /// Propagate memory spaces through view-like ops until convergence.
  /// Includes HBM inference for copy sources, which must interleave with
  /// propagation (view ops need their source memspace propagated first).
  void propagateMemSpaces(func::FuncOp func) {
    bool changed = true;
    while (changed) {
      changed = false;
      func.walk([&](Operation *op) {
        TypeSwitch<Operation *>(op)
          .Case<memref::SubViewOp>([&](auto op) {
            changed |= propagateMemSpace(op.getSource(), op.getResult());
            changed |= propagateMemSpace(op.getResult(), op.getSource());
          })
          .Case<memref::CollapseShapeOp, memref::ExpandShapeOp>([&](auto op) {
            changed |= propagateMemSpace(op.getSrc(), op.getResult());
            changed |= propagateMemSpace(op.getResult(), op.getSrc());
          })
          .Case<memref::ReshapeOp>([&](auto op) {
            changed |= propagateMemSpace(op.getSource(), op.getResult());
            changed |= propagateMemSpace(op.getResult(), op.getSource());
          })
          .Case<memref::CastOp>([&](auto op) {
            changed |= propagateMemSpace(op.getSource(), op.getResult());
          });
      });
      changed |= inferHbmForCopySources(func);
    }
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Processing function: " << func.getName() << "\n");

    annotateInputOutput(func);
    if (func.isDeclaration())
      return;

    applyAnnotations(func);
    resolveSubViewConflicts(func);
    propagateMemSpaces(func);
  }
};
} // namespace

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createAnnotateMemorySpacePass() {
  return std::make_unique<NkipyAnnotateMemorySpacePass>();
}
} // namespace nkipy
} // namespace mlir
