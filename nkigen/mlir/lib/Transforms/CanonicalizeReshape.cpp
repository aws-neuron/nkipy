//===- CanonicalizeReshape.cpp - Canonicalize memref reshape ops ----------===//
//
// Post-bufferization pass that canonicalizes reshape (expand_shape /
// collapse_shape) operations on memrefs.
//
// Runs after annotate-memory-space so all memrefs have explicit memory spaces
// (#nisa.mem<sbuf>, #nisa.mem<shared_hbm>, etc.) and partition_dim is
// guaranteed to be 0 (from canonicalize-partition-dim).
//
// Classification logic:
//
// +-------------------+--------+-----------+-----------------------------+
// | Reshape type      | MemSpc | Pdim(d0)? | Action                      |
// +-------------------+--------+-----------+-----------------------------+
// | Any               | HBM    | N/A       | View (no partition concept) |
// | Merge (collapse)  | SBUF   | any       | View (contiguous in memory) |
// | Split (expand)    | SBUF   | no        | View (partition unchanged)  |
// | Split (expand)    | SBUF   | yes       | Copy (needs modulo for      |
// |                   |        |           | partition reassignment)     |
// | Split N->(N,1)    | SBUF   | trivial   | View (no modulo needed)     |
// +-------------------+--------+-----------+-----------------------------+
//
// Additionally, returned expand_shape views of function arguments need
// alloc+copy to ensure separate HBM output allocations.
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using nkipy::isSbuf;
using nkipy::isAnyHbm;

namespace {

/// Trace through memref view ops to find the base memref.
static Value traceToBase(Value v) {
  while (auto defOp = v.getDefiningOp()) {
    if (auto op = dyn_cast<memref::CollapseShapeOp>(defOp))
      v = op.getSrc();
    else if (auto op = dyn_cast<memref::ExpandShapeOp>(defOp))
      v = op.getSrc();
    else if (auto op = dyn_cast<memref::CastOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::SubViewOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::ReinterpretCastOp>(defOp))
      v = op.getSource();
    else
      break;
  }
  return v;
}

/// Check if an expand_shape splits the partition dim (dim 0).
/// After canonicalize-partition-dim, partition dim is always 0.
/// A split of dim 0 means the reassociation group for src dim 0 maps to
/// multiple dst dims.
static bool splitsPartitionDim(memref::ExpandShapeOp expandOp) {
  auto reassoc = expandOp.getReassociationIndices();
  // Reassociation is indexed by src dims.  If src dim 0's group has
  // multiple dst dims, the partition dim is being split.
  if (reassoc.empty())
    return false;
  return reassoc[0].size() > 1;
}

/// Check if a partition dim split is trivial: N -> (N, 1) or (1, N).
/// Trivial splits don't need modulo because one of the factors is 1.
static bool isTrivialPartitionSplit(memref::ExpandShapeOp expandOp) {
  if (!splitsPartitionDim(expandOp))
    return false;

  auto resultType = cast<MemRefType>(expandOp.getResultType());
  auto reassoc = expandOp.getReassociationIndices();
  // Check the dims in the partition group
  for (int64_t dstIdx : reassoc[0]) {
    int64_t dimSize = resultType.getShape()[dstIdx];
    if (dimSize == 1)
      return true;  // One factor is 1, no modulo needed
  }
  return false;
}

struct CanonicalizeReshapePass
    : public PassWrapper<CanonicalizeReshapePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeReshapePass)

  StringRef getArgument() const final {
    return "canonicalize-reshape";
  }

  StringRef getDescription() const final {
    return "Canonicalize memref reshape ops based on mem_space and partition_dim";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // ---------------------------------------------------------------
    // Phase 0: Convert memref.reshape to memref.reinterpret_cast.
    //
    // Category 2 reshapes (no contiguous dim grouping) are emitted by the
    // tracer as tensor.reshape, which bufferizes to memref.reshape.
    // Since the data is contiguous in row-major order, we can replace with
    // a reinterpret_cast using contiguous (row-major) strides for the new
    // shape — a true zero-cost view reinterpretation.
    // ---------------------------------------------------------------
    SmallVector<memref::ReshapeOp> reshapeOps;
    func.walk([&](memref::ReshapeOp op) { reshapeOps.push_back(op); });

    for (auto reshapeOp : reshapeOps) {
      auto srcType = cast<MemRefType>(reshapeOp.getSource().getType());
      auto resultType = cast<MemRefType>(reshapeOp.getResult().getType());

      OpBuilder builder(reshapeOp);
      Location loc = reshapeOp.getLoc();

      // Decompose into collapse_shape(→1D) + expand_shape(→target).
      // Both are pure views on contiguous memory — no data movement.
      // FoldHbmReshapePattern in linalg-to-nisa handles these: the DMA
      // addresses the alloc with its original shape, while the return
      // reinterprets the same buffer via view() with the target shape.
      int64_t totalElems = 1;
      for (int64_t d : srcType.getShape())
        totalElems *= d;

      // Step 1: collapse source to 1D.
      auto flatType = MemRefType::get(
          {totalElems}, srcType.getElementType(),
          MemRefLayoutAttrInterface{}, srcType.getMemorySpace());
      SmallVector<ReassociationIndices> collapseReassoc = {{}};
      for (int64_t i = 0; i < srcType.getRank(); ++i)
        collapseReassoc[0].push_back(i);
      auto collapseOp = builder.create<memref::CollapseShapeOp>(
          loc, flatType, reshapeOp.getSource(), collapseReassoc);

      // Step 2: expand 1D to target shape.
      auto expandType = MemRefType::get(
          resultType.getShape(), resultType.getElementType(),
          MemRefLayoutAttrInterface{}, resultType.getMemorySpace());
      SmallVector<ReassociationIndices> expandReassoc = {{}};
      for (int64_t i = 0; i < resultType.getRank(); ++i)
        expandReassoc[0].push_back(i);
      auto expandOp = builder.create<memref::ExpandShapeOp>(
          loc, expandType, collapseOp.getResult(), expandReassoc);

      reshapeOp.getResult().replaceAllUsesWith(expandOp.getResult());
      reshapeOp->erase();

      llvm::errs() << "[CanonicalizeReshape] Category 2 reshape: "
                   << "decomposed to collapse(1D)+expand(target)\n";
    }

    // ---------------------------------------------------------------
    // Phase 1: Classify expand_shape ops and insert copies where needed.
    //
    // collapse_shape (merge) is always a view -- merging dims never needs
    // modulo regardless of mem_space or partition_dim involvement.
    //
    // expand_shape (split) needs analysis:
    //   - HBM: always view (no partition concept)
    //   - SBUF, non-partition split: view
    //   - SBUF, partition dim split: copy (NISA has no modulo)
    //     Exception: trivial split N->(N,1)/(1,N) is view
    // ---------------------------------------------------------------
    SmallVector<memref::ExpandShapeOp> expandOps;
    func.walk([&](memref::ExpandShapeOp op) { expandOps.push_back(op); });

    for (auto expandOp : expandOps) {
      auto srcType = cast<MemRefType>(expandOp.getSrc().getType());

      // HBM: always a view, no partition concept
      if (isAnyHbm(srcType) || !isSbuf(srcType))
        continue;

      // SBUF: check if partition dim (dim 0) is being split
      if (!splitsPartitionDim(expandOp))
        continue;

      // Trivial split N->(N,1) or (1,N): no modulo needed, keep as view
      if (isTrivialPartitionSplit(expandOp))
        continue;

      // SBUF partition dim split that needs modulo -- NISA can't do this
      // as a view.  Insert alloc + copy to materialize the reshape.
      //
      // TODO: This should ideally be a tiled copy loop that explicitly
      // computes the partition reassignment.  For now, emit alloc + copy
      // and let downstream passes handle the lowering.
      OpBuilder builder(expandOp);
      Location loc = expandOp.getLoc();

      auto resultType = cast<MemRefType>(expandOp.getResultType());
      auto allocOp = builder.create<memref::AllocOp>(loc, resultType);

      // Copy from the expanded view into the fresh allocation.
      // The expand_shape itself is a valid view for memref.copy's purposes.
      builder.create<memref::CopyOp>(
          loc, expandOp.getResult(), allocOp.getResult());

      expandOp.getResult().replaceAllUsesExcept(
          allocOp.getResult(), allocOp->getNextNode());

      llvm::errs() << "[CanonicalizeReshape] SBUF partition dim split: "
                   << "inserted alloc+copy for expand_shape\n";
    }

    // ---------------------------------------------------------------
    // Phase 2: Returned view ops of function arguments.
    //
    // After bufferization, view ops (expand_shape, reinterpret_cast)
    // produce no allocation, but NISA requires function outputs to be
    // separate HBM allocations.  Insert alloc + copy to materialize.
    //
    // collapse_shape already gets alloc+copy from bufferization.
    // ---------------------------------------------------------------
    func.walk([&](func::ReturnOp returnOp) {
      for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        Value retVal = returnOp.getOperand(i);
        if (!isa<MemRefType>(retVal.getType()))
          continue;

        Value base = traceToBase(retVal);
        if (!isa<BlockArgument>(base))
          continue;
        if (retVal == base)
          continue;

        Operation *defOp = retVal.getDefiningOp();
        if (!defOp)
          continue;

        // Handle expand_shape views.
        if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(defOp)) {
          Value expandSrc = expandOp.getSrc();
          if (!isa<BlockArgument>(traceToBase(expandSrc)))
            continue;

          // Check if this expand_shape already has a non-arg source
          // (Phase 1 may have already inserted a copy)
          if (expandSrc.getDefiningOp<memref::AllocOp>())
            continue;

          auto srcType = cast<MemRefType>(expandSrc.getType());
          auto oldResultType = cast<MemRefType>(expandOp.getResultType());
          auto contigResultType = MemRefType::get(
              oldResultType.getShape(), oldResultType.getElementType(),
              MemRefLayoutAttrInterface{}, oldResultType.getMemorySpace());
          Location loc = expandOp.getLoc();

          if (srcType.getRank() < 2) {
            // Source is 1D: NISA DMA requires at least 2D tiles.
            // Alloc with the expanded (2D) shape and copy from the
            // expand_shape view to avoid a 1D DMA.
            OpBuilder builder(expandOp->getNextNode());

            auto allocOp = builder.create<memref::AllocOp>(
                loc, contigResultType);
            auto copyOp = builder.create<memref::CopyOp>(
                loc, expandOp.getResult(), allocOp.getResult());

            expandOp.getResult().replaceAllUsesExcept(
                allocOp.getResult(), {copyOp});
          } else {
            // Source is 2D+: alloc+copy before expand, then rebuild expand.
            auto contigSrcType = MemRefType::get(
                srcType.getShape(), srcType.getElementType(),
                MemRefLayoutAttrInterface{}, srcType.getMemorySpace());
            OpBuilder builder(expandOp);

            auto allocOp = builder.create<memref::AllocOp>(
                loc, contigSrcType);
            builder.create<memref::CopyOp>(
                loc, expandSrc, allocOp.getResult());

            auto newExpandOp = builder.create<memref::ExpandShapeOp>(
                loc, contigResultType, allocOp.getResult(),
                expandOp.getReassociationIndices(),
                expandOp.getMixedOutputShape());

            expandOp.getResult().replaceAllUsesWith(newExpandOp.getResult());
            expandOp->erase();
          }

          // Update function signature return type.
          auto funcType = func.getFunctionType();
          SmallVector<Type> newResultTypes(funcType.getResults());
          newResultTypes[i] = contigResultType;
          func.setFunctionType(FunctionType::get(
              func.getContext(), funcType.getInputs(), newResultTypes));

          llvm::errs() << "[CanonicalizeReshape] Returned view of func arg: "
                       << "inserted alloc+copy for expand_shape\n";
          continue;
        }

      }
    });

    // ---------------------------------------------------------------
    // Phase 3: Direct return of function arguments (identity reshape).
    //
    // When a function argument is returned directly (no reshape at all),
    // there's no allocation for the output.  NISA requires function outputs
    // to be separate HBM allocations.  Insert alloc + copy.
    //
    // Example: np.reshape(x, x.shape) -> identity -> return %arg0
    // ---------------------------------------------------------------
    func.walk([&](func::ReturnOp returnOp) {
      for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        Value retVal = returnOp.getOperand(i);
        if (!isa<MemRefType>(retVal.getType()))
          continue;

        // Only handle direct return of block arguments
        if (!isa<BlockArgument>(retVal))
          continue;

        auto retType = cast<MemRefType>(retVal.getType());
        auto contigType = MemRefType::get(
            retType.getShape(), retType.getElementType(),
            MemRefLayoutAttrInterface{}, retType.getMemorySpace());
        OpBuilder builder(returnOp);
        Location loc = returnOp.getLoc();

        auto allocOp = builder.create<memref::AllocOp>(loc, contigType);
        builder.create<memref::CopyOp>(loc, retVal, allocOp.getResult());
        returnOp.setOperand(i, allocOp.getResult());

        // Update function signature to match contiguous type
        auto funcType = func.getFunctionType();
        SmallVector<Type> newResultTypes(funcType.getResults());
        newResultTypes[i] = contigType;
        func.setFunctionType(FunctionType::get(
            func.getContext(), funcType.getInputs(), newResultTypes));

        llvm::errs() << "[CanonicalizeReshape] Direct return of func arg: "
                     << "inserted alloc+copy\n";
      }
    });
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<func::FuncOp>>
createCanonicalizeReshapePass() {
  return std::make_unique<CanonicalizeReshapePass>();
}

} // namespace nkipy
} // namespace mlir
