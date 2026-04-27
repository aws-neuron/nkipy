#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "eliminate-same-memspace-copy"

using namespace mlir;
using nkipy::getBaseMemRef;
using nkipy::getNkipyMemSpace;
using nkipy::isAnyHbm;

namespace mlir {
namespace nkipy {

namespace {

/// Check if a copy is between the same memory space (e.g., SBUF→SBUF).
static bool isSameMemSpaceCopy(memref::CopyOp copyOp) {
  auto srcMemSpace = getNkipyMemSpace(copyOp.getSource().getType());
  auto dstMemSpace = getNkipyMemSpace(copyOp.getTarget().getType());
  if (!srcMemSpace || !dstMemSpace)
    return false;
  return *srcMemSpace == *dstMemSpace;
}

/// Check if two subview operations access the exact same memory region.
/// They must have the same base, offsets, sizes, and strides.
static bool isSameRegion(memref::SubViewOp a, memref::SubViewOp b) {
  return getBaseMemRef(a.getSource()) == getBaseMemRef(b.getSource()) &&
         a.getMixedOffsets() == b.getMixedOffsets() &&
         a.getMixedSizes() == b.getMixedSizes() &&
         a.getMixedStrides() == b.getMixedStrides();
}

/// Check if src and dst of a copy point to the exact same memory region.
/// If so, the copy is a no-op and can be eliminated.
static bool isCopyToSelf(memref::CopyOp copyOp) {
  Value src = copyOp.getSource();
  Value dst = copyOp.getTarget();
  
  // Trivial case: same SSA value
  if (src == dst)
    return true;
  
  // Check if both are subviews of the same base with same parameters
  auto srcSubview = src.getDefiningOp<memref::SubViewOp>();
  auto dstSubview = dst.getDefiningOp<memref::SubViewOp>();
  
  if (srcSubview && dstSubview)
    return isSameRegion(srcSubview, dstSubview);
  
  return false;
}

/// Pattern to eliminate self-copies (where src and dst point to same region).
/// These are no-ops and can be erased directly.
struct EliminateSelfCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (!isCopyToSelf(copyOp))
      return failure();
    
    LLVM_DEBUG(llvm::dbgs() << "Eliminating self-copy: " << copyOp << "\n");
    rewriter.eraseOp(copyOp);
    return success();
  }
};

/// Pattern to eliminate same-memory-space copies where dst is a fresh alloc.
/// Transforms:
///   %alloc = memref.alloc() : memref<...xf32, #sbuf>
///   memref.copy %src, %alloc : memref<...xf32, #sbuf> to memref<...xf32, #sbuf>
///   use(%alloc)
/// Into:
///   use(%src)
struct EliminateAllocCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // Only eliminate same memory space copies
    if (!isSameMemSpaceCopy(copyOp))
      return failure();
    
    Value src = copyOp.getSource();
    Value dst = copyOp.getTarget();
    
    // Only eliminate SBUF→SBUF copies (not PSUM→PSUM which doesn't make sense)
    auto srcMemSpace = getNkipyMemSpace(src.getType());
    if (*srcMemSpace != nkipy::MemSpaceEnum::Sbuf)
      return failure();
    
    // Destination must be a fresh allocation (not written to before this copy)
    auto allocOp = dst.getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();
    
    // Check that the allocation hasn't been written to before this copy
    // by ensuring no user of allocOp in the same block comes before copyOp.
    // Users in nested regions (e.g., loop bodies) execute after the copy,
    // so they don't count as "before".
    // Verify the alloc hasn't been written to before this copy.
    Block *copyBlock = copyOp->getBlock();
    for (Operation *user : allocOp->getUsers()) {
      if (user == copyOp.getOperation())
        continue;
      if (user->getBlock() != copyBlock)
        continue; // nested regions execute after the copy
      if (user->isBeforeInBlock(copyOp.getOperation()))
        return failure(); // another use before the copy — bail out
    }
    
    // Check types are compatible (shapes should match for direct replacement)
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());
    
    if (srcType.getShape() != dstType.getShape())
      return failure();
    
    if (srcType.getElementType() != dstType.getElementType())
      return failure();
    
    LLVM_DEBUG(llvm::dbgs() << "Eliminating SBUF copy: " << copyOp << "\n");
    rewriter.replaceAllUsesWith(dst, src);
    
    // Erase the copy operation
    rewriter.eraseOp(copyOp);
    
    // Erase the now-unused allocation
    if (allocOp->use_empty())
      rewriter.eraseOp(allocOp);
    
    return success();
  }
};

/// Helper: convert an OpFoldResult to a Value, materializing constants.
static Value materialize(OpFoldResult ofr, Location loc,
                         PatternRewriter &rewriter) {
  if (auto attr = dyn_cast<Attribute>(ofr))
    return rewriter.create<arith::ConstantIndexOp>(
        loc, cast<IntegerAttr>(attr).getInt());
  return cast<Value>(ofr);
}

/// Helper: check if an OpFoldResult is a static zero.
static bool isStaticZero(OpFoldResult ofr) {
  if (auto attr = dyn_cast<Attribute>(ofr))
    return cast<IntegerAttr>(attr).getInt() == 0;
  return false;
}

/// Check if defOp properly dominates insertionPt.
/// A value dominates if it's defined before the insertion point in the same
/// block, or if it's defined before the ancestor op in an ancestor block.
static bool properlyDominates(Operation *defOp, Operation *insertionPt) {
  Block *insertBlock = insertionPt->getBlock();
  // Same block: defOp must come before insertionPt
  if (defOp->getBlock() == insertBlock)
    return defOp->isBeforeInBlock(insertionPt);
  // Walk up from insertion point to find if defOp is in an ancestor block,
  // and if so, check it comes before the child region's parent op.
  for (Operation *ancestor = insertBlock->getParentOp(); ancestor;
       ancestor = ancestor->getBlock()
                      ? ancestor->getBlock()->getParentOp()
                      : nullptr) {
    if (defOp->getBlock() == ancestor->getBlock())
      return defOp->isBeforeInBlock(ancestor);
  }
  return false;
}

/// Ensure an OpFoldResult is usable at the given insertion point.
/// If it's a static attribute, just return it. If it's a Value whose defining
/// op doesn't dominate the insertion point, clone the defining op there.
static OpFoldResult ensureDominates(OpFoldResult ofr, Location loc,
                                    PatternRewriter &rewriter) {
  if (isa<Attribute>(ofr))
    return ofr;
  Value v = cast<Value>(ofr);
  Operation *insertionPt = rewriter.getInsertionBlock()
                               ? &*rewriter.getInsertionPoint()
                               : nullptr;
  if (!insertionPt)
    return ofr;
  // Block arguments dominate their block and all nested blocks
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    Block *argBlock = blockArg.getOwner();
    // Check the arg's block is the same or an ancestor of insertionPt
    for (Block *b = insertionPt->getBlock(); b;
         b = b->getParentOp() ? b->getParentOp()->getBlock() : nullptr) {
      if (b == argBlock)
        return ofr;
    }
    // Block arg doesn't dominate - fall through to clone logic
    // (shouldn't normally happen)
  }
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return ofr;
  if (properlyDominates(defOp, insertionPt))
    return ofr;
  // defOp doesn't dominate - clone it at the insertion point.
  // First ensure its operands dominate too.
  IRMapping mapping;
  for (Value operand : defOp->getOperands()) {
    OpFoldResult ensured = ensureDominates(operand, loc, rewriter);
    if (auto ensuredVal = dyn_cast<Value>(ensured)) {
      if (ensuredVal != operand)
        mapping.map(operand, ensuredVal);
    }
  }
  Operation *cloned = rewriter.clone(*defOp, mapping);
  return cloned->getResult(0);
}

/// Pattern to eliminate intermediate SharedHBM allocations that are written
/// to via tiled copies and then bulk-copied to the output via reshape+copy.
///
/// Matches:
///   %tmp = memref.alloc() : memref<MxNxf32, shared_hbm>
///   memref.copy %tile, subview(%tmp, [row, col])       // tiled SBUF→HBM writes
///   %reshape = memref.reshape %tmp(...) -> memref<1xMxNxf32>
///   memref.copy %reshape, subview(%out, [batch, 0, 0]) // HBM→HBM bulk copy
///
/// Transforms to:
///   memref.copy %tile, subview(%out, [batch, row, col]) // write directly to output
///
struct EliminateHbmIntermediatePattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // 1. Source must come from a memref.reshape or memref.expand_shape
    //    (expand_shape is produced when np.expand_dims emits
    //    tensor.expand_shape instead of tensor.reshape)
    Operation *expandOp = nullptr;
    Value intermediateAllocResult;

    if (auto reshapeOp = copyOp.getSource().getDefiningOp<memref::ReshapeOp>()) {
      expandOp = reshapeOp;
      intermediateAllocResult = reshapeOp.getSource();
    } else if (auto expandShapeOp =
                   copyOp.getSource().getDefiningOp<memref::ExpandShapeOp>()) {
      expandOp = expandShapeOp;
      intermediateAllocResult = expandShapeOp.getSrc();
    } else {
      return failure();
    }

    // 2. Reshape/expand source must be a fresh alloc in HBM or SharedHBM
    auto intermediateAlloc =
        intermediateAllocResult.getDefiningOp<memref::AllocOp>();
    if (!intermediateAlloc)
      return failure();

    if (!isAnyHbm(intermediateAlloc.getResult().getType()))
      return failure();

    // 3. Destination must be a subview of an HBM or SharedHBM buffer
    auto dstSubview = copyOp.getTarget().getDefiningOp<memref::SubViewOp>();
    if (!dstSubview)
      return failure();

    Value outputBase = dstSubview.getSource();
    if (!isAnyHbm(outputBase.getType()))
      return failure();

    // 4. Verify reshape/expand is a trivial leading-1 expansion (MxN → 1xMxN)
    auto intermediateType =
        cast<MemRefType>(intermediateAlloc.getResult().getType());
    auto expandResultType = cast<MemRefType>(expandOp->getResult(0).getType());
    auto intermediateShape = intermediateType.getShape();
    auto expandShape = expandResultType.getShape();
    if (expandShape.size() != intermediateShape.size() + 1)
      return failure();
    if (expandShape[0] != 1)
      return failure();
    for (unsigned i = 0; i < intermediateShape.size(); ++i) {
      if (expandShape[i + 1] != intermediateShape[i])
        return failure();
    }

    // 5. Reshape/expand must have exactly one use (this copy)
    if (!expandOp->getResult(0).hasOneUse())
      return failure();

    // 6. Collect all subviews of the intermediate alloc.
    //    Only subviews and the reshape/expand are allowed as users.
    SmallVector<memref::SubViewOp> intermediateSubviews;
    for (Operation *user : intermediateAlloc->getUsers()) {
      if (user == expandOp)
        continue;
      auto subview = dyn_cast<memref::SubViewOp>(user);
      if (!subview)
        return failure();
      intermediateSubviews.push_back(subview);
    }

    // 7. Get batch offset and other offsets from the dst subview
    SmallVector<OpFoldResult> dstOffsets = dstSubview.getMixedOffsets();
    auto outputBaseType = cast<MemRefType>(outputBase.getType());

    LLVM_DEBUG(llvm::dbgs() << "Eliminating HBM intermediate: "
                           << *intermediateAlloc << "\n");

    // 8. For each subview of the intermediate, create a new rank-reducing
    //    subview of the output base with the batch offset prepended.
    for (auto subview : intermediateSubviews) {
      SmallVector<OpFoldResult> tileOffsets = subview.getMixedOffsets();
      SmallVector<OpFoldResult> tileSizes = subview.getMixedSizes();
      SmallVector<OpFoldResult> tileStrides = subview.getMixedStrides();
      Location loc = subview.getLoc();

      // New offsets: [batch, tile_row + dst_row, tile_col + dst_col, ...]
      // Use ensureDominates because dstOffsets come from the dstSubview which
      // is defined after the nested loops, but subview is inside the loops.
      rewriter.setInsertionPoint(subview);
      SmallVector<OpFoldResult> newOffsets;
      newOffsets.push_back(ensureDominates(dstOffsets[0], loc, rewriter));
      for (unsigned i = 0; i < tileOffsets.size(); ++i) {
        unsigned dstIdx = i + 1; // skip batch dim in dst
        if (dstIdx < dstOffsets.size() && !isStaticZero(dstOffsets[dstIdx])) {
          // Non-zero dst offset: add to tile offset
          Value sum = rewriter.create<arith::AddIOp>(
              loc, materialize(tileOffsets[i], loc, rewriter),
              materialize(ensureDominates(dstOffsets[dstIdx], loc, rewriter),
                          loc, rewriter));
          newOffsets.push_back(sum);
        } else {
          newOffsets.push_back(tileOffsets[i]);
        }
      }

      // New sizes: [1, tile_sizes...]
      SmallVector<OpFoldResult> newSizes;
      newSizes.push_back(rewriter.getIndexAttr(1));
      for (auto sz : tileSizes)
        newSizes.push_back(sz);

      // New strides: [1, tile_strides...]
      SmallVector<OpFoldResult> newStrides;
      newStrides.push_back(rewriter.getIndexAttr(1));
      for (auto st : tileStrides)
        newStrides.push_back(st);

      // Infer rank-reduced result type (drops the leading size-1 batch dim)
      auto originalResultType = cast<MemRefType>(subview.getResult().getType());
      auto reducedType = memref::SubViewOp::inferRankReducedResultType(
          originalResultType.getShape(), outputBaseType, newOffsets, newSizes,
          newStrides);

      auto newSubview = rewriter.create<memref::SubViewOp>(
          loc, reducedType, outputBase, newOffsets, newSizes, newStrides);

      rewriter.replaceOp(subview, newSubview.getResult());
    }

    // 9. Erase the HBM→HBM copy
    rewriter.eraseOp(copyOp);

    // 10. Clean up dead ops
    if (expandOp->use_empty())
      rewriter.eraseOp(expandOp);
    if (intermediateAlloc->use_empty())
      rewriter.eraseOp(intermediateAlloc);

    // dstSubview will be cleaned up by canonicalize (now has no users)

    return success();
  }
};

struct EliminateSameMemSpaceCopyPass
    : public EliminateSameMemSpaceCopyBase<EliminateSameMemSpaceCopyPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<EliminateSelfCopyPattern>(&getContext());
    patterns.add<EliminateAllocCopyPattern>(&getContext());
    patterns.add<EliminateHbmIntermediatePattern>(&getContext());

    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createEliminateSameMemSpaceCopyPass() {
  return std::make_unique<EliminateSameMemSpaceCopyPass>();
}

} // namespace nkipy
} // namespace mlir
