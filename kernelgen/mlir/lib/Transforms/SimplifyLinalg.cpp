//===- SimplifyLinalg.cpp - Simplify linalg ops for NISA lowering ---------===//
//
// Pre-processing pass that simplifies linalg operations before linalg-to-nisa.
//
// Transformations:
// 1. Rewrites >2D SBUF linalg.transpose with unit dims to 2D.
//    NISA dma_transpose only supports [1,0] (2D) or [2,1,0] (3D full reverse).
//    Collapses SBUF allocs to 2D + expand_shape views, rewrites transpose or
//    emits copy (when non-unit dims keep order, i.e. just a reshape).
//
// 2. Converts trivial-broadcast linalg.generic ops to named linalg ops.
//    After tiling, broadcasts become same-shape operations. This converts
//    them to named ops (linalg.mul, etc.) so LinalgToNisa patterns can match.
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "nkipy/Dialect/NkipyOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

using nkipy::isHbm;
using nkipy::isSbuf;

/// Elementwise arith kinds recognized in linalg.generic bodies.  Kept local
/// to this file so we do not depend on nki::nisa::ArithOpKind.
enum class LocalArithKind {
  ADD, SUBTRACT, MULTIPLY, DIVIDE, MOD, MODINT,
  ISEQ, ISGT, ISGE, ISLE, ISLT, ISNE,
};

/// Map arith dialect binary op to a local arith kind used to classify the
/// body of linalg.generic ops.
static std::optional<LocalArithKind>
getArithOpKindFromBodyOp(Operation *op) {
  auto kind = llvm::TypeSwitch<Operation *, std::optional<LocalArithKind>>(op)
      .Case<arith::AddFOp, arith::AddIOp>(
          [](auto) { return LocalArithKind::ADD; })
      .Case<arith::SubFOp, arith::SubIOp>(
          [](auto) { return LocalArithKind::SUBTRACT; })
      .Case<arith::MulFOp, arith::MulIOp>(
          [](auto) { return LocalArithKind::MULTIPLY; })
      .Case<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(
          [](auto) { return LocalArithKind::DIVIDE; })
      .Case<arith::RemFOp>([](auto) { return LocalArithKind::MOD; })
      .Case<arith::RemSIOp>([](auto) { return LocalArithKind::MODINT; })
      .Default([](Operation *) { return std::nullopt; });
  if (kind)
    return kind;

  // Comparison pattern: arith.uitofp(arith.cmpf(...))
  if (auto uitofp = dyn_cast<arith::UIToFPOp>(op)) {
    if (auto cmpf = uitofp.getIn().getDefiningOp<arith::CmpFOp>()) {
      switch (cmpf.getPredicate()) {
      case arith::CmpFPredicate::OEQ: return LocalArithKind::ISEQ;
      case arith::CmpFPredicate::OGT: return LocalArithKind::ISGT;
      case arith::CmpFPredicate::OGE: return LocalArithKind::ISGE;
      case arith::CmpFPredicate::OLE: return LocalArithKind::ISLE;
      case arith::CmpFPredicate::OLT: return LocalArithKind::ISLT;
      case arith::CmpFPredicate::ONE: return LocalArithKind::ISNE;
      default: return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Preprocessing: Decompose high-rank transposes to loops of 2D transposes
//===----------------------------------------------------------------------===//

/// Decompose N-D linalg.transpose where exactly 2 dimensions are swapped
/// (and the rest are identity) into a loop nest over the identity dims
/// with a 2D transpose (or copy) on the swapped pair.
///
/// Example: [0, 2, 1, 3] on memref<2x128x2x128> (multi-head attention reshape)
///   → scf.for batch = 0..2:
///       scf.for head = 0..2:
///         linalg.transpose [1, 0] on memref<128x128> tiles
///
/// This handles transposes across any memory space (HBM, SharedHBM, SBUF).
static void decomposeHighRankTranspose(func::FuncOp func) {
  SmallVector<linalg::TransposeOp> worklist;
  func.walk([&](linalg::TransposeOp op) { worklist.push_back(op); });

  for (auto transposeOp : worklist) {
    auto srcType = dyn_cast<MemRefType>(transposeOp.getInput().getType());
    auto dstType = dyn_cast<MemRefType>(transposeOp.getInit().getType());
    if (!srcType || !dstType)
      continue;

    int64_t rank = srcType.getRank();
    // Only handle rank > 3.  Rank-2 is already native NISA.
    // Rank-3 with unit dims is handled by rewriteSbufTransposeTo2D.
    if (rank <= 3)
      continue;

    // Find which dimensions are swapped vs identity.
    auto perm = transposeOp.getPermutation();
    SmallVector<int64_t> swappedDims;   // dims where perm[i] != i
    SmallVector<int64_t> identityDims;  // dims where perm[i] == i
    for (int64_t i = 0; i < rank; i++) {
      if (perm[i] != i)
        swappedDims.push_back(i);
      else
        identityDims.push_back(i);
    }

    // Only handle the case where exactly 2 dims are swapped.
    if (swappedDims.size() != 2)
      continue;

    int64_t d0 = swappedDims[0];  // first swapped dim
    int64_t d1 = swappedDims[1];  // second swapped dim
    // Verify they are actually swapped with each other
    if (perm[d0] != d1 || perm[d1] != d0)
      continue;

    auto srcShape = srcType.getShape();
    auto dstShape = dstType.getShape();

    OpBuilder b(transposeOp);
    Location loc = transposeOp.getLoc();

    // Helper: collapse a subview slice to 2D by grouping unit dims with
    // their neighbors.  dimA is the position of the first non-unit dim;
    // everything up to and including dimA goes in group 0.
    auto collapse2D = [&](Value slice, int64_t dimA, int64_t dimB,
                          MemRefType origType) -> Value {
      auto sliceType = cast<MemRefType>(slice.getType());
      if (sliceType.getRank() == 2)
        return slice;

      SmallVector<ReassociationIndices> reassoc;
      ReassociationIndices group0, group1;
      bool seenFirst = false;
      for (int64_t i = 0; i < (int64_t)sliceType.getRank(); i++) {
        if (!seenFirst || i <= dimA) {
          group0.push_back(i);
          if (sliceType.getShape()[i] > 1) seenFirst = true;
        } else {
          group1.push_back(i);
        }
      }
      if (group1.empty())
        return slice;
      reassoc = {group0, group1};
      return b.create<memref::CollapseShapeOp>(loc, slice, reassoc);
    };

    // ----------------------------------------------------------------
    // Optimization: when one swapped dim is small (e.g. n_heads=2),
    // iterate over it in the loop nest and emit a 2D *copy* instead of
    // a 2D transpose.  This avoids collapse_shape tiles whose non-unit
    // strides are lost by getBaseAndOffsets in linalg-to-nisa, causing
    // wrong column interleaving in NISA DMA access patterns.
    //
    // Example: perm [0,2,1,3] on (2,128,2,128)
    //   Before: loop batch(2) × hd(128) = 256 iters, inner (128,2) transpose
    //   After:  loop batch(2) × head(2) = 4 iters, inner (128,128) copy
    // ----------------------------------------------------------------
    constexpr int64_t kSmallSwapDimThreshold = 16;
    int64_t smallSwapDim = -1, largeSwapDim = -1;
    {
      int64_t s0 = srcShape[d0], s1 = srcShape[d1];
      if (s0 <= kSmallSwapDimThreshold && s0 < s1) {
        smallSwapDim = d0; largeSwapDim = d1;
      } else if (s1 <= kSmallSwapDimThreshold && s1 < s0) {
        smallSwapDim = d1; largeSwapDim = d0;
      }
    }

    // Find the largest identity dim — kept in the inner tile for efficiency.
    int64_t keepIdDim = -1;
    int64_t keepIdSize = 0;
    for (int64_t idim : identityDims) {
      if (srcShape[idim] > keepIdSize) {
        keepIdSize = srcShape[idim];
        keepIdDim = idim;
      }
    }

    // Use the optimized copy path when a small swapped dim exists and
    // there is a large identity dim to pair with the large swapped dim
    // for the 2D inner tile.
    bool useSmallSwapOpt = (smallSwapDim >= 0 && keepIdDim >= 0 &&
                            keepIdSize > 1);

    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

    if (useSmallSwapOpt) {
      // Inverse permutation: invPerm[srcDim] = dstDim
      SmallVector<int64_t> invPerm(rank);
      for (int64_t j = 0; j < rank; j++)
        invPerm[perm[j]] = j;

      // Build loop nest over:
      //   - identity dims (except keepIdDim)
      //   - small swapped dim (offset goes at different positions in src vs dst)
      struct LoopInfo { int64_t srcDim; int64_t dstDim; };
      SmallVector<LoopInfo> loopInfos;
      SmallVector<Value> loopIVs;
      auto addLoop = [&](int64_t srcDim, int64_t dstDim, int64_t size) {
        Value ub = b.create<arith::ConstantIndexOp>(loc, size);
        auto loop = b.create<scf::ForOp>(loc, c0, ub, c1);
        b.setInsertionPointToStart(loop.getBody());
        loopInfos.push_back({srcDim, dstDim});
        loopIVs.push_back(loop.getInductionVar());
      };

      for (int64_t idim : identityDims) {
        if (idim == keepIdDim) continue;
        addLoop(idim, idim, srcShape[idim]);
      }
      addLoop(smallSwapDim, invPerm[smallSwapDim], srcShape[smallSwapDim]);

      // Build src subview
      SmallVector<OpFoldResult> srcOffsets(rank, b.getIndexAttr(0));
      SmallVector<OpFoldResult> srcSizes;
      SmallVector<OpFoldResult> srcStrides(rank, b.getIndexAttr(1));
      for (int64_t i = 0; i < rank; i++) {
        bool inLoop = false;
        for (size_t li = 0; li < loopInfos.size(); li++) {
          if (loopInfos[li].srcDim == i) {
            srcOffsets[i] = OpFoldResult(loopIVs[li]);
            srcSizes.push_back(b.getIndexAttr(1));
            inLoop = true; break;
          }
        }
        if (!inLoop)
          srcSizes.push_back(b.getIndexAttr(srcShape[i]));
      }
      auto srcSlice = b.create<memref::SubViewOp>(
          loc, transposeOp.getInput(), srcOffsets, srcSizes, srcStrides);

      // Build dst subview (loop IVs placed at dstDim positions)
      SmallVector<OpFoldResult> dstOffsets(rank, b.getIndexAttr(0));
      SmallVector<OpFoldResult> dstSizes;
      SmallVector<OpFoldResult> dstStrides(rank, b.getIndexAttr(1));
      for (int64_t i = 0; i < rank; i++) {
        bool inLoop = false;
        for (size_t li = 0; li < loopInfos.size(); li++) {
          if (loopInfos[li].dstDim == i) {
            dstOffsets[i] = OpFoldResult(loopIVs[li]);
            dstSizes.push_back(b.getIndexAttr(1));
            inLoop = true; break;
          }
        }
        if (!inLoop)
          dstSizes.push_back(b.getIndexAttr(dstShape[i]));
      }
      auto dstSlice = b.create<memref::SubViewOp>(
          loc, transposeOp.getInit(), dstOffsets, dstSizes, dstStrides);

      // Collapse to 2D: the inner tile has two non-unit dims
      // (largeSwapDim and keepIdDim in src; their permuted positions in dst).
      int64_t srcDimA = std::min(largeSwapDim, keepIdDim);
      int64_t dstDimA = std::min(invPerm[largeSwapDim], keepIdDim);
      Value src2d = collapse2D(srcSlice, srcDimA, -1, srcType);
      Value dst2d = collapse2D(dstSlice, dstDimA, -1, dstType);

      // Emit copy — no transpose needed since the two inner dims are in
      // the same relative order in src and dst.
      auto src2dType = cast<MemRefType>(src2d.getType());
      auto dst2dType = cast<MemRefType>(dst2d.getType());
      if (isSbuf(src2dType) || isSbuf(dst2dType)) {
        b.create<memref::CopyOp>(loc, src2d, dst2d);
      } else {
        // HBM→HBM: route through SBUF temp.
        auto sbufMemSpace = nkipy::MemSpaceEnumAttr::get(
            b.getContext(), nkipy::MemSpaceEnum::Sbuf);
        auto shape2d = src2dType.getShape();
        auto sbufType = MemRefType::get(
            shape2d, src2dType.getElementType(), nullptr, sbufMemSpace);
        auto sbufTemp = b.create<memref::AllocOp>(loc, sbufType);
        b.create<memref::CopyOp>(loc, src2d, sbufTemp);
        b.create<memref::CopyOp>(loc, sbufTemp, dst2d);
      }

      transposeOp.erase();
      LLVM_DEBUG(llvm::dbgs() << "[SimplifyLinalg] Decomposed " << rank
                   << "D transpose to loop of 2D copies"
                      " (small-swap-dim opt)\n");
      continue;
    }

    // ----------------------------------------------------------------
    // Original path: loop over all identity dims, transpose inner 2D.
    // ----------------------------------------------------------------
    SmallVector<Value> ivs;
    for (int64_t idim : identityDims) {
      Value ub = b.create<arith::ConstantIndexOp>(loc, srcShape[idim]);
      auto loop = b.create<scf::ForOp>(loc, c0, ub, c1);
      b.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
    }

    // Build subview for src: take a 2D slice at the swapped dims
    SmallVector<OpFoldResult> srcOffsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> srcSizes;
    SmallVector<OpFoldResult> srcStrides(rank, b.getIndexAttr(1));
    unsigned ivIdx = 0;
    for (int64_t i = 0; i < rank; i++) {
      if (perm[i] == i) {
        srcOffsets[i] = OpFoldResult(ivs[ivIdx++]);
        srcSizes.push_back(b.getIndexAttr(1));
      } else {
        srcSizes.push_back(b.getIndexAttr(srcShape[i]));
      }
    }
    auto srcSlice = b.create<memref::SubViewOp>(
        loc, transposeOp.getInput(), srcOffsets, srcSizes, srcStrides);

    // Build subview for dst: same loop IVs but at the permuted positions
    SmallVector<OpFoldResult> dstOffsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> dstSizes;
    SmallVector<OpFoldResult> dstStrides(rank, b.getIndexAttr(1));
    ivIdx = 0;
    for (int64_t i = 0; i < rank; i++) {
      if (perm[i] == i) {
        dstOffsets[i] = OpFoldResult(ivs[ivIdx++]);
        dstSizes.push_back(b.getIndexAttr(1));
      } else {
        dstSizes.push_back(b.getIndexAttr(dstShape[i]));
      }
    }
    auto dstSlice = b.create<memref::SubViewOp>(
        loc, transposeOp.getInit(), dstOffsets, dstSizes, dstStrides);

    Value src2d = collapse2D(srcSlice, d0, d1, srcType);
    Value dst2d = collapse2D(dstSlice, d0, d1, dstType);

    // The 2D transpose is [1, 0] since the two swapped dims are now the
    // only dims.
    //
    // For SBUF-involved transposes: emit linalg.transpose directly.
    // For HBM-only transposes: route through an SBUF temp since there's no
    // HBM→HBM transpose in hardware.  Pattern: load → transpose in SBUF → store.
    auto src2dType = cast<MemRefType>(src2d.getType());
    auto dst2dType = cast<MemRefType>(dst2d.getType());
    if (isSbuf(src2dType) || isSbuf(dst2dType)) {
      auto newOp = b.create<linalg::TransposeOp>(
          loc, src2d, dst2d, ArrayRef<int64_t>{1, 0});
      if (auto id = transposeOp->getAttr("nkipy.op_id"))
        newOp->setAttr("nkipy.op_id", id);
    } else {
      // HBM→HBM: allocate SBUF temp, load src, transpose in SBUF, store to dst.
      auto sbufMemSpace = nkipy::MemSpaceEnumAttr::get(
          b.getContext(), nkipy::MemSpaceEnum::Sbuf);
      auto srcShape2d = src2dType.getShape();
      auto dstShape2d = dst2dType.getShape();
      auto sbufSrcType = MemRefType::get(
          srcShape2d, src2dType.getElementType(), nullptr, sbufMemSpace);
      auto sbufDstType = MemRefType::get(
          dstShape2d, dst2dType.getElementType(), nullptr, sbufMemSpace);
      auto sbufSrc = b.create<memref::AllocOp>(loc, sbufSrcType);
      auto sbufDst = b.create<memref::AllocOp>(loc, sbufDstType);
      b.create<memref::CopyOp>(loc, src2d, sbufSrc);       // HBM → SBUF
      auto newOp = b.create<linalg::TransposeOp>(           // transpose in SBUF
          loc, sbufSrc, sbufDst, ArrayRef<int64_t>{1, 0});
      if (auto id = transposeOp->getAttr("nkipy.op_id"))
        newOp->setAttr("nkipy.op_id", id);
      b.create<memref::CopyOp>(loc, sbufDst, dst2d);       // SBUF → HBM
    }

    transposeOp.erase();
    LLVM_DEBUG(llvm::dbgs() << "[SimplifyLinalg] Decomposed " << rank
                 << "D transpose to loop of 2D transposes\n");
  }
}

//===----------------------------------------------------------------------===//
// Preprocessing: Collapse >2D SBUF transpose to 2D
//===----------------------------------------------------------------------===//

/// Replace a >2D SBUF memref.alloc (with unit dims, effectively 2D) with a
/// true 2D alloc + expand_shape view.  Returns the 2D alloc Value, or nullptr
/// if the operand is not a collapsible SBUF alloc.
static Value collapseSbufAllocTo2D(Value operand, int64_t dim0, int64_t dim1,
                                   SmallVector<ReassociationIndices> &reassoc) {
  auto allocOp = operand.getDefiningOp<memref::AllocOp>();
  if (!allocOp)
    return nullptr;
  auto oldType = allocOp.getType();
  auto newType = MemRefType::get(
      {dim0, dim1}, oldType.getElementType(),
      /*layout=*/nullptr, oldType.getMemorySpace());

  OpBuilder b(allocOp);
  Location loc = allocOp.getLoc();
  auto newAlloc = b.create<memref::AllocOp>(loc, newType,
                                             allocOp.getAlignmentAttr());
  auto expandOp = b.create<memref::ExpandShapeOp>(
      loc, oldType, newAlloc, reassoc);

  allocOp.replaceAllUsesWith(expandOp.getResult());
  allocOp.erase();
  return newAlloc.getResult();
}

/// Rewrite >2D SBUF linalg.transpose (with unit dims) to a true 2D transpose.
static void rewriteSbufTransposeTo2D(func::FuncOp func) {
  SmallVector<linalg::TransposeOp> worklist;
  func.walk([&](linalg::TransposeOp op) { worklist.push_back(op); });

  for (auto transposeOp : worklist) {
    auto srcType = dyn_cast<MemRefType>(transposeOp.getInput().getType());
    auto dstType = dyn_cast<MemRefType>(transposeOp.getInit().getType());
    if (!srcType || !dstType || srcType.getRank() <= 2)
      continue;
    if (!isSbuf(srcType) || !isSbuf(dstType))
      continue;

    // Must have exactly 2 non-unit dims to collapse to 2D.
    auto shape = srcType.getShape();
    unsigned nonUnitCount = 0;
    SmallVector<unsigned> nonUnitDims;
    for (unsigned i = 0; i < shape.size(); i++) {
      if (shape[i] != 1) { nonUnitCount++; nonUnitDims.push_back(i); }
    }
    if (nonUnitCount != 2)
      continue;

    // Only rewrite when the non-unit dims are actually transposed.
    auto perm = transposeOp.getPermutation();
    unsigned dstPos0 = 0, dstPos1 = 0;
    for (unsigned d = 0; d < perm.size(); d++) {
      if (perm[d] == (int64_t)nonUnitDims[0]) dstPos0 = d;
      if (perm[d] == (int64_t)nonUnitDims[1]) dstPos1 = d;
    }
    bool needsTranspose = (dstPos0 > dstPos1);

    // Helper: compute 2D collapse params for a given shape.
    auto computeCollapse = [](ArrayRef<int64_t> sh,
                              int64_t &d0, int64_t &d1,
                              SmallVector<ReassociationIndices> &ra) {
      int64_t rank = sh.size();
      unsigned firstNonUnit = 0;
      for (unsigned i = 0; i < sh.size(); i++)
        if (sh[i] != 1) { firstNonUnit = i; break; }
      d0 = 1;
      for (unsigned i = 0; i <= firstNonUnit; i++) d0 *= sh[i];
      d1 = 1;
      for (unsigned i = firstNonUnit + 1; i < (unsigned)rank; i++) d1 *= sh[i];
      ra.clear();
      ReassociationIndices g0, g1;
      for (int64_t i = 0; i <= (int64_t)firstNonUnit; i++) g0.push_back(i);
      for (int64_t i = firstNonUnit + 1; i < rank; i++) g1.push_back(i);
      ra.push_back(g0);
      ra.push_back(g1);
    };

    int64_t srcDim0, srcDim1, dstDim0, dstDim1;
    SmallVector<ReassociationIndices> srcReassoc, dstReassoc;
    computeCollapse(srcType.getShape(), srcDim0, srcDim1, srcReassoc);
    computeCollapse(dstType.getShape(), dstDim0, dstDim1, dstReassoc);

    // Replace both allocs with 2D + expand_shape.
    Value src2d = collapseSbufAllocTo2D(transposeOp.getInput(),
                                         srcDim0, srcDim1, srcReassoc);
    Value dst2d = collapseSbufAllocTo2D(transposeOp.getInit(),
                                         dstDim0, dstDim1, dstReassoc);
    if (!src2d || !dst2d)
      continue;

    // Redirect dealloc and copy ops from expand_shape (3D) to 2D alloc.
    for (auto *val : {&src2d, &dst2d}) {
      for (auto *user : val->getUsers()) {
        auto expand = dyn_cast<memref::ExpandShapeOp>(user);
        if (!expand) continue;
        for (auto *eu : llvm::make_early_inc_range(expand->getUsers())) {
          if (auto dealloc = dyn_cast<memref::DeallocOp>(eu)) {
            dealloc.getMemrefMutable().assign(*val);
          } else if (auto copyOp = dyn_cast<memref::CopyOp>(eu)) {
            OpBuilder cb(copyOp);
            Location cl = copyOp.getLoc();
            bool sbufIsDst = (copyOp.getTarget() == expand.getResult());
            Value hbmOperand = sbufIsDst ? copyOp.getSource()
                                         : copyOp.getTarget();
            auto hbmType = cast<MemRefType>(hbmOperand.getType());
            int64_t hbmRank = hbmType.getRank();

            // Build a rank-reducing subview: [1,128,128] -> [128,128]
            SmallVector<OpFoldResult> offsets(hbmRank,
                                              cb.getI64IntegerAttr(0));
            SmallVector<OpFoldResult> sizes;
            for (int64_t s : hbmType.getShape())
              sizes.push_back(cb.getI64IntegerAttr(s));
            SmallVector<OpFoldResult> strides(hbmRank,
                                              cb.getI64IntegerAttr(1));

            auto sbufType = cast<MemRefType>(val->getType());
            auto hbm2dType = memref::SubViewOp::inferRankReducedResultType(
                sbufType.getShape(), hbmType, offsets, sizes, strides);
            auto hbm2d = cb.create<memref::SubViewOp>(
                cl, cast<MemRefType>(hbm2dType), hbmOperand,
                offsets, sizes, strides);

            if (sbufIsDst) {
              cb.create<memref::CopyOp>(cl, hbm2d, *val);
            } else {
              cb.create<memref::CopyOp>(cl, *val, hbm2d);
            }
            copyOp.erase();
          }
        }
      }
    }

    OpBuilder b(transposeOp);
    Location loc = transposeOp.getLoc();

    if (needsTranspose) {
      // Real transpose: create 2D linalg.transpose with [1, 0].
      auto newOp = b.create<linalg::TransposeOp>(
          loc, src2d, dst2d, ArrayRef<int64_t>{1, 0});
      if (auto id = transposeOp->getAttr("nkipy.op_id"))
        newOp->setAttr("nkipy.op_id", id);
    } else {
      // Just moving unit dim -- same data layout. Emit a copy.
      b.create<memref::CopyOp>(loc, src2d, dst2d);
    }

    transposeOp.erase();
  }
}

//===----------------------------------------------------------------------===//
// Preprocessing: Canonicalize trivial-broadcast generics to named ops
//===----------------------------------------------------------------------===//

/// Convert linalg.generic with trivial broadcast to named ops.
/// After tiling, a broadcast like (128x4x64) * (128x1x64) becomes
/// (128x1x64) * (128x1x64) -- the broadcast dim is now size 1 on both sides.
/// The generic still carries broadcast indexing maps but is effectively
/// a same-shape elementwise op. Convert it to a named op (linalg.mul, etc.)
/// so the existing LinalgElementwiseToNisaPattern can handle it.
static void canonicalizeTrivialBroadcastGenerics(func::FuncOp func) {
  SmallVector<linalg::GenericOp> toConvert;
  func.walk([&](linalg::GenericOp op) {
    // 2 inputs, 1 output, all parallel
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
      return;
    if (!llvm::all_of(op.getIteratorTypesArray(),
            [](utils::IteratorType t) {
              return t == utils::IteratorType::parallel;
            }))
      return;

    // Single binary arith op in body
    Operation *binaryOp = nullptr;
    for (Operation &bodyOp : op.getRegion().front().without_terminator()) {
      if (getArithOpKindFromBodyOp(&bodyOp)) {
        if (binaryOp) return; // multiple ops
        binaryOp = &bodyOp;
      }
    }
    if (!binaryOp) return;

    // Must be a direct binary op (not a wrapped pattern like uitofp(andi(...)))
    if (binaryOp->getNumOperands() != 2) return;

    // Both operands must be block args (not constants)
    if (binaryOp->getOperand(0).getDefiningOp<arith::ConstantOp>() ||
        binaryOp->getOperand(1).getDefiningOp<arith::ConstantOp>())
      return;

    // Check all indexing maps are identity or trivial broadcast
    auto maps = op.getIndexingMapsArray();
    auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
    if (!outType) return;

    for (auto &map : maps) {
      if (map.isIdentity()) continue;
      for (unsigned i = 0; i < map.getNumResults(); ++i) {
        auto expr = map.getResult(i);
        if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
          if (constExpr.getValue() == 0 && outType.getDimSize(i) == 1)
            continue; // trivial broadcast
          return; // non-trivial
        }
        if (!isa<AffineDimExpr>(expr)) return;
      }
    }

    toConvert.push_back(op);
  });

  for (auto op : toConvert) {
    Operation *binaryOp = nullptr;
    for (Operation &bodyOp : op.getRegion().front().without_terminator()) {
      if (getArithOpKindFromBodyOp(&bodyOp)) {
        binaryOp = &bodyOp;
        break;
      }
    }

    // Figure out operand order: body may swap block args
    Block &body = op.getRegion().front();
    Value lhs = op.getDpsInputs()[0];
    Value rhs = op.getDpsInputs()[1];
    if (binaryOp->getOperand(0) == body.getArgument(1) &&
        binaryOp->getOperand(1) == body.getArgument(0))
      std::swap(lhs, rhs);

    OpBuilder builder(op);
    Value output = op.getDpsInits()[0];

    Operation *namedOp = nullptr;
    auto kind = *getArithOpKindFromBodyOp(binaryOp);
    switch (kind) {
    case LocalArithKind::ADD:
      namedOp = builder.create<linalg::AddOp>(
          op.getLoc(), ValueRange{lhs, rhs}, ValueRange{output});
      break;
    case LocalArithKind::SUBTRACT:
      namedOp = builder.create<linalg::SubOp>(
          op.getLoc(), ValueRange{lhs, rhs}, ValueRange{output});
      break;
    case LocalArithKind::MULTIPLY:
      namedOp = builder.create<linalg::MulOp>(
          op.getLoc(), ValueRange{lhs, rhs}, ValueRange{output});
      break;
    case LocalArithKind::DIVIDE:
      namedOp = builder.create<linalg::DivOp>(
          op.getLoc(), ValueRange{lhs, rhs}, ValueRange{output});
      break;
    default:
      continue; // skip unsupported ops
    }

    // Copy over any relevant attrs (like nkipy.op_id)
    if (auto opId = op->getAttr("nkipy.op_id"))
      namedOp->setAttr("nkipy.op_id", opId);

    op.replaceAllUsesWith(namedOp->getResults());
    op.erase();
  }
}

//===----------------------------------------------------------------------===//
// Preprocessing: Replace SBUF gather operands with HBM originals
//===----------------------------------------------------------------------===//

/// For nkipy.gather ops, replace SBUF source/indices with their HBM origins.
/// nisa.dma_copy_indirect requires the source table in HBM. The annotation
/// pass may have copied source/indices to SBUF — undo that so linalg-to-nisa
/// sees HBM operands and can emit dma_copy_indirect directly.
static void prepareGatherForNisaLowering(func::FuncOp func) {
  SmallVector<nkipy::GatherOp> gatherOps;
  func.walk([&](nkipy::GatherOp op) { gatherOps.push_back(op); });

  for (auto gatherOp : gatherOps) {
    // Check source (operand 0) and indices (operand 1).
    for (unsigned idx : {0u, 1u}) {
      Value operand = gatherOp->getOperand(idx);
      auto memrefType = dyn_cast<MemRefType>(operand.getType());
      if (!memrefType || !isSbuf(memrefType))
        continue;

      // Find the memref.copy that writes HBM data into this SBUF alloc.
      Value hbmSource = nullptr;
      memref::CopyOp deadCopy = nullptr;
      for (auto *user : operand.getUsers()) {
        auto copyOp = dyn_cast<memref::CopyOp>(user);
        if (!copyOp || copyOp.getTarget() != operand)
          continue;
        Value base = nkipy::getBaseMemRef(copyOp.getSource());
        if (isHbm(cast<MemRefType>(base.getType()))) {
          hbmSource = copyOp.getSource();
          deadCopy = copyOp;
          break;
        }
      }
      if (!hbmSource)
        continue;

      // Replace the gather operand with the HBM source.
      gatherOp->setOperand(idx, hbmSource);

      // Erase the dead copy.
      deadCopy->erase();

      // If the SBUF alloc has no remaining readers, erase it + dealloc.
      if (auto allocOp = operand.getDefiningOp<memref::AllocOp>()) {
        SmallVector<Operation *> toErase;
        bool canErase = true;
        for (auto *user : allocOp->getResult(0).getUsers()) {
          if (isa<memref::DeallocOp>(user))
            toErase.push_back(user);
          else {
            canErase = false;
            break;
          }
        }
        if (canErase) {
          for (auto *op : toErase)
            op->erase();
          allocOp->erase();
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct SimplifyLinalgPass
    : public PassWrapper<SimplifyLinalgPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyLinalgPass)

  StringRef getArgument() const final { return "simplify-linalg"; }

  StringRef getDescription() const final {
    return "Prepare linalg operations for NISA lowering";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Decompose high-rank transposes (e.g., 4D [0,2,1,3] from multi-head
    // attention reshaping) into loops of 2D transposes.  This handles any
    // memory space (HBM, SharedHBM, SBUF) and any rank where exactly 2
    // dimensions are swapped.  Must run before rewriteSbufTransposeTo2D.
    decomposeHighRankTranspose(func);

    // Rewrite >2D SBUF transpose to 2D.
    // NISA dma_transpose only supports [1,0] (2D) or [2,1,0] (3D full reverse).
    // A 3D transpose [0,2,1] on [1,128,128] must use 2D [1,0] on [128,128].
    // Replace the >2D SBUF allocs with 2D allocs + expand_shape views.
    // After pattern rewriting, getBaseAndOffsets traces through expand_shape
    // so NISA ops use the 2D base. The expand_shape becomes dead -> DCE'd.
    rewriteSbufTransposeTo2D(func);

    // Convert trivial-broadcast linalg.generic to named ops
    canonicalizeTrivialBroadcastGenerics(func);

    // Replace SBUF gather operands with HBM originals for dma_copy_indirect
    prepareGatherForNisaLowering(func);
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<func::FuncOp>> createSimplifyLinalgPass() {
  return std::make_unique<SimplifyLinalgPass>();
}

} // namespace nkipy
} // namespace mlir
