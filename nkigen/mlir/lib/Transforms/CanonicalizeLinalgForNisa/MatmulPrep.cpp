//===- MatmulPrep.cpp - Prepare matmul-shaped ops for NISA ---------------===//
//
// Part of the "canonicalize linalg for NISA" family of passes.  Collects
// rewrites that prepare matmul-like ops for NISA lowering.
//
// Passes in this file:
//   --remove-redundant-zero-fill : Remove `linalg.fill(0)` ops whose only
//     users are matmul-like.  NISA matmul auto-zeros PSUM, so the fill
//     would otherwise become a redundant memref.copy / nisa.memset.
//
//   --decompose-batch-matmul : Rewrite `linalg.batch_matmul [B,M,N]` to
//     `scf.for` + `linalg.matmul`.  NISA's tensor engine only supports 2D
//     matmul, so every non-{M,K,N} dimension must be decomposed into a
//     surrounding loop (tile_size on batch dims must be 1).  User
//     annotations on the bmm are forwarded: the `nkipy.layout` is
//     cloned onto the `scf.for` result; the `nkipy.tile_op` is split so
//     the inner `linalg.matmul` carries the 2D (M,N) tile plus the K
//     reduction tile.  Pdim/mem-space bridging (extra transposes/copies
//     when the user's annotation doesn't match NISA's hardware contract)
//     is the job of downstream passes.
//
// Both passes run on tensor IR before tiling/bufferization.
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "nkipy/Dialect/NkipyOps.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

static bool isMatmulLikeOp(Operation *op) {
  return isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
             linalg::MatmulTransposeBOp, linalg::BatchMatmulOp,
             linalg::BatchMatmulTransposeAOp,
             linalg::BatchMatmulTransposeBOp>(op);
}

/// Check if a value is defined by arith.constant with a zero value.
static bool isZeroConstant(Value value) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;

  if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
    return intAttr.getValue().isZero();
  if (auto fpAttr = dyn_cast<FloatAttr>(constOp.getValue()))
    return fpAttr.getValue().isZero();
  return false;
}

/// Remove linalg.fill(zero) when all users of the fill result are matmul-like.
struct RemoveZeroFillBeforeMatmul : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    // Check fill value is zero
    if (!isZeroConstant(fillOp.getInputs()[0]))
      return failure();

    // The fill must have exactly one result (the filled tensor)
    if (fillOp.getNumResults() != 1)
      return failure();

    Value fillResult = fillOp.getResult(0);

    // All users must be matmul-like ops
    for (Operation *user : fillResult.getUsers()) {
      if (!isMatmulLikeOp(user))
        return failure();
    }

    // Replace fill result with the unfilled output tensor
    Value outputTensor = fillOp.getOutputs()[0];
    llvm::errs() << "[RemoveRedundantZeroFill] Removing zero fill before "
                    "matmul: "
                 << *fillOp << "\n";
    rewriter.replaceOp(fillOp, outputTensor);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct RemoveRedundantZeroFillPass
    : public PassWrapper<RemoveRedundantZeroFillPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveRedundantZeroFillPass)

  StringRef getArgument() const final { return "remove-redundant-zero-fill"; }

  StringRef getDescription() const final {
    return "Remove linalg.fill ops with zero values when only used by "
           "matmul-like ops (NISA matmul auto-zeros PSUM)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<RemoveZeroFillBeforeMatmul>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      llvm::errs()
          << "[RemoveRedundantZeroFill] Pattern application failed\n";
      signalPassFailure();
      return;
    }

    llvm::errs() << "[RemoveRedundantZeroFill] Pass completed successfully\n";
  }
};

//===----------------------------------------------------------------------===//
// DecomposeBatchMatmul: linalg.batch_matmul -> scf.for + linalg.matmul
//===----------------------------------------------------------------------===//

/// Decompose one batch_matmul op into an scf.for of 2D linalg.matmul.
/// Forwards user annotations (nkipy.layout / nkipy.tile_op) onto the
/// loop result and the inner matmul respectively.  Returns failure on
/// unsupported shapes or mis-specified batch tiles.
static LogicalResult decomposeOneBatchMatmul(linalg::BatchMatmulOp bmmOp) {
  Value lhs = bmmOp.getInputs()[0];
  Value rhs = bmmOp.getInputs()[1];
  Value init = bmmOp.getOutputs()[0];

  auto initType = cast<RankedTensorType>(init.getType());
  if (initType.getRank() != 3 || !initType.hasStaticShape()) {
    return bmmOp.emitError(
        "decompose-batch-matmul: unsupported batch_matmul shape "
        "(only static rank-3 [B,M,N] is supported)");
  }

  // Collect paired annotations on the bmm result.
  SmallVector<nkipy::LayoutOp> layoutOps;
  SmallVector<nkipy::TileOp> tileOps;
  for (Operation *user : bmmOp.getResult(0).getUsers()) {
    if (auto lay = dyn_cast<nkipy::LayoutOp>(user))
      layoutOps.push_back(lay);
    else if (auto t = dyn_cast<nkipy::TileOp>(user))
      tileOps.push_back(t);
  }

  // NISA only has 2D matmul.  Any tile on the batch dim other than 1 is
  // ill-defined — decomposition iterates the batch one at a time.
  for (auto lay : layoutOps) {
    if (auto ts = lay.getTileSizeAttr()) {
      auto arr = ts.asArrayRef();
      if (!arr.empty() && arr[0] != 1)
        return lay.emitError(
            "decompose-batch-matmul: layout tile_size[0] on a "
            "batch_matmul must be 1 (batch dim cannot be tiled; it is "
            "iterated)");
    }
  }
  for (auto t : tileOps) {
    if (auto ts = t.getLoopTileSizeAttr()) {
      auto arr = ts.asArrayRef();
      if (!arr.empty() && arr[0] != 1)
        return t.emitError(
            "decompose-batch-matmul: tile_op tile_size[0] on a "
            "batch_matmul must be 1 (batch dim cannot be tiled; it is "
            "iterated)");
    }
  }

  Location loc = bmmOp.getLoc();
  int64_t B = initType.getShape()[0];
  int64_t M = initType.getShape()[1];
  int64_t N = initType.getShape()[2];
  Type elemTy = initType.getElementType();

  OpBuilder builder(bmmOp);

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value cB = builder.create<arith::ConstantIndexOp>(loc, B);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  // Accumulator stays in [B,M,N] layout.  Downstream passes decide
  // whether to transpose / copy it based on annotations.
  auto forOp = builder.create<scf::ForOp>(loc, c0, cB, c1, ValueRange{init});

  builder.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  Value acc = forOp.getRegionIterArg(0);

  // Rank-reducing 2D slices from the 3D operands and accumulator.
  auto extract2D = [&](Value src) -> Value {
    auto srcType = cast<RankedTensorType>(src.getType());
    auto shape = srcType.getShape();
    auto sliceType = RankedTensorType::get({shape[1], shape[2]},
                                            srcType.getElementType());
    SmallVector<OpFoldResult> offsets = {iv, builder.getIndexAttr(0),
                                          builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(1),
                                        builder.getIndexAttr(shape[1]),
                                        builder.getIndexAttr(shape[2])};
    SmallVector<OpFoldResult> strides(3, builder.getIndexAttr(1));
    return builder.create<tensor::ExtractSliceOp>(loc, sliceType, src,
                                                   offsets, sizes, strides);
  };

  Value lhsSlice = extract2D(lhs);
  Value rhsSlice = extract2D(rhs);

  auto mmType = RankedTensorType::get({M, N}, elemTy);
  SmallVector<OpFoldResult> sliceOffsets = {iv, builder.getIndexAttr(0),
                                             builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> sliceSizes = {builder.getIndexAttr(1),
                                           builder.getIndexAttr(M),
                                           builder.getIndexAttr(N)};
  SmallVector<OpFoldResult> sliceStrides(3, builder.getIndexAttr(1));

  Value initSlice = builder.create<tensor::ExtractSliceOp>(
      loc, mmType, acc, sliceOffsets, sliceSizes, sliceStrides);
  auto matmulOp = builder.create<linalg::MatmulOp>(
      loc, TypeRange{mmType}, ValueRange{lhsSlice, rhsSlice},
      ValueRange{initSlice});
  Value inserted = builder.create<tensor::InsertSliceOp>(
      loc, matmulOp.getResult(0), acc, sliceOffsets, sliceSizes,
      sliceStrides);

  // Preserve nkipy.op_id across the decomposition so downstream tiling
  // can address the inner matmul.
  if (auto opIdAttr = bmmOp->getAttrOfType<IntegerAttr>("nkipy.op_id"))
    matmulOp->setAttr("nkipy.op_id", opIdAttr);

  builder.create<scf::YieldOp>(loc, ValueRange{inserted});

  Value forResult = forOp.getResult(0);

  // --- Annotation forwarding --------------------------------------------
  // nkipy.layout describes where the accumulator lives; clone it onto
  // the scf.for result.  If the layout has no tile_size, derive one
  // from the user's tile_op by dropping the trailing K entry — that
  // gives the value-shape placement tile for the bmm result.
  // (After this pass runs, infer-layout no longer visits the scf.for
  // result, so we have to populate its layout tile_size here.)
  DenseI64ArrayAttr derivedLayoutTile;
  if (!tileOps.empty()) {
    if (auto ts = tileOps.front().getLoopTileSizeAttr()) {
      auto arr = ts.asArrayRef();
      if (arr.size() >= 2) {
        SmallVector<int64_t> dropK(arr.begin(), arr.end() - 1);
        derivedLayoutTile = DenseI64ArrayAttr::get(bmmOp.getContext(), dropK);
      }
    }
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(forOp);
  for (auto lay : layoutOps) {
    DenseI64ArrayAttr layoutTile = lay.getTileSizeAttr();
    if (!layoutTile)
      layoutTile = derivedLayoutTile;
    builder.create<nkipy::LayoutOp>(
        lay.getLoc(), forResult, lay.getMemSpaceAttr(),
        lay.getPartitionDimAttr(), layoutTile);
  }

  // nkipy.tile_op belongs on the inner matmul: strip the leading batch
  // entry from tile_size [B, M, N, K] -> [M, N, K] (one entry per
  // iterator, in linalg iterator order).
  for (auto t : tileOps) {
    DenseI64ArrayAttr innerTileSize;
    if (auto ts = t.getLoopTileSizeAttr()) {
      auto arr = ts.asArrayRef();
      if (arr.size() >= 2) {
        SmallVector<int64_t> inner(arr.begin() + 1, arr.end());
        innerTileSize = DenseI64ArrayAttr::get(bmmOp.getContext(), inner);
      }
    }
    if (innerTileSize) {
      OpBuilder innerBuilder(matmulOp);
      innerBuilder.setInsertionPointAfter(matmulOp);
      innerBuilder.create<nkipy::TileOp>(
          t.getLoc(), matmulOp.getResult(0), innerTileSize);
    }
  }

  // Replace non-annotation uses of the bmm result with the for result,
  // then erase the old annotations and the bmm.
  SmallVector<OpOperand *> usesToReplace;
  for (OpOperand &use : bmmOp.getResult(0).getUses()) {
    Operation *owner = use.getOwner();
    if (!isa<nkipy::LayoutOp>(owner) && !isa<nkipy::TileOp>(owner))
      usesToReplace.push_back(&use);
  }
  for (OpOperand *use : usesToReplace)
    use->set(forResult);

  for (Operation *user : llvm::make_early_inc_range(
           bmmOp.getResult(0).getUsers())) {
    if (isa<nkipy::LayoutOp>(user) || isa<nkipy::TileOp>(user))
      user->erase();
  }
  bmmOp.erase();

  llvm::errs() << "[DecomposeBatchMatmul] decomposed B=" << B
               << " M=" << M << " N=" << N << "\n";
  return success();
}

struct DecomposeBatchMatmulPass
    : public PassWrapper<DecomposeBatchMatmulPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeBatchMatmulPass)

  StringRef getArgument() const final { return "decompose-batch-matmul"; }

  StringRef getDescription() const final {
    return "Rewrite linalg.batch_matmul to scf.for + linalg.matmul; NISA "
           "has no bmm engine, so every non-{M,K,N} dim must be iterated";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<nkipy::NkipyDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<linalg::BatchMatmulOp> bmms;
    func.walk([&](linalg::BatchMatmulOp op) { bmms.push_back(op); });

    for (auto bmm : bmms) {
      if (failed(decomposeOneBatchMatmul(bmm))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<ModuleOp>> createRemoveRedundantZeroFillPass() {
  return std::make_unique<RemoveRedundantZeroFillPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeBatchMatmulPass() {
  return std::make_unique<DecomposeBatchMatmulPass>();
}

} // namespace nkipy
} // namespace mlir
