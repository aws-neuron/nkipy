//===- RemoveRedundantZeroFill.cpp - Remove zero fills before matmul ------===//
//
// This pass removes linalg.fill operations that fill with zero and are only
// consumed by matmul-like operations. NISA matmul hardware initializes PSUM
// accumulators to zero automatically (psum_zero_region), so the zero fill is
// redundant.
//
// This runs on tensor IR (before tiling/bufferization) so the fill is removed
// early, before it becomes a memref.copy chain that is harder to optimize.
//
// Pattern:
//   %cst = arith.constant 0.0 : f32
//   %empty = tensor.empty() : tensor<...>
//   %filled = linalg.fill ins(%cst) outs(%empty) -> tensor<...>
//   %result = linalg.matmul ins(%a, %b) outs(%filled) -> tensor<...>
//
// After:
//   %empty = tensor.empty() : tensor<...>
//   %result = linalg.matmul ins(%a, %b) outs(%empty) -> tensor<...>
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<ModuleOp>> createRemoveRedundantZeroFillPass() {
  return std::make_unique<RemoveRedundantZeroFillPass>();
}

} // namespace nkipy
} // namespace mlir
