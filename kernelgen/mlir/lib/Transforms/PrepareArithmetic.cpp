//===- PrepareArithmetic.cpp - Prepare arithmetic ops for NISA lowering ===//
//
// This pass prepares arithmetic operations for NISA lowering by transforming
// operations that don't have direct NISA equivalents.
//
// Transformations:
// - linalg.div(A, B) -> linalg.mul(A, linalg.reciprocal(B))
//   NISA's tensor_tensor_arith doesn't support DIVIDE, so we convert division
//   to multiplication by reciprocal.
// - linalg.generic { arith.divf(%a, %b) } with broadcast indexing maps
//   -> linalg.reciprocal(B) + linalg.generic { arith.mulf(%a, %recip) }
//   Handles broadcast tensor-tensor division (e.g. tensor<MxN> / tensor<Mx1>).
//
// This pass runs before tiling so that the generated reciprocal operations
// get tiled and bufferized normally.
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "nkipy/Dialect/NkipyOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper: Clone nkipy.annotate ops from one value to another
//===----------------------------------------------------------------------===//

static void cloneAnnotations(Value oldValue, Value newValue,
                             PatternRewriter &rewriter) {
  for (Operation *user : oldValue.getUsers()) {
    if (auto annotateOp = dyn_cast<nkipy::AnnotateOp>(user)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfterValue(newValue);
      rewriter.create<nkipy::AnnotateOp>(
          annotateOp.getLoc(), newValue,
          annotateOp.getMemSpaceAttr(), annotateOp.getPartitionDimAttr(),
          annotateOp.getTileSizeAttr(), annotateOp.getReductionTileAttr());
    }
  }
}

//===----------------------------------------------------------------------===//
// Pattern: Any linalg op with division -> reciprocal + multiply
//===----------------------------------------------------------------------===//

/// Convert any linalg op containing division to use reciprocal+multiply.
/// Matches both the named linalg.div and linalg.generic with arith.divf.
struct ConvertDivToReciprocal
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (auto divOp = dyn_cast<linalg::DivOp>(linalgOp.getOperation()))
      return handleNamedDiv(divOp, rewriter);

    auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
    if (!genericOp || genericOp.getNumDpsInits() != 1 ||
        !isAllParallel(genericOp))
      return failure();

    arith::DivFOp divFOp = findUniqueDivF(genericOp);
    if (!divFOp)
      return failure();

    unsigned numInputs = genericOp.getNumDpsInputs();
    if (numInputs == 1)
      return handleScalarDiv(genericOp, divFOp, rewriter);
    if (numInputs == 2)
      return handleBroadcastDiv(genericOp, divFOp, rewriter);
    return failure();
  }

private:
  // --- Helpers ---

  static arith::DivFOp findUniqueDivF(linalg::GenericOp op) {
    arith::DivFOp found = nullptr;
    for (Operation &bodyOp : op.getRegion().front().without_terminator()) {
      if (auto d = dyn_cast<arith::DivFOp>(&bodyOp)) {
        if (found)
          return nullptr;
        found = d;
      }
    }
    return found;
  }

  static linalg::ReciprocalOp createReciprocal(PatternRewriter &rewriter,
                                                Location loc, Value input) {
    auto inputType = cast<RankedTensorType>(input.getType());
    auto recipOut = rewriter.create<tensor::EmptyOp>(
        loc, inputType.getShape(), inputType.getElementType());
    return rewriter.create<linalg::ReciprocalOp>(
        loc, TypeRange{inputType}, ValueRange{input},
        ValueRange{recipOut.getResult()});
  }

  static bool isAllParallel(linalg::GenericOp op) {
    return llvm::all_of(op.getIteratorTypesArray(),
        [](utils::IteratorType t) {
          return t == utils::IteratorType::parallel;
        });
  }

  /// Create reciprocal(divisor) and mul(numerator, recip), replacing origOp.
  void replaceWithReciprocalMul(Operation *origOp, Value numerator,
                                Value divisor, Value output,
                                PatternRewriter &rewriter) const {
    Location loc = origOp->getLoc();
    auto outType = cast<RankedTensorType>(output.getType());

    auto recipOp = createReciprocal(rewriter, loc, divisor);
    cloneAnnotations(origOp->getResult(0), recipOp.getResult(0), rewriter);

    auto mulOp = rewriter.create<linalg::MulOp>(
        loc, TypeRange{outType},
        ValueRange{numerator, recipOp.getResult(0)}, ValueRange{output});
    rewriter.replaceOp(origOp, mulOp.getResults());
  }

  // --- Case handlers ---

  /// linalg.div(A, B) → mul(A, reciprocal(B))
  LogicalResult handleNamedDiv(linalg::DivOp op,
                               PatternRewriter &rewriter) const {
    llvm::errs() << "[PrepareArithmetic] Converting linalg.div to "
                    "mul+reciprocal\n";
    replaceWithReciprocalMul(op, op.getInputs()[0], op.getInputs()[1],
                             op.getOutputs()[0], rewriter);
    return success();
  }

  /// 1-input generic with divf involving a constant.
  LogicalResult handleScalarDiv(linalg::GenericOp op, arith::DivFOp divOp,
                                PatternRewriter &rewriter) const {
    Value divLhs = divOp.getLhs();
    Value divRhs = divOp.getRhs();
    auto lhsConst = divLhs.getDefiningOp<arith::ConstantOp>();
    auto rhsConst = divRhs.getDefiningOp<arith::ConstantOp>();

    if ((!lhsConst && !rhsConst) || (lhsConst && rhsConst))
      return failure();

    // divf(tensor, scalar) → mulf(tensor, 1/scalar) in body
    if (rhsConst) {
      auto floatAttr = dyn_cast<FloatAttr>(rhsConst.getValue());
      if (!floatAttr || floatAttr.getValueAsDouble() == 0.0)
        return failure();
      double recipVal = 1.0 / floatAttr.getValueAsDouble();
      llvm::errs() << "[PrepareArithmetic] Converting divf(tensor, "
                   << floatAttr.getValueAsDouble() << ") to mulf(tensor, "
                   << recipVal << ")\n";

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(divOp);
      auto recipConst = rewriter.create<arith::ConstantOp>(
          divOp.getLoc(),
          rewriter.getFloatAttr(floatAttr.getType(), recipVal));
      rewriter.replaceOpWithNewOp<arith::MulFOp>(divOp, divLhs,
                                                  recipConst.getResult());
      return success();
    }

    // divf(scalar, tensor) → reciprocal(tensor) [* scalar]
    auto floatAttr = dyn_cast<FloatAttr>(lhsConst.getValue());
    if (!floatAttr)
      return failure();
    double scalarVal = floatAttr.getValueAsDouble();
    llvm::errs() << "[PrepareArithmetic] Converting divf(" << scalarVal
                 << ", tensor) to reciprocal\n";

    Location loc = op.getLoc();
    Value input = op.getDpsInputs()[0];
    Value output = op.getDpsInits()[0];
    auto outputType = cast<RankedTensorType>(output.getType());

    if (scalarVal == 1.0) {
      auto recipOp = createReciprocal(rewriter, loc, input);
      cloneAnnotations(op.getResult(0), recipOp.getResult(0), rewriter);
      rewriter.replaceOp(op, recipOp.getResults());
    } else {
      // Create fill(scalar) as the numerator, then use shared helper
      auto fillEmpty = rewriter.create<tensor::EmptyOp>(
          loc, outputType.getShape(), outputType.getElementType());
      auto scalarCst = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(
              cast<FloatType>(outputType.getElementType()), scalarVal));
      auto fillOp = rewriter.create<linalg::FillOp>(
          loc, TypeRange{outputType}, ValueRange{scalarCst.getResult()},
          ValueRange{fillEmpty.getResult()});
      replaceWithReciprocalMul(op, fillOp.getResult(0), input, output,
                               rewriter);
    }
    return success();
  }

  /// 2-input generic with divf between two block arguments.
  LogicalResult handleBroadcastDiv(linalg::GenericOp op, arith::DivFOp divOp,
                                   PatternRewriter &rewriter) const {
    auto rhsArg = dyn_cast<BlockArgument>(divOp.getRhs());
    if (!rhsArg || !isa<BlockArgument>(divOp.getLhs()))
      return failure();

    unsigned rhsArgNum = rhsArg.getArgNumber();
    if (rhsArgNum >= op.getNumDpsInputs())
      return failure();

    llvm::errs() << "[PrepareArithmetic] Converting broadcast divf to "
                    "reciprocal+mulf\n";

    Location loc = op.getLoc();
    Value rhsInput = op.getDpsInputs()[rhsArgNum];
    auto recipOp = createReciprocal(rewriter, loc, rhsInput);

    // Clone generic with reciprocal replacing the rhs input, divf→mulf in body
    SmallVector<Value> newInputs(op.getDpsInputs());
    newInputs[rhsArgNum] = recipOp.getResult(0);

    auto newGeneric = rewriter.create<linalg::GenericOp>(
        loc, op.getResultTypes(), newInputs, op.getDpsInits(),
        op.getIndexingMapsArray(), op.getIteratorTypesArray());

    rewriter.cloneRegionBefore(op.getRegion(), newGeneric.getRegion(),
                               newGeneric.getRegion().end());

    for (Operation &bodyOp : newGeneric.getRegion().front().without_terminator()) {
      if (auto d = dyn_cast<arith::DivFOp>(&bodyOp)) {
        rewriter.setInsertionPoint(d);
        rewriter.replaceOpWithNewOp<arith::MulFOp>(d, d.getLhs(), d.getRhs());
        break;
      }
    }

    rewriter.replaceOp(op, newGeneric.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct PrepareArithmeticPass
    : public PassWrapper<PrepareArithmeticPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareArithmeticPass)

  StringRef getArgument() const final { return "prepare-arithmetic"; }

  StringRef getDescription() const final {
    return "Prepare arithmetic operations for NISA lowering";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<nkipy::NkipyDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    // Add arithmetic preparation patterns
    patterns.add<ConvertDivToReciprocal>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      llvm::errs() << "[PrepareArithmetic] Pattern application failed\n";
      signalPassFailure();
      return;
    }

    llvm::errs() << "[PrepareArithmetic] Pass completed successfully\n";
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<ModuleOp>> createPrepareArithmeticPass() {
  return std::make_unique<PrepareArithmeticPass>();
}

} // namespace nkipy
} // namespace mlir
