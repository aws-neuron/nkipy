#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyOps.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {

namespace {

/// After inlining a reference body into post-bufferization IR, there may be
/// tensor.extract(bufferization.to_tensor(memref), indices) patterns that
/// canonicalize alone cannot fold.  Walk the function and replace them with
/// memref.load(memref, indices).
static void foldTensorExtractOfToTensor(func::FuncOp func) {
  SmallVector<tensor::ExtractOp> toFold;
  func.walk([&](tensor::ExtractOp extractOp) {
    if (extractOp.getTensor().getDefiningOp<bufferization::ToTensorOp>())
      toFold.push_back(extractOp);
  });
  for (auto extractOp : toFold) {
    auto toTensor =
        extractOp.getTensor().getDefiningOp<bufferization::ToTensorOp>();
    OpBuilder b(extractOp);
    Value loaded = b.create<memref::LoadOp>(
        extractOp.getLoc(), toTensor.getBuffer(), extractOp.getIndices());
    extractOp.getResult().replaceAllUsesWith(loaded);
    extractOp.erase();
  }
}

struct InlineNkipyReferencePass
    : public InlineNkipyReferenceBase<InlineNkipyReferencePass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect nkipy ops with non-empty reference_impl regions.
    SmallVector<Operation *> opsToInline;
    func.walk([&](Operation *op) {
      if (!op->getDialect() ||
          op->getDialect()->getNamespace() != "nkipy")
        return;
      if (isa<nkipy::AnnotateOp>(op) || isa<nkipy::YieldOp>(op))
        return;
      if (op->getNumRegions() == 0)
        return;
      Region &region = op->getRegion(0);
      if (region.empty())
        return;
      opsToInline.push_back(op);
    });

    for (Operation *op : opsToInline)
      inlineReferenceRegion(op);

    // Clean up tensor.extract(to_tensor(memref)) → memref.load(memref)
    // patterns left after inlining into post-bufferization IR.
    foldTensorExtractOfToTensor(func);
  }

  void inlineReferenceRegion(Operation *nkipyOp) {
    Region &region = nkipyOp->getRegion(0);
    Block &refBlock = region.front();

    // Build value mapping: block args → op operands.
    // Post-bufferization, operands may be memrefs while block args are tensors.
    // Insert to_tensor conversions so the reference body works correctly.
    IRMapping mapping;
    OpBuilder builder(nkipyOp);
    for (unsigned i = 0; i < refBlock.getNumArguments(); ++i) {
      Value blockArg = refBlock.getArgument(i);
      Value operand = nkipyOp->getOperand(i);
      if (isa<TensorType>(blockArg.getType()) &&
          isa<MemRefType>(operand.getType())) {
        operand = builder.create<bufferization::ToTensorOp>(
            nkipyOp->getLoc(), blockArg.getType(), operand);
      }
      mapping.map(blockArg, operand);
    }

    // Clone each op (except the yield) before the nkipy op.
    SmallVector<Value> yieldValues;

    for (Operation &innerOp : llvm::make_early_inc_range(refBlock)) {
      if (isa<nkipy::YieldOp>(innerOp)) {
        for (Value v : innerOp.getOperands())
          yieldValues.push_back(mapping.lookupOrDefault(v));
      } else {
        Operation *cloned = builder.clone(innerOp, mapping);
        (void)cloned;
      }
    }

    // Post-bufferization: the inlined body yields tensors, but the outer code
    // reads from the DPS output memref. Copy each result into its DPS init.
    if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(nkipyOp)) {
      for (unsigned i = 0; i < yieldValues.size(); ++i) {
        if (!isa<TensorType>(yieldValues[i].getType()))
          continue;
        auto inits = dpsOp.getDpsInits();
        if (i >= inits.size() || !isa<MemRefType>(inits[i].getType()))
          continue;
        auto tensorType = cast<RankedTensorType>(yieldValues[i].getType());
        auto bufType = MemRefType::get(tensorType.getShape(),
                                        tensorType.getElementType());
        auto buf = builder.create<bufferization::ToBufferOp>(
            nkipyOp->getLoc(), bufType, yieldValues[i]);
        builder.create<memref::CopyOp>(nkipyOp->getLoc(), buf, inits[i]);
      }
    }

    // Replace uses and erase.
    assert(yieldValues.size() == nkipyOp->getNumResults());
    for (unsigned i = 0; i < nkipyOp->getNumResults(); ++i)
      nkipyOp->getResult(i).replaceAllUsesWith(yieldValues[i]);
    nkipyOp->erase();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createInlineNkipyReferencePass() {
  return std::make_unique<InlineNkipyReferencePass>();
}

} // namespace nkipy
} // namespace mlir
