//===- NkipyTransformOps.cpp - Nkipy Transform Operations -----------------===//
//
// Implementation of custom transform dialect operations for NKIPyKernelGen.
//
//===----------------------------------------------------------------------===//

#include "nkipy/TransformOps/NkipyTransformOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "nkipy/TransformOps/NkipyTransformOps.cpp.inc"

namespace {

//===----------------------------------------------------------------------===//
// PromoteTensorOp helpers
//===----------------------------------------------------------------------===//

/// Return true if the operand may be read from by its owner. This is currently
/// very conservative and only looks inside linalg operations to prevent
/// unintentional data loss.
static bool mayBeRead(OpOperand &operand) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(operand.getOwner());

  // Be conservative about ops we cannot analyze deeper.
  if (!linalgOp)
    return true;

  // Look inside linalg ops.
  Value blockArgument = linalgOp.getMatchingBlockArgument(&operand);
  return !blockArgument.use_empty();
}

/// Return true if the value may be read through any of its uses.
static bool mayBeRead(Value value) {
  // If the value has a reference semantics, it
  // may be read through any alias...
  if (!isa<TensorType, FloatType, IntegerType>(value.getType()))
    return true;
  return llvm::any_of(value.getUses(),
                      static_cast<bool (&)(OpOperand &)>(mayBeRead));
}

} // namespace

//===----------------------------------------------------------------------===//
// PromoteTensorOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::PromoteTensorOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  SmallVector<Value> promoted;
  for (Value tensor : state.getPayloadValues(getTensor())) {
    auto type = dyn_cast<RankedTensorType>(tensor.getType());
    if (!type) {
      return emitSilenceableError() << "non-tensor type: " << tensor;
    }

    Operation *definingOp = tensor.getDefiningOp();
    if (definingOp)
      rewriter.setInsertionPointAfter(definingOp);
    else
      rewriter.setInsertionPointToStart(cast<BlockArgument>(tensor).getOwner());

    // Check this before we emit operations using this value.
    bool needsMaterialization = mayBeRead(tensor);

    SmallVector<Value> dynamicDims;
    llvm::SmallPtrSet<Operation *, 4> preservedOps;
    for (auto [pos, dim] : llvm::enumerate(type.getShape())) {
      if (!ShapedType::isDynamic(dim))
        continue;
      Value cst =
          rewriter.create<arith::ConstantIndexOp>(tensor.getLoc(), static_cast<int64_t>(pos));
      auto dimOp =
          rewriter.create<tensor::DimOp>(tensor.getLoc(), tensor, cst);
      preservedOps.insert(dimOp);
      dynamicDims.push_back(dimOp);
    }
    auto allocation = rewriter.create<bufferization::AllocTensorOp>(
        tensor.getLoc(), type, dynamicDims);
    // Set memory space if provided.
    if (getMemorySpaceAttr())
      allocation.setMemorySpaceAttr(getMemorySpaceAttr());
    Value allocated = allocation;

    // Only insert a materialization (typically bufferizes to a copy) when the
    // value may be read from.
    if (needsMaterialization) {
      auto copy = rewriter.create<bufferization::MaterializeInDestinationOp>(
          tensor.getLoc(), tensor, allocated);
      preservedOps.insert(copy);
      promoted.push_back(copy.getResult());
    } else {
      promoted.push_back(allocated);
    }
    rewriter.replaceAllUsesExcept(tensor, promoted.back(), preservedOps);
  }
  results.setValues(cast<OpResult>(getPromoted()), promoted);
  return DiagnosedSilenceableFailure::success();
}

void transform::PromoteTensorOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTensorMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform dialect extension registration
//===----------------------------------------------------------------------===//

namespace {

class NkipyTransformDialectExtension
    : public transform::TransformDialectExtension<
          NkipyTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NkipyTransformDialectExtension)

  NkipyTransformDialectExtension() {
    registerTransformOps<
#define GET_OP_LIST
#include "nkipy/TransformOps/NkipyTransformOps.cpp.inc"
        >();
  }
};

} // namespace

void mlir::nkipy::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<NkipyTransformDialectExtension>();
}
