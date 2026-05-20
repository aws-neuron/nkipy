//===- NkipyTransformOps.cpp - Nkipy Transform Operations -----------------===//
//
// Implementation of custom transform dialect operations for NKIPyKernelGen.
//
//===----------------------------------------------------------------------===//

#include "nkipy/TransformOps/NkipyTransformOps.h"

#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyOps.h"

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

/// Return the mem_space that `value` already lives in, if known.
/// Walks back through no-alloc aliasing ops (tensor.extract_slice,
/// linalg.transpose, bufferization.materialize_in_destination) so
/// promote_tensor can skip alloc+copy whenever any aliased value on the
/// chain already carries a mem_space annotation — either via a
/// nkipy.layout side op or via the root bufferization.alloc_tensor.
static std::optional<nkipy::MemSpaceEnum>
findExistingMemSpace(Value value) {
  Value v = value;
  while (v) {
    // nkipy.layout side op attached to the current alias.
    for (Operation *user : v.getUsers()) {
      auto layout = dyn_cast<nkipy::LayoutOp>(user);
      if (!layout || layout.getTarget() != v)
        continue;
      if (auto ms = layout.getMemSpace())
        return *ms;
    }
    // Root of the walk: an alloc_tensor whose memory_space is the answer.
    if (auto allocTensor =
            v.getDefiningOp<bufferization::AllocTensorOp>()) {
      if (auto ms = allocTensor.getMemorySpaceAttr()) {
        if (auto nkipyMs = dyn_cast<nkipy::MemSpaceEnumAttr>(ms))
          return nkipyMs.getValue();
      }
      return std::nullopt;  // untyped alloc; stop walking.
    }
    // Step through one level of no-alloc aliasing.
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return std::nullopt;
    if (auto extract = dyn_cast<tensor::ExtractSliceOp>(defOp))
      v = extract.getSource();
    else if (auto materialize =
                 dyn_cast<bufferization::MaterializeInDestinationOp>(defOp))
      v = materialize.getDest();
    else if (auto transposeOp = dyn_cast<linalg::TransposeOp>(defOp))
      v = transposeOp.getInit();
    else
      return std::nullopt;
  }
  return std::nullopt;
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
  // Extract the target mem_space (if requested) so we can check whether
  // the source already lives there and skip the alloc+copy.
  std::optional<nkipy::MemSpaceEnum> targetMs;
  if (auto msAttr = getMemorySpaceAttr())
    if (auto nkipyMs = dyn_cast<nkipy::MemSpaceEnumAttr>(msAttr))
      targetMs = nkipyMs.getValue();

  for (Value tensor : state.getPayloadValues(getTensor())) {
    auto type = dyn_cast<RankedTensorType>(tensor.getType());
    if (!type) {
      return emitSilenceableError() << "non-tensor type: " << tensor;
    }

    // Source-aware early exit: if the tensor is already annotated (via
    // nkipy.layout or an existing alloc_tensor) to live in the target
    // memory space, there is no promotion to do.  This kills redundant
    // SBUF→SBUF copies that eliminate-same-memspace-copy used to clean
    // up post-bufferize.
    if (targetMs) {
      auto sourceMs = findExistingMemSpace(tensor);
      if (sourceMs && *sourceMs == *targetMs) {
        promoted.push_back(tensor);
        continue;
      }
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
