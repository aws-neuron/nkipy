#include "nkipy/Dialect/NkipyOps.h"
#include "nkipy/Dialect/NkipyDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AnnotateOp — BufferizableOpInterface
//===----------------------------------------------------------------------===//

bool nkipy::AnnotateOp::bufferizesToMemoryRead(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return false;
}

bool nkipy::AnnotateOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return false;
}

bufferization::AliasingValueList nkipy::AnnotateOp::getAliasingValues(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return {};
}

LogicalResult nkipy::AnnotateOp::bufferize(
    RewriterBase &rewriter, const bufferization::BufferizationOptions &options,
    bufferization::BufferizationState &state) {
  Value target = getTarget();
  // If the target is already a memref, nothing to do.
  if (isa<MemRefType>(target.getType()))
    return success();

  FailureOr<Value> buffer =
      bufferization::getBuffer(rewriter, target, options, state);
  if (failed(buffer))
    return failure();

  rewriter.create<nkipy::AnnotateOp>(
      getLoc(), *buffer, getMemSpaceAttr(), getPartitionDimAttr(),
      getTileSizeAttr(), getReductionTileAttr());
  rewriter.eraseOp(getOperation());
  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp — BufferizableOpInterface
//===----------------------------------------------------------------------===//

bool nkipy::GatherOp::bufferizesToMemoryRead(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  // Source and indices are read; output (DPS init) is only written.
  return !isDpsInit(&opOperand);
}

bool nkipy::GatherOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return isDpsInit(&opOperand);
}

bufferization::AliasingValueList nkipy::GatherOp::getAliasingValues(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  // DPS: the output buffer aliases the result.
  if (isDpsInit(&opOperand))
    return {{getResult(), bufferization::BufferRelation::Equivalent}};
  return {};
}

LogicalResult nkipy::GatherOp::bufferize(
    RewriterBase &rewriter, const bufferization::BufferizationOptions &options,
    bufferization::BufferizationState &state) {
  FailureOr<Value> srcBuf =
      bufferization::getBuffer(rewriter, getSource(), options, state);
  FailureOr<Value> idxBuf =
      bufferization::getBuffer(rewriter, getIndices(), options, state);
  FailureOr<Value> outBuf =
      bufferization::getBuffer(rewriter, getOutput(), options, state);
  if (failed(srcBuf) || failed(idxBuf) || failed(outBuf))
    return failure();

  // Create memref-based gather that writes into the output buffer.
  auto newGather = rewriter.create<nkipy::GatherOp>(
      getLoc(), (*outBuf).getType(), *srcBuf, *idxBuf, *outBuf);

  // Move the reference_impl region. The body retains tensor-typed block args;
  // inline-nkipy-reference handles the memref→tensor conversion at inline time.
  rewriter.inlineRegionBefore(getReferenceImpl(), newGather.getReferenceImpl(),
                              newGather.getReferenceImpl().begin());

  // DPS: replace the tensor result with the output buffer.
  bufferization::replaceOpWithBufferizedValues(rewriter, getOperation(),
                                               *outBuf);
  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp — DestinationStyleOpInterface
//===----------------------------------------------------------------------===//

MutableOperandRange nkipy::GatherOp::getDpsInitsMutable() {
  return getOutputMutable();
}

//===----------------------------------------------------------------------===//
// GatherOp — TilingInterface
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> nkipy::GatherOp::getLoopIteratorTypes() {
  auto resultType = cast<ShapedType>(getResult().getType());
  return SmallVector<utils::IteratorType>(
      resultType.getRank(), utils::IteratorType::parallel);
}

SmallVector<Range> nkipy::GatherOp::getIterationDomain(OpBuilder &b) {
  auto resultType = cast<ShapedType>(getResult().getType());
  SmallVector<Range> domain;
  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    domain.push_back(Range{b.getIndexAttr(0),
                           b.getIndexAttr(resultType.getDimSize(i)),
                           b.getIndexAttr(1)});
  }
  return domain;
}

FailureOr<TilingResult>
nkipy::GatherOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();

  auto sourceType = cast<ShapedType>(getSource().getType());
  auto resultType = cast<ShapedType>(getResult().getType());
  int64_t rank = resultType.getRank();

  // --- Slice indices: indices[i_off : i_off + tN] ---
  SmallVector<OpFoldResult> idxOffsets = {offsets[0]};
  SmallVector<OpFoldResult> idxSizes = {sizes[0]};
  SmallVector<OpFoldResult> idxStrides = {b.getIndexAttr(1)};
  Value indicesTile = b.create<tensor::ExtractSliceOp>(
      loc, getIndices(), idxOffsets, idxSizes, idxStrides);

  // --- Slice source: source[0:V, j_off : j_off+tH] ---
  // All V rows are needed (indices can reference any row); only the
  // embedding/free dimensions are sliced.
  SmallVector<OpFoldResult> srcOffsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> srcSizes;
  SmallVector<OpFoldResult> srcStrides(rank, b.getIndexAttr(1));
  srcSizes.push_back(b.getIndexAttr(sourceType.getDimSize(0)));  // V (all rows)
  for (int64_t d = 1; d < rank; ++d) {
    srcOffsets[d] = offsets[d];
    srcSizes.push_back(sizes[d]);
  }
  Value sourceTile = b.create<tensor::ExtractSliceOp>(
      loc, getSource(), srcOffsets, srcSizes, srcStrides);

  // --- Slice output (DPS init): output[i_off:, j_off:] ---
  SmallVector<OpFoldResult> outOffsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> outSizes(sizes.begin(), sizes.end());
  SmallVector<OpFoldResult> outStrides(rank, b.getIndexAttr(1));
  Value outputTile = b.create<tensor::ExtractSliceOp>(
      loc, getOutput(), outOffsets, outSizes, outStrides);

  // --- Build tiled result type ---
  SmallVector<int64_t> tiledShape;
  for (auto s : sizes) {
    if (auto attr = getConstantIntValue(s))
      tiledShape.push_back(*attr);
    else
      tiledShape.push_back(ShapedType::kDynamic);
  }
  auto tiledResultType = RankedTensorType::get(
      tiledShape, resultType.getElementType());

  // --- Create tiled gather ---
  auto tiledGather = b.create<nkipy::GatherOp>(
      loc, tiledResultType, sourceTile, indicesTile, outputTile);

  // --- Clone reference_impl into the tiled gather ---
  // The reference body is used by InlineNkipyReference for LLVM CPU
  // simulation.  We clone the original region, adjusting block-arg types
  // and fixing up shape-dependent ops (tensor.empty, linalg result types).
  Region &origRegion = getReferenceImpl();
  if (!origRegion.empty()) {
    Region &newRegion = tiledGather.getReferenceImpl();
    Block &origBlock = origRegion.front();

    // Create new block with tiled operand types (source_tile, indices_tile).
    Block *newBlock = new Block();
    newRegion.push_back(newBlock);
    newBlock->addArgument(sourceTile.getType(), loc);
    newBlock->addArgument(indicesTile.getType(), loc);

    // Map original block args → new block args.
    IRMapping mapping;
    mapping.map(origBlock.getArgument(0), newBlock->getArgument(0));
    mapping.map(origBlock.getArgument(1), newBlock->getArgument(1));

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(newBlock);

    for (Operation &op : origBlock) {
      if (isa<tensor::EmptyOp>(&op)) {
        // Replace tensor.empty with the tiled output shape.
        auto newEmpty = b.create<tensor::EmptyOp>(
            loc, tiledResultType.getShape(),
            tiledResultType.getElementType());
        mapping.map(op.getResult(0), newEmpty.getResult());
      } else {
        Operation *cloned = b.clone(op, mapping);
        // Fix result types for DPS linalg ops: the cloned result type is
        // still the original shape, but the (remapped) init operand has the
        // tiled shape.  Align result types with init types.
        if (auto linalgOp = dyn_cast<linalg::LinalgOp>(cloned)) {
          auto inits = linalgOp.getDpsInits();
          for (unsigned i = 0; i < cloned->getNumResults(); ++i) {
            if (i < inits.size())
              cloned->getResult(i).setType(inits[i].getType());
          }
        }
      }
    }
  }

  return TilingResult{{tiledGather.getOperation()},
                      {tiledGather.getResult()},
                      {}};
}

LogicalResult nkipy::GatherOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0)
    return failure();
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

#define GET_OP_CLASSES
#include "nkipy/Dialect/NkipyOps.cpp.inc"