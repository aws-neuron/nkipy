//===- KnobDrivenTiling.cpp - Generate Transform dialect for tiling -------===//
//
// This pass generates Transform dialect sequences for tiling operations that
// implement TilingInterface, based on knob annotations.  Linalg ops get
// op-specific treatment (matmul blocking, reduction interleaving, etc.);
// other TilingInterface ops (e.g., nkipy.gather) get elementwise-like tiling.
//
// The pass adds a transform.named_sequence @__transform_main to the module.
// Run --transform-interpreter afterwards to apply the generated transforms.
//
//===----------------------------------------------------------------------===//

#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/HardwareConstants.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "nkipy/Transforms/OpClassification.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "nkipy/TransformOps/NkipyTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include "nkipy/Dialect/NkipyAttrs.h"

#include <map>
#include <string>

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {

namespace {

/// Structure to hold knob information
struct KnobInfo {
  SmallVector<int64_t> tileSize;
  SmallVector<int64_t> reductionTile;  // Tile sizes for reduction dims (e.g., K for matmul)
  int64_t opId = -1;  // nkipy.op_id for per-instance matching, -1 if not set
  int numDpsInputs = -1;  // Number of DPS inputs, -1 if unknown
  bool isElementwise = false;  // Whether this op is elementwise (verified during extraction)
  bool isReduction = false;  // Whether this op has both parallel and reduction iterators
  SmallVector<utils::IteratorType> iteratorTypes;  // For generic ops (needed for reduction tiling)
  SmallVector<int64_t> matmulDims;  // [M, N, K] for matmul ops (for dynamic blocking)

  bool isValid() const { return !tileSize.empty(); }
};

/// Check if an op name corresponds to a transpose operation.
bool isTransposeOp(StringRef opName) {
  return opName == "linalg.transpose";
}

//===----------------------------------------------------------------------===//
// Transform dialect helpers
//===----------------------------------------------------------------------===//

/// Emit transform.structured.match for a linalg op by name and optional op_id.
Value emitMatch(OpBuilder &builder, Location loc, Value moduleArg,
                StringRef opName, DictionaryAttr opAttrs) {
  auto anyOpType = transform::AnyOpType::get(builder.getContext());
  return builder.create<transform::MatchOp>(
      loc, anyOpType, moduleArg,
      builder.getStrArrayAttr({opName}),
      /*matchInterfaceEnum=*/transform::MatchInterfaceEnumAttr{},
      /*opAttrs=*/opAttrs,
      /*filterResultType=*/TypeAttr{},
      /*filterOperandTypes=*/ArrayAttr{}).getResult();
}

/// Emit transform.structured.tile_using_for. Returns the tiled op (result 0).
Value emitTile(OpBuilder &builder, Location loc, Value target,
               ArrayRef<int64_t> tileSizes) {
  auto anyOpType = transform::AnyOpType::get(builder.getContext());
  SmallVector<bool> scalableSizes(tileSizes.size(), false);

  // 1 result for tiled op + 1 per non-zero tile (loop handles)
  SmallVector<Type> resultTypes;
  resultTypes.push_back(anyOpType);
  for (int64_t t : tileSizes)
    if (t != 0) resultTypes.push_back(anyOpType);

  auto tileOp = builder.create<transform::TileUsingForOp>(
      loc, TypeRange(resultTypes), target, ValueRange{},
      tileSizes, ArrayRef<int64_t>{}, ArrayRef<bool>(scalableSizes));
  return tileOp.getResult(0);
}

/// Emit promote_tensor for a specific DPS operand position.
void emitPromoteOperand(OpBuilder &builder, Location loc, Value tiledOp,
                        int64_t operandIdx, Attribute memSpace) {
  auto anyValueType = transform::AnyValueType::get(builder.getContext());
  SmallVector<int64_t> position = {operandIdx};
  auto getOp = builder.create<transform::GetOperandOp>(
      loc, anyValueType, tiledOp, ArrayRef<int64_t>(position),
      /*is_inverted=*/false, /*is_all=*/false);
  builder.create<transform::PromoteTensorOp>(
      loc, anyValueType, getOp.getResult(), /*memory_space=*/memSpace);
}

/// Promote all DPS inputs and the output to SBUF.
void emitPromoteAllToSbuf(OpBuilder &builder, Location loc, Value tiledOp,
                          int numInputs) {
  auto sbufMemSpace = nkipy::MemSpaceEnumAttr::get(
      builder.getContext(), nkipy::MemSpaceEnum::Sbuf);
  for (int i = 0; i < numInputs; ++i)
    emitPromoteOperand(builder, loc, tiledOp, i, sbufMemSpace);
  emitPromoteOperand(builder, loc, tiledOp, numInputs, sbufMemSpace);
}

/// Log tile sizes for debugging.
void logTileSizes(StringRef label, ArrayRef<int64_t> tiles) {
  llvm::errs() << "[KnobDrivenTiling] " << label << ": [";
  for (size_t i = 0; i < tiles.size(); ++i) {
    llvm::errs() << tiles[i];
    if (i + 1 < tiles.size()) llvm::errs() << ", ";
  }
  llvm::errs() << "]\n";
}

/// Validate matmul tile sizes against matrix dimensions.
/// Returns empty string if valid, error message if invalid.
std::string validateMatmulTileSize(linalg::LinalgOp linalgOp, const KnobInfo &knob) {
  Value output = linalgOp.getDpsInits()[0];
  auto outputType = dyn_cast<ShapedType>(output.getType());
  if (!outputType || !outputType.hasStaticShape() || outputType.getRank() < 2)
    return "";

  int64_t outputRank = outputType.getRank();
  if (knob.tileSize.size() != static_cast<size_t>(outputRank)) {
    return "matmul tile_size has " + std::to_string(knob.tileSize.size()) +
           " elements but output tensor has rank " + std::to_string(outputRank) +
           "; tile_size must match output rank";
  }

  if (knob.reductionTile.empty())
    return "matmul requires reduction_tile (e.g., reduction_tile=[K]), got none";

  int64_t tileM = knob.tileSize[outputRank - 2];
  int64_t tileN = knob.tileSize[outputRank - 1];
  int64_t tileK = knob.reductionTile[0];

  int64_t dimM = outputType.getDimSize(outputRank - 2);
  int64_t dimN = outputType.getDimSize(outputRank - 1);

  if (tileM > dimM) {
    return "matmul tile_size M (" + std::to_string(tileM) +
           ") is larger than M dimension (" + std::to_string(dimM) + ")";
  }
  if (tileN > dimN) {
    return "matmul tile_size N (" + std::to_string(tileN) +
           ") is larger than N dimension (" + std::to_string(dimN) + ")";
  }

  Value lhs = linalgOp.getDpsInputs()[0];
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  int64_t dimK = -1;
  if (lhsType && lhsType.hasStaticShape() && lhsType.getRank() >= 2)
    dimK = lhsType.getDimSize(lhsType.getRank() - 1);

  if (dimK > 0 && tileK > dimK) {
    return "matmul K tile (" + std::to_string(tileK) +
           ") is larger than K dimension (" + std::to_string(dimK) + ")";
  }
  
  return "";
}

/// Validate that tensor dimensions are large enough for tiling.
/// Returns empty string if valid, error message if invalid.
std::string validateElementwiseTileSize(linalg::LinalgOp linalgOp, const KnobInfo &knob) {
  if (linalgOp.getDpsInits().empty())
    return "";
  
  Value output = linalgOp.getDpsInits()[0];
  auto outputType = dyn_cast<ShapedType>(output.getType());
  if (!outputType || !outputType.hasStaticShape())
    return ""; // Can't validate dynamic shapes
  
  int64_t rank = outputType.getRank();
  if (knob.tileSize.size() != static_cast<size_t>(rank)) {
    return "tile_size has " + std::to_string(knob.tileSize.size()) +
           " elements but tensor has rank " + std::to_string(rank) +
           "; tile_size must have exactly one element per dimension";
  }
  
  for (size_t i = 0; i < knob.tileSize.size(); ++i) {
    int64_t tile = knob.tileSize[i];
    int64_t dim = outputType.getDimSize(i);
    if (tile > dim) {
      return "tile_size[" + std::to_string(i) + "]=" + std::to_string(tile) + 
             " is larger than dimension[" + std::to_string(i) + "]=" + std::to_string(dim);
    }
  }
  
  return "";
}

/// Validate tile sizes for reduction generic ops.
/// tile_size must match the number of parallel dims, reduction_tile must match reduction dims.
std::string validateReductionTileSize(linalg::LinalgOp linalgOp, const KnobInfo &knob) {
  auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
  if (!genericOp)
    return "expected linalg.generic for reduction validation";

  auto iterTypes = genericOp.getIteratorTypesArray();
  size_t numParallel = 0, numReduction = 0;
  for (auto t : iterTypes) {
    if (t == utils::IteratorType::parallel) numParallel++;
    else if (t == utils::IteratorType::reduction) numReduction++;
  }

  if (knob.tileSize.size() != numParallel) {
    return "reduction op tile_size has " + std::to_string(knob.tileSize.size()) +
           " elements but op has " + std::to_string(numParallel) +
           " parallel dimensions; tile_size must match parallel dim count";
  }

  if (knob.reductionTile.empty()) {
    return "reduction op requires reduction_tile, got none";
  }

  if (knob.reductionTile.size() != numReduction) {
    return "reduction op reduction_tile has " + std::to_string(knob.reductionTile.size()) +
           " elements but op has " + std::to_string(numReduction) +
           " reduction dimensions; reduction_tile must match reduction dim count";
  }

  return "";
}

/// Extract knob map from nkipy.annotate operations
/// Only extracts knobs that have valid tile_size attributes
/// Returns empty map and sets errorMsg if validation fails
std::map<std::string, std::vector<KnobInfo>> extractKnobsByOpType(
    ModuleOp module, std::string &errorMsg) {
  std::map<std::string, std::vector<KnobInfo>> knobsByOp;
  errorMsg = "";
  
  module.walk([&](func::FuncOp func) {
    if (!errorMsg.empty()) return; // Already failed
    
    // First, build a map of Value → KnobInfo.
    // Skip annotations nested inside nkipy regions (reference_impl bodies).
    DenseMap<Value, KnobInfo> valueToKnob;
    func.walk([&](nkipy::AnnotateOp annotateOp) {
      if (isInsideNkipyRegion(annotateOp))
        return;
      Value target = annotateOp.getTarget();
      KnobInfo info;
      
      // Extract tile_size attribute
      if (auto tileSizeAttr = annotateOp.getTileSizeAttr()) {
        auto arrayRef = tileSizeAttr.asArrayRef();
        info.tileSize.assign(arrayRef.begin(), arrayRef.end());
      }

      // Extract reduction_tile attribute (for matmul K dimension, etc.)
      if (auto reductionTileAttr = annotateOp.getReductionTileAttr()) {
        auto arrayRef = reductionTileAttr.asArrayRef();
        info.reductionTile.assign(arrayRef.begin(), arrayRef.end());
      }
      
      // Only add to map if it has valid tile_size
      if (info.isValid()) {
        valueToKnob[target] = info;
      }
    });
    
    // Then, group knobs by op type, validating each.
    // Walk all ops with TilingInterface (covers linalg ops and any nkipy ops
    // that implement TilingInterface, e.g., nkipy.gather).
    // Skip ops nested inside nkipy regions (e.g., the linalg.generic inside
    // nkipy.gather's reference_impl — it must not be tiled independently).
    func.walk([&](Operation *op) {
      if (!errorMsg.empty()) return WalkResult::interrupt();
      if (!isa<TilingInterface>(op))
        return WalkResult::advance();
      if (op->getNumResults() == 0)
        return WalkResult::advance();
      if (isInsideNkipyRegion(op))
        return WalkResult::advance();

      for (Value result : op->getResults()) {
        auto it = valueToKnob.find(result);
        if (it != valueToKnob.end()) {
          std::string opName = op->getName().getStringRef().str();

          // Copy knob and extract op_id
          KnobInfo knobWithId = it->second;
          if (auto opIdAttr = op->getAttrOfType<IntegerAttr>("nkipy.op_id")) {
            knobWithId.opId = opIdAttr.getInt();
          }

          // Validate tile sizes against dimensions
          std::string validationError;

          // Linalg-specific classification
          if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
            if (isMatmulOp(opName)) {
              // Capture matrix dimensions for dynamic blocking.
              Value output = linalgOp.getDpsInits()[0];
              auto outType = dyn_cast<ShapedType>(output.getType());
              Value lhs = linalgOp.getDpsInputs()[0];
              auto lhsType = dyn_cast<ShapedType>(lhs.getType());
              if (outType && outType.hasStaticShape() && outType.getRank() >= 2 &&
                  lhsType && lhsType.hasStaticShape() && lhsType.getRank() >= 2) {
                int64_t r = outType.getRank();
                knobWithId.matmulDims = {
                    outType.getDimSize(r - 2),   // M
                    outType.getDimSize(r - 1),   // N
                    lhsType.getDimSize(lhsType.getRank() - 1)  // K
                };
              }
              validationError = validateMatmulTileSize(linalgOp, knobWithId);
            } else if (isTransposeOp(opName)) {
              knobWithId.isElementwise = true;
              knobWithId.numDpsInputs = linalgOp.getNumDpsInputs();
              validationError = validateElementwiseTileSize(linalgOp, knobWithId);
            } else if (isElementwiseOp(linalgOp)) {
              knobWithId.isElementwise = true;
              knobWithId.numDpsInputs = linalgOp.getNumDpsInputs();
              validationError = validateElementwiseTileSize(linalgOp, knobWithId);
            } else if (isReductionGeneric(linalgOp)) {
              knobWithId.isReduction = true;
              knobWithId.numDpsInputs = linalgOp.getNumDpsInputs();
              knobWithId.iteratorTypes = linalgOp.getIteratorTypesArray();
              validationError = validateReductionTileSize(linalgOp, knobWithId);
            }
          } else {
            // Non-linalg TilingInterface op (e.g., nkipy.gather).
            // Default to elementwise-like tiling: tile all dims, promote.
            knobWithId.isElementwise = true;
            if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(op))
              knobWithId.numDpsInputs = dstOp.getNumDpsInputs();
          }

          if (!validationError.empty()) {
            errorMsg = validationError;
            return WalkResult::interrupt();
          }

          knobsByOp[opName].push_back(knobWithId);

          llvm::errs() << "[KnobDrivenTiling] Found knob for " << opName;
          if (knobWithId.opId >= 0) {
            llvm::errs() << " (op_id=" << knobWithId.opId << ")";
          }
          llvm::errs() << ": tile_size=[";
          for (size_t i = 0; i < it->second.tileSize.size(); ++i) {
            llvm::errs() << it->second.tileSize[i];
            if (i + 1 < it->second.tileSize.size()) llvm::errs() << ", ";
          }
          llvm::errs() << "]";
          if (!it->second.reductionTile.empty()) {
            llvm::errs() << ", reduction_tile=[";
            for (size_t i = 0; i < it->second.reductionTile.size(); ++i) {
              llvm::errs() << it->second.reductionTile[i];
              if (i + 1 < it->second.reductionTile.size()) llvm::errs() << ", ";
            }
            llvm::errs() << "]";
          }
          llvm::errs() << "\n";
          break;
        }
      }
      return WalkResult::advance();
    });
  });
  
  return knobsByOp;
}

/// Build tiling + SBUF promotion for elementwise (and transpose) operations.
void buildElementwiseTiling(OpBuilder &builder, Location loc,
                            Value moduleArg,
                            const std::string &opName,
                            const KnobInfo &knob,
                            DictionaryAttr opAttrs) {
  logTileSizes(opName + " elementwise tile_size", knob.tileSize);

  Value matched = emitMatch(builder, loc, moduleArg, opName, opAttrs);
  Value tiledOp = emitTile(builder, loc, matched, knob.tileSize);

  int numInputs = knob.numDpsInputs >= 0 ? knob.numDpsInputs
      : (isNamedUnaryElementwiseOp(opName) ? 1 : 2);
  emitPromoteAllToSbuf(builder, loc, tiledOp, numInputs);

  llvm::errs() << "[KnobDrivenTiling] Elementwise: promoted " << numInputs
               << " inputs + 1 output to SBUF\n";
}

/// Build tiling + SBUF promotion for reduction operations.
/// Interleaves tile_size (parallel) and reduction_tile (reduction) based on
/// iterator types, then tiles and promotes all operands.
void buildReductionTiling(OpBuilder &builder, Location loc,
                          Value moduleArg,
                          const std::string &opName,
                          const KnobInfo &knob,
                          DictionaryAttr opAttrs) {
  // Interleave parallel and reduction tile sizes by iterator type order.
  SmallVector<int64_t> combinedTileSizes;
  int parallelIdx = 0, reductionIdx = 0;
  for (auto iterType : knob.iteratorTypes) {
    if (iterType == utils::IteratorType::parallel) {
      combinedTileSizes.push_back(
          parallelIdx < static_cast<int>(knob.tileSize.size())
              ? knob.tileSize[parallelIdx++] : 0);
    } else {
      combinedTileSizes.push_back(
          reductionIdx < static_cast<int>(knob.reductionTile.size())
              ? knob.reductionTile[reductionIdx++] : 0);
    }
  }

  logTileSizes(opName + " reduction combined_tile_sizes", combinedTileSizes);

  Value matched = emitMatch(builder, loc, moduleArg, opName, opAttrs);
  Value tiledOp = emitTile(builder, loc, matched, combinedTileSizes);

  int numInputs = knob.numDpsInputs >= 0 ? knob.numDpsInputs : 1;
  emitPromoteAllToSbuf(builder, loc, tiledOp, numInputs);

  llvm::errs() << "[KnobDrivenTiling] Reduction: promoted " << numInputs
               << " inputs + 1 output to SBUF\n";
}

/// Build the transform sequence for matmul with dynamic blocking.
///
/// Uses 2-tile blocking when dimensions are large enough (BLOCK = TILE * 2),
/// degenerating to 1-tile blocking (BLOCK = TILE) for small dimensions.
///
/// Generated loop structure:
///   for block_m in [0, M, BLOCK_M):         // BLOCK_M = TILE_M * blocksM
///     LOAD LHS to SBUF                      // Reused across N-blocks
///     for block_n in [0, N, BLOCK_N):       // BLOCK_N = TILE_N * blocksN
///       LOAD RHS to SBUF                    // Reused within block
///       for tile_m in [0, BLOCK_M, TILE_M): // trip count = blocksM (1 or 2)
///         for tile_n in [0, BLOCK_N, TILE_N): // trip count = blocksN (1 or 2)
///           ALLOC psum buffer
///           for k in [0, K, TILE_K):
///             psum += matmul(lhs_tile, rhs_tile)
///           STORE result tile
///
/// Returns false if the knob is invalid for matmul.
bool buildMatmulBlockingTransforms(OpBuilder &builder, Location loc,
                                    Value moduleArg,
                                    const std::string &opName,
                                    const KnobInfo &knob,
                                    DictionaryAttr opAttrs) {
  if (knob.tileSize.size() < 2) {
    llvm::errs() << "[KnobDrivenTiling] Matmul tile_size must have >= 2 elements\n";
    return false;
  }
  if (knob.reductionTile.empty()) {
    llvm::errs() << "[KnobDrivenTiling] Matmul requires reduction_tile\n";
    return false;
  }

  size_t tsz = knob.tileSize.size();
  int64_t tileM = knob.tileSize[tsz - 2];
  int64_t tileN = knob.tileSize[tsz - 1];
  int64_t tileK = knob.reductionTile[0];

  // Dynamic blocking: use block size 2 if dimension is large enough,
  // otherwise degenerate to block size 1 (no blocking, less data reuse).
  int64_t blocksM = 2, blocksN = 2;
  if (knob.matmulDims.size() == 3) {
    int64_t dimM = knob.matmulDims[0];
    int64_t dimN = knob.matmulDims[1];
    if (dimM < tileM * 2) blocksM = 1;
    if (dimN < tileN * 2) blocksN = 1;
  }
  int64_t blockM = tileM * blocksM;
  int64_t blockN = tileN * blocksN;

  llvm::errs() << "[KnobDrivenTiling] Matmul: TILE=[" << tileM << "," << tileN
               << "," << tileK << "], BLOCK=[" << blockM << "," << blockN
               << "] (blocksM=" << blocksM << ", blocksN=" << blocksN << ")\n";

  auto anyOpType = transform::AnyOpType::get(builder.getContext());
  auto sbufMemSpace = nkipy::MemSpaceEnumAttr::get(
      builder.getContext(), nkipy::MemSpaceEnum::Sbuf);
  auto psumMemSpace = nkipy::MemSpaceEnumAttr::get(
      builder.getContext(), nkipy::MemSpaceEnum::Psum);

  // --- Match ---
  Value matmul = emitMatch(builder, loc, moduleArg, opName, opAttrs);

  // --- Level 1: Block-level tiling ---

  // Tile M blocks
  Value blockMTiled = emitTile(builder, loc, matmul, {blockM, 0, 0});

  // Transpose matmul: matmul(A,B) → matmul_transpose_a(transpose(A), B)
  auto transposeMatmul = builder.create<transform::TransposeMatmulOp>(
      loc, anyOpType, blockMTiled,
      transform::TransposeMatmulInput::lhs);
  Value transposedMatmul = transposeMatmul.getResult();

  // Promote the transpose output to SBUF (inserted by TransposeMatmulOp).
  // Use GetProducerOfOperand to target only this specific transpose,
  // not user-provided transposes that have nkipy.op_id.
  auto getTransposeOp = builder.create<transform::GetProducerOfOperand>(
      loc, anyOpType, transposedMatmul, /*operand_number=*/0);
  emitPromoteOperand(builder, loc, getTransposeOp.getResult(), 1, sbufMemSpace);

  // Promote LHS at block-M level (reused across all N-blocks)
  emitPromoteOperand(builder, loc, transposedMatmul, 0, sbufMemSpace);

  // Tile N blocks
  Value blockNTiled = emitTile(builder, loc, transposedMatmul, {0, blockN, 0});

  // Promote RHS at block-N level (reused within this N-block)
  emitPromoteOperand(builder, loc, blockNTiled, 1, sbufMemSpace);

  // --- Level 2: Tile-level tiling (within blocks) ---

  Value tileMTiled = emitTile(builder, loc, blockNTiled, {tileM, 0, 0});
  Value tileNTiled = emitTile(builder, loc, tileMTiled, {0, tileN, 0});

  // Promote output to PSUM (accumulator for partial sums)
  emitPromoteOperand(builder, loc, tileNTiled, 2, psumMemSpace);

  // Tile K (innermost reduction loop)
  emitTile(builder, loc, tileNTiled, {0, 0, tileK});

  return true;
}

struct NkipyKnobDrivenTilingPass
    : public KnobDrivenTilingBase<NkipyKnobDrivenTilingPass> {
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
  }
    
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Note: batch_matmul -> loop + matmul conversion is now handled by
    // canonicalize-partition-dim pass. All matmul ops here are rank-2.

    // Extract knobs from annotations (only those with valid tile_size)
    std::string validationError;
    auto knobsByOp = extractKnobsByOpType(module, validationError);
    
    // Check for validation errors (tile sizes too large for dimensions)
    if (!validationError.empty()) {
      module->emitError() << "[KnobDrivenTiling] Invalid tile configuration: " 
                          << validationError;
      return signalPassFailure();
    }
    
    if (knobsByOp.empty()) {
      llvm::errs() << "[KnobDrivenTiling] No knob annotations found\n";
      // Still emit an empty __transform_main so transform-interpreter
      // doesn't crash.  This happens when a kernel has only data-movement
      // ops (e.g. np.concatenate) and no compute linalg ops.
      OpBuilder emptyBuilder(ctx);
      module->setAttr("transform.with_named_sequence",
                       emptyBuilder.getUnitAttr());
      emptyBuilder.setInsertionPointToEnd(module.getBody());
      auto anyOpType = transform::AnyOpType::get(ctx);
      auto emptySeq = emptyBuilder.create<transform::NamedSequenceOp>(
          module.getLoc(), "__transform_main",
          TypeAttr::get(FunctionType::get(ctx, {anyOpType}, {})),
          /*sym_visibility=*/StringAttr{},
          /*arg_attrs=*/ArrayAttr{},
          /*res_attrs=*/ArrayAttr{});
      emptySeq.addEntryBlock();
      emptySeq.setArgAttr(0, "transform.readonly",
                           emptyBuilder.getUnitAttr());
      emptyBuilder.setInsertionPointToStart(&emptySeq.getBody().front());
      emptyBuilder.create<transform::YieldOp>(module.getLoc());
      return;
    }
    
    OpBuilder builder(ctx);
    Location loc = module.getLoc();
    
    // Add transform.with_named_sequence attribute to module
    module->setAttr("transform.with_named_sequence", builder.getUnitAttr());
    
    // Create transform.named_sequence @__transform_main at end of module
    builder.setInsertionPointToEnd(module.getBody());
    
    auto anyOpType = transform::AnyOpType::get(ctx);
    
    // Create the named sequence with proper signature
    auto namedSeq = builder.create<transform::NamedSequenceOp>(
        loc,
        "__transform_main",
        TypeAttr::get(FunctionType::get(ctx, {anyOpType}, {})),
        /*sym_visibility=*/StringAttr{},
        /*arg_attrs=*/ArrayAttr{},
        /*res_attrs=*/ArrayAttr{});
    
    // Add entry block with module argument
    namedSeq.addEntryBlock();
    
    // Add attribute to mark the argument as readonly
    namedSeq.setArgAttr(0, "transform.readonly", builder.getUnitAttr());
    
    // Get the module argument for use in generated transforms
    Value moduleArg = namedSeq.getBody().getArgument(0);
    
    // Set insertion point to the start of the named sequence body
    builder.setInsertionPointToStart(&namedSeq.getBody().front());
    
    bool hasAnyTransforms = false;
    
    // Process all ops with knobs (per-instance)
    for (const auto &[opName, knobs] : knobsByOp) {
      for (const KnobInfo &knob : knobs) {
        // Build op_attrs for per-instance matching if op_id is set
        DictionaryAttr opAttrs;
        if (knob.opId >= 0) {
          opAttrs = builder.getDictionaryAttr({
            builder.getNamedAttr("nkipy.op_id", builder.getI64IntegerAttr(knob.opId))
          });
        }
        
        if (isMatmulOp(opName)) {
          // Matmul gets special 6-level blocking treatment
          if (!buildMatmulBlockingTransforms(builder, loc, moduleArg, opName, knob, opAttrs)) {
            llvm::errs() << "[KnobDrivenTiling] Failed to build matmul transforms\n";
            continue;
          }
          hasAnyTransforms = true;
        } else if (knob.isElementwise) {
          // Single-level tiling for elementwise ops (named or elementwise generic)
          buildElementwiseTiling(builder, loc, moduleArg, opName, knob, opAttrs);
          hasAnyTransforms = true;
        } else if (knob.isReduction) {
          // Single-level tiling for reduction ops (generic with reduction iterators)
          buildReductionTiling(builder, loc, moduleArg, opName, knob, opAttrs);
          hasAnyTransforms = true;
        } else {
          llvm::errs() << "[KnobDrivenTiling] Unknown op type: " << opName << " - skipping\n";
        }
      }
    }
    
    // Add transform.yield at the end
    builder.create<transform::YieldOp>(loc);
    
    if (!hasAnyTransforms) {
      // No transforms generated - clean up
      namedSeq.erase();
      module->removeAttr("transform.with_named_sequence");
      llvm::errs() << "[KnobDrivenTiling] No transforms generated\n";
      return;
    }
    
    llvm::errs() << "[KnobDrivenTiling] Generated transform sequence\n";
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createKnobDrivenTilingPass() {
  return std::make_unique<NkipyKnobDrivenTilingPass>();
}

} // namespace nkipy
} // namespace mlir
