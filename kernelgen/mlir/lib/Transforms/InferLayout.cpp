//===- InferLayout.cpp - Infer layout (tiling + placement) for all ops -----===//
//
// This pass infers layout information (tile_size, mem_space, partition_dim,
// reduction_tile) for operations that lack explicit user annotations.
//
// Algorithm:
//   1. Collect existing user annotations into a map (no IR mutation).
//   2. BFS propagation from user annotations through elementwise chains
//      (forward + backward along SSA edges).
//   3. Matmul seeding: for unannotated matmul ops, apply hardware-derived
//      defaults.  Validate any user annotations that reached matmul operands.
//   4. BFS propagation from matmul seeds.
//   5. Elementwise fallback: for any remaining unannotated ops, apply defaults.
//   6. BFS propagation from fallback seeds.
//   7. Error if any linalg op still has no annotation.
//   8. Materialize: create nkipy.annotate ops for all newly-inferred layouts.
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"

#include <deque>

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {

namespace {

/// Parsed matmul dimensions: A[M,K] x B[K,N] -> C[M,N].
/// For batch_matmul: A[B,M,K] x B[B,K,N] -> C[B,M,N].
struct MatmulDims {
  int64_t M, K, N;
  int64_t B = 0;  // 0 means non-batched

  bool isBatched() const { return B > 0; }

  /// Try to parse from a matmul op. Returns std::nullopt on failure.
  static std::optional<MatmulDims> parse(linalg::LinalgOp matmulOp) {
    SmallVector<Value> inputs(matmulOp.getDpsInputs());
    if (inputs.size() < 2)
      return std::nullopt;
    auto typeA = dyn_cast<ShapedType>(inputs[0].getType());
    auto typeB = dyn_cast<ShapedType>(inputs[1].getType());
    if (!typeA || !typeB ||
        !typeA.hasStaticShape() || !typeB.hasStaticShape())
      return std::nullopt;
    int64_t rankA = typeA.getRank();
    if (rankA == 3) {
      // batch_matmul: A[B,M,K] x B[B,K,N]
      return MatmulDims{typeA.getShape()[1], typeA.getShape()[2],
                        typeB.getShape()[2], typeA.getShape()[0]};
    }
    return MatmulDims{typeA.getShape()[0], typeA.getShape()[1],
                      typeB.getShape()[1]};
  }
};

//===----------------------------------------------------------------------===//
// Layout entry for tracking tiling + placement annotations
//===----------------------------------------------------------------------===//

struct LayoutEntry {
  DenseI64ArrayAttr tileSize;
  DenseI64ArrayAttr reductionTile;
  MemSpaceEnumAttr memSpace;
  IntegerAttr partitionDim;  // UI32Attr, optional
  int seedId = -1;           // Which seed originated this entry
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct NkipyInferLayoutPass
    : public InferLayoutBase<NkipyInferLayoutPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<nkipy::NkipyDialect>();
  }

  /// Check if a value traces back to a function argument (possibly through
  /// tensor.extract_slice).  Such values live in HBM and cannot be placed
  /// in SBUF.
  bool tracesToFuncArg(Value val) {
    while (val) {
      if (isa<BlockArgument>(val))
        return true;
      Operation *def = val.getDefiningOp();
      if (!def)
        return false;
      if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(def)) {
        val = extractOp.getSource();
        continue;
      }
      return false;
    }
    return false;
  }

  /// Determine the appropriate memory space for a value.  If the value
  /// already has a known mem_space (user annotation) respect it; if it
  /// traces to a function argument, use SharedHBM; otherwise default to SBUF.
  MemSpaceEnumAttr inferMemSpace(Value val,
                                 const DenseMap<Value, LayoutEntry> &annotatedValues,
                                 MLIRContext *ctx) {
    auto it = annotatedValues.find(val);
    if (it != annotatedValues.end() && it->second.memSpace)
      return it->second.memSpace;
    if (tracesToFuncArg(val))
      return getSharedHbmMemSpace(ctx);
    return getSbufMemSpace(ctx);
  }

  /// Create a default MemSpaceEnumAttr for SBUF.
  MemSpaceEnumAttr getSbufMemSpace(MLIRContext *ctx) {
    return MemSpaceEnumAttr::get(ctx, MemSpaceEnum::Sbuf);
  }

  MemSpaceEnumAttr getSharedHbmMemSpace(MLIRContext *ctx) {
    return MemSpaceEnumAttr::get(ctx, MemSpaceEnum::SharedHbm);
  }

  /// Create a UI32 IntegerAttr for partition_dim.
  IntegerAttr makePartitionDimAttr(MLIRContext *ctx, unsigned pdim) {
    return IntegerAttr::get(IntegerType::get(ctx, 32, IntegerType::Unsigned),
                            pdim);
  }

  /// Compute the default elementwise tile_size for a given shape and
  /// partition_dim.  Rule: partition dim gets min(size, 128), last dim
  /// (in original coordinates, excluding partition dim) gets full extent,
  /// middle dims get 1.
  SmallVector<int64_t> computeElementwiseTileSize(ArrayRef<int64_t> shape,
                                                   unsigned partDim) {
    unsigned rank = shape.size();
    SmallVector<int64_t> tile(rank, 1);

    // Partition dim: capped at MAX_PARTITION_DIM.
    tile[partDim] = std::min(shape[partDim], MAX_PARTITION_DIM);

    // Free dim: the last dim that is not the partition dim.
    unsigned freeDim = rank - 1;
    if (freeDim == partDim && rank > 1)
      freeDim = rank - 2;
    tile[freeDim] = shape[freeDim];

    return tile;
  }

  /// Try to compute a propagated layout for targetVal, given sourceVal's
  /// layout.  Does NOT mutate IR or annotatedValues.  Returns true on
  /// success, with the result in outLayout.
  bool computePropagatedLayout(Value sourceVal, Value targetVal,
                               const LayoutEntry &sourceLayout,
                               linalg::LinalgOp targetLinalgOp,
                               const DenseMap<Value, LayoutEntry> &annotatedValues,
                               MLIRContext *ctx,
                               LayoutEntry &outLayout) {
    auto targetType = dyn_cast<ShapedType>(targetVal.getType());
    if (!targetType || !targetType.hasStaticShape())
      return false;

    auto sourceType = dyn_cast<ShapedType>(sourceVal.getType());
    if (!sourceType || !sourceType.hasStaticShape())
      return false;

    ArrayRef<int64_t> targetShape = targetType.getShape();
    ArrayRef<int64_t> sourceShape = sourceType.getShape();

    SmallVector<int64_t> tileValues(sourceLayout.tileSize.asArrayRef());

    // If the source tile has fewer elements than the source shape rank,
    // this is a reduction result with stripped dims.  Reconstruct the
    // full-rank tile by re-inserting reduction_tile values at the reduced
    // dimensions (where sourceShape[i] == 1 && targetShape[i] > 1).
    // Only applies to generic reductions (sum, mean), NOT matmul.
    if (tileValues.size() < sourceShape.size() && sourceLayout.reductionTile) {
      Operation *sourceDefOp = sourceVal.getDefiningOp();
      auto sourceLinalgOp = sourceDefOp
          ? dyn_cast<linalg::LinalgOp>(sourceDefOp) : nullptr;
      if (sourceLinalgOp && isReductionGeneric(sourceLinalgOp) &&
          !isMatmulOp(sourceLinalgOp)) {
        ArrayRef<int64_t> redTile = sourceLayout.reductionTile.asArrayRef();
        SmallVector<int64_t> expanded;
        size_t tileIdx = 0, redIdx = 0;
        for (size_t i = 0; i < sourceShape.size(); i++) {
          if (sourceShape[i] == 1 && targetShape.size() > i &&
              targetShape[i] > 1 && redIdx < redTile.size()) {
            expanded.push_back(redTile[redIdx++]);
          } else if (tileIdx < tileValues.size()) {
            expanded.push_back(tileValues[tileIdx++]);
          } else {
            expanded.push_back(1);
          }
        }
        tileValues = expanded;
      }
    }

    if (targetShape != sourceShape) {
      if (targetShape.size() != sourceShape.size())
        return false;
      bool broadcastable = true;
      for (size_t i = 0; i < targetShape.size(); i++) {
        if (targetShape[i] != sourceShape[i] &&
            targetShape[i] != 1 && sourceShape[i] != 1) {
          broadcastable = false;
          break;
        }
      }
      if (!broadcastable)
        return false;

      for (size_t i = 0; i < tileValues.size() && i < targetShape.size(); i++) {
        tileValues[i] = std::min(tileValues[i], targetShape[i]);
      }
    }

    // Clamp partition tile to the minimum partition dim across all inputs.
    // This ensures that after tiling, broadcast operands with smaller
    // partition dims (e.g., [1,N] in add(a[128,N], b[1,N])) produce
    // matching tile shapes for NISA ops like tensor_tensor_arith.
    if (isElementwiseOp(targetLinalgOp) || isReductionGeneric(targetLinalgOp)) {
      unsigned partDim = sourceLayout.partitionDim
          ? sourceLayout.partitionDim.getValue().getZExtValue() : 0;
      if (partDim < tileValues.size()) {
        int64_t minPartSize = tileValues[partDim];
        for (Value input : targetLinalgOp.getDpsInputs()) {
          auto inType = dyn_cast<ShapedType>(input.getType());
          if (!inType || !inType.hasStaticShape())
            continue;
          if (partDim < (unsigned)inType.getRank())
            minPartSize = std::min(minPartSize, inType.getShape()[partDim]);
        }
        if (minPartSize < tileValues[partDim]) {
          tileValues[partDim] = minPartSize;
          llvm::errs() << "[InferLayout] Clamped partition tile to "
                       << minPartSize << " for broadcast operand\n";
        }
      }
    }

    // Handle reduction ops: strip tile dimensions for size-1 output dims.
    DenseI64ArrayAttr inferredReductionTile;
    if (isReductionGeneric(targetLinalgOp)) {
      SmallVector<int64_t> stripped;
      SmallVector<int64_t> reductionTileValues;

      bool needInferReductionTile = !sourceLayout.reductionTile;

      Value reductionInput = targetLinalgOp.getDpsInputs()[0];
      auto inputType = dyn_cast<ShapedType>(reductionInput.getType());
      ArrayRef<int64_t> inputTile;
      if (needInferReductionTile) {
        auto inputIt = annotatedValues.find(reductionInput);
        if (inputIt != annotatedValues.end())
          inputTile = inputIt->second.tileSize.asArrayRef();
      }

      for (size_t i = 0; i < tileValues.size() && i < targetShape.size(); i++) {
        if (targetShape[i] > 1) {
          stripped.push_back(tileValues[i]);
        } else if (needInferReductionTile) {
          if (!inputTile.empty() && i < inputTile.size())
            reductionTileValues.push_back(inputTile[i]);
          else if (inputType && inputType.hasStaticShape() &&
                   i < (size_t)inputType.getRank())
            reductionTileValues.push_back(inputType.getShape()[i]);
        }
      }
      tileValues = stripped;

      if (!reductionTileValues.empty())
        inferredReductionTile =
            DenseI64ArrayAttr::get(ctx, reductionTileValues);
    }

    outLayout.tileSize = DenseI64ArrayAttr::get(ctx, tileValues);
    outLayout.reductionTile =
        inferredReductionTile ? inferredReductionTile : sourceLayout.reductionTile;
    // Always default to SBUF for propagated layouts.  mem_space is
    // determined by the value's role (return value → SharedHbm, else SBUF),
    // not by propagation from neighbors.
    outLayout.memSpace = getSbufMemSpace(ctx);
    outLayout.partitionDim = sourceLayout.partitionDim;
    return true;
  }

  /// Check if two layouts conflict.  Returns a human-readable description
  /// of the conflict, or empty string if they are compatible.
  ///
  /// Tile sizes are compatible if one divides the other in every dimension
  /// (the consumer with the smaller tile can subdivide the producer's tiles).
  /// mem_space is not checked: propagation defaults to SBUF while user
  /// annotations may specify SharedHbm — this is expected, not a conflict.
  std::string describeConflict(const LayoutEntry &existing,
                               const LayoutEntry &proposed) {
    if (existing.partitionDim && proposed.partitionDim &&
        existing.partitionDim.getInt() != proposed.partitionDim.getInt()) {
      return "conflicting partition_dim: existing=" +
             std::to_string(existing.partitionDim.getInt()) +
             " vs proposed=" +
             std::to_string(proposed.partitionDim.getInt());
    }
    if (existing.tileSize && proposed.tileSize) {
      auto existArr = existing.tileSize.asArrayRef();
      auto propArr = proposed.tileSize.asArrayRef();
      if (existArr.size() == propArr.size()) {
        for (size_t i = 0; i < existArr.size(); i++) {
          int64_t larger = std::max(existArr[i], propArr[i]);
          int64_t smaller = std::min(existArr[i], propArr[i]);
          if (smaller > 0 && larger % smaller != 0) {
            std::string msg = "incompatible tile_size: [";
            for (auto [j, v] : llvm::enumerate(existArr)) {
              if (j > 0) msg += ", ";
              msg += std::to_string(v);
            }
            msg += "] vs [";
            for (auto [j, v] : llvm::enumerate(propArr)) {
              if (j > 0) msg += ", ";
              msg += std::to_string(v);
            }
            msg += "] (one must divide the other in each dimension)";
            return msg;
          }
        }
      }
    }
    return "";  // no conflict
  }

  /// Compute the matmul-specific layout for an operand, given the matmul's
  /// shapes.  operandIdx: 0 = A (stationary), 1 = B (moving).
  /// Returns true on success.
  bool computeMatmulOperandLayout(linalg::LinalgOp matmulOp,
                                  unsigned operandIdx,
                                  Value operandValue,
                                  const LayoutEntry &resultLayout,
                                  const DenseMap<Value, LayoutEntry> &annotatedValues,
                                  MLIRContext *ctx,
                                  LayoutEntry &outLayout) {
    auto dims = MatmulDims::parse(matmulOp);
    if (!dims)
      return false;

    outLayout.memSpace = inferMemSpace(operandValue, annotatedValues, ctx);
    outLayout.seedId = resultLayout.seedId;

    if (dims->isBatched()) {
      // batch_matmul: operands are [B,M,K] and [B,K,N]
      // Tile batch dim to 1 (will be looped over by CanonicalizePartitionDim)
      if (operandIdx == 0) {
        // A (stationary): [B, M, K], partition_dim=2 (K)
        outLayout.tileSize = DenseI64ArrayAttr::get(
            ctx, {1,
                  std::min(dims->M, MAX_PARTITION_DIM),
                  std::min(dims->K, MAX_PARTITION_DIM)});
        outLayout.partitionDim = makePartitionDimAttr(ctx, 2);
      } else {
        // B (moving): [B, K, N], partition_dim=1 (K)
        outLayout.tileSize = DenseI64ArrayAttr::get(
            ctx, {1,
                  std::min(dims->K, MAX_PARTITION_DIM),
                  std::min(dims->N, MAX_FREE_DIM_MATMUL)});
        outLayout.partitionDim = makePartitionDimAttr(ctx, 1);
      }
    } else if (operandIdx == 0) {
      // A (stationary): [M, K], partition_dim=1 (K)
      outLayout.tileSize = DenseI64ArrayAttr::get(
          ctx, {std::min(dims->M, MAX_PARTITION_DIM),
                std::min(dims->K, MAX_PARTITION_DIM)});
      outLayout.partitionDim = makePartitionDimAttr(ctx, 1);
    } else {
      // B (moving): [K, N], partition_dim=0 (K)
      outLayout.tileSize = DenseI64ArrayAttr::get(
          ctx, {std::min(dims->K, MAX_PARTITION_DIM),
                std::min(dims->N, MAX_FREE_DIM_MATMUL)});
      outLayout.partitionDim = makePartitionDimAttr(ctx, 0);
    }
    return true;
  }

  /// Result of tryInsertLayout: whether the value was newly inserted,
  /// already existed (compatible), or conflicted.
  enum class InsertResult { Inserted, Exists, Conflict };

  /// Try to insert a propagated layout for targetValue.  If already annotated,
  /// check for conflicts.  On success (Inserted), adds to queue.
  InsertResult tryInsertLayout(Value targetValue,
                               const LayoutEntry &propagated,
                               const LayoutEntry &sourceLayout,
                               Operation *targetOp,
                               std::deque<Value> &queue,
                               DenseMap<Value, LayoutEntry> &annotatedValues,
                               int &numInferred,
                               StringRef direction) {
    auto existingIt = annotatedValues.find(targetValue);
    if (existingIt != annotatedValues.end()) {
      if (existingIt->second.seedId != sourceLayout.seedId) {
        std::string conflict =
            describeConflict(existingIt->second, propagated);
        if (!conflict.empty()) {
          llvm::errs() << "[InferLayout] CONFLICT (" << direction << "): "
                       << "current seed=" << sourceLayout.seedId
                       << " existing seed=" << existingIt->second.seedId
                       << " at " << targetOp->getName() << "\n";
          targetOp->emitError("infer-layout: ") << conflict;
          return InsertResult::Conflict;
        }
      }
      return InsertResult::Exists;
    }

    LayoutEntry entry = propagated;
    entry.seedId = sourceLayout.seedId;
    annotatedValues[targetValue] = entry;
    queue.push_back(targetValue);
    numInferred++;
    llvm::errs() << "[InferLayout] " << direction << "-propagated to "
                 << targetOp->getName() << "\n";
    return InsertResult::Inserted;
  }

  /// BFS propagation from all values currently in the queue.
  /// Explores both forward (to consumers) and backward (to producers)
  /// along SSA edges through elementwise/reduction chains.
  /// Only modifies annotatedValues (no IR mutation).
  int bfsPropagation(std::deque<Value> &queue,
                     DenseMap<Value, LayoutEntry> &annotatedValues,
                     MLIRContext *ctx,
                     bool &hasConflict) {
    int numInferred = 0;

    while (!queue.empty()) {
      Value current = queue.front();
      queue.pop_front();

      if (hasConflict)
        break;

      auto it = annotatedValues.find(current);
      if (it == annotatedValues.end())
        continue;
      LayoutEntry layout = it->second;

      Operation *defOp = current.getDefiningOp();
      auto defLinalgOp = defOp ? dyn_cast<linalg::LinalgOp>(defOp) : nullptr;

      // --- Backward: from result to producer inputs ---
      if (defLinalgOp) {
        bool isMatmul = isMatmulOp(defLinalgOp);
        SmallVector<Value> inputs(defLinalgOp.getDpsInputs());

        for (auto [idx, input] : llvm::enumerate(inputs)) {
          Operation *producerOp = input.getDefiningOp();
          if (!producerOp)
            continue;

          Value targetValue = input;
          linalg::LinalgOp targetLinalgOp;
          if (!isMatmul) {
            targetLinalgOp = dyn_cast<linalg::LinalgOp>(producerOp);
            if (!targetLinalgOp || !isAnnotatableOp(targetLinalgOp) ||
                targetLinalgOp->getNumResults() == 0)
              continue;
            targetValue = targetLinalgOp->getResult(0);
          }

          LayoutEntry propagated;
          bool computed = isMatmul
              ? computeMatmulOperandLayout(defLinalgOp, idx, targetValue,
                                           layout, annotatedValues, ctx,
                                           propagated)
              : computePropagatedLayout(current, targetValue, layout,
                                        targetLinalgOp, annotatedValues, ctx,
                                        propagated);
          if (!computed)
            continue;

          auto result = tryInsertLayout(targetValue, propagated, layout,
                                        producerOp, queue, annotatedValues,
                                        numInferred, "Backward");
          if (result == InsertResult::Conflict) {
            hasConflict = true;
            return numInferred;
          }
        }
      }

      // --- Forward: from a value to its consumer op results ---
      for (Operation *userOp : current.getUsers()) {
        auto userLinalgOp = dyn_cast<linalg::LinalgOp>(userOp);
        if (!userLinalgOp)
          continue;
        if (!isElementwiseOp(userLinalgOp) && !isReductionGeneric(userLinalgOp))
          continue;
        if (userLinalgOp->getNumResults() == 0)
          continue;

        Value userResult = userLinalgOp->getResult(0);

        LayoutEntry propagated;
        if (!computePropagatedLayout(current, userResult, layout,
                                     userLinalgOp, annotatedValues,
                                     ctx, propagated))
          continue;

        auto result = tryInsertLayout(userResult, propagated, layout,
                                      userOp, queue, annotatedValues,
                                      numInferred, "Forward");
        if (result == InsertResult::Conflict) {
          hasConflict = true;
          return numInferred;
        }
      }
    }

    return numInferred;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    llvm::errs() << "[InferLayout] Processing function: "
                 << func.getName() << "\n";

    // This map tracks the layout assignment for each Value.
    // It is built up across all phases.  IR is only mutated at the very end.
    DenseMap<Value, LayoutEntry> annotatedValues;
    // Track which values already had user-created nkipy.annotate ops
    // so we don't duplicate them during materialization.
    DenseSet<Value> userAnnotatedValues;
    int numInferred = 0;
    int nextSeedId = 0;

    // Collect values that flow into func.return — these must be SharedHbm.
    DenseSet<Value> returnValues;
    func.walk([&](func::ReturnOp returnOp) {
      for (Value operand : returnOp.getOperands())
        returnValues.insert(operand);
    });

    // ================================================================
    // Phase 1: Collect existing user annotations
    // ================================================================
    func.walk([&](nkipy::AnnotateOp annotateOp) {
      if (isInsideNkipyRegion(annotateOp))
        return;
      if (auto tileSizeAttr = annotateOp.getTileSizeAttr()) {
        LayoutEntry entry;
        entry.tileSize = tileSizeAttr;
        entry.reductionTile = annotateOp.getReductionTileAttr();
        entry.memSpace = annotateOp.getMemSpaceAttr();
        entry.partitionDim = annotateOp.getPartitionDimAttr();
        entry.seedId = nextSeedId++;
        annotatedValues[annotateOp.getTarget()] = entry;
        userAnnotatedValues.insert(annotateOp.getTarget());
      }
    });

    llvm::errs() << "[InferLayout] Found " << annotatedValues.size()
                 << " user annotation(s)\n";

    // ================================================================
    // Phase 2: BFS propagation from user annotations
    // ================================================================
    bool hasConflict = false;
    {
      std::deque<Value> queue;
      for (auto &kv : annotatedValues)
        queue.push_back(kv.first);
      numInferred += bfsPropagation(queue, annotatedValues, ctx, hasConflict);
    }
    if (hasConflict)
      return signalPassFailure();

    // ================================================================
    // Phase 3: Matmul seeding
    // ================================================================
    {
      std::deque<Value> matmulSeeds;

      func.walk([&](linalg::LinalgOp linalgOp) {
        if (isInsideNkipyRegion(linalgOp))
          return;
        if (!isMatmulOp(linalgOp) || linalgOp->getNumResults() == 0)
          return;

        Value resultVal = linalgOp->getResult(0);
        if (annotatedValues.count(resultVal))
          return;

        auto dims = MatmulDims::parse(linalgOp);
        if (!dims)
          return;

        // Seed result C[M,N] (or C[B,M,N] for batch_matmul):
        // partition_dim=0 (M for 2D, B for 3D — batch dim gets looped over)
        LayoutEntry cLayout;
        if (dims->isBatched()) {
          // batch_matmul: tile_size=[1, M_tile, N_tile] — batch dim tiled to 1
          // (CanonicalizePartitionDim will loop over batch and drop it)
          cLayout.tileSize = DenseI64ArrayAttr::get(
              ctx,
              {1,
               std::min(dims->M, MAX_PARTITION_DIM),
               std::min(dims->N, MAX_FREE_DIM_MATMUL)});
        } else {
          cLayout.tileSize = DenseI64ArrayAttr::get(
              ctx,
              {std::min(dims->M, MAX_PARTITION_DIM),
               std::min(dims->N, MAX_FREE_DIM_MATMUL)});
        }
        cLayout.reductionTile = DenseI64ArrayAttr::get(
            ctx, {std::min(dims->K, MAX_PARTITION_DIM)});
        cLayout.memSpace = getSbufMemSpace(ctx);
        cLayout.partitionDim = makePartitionDimAttr(ctx, 0);
        cLayout.seedId = nextSeedId++;

        annotatedValues[resultVal] = cLayout;
        matmulSeeds.push_back(resultVal);
        numInferred++;
        llvm::errs() << "[InferLayout] Matmul-seeded result C\n";

        // Operands A and B are seeded during BFS backward propagation
        // from the matmul result via computeMatmulOperandLayout().
      });

      // Phase 4: BFS propagation from matmul seeds
      numInferred += bfsPropagation(matmulSeeds, annotatedValues, ctx,
                                    hasConflict);
    }
    if (hasConflict)
      return signalPassFailure();

    // ================================================================
    // Phase 5: Elementwise / standalone fallback
    // ================================================================
    {
      std::deque<Value> fallbackSeeds;

      func.walk([&](linalg::LinalgOp linalgOp) {
        if (isInsideNkipyRegion(linalgOp))
          return;
        if (!isAnnotatableOp(linalgOp))
          return;
        if (linalgOp->getNumResults() == 0)
          return;

        Value result = linalgOp->getResult(0);
        if (annotatedValues.count(result))
          return;

        auto resultType = dyn_cast<ShapedType>(result.getType());
        if (!resultType || !resultType.hasStaticShape())
          return;

        ArrayRef<int64_t> shape = resultType.getShape();
        unsigned partDim = 0;

        LayoutEntry layout;
        layout.memSpace = getSbufMemSpace(ctx);
        layout.partitionDim = makePartitionDimAttr(ctx, partDim);
        layout.seedId = nextSeedId++;

        if (isReductionGeneric(linalgOp)) {
          // For reduction ops, tile_size covers only parallel dims and
          // reduction_tile covers only reduction dims.
          auto genericOp = cast<linalg::GenericOp>(linalgOp.getOperation());
          auto iterTypes = genericOp.getIteratorTypesArray();

          // Get input shape to determine reduction dim sizes.
          Value reductionInput = linalgOp.getDpsInputs()[0];
          auto inputType = dyn_cast<ShapedType>(reductionInput.getType());

          SmallVector<int64_t> parallelTile;
          SmallVector<int64_t> reductionTile;
          for (size_t i = 0; i < iterTypes.size(); i++) {
            if (iterTypes[i] == utils::IteratorType::parallel) {
              int64_t dimSize = shape[parallelTile.size()];
              if (parallelTile.size() == partDim)
                parallelTile.push_back(std::min(dimSize, MAX_PARTITION_DIM));
              else
                parallelTile.push_back(dimSize);
            } else {
              int64_t redDimSize = (inputType && inputType.hasStaticShape() &&
                                    i < (size_t)inputType.getRank())
                                       ? inputType.getShape()[i]
                                       : 1;
              reductionTile.push_back(redDimSize);
            }
          }

          layout.tileSize = DenseI64ArrayAttr::get(ctx, parallelTile);
          layout.reductionTile = DenseI64ArrayAttr::get(ctx, reductionTile);
        } else {
          SmallVector<int64_t> tile = computeElementwiseTileSize(shape, partDim);
          // Clamp partition tile for broadcast inputs.
          int64_t minPartSize = tile[partDim];
          for (Value input : linalgOp.getDpsInputs()) {
            auto inType = dyn_cast<ShapedType>(input.getType());
            if (!inType || !inType.hasStaticShape())
              continue;
            if (partDim < (unsigned)inType.getRank())
              minPartSize = std::min(minPartSize, inType.getShape()[partDim]);
          }
          tile[partDim] = minPartSize;
          layout.tileSize = DenseI64ArrayAttr::get(ctx, tile);
        }

        annotatedValues[result] = layout;
        fallbackSeeds.push_back(result);
        numInferred++;
        llvm::errs() << "[InferLayout] Fallback-seeded "
                     << linalgOp->getName() << "\n";
      });

      // Phase 6: BFS propagation from fallback seeds
      numInferred += bfsPropagation(fallbackSeeds, annotatedValues, ctx,
                                    hasConflict);
    }
    if (hasConflict)
      return signalPassFailure();

    // ================================================================
    // Phase 7: Error check — all linalg ops should be annotated
    // ================================================================
    bool hasError = false;
    func.walk([&](linalg::LinalgOp linalgOp) {
      if (isInsideNkipyRegion(linalgOp))
        return;
      if (!isAnnotatableOp(linalgOp))
        return;
      if (linalgOp->getNumResults() == 0)
        return;
      Value result = linalgOp->getResult(0);
      if (!annotatedValues.count(result)) {
        linalgOp.emitError("infer-layout: unable to determine layout for op");
        hasError = true;
      }
    });
    if (hasError)
      return signalPassFailure();

    // ================================================================
    // Phase 7.5b: Override mem_space for return values to SharedHbm
    // ================================================================
    for (Value retVal : returnValues) {
      auto it = annotatedValues.find(retVal);
      if (it != annotatedValues.end()) {
        it->second.memSpace = getSharedHbmMemSpace(ctx);
      }
    }

    // ================================================================
    // Phase 7.6: Fill in missing mem_space for all entries
    // ================================================================
    for (auto &kv : annotatedValues) {
      LayoutEntry &layout = kv.second;
      if (!layout.memSpace) {
        layout.memSpace = inferMemSpace(kv.first, annotatedValues, ctx);
      }
    }

    // ================================================================
    // Phase 8: Materialize — create nkipy.annotate ops for all
    //          newly-inferred layouts, and update user-annotated ops
    //          that had mem_space filled in.
    // ================================================================
    OpBuilder builder(ctx);
    for (auto &kv : annotatedValues) {
      Value val = kv.first;
      const LayoutEntry &layout = kv.second;

      if (userAnnotatedValues.count(val)) {
        // Update existing annotate op if mem_space was inferred
        for (Operation *user : val.getUsers()) {
          if (auto annotateOp = dyn_cast<nkipy::AnnotateOp>(user)) {
            if (!annotateOp.getMemSpace() && layout.memSpace) {
              annotateOp.setMemSpaceAttr(layout.memSpace);
            }
            break;
          }
        }
        continue;
      }

      builder.setInsertionPointAfterValue(val);
      builder.create<nkipy::AnnotateOp>(
          val.getLoc(), val,
          layout.memSpace, layout.partitionDim,
          layout.tileSize, layout.reductionTile);
    }

    llvm::errs() << "[InferLayout] Inferred " << numInferred
                 << " layout annotation(s)\n";
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createInferLayoutPass() {
  return std::make_unique<NkipyInferLayoutPass>();
}

} // namespace nkipy
} // namespace mlir
