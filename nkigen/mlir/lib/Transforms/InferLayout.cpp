//===- InferLayout.cpp - Infer layout (tiling + placement) for all ops -----===//
//
// This pass infers layout information (tile_size, mem_space, partition_dim)
// for operations that lack explicit user annotations.
//
// tile_size has one entry per iterator of the producing op, in linalg
// iterator order (entry i applies to iterator i — no reordering).  For
// elementwise ops this matches the output rank; for reductions it matches
// the input rank; for matmul it is rank 3 ([M, N, K], matching the matmul
// iterator order [parallel, parallel, reduction]).
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
//   8. Materialize: create nkipy.layout + nkipy.tile_op ops for all
//      newly-inferred layouts.
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
  DenseI64ArrayAttr tileSize;  // One entry per iterator, in linalg iterator
                               // order (entry i -> iterator i).
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

  /// SBUF partition limit (== max value the partition dim can take in a tile)
  /// for the configured target.  Falls back to trn2 if the option is unset
  /// or the target is unknown.
  int64_t maxPartitionDim() const {
    StringRef t = target.empty() ? StringRef("trn2") : StringRef(target);
    if (auto n = nkipy::getSbufNumPartitions(t))
      return *n;
    if (auto n = nkipy::getSbufNumPartitions("trn2"))
      return *n;
    return 128;
  }

  /// Per-tile cap on the matmul output free dim for the configured target.
  int64_t matmulFreeDimCap() const {
    StringRef t = target.empty() ? StringRef("trn2") : StringRef(target);
    if (auto n = nkipy::getMatmulFreeDimTileCap(t))
      return *n;
    if (auto n = nkipy::getMatmulFreeDimTileCap("trn2"))
      return *n;
    return 512;
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

    // Partition dim: capped at maxPartitionDim().
    tile[partDim] = std::min(shape[partDim], maxPartitionDim());

    // Free dim: the last dim that is not the partition dim.
    unsigned freeDim = rank - 1;
    if (freeDim == partDim && rank > 1)
      freeDim = rank - 2;
    tile[freeDim] = shape[freeDim];

    return tile;
  }

  /// Project an iter-space tile (the form stored in LayoutEntry.tileSize)
  /// down to the producing op's result shape.  For elementwise the
  /// iter-space already matches the shape; for matmul we drop the
  /// trailing K entry; for reductions we drop reduction-iter entries
  /// (or clamp them to 1 in keepdims=True case).  Returns the original
  /// tile if the projection doesn't apply or the rank already matches.
  ///
  /// `iterTile`'s entries correspond 1-to-1 to the op's iterator types.
  static DenseI64ArrayAttr
  projectIterTileToValueShape(Value val, DenseI64ArrayAttr iterTile) {
    if (!iterTile)
      return iterTile;
    auto shapedType = dyn_cast<ShapedType>(val.getType());
    if (!shapedType || !shapedType.hasStaticShape())
      return iterTile;
    int64_t valueRank = shapedType.getRank();
    ArrayRef<int64_t> arr = iterTile.asArrayRef();

    Operation *defOp = val.getDefiningOp();
    auto linalgOp = defOp ? dyn_cast<linalg::LinalgOp>(defOp) : nullptr;

    // Matmul: iter-space is rank value+1 (trailing K); drop trailing.
    if (linalgOp && isMatmulOp(linalgOp) &&
        (int64_t)arr.size() == valueRank + 1) {
      SmallVector<int64_t> projected(arr.begin(), arr.end() - 1);
      return DenseI64ArrayAttr::get(val.getContext(), projected);
    }

    // Reduction: walk iter types; drop reduction entries (no keepdims),
    // or clamp them to 1 at size-1 axes (keepdims).  Note: the keepdims
    // case has the same rank as the iter-space, so we still need to run
    // through this to clamp the reduction-axis tiles to 1.
    if (linalgOp && isReductionGeneric(linalgOp)) {
      auto iterTypes = linalgOp.getIteratorTypesArray();
      if (arr.size() != iterTypes.size())
        return iterTile;
      ArrayRef<int64_t> shape = shapedType.getShape();
      bool keepdims = (int64_t)iterTypes.size() == valueRank;
      SmallVector<int64_t> projected;
      for (size_t i = 0; i < iterTypes.size(); i++) {
        if (iterTypes[i] == utils::IteratorType::reduction) {
          if (keepdims)
            projected.push_back(1);
          // else drop
        } else {
          projected.push_back(keepdims && shape[i] == 1 ? 1 : arr[i]);
        }
      }
      if ((int64_t)projected.size() == valueRank)
        return DenseI64ArrayAttr::get(val.getContext(), projected);
    }

    return iterTile;
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

    // tile_size in LayoutEntry has one entry per iterator (linalg iterator
    // order).  For elementwise it equals the value's shape; for a reduction
    // result it equals the input shape; for a matmul result it is rank
    // value+1 ([..., M, N, K]).
    //
    // Step 1: project the source tile down to source's value-shape so the
    // remaining propagation logic can operate on shape-rank tiles.
    DenseI64ArrayAttr projected =
        projectIterTileToValueShape(sourceVal, sourceLayout.tileSize);
    SmallVector<int64_t> tileValues(projected.asArrayRef());
    if (tileValues.size() != sourceShape.size())
      return false;

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

    // Step 2: lift tile up to the target's iter-space.
    // For a matmul target, iter-space rank == output rank + 1 (trailing K);
    // append the input's K dim (or its tile, if available).
    if (isMatmulOp(targetLinalgOp)) {
      Value lhs = targetLinalgOp.getDpsInputs()[0];
      auto lhsType = dyn_cast<ShapedType>(lhs.getType());
      if (lhsType && lhsType.hasStaticShape() && lhsType.getRank() >= 2) {
        int64_t kDim = lhsType.getShape()[lhsType.getRank() - 1];
        auto inputIt = annotatedValues.find(lhs);
        if (inputIt != annotatedValues.end() && inputIt->second.tileSize) {
          ArrayRef<int64_t> inputTile = inputIt->second.tileSize.asArrayRef();
          if (inputTile.size() == (size_t)lhsType.getRank())
            kDim = inputTile.back();
        }
        tileValues.push_back(std::min(kDim, maxPartitionDim()));
      }
    }
    // For a reduction target, iter-space rank == input rank; insert tiles
    // for the reduction iterators from the input shape (or input's tile, if
    // available).  isReductionGeneric() implies linalg.generic so this
    // doesn't overlap with the matmul branch above.
    else if (isReductionGeneric(targetLinalgOp)) {
      Value reductionInput = targetLinalgOp.getDpsInputs()[0];
      auto inType = dyn_cast<ShapedType>(reductionInput.getType());
      if (!inType || !inType.hasStaticShape())
        return false;
      ArrayRef<int64_t> inputShape = inType.getShape();
      ArrayRef<int64_t> inputTile;
      auto inputIt = annotatedValues.find(reductionInput);
      if (inputIt != annotatedValues.end() && inputIt->second.tileSize &&
          inputIt->second.tileSize.size() == (int64_t)inputShape.size())
        inputTile = inputIt->second.tileSize.asArrayRef();

      auto iterTypes = targetLinalgOp.getIteratorTypesArray();
      SmallVector<int64_t> lifted;
      size_t parallelIdx = 0;
      for (size_t i = 0; i < iterTypes.size(); i++) {
        if (iterTypes[i] == utils::IteratorType::reduction) {
          int64_t v = !inputTile.empty() && i < inputTile.size()
                          ? inputTile[i]
                          : inputShape[i];
          lifted.push_back(v);
        } else {
          if (parallelIdx >= tileValues.size())
            return false;
          lifted.push_back(tileValues[parallelIdx++]);
        }
      }
      tileValues = lifted;
    }

    outLayout.tileSize = DenseI64ArrayAttr::get(ctx, tileValues);
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
                  std::min(dims->M, maxPartitionDim()),
                  std::min(dims->K, maxPartitionDim())});
        outLayout.partitionDim = makePartitionDimAttr(ctx, 2);
      } else {
        // B (moving): [B, K, N], partition_dim=1 (K)
        outLayout.tileSize = DenseI64ArrayAttr::get(
            ctx, {1,
                  std::min(dims->K, maxPartitionDim()),
                  std::min(dims->N, matmulFreeDimCap())});
        outLayout.partitionDim = makePartitionDimAttr(ctx, 1);
      }
    } else if (operandIdx == 0) {
      // A (stationary): [M, K], partition_dim=1 (K)
      outLayout.tileSize = DenseI64ArrayAttr::get(
          ctx, {std::min(dims->M, maxPartitionDim()),
                std::min(dims->K, maxPartitionDim())});
      outLayout.partitionDim = makePartitionDimAttr(ctx, 1);
    } else {
      // B (moving): [K, N], partition_dim=0 (K)
      outLayout.tileSize = DenseI64ArrayAttr::get(
          ctx, {std::min(dims->K, maxPartitionDim()),
                std::min(dims->N, matmulFreeDimCap())});
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
    // Track which values already had user-created nkipy.layout/tile_op ops
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
    // Phase 1: Collect existing user annotations (nkipy.layout + nkipy.tile_op)
    // ================================================================
    // tile_op carries the iter-space tile (full input shape for reductions,
    // [..., M, N, K] for matmul, output shape for elementwise).  layout's
    // tile_size is the value-shape physical tile.  LayoutEntry tracks the
    // iter-space form, so prefer tile_op when both are present.
    DenseMap<Value, DenseI64ArrayAttr> loopTileSizeByValue;
    func.walk([&](nkipy::TileOp tileOp) {
      if (isInsideNkipyRegion(tileOp))
        return;
      if (auto lt = tileOp.getLoopTileSizeAttr())
        loopTileSizeByValue[tileOp.getTarget()] = lt;
    });

    func.walk([&](nkipy::LayoutOp layoutOp) {
      if (isInsideNkipyRegion(layoutOp))
        return;
      Value target = layoutOp.getTarget();
      DenseI64ArrayAttr tileSizeAttr;
      auto lt = loopTileSizeByValue.find(target);
      if (lt != loopTileSizeByValue.end())
        tileSizeAttr = lt->second;
      else
        tileSizeAttr = layoutOp.getTileSizeAttr();
      if (!tileSizeAttr)
        return;
      LayoutEntry entry;
      entry.tileSize = tileSizeAttr;
      entry.memSpace = layoutOp.getMemSpaceAttr();
      entry.partitionDim = layoutOp.getPartitionDimAttr();
      entry.seedId = nextSeedId++;
      annotatedValues[target] = entry;
      userAnnotatedValues.insert(target);
    });

    // Handle the case of a tile_op with no matching layout (loop-tile-only).
    for (auto &kv : loopTileSizeByValue) {
      if (annotatedValues.count(kv.first))
        continue;
      LayoutEntry entry;
      entry.tileSize = kv.second;
      entry.seedId = nextSeedId++;
      annotatedValues[kv.first] = entry;
      userAnnotatedValues.insert(kv.first);
    }

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

        // Seed result C with a matmul iter-space tile [M, N, K] (or
        // [B, M, N, K] for batch_matmul; bmm is decomposed before this
        // pass runs, so the batched branch is defensive).
        // partition_dim=0 (M for 2D, B for 3D — batch dim gets looped over).
        LayoutEntry cLayout;
        if (dims->isBatched()) {
          cLayout.tileSize = DenseI64ArrayAttr::get(
              ctx,
              {1,
               std::min(dims->M, maxPartitionDim()),
               std::min(dims->N, matmulFreeDimCap()),
               std::min(dims->K, maxPartitionDim())});
        } else {
          cLayout.tileSize = DenseI64ArrayAttr::get(
              ctx,
              {std::min(dims->M, maxPartitionDim()),
               std::min(dims->N, matmulFreeDimCap()),
               std::min(dims->K, maxPartitionDim())});
        }
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
          // tile_size is iter-space order: walk iterator types and emit one
          // entry per axis (parallel from output shape, reduction from input
          // shape).  Partition dim is the leading parallel iterator.
          auto genericOp = cast<linalg::GenericOp>(linalgOp.getOperation());
          auto iterTypes = genericOp.getIteratorTypesArray();

          Value reductionInput = linalgOp.getDpsInputs()[0];
          auto inputType = dyn_cast<ShapedType>(reductionInput.getType());

          SmallVector<int64_t> tile;
          size_t outIdx = 0;
          unsigned parallelCount = 0;
          for (size_t i = 0; i < iterTypes.size(); i++) {
            if (iterTypes[i] == utils::IteratorType::parallel) {
              int64_t dimSize = (outIdx < shape.size()) ? shape[outIdx] : 1;
              if (parallelCount == partDim)
                tile.push_back(std::min(dimSize, maxPartitionDim()));
              else
                tile.push_back(dimSize);
              outIdx++;
              parallelCount++;
            } else {
              int64_t redDimSize = (inputType && inputType.hasStaticShape() &&
                                    i < (size_t)inputType.getRank())
                                       ? inputType.getShape()[i]
                                       : 1;
              tile.push_back(redDimSize);
            }
          }

          layout.tileSize = DenseI64ArrayAttr::get(ctx, tile);
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
    // Phase 8: Materialize — create nkipy.layout + nkipy.tile_op ops for all
    //          newly-inferred layouts, and update user-annotated ops
    //          that had mem_space filled in.
    // ================================================================
    OpBuilder builder(ctx);
    for (auto &kv : annotatedValues) {
      Value val = kv.first;
      const LayoutEntry &layout = kv.second;

      // The layout op carries a value-shape tile; the tile_op carries the
      // iter-space tile.  These differ for matmul/reduction results.
      DenseI64ArrayAttr layoutTile =
          projectIterTileToValueShape(val, layout.tileSize);

      if (userAnnotatedValues.count(val)) {
        // User already supplied an annotation.  Update existing nkipy.layout
        // with the inferred mem_space / partition_dim / tile_size if they
        // were missing, or synthesize one if it wasn't written by the user.
        nkipy::LayoutOp existingLayout;
        for (Operation *user : val.getUsers()) {
          if (auto layoutOp = dyn_cast<nkipy::LayoutOp>(user)) {
            existingLayout = layoutOp;
            break;
          }
        }
        if (existingLayout) {
          if (!existingLayout.getMemSpace() && layout.memSpace)
            existingLayout.setMemSpaceAttr(layout.memSpace);
          if (!existingLayout.getTileSize() && layoutTile)
            existingLayout.setTileSizeAttr(layoutTile);
          if (!existingLayout.getPartitionDim() && layout.partitionDim)
            existingLayout.setPartitionDimAttr(layout.partitionDim);
        } else {
          // tile_op was written by the user, but no layout — synthesize one.
          builder.setInsertionPointAfterValue(val);
          builder.create<nkipy::LayoutOp>(
              val.getLoc(), val, layout.memSpace, layout.partitionDim,
              layoutTile);
        }
        // Ensure a tile_op exists carrying the iter-space tile
        // so KnobDrivenTiling has something to consume.
        bool hasTileOp = false;
        for (Operation *user : val.getUsers()) {
          if (isa<nkipy::TileOp>(user)) {
            hasTileOp = true;
            break;
          }
        }
        if (!hasTileOp && layout.tileSize) {
          builder.setInsertionPointAfterValue(val);
          builder.create<nkipy::TileOp>(
              val.getLoc(), val, layout.tileSize);
        }
        continue;
      }

      builder.setInsertionPointAfterValue(val);
      builder.create<nkipy::LayoutOp>(
          val.getLoc(), val, layout.memSpace, layout.partitionDim,
          layoutTile);
      if (layout.tileSize) {
        builder.create<nkipy::TileOp>(val.getLoc(), val, layout.tileSize);
      }
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
