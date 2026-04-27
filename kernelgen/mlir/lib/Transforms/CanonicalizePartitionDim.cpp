//===- CanonicalizePartitionDim.cpp - Ensure partition_dim=0 everywhere ----===//
//
// This pass inserts transposes so that partition_dim=0 holds for all annotated
// tensors.  NISA hardware assumes dimension 0 is always the partition
// dimension, and every downstream pass relies on this.
//
// Algorithm:
//   1. Collect all nkipy.annotate ops with partition_dim != 0.
//   2. For each such annotation, BFS through the connected elementwise
//      component to find all values that share the same non-zero partition_dim.
//   3. At component boundaries (inputs from non-elementwise producers, outputs
//      to non-elementwise consumers), insert linalg.transpose to move
//      partition_dim to position 0.
//   4. Rewrite all elementwise ops inside the component with permuted shapes.
//   5. Update all nkipy.annotate ops: partition_dim -> 0, permute tile_size.
//
// The pass runs BEFORE assign-linalg-op-ids so that new transpose ops get IDs
// (needed for knob-driven tiling). It runs AFTER infer-layout so that all
// tensors in the chain already have partition_dim annotations.
//
//===----------------------------------------------------------------------===//

#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/HardwareConstants.h"
#include "nkipy/Transforms/OpClassification.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Build the permutation that moves dimension `partDim` to position 0
/// and shifts the others right.  E.g. for rank=4, partDim=2:
///   [2, 0, 1, 3]
static SmallVector<int64_t> buildPermutation(int64_t rank, int64_t partDim) {
  SmallVector<int64_t> perm;
  perm.push_back(partDim);
  for (int64_t i = 0; i < rank; ++i) {
    if (i != partDim)
      perm.push_back(i);
  }
  return perm;
}

/// Build the inverse permutation.  E.g. for perm=[2,0,1,3]:
///   inv=[1,2,0,3]
static SmallVector<int64_t> invertPermutation(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> inv(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    inv[perm[i]] = i;
  return inv;
}

/// Apply a permutation to a vector.
template <typename T>
static SmallVector<T> permuteVector(ArrayRef<T> vec, ArrayRef<int64_t> perm) {
  SmallVector<T> result;
  for (int64_t p : perm)
    result.push_back(vec[p]);
  return result;
}

/// For >2D transposes: equalize tile sizes of swapped dim pairs so that
/// after tiling, the tiled transpose has ≤2 non-unit dims (required by
/// linalg-to-nisa, since NISA ops are 2D).
/// E.g. perm=[1,0,2], tile=[4,128,128] → tile=[4,4,128]
///
/// Exception: when one of the swapped dims already has tile=1, the
/// transpose along that pair is trivial (a reshape), so equalization
/// would be harmful — it would shrink the non-unit dim to 1 and create
/// tile size mismatches with downstream consumers.
static void equalizeSwappedTileDims(SmallVector<int64_t> &tile,
                                    ArrayRef<int64_t> perm) {
  for (int64_t i = 0; i < static_cast<int64_t>(tile.size()); ++i) {
    int64_t j = perm[i];
    if (j != i && j < static_cast<int64_t>(tile.size())) {
      if (tile[i] == 1 || tile[j] == 1)
        continue;
      int64_t minTile = std::min(tile[i], tile[j]);
      tile[i] = minTile;
      tile[j] = minTile;
    }
  }
}

/// Permute a reduced-rank tile_size (from a reduction op) through a
/// full-rank permutation. Expands to full rank using 1s for size-1 dims,
/// permutes, then strips back to parallel dims only.
static DenseI64ArrayAttr permuteReducedTileSize(
    ArrayRef<int64_t> oldTileSize, ArrayRef<int64_t> perm,
    ArrayRef<int64_t> invPerm, ArrayRef<int64_t> permutedShape,
    int64_t rank, MLIRContext *ctx) {
  // Recover original shape to find parallel dims.
  SmallVector<int64_t> origShape = permuteVector<int64_t>(permutedShape, invPerm);
  SmallVector<int64_t> origParDims;
  for (int64_t i = 0; i < rank; ++i) {
    if (origShape[i] > 1)
      origParDims.push_back(i);
  }

  if (static_cast<int64_t>(oldTileSize.size()) !=
      static_cast<int64_t>(origParDims.size()))
    return {};

  // Build full-rank tile with 1s for size-1 dims, then permute.
  SmallVector<int64_t> fullTile(rank, 1);
  for (size_t i = 0; i < origParDims.size(); ++i)
    fullTile[origParDims[i]] = oldTileSize[i];
  SmallVector<int64_t> permFull = permuteVector<int64_t>(fullTile, perm);

  // Strip back to only parallel dims in the permuted shape.
  SmallVector<int64_t> newTileSize;
  for (int64_t i = 0; i < rank; ++i) {
    if (permutedShape[i] > 1)
      newTileSize.push_back(permFull[i]);
  }
  return DenseI64ArrayAttr::get(ctx, newTileSize);
}

/// Emit an nkipy.annotate for a boundary transpose result.
/// Applies >2D tile equalization, then creates the annotation with
/// partition_dim=0 and the given mem_space/tile_size.
static void annotateBoundaryTranspose(OpBuilder &builder, Location loc,
                                      Value transposed,
                                      MemSpaceEnumAttr memSpace,
                                      DenseI64ArrayAttr tileSize,
                                      ArrayRef<int64_t> perm, int64_t rank) {
  DenseI64ArrayAttr finalTileSize;
  if (tileSize) {
    SmallVector<int64_t> tileSizeVec(tileSize.asArrayRef());
    if (rank > 2)
      equalizeSwappedTileDims(tileSizeVec, perm);
    finalTileSize = DenseI64ArrayAttr::get(builder.getContext(), tileSizeVec);
  }
  auto zeroPdAttr = builder.getIntegerAttr(
      builder.getIntegerType(32, /*isSigned=*/false), 0);
  builder.create<nkipy::AnnotateOp>(
      loc, transposed, memSpace, zeroPdAttr,
      finalTileSize, /*reduction_tile=*/DenseI64ArrayAttr{});
}

/// Wrappers that accept Operation* for use in BFS traversal where we
/// iterate over generic Operations rather than typed LinalgOps.
static bool isElementwiseOp(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  return linalgOp && ::mlir::nkipy::isElementwiseOp(linalgOp);
}

static bool isReductionGeneric(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  return linalgOp && ::mlir::nkipy::isReductionGeneric(linalgOp);
}

static bool isMatmulOp(Operation *op) {
  return ::mlir::nkipy::isMatmulOp(op->getName().getStringRef());
}

/// Get the partition_dim for a value from its nkipy.annotate op, if any.
/// Returns -1 if no annotation or no partition_dim.
static int64_t getPartitionDim(Value val,
                               DenseMap<Value, int64_t> &partDimMap) {
  auto it = partDimMap.find(val);
  if (it != partDimMap.end())
    return it->second;
  return -1;
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct NkipyCanonicalizePartitionDimPass
    : public CanonicalizePartitionDimBase<NkipyCanonicalizePartitionDimPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<nkipy::NkipyDialect>();
  }

  /// Convert ALL batch_matmul ops to scf.for + matmul.
  ///
  /// SBUF output: produces MxBxN (M at dim 0 = partition), so LegalizeLayout
  /// can properly expand it to physical format.
  ///
  /// HBM output (or no annotation): produces BxMxN (standard layout), same
  /// as the original preprocessBatchedOps in KnobDrivenTiling.
  ///
  /// Returns the set of SBUF converted output values (the for-loop results)
  /// so the partition_dim component processing can skip boundary
  /// transposes for these values (they are already in MxBxN order).
  DenseSet<Value> preprocessBatchMatmul(func::FuncOp func) {
    DenseSet<Value> convertedBmmOutputs;

    SmallVector<linalg::BatchMatmulOp> batchOps;
    func.walk([&](linalg::BatchMatmulOp op) {
      batchOps.push_back(op);
    });

    for (auto bmmOp : batchOps) {
      Value lhs = bmmOp.getInputs()[0];
      Value rhs = bmmOp.getInputs()[1];
      Value init = bmmOp.getOutputs()[0];

      auto initType = cast<RankedTensorType>(init.getType());
      if (initType.getRank() != 3 || !initType.hasStaticShape())
        continue;

      // Check if the output is annotated as SBUF.
      bool isSbuf = false;
      SmallVector<nkipy::AnnotateOp> annotateOps;
      for (auto *user : bmmOp.getResult(0).getUsers()) {
        if (auto ann = dyn_cast<nkipy::AnnotateOp>(user)) {
          annotateOps.push_back(ann);
          if (auto memSpace = ann.getMemSpaceAttr()) {
            if (memSpace.getValue() == nkipy::MemSpaceEnum::Sbuf)
              isSbuf = true;
          }
        }
      }

      Location loc = bmmOp.getLoc();
      // BxMxN shape
      int64_t B = initType.getShape()[0];
      int64_t M = initType.getShape()[1];
      int64_t N = initType.getShape()[2];
      Type elemTy = initType.getElementType();

      OpBuilder builder(bmmOp);

      // For SBUF: output is MxBxN (partition-correct, M at dim 0).
      //   Create a new MxBxN tensor.empty (no fill needed — NISA matmul
      //   auto-zeroes PSUM, so the init value is unused by hardware).
      // For HBM: output is BxMxN (standard layout).
      //   Reuse the original init tensor directly.
      Value loopInit;
      if (isSbuf) {
        loopInit = builder.create<tensor::EmptyOp>(
            loc, SmallVector<int64_t>{M, B, N}, elemTy);
      } else {
        loopInit = init;
      }

      // Create loop bounds: for %b = 0 to B step 1
      Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value cB = builder.create<arith::ConstantIndexOp>(loc, B);
      Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

      auto forOp = builder.create<scf::ForOp>(loc, c0, cB, c1,
                                               ValueRange{loopInit});

      // Build loop body
      builder.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      Value acc = forOp.getRegionIterArg(0);

      // Extract 2D slices from LHS and RHS (rank-reducing from BxMxK/BxKxN)
      auto extract2DFromBatch = [&](Value src) -> Value {
        auto srcType = cast<RankedTensorType>(src.getType());
        auto shape = srcType.getShape(); // [B, dim1, dim2]
        auto sliceType = RankedTensorType::get(
            {shape[1], shape[2]}, srcType.getElementType());

        SmallVector<OpFoldResult> offsets = {
            iv, builder.getIndexAttr(0), builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes = {
            builder.getIndexAttr(1),
            builder.getIndexAttr(shape[1]),
            builder.getIndexAttr(shape[2])};
        SmallVector<OpFoldResult> strides = {
            builder.getIndexAttr(1), builder.getIndexAttr(1),
            builder.getIndexAttr(1)};

        return builder.create<tensor::ExtractSliceOp>(
            loc, sliceType, src, offsets, sizes, strides);
      };

      Value lhsSlice = extract2DFromBatch(lhs);
      Value rhsSlice = extract2DFromBatch(rhs);

      // Extract init slice from accumulator and run matmul.
      auto mmType = RankedTensorType::get({M, N}, elemTy);
      SmallVector<OpFoldResult> extractOffsets, extractSizes;
      SmallVector<OpFoldResult> extractStrides = {
          builder.getIndexAttr(1), builder.getIndexAttr(1),
          builder.getIndexAttr(1)};

      Value inserted;
      linalg::MatmulOp matmulOp;
      if (isSbuf) {
        // SBUF (MxBxN): use a separate 2D tensor for matmul output.
        // We cannot use a collapse_shape view of the 3D acc because after
        // LegalizeLayout expands the 3D alloc to 5D, the collapsed strides
        // produce an interleaved 2D view that corrupts the matmul output.
        // Instead: matmul → separate 2D → rank-expanding insert_slice into 3D.
        Value mmInit = builder.create<tensor::EmptyOp>(
            loc, SmallVector<int64_t>{M, N}, elemTy);

        matmulOp = builder.create<linalg::MatmulOp>(
            loc, TypeRange{mmType},
            ValueRange{lhsSlice, rhsSlice}, ValueRange{mmInit});

        // Expand 2D [M, N] → 3D [M, 1, N] so the insert_slice is same-rank.
        // This avoids rank-reducing subviews after bufferization, which
        // LegalizeLayout's tileCopyAndTranspose cannot handle.
        auto expandedType = RankedTensorType::get({M, 1, N}, elemTy);
        SmallVector<ReassociationIndices> reassoc = {{0}, {1, 2}};
        Value expanded = builder.create<tensor::ExpandShapeOp>(
            loc, expandedType, matmulOp.getResult(0), reassoc);

        // Same-rank insert: 3D [M, 1, N] → 3D acc at [0, b, 0] [M, 1, N]
        extractOffsets = {builder.getIndexAttr(0), iv, builder.getIndexAttr(0)};
        extractSizes = {builder.getIndexAttr(M), builder.getIndexAttr(1),
                        builder.getIndexAttr(N)};
        inserted = builder.create<tensor::InsertSliceOp>(
            loc, expanded, acc,
            extractOffsets, extractSizes, extractStrides);
      } else {
        // HBM (BxMxN): rank-reducing extract [%b,0,0][1,M,N] → MxN
        extractOffsets = {iv, builder.getIndexAttr(0), builder.getIndexAttr(0)};
        extractSizes = {builder.getIndexAttr(1), builder.getIndexAttr(M),
                        builder.getIndexAttr(N)};

        Value initSlice = builder.create<tensor::ExtractSliceOp>(
            loc, mmType, acc, extractOffsets, extractSizes, extractStrides);

        matmulOp = builder.create<linalg::MatmulOp>(
            loc, TypeRange{mmType},
            ValueRange{lhsSlice, rhsSlice}, ValueRange{initSlice});

        inserted = builder.create<tensor::InsertSliceOp>(
            loc, matmulOp.getResult(0), acc,
            extractOffsets, extractSizes, extractStrides);
      }

      // Copy nkipy.op_id attribute
      if (auto opIdAttr = bmmOp->getAttrOfType<IntegerAttr>("nkipy.op_id"))
        matmulOp->setAttr("nkipy.op_id", opIdAttr);

      builder.create<scf::YieldOp>(loc, ValueRange{inserted});

      // Handle annotations: create separate annotations for the matmul
      // result and the 3D accumulator.
      for (auto ann : annotateOps) {
        // 2D tile for matmul: drop batch dim from original tile_size
        DenseI64ArrayAttr newTileSize;
        if (auto ts = ann.getTileSizeAttr()) {
          auto arr = ts.asArrayRef();
          if (arr.size() >= 2) {
            SmallVector<int64_t> adjusted(arr.begin() + 1, arr.end());
            newTileSize = DenseI64ArrayAttr::get(
                func.getContext(), adjusted);
          }
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(matmulOp);

        if (isSbuf) {
          // Matmul output: annotate as SharedHbm so LegalizeLayout won't
          // touch it (it's a temporary that gets copied into the 3D SBUF acc).
          auto hbmAttr = nkipy::MemSpaceEnumAttr::get(
              func.getContext(), nkipy::MemSpaceEnum::SharedHbm);
          builder.create<nkipy::AnnotateOp>(
              ann.getLoc(), matmulOp.getResult(0),
              hbmAttr, /*partitionDim=*/IntegerAttr{},
              newTileSize, ann.getReductionTileAttr());

          // 3D accumulator: annotate as SBUF with 3D tile (partition, batch, free).
          // Tile is [M_tile, 1, N_tile] from the original [B_tile, M_tile, N_tile].
          DenseI64ArrayAttr accTileSize;
          if (auto ts = ann.getTileSizeAttr()) {
            auto arr = ts.asArrayRef();
            if (arr.size() >= 3) {
              // Original: [B, M, N] → transposed: [M, B, N] → tile: [M_tile, B_tile, N_tile]
              SmallVector<int64_t> accTile = {arr[1], arr[0], arr[2]};
              accTileSize = DenseI64ArrayAttr::get(func.getContext(), accTile);
            }
          }
          builder.setInsertionPointAfter(forOp);
          builder.create<nkipy::AnnotateOp>(
              ann.getLoc(), forOp.getResult(0),
              ann.getMemSpaceAttr(), /*partitionDim=*/IntegerAttr{},
              accTileSize, /*reductionTile=*/DenseI64ArrayAttr{});
        } else {
          // HBM: annotate matmul result with original mem_space
          builder.create<nkipy::AnnotateOp>(
              ann.getLoc(), matmulOp.getResult(0),
              ann.getMemSpaceAttr(), /*partitionDim=*/IntegerAttr{},
              newTileSize, ann.getReductionTileAttr());
        }

        ann.erase();
      }

      // Replace all uses of batch_matmul result with for loop result.
      Value forResult = forOp.getResult(0);

      // Collect non-annotate uses before replacing
      SmallVector<OpOperand *> usesToReplace;
      for (OpOperand &use : bmmOp.getResult(0).getUses()) {
        if (!isa<nkipy::AnnotateOp>(use.getOwner()))
          usesToReplace.push_back(&use);
      }

      // Replace uses
      for (OpOperand *use : usesToReplace)
        use->set(forResult);

      // Erase remaining annotate uses and the bmm op
      for (auto *user : llvm::make_early_inc_range(
               bmmOp.getResult(0).getUsers())) {
        if (isa<nkipy::AnnotateOp>(user))
          user->erase();
      }
      bmmOp.erase();

      if (isSbuf) {
        convertedBmmOutputs.insert(forResult);
        llvm::errs() << "[CanonicalizePartitionDim] Converted batch_matmul "
                     << "(B=" << B << ", M=" << M << ", N=" << N
                     << ") to loop + matmul with MxBxN output (SBUF)\n";
      } else {
        llvm::errs() << "[CanonicalizePartitionDim] Converted batch_matmul "
                     << "(B=" << B << ", M=" << M << ", N=" << N
                     << ") to loop + matmul with BxMxN output (HBM)\n";
      }
    }

    return convertedBmmOutputs;
  }

  /// Collect annotations with partition_dim info from the function.
  void collectAnnotations(
      func::FuncOp func,
      DenseMap<Value, int64_t> &partDimMap,
      DenseMap<Value, nkipy::AnnotateOp> &valueAnnotateMap,
      SmallVector<nkipy::AnnotateOp> &nonZeroAnnotations) {
    func.walk([&](nkipy::AnnotateOp annotateOp) {
      valueAnnotateMap[annotateOp.getTarget()] = annotateOp;
      auto partDimAttr = annotateOp.getPartitionDimAttr();
      if (!partDimAttr)
        return;
      uint32_t partDim = partDimAttr.getUInt();
      partDimMap[annotateOp.getTarget()] = partDim;
      if (partDim != 0)
        nonZeroAnnotations.push_back(annotateOp);
    });
  }

  /// BFS from seedOp to find connected elementwise/reduction component.
  llvm::SetVector<Operation *> findComponent(Operation *seedOp) {
    llvm::SetVector<Operation *> componentOps;

    auto canInclude = [](Operation *op) {
      return isElementwiseOp(op) || isReductionGeneric(op);
    };

    if (!seedOp || !canInclude(seedOp))
      return componentOps;

    SmallVector<Operation *> bfsQueue;
    bfsQueue.push_back(seedOp);
    componentOps.insert(seedOp);

    while (!bfsQueue.empty()) {
      Operation *op = bfsQueue.pop_back_val();

      // Backward through DPS inputs.
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        for (Value input : linalgOp.getDpsInputs()) {
          Operation *producer = input.getDefiningOp();
          if (producer && canInclude(producer) &&
              !componentOps.count(producer)) {
            componentOps.insert(producer);
            bfsQueue.push_back(producer);
          }
        }
      }

      // Forward through uses.
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (isa<nkipy::AnnotateOp>(user))
            continue;
          if (canInclude(user) && !componentOps.count(user)) {
            componentOps.insert(user);
            bfsQueue.push_back(user);
          }
        }
      }
    }

    return componentOps;
  }

  /// Find boundary inputs: values used by component ops but defined outside.
  /// Skips tensor.empty and linalg.fill (recreated with permuted shapes).
  llvm::SetVector<Value> findBoundaryInputs(
      const llvm::SetVector<Operation *> &componentOps) {
    llvm::SetVector<Value> boundaryInputs;
    for (Operation *op : componentOps) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp || !componentOps.count(defOp)) {
          if (defOp && (isa<tensor::EmptyOp>(defOp) ||
                        isa<linalg::FillOp>(defOp)))
            continue;
          boundaryInputs.insert(operand);
        }
      }
    }
    return boundaryInputs;
  }

  /// Find boundary outputs: results of component ops used outside.
  llvm::SetVector<Value> findBoundaryOutputs(
      const llvm::SetVector<Operation *> &componentOps) {
    llvm::SetVector<Value> boundaryOutputs;
    for (Operation *op : componentOps) {
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (isa<nkipy::AnnotateOp>(user))
            continue;
          if (!componentOps.count(user)) {
            boundaryOutputs.insert(result);
            break;
          }
        }
      }
    }
    return boundaryOutputs;
  }

  /// Insert input boundary transposes (original -> permuted).
  void insertInputTransposes(
      OpBuilder &builder, const llvm::SetVector<Value> &boundaryInputs,
      ArrayRef<int64_t> perm, int64_t rank,
      DenseI64ArrayAttr seedTileSizeAttr,
      const DenseSet<Value> &convertedBmmOutputs,
      IRMapping &valueMapping) {
    for (Value input : boundaryInputs) {
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputType || inputType.getRank() != rank)
        continue;

      if (convertedBmmOutputs.count(input)) {
        valueMapping.map(input, input);
        llvm::errs() << "[CanonicalizePartitionDim] Skipping boundary "
                     << "transpose for converted BMM output (already MxBxN)\n";
        continue;
      }

      SmallVector<int64_t> newShape =
          permuteVector<int64_t>(inputType.getShape(), perm);

      if (input.getDefiningOp())
        builder.setInsertionPointAfter(input.getDefiningOp());
      else
        builder.setInsertionPointToStart(input.getParentBlock());

      Location loc = input.getLoc();
      Value init = builder.create<tensor::EmptyOp>(
          loc, newShape, inputType.getElementType());
      auto transposeOp =
          builder.create<linalg::TransposeOp>(loc, input, init, perm);
      Value transposed = transposeOp.getResult()[0];
      valueMapping.map(input, transposed);

      DenseI64ArrayAttr transposeTileSize;
      if (seedTileSizeAttr) {
        SmallVector<int64_t> permutedTile =
            permuteVector<int64_t>(seedTileSizeAttr.asArrayRef(), perm);
        transposeTileSize =
            DenseI64ArrayAttr::get(builder.getContext(), permutedTile);
      }
      auto sbufAttr = nkipy::MemSpaceEnumAttr::get(
          builder.getContext(), nkipy::MemSpaceEnum::Sbuf);
      annotateBoundaryTranspose(builder, loc, transposed,
                                sbufAttr, transposeTileSize, perm, rank);
    }
  }

  /// Rewrite component ops with permuted shapes.
  void rewriteComponentOps(
      OpBuilder &builder, func::FuncOp func,
      const llvm::SetVector<Operation *> &componentOps,
      ArrayRef<int64_t> perm, int64_t rank,
      IRMapping &valueMapping) {
    // Process in topological order.
    SmallVector<Operation *> topoOrder;
    func.walk([&](Operation *op) {
      if (componentOps.count(op))
        topoOrder.push_back(op);
    });

    for (Operation *op : topoOrder) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp)
        continue;

      // Replace operands with mapped (transposed) values.
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        if (Value mapped = valueMapping.lookupOrNull(op->getOperand(i)))
          op->setOperand(i, mapped);
      }

      // Recreate init operands with permuted shape.
      for (auto [idx, initOperand] :
           llvm::enumerate(linalgOp.getDpsInits())) {
        if (auto emptyOp = initOperand.getDefiningOp<tensor::EmptyOp>()) {
          auto emptyType =
              dyn_cast<RankedTensorType>(emptyOp.getResult().getType());
          if (!emptyType || emptyType.getRank() != rank)
            continue;
          SmallVector<int64_t> newShape =
              permuteVector<int64_t>(emptyType.getShape(), perm);
          builder.setInsertionPoint(emptyOp);
          auto newEmpty = builder.create<tensor::EmptyOp>(
              emptyOp.getLoc(), newShape, emptyType.getElementType());
          emptyOp.getResult().replaceAllUsesWith(newEmpty.getResult());
          emptyOp->erase();
          continue;
        }

        if (auto fillOp = dyn_cast<linalg::FillOp>(
                initOperand.getDefiningOp())) {
          auto fillEmptyOp =
              fillOp.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
          if (!fillEmptyOp)
            continue;
          auto emptyType = dyn_cast<RankedTensorType>(
              fillEmptyOp.getResult().getType());
          if (!emptyType || emptyType.getRank() != rank)
            continue;
          SmallVector<int64_t> newShape =
              permuteVector<int64_t>(emptyType.getShape(), perm);
          builder.setInsertionPoint(fillEmptyOp);
          auto newEmpty = builder.create<tensor::EmptyOp>(
              fillEmptyOp.getLoc(), newShape, emptyType.getElementType());
          fillEmptyOp.getResult().replaceAllUsesWith(newEmpty.getResult());
          fillEmptyOp->erase();
          fillOp->getResult(0).setType(
              RankedTensorType::get(newShape, emptyType.getElementType()));
        }
      }

      // Update result types.
      for (Value result : op->getResults()) {
        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        if (!resultType || resultType.getRank() != rank)
          continue;
        SmallVector<int64_t> newShape =
            permuteVector<int64_t>(resultType.getShape(), perm);
        result.setType(
            RankedTensorType::get(newShape, resultType.getElementType()));
      }

      // Permute indexing maps of linalg.generic ops.
      // Named ops (add, mul, exp, etc.) have implicit identity maps that
      // remain valid after consistent shape permutation.  But generic ops
      // may have non-identity maps (e.g. broadcast: (d0,d1,d2)->(0,d1,d2))
      // that must be permuted to match the new shape layout.
      if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        MLIRContext *ctx = genericOp.getContext();
        SmallVector<int64_t> invPerm = invertPermutation(perm);

        // Build dimension remapping: d_j -> d_{invPerm[j]}.
        SmallVector<AffineExpr> dimReplacements;
        for (int64_t j = 0; j < rank; ++j)
          dimReplacements.push_back(getAffineDimExpr(invPerm[j], ctx));

        SmallVector<AffineMap> newMaps;
        for (AffineMap map : genericOp.getIndexingMapsArray()) {
          unsigned numResults = map.getNumResults();

          SmallVector<AffineExpr> exprs;
          if (static_cast<int64_t>(numResults) == rank) {
            // Full-rank map: reorder result positions by perm, then
            // remap dimension references.
            for (int64_t i = 0; i < rank; ++i)
              exprs.push_back(map.getResult(perm[i]));
          } else {
            // Reduced-rank map (e.g. reduction output): keep result
            // order, only remap dimension references.
            for (unsigned i = 0; i < numResults; ++i)
              exprs.push_back(map.getResult(i));
          }

          SmallVector<AffineExpr> finalExprs;
          for (AffineExpr expr : exprs)
            finalExprs.push_back(
                expr.replaceDimsAndSymbols(dimReplacements, {}));

          newMaps.push_back(
              AffineMap::get(map.getNumDims(), 0, finalExprs, ctx));
        }
        genericOp.setIndexingMapsAttr(
            builder.getAffineMapArrayAttr(newMaps));
      }
    }
  }

  /// Insert output boundary transposes (permuted -> original) and rewire uses.
  void insertOutputTransposes(
      OpBuilder &builder,
      const llvm::SetVector<Value> &boundaryOutputs,
      const llvm::SetVector<Operation *> &componentOps,
      ArrayRef<int64_t> invPerm, int64_t rank,
      DenseI64ArrayAttr seedTileSizeAttr,
      DenseMap<Value, nkipy::AnnotateOp> &valueAnnotateMap,
      nkipy::AnnotateOp annotateOp, int64_t partDim) {
    for (Value output : boundaryOutputs) {
      auto outputType = dyn_cast<RankedTensorType>(output.getType());
      if (!outputType)
        continue;

      SmallVector<int64_t> origShape =
          permuteVector<int64_t>(outputType.getShape(), invPerm);

      builder.setInsertionPointAfterValue(output);
      Location loc = output.getLoc();
      Value init = builder.create<tensor::EmptyOp>(
          loc, origShape, outputType.getElementType());
      auto transposeOp =
          builder.create<linalg::TransposeOp>(loc, output, init, invPerm);
      Value transposedBack = transposeOp.getResult()[0];

      // Derive tile_size and mem_space for the output annotation.
      DenseI64ArrayAttr outputTileSize;
      MemSpaceEnumAttr outputMemSpace;
      auto annIt = valueAnnotateMap.find(output);
      if (annIt != valueAnnotateMap.end()) {
        outputMemSpace = annIt->second.getMemSpaceAttr();
        outputTileSize = annIt->second.getTileSizeAttr();
      }
      if (!outputTileSize && seedTileSizeAttr)
        outputTileSize = seedTileSizeAttr;

      // Expand reduced-rank tile_size to full rank.
      if (outputTileSize &&
          static_cast<int64_t>(outputTileSize.size()) < rank) {
        SmallVector<int64_t> expanded;
        ArrayRef<int64_t> reduced = outputTileSize.asArrayRef();
        size_t ri = 0;
        for (int64_t i = 0; i < rank; ++i) {
          if (origShape[i] > 1 && ri < reduced.size())
            expanded.push_back(reduced[ri++]);
          else
            expanded.push_back(1);
        }
        outputTileSize =
            DenseI64ArrayAttr::get(builder.getContext(), expanded);
      }

      // Clamp dim 0 to MAX_PARTITION_DIM.
      if (outputTileSize) {
        SmallVector<int64_t> tileSizeVec(outputTileSize.asArrayRef());
        if (tileSizeVec[0] > MAX_PARTITION_DIM) {
          annotateOp.emitWarning("partition_dim=")
              << partDim << ": clamping boundary transpose tile_size[0] "
              << "from " << tileSizeVec[0] << " to " << MAX_PARTITION_DIM;
          tileSizeVec[0] = MAX_PARTITION_DIM;
        }
        outputTileSize =
            DenseI64ArrayAttr::get(builder.getContext(), tileSizeVec);
      }

      annotateBoundaryTranspose(builder, loc, transposedBack,
                                outputMemSpace, outputTileSize,
                                invPerm, rank);

      // Rewire non-component uses to the inverse-transposed value.
      SmallVector<OpOperand *> usesToReplace;
      for (OpOperand &use : output.getUses()) {
        Operation *user = use.getOwner();
        if (user == transposeOp || isa<nkipy::AnnotateOp>(user))
          continue;
        if (!componentOps.count(user))
          usesToReplace.push_back(&use);
      }
      for (OpOperand *use : usesToReplace)
        use->set(transposedBack);
    }
  }

  /// Update nkipy.annotate ops for values in the component:
  /// set partition_dim=0 and permute tile_size.
  void updateComponentAnnotations(
      OpBuilder &builder, func::FuncOp func,
      const llvm::SetVector<Operation *> &componentOps,
      ArrayRef<int64_t> perm, ArrayRef<int64_t> invPerm, int64_t rank) {
    func.walk([&](nkipy::AnnotateOp annOp) {
      Value annTarget = annOp.getTarget();
      Operation *defOp = annTarget.getDefiningOp();
      if (!defOp || !componentOps.count(defOp))
        return;

      annOp.setPartitionDimAttr(builder.getIntegerAttr(
          builder.getIntegerType(32, /*isSigned=*/false), 0));

      if (auto tileSizeAttr = annOp.getTileSizeAttr()) {
        ArrayRef<int64_t> oldTileSize = tileSizeAttr.asArrayRef();
        if (static_cast<int64_t>(oldTileSize.size()) == rank) {
          SmallVector<int64_t> newTileSize =
              permuteVector<int64_t>(oldTileSize, perm);
          annOp.setTileSizeAttr(
              DenseI64ArrayAttr::get(builder.getContext(), newTileSize));
        } else {
          auto annTargetType =
              dyn_cast<RankedTensorType>(annTarget.getType());
          if (annTargetType) {
            auto result = permuteReducedTileSize(
                oldTileSize, perm, invPerm,
                annTargetType.getShape(), rank, builder.getContext());
            if (result)
              annOp.setTileSizeAttr(result);
          }
        }
      }
    });
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::errs() << "[CanonicalizePartitionDim] Processing function: "
                 << func.getName() << "\n";

    // Phase 0: Convert batch_matmul to loop + matmul.
    DenseSet<Value> convertedBmmOutputs = preprocessBatchMatmul(func);

    // Phase 1: Collect partition_dim annotations.
    DenseMap<Value, int64_t> partDimMap;
    DenseMap<Value, nkipy::AnnotateOp> valueAnnotateMap;
    SmallVector<nkipy::AnnotateOp> nonZeroAnnotations;
    collectAnnotations(func, partDimMap, valueAnnotateMap, nonZeroAnnotations);

    if (nonZeroAnnotations.empty()) {
      llvm::errs() << "[CanonicalizePartitionDim] No non-zero partition_dim "
                      "annotations found\n";
      return;
    }

    // Phase 2: Process each non-zero partition_dim component.
    OpBuilder builder(func.getContext());
    int numTransposed = 0;
    DenseSet<Operation *> processedOps;

    for (nkipy::AnnotateOp annotateOp : nonZeroAnnotations) {
      Value target = annotateOp.getTarget();
      Operation *seedOp = target.getDefiningOp();

      if (seedOp && processedOps.count(seedOp))
        continue;

      auto tensorType = dyn_cast<RankedTensorType>(target.getType());
      if (!tensorType) {
        annotateOp.emitError("partition_dim != 0 on non-tensor type");
        return signalPassFailure();
      }

      int64_t partDim = partDimMap[target];
      int64_t rank = tensorType.getRank();
      if (partDim >= rank) {
        annotateOp.emitError("partition_dim ")
            << partDim << " >= tensor rank " << rank;
        return signalPassFailure();
      }

      SmallVector<int64_t> perm = buildPermutation(rank, partDim);
      SmallVector<int64_t> invPerm = invertPermutation(perm);
      auto seedTileSizeAttr = annotateOp.getTileSizeAttr();

      // Validate partition tile size fits hardware.
      if (seedTileSizeAttr) {
        ArrayRef<int64_t> tileVals = seedTileSizeAttr.asArrayRef();
        if (static_cast<int64_t>(tileVals.size()) > partDim &&
            tileVals[partDim] > MAX_PARTITION_DIM) {
          annotateOp.emitError("tile_size[")
              << partDim << "] = " << tileVals[partDim]
              << " exceeds hardware partition limit " << MAX_PARTITION_DIM;
          return signalPassFailure();
        }
      }

      // Skip non-linalg ops (partition_dim is informational only).
      if (seedOp && !isa<linalg::LinalgOp>(seedOp)) {
        processedOps.insert(seedOp);
        continue;
      }

      // Error on matmul with partition_dim != 0.
      if (seedOp && isMatmulOp(seedOp)) {
        annotateOp.emitError(
            "partition_dim != 0 on matmul/bmm is not supported. "
            "Please annotate downstream elementwise ops instead.");
        return signalPassFailure();
      }

      // Step 1: Find connected component.
      auto componentOps = findComponent(seedOp);
      for (Operation *op : componentOps)
        processedOps.insert(op);

      if (componentOps.empty()) {
        annotateOp.emitError(
            "partition_dim != 0 on an unsupported op. "
            "Supported: elementwise and reduction ops.");
        return signalPassFailure();
      }

      // Step 2: Find boundaries.
      auto boundaryInputs = findBoundaryInputs(componentOps);
      auto boundaryOutputs = findBoundaryOutputs(componentOps);

      // Step 3: Insert input boundary transposes.
      IRMapping valueMapping;
      insertInputTransposes(builder, boundaryInputs, perm, rank,
                            seedTileSizeAttr, convertedBmmOutputs,
                            valueMapping);

      // Step 4: Rewrite component ops with permuted shapes.
      rewriteComponentOps(builder, func, componentOps, perm, rank,
                          valueMapping);

      // Step 5: Insert output boundary transposes.
      insertOutputTransposes(builder, boundaryOutputs, componentOps,
                             invPerm, rank, seedTileSizeAttr,
                             valueAnnotateMap, annotateOp, partDim);

      // Step 6: Update annotations.
      updateComponentAnnotations(builder, func, componentOps,
                                 perm, invPerm, rank);

      numTransposed += componentOps.size();
      llvm::errs() << "[CanonicalizePartitionDim] Processed component of "
                   << componentOps.size() << " ops with partition_dim="
                   << partDim << "\n";
    }

    llvm::errs() << "[CanonicalizePartitionDim] Rewritten " << numTransposed
                 << " op(s) total\n";
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createCanonicalizePartitionDimPass() {
  return std::make_unique<NkipyCanonicalizePartitionDimPass>();
}

} // namespace nkipy
} // namespace mlir
