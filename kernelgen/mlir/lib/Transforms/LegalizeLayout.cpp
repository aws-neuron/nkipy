//===- LegalizeLayout.cpp - Legalize SBUF tensor layouts ------------------===//
//
// This pass transforms SBUF tensor layouts to satisfy NKI hardware constraints
// where the first dimension (partition dimension) must be ≤128.
//
// The pass identifies SBUF tensors via:
// 1. bufferization.alloc_tensor with memory_space = #nisa.mem<sbuf>
// 2. nkipy.annotate ops with mem_space = Sbuf (traced back to tensor.empty)
//
// For each SBUF tensor needing legalization, the pass:
// 1. Computes the target (R+2)-D physical shape based on tile_size annotation
// 2. Propagates the shape change through the entire use-def chain via BFS
// 3. Updates scf.for init_args, block args, and results
// 4. Transforms extract_slice to (R+2)-D indexing + collapse_shape
// 5. Transforms insert_slice with expand_shape + (R+2)-D indexing
//
// Prerequisites:
// - Runs after knob-driven-tiling + transform-interpreter
// - Runs after canonicalize-loop-step (loops have step=1)
//
//===----------------------------------------------------------------------===//

#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/HardwareConstants.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <queue>

#define DEBUG_TYPE "legalize-layout"

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {

namespace {

/// Structure to hold layout transformation info for an SBUF tensor.
///
/// For a rank-R tensor [d_0, ..., d_{R-1}] with tile [t_0, ..., t_{R-1}]:
///   numBlocks[i] = d_i / t_i
///   Physical shape (rank R+2): [t_0, numBlocks[0], ..., numBlocks[R-1], t_{R-1}]
///
/// Key constraint: only dim 0 (partition) and dim R-1 (free) may have tile > 1.
/// All middle dims (1..R-2) must have tile = 1. This means single-block subviews
/// have shape [partTile, 1, ..., 1, freeTile] which can be collapsed directly to
/// 2D [partTile, freeTile] for NISA consumption.
///
/// For R=2: physical is 4D [t0, nB0, nB1, t1], collapse [[0,1,2],[3]] → 2D.
struct LayoutInfo {
  Value originalValue;              // The original tensor allocation
  SmallVector<int64_t> origShape;   // [d_0, ..., d_{R-1}]
  SmallVector<int64_t> tileSize;    // [t_0, ..., t_{R-1}]
  SmallVector<int64_t> numBlocks;   // [d_0/t_0, ..., d_{R-1}/t_{R-1}]

  int64_t rank() const { return origShape.size(); }
  int64_t physicalRank() const { return rank() + 2; }

  /// Physical shape: [tileSize[0], numBlocks[0], ..., numBlocks[R-1], tileSize[R-1]]
  SmallVector<int64_t> getPhysicalShape() const {
    SmallVector<int64_t> shape;
    shape.push_back(tileSize[0]);
    for (auto nb : numBlocks)
      shape.push_back(nb);
    shape.push_back(tileSize.back());
    return shape;
  }

  /// Collapse reassociation from (R+2)-D physical layout directly to 2D:
  /// [[0, 1, ..., R], [R+1]] → [partTile, freeTile]
  ///
  /// This works for single-block subviews where all numBlocks dims are 1.
  /// For R=2: [[0,1,2], [3]] on [128,1,1,128] → [128, 128] (same as legacy).
  SmallVector<ReassociationIndices> getCollapseReassociation() const {
    int64_t R = rank();
    SmallVector<ReassociationIndices> reassoc;
    ReassociationIndices group0;
    for (int64_t i = 0; i <= R; i++)
      group0.push_back(i);             // partTile + all numBlocks dims
    reassoc.push_back(group0);
    reassoc.push_back({R + 1});         // freeTile
    return reassoc;
  }

};

/// Build collapse reassociation from (R+2)-D to 2D: [[0, 1, ..., R], [R+1]]
///
/// For single-block subviews with shape [partTile, 1, ..., 1, freeTile],
/// this produces [partTile, freeTile] (2D) suitable for NISA ops.
static SmallVector<ReassociationIndices> build2DCollapseFromPhysical(
    int64_t physicalRank) {
  int64_t R = physicalRank - 2;  // logical rank
  SmallVector<ReassociationIndices> reassoc;
  ReassociationIndices group0;
  for (int64_t i = 0; i <= R; i++)
    group0.push_back(i);
  reassoc.push_back(group0);
  reassoc.push_back({R + 1});
  return reassoc;
}

/// Build collapse reassociation from R-D to 2D: [[0, 1, ..., R-2], [R-1]]
///
/// For R-D tiles with shape [partTile, 1, ..., 1, freeTile] (where middle
/// dims have tile=1), this produces [partTile, freeTile] (2D).
/// For R=2 this is [[0], [1]] (identity, no-op).
static SmallVector<ReassociationIndices> build2DCollapseFromLogical(
    int64_t logicalRank) {
  SmallVector<ReassociationIndices> reassoc;
  ReassociationIndices group0;
  for (int64_t i = 0; i < logicalRank - 1; i++)
    group0.push_back(i);
  reassoc.push_back(group0);
  reassoc.push_back({logicalRank - 1});
  return reassoc;
}

/// Result of createBlockLoopNest: IVs and the builder insertion point is
/// set to the innermost loop body.
struct BlockLoopNest {
  SmallVector<Value> ivs; // One IV per logical dim
};

/// Create an R-level scf.for loop nest iterating over block indices.
/// After this call, builder insertion point is inside the innermost loop body.
static BlockLoopNest createBlockLoopNest(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> numBlocks) {
  BlockLoopNest result;
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  for (int64_t nb : numBlocks) {
    Value ub = builder.create<arith::ConstantIndexOp>(loc, nb);
    auto loop = builder.create<scf::ForOp>(loc, c0, ub, c1);
    builder.setInsertionPointToStart(loop.getBody());
    result.ivs.push_back(loop.getInductionVar());
  }
  return result;
}

/// Create a single-tile subview of a physical (R+2)-D SBUF buffer and
/// collapse it to 2D [partTile, freeTile].
///
/// Physical layout: [partTile, numBlocks[0], ..., numBlocks[R-1], freeTile]
/// Subview:         [0, iv0, ..., ivR-1, 0] / [partTile, 1, ..., 1, freeTile]
/// Collapse:        [[0, 1, ..., R], [R+1]] → [partTile, freeTile]
static Value createTileSubviewAndCollapse(
    OpBuilder &builder, Location loc, Value physBuf,
    int64_t partTile, int64_t freeTile, int64_t R,
    ArrayRef<Value> blockIVs) {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  offsets.push_back(builder.getIndexAttr(0));
  for (Value iv : blockIVs)
    offsets.push_back(OpFoldResult(iv));
  offsets.push_back(builder.getIndexAttr(0));

  sizes.push_back(builder.getIndexAttr(partTile));
  for (int64_t i = 0; i < R; i++)
    sizes.push_back(builder.getIndexAttr(1));
  sizes.push_back(builder.getIndexAttr(freeTile));

  for (int64_t i = 0; i < R + 2; i++)
    strides.push_back(builder.getIndexAttr(1));

  auto subview = builder.create<memref::SubViewOp>(
      loc, physBuf, offsets, sizes, strides);
  return builder.create<memref::CollapseShapeOp>(
      loc, subview, build2DCollapseFromPhysical(R + 2));
}

static bool isSbuf(Attribute memSpaceAttr) {
  if (auto a = dyn_cast_or_null<nkipy::MemSpaceEnumAttr>(memSpaceAttr))
    return a.getValue() == nkipy::MemSpaceEnum::Sbuf;
  return false;
}

static bool isHbm(Attribute memSpaceAttr) {
  if (auto a = dyn_cast_or_null<nkipy::MemSpaceEnumAttr>(memSpaceAttr))
    return a.getValue() == nkipy::MemSpaceEnum::Hbm ||
           a.getValue() == nkipy::MemSpaceEnum::SharedHbm;
  return false;
}

/// Check if a copy/transpose needs tiling (HBM↔SBUF transfer).
static bool needsTiledTransfer(MemRefType srcType, MemRefType dstType) {
  bool srcH = isHbm(srcType.getMemorySpace());
  bool srcS = isSbuf(srcType.getMemorySpace());
  bool dstH = isHbm(dstType.getMemorySpace());
  bool dstS = isSbuf(dstType.getMemorySpace());
  return (srcH && dstS) || (srcS && dstH);
}

/// Look through memref.cast ops to find the base value.
/// memref.cast changes static type information but doesn't create new data.
static Value lookThroughCast(Value v) {
  while (auto castOp = v.getDefiningOp<memref::CastOp>()) {
    v = castOp.getSource();
  }
  return v;
}

/// Like lookThroughCast, but also tries to resolve through collapse_shape
/// when the base alloc is in valueMapping (i.e., was actually legalized).
/// This handles chains like: legalized_alloc → collapse_shape → cast → user
/// where Step 1.5 replaced the Phase 0 collapse with one from the legalized alloc.
static Value lookThroughCastAndResolve(Value v, IRMapping &valueMapping) {
  v = lookThroughCast(v);
  // If the value after cast-stripping is in valueMapping, we're done.
  if (valueMapping.lookupOrNull(v))
    return v;
  // Try going through collapse_shape to find a mapped (legalized) base alloc.
  if (auto collapseOp = v.getDefiningOp<memref::CollapseShapeOp>()) {
    Value src = lookThroughCast(collapseOp.getSrc());
    if (valueMapping.lookupOrNull(src))
      return src;
  }
  return v;
}

/// Walk backward from `current` through subviews/casts to find a
/// memref.collapse_shape in the defining chain. Returns it if found, or nullptr.
static memref::CollapseShapeOp findCollapseInDefChain(Value current) {
  Value v = current;
  while (v) {
    if (auto collapseOp = v.getDefiningOp<memref::CollapseShapeOp>())
      return collapseOp;
    if (auto subviewOp = v.getDefiningOp<memref::SubViewOp>()) {
      v = subviewOp.getSource();
      continue;
    }
    if (auto castOp = v.getDefiningOp<memref::CastOp>()) {
      v = castOp.getSource();
      continue;
    }
    break;
  }
  return nullptr;
}

/// Given a tile shape from a collapsed domain (e.g. [128, 128] from a 2D view),
/// expand it back to the alloc's rank using the collapse_shape's reassociation
/// indices.
///
/// For each reassociation group with multiple source dims, try to assign the
/// full collapsed tile to a single source dim (keeping other dims at 1).
/// This preserves the middle-tile-must-be-1 invariant for legalize-layout.
///
/// Priority:
///   1. Exact match: a dim whose size == the collapsed tile.
///   2. Outermost dim that can hold the tile (srcShape[dim] >= tile).
///   3. Fallback: distribute across dims from innermost outward.
///
/// Example: reassoc = [[0, 1], [2]], srcShape = [4, 128, 64], tile = [128, 64]
///   group [0, 1], srcShape=[4,128], tile=128:
///     dim 1 (128) == 128 → exact match → [1, 128]
///   group [2], tile=64: → [64]
///   result: [1, 128, 64]
static SmallVector<int64_t> expandTileShape(
    ArrayRef<int64_t> collapsedTile,
    memref::CollapseShapeOp collapseOp) {
  auto reassoc = collapseOp.getReassociationIndices();
  auto srcType = cast<MemRefType>(collapseOp.getSrc().getType());
  auto srcShape = srcType.getShape();
  SmallVector<int64_t> expanded(srcShape.size(), 1);

  for (size_t g = 0; g < reassoc.size(); g++) {
    auto &group = reassoc[g];
    int64_t tile = collapsedTile[g];

    if (group.size() == 1) {
      expanded[group[0]] = tile;
      continue;
    }

    // Try to assign the full tile to a single dim in the group.
    // Priority: exact match first, then outermost dim that can hold it.
    int bestIdx = -1;
    for (size_t i = 0; i < group.size(); i++) {
      int64_t srcDim = srcShape[group[i]];
      if (srcDim == tile) {
        bestIdx = static_cast<int>(i);
        break;  // Exact match — stop looking.
      }
      if (bestIdx == -1 && srcDim >= tile) {
        bestIdx = static_cast<int>(i);  // First (outermost) holder.
      }
    }

    if (bestIdx >= 0) {
      // Assign full tile to the chosen dim; rest stay 1.
      expanded[group[bestIdx]] = tile;
    } else {
      // No single dim can hold the tile — distribute from innermost outward.
      int64_t remaining = tile;
      for (int i = static_cast<int>(group.size()) - 1; i >= 0; i--) {
        int64_t srcDim = srcShape[group[i]];
        int64_t tileForDim = std::min(srcDim, remaining);
        expanded[group[i]] = tileForDim;
        if (tileForDim > 0)
          remaining /= tileForDim;
      }
    }
  }
  return expanded;
}

/// Trace a value forward through uses to find ALL linalg operands it feeds into.
/// Returns a vector of (linalgOp, operandIndex, operandShape) tuples.
///
/// Post-bufferization version: follows memref.subview ops to find linalg users.
/// Used to collect tile sizes from all linalg uses of an SBUF allocation.
static void traceToLinalgOperands(
    Value val,
    SmallVector<std::tuple<linalg::LinalgOp, unsigned, SmallVector<int64_t>>> &results) {

  // Track visited values to avoid infinite loops
  llvm::SmallPtrSet<Value, 16> visited;
  std::queue<Value> workList;
  workList.push(val);

  while (!workList.empty()) {
    Value current = workList.front();
    workList.pop();

    if (visited.contains(current))
      continue;
    visited.insert(current);

    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();

      // Check if this is a linalg op
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(user)) {
        // Record the operand shape as tile size.
        // If a collapse_shape exists anywhere in the def chain back to the alloc,
        // expand the tile shape back to the alloc's rank using the reassociation.
        // This handles chains like: alloc[MxBxN] → subview → collapse_shape → subview → linalg
        MemRefType operandType = dyn_cast<MemRefType>(current.getType());
        if (operandType) {
          SmallVector<int64_t> tileShape(operandType.getShape().begin(),
                                         operandType.getShape().end());
          if (auto collapseOp = findCollapseInDefChain(current)) {
            // Only expand if tile rank matches the collapsed rank (number of
            // reassociation groups).  If they differ, the tile went through
            // additional rank changes (e.g., a subview that dropped dims after
            // the collapse_shape) and cannot be directly expanded back.
            // Skip recording this tile — it doesn't represent the alloc's tiling.
            auto reassoc = collapseOp.getReassociationIndices();
            if (tileShape.size() == reassoc.size()) {
              tileShape = expandTileShape(tileShape, collapseOp);
            } else {
              LLVM_DEBUG(llvm::dbgs() << "   Skipping linalg operand (tile rank "
                           << tileShape.size() << " != collapsed rank "
                           << reassoc.size() << ")\n");
              continue;
            }
          }
          LLVM_DEBUG({
            llvm::dbgs() << "   Found linalg operand with shape [";
            llvm::interleave(tileShape, llvm::dbgs(), "x");
            llvm::dbgs() << "] in " << linalgOp->getName() << "\n";
          });
          // Find which operand index this use corresponds to
          for (unsigned i = 0; i < linalgOp->getNumOperands(); ++i) {
            if (linalgOp->getOperand(i) == current) {
              results.push_back({linalgOp, i, tileShape});
              break;
            }
          }
        }
        continue;  // Don't stop - keep looking for more linalg uses
      }

      // Follow through memref.subview (post-bufferization)
      // Don't record shape here - we'll record it when we see actual uses
      // (linalg ops or copy destinations)
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
        workList.push(subviewOp.getResult());
        continue;
      }

      // Follow through memref.cast (type refinement, doesn't change data)
      if (auto castOp = dyn_cast<memref::CastOp>(user)) {
        workList.push(castOp.getResult());
        continue;
      }

      // Follow through memref.collapse_shape (created by Phase 0's foldReshapeIntoAlloc)
      if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(user)) {
        workList.push(collapseOp.getResult());
        continue;
      }

      // Follow through memref.expand_shape
      if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(user)) {
        workList.push(expandOp.getResult());
        continue;
      }

      // Record shape when current value is used as copy destination
      // This captures tile sizes for buffers that are written via copy.
      // If a collapse_shape exists in the def chain, expand back to alloc rank.
      if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
        if (copyOp.getTarget() == current) {
          MemRefType memrefType = dyn_cast<MemRefType>(current.getType());
          if (memrefType) {
            SmallVector<int64_t> tileShape(memrefType.getShape().begin(),
                                           memrefType.getShape().end());
            if (auto collapseOp = findCollapseInDefChain(current)) {
              auto reassoc = collapseOp.getReassociationIndices();
              if (tileShape.size() == reassoc.size()) {
                tileShape = expandTileShape(tileShape, collapseOp);
              } else {
                LLVM_DEBUG(llvm::dbgs() << "   Skipping copy dest (tile rank "
                             << tileShape.size() << " != collapsed rank "
                             << reassoc.size() << ")\n");
                continue;
              }
            }
            LLVM_DEBUG({
              llvm::dbgs() << "   Found copy destination tile shape [";
              llvm::interleave(tileShape, llvm::dbgs(), "x");
              llvm::dbgs() << "]\n";
            });
            results.push_back({linalg::LinalgOp(nullptr), 0, tileShape});
          }
        }
        // Don't follow through copies
        continue;
      }
    }
  }
}

struct NkipyLegalizeLayoutPass
    : public LegalizeLayoutBase<NkipyLegalizeLayoutPass> {
  
  // Track if any errors occurred during the pass
  bool hasError = false;
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<bufferization::BufferizationDialect>();
  }

  /// Try to compute a reassociation map that collapses newShape back to origShape.
  /// Greedily groups consecutive dims of newShape whose product equals each dim
  /// of origShape. Works for any rank combination.
  ///
  /// Example: origShape=[256,64], newShape=[256,1,64]
  ///   dim 0: 256 == 256 → group [0]
  ///   dim 1: 1*64 == 64 → group [1,2]
  ///   result: [[0], [1,2]]
  static bool computeCollapseReassociation(
      ArrayRef<int64_t> newShape,
      ArrayRef<int64_t> origShape,
      SmallVector<ReassociationIndices> &reassociation) {
    reassociation.clear();
    int64_t newIdx = 0;
    int64_t newRank = newShape.size();

    for (int64_t origDim : origShape) {
      ReassociationIndices group;
      int64_t product = 1;

      while (newIdx < newRank && product < origDim) {
        product *= newShape[newIdx];
        group.push_back(newIdx);
        newIdx++;
      }

      if (product != origDim || group.empty())
        return false;

      reassociation.push_back(group);
    }

    return newIdx == newRank;
  }

  /// Pre-processing: fold alloc+reshape patterns into a single higher-rank alloc.
  ///
  /// Transforms:
  ///   %a = memref.alloc() : memref<OrigShape, sbuf>
  ///   memref.copy %src, %a
  ///   %r = memref.reshape %a(%shape) -> memref<NewShape, sbuf>
  /// Into:
  ///   %a = memref.alloc() : memref<NewShape, sbuf>
  ///   %v = memref.collapse_shape %a [...] -> memref<OrigShape, sbuf>
  ///   memref.copy %src, %v
  ///   // uses of %r replaced with %a
  ///
  /// This eliminates lower-rank SBUF allocs that can't be legalized because
  /// their tile sizes are only discoverable through the reshape.
  void foldReshapeIntoAlloc(func::FuncOp func) {
    SmallVector<memref::ReshapeOp> reshapesToFold;

    func.walk([&](memref::ReshapeOp reshapeOp) {
      Value source = reshapeOp.getSource();
      auto sourceType = cast<MemRefType>(source.getType());
      auto resultType = cast<MemRefType>(reshapeOp.getResult().getType());

      // Only handle SBUF allocs
      if (!isSbuf(sourceType.getMemorySpace()))
        return;
      auto allocOp = source.getDefiningOp<memref::AllocOp>();
      if (!allocOp)
        return;

      // Both must have identity layout
      if (!sourceType.getLayout().isIdentity() ||
          !resultType.getLayout().isIdentity())
        return;

      // Must be able to compute a valid collapse reassociation
      SmallVector<ReassociationIndices> reassociation;
      if (!computeCollapseReassociation(
              resultType.getShape(), sourceType.getShape(), reassociation))
        return;

      reshapesToFold.push_back(reshapeOp);
    });

    for (auto reshapeOp : reshapesToFold) {
      Value source = reshapeOp.getSource();
      auto allocOp = source.getDefiningOp<memref::AllocOp>();
      auto sourceType = cast<MemRefType>(source.getType());
      auto resultType = cast<MemRefType>(reshapeOp.getResult().getType());

      SmallVector<ReassociationIndices> reassociation;
      computeCollapseReassociation(
          resultType.getShape(), sourceType.getShape(), reassociation);

      OpBuilder builder(allocOp);

      // Create new alloc with the reshaped (higher-rank) type
      auto newAllocType = MemRefType::get(
          resultType.getShape(),
          resultType.getElementType(),
          /*layout=*/nullptr,
          sourceType.getMemorySpace());
      auto newAlloc = builder.create<memref::AllocOp>(
          allocOp.getLoc(), newAllocType, allocOp.getAlignmentAttr());

      // Create collapse_shape to provide the original shape for existing copy users
      auto collapseOp = builder.create<memref::CollapseShapeOp>(
          allocOp.getLoc(), newAlloc.getResult(), reassociation);

      LLVM_DEBUG(llvm::dbgs() << " Folded reshape into alloc: "
                   << sourceType << " -> " << newAllocType << "\n");

      // Replace uses: old alloc users → collapse_shape, old reshape users → new alloc
      allocOp.getResult().replaceAllUsesWith(collapseOp.getResult());
      reshapeOp.getResult().replaceAllUsesWith(newAlloc.getResult());

      reshapeOp.erase();
      allocOp.erase();
    }

    // Also fold memref.expand_shape on SBUF allocs.
    // expand_shape increases rank (e.g., [1,128] → [1,128,1] from expand_dims).
    // Without folding, the alloc is legalized at its original rank but the
    // expand_shape result has a different rank, causing downstream mismatches.
    SmallVector<memref::ExpandShapeOp> expandsToFold;

    func.walk([&](memref::ExpandShapeOp expandOp) {
      Value source = expandOp.getSrc();
      auto sourceType = cast<MemRefType>(source.getType());

      if (!isSbuf(sourceType.getMemorySpace()))
        return;
      auto allocOp = source.getDefiningOp<memref::AllocOp>();
      if (!allocOp)
        return;
      if (!sourceType.getLayout().isIdentity())
        return;

      expandsToFold.push_back(expandOp);
    });

    for (auto expandOp : expandsToFold) {
      Value source = expandOp.getSrc();
      auto allocOp = source.getDefiningOp<memref::AllocOp>();
      auto sourceType = cast<MemRefType>(source.getType());
      auto resultType = cast<MemRefType>(expandOp.getResult().getType());

      OpBuilder builder(allocOp);

      // Create new alloc with the expanded (higher-rank) type
      auto newAllocType = MemRefType::get(
          resultType.getShape(),
          resultType.getElementType(),
          /*layout=*/nullptr,
          sourceType.getMemorySpace());
      auto newAlloc = builder.create<memref::AllocOp>(
          allocOp.getLoc(), newAllocType, allocOp.getAlignmentAttr());

      // Create collapse_shape to provide the original shape for existing users
      // (e.g., HBM→SBUF copies that wrote into the original 2D alloc)
      auto reassociation = expandOp.getReassociationIndices();
      auto collapseOp = builder.create<memref::CollapseShapeOp>(
          allocOp.getLoc(), newAlloc.getResult(), reassociation);

      LLVM_DEBUG(llvm::dbgs() << " Folded expand_shape into alloc: "
                   << sourceType << " -> " << newAllocType << "\n");

      // Replace uses: old alloc users → collapse_shape, expand_shape users → new alloc
      allocOp.getResult().replaceAllUsesWith(collapseOp.getResult());
      expandOp.getResult().replaceAllUsesWith(newAlloc.getResult());

      expandOp.erase();
      allocOp.erase();
    }
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    hasError = false;

    LLVM_DEBUG(llvm::dbgs() << "[LegalizeLayout] Processing function: "
                 << func.getName() << "\n");

    // Phase 0: Fold alloc+reshape patterns into single higher-rank allocs
    // This handles SBUF allocs created by expand_dims bufferization where the
    // alloc has a lower rank than the reshape result. Without folding, the
    // alloc's tile sizes can't be determined (they're only visible through
    // the reshape's downstream subviews).
    foldReshapeIntoAlloc(func);

    // Phase 1: Identify all SBUF tensors needing legalization
    SmallVector<LayoutInfo> layoutInfos = findSbufTensorsToLegalize(func);

    if (hasError) {
      llvm::errs() << "[LegalizeLayout] FAILED in Phase 1 (find SBUF tensors)\n";
      signalPassFailure();
      return;
    }

    if (layoutInfos.empty()) {
      llvm::errs() << "[LegalizeLayout] No SBUF tensors need legalization\n";
      // Still need to decompose HBM fills even when no SBUF layout changes
      decomposeHbmFills(func);
      return;
    }

    llvm::errs() << "[LegalizeLayout] Found " << layoutInfos.size()
                 << " SBUF tensor(s) to legalize:\n";
    for (auto &info : layoutInfos) {
      llvm::errs() << "  tensor<";
      llvm::interleave(info.origShape, llvm::errs(), "x");
      llvm::errs() << "> tile=[";
      llvm::interleave(info.tileSize, llvm::errs(), ",");
      llvm::errs() << "] numBlocks=[";
      llvm::interleave(info.numBlocks, llvm::errs(), ",");
      llvm::errs() << "] -> physical<";
      auto phys = info.getPhysicalShape();
      llvm::interleave(phys, llvm::errs(), "x");
      llvm::errs() << ">\n";
    }

    // Phase 2: Transform allocations and subviews to physical (R+2)-D layout
    IRMapping valueMapping;
    transformToPhysicalLayout(func, layoutInfos, valueMapping);

    if (hasError) {
      llvm::errs() << "[LegalizeLayout] FAILED in Phase 2 (transform to physical layout)\n";
      signalPassFailure();
      return;
    }

    // Phase 3a: Tile HBM↔SBUF copies and transposes
    // These ops require same-rank inputs/outputs, so we generate tiled loops
    tileCopyAndTranspose(func, layoutInfos, valueMapping);

    if (hasError) {
      llvm::errs() << "[LegalizeLayout] FAILED in Phase 3a (tile copy/transpose)\n";
      signalPassFailure();
      return;
    }

    // Phase 3b: Decompose linalg.fill on HBM
    // nisa.memset only supports SBUF/PSUM, so fill on HBM is decomposed into:
    //   alloc SBUF temp → linalg.fill SBUF → scf.for { memref.copy SBUF → HBM }
    decomposeHbmFills(func);

    if (hasError) {
      llvm::errs() << "[LegalizeLayout] FAILED in Phase 3b (decompose HBM fills)\n";
      signalPassFailure();
      return;
    }

    // Phase 4: Fix rank mismatches
    // After Phase 2, some ops have (R+2)-D operands but expect R-D:
    // - linalg ops: (R+2)-D operands but indexing maps expect R-D
    // - memref.copy: PSUM↔SBUF where PSUM is R-D and SBUF is now (R+2)-D
    // Insert collapse_shape to convert (R+2)-D -> R-D where needed.
    fixRankMismatches(func);

    if (hasError) {
      llvm::errs() << "[LegalizeLayout] FAILED in Phase 4 (fix rank mismatches)\n";
      signalPassFailure();
      return;
    }

    llvm::errs() << "[LegalizeLayout] Pass completed successfully\n";
  }
  
  /// Build a map from 2D values to their layout info for quick lookup
  DenseMap<Value, LayoutInfo*> buildValueToLayoutMap(SmallVector<LayoutInfo> &layoutInfos) {
    DenseMap<Value, LayoutInfo*> valueMap;
    for (auto &info : layoutInfos) {
      valueMap[info.originalValue] = &info;
    }
    return valueMap;
  }
  
  /// Transform all identified SBUF memrefs to (R+2)-D physical layout
  ///
  /// Post-bufferization algorithm:
  /// 1. Transform memref.alloc to (R+2)-D shape
  /// 2. Collect all ops using transformed allocations via BFS
  /// 3. Transform memref.subview ops to use (R+2)-D indexing
  /// 
  /// Note: memref.copy and linalg.transpose are handled separately in
  /// tileCopyAndTranspose() because they require same-rank src/dst.
  void transformToPhysicalLayout(func::FuncOp func, SmallVector<LayoutInfo> &layoutInfos,
                                 IRMapping &valueMapping) {
    if (layoutInfos.empty())
      return;
    
    OpBuilder builder(func.getContext());
    auto valueMap = buildValueToLayoutMap(layoutInfos);
    
    // Step 1: Transform allocations (memref.alloc)
    for (auto &info : layoutInfos) {
      transformAllocation(builder, info, valueMapping);
      if (hasError) return;
    }
    
    // Step 1.5: Update collapse_shape ops from Phase 0 (foldReshapeIntoAlloc)
    //
    // Phase 0 creates: alloc(R-D) → collapse_shape → alloc_expanded((R+k)-D)
    // After Step 1, alloc_expanded is mapped to a new (R+k+2)-D physical alloc.
    // The collapse_shape still references the old alloc_expanded — update it to
    // reference the new physical alloc with a composed reassociation.
    for (auto &info : layoutInfos) {
      Value origAlloc = info.originalValue;
      Value legAlloc = valueMapping.lookupOrNull(origAlloc);
      if (!legAlloc) continue;

      // Walk through reinterpret_cast to get the raw identity-layout alloc.
      // Narrow buffers use reinterpret_cast for strided layout, but
      // collapse_shape requires contiguous (identity) layout.
      Value rawAlloc = legAlloc;
      while (auto reinterpret = rawAlloc.getDefiningOp<memref::ReinterpretCastOp>())
        rawAlloc = reinterpret.getSource();

      SmallVector<memref::CollapseShapeOp> collapseOps;
      for (auto &use : origAlloc.getUses()) {
        if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(use.getOwner()))
          collapseOps.push_back(collapseOp);
      }

      for (auto collapseOp : collapseOps) {
        // Map the R-D → targetRank reassociation through the (R+2)-D physical layout.
        // Physical layout convention: [t0, nB0, nB1, ..., nB_{R-1}, t_{R-1}]
        // So R-dim i maps to physical dims:
        //   i=0     → {0, 1}      (tile0, numBlocks0)
        //   0<i<R-1 → {i+1}       (numBlocks_i)
        //   i=R-1   → {R, R+1}    (numBlocks_{R-1}, tile_{R-1})
        auto oldReassoc = collapseOp.getReassociationIndices();
        int64_t R = info.rank();

        SmallVector<ReassociationIndices> newReassoc;
        for (auto &group : oldReassoc) {
          ReassociationIndices newGroup;
          for (int64_t rdim : group) {
            if (rdim == 0) {
              newGroup.push_back(0);
              newGroup.push_back(1);
            } else if (rdim == R - 1) {
              newGroup.push_back(R);
              newGroup.push_back(R + 1);
            } else {
              newGroup.push_back(rdim + 1);
            }
          }
          newReassoc.push_back(newGroup);
        }

        OpBuilder b(collapseOp);
        auto newCollapse = b.create<memref::CollapseShapeOp>(
            collapseOp.getLoc(), rawAlloc, newReassoc);
        collapseOp.getResult().replaceAllUsesWith(newCollapse.getResult());
        collapseOp.erase();

        LLVM_DEBUG(llvm::dbgs() << " Updated Phase 0 collapse_shape to "
                     << "reference legalized alloc\n");
      }
    }

    // Step 2: Collect which ops need transformation via BFS (into a set)
    auto opsToTransform = collectOpsToTransform(layoutInfos, valueMapping);
    
    // Step 3: Walk the function in program order (topological order)
    // and transform subview ops that are in the set.
    // Note: memref.copy is handled separately in tileCopyAndTranspose()
    //
    // IMPORTANT: We collect ops to transform first, then do all transforms,
    // then do cleanup. This is because replaceAllUsesWith on a parent subview
    // would modify the child subview's source operand before we transform it.
    SmallVector<std::pair<memref::SubViewOp, memref::SubViewOp>> subviewReplacements;

    func.walk([&](Operation *op) {
      if (hasError)
        return WalkResult::interrupt();

      // Only process ops that need transformation
      if (!opsToTransform.contains(op))
        return WalkResult::advance();

      if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
        auto newSubview = transformSubview(builder, subviewOp, valueMapping, valueMap);
        if (newSubview) {
          subviewReplacements.push_back({subviewOp, newSubview});
        }
      }

      return WalkResult::advance();
    });

    // Step 4: Now do subview replacements and cleanup
    for (auto &[oldOp, newOp] : subviewReplacements) {
      oldOp.getResult().replaceAllUsesWith(newOp.getResult());
      oldOp.erase();
    }

    // Step 5: Redirect linalg ops that still reference the original alloc
    // (e.g., fill, generic) to a collapse_shape view of the legalized alloc.
    // This handles untiled ops that KnobDrivenTiling left as full-buffer accesses.
    redirectDirectLinalgUses(layoutInfos, valueMapping);
  }
  
  /// Redirect linalg ops that directly reference original (stale) alloc values.
  ///
  /// Two cases based on whether the linalg op mixes legalized/non-legalized operands:
  ///  - All-SBUF (e.g. fill): redirect to collapse_shape view — interleaving is
  ///    consistent across all operands so element-wise computation is correct.
  ///  - Mixed (e.g. generic with SBUF input + non-SBUF output): copy legalized
  ///    data block-by-block into a sequential temp, then redirect to temp.
  void redirectDirectLinalgUses(SmallVector<LayoutInfo> &layoutInfos,
                                IRMapping &valueMapping) {
    DenseMap<Value, LayoutInfo*> origToLayout;
    for (auto &info : layoutInfos)
      origToLayout[info.originalValue] = &info;

    // Classify linalg ops BEFORE any redirects (to avoid stale operand checks)
    struct UseRecord {
      linalg::LinalgOp op;
      unsigned operandIdx;
      LayoutInfo *info;
      bool mixed;
    };
    SmallVector<UseRecord> records;

    for (auto &info : layoutInfos) {
      Value origAlloc = info.originalValue;
      if (!valueMapping.lookupOrNull(origAlloc)) continue;

      for (OpOperand &use : origAlloc.getUses()) {
        auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner());
        if (!linalgOp) continue;

        // Skip TransposeOp — handled separately by Phase 3a (tileCopyAndTranspose)
        if (isa<linalg::TransposeOp>(use.getOwner())) continue;

        // Mixed = any rank≥2 operand is NOT a legalized orig alloc
        bool mixed = false;
        for (unsigned i = 0; i < linalgOp->getNumOperands(); ++i) {
          auto mt = dyn_cast<MemRefType>(linalgOp->getOperand(i).getType());
          if (mt && mt.getRank() >= 2 && !origToLayout.count(linalgOp->getOperand(i))) {
            mixed = true;
            break;
          }
        }
        records.push_back({linalgOp, use.getOperandNumber(), &info, mixed});
      }
    }

    // Cache one collapse_shape per legalized alloc (for all-SBUF ops)
    DenseMap<Value, Value> collapseCache;
    // Ops replaced by tiled versions that need to be erased
    llvm::SmallPtrSet<Operation*, 4> opsToErase;

    for (auto &rec : records) {
      if (opsToErase.contains(rec.op)) continue;
      Value legAlloc = valueMapping.lookup(rec.info->originalValue);

      if (!rec.mixed) {
        auto legType = cast<MemRefType>(legAlloc.getType());

        if (!legType.getLayout().isIdentity()) {
          // Partition-contiguous strides (e.g. [1,128,128,1] for narrow
          // buffers): collapse_shape on the full alloc fails because the
          // strides aren't row-major contiguous when numBlocks dims > 1.
          // Generate a per-tile loop instead: subview each tile (middle dims
          // become 1 → collapse works), then clone the linalg op on each tile.
          int64_t R = rec.info->rank();
          OpBuilder b(rec.op);
          Location loc = rec.op.getLoc();

          auto nest = createBlockLoopNest(b, loc, rec.info->numBlocks);
          Value tileCollapsed = createTileSubviewAndCollapse(
              b, loc, legAlloc, rec.info->tileSize[0],
              rec.info->tileSize[R - 1], R, nest.ivs);

          // Clone the linalg op into the loop body, redirecting the
          // legalized operand to the tile-sized collapsed view
          auto *cloned = b.clone(*rec.op);
          cloned->setOperand(rec.operandIdx, tileCollapsed);
          // Mark original for erasure (will be erased after loop)
          opsToErase.insert(rec.op);
          LLVM_DEBUG(llvm::dbgs() << " Tiled " << rec.op->getName()
                       << " over " << R << " block dims for strided buffer\n");
        } else {
          // Default row-major layout: collapse_shape on full alloc works.
          Value &collapsed = collapseCache[rec.info->originalValue];
          if (!collapsed) {
            OpBuilder b(legAlloc.getContext());
            b.setInsertionPointAfterValue(legAlloc);
            collapsed = b.create<memref::CollapseShapeOp>(
                legAlloc.getLoc(), legAlloc,
                build2DCollapseFromPhysical(rec.info->physicalRank()));
          }
          rec.op->setOperand(rec.operandIdx, collapsed);
          LLVM_DEBUG(llvm::dbgs() << " Redirected " << rec.op->getName()
                       << " operand " << rec.operandIdx << " to collapsed view\n");
        }
      } else {
        // Mixed: deinterleave via block-by-block copy into sequential temp
        OpBuilder b(rec.op);
        Location loc = rec.op.getLoc();
        int64_t R = rec.info->rank();
        auto elemTy = cast<MemRefType>(legAlloc.getType()).getElementType();
        auto temp = b.create<memref::AllocOp>(
            loc, MemRefType::get(rec.info->origShape, elemTy));

        auto nest = createBlockLoopNest(b, loc, rec.info->numBlocks);
        Value legC = createTileSubviewAndCollapse(
            b, loc, legAlloc, rec.info->tileSize[0],
            rec.info->tileSize[R - 1], R, nest.ivs);

        // Temp subview [b0*t0, ...][t0, ...]
        SmallVector<OpFoldResult> so, ss, sr;
        for (int64_t d = 0; d < R; d++) {
          if (rec.info->tileSize[d] == 1) {
            so.push_back(OpFoldResult(nest.ivs[d]));
          } else {
            Value ts = b.create<arith::ConstantIndexOp>(loc, rec.info->tileSize[d]);
            so.push_back(OpFoldResult(b.create<arith::MulIOp>(loc, nest.ivs[d], ts)));
          }
          ss.push_back(b.getIndexAttr(rec.info->tileSize[d]));
          sr.push_back(b.getIndexAttr(1));
        }
        auto seqSV = b.create<memref::SubViewOp>(loc, temp, so, ss, sr);
        b.create<memref::CopyOp>(loc, legC, seqSV);

        rec.op->setOperand(rec.operandIdx, temp.getResult());
        LLVM_DEBUG(llvm::dbgs() << " Deinterleaved " << rec.op->getName()
                     << " operand " << rec.operandIdx << " via temp copy\n");
      }
    }

    // Erase original ops that were replaced by tiled versions
    for (auto *op : opsToErase)
      op->erase();
  }

  /// Convert an R-D offset to a block index by dividing by tile size.
  /// 
  /// NOTE: This is fragile! It assumes offsets after canonicalize-loop-step
  /// are either:
  ///   1. Static constants (divisible by tileSize)
  ///   2. Dynamic values of the form: arith.muli %idx, %tileSize
  /// 
  /// For case 2, we pattern-match and extract %idx directly (avoiding division).
  /// If the pattern doesn't match, we emit an error.
  std::optional<OpFoldResult> computeBlockIndex(OpBuilder &builder,
                                                 OpFoldResult offset2D,
                                                 int64_t tileSize,
                                                 Location loc) {
    // Special case: tile_size == 1, block index equals the offset directly.
    // This handles middle dims of R-D tensors where tile=1 and the offset
    // is a bare loop induction variable (not wrapped in arith.muli).
    if (tileSize == 1) {
      return offset2D;
    }

    // Case 1: Static constant offset
    if (auto attr = dyn_cast<Attribute>(offset2D)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        int64_t val = intAttr.getInt();
        if (val % tileSize != 0) {
          llvm::errs() << "[LegalizeLayout] Error: static offset " << val
                     << " not divisible by tile size " << tileSize << "\n";
          return std::nullopt;
        }
        return builder.getIndexAttr(val / tileSize);
      }
    }
    
    // Case 2: Dynamic value - try to pattern match arith.muli %idx, %multiplier
    // where multiplier is divisible by tileSize
    Value offsetVal = cast<Value>(offset2D);
    if (auto mulOp = offsetVal.getDefiningOp<arith::MulIOp>()) {
      // Check if RHS is a constant divisible by tileSize
      if (auto constOp = mulOp.getRhs().getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          int64_t multiplier = intAttr.getInt();
          if (multiplier % tileSize == 0) {
            int64_t scale = multiplier / tileSize;
            if (scale == 1) {
              // offset = idx * tileSize, so block_index = idx
              return OpFoldResult(mulOp.getLhs());
            } else {
              // offset = idx * (scale * tileSize), so block_index = idx * scale
              Value scaleVal = builder.create<arith::ConstantIndexOp>(loc, scale);
              Value blockIdx = builder.create<arith::MulIOp>(loc, mulOp.getLhs(), scaleVal);
              return OpFoldResult(blockIdx);
            }
          }
        }
      }
      // Also check LHS in case constant is on the left
      if (auto constOp = mulOp.getLhs().getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          int64_t multiplier = intAttr.getInt();
          if (multiplier % tileSize == 0) {
            int64_t scale = multiplier / tileSize;
            if (scale == 1) {
              return OpFoldResult(mulOp.getRhs());
            } else {
              Value scaleVal = builder.create<arith::ConstantIndexOp>(loc, scale);
              Value blockIdx = builder.create<arith::MulIOp>(loc, mulOp.getRhs(), scaleVal);
              return OpFoldResult(blockIdx);
            }
          }
        }
      }
    }
    
    // Case 3: Check if it's just 0 (common for first dimension)
    if (auto constOp = offsetVal.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        int64_t val = intAttr.getInt();
        if (val % tileSize != 0) {
          llvm::errs() << "[LegalizeLayout] Error: constant offset " << val
                       << " not divisible by tile size " << tileSize << "\n";
          return std::nullopt;
        }
        return builder.getIndexAttr(val / tileSize);
      }
    }
   
    llvm::errs() << "[LegalizeLayout] Error: cannot compute block index from dynamic offset. "
                 << "Expected pattern: constant (divisible by " << tileSize << ") or "
                 << "arith.muli %idx, " << tileSize << "\n";
    return std::nullopt;
  }
  
  /// Transform memref.subview to physical-layout subview with correct sizes/offsets
  ///
  /// For rank-R tensor with tile [t_0, ..., t_{R-1}]:
  ///   R-D subview: [off_0, ..., off_{R-1}][sz_0, ..., sz_{R-1}]
  ///   (R+2)-D subview: [0, off_0/t_0, ..., off_{R-1}/t_{R-1}, 0]
  ///                    [t_0, sz_0/t_0, ..., sz_{R-1}/t_{R-1}, t_{R-1}]
  ///
  /// The first dim (partition tile) and last dim (free tile) always span full tiles.
  /// The middle R dims are block indices computed from the original offsets/sizes.
  memref::SubViewOp transformSubview(OpBuilder &builder, memref::SubViewOp op,
                                     IRMapping &valueMapping,
                                     DenseMap<Value, LayoutInfo*> &valueMap) {
    Value source = op.getSource();

    // Find the physical source value
    Value sourcePhys = valueMapping.lookupOrNull(source);
    if (!sourcePhys) {
      // Source wasn't transformed - skip
      return nullptr;
    }

    // Find layout info for this source
    LayoutInfo *info = findLayoutInfo(source, valueMapping, valueMap);
    if (!info) {
      LLVM_DEBUG(llvm::dbgs() << " Warning: subview source not in layout map\n");
      return nullptr;
    }

    int64_t R = info->rank();

    // Get the R-D offsets and sizes from original subview
    auto mixedOffsets = op.getMixedOffsets();
    auto mixedSizes = op.getMixedSizes();
    if ((int64_t)mixedOffsets.size() != R || (int64_t)mixedSizes.size() != R) {
      llvm::errs() << "[LegalizeLayout] Error: expected " << R << "D subview but got "
                   << mixedOffsets.size() << "D offsets, "
                   << mixedSizes.size() << "D sizes\n";
      hasError = true;
      return nullptr;
    }

    builder.setInsertionPoint(op);
    Location loc = op.getLoc();

    // Get static sizes (we require them to be static for now)
    auto staticSizes = op.getStaticSizes();
    for (int64_t i = 0; i < R; i++) {
      if (staticSizes[i] == ShapedType::kDynamic) {
        llvm::errs() << "[LegalizeLayout] Error: dynamic subview sizes not supported\n";
        hasError = true;
        return nullptr;
      }
    }

    // Validate divisibility for all dims
    for (int64_t i = 0; i < R; i++) {
      if (staticSizes[i] % info->tileSize[i] != 0) {
        llvm::errs() << "[LegalizeLayout] Error: subview size " << staticSizes[i]
                     << " in dim " << i << " not divisible by tile size " << info->tileSize[i] << "\n";
        hasError = true;
        return nullptr;
      }
    }

    // Compute block-index offsets from R-D offsets
    SmallVector<std::optional<OpFoldResult>> blockIndices;
    for (int64_t i = 0; i < R; i++) {
      auto idx = computeBlockIndex(builder, mixedOffsets[i], info->tileSize[i], loc);
      if (!idx) {
        hasError = true;
        return nullptr;
      }
      blockIndices.push_back(idx);
    }

    // Build (R+2)-dim offsets: [0, blockIdx_0, ..., blockIdx_{R-1}, 0]
    SmallVector<OpFoldResult> offsetsPhys;
    offsetsPhys.push_back(builder.getIndexAttr(0));   // Partition tile: always 0
    for (int64_t i = 0; i < R; i++)
      offsetsPhys.push_back(*blockIndices[i]);
    offsetsPhys.push_back(builder.getIndexAttr(0));   // Free tile: always 0

    // Build (R+2)-dim sizes: [tileSize[0], sz_0/t_0, ..., sz_{R-1}/t_{R-1}, tileSize[R-1]]
    SmallVector<OpFoldResult> sizesPhys;
    sizesPhys.push_back(builder.getIndexAttr(info->tileSize[0]));
    for (int64_t i = 0; i < R; i++)
      sizesPhys.push_back(builder.getIndexAttr(staticSizes[i] / info->tileSize[i]));
    sizesPhys.push_back(builder.getIndexAttr(info->tileSize[R - 1]));

    // Build (R+2)-dim strides: all 1s
    SmallVector<OpFoldResult> stridesPhys(info->physicalRank(), builder.getIndexAttr(1));

    // Create the new physical subview
    // Result type is inferred - rank reduction happens for dims with size=1
    auto newSubview = builder.create<memref::SubViewOp>(
        loc, sourcePhys, offsetsPhys, sizesPhys, stridesPhys);

    // Add mapping from original result to new result
    valueMapping.map(op.getResult(), newSubview.getResult());

    // Add layout info for the new subview result so nested subviews can find it
    valueMap[newSubview.getResult()] = info;

    LLVM_DEBUG(llvm::dbgs() << " Transformed subview: "
                 << op.getSourceType() << " -> " << newSubview.getType() << "\n");

    return newSubview;
  }
  
  /// Transform a memref allocation to physical layout shape
  ///
  /// Post-bufferization: transforms memref.alloc from R-D to (R+2)-D shape
  /// Input:  %alloc = memref.alloc() : memref<d0x...xd_{R-1}xf32, #nisa.mem<sbuf>>
  /// Output: %alloc_phys = memref.alloc() : memref<t0 x nB0 x ... x nB_{R-1} x t_{R-1} x f32, #nisa.mem<sbuf>>
  void transformAllocation(OpBuilder &builder, LayoutInfo &info, IRMapping &valueMapping) {
    Value origValue = info.originalValue;
    auto physShape = info.getPhysicalShape();
    
    // Post-bufferization: must be memref.alloc
    auto allocOp = origValue.getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
      llvm::errs() << "[LegalizeLayout] Error: expected memref.alloc but got ";
      if (auto defOp = origValue.getDefiningOp())
        llvm::errs() << defOp->getName() << "\n";
      else
        llvm::errs() << "block argument\n";
      hasError = true;
      return;
    }
    
    auto origType = allocOp.getType();

    // Determine layout for the physical memref.
    //
    // For narrow SBUF buffers (free dim < 128) with multiple tile rows,
    // default row-major strides make the partition dimension non-contiguous.
    // Example: shape [128, 2, 1, 1] gets strides [2, 1, 1, 1], so accessing
    // tile 1 gives stride-2 partition access [[2,128],[1,1]] — hardware rejects
    // because the free dim (1 element) doesn't fill a full partition width (128).
    // When free dim = 128, the DMA engine handles arbitrary partition strides.
    //
    // Fix: use a partition-contiguous layout where dim 0 has stride 1 and the
    // block dimensions are outer. For [128, 2, 1, 1] this gives strides
    // [1, 128, 128, 1], so each tile's partitions are contiguous in memory.
    MemRefLayoutAttrInterface layout;
    int64_t R = info.rank();
    int64_t tileN = info.tileSize[R - 1];
    int64_t numBlocksM = info.numBlocks[0];
    if (tileN < 128 && numBlocksM > 1) {
      int64_t t0 = info.tileSize[0];
      int64_t physRank = physShape.size();
      SmallVector<int64_t> strides(physRank);
      // Partition dim (first) is contiguous
      strides[0] = 1;
      // Free dim (last): stride = t0 so that partition elements are contiguous
      // for each free index (avoids aliasing with partition stride in flat memory)
      strides[physRank - 1] = t0;
      // Block dims from right to left: each block region is t0 * tileN elements
      int64_t stride = t0 * tileN;
      for (int64_t i = R; i >= 1; --i) {
        strides[i] = stride;
        stride *= physShape[i];
      }
      layout = StridedLayoutAttr::get(builder.getContext(), /*offset=*/0, strides);
      LLVM_DEBUG(llvm::dbgs() << " Using partition-contiguous strides for narrow buffer\n");
    }

    builder.setInsertionPoint(allocOp);
    Location loc = allocOp.getLoc();

    // Always allocate with default (identity) layout — LLVM lowering requires
    // contiguous allocs. For strided layouts, apply a reinterpret_cast after.
    auto defaultType = MemRefType::get(
        physShape,
        origType.getElementType(),
        /*layout=*/nullptr,
        origType.getMemorySpace()
    );

    auto newAlloc = builder.create<memref::AllocOp>(
        loc,
        defaultType,
        /*dynamicSizes=*/ValueRange{},
        /*symbolOperands=*/ValueRange{},
        allocOp.getAlignmentAttr()
    );

    Value result = newAlloc.getResult();

    // For narrow buffers: reinterpret_cast to the partition-contiguous layout
    if (layout) {
      auto stridedType = MemRefType::get(
          physShape,
          origType.getElementType(),
          layout,
          origType.getMemorySpace()
      );
      SmallVector<OpFoldResult> sizes, strides;
      for (auto s : physShape)
        sizes.push_back(builder.getIndexAttr(s));
      auto stridedAttr = cast<StridedLayoutAttr>(layout);
      for (auto s : stridedAttr.getStrides())
        strides.push_back(builder.getIndexAttr(s));
      auto reinterpret = builder.create<memref::ReinterpretCastOp>(
          loc, stridedType, newAlloc.getResult(),
          /*offset=*/builder.getIndexAttr(0), sizes, strides);
      result = reinterpret.getResult();
    }

    valueMapping.map(origValue, result);
    // Also self-map the legalized alloc so lookThroughCastAndResolve can
    // identify it when tracing through collapse_shape ops from Step 1.5.
    // Without this, the legalized alloc is only a VALUE in the mapping
    // (not a KEY), so lookupOrNull would miss it.
    valueMapping.map(result, result);
    LLVM_DEBUG(llvm::dbgs() << " Transformed memref.alloc to physical layout: "
                 << result.getType() << "\n");
  }
  
  /// Collect all memref operations that need transformation via BFS
  /// 
  /// Post-bufferization: memrefs are modified in-place, so we follow:
  ///   alloc -> copy (src or dest) -> subview -> linalg
  ///
  /// Returns a set of ops that need transformation
  /// (iteration order will be determined by func.walk for topological order)
  llvm::SmallPtrSet<Operation*, 16> collectOpsToTransform(SmallVector<LayoutInfo> &layoutInfos,
                                                          IRMapping &valueMapping) {
    llvm::SmallPtrSet<Operation*, 16> opsToTransform;
    llvm::SmallPtrSet<Value, 16> visited;
    std::queue<Value> worklist;

    for (auto &info : layoutInfos) {
      worklist.push(info.originalValue);
    }

    while (!worklist.empty()) {
      Value current = worklist.front();
      worklist.pop();

      if (visited.contains(current))
        continue;
      visited.insert(current);

      // For each use of current value
      for (OpOperand &use : current.getUses()) {
        Operation *user = use.getOwner();

        if (opsToTransform.contains(user))
          continue;

        // Skip dealloc and annotate ops
        if (isa<memref::DeallocOp, nkipy::AnnotateOp>(user)) {
          continue;
        }

        // Add to set if it's a copy or subview that needs transformation
        if (isa<memref::CopyOp, memref::SubViewOp>(user)) {
          opsToTransform.insert(user);
        }

        // Follow through subview result - linalg ops use this
        if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
          worklist.push(subviewOp.getResult());
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << " Collected " << opsToTransform.size()
                 << " ops to transform\n");
    return opsToTransform;
  }
  
  /// Tile HBM↔SBUF copy and transpose operations
  ///
  /// memref.copy and linalg.transpose require same-rank src/dst.
  /// When one operand is R-D HBM and the other is (R+2)-D SBUF, we need to:
  /// 1. Generate a tiled R-level loop nest (over block dimensions)
  /// 2. Create subviews of both R-D HBM and (R+2)-D SBUF for each tile
  /// 3. Collapse SBUF subview to R-D, then copy/transpose tile by tile
  ///
  /// Cases handled:
  /// - HBM→SBUF / SBUF→HBM: R-level loop with collapse
  /// - SBUF→SBUF: R-level loop with permuted block indices
  /// - PSUM↔HBM, PSUM↔SBUF: No transform needed (already tile-sized)
  void tileCopyAndTranspose(func::FuncOp func, SmallVector<LayoutInfo> &layoutInfos,
                            IRMapping &valueMapping) {
    OpBuilder builder(func.getContext());
    auto valueMap = buildValueToLayoutMap(layoutInfos);
    
    // Collect copy and transpose ops to tile (can't modify while walking)
    SmallVector<memref::CopyOp> copiesToTile;
    SmallVector<linalg::TransposeOp> transposesToTile;
    
    func.walk([&](Operation *op) {
      if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        // Check if this copy needs tiling (HBM↔SBUF transfer)
        Value src = copyOp.getSource();
        Value dst = copyOp.getTarget();
        auto srcType = cast<MemRefType>(src.getType());
        auto dstType = cast<MemRefType>(dst.getType());
        
        if (needsTiledTransfer(srcType, dstType)) {
          copiesToTile.push_back(copyOp);
        }
      } else if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
        Value input = transposeOp.getDpsInputs()[0];
        Value output = transposeOp.getDpsInits()[0];
        // Look through casts (SBUF→SBUF transpose input may be a cast result)
        Value inputBase = lookThroughCast(input);
        auto inputBaseType = cast<MemRefType>(inputBase.getType());
        auto outputType = cast<MemRefType>(output.getType());

        // Tile HBM↔SBUF transposes (existing) and SBUF→SBUF transposes (new)
        if (needsTiledTransfer(inputBaseType, outputType) ||
            (isSbuf(inputBaseType.getMemorySpace()) &&
             isSbuf(outputType.getMemorySpace()))) {
          transposesToTile.push_back(transposeOp);
        }
      }
    });
    
    LLVM_DEBUG(llvm::dbgs() << " Found " << copiesToTile.size() 
                 << " copies and " << transposesToTile.size() 
                 << " transposes to tile\n");
    
    // Process copies
    for (auto copyOp : copiesToTile) {
      tileMemrefCopy(builder, copyOp, valueMapping, valueMap);
      if (hasError) return;
    }
    
    // Process transposes  
    for (auto transposeOp : transposesToTile) {
      tileTranspose(builder, transposeOp, valueMapping, valueMap);
      if (hasError) return;
    }
  }
  
  /// Tile a memref.copy between R-D HBM and (R+2)-D SBUF
  ///
  /// Generates an R-level loop nest (one scf.for per block dimension).
  /// For each block: creates an R-D HBM subview and an (R+2)-D SBUF subview,
  /// collapses the SBUF subview to R-D, then copies.
  /// Look through collapse_shape/expand_shape to find a value that has a
  /// mapping in valueMapping.  This handles the pattern created by Phase 0
  /// (foldReshapeIntoAlloc) where a collapse_shape sits between the legalized
  /// alloc and the copy.
  Value resolveToMappedValue(Value v, IRMapping &valueMapping) {
    // Direct lookup first
    if (Value mapped = valueMapping.lookupOrNull(v))
      return mapped;

    // Look through collapse_shape: the source alloc may be mapped
    if (auto collapseOp = v.getDefiningOp<memref::CollapseShapeOp>()) {
      if (Value mapped = valueMapping.lookupOrNull(collapseOp.getSrc()))
        return mapped;
    }

    // Look through expand_shape
    if (auto expandOp = v.getDefiningOp<memref::ExpandShapeOp>()) {
      if (Value mapped = valueMapping.lookupOrNull(expandOp.getSrc()))
        return mapped;
    }

    return v;
  }

  void tileMemrefCopy(OpBuilder &builder, memref::CopyOp op,
                      IRMapping &valueMapping,
                      DenseMap<Value, LayoutInfo*> &valueMap) {
    Value src = op.getSource();
    Value dst = op.getTarget();

    // Look up mapped values (transformations from Phase 2)
    // Also look through collapse_shape/expand_shape created by Phase 0
    Value srcMapped = resolveToMappedValue(src, valueMapping);
    Value dstMapped = resolveToMappedValue(dst, valueMapping);

    auto srcType = cast<MemRefType>(srcMapped.getType());
    auto dstType = cast<MemRefType>(dstMapped.getType());

    // Determine which is HBM (R-D) and which is SBUF ((R+2)-D after transformation)
    bool srcIsSbuf = isSbuf(srcType.getMemorySpace());
    bool dstIsSbuf = isSbuf(dstType.getMemorySpace());

    Value bufHBM = srcIsSbuf ? dst : src;           // HBM stays R-D
    Value bufSBUF = srcIsSbuf ? srcMapped : dstMapped; // SBUF is now (R+2)-D
    auto bufHBMType = cast<MemRefType>(bufHBM.getType());
    auto bufSBUFType = cast<MemRefType>(bufSBUF.getType());

    int64_t hbmRank = bufHBMType.getRank();

    // If SBUF wasn't transformed (partition dim ≤ 128), skip tiling
    if (bufSBUFType.getRank() <= hbmRank) {
      LLVM_DEBUG(llvm::dbgs() << " Skipping tiling for copy (SBUF not transformed): "
                   << srcType << " -> " << dstType << "\n");
      return;
    }

    // Extract tile info from physical shape: [partTile, nB_0, ..., nB_{R-1}, freeTile]
    auto physShape = bufSBUFType.getShape();
    int64_t R = physShape.size() - 2;  // logical rank
    int64_t partTile = physShape[0];
    int64_t freeTile = physShape[R + 1];

    // Collect tile sizes for each logical dim from the physical shape
    // tileSize[0] = partTile, tileSize[R-1] = freeTile, middle = 1
    SmallVector<int64_t> tileSizePerDim(R);
    tileSizePerDim[0] = partTile;
    for (int64_t i = 1; i < R - 1; i++)
      tileSizePerDim[i] = 1;  // middle dims always tile=1
    if (R > 1)
      tileSizePerDim[R - 1] = freeTile;

    // Check if the copy's SBUF operand goes through a collapse_shape (Phase 0
    // pattern).  When Phase 0 folds a reshape into the alloc, the copy still
    // operates at the original (lower) rank via a collapse_shape view.  We need
    // the reassociation to build the HBM subview at its actual rank.
    SmallVector<SmallVector<int64_t>> hbmReassoc;
    Value sbufCopyOperand = srcIsSbuf ? src : dst;
    if (hbmRank < R) {
      if (auto collapseOp = sbufCopyOperand.getDefiningOp<memref::CollapseShapeOp>()) {
        for (auto &indices : collapseOp.getReassociationIndices())
          hbmReassoc.push_back(SmallVector<int64_t>(indices.begin(), indices.end()));
      }
      if (hbmReassoc.empty()) {
        llvm::errs() << "[LegalizeLayout] Error: HBM rank " << hbmRank
                     << " < SBUF logical rank " << R
                     << " but no collapse_shape found\n";
        hasError = true;
        return;
      }
    }

    builder.setInsertionPoint(op);
    Location loc = op.getLoc();

    // Generate R-level nested loop nest (one scf.for per block dim)
    SmallVector<int64_t> numBlocksVec(physShape.begin() + 1, physShape.begin() + 1 + R);
    auto nest = createBlockLoopNest(builder, loc, numBlocksVec);
    auto &blockIdxVars = nest.ivs;
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);

    // Build HBM subview.
    // When hbmRank == R: straightforward R-D subview.
    // When hbmRank < R (Phase 0 pattern): build hbmRank-D subview using the
    // collapse_shape reassociation to map R block indices to hbmRank HBM dims.
    SmallVector<OpFoldResult> offsetsHBM, sizesHBM, stridesHBM;
    if (hbmReassoc.empty()) {
      // Normal case: HBM and SBUF have the same logical rank
      for (int64_t i = 0; i < R; i++) {
        if (tileSizePerDim[i] == 1) {
          offsetsHBM.push_back(OpFoldResult(blockIdxVars[i]));
        } else {
          Value tileSizeVal = builder.create<arith::ConstantIndexOp>(loc, tileSizePerDim[i]);
          Value offset = builder.create<arith::MulIOp>(loc, blockIdxVars[i], tileSizeVal);
          offsetsHBM.push_back(OpFoldResult(offset));
        }
        sizesHBM.push_back(builder.getIndexAttr(tileSizePerDim[i]));
        stridesHBM.push_back(builder.getIndexAttr(1));
      }
    } else {
      // Phase 0 pattern: HBM has fewer dims than SBUF logical rank.
      // Each HBM dim corresponds to a group of SBUF logical dims via the
      // collapse_shape reassociation.
      // For each group, compute:
      //   size = product of tileSizePerDim[i] for i in group
      //   offset = linearized index from block indices within the group
      for (auto &group : hbmReassoc) {
        int64_t combinedSize = 1;
        for (int64_t idx : group)
          combinedSize *= tileSizePerDim[idx];

        // Compute linearized offset within this group.
        // offset = sum_i( blockIdx[group[i]] * product(tileSizePerDim[group[j]] for j>i) )
        Value offset = nullptr;
        int64_t innerProduct = combinedSize;
        for (int64_t i = 0; i < (int64_t)group.size(); i++) {
          int64_t dimIdx = group[i];
          innerProduct /= tileSizePerDim[dimIdx];
          if (innerProduct == 1 && !offset) {
            // Last or only contributing dim: offset += blockIdx * tileSize
            if (tileSizePerDim[dimIdx] == 1) {
              offset = blockIdxVars[dimIdx];
            } else {
              Value ts = builder.create<arith::ConstantIndexOp>(loc, tileSizePerDim[dimIdx]);
              offset = builder.create<arith::MulIOp>(loc, blockIdxVars[dimIdx], ts);
            }
          } else if (innerProduct >= 1) {
            int64_t stride = tileSizePerDim[dimIdx] * innerProduct;
            Value strideVal = builder.create<arith::ConstantIndexOp>(loc, stride);
            Value term = builder.create<arith::MulIOp>(loc, blockIdxVars[dimIdx], strideVal);
            offset = offset ? builder.create<arith::AddIOp>(loc, offset, term).getResult()
                            : term;
          }
        }
        if (!offset)
          offset = c0;

        offsetsHBM.push_back(OpFoldResult(offset));
        sizesHBM.push_back(builder.getIndexAttr(combinedSize));
        stridesHBM.push_back(builder.getIndexAttr(1));
      }
    }

    auto tileHBM = builder.create<memref::SubViewOp>(
        loc, bufHBM, offsetsHBM, sizesHBM, stridesHBM);

    // Build (R+2)-D SBUF subview and collapse to 2D
    Value tileSBUFCollapsed = createTileSubviewAndCollapse(
        builder, loc, bufSBUF, partTile, freeTile, R, blockIdxVars);

    // For R>2 (or hbmRank>2), collapse the HBM tile to 2D:
    // [[0, ..., N-2], [N-1]] where N = actual HBM tile rank
    int64_t hbmTileRank = hbmReassoc.empty() ? R : hbmRank;
    Value tileHBM2D = tileHBM;
    if (hbmTileRank > 2) {
      auto reassocHBM = build2DCollapseFromLogical(hbmTileRank);
      tileHBM2D = builder.create<memref::CollapseShapeOp>(
          loc, tileHBM, reassocHBM);
    }

    // Create copy for this tile (both are now 2D)
    if (srcIsSbuf) {
      builder.create<memref::CopyOp>(loc, tileSBUFCollapsed, tileHBM2D);
    } else {
      builder.create<memref::CopyOp>(loc, tileHBM2D, tileSBUFCollapsed);
    }

    LLVM_DEBUG(llvm::dbgs() << " Tiled memref.copy: " << srcType
                 << " -> " << dstType << "\n");

    // Erase original copy
    op.erase();
  }

  /// Decompose linalg.fill on HBM into SBUF fill + tiled copy to HBM.
  ///
  /// nisa.memset only supports SBUF/PSUM destinations, so a fill on HBM must
  /// be decomposed before linalg-to-nisa.  The pattern mirrors tileMemrefCopy:
  ///
  ///   %sbuf = memref.alloc() : memref<PxFxf32, #nisa.mem<sbuf>>
  ///   linalg.fill ins(%cst) outs(%sbuf)
  ///   scf.for %i = 0 to numBlocks step 1 {
  ///     %hbm_tile = memref.subview %hbm[%i*P, 0][P, F][1, 1]
  ///     memref.copy %sbuf, %hbm_tile
  ///   }
  ///
  /// where P = min(partition_dim, 128).
  void decomposeHbmFills(func::FuncOp func) {
    SmallVector<linalg::FillOp> fillsToDecompose;

    func.walk([&](linalg::FillOp fillOp) {
      Value output = fillOp.getOutputs()[0];
      auto outputType = dyn_cast<MemRefType>(output.getType());
      if (!outputType)
        return;
      if (!isHbm(outputType.getMemorySpace()))
        return;
      // Only handle static shapes
      if (!outputType.hasStaticShape())
        return;
      // Need at least 2D for SBUF (partition + free)
      if (outputType.getRank() < 2)
        return;
      fillsToDecompose.push_back(fillOp);
    });

    if (fillsToDecompose.empty())
      return;

    LLVM_DEBUG(llvm::dbgs() << " Decomposing " << fillsToDecompose.size()
                 << " linalg.fill on HBM\n");

    OpBuilder builder(func.getContext());

    for (auto fillOp : fillsToDecompose) {
      builder.setInsertionPoint(fillOp);
      Location loc = fillOp.getLoc();

      Value scalarValue = fillOp.getInputs()[0];
      Value hbmBuf = fillOp.getOutputs()[0];
      auto hbmType = cast<MemRefType>(hbmBuf.getType());
      auto hbmShape = hbmType.getShape();
      int64_t rank = hbmType.getRank();

      // Partition dim capped at MAX_PARTITION_DIM (128)
      int64_t partDim = hbmShape[0];
      int64_t partTile = std::min(partDim, MAX_PARTITION_DIM);
      int64_t numBlocks = (partDim + partTile - 1) / partTile;

      // Build SBUF shape: [partTile, numBlocks * hbmShape[1], hbmShape[2], ...]
      // Fold numBlocks into dim 1 so SBUF holds all the data,
      // matching the DMA loading pattern in tileMemrefCopy.
      SmallVector<int64_t> sbufShape;
      sbufShape.push_back(partTile);
      sbufShape.push_back(numBlocks * hbmShape[1]);
      for (int64_t i = 2; i < rank; ++i)
        sbufShape.push_back(hbmShape[i]);

      // Alloc SBUF temp
      auto sbufMemSpace = nkipy::MemSpaceEnumAttr::get(
          builder.getContext(), nkipy::MemSpaceEnum::Sbuf);
      auto sbufType = MemRefType::get(
          sbufShape, hbmType.getElementType(), nullptr, sbufMemSpace);
      auto sbufAlloc = builder.create<memref::AllocOp>(loc, sbufType);

      // Fill the entire SBUF with the constant
      builder.create<linalg::FillOp>(loc, scalarValue, sbufAlloc.getResult());

      if (numBlocks == 1) {
        // Fits in one SBUF tile — single copy, no loop needed
        builder.create<memref::CopyOp>(loc, sbufAlloc.getResult(), hbmBuf);
      } else {
        // Build scf.for loop over blocks, subviewing both SBUF and HBM
        int64_t freeDim = hbmShape[1];

        Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
        Value numBlocksVal =
            builder.create<arith::ConstantIndexOp>(loc, numBlocks);
        Value partTileVal =
            builder.create<arith::ConstantIndexOp>(loc, partTile);
        Value freeDimVal =
            builder.create<arith::ConstantIndexOp>(loc, freeDim);

        auto loop = builder.create<scf::ForOp>(loc, c0, numBlocksVal, c1);
        builder.setInsertionPointToStart(loop.getBody());
        Value iv = loop.getInductionVar();

        // SBUF subview: offsets=[0, iv*freeDim, 0, ...],
        //               sizes=[partTile, freeDim, hbmShape[2], ...]
        Value sbufDim1Offset =
            builder.create<arith::MulIOp>(loc, iv, freeDimVal);
        SmallVector<OpFoldResult> sbufOffsets, sbufSizes, sbufStrides;
        sbufOffsets.push_back(builder.getIndexAttr(0));
        sbufOffsets.push_back(OpFoldResult(sbufDim1Offset));
        sbufSizes.push_back(builder.getIndexAttr(partTile));
        sbufSizes.push_back(builder.getIndexAttr(freeDim));
        sbufStrides.push_back(builder.getIndexAttr(1));
        sbufStrides.push_back(builder.getIndexAttr(1));
        for (int64_t i = 2; i < rank; ++i) {
          sbufOffsets.push_back(builder.getIndexAttr(0));
          sbufSizes.push_back(builder.getIndexAttr(hbmShape[i]));
          sbufStrides.push_back(builder.getIndexAttr(1));
        }
        auto sbufTile = builder.create<memref::SubViewOp>(
            loc, sbufAlloc.getResult(), sbufOffsets, sbufSizes, sbufStrides);

        // HBM subview: offsets=[iv*partTile, 0, ...],
        //              sizes=[partTile, freeDim, hbmShape[2], ...]
        Value hbmPartOffset =
            builder.create<arith::MulIOp>(loc, iv, partTileVal);
        SmallVector<OpFoldResult> hbmOffsets, hbmSizes, hbmStrides;
        hbmOffsets.push_back(OpFoldResult(hbmPartOffset));
        hbmSizes.push_back(builder.getIndexAttr(partTile));
        hbmStrides.push_back(builder.getIndexAttr(1));
        for (int64_t i = 1; i < rank; ++i) {
          hbmOffsets.push_back(builder.getIndexAttr(0));
          hbmSizes.push_back(builder.getIndexAttr(hbmShape[i]));
          hbmStrides.push_back(builder.getIndexAttr(1));
        }
        auto hbmTile = builder.create<memref::SubViewOp>(
            loc, hbmBuf, hbmOffsets, hbmSizes, hbmStrides);

        // Copy SBUF tile → HBM tile
        builder.create<memref::CopyOp>(loc, sbufTile, hbmTile);

        builder.setInsertionPointAfter(loop);
      }

      LLVM_DEBUG(llvm::dbgs() << " Decomposed linalg.fill on HBM "
                   << hbmType << " -> SBUF " << sbufType << "\n");

      // Erase original fill
      fillOp.erase();
    }
  }

  /// Tile a linalg.transpose between HBM and SBUF (or SBUF→SBUF)
  ///
  /// Generalized for rank-R tensors with (R+2)-D physical layout.
  /// Generates an R-level loop nest over block indices, applies permutation
  /// to map output block indices to source block indices.
  void tileTranspose(OpBuilder &builder, linalg::TransposeOp op,
                     IRMapping &valueMapping,
                     DenseMap<Value, LayoutInfo*> &valueMap) {
    Value input = op.getDpsInputs()[0];
    Value output = op.getDpsInits()[0];

    // Look through casts (and collapse_shape if the base alloc was legalized)
    // to find the base alloc for valueMapping lookup.
    // This handles chains like: legalized_alloc → collapse_shape → cast → transpose
    // where Step 1.5 replaced the Phase 0 collapse with one from the legalized alloc.
    Value inputBase = lookThroughCastAndResolve(input, valueMapping);

    // Look up mapped values (transformations from Phase 2)
    Value inputMapped = valueMapping.lookupOrDefault(inputBase);
    Value outputMapped = valueMapping.lookupOrDefault(output);

    auto inputMappedType = cast<MemRefType>(inputMapped.getType());
    auto outputMappedType = cast<MemRefType>(outputMapped.getType());

    // Determine memory spaces
    bool inputIsSbuf = isSbuf(inputMappedType.getMemorySpace());
    bool outputIsSbuf = isSbuf(outputMappedType.getMemorySpace());

    auto permutation = op.getPermutation();

    // Handle SBUF→SBUF transpose: at least one side is (R+2)-D after legalization
    if (inputIsSbuf && outputIsSbuf) {
      auto dstPhysShape = outputMappedType.getShape();
      auto srcPhysShape = inputMappedType.getShape();

      // If NEITHER side was expanded to physical layout (both mapped ranks
      // equal the original ranks), leave the transpose for linalg-to-nisa.
      // This handles boundary transposes from canonicalize-partition-dim
      // which operate on tile-sized buffers (partition dim ≤ 128).
      int64_t origInputRank = cast<MemRefType>(input.getType()).getRank();
      int64_t origOutputRank = cast<MemRefType>(output.getType()).getRank();
      if (inputMappedType.getRank() == origInputRank &&
          outputMappedType.getRank() == origOutputRank) {
        LLVM_DEBUG(llvm::dbgs() << " Skipping SBUF→SBUF transpose "
                     << "(not expanded to physical layout): "
                     << inputMappedType << " -> " << outputMappedType << "\n");
        return;
      }

      bool inputLegalized = (inputMappedType.getRank() != origInputRank);
      bool outputLegalized = (outputMappedType.getRank() != origOutputRank);

      // Determine physical parameters from the legalized side.
      // If both legalized: use output. If only one: use that one.
      auto physRefShape = outputLegalized ? dstPhysShape : srcPhysShape;
      int64_t R = physRefShape.size() - 2;  // logical rank
      int64_t partTile = physRefShape[0];
      int64_t freeTile = physRefShape[R + 1];

      builder.setInsertionPoint(op);
      Location loc = op.getLoc();

      // R-level loop nest over block indices (from the legalized side)
      SmallVector<int64_t> numBlocksRef(physRefShape.begin() + 1,
                                        physRefShape.begin() + 1 + R);
      auto nest = createBlockLoopNest(builder, loc, numBlocksRef);

      // Tile sizes per logical dim (for R-D subview of non-legalized side)
      SmallVector<int64_t> tileSizePerDim(R);
      tileSizePerDim[0] = partTile;
      for (int64_t i = 1; i < R - 1; i++)
        tileSizePerDim[i] = 1;
      if (R > 1)
        tileSizePerDim[R - 1] = freeTile;

      // Helper: create a tile from the legalized (R+2)-D side
      auto makeLegalizedTile = [&](Value buf, ArrayRef<Value> ivs) {
        return createTileSubviewAndCollapse(
            builder, loc, buf, partTile, freeTile, R, ivs);
      };

      // Helper: create a tile from the non-legalized R-D side (plain subview)
      auto makeNonLegalizedTile = [&](Value buf, ArrayRef<Value> ivs) -> Value {
        SmallVector<OpFoldResult> offsets, sizes, strides;
        for (int64_t i = 0; i < R; i++) {
          if (tileSizePerDim[i] == 1) {
            offsets.push_back(OpFoldResult(ivs[i]));
          } else {
            Value ts = builder.create<arith::ConstantIndexOp>(loc, tileSizePerDim[i]);
            offsets.push_back(OpFoldResult(
                builder.create<arith::MulIOp>(loc, ivs[i], ts)));
          }
          sizes.push_back(builder.getIndexAttr(tileSizePerDim[i]));
          strides.push_back(builder.getIndexAttr(1));
        }
        auto tile = builder.create<memref::SubViewOp>(loc, buf, offsets, sizes, strides);
        // Collapse R-D to 2D if needed
        Value result = tile;
        if (R > 2) {
          result = builder.create<memref::CollapseShapeOp>(
              loc, tile, build2DCollapseFromLogical(R));
        }
        return result;
      };

      // Dest tile: straight block indices
      SmallVector<Value> permutedIVs;
      for (int64_t i = 0; i < R; i++)
        permutedIVs.push_back(nest.ivs[permutation[i]]);

      Value dstTile = outputLegalized
          ? makeLegalizedTile(outputMapped, nest.ivs)
          : makeNonLegalizedTile(outputMapped, nest.ivs);

      // Source tile: apply permutation to output block indices
      Value srcTile = inputLegalized
          ? makeLegalizedTile(inputMapped, permutedIVs)
          : makeNonLegalizedTile(inputMapped, permutedIVs);

      // Compute 2D permutation from R-D permutation.
      SmallVector<int64_t> invPerm(R);
      for (int64_t i = 0; i < R; i++)
        invPerm[permutation[i]] = i;
      SmallVector<int64_t> perm2D = (invPerm[0] < invPerm[R - 1])
          ? SmallVector<int64_t>{0, 1}   // partition stays first
          : SmallVector<int64_t>{1, 0};   // partition and free swap

      // If perm2D is identity [0,1], the transpose is a no-op (e.g. swapping
      // dims where one is size 1).  Emit memref.copy instead of a transpose
      // so linalg-to-nisa can convert it to nisa.tensor_copy.
      if (perm2D[0] == 0 && perm2D[1] == 1)
        builder.create<memref::CopyOp>(loc, srcTile, dstTile);
      else
        builder.create<linalg::TransposeOp>(loc, srcTile, dstTile, perm2D);

      LLVM_DEBUG(llvm::dbgs() << " Tiled SBUF→SBUF linalg.transpose: "
                   << inputMappedType << " -> " << outputMappedType << "\n");

      op.erase();
      return;
    }

    // HBM↔SBUF case
    Value bufSBUF = inputIsSbuf ? inputMapped : outputMapped;
    Value bufHBM = inputIsSbuf ? output : input;
    auto bufSBUFType = cast<MemRefType>(bufSBUF.getType());
    auto bufHBMType = cast<MemRefType>(bufHBM.getType());
    int64_t hbmRank = bufHBMType.getRank();

    // If SBUF wasn't transformed, skip tiling
    if (bufSBUFType.getRank() <= hbmRank) {
      LLVM_DEBUG(llvm::dbgs() << " Skipping tiling for transpose (SBUF not transformed): "
                   << inputMappedType << " -> " << outputMappedType << "\n");
      return;
    }

    // Extract tile info from SBUF physical shape
    auto physShape = bufSBUFType.getShape();
    int64_t R = physShape.size() - 2;
    int64_t partTile = physShape[0];
    int64_t freeTile = physShape[R + 1];

    // Collect tile sizes per logical dim
    SmallVector<int64_t> tileSizePerDim(R);
    tileSizePerDim[0] = partTile;
    for (int64_t i = 1; i < R - 1; i++)
      tileSizePerDim[i] = 1;
    if (R > 1)
      tileSizePerDim[R - 1] = freeTile;

    builder.setInsertionPoint(op);
    Location loc = op.getLoc();

    // R-level loop nest over SBUF block indices
    SmallVector<int64_t> numBlocksVec(physShape.begin() + 1,
                                      physShape.begin() + 1 + R);
    auto nest = createBlockLoopNest(builder, loc, numBlocksVec);
    auto &blockIdxVars = nest.ivs;

    // Build SBUF (R+2)-D subview and collapse to 2D
    Value tileSBUFCollapsed = createTileSubviewAndCollapse(
        builder, loc, bufSBUF, partTile, freeTile, R, blockIdxVars);

    // Build HBM R-D subview with permuted offsets
    // For transpose: HBM dim j corresponds to SBUF dim invPerm[j]
    // So HBM offset at dim j = blockIdxVars[invPerm[j]] * tileSizePerDim[invPerm[j]]
    // HBM size at dim j = tileSizePerDim[invPerm[j]]
    SmallVector<int64_t> invPerm(R);
    for (int64_t i = 0; i < R; i++)
      invPerm[permutation[i]] = i;

    SmallVector<OpFoldResult> offsetsHBM, sizesHBM, stridesHBM;
    for (int64_t j = 0; j < R; j++) {
      int64_t srcDim = invPerm[j];
      int64_t tileForDim = tileSizePerDim[srcDim];
      if (tileForDim == 1) {
        offsetsHBM.push_back(OpFoldResult(blockIdxVars[srcDim]));
      } else {
        Value tileSizeVal = builder.create<arith::ConstantIndexOp>(loc, tileForDim);
        Value offset = builder.create<arith::MulIOp>(loc, blockIdxVars[srcDim], tileSizeVal);
        offsetsHBM.push_back(OpFoldResult(offset));
      }
      sizesHBM.push_back(builder.getIndexAttr(tileForDim));
      stridesHBM.push_back(builder.getIndexAttr(1));
    }

    auto tileHBM = builder.create<memref::SubViewOp>(
        loc, bufHBM, offsetsHBM, sizesHBM, stridesHBM);

    // For R>2, collapse HBM R-D to 2D.
    // HBM tile has permuted sizes: non-unit dims at positions permutation[0]
    // (partition) and permutation[R-1] (free). Split into 2 contiguous groups
    // at the first non-unit dim boundary.
    Value tileHBM2D = tileHBM;
    if (R > 2) {
      int64_t partPosInHBM = permutation[0];
      int64_t freePosInHBM = permutation[R - 1];
      int64_t splitAt = std::min(partPosInHBM, freePosInHBM) + 1;
      SmallVector<ReassociationIndices> reassocHBM;
      ReassociationIndices g0, g1;
      for (int64_t i = 0; i < splitAt; i++) g0.push_back(i);
      for (int64_t i = splitAt; i < R; i++) g1.push_back(i);
      reassocHBM = {g0, g1};
      tileHBM2D = builder.create<memref::CollapseShapeOp>(loc, tileHBM, reassocHBM);
    }

    // Compute 2D permutation: check relative order of partition and free dims
    SmallVector<int64_t> perm2D = (invPerm[0] < invPerm[R - 1])
        ? SmallVector<int64_t>{0, 1}
        : SmallVector<int64_t>{1, 0};

    // Create tiled 2D transpose with correct src/dst based on direction
    if (outputIsSbuf) {
      // HBM input → SBUF output
      builder.create<linalg::TransposeOp>(loc, tileHBM2D, tileSBUFCollapsed, perm2D);
    } else {
      // SBUF input → HBM output
      builder.create<linalg::TransposeOp>(loc, tileSBUFCollapsed, tileHBM2D, perm2D);
    }

    LLVM_DEBUG(llvm::dbgs() << " Tiled linalg.transpose: " << inputMappedType
                 << " -> " << outputMappedType << "\n");

    // Erase original transpose
    op.erase();
  }
  
  /// Fix rank mismatches by collapsing all operands directly to 2D.
  ///
  /// After transforming subviews to (R+2)-D, and given that middle dims always
  /// have tile=1, we collapse everything directly to 2D [partTile, freeTile]:
  ///
  /// 1. linalg ops: collapse (R+2)-D SBUF and R-D non-SBUF operands to 2D.
  ///    For R>2, also reconstruct the linalg op with 2D identity maps since
  ///    the original maps have R dims.
  /// 2. memref.copy: collapse the higher-rank operand to 2D to match the other.
  void fixRankMismatches(func::FuncOp func) {
    OpBuilder builder(func.getContext());

    // --- Phase 1: Fix linalg ops ---
    // Only collect ops where an operand's rank doesn't match its indexing map,
    // which signals the layout transformation changed the operand's rank.
    // Ops where rank matches the map (e.g. 3D SBUF fills from
    // decomposeHbmFills) are already consistent and must not be touched.
    SmallVector<linalg::LinalgOp> linalgOpsToFix;
    func.walk([&](linalg::LinalgOp op) {
      auto maps = op.getIndexingMapsArray();
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        auto memrefType = dyn_cast<MemRefType>(op->getOperand(i).getType());
        if (!memrefType)
          continue;
        if (memrefType.getRank() != (int64_t)maps[i].getNumResults()) {
          linalgOpsToFix.push_back(op);
          return;
        }
      }
    });

    for (auto linalgOp : linalgOpsToFix) {
      builder.setInsertionPoint(linalgOp);
      Location loc = linalgOp.getLoc();
      auto indexingMaps = linalgOp.getIndexingMapsArray();

      // --- Special case: TransposeOp with one expanded operand ---
      //
      // When canonicalize-partition-dim inserts boundary transposes and one
      // operand comes from a legalized (expanded) alloc while the other is a
      // tile-sized alloc (not legalized), the generic 2D collapse path would
      // create an identity copy where the tile-sized alloc's partition dim
      // doesn't match the tile (e.g., base dim0=1 but tile wants 128).
      //
      // Fix: replace the tile-sized alloc with a 2D alloc whose dim0 matches
      // the partition count, then emit a 2D copy or transpose.
      if (auto transposeOp = dyn_cast<linalg::TransposeOp>(linalgOp.getOperation())) {
        Value input = transposeOp.getInput();
        Value output = transposeOp.getInit();
        auto inputType = cast<MemRefType>(input.getType());
        auto outputType = cast<MemRefType>(output.getType());
        int64_t inputRank = inputType.getRank();
        int64_t outputRank = outputType.getRank();
        int64_t logicalRank = (int64_t)transposeOp.getPermutation().size();

        bool inputExpanded = (inputRank > logicalRank);
        bool outputExpanded = (outputRank > logicalRank);

        // Only handle the asymmetric case: one side expanded, other not
        if (inputExpanded != outputExpanded) {
          Value expandedVal = inputExpanded ? input : output;
          Value nonExpandedVal = inputExpanded ? output : input;
          auto nonExpandedType = cast<MemRefType>(nonExpandedVal.getType());

          // Non-expanded operand must be a direct AllocOp for replacement
          auto nonExpandedAlloc = nonExpandedVal.getDefiningOp<memref::AllocOp>();
          if (nonExpandedAlloc) {
            // Step 1: Collapse expanded operand to 2D
            auto expandedRank = cast<MemRefType>(expandedVal.getType()).getRank();
            auto reassocExpanded = build2DCollapseFromPhysical(expandedRank);
            auto collapsed2D = builder.create<memref::CollapseShapeOp>(
                loc, expandedVal, reassocExpanded);
            auto collapsed2DType = cast<MemRefType>(collapsed2D.getType());

            // Step 2: Compute 2D shape for the non-expanded alloc
            // Use same logic as simplify-linalg's computeCollapse: merge dims
            // up to (and including) the first non-unit dim into dim 0, rest
            // into dim 1.
            auto neShape = nonExpandedType.getShape();
            unsigned firstNonUnit = 0;
            for (unsigned i = 0; i < neShape.size(); i++)
              if (neShape[i] != 1) { firstNonUnit = i; break; }

            int64_t d0 = 1, d1 = 1;
            for (unsigned i = 0; i <= firstNonUnit; i++) d0 *= neShape[i];
            for (unsigned i = firstNonUnit + 1;
                 i < (unsigned)nonExpandedType.getRank(); i++)
              d1 *= neShape[i];

            SmallVector<ReassociationIndices> allocReassoc;
            {
              ReassociationIndices g0, g1;
              for (int64_t i = 0; i <= (int64_t)firstNonUnit; i++)
                g0.push_back(i);
              for (int64_t i = firstNonUnit + 1;
                   i < nonExpandedType.getRank(); i++)
                g1.push_back(i);
              allocReassoc.push_back(g0);
              allocReassoc.push_back(g1);
            }

            auto new2DType = MemRefType::get(
                {d0, d1}, nonExpandedType.getElementType(),
                /*layout=*/nullptr, nonExpandedType.getMemorySpace());

            // Collect downstream copy users of the old alloc BEFORE erasing
            SmallVector<memref::CopyOp> downstreamCopies;
            for (auto *user : nonExpandedAlloc.getResult().getUsers()) {
              if (user == transposeOp.getOperation())
                continue; // skip the transpose itself
              if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
                if (copyOp.getSource() == nonExpandedAlloc.getResult() ||
                    copyOp.getTarget() == nonExpandedAlloc.getResult())
                  downstreamCopies.push_back(copyOp);
              }
            }

            // Create 2D alloc at the old alloc's position
            OpBuilder allocBuilder(nonExpandedAlloc);
            auto new2DAlloc = allocBuilder.create<memref::AllocOp>(
                nonExpandedAlloc.getLoc(), new2DType,
                nonExpandedAlloc.getAlignmentAttr());

            // Step 3: Create 2D copy or transpose replacing the original
            auto cShape = collapsed2DType.getShape();
            bool shapesMatch = (cShape[0] == d0 && cShape[1] == d1);

            if (shapesMatch) {
              // Unit-dim-only movement: just copy
              if (inputExpanded)
                builder.create<memref::CopyOp>(
                    loc, collapsed2D.getResult(), new2DAlloc.getResult());
              else
                builder.create<memref::CopyOp>(
                    loc, new2DAlloc.getResult(), collapsed2D.getResult());
            } else {
              // Real transpose between 2D operands
              if (inputExpanded)
                builder.create<linalg::TransposeOp>(
                    loc, collapsed2D.getResult(), new2DAlloc.getResult(),
                    ArrayRef<int64_t>{1, 0});
              else
                builder.create<linalg::TransposeOp>(
                    loc, new2DAlloc.getResult(), collapsed2D.getResult(),
                    ArrayRef<int64_t>{1, 0});
            }

            // Step 4: Redirect downstream copies to use 2D paths
            // Instead of copying through expand_shape (which loses indexing
            // info in getBaseAndOffsets), collapse the destination to 2D
            // and copy directly from the 2D alloc.
            for (auto copyOp : downstreamCopies) {
              bool isSource =
                  (copyOp.getSource() == nonExpandedAlloc.getResult());
              Value otherOperand =
                  isSource ? copyOp.getTarget() : copyOp.getSource();
              auto otherType = cast<MemRefType>(otherOperand.getType());

              OpBuilder copyBuilder(copyOp);
              if (otherType.getRank() > 2) {
                // Collapse the other operand from R-D to 2D
                auto otherReassoc =
                    build2DCollapseFromLogical(otherType.getRank());
                auto collapsedOther =
                    copyBuilder.create<memref::CollapseShapeOp>(
                        loc, otherOperand, otherReassoc);
                if (isSource)
                  copyBuilder.create<memref::CopyOp>(
                      loc, new2DAlloc.getResult(),
                      collapsedOther.getResult());
                else
                  copyBuilder.create<memref::CopyOp>(
                      loc, collapsedOther.getResult(),
                      new2DAlloc.getResult());
              } else {
                if (isSource)
                  copyBuilder.create<memref::CopyOp>(
                      loc, new2DAlloc.getResult(), otherOperand);
                else
                  copyBuilder.create<memref::CopyOp>(
                      loc, otherOperand, new2DAlloc.getResult());
              }
              copyOp.erase();
            }

            LLVM_DEBUG(llvm::dbgs()
                << " Handled TransposeOp with expanded operand: "
                << inputType << " -> " << outputType
                << " → 2D " << collapsed2DType << " / " << new2DType << "\n");

            transposeOp->erase();
            nonExpandedAlloc.erase();
            continue;
          }
        }
      }

      bool needsReconstruction = false;

      // Collapse operands whose rank doesn't match the indexing map to 2D
      for (unsigned i = 0; i < linalgOp->getNumOperands(); ++i) {
        auto memrefType = dyn_cast<MemRefType>(linalgOp->getOperand(i).getType());
        if (!memrefType)
          continue;

        int64_t operandRank = memrefType.getRank();
        int64_t expectedRank = indexingMaps[i].getNumResults();

        // Skip operands that already match their indexing map
        if (operandRank == expectedRank)
          continue;

        // Choose collapse based on whether operand was transformed
        SmallVector<ReassociationIndices> reassoc;
        if (operandRank > expectedRank) {
          // SBUF operand transformed to (R+2)-D → collapse to 2D
          reassoc = build2DCollapseFromPhysical(operandRank);
        } else {
          // Non-transformed operand (e.g., PSUM R-D where R>2) → collapse to 2D
          reassoc = build2DCollapseFromLogical(operandRank);
        }

        auto collapsed = builder.create<memref::CollapseShapeOp>(
            loc, linalgOp->getOperand(i), reassoc);
        linalgOp->setOperand(i, collapsed.getResult());

        LLVM_DEBUG(llvm::dbgs() << " Collapsed operand " << i
                     << " of " << linalgOp->getName() << ": "
                     << memrefType << " -> " << collapsed.getType() << "\n");

        if (expectedRank > 2)
          needsReconstruction = true;
      }

      // For R>2: the linalg op's indexing maps still have R dims but operands
      // are now 2D. Reconstruct with 2D identity maps and parallel iterators.
      // This only applies to elementwise ops (identity maps, all parallel).
      if (needsReconstruction) {
        // Before gathering operands, collapse any remaining rank>2 operands
        // to 2D. These are non-legalized operands (rank == expectedRank) that
        // still need collapsing because the op is being reconstructed as 2D.
        // E.g. memref<128x1x128> (tile-sized 3D SBUF, partition ≤ 128) → 2D.
        for (unsigned i = 0; i < linalgOp->getNumOperands(); ++i) {
          auto mt = dyn_cast<MemRefType>(linalgOp->getOperand(i).getType());
          if (!mt || mt.getRank() <= 2)
            continue;
          auto reassoc = build2DCollapseFromLogical(mt.getRank());
          auto collapsed = builder.create<memref::CollapseShapeOp>(
              loc, linalgOp->getOperand(i), reassoc);
          linalgOp->setOperand(i, collapsed.getResult());
          LLVM_DEBUG(llvm::dbgs() << " Collapsed remaining operand " << i
                       << " of " << linalgOp->getName() << ": "
                       << mt << " -> " << collapsed.getType() << "\n");
        }

        // Gather collapsed operands (now all 2D)
        SmallVector<Value> inputs(linalgOp.getDpsInputs());
        SmallVector<Value> outputs(linalgOp.getDpsInits());

        // Try to create a fresh named op of the same type.
        // Named ops (AddOp, SubOp, etc.) infer correct 2D indexing maps
        // automatically from the operand types.
        Operation *newOp = nullptr;
        if (isa<linalg::AddOp>(linalgOp.getOperation())) {
          newOp = builder.create<linalg::AddOp>(loc, inputs, outputs);
        } else if (isa<linalg::SubOp>(linalgOp.getOperation())) {
          newOp = builder.create<linalg::SubOp>(loc, inputs, outputs);
        } else if (isa<linalg::MulOp>(linalgOp.getOperation())) {
          newOp = builder.create<linalg::MulOp>(loc, inputs, outputs);
        } else if (isa<linalg::NegFOp>(linalgOp.getOperation())) {
          newOp = builder.create<linalg::NegFOp>(loc, inputs, outputs);
        } else if (isa<linalg::ExpOp>(linalgOp.getOperation())) {
          newOp = builder.create<linalg::ExpOp>(loc, inputs, outputs);
        } else if (isa<linalg::ReciprocalOp>(linalgOp.getOperation())) {
          newOp = builder.create<linalg::ReciprocalOp>(loc, inputs, outputs);
        } else if (isa<linalg::MaxOp>(linalgOp.getOperation())) {
          newOp = builder.create<linalg::MaxOp>(loc, inputs, outputs);
        } else if (isa<linalg::MinOp>(linalgOp.getOperation())) {
          newOp = builder.create<linalg::MinOp>(loc, inputs, outputs);
        } else {
          // Fall back to linalg.generic for unrecognized ops.
          // Build 2D indexing maps per-operand: if an operand's dim is 1
          // where other operands have a larger dim, use a constant-0
          // (broadcast/reduction) map for that dimension.
          auto ctx = linalgOp->getContext();

          // Determine the "full" 2D shape (max across all operands per dim).
          int64_t fullShape[2] = {1, 1};
          SmallVector<Value> allOperands;
          allOperands.append(inputs.begin(), inputs.end());
          allOperands.append(outputs.begin(), outputs.end());
          for (Value v : allOperands) {
            auto mt = dyn_cast<MemRefType>(v.getType());
            if (!mt || mt.getRank() != 2)
              continue;
            for (int d = 0; d < 2; d++)
              fullShape[d] = std::max(fullShape[d], mt.getShape()[d]);
          }

          // Check if the original op has reduction iterators → dim 1 is
          // a reduction dimension in the collapsed 2D space.
          auto origIterTypes = linalgOp.getIteratorTypesArray();
          bool hasReduction = llvm::any_of(origIterTypes,
              [](utils::IteratorType t) {
                return t == utils::IteratorType::reduction;
              });

          SmallVector<utils::IteratorType> newIterTypes = {
              utils::IteratorType::parallel,
              hasReduction ? utils::IteratorType::reduction
                           : utils::IteratorType::parallel};

          // Build per-operand map: identity if shape matches full,
          // constant-0 for dims that are broadcast/reduced (size 1).
          auto d0 = getAffineDimExpr(0, ctx);
          auto d1 = getAffineDimExpr(1, ctx);
          auto c0 = getAffineConstantExpr(0, ctx);

          SmallVector<AffineMap> newMaps;
          for (Value v : allOperands) {
            auto mt = cast<MemRefType>(v.getType());
            AffineExpr e0 = (mt.getShape()[0] < fullShape[0]) ? c0 : d0;
            AffineExpr e1 = (mt.getShape()[1] < fullShape[1]) ? c0 : d1;
            newMaps.push_back(AffineMap::get(2, 0, {e0, e1}, ctx));
          }

          auto genericOp = builder.create<linalg::GenericOp>(
              loc, /*resultTypes=*/TypeRange{}, inputs, outputs,
              newMaps, newIterTypes);
          genericOp.getRegion().takeBody(linalgOp->getRegion(0));
          newOp = genericOp;
        }

        // Copy relevant attributes (e.g., nkipy.op_id)
        for (auto attr : linalgOp->getAttrs()) {
          if (attr.getName().strref().starts_with("nkipy."))
            newOp->setAttr(attr.getName(), attr.getValue());
        }

        LLVM_DEBUG(llvm::dbgs() << " Reconstructed " << linalgOp->getName()
                     << " as " << newOp->getName() << " with 2D operands\n");

        linalgOp->erase();
      }
    }

    // --- Phase 2: Fix memref.copy rank mismatches ---
    // Only fix copies involving at least one SBUF operand (affected by layout
    // transformation).  HBM↔HBM copies at rank > 2 are perfectly valid and
    // must not be collapsed.
    SmallVector<memref::CopyOp> copiesToFix;
    func.walk([&](memref::CopyOp copyOp) {
      auto srcType = cast<MemRefType>(copyOp.getSource().getType());
      auto dstType = cast<MemRefType>(copyOp.getTarget().getType());
      int64_t srcRank = srcType.getRank();
      int64_t dstRank = dstType.getRank();

      // Only collect copies with an actual rank mismatch (one side was
      // transformed to (R+2)-D by the layout pass while the other stayed R-D).
      // Same-rank copies are valid regardless of rank and must not be touched
      // (e.g. 3D SBUF temps created by decomposeHbmFills).
      if (srcRank != dstRank)
        copiesToFix.push_back(copyOp);
    });

    for (auto copyOp : copiesToFix) {
      Value src = copyOp.getSource();
      Value dst = copyOp.getTarget();
      auto srcType = cast<MemRefType>(src.getType());
      auto dstType = cast<MemRefType>(dst.getType());
      int64_t srcRank = srcType.getRank();
      int64_t dstRank = dstType.getRank();

      builder.setInsertionPoint(copyOp);

      // Collapse each operand to 2D if rank > 2.
      // - Higher-rank side was transformed to physical layout → use physical collapse
      // - Equal/lower-rank side is untransformed → use logical collapse
      // Safety: middle dims must be 1 (design constraint) for collapse to produce
      // matching 2D shapes [partTile, freeTile] from both sides.
      if (srcRank > 2) {
        // Verify middle dims are 1 before collapsing
        auto srcShape = srcType.getShape();
        bool middleDimsUnit = true;
        for (int64_t i = 1; i < srcRank - 1; i++) {
          if (srcShape[i] != 1) { middleDimsUnit = false; break; }
        }
        if (!middleDimsUnit) {
          llvm::errs() << "[LegalizeLayout] Error: copy src has non-unit middle dims, "
                       << "cannot safely collapse to 2D: " << srcType << "\n";
          hasError = true;
          return;
        }
        auto reassoc = (srcRank > dstRank)
            ? build2DCollapseFromPhysical(srcRank)
            : build2DCollapseFromLogical(srcRank);
        auto collapsed = builder.create<memref::CollapseShapeOp>(
            copyOp.getLoc(), src, reassoc);
        copyOp->setOperand(0, collapsed.getResult());
        LLVM_DEBUG(llvm::dbgs() << " Collapsed copy src: "
                     << srcType << " -> " << collapsed.getType() << "\n");
      }
      if (dstRank > 2) {
        auto dstShape = dstType.getShape();
        bool middleDimsUnit = true;
        for (int64_t i = 1; i < dstRank - 1; i++) {
          if (dstShape[i] != 1) { middleDimsUnit = false; break; }
        }
        if (!middleDimsUnit) {
          llvm::errs() << "[LegalizeLayout] Error: copy dst has non-unit middle dims, "
                       << "cannot safely collapse to 2D: " << dstType << "\n";
          hasError = true;
          return;
        }
        auto reassoc = (dstRank > srcRank)
            ? build2DCollapseFromPhysical(dstRank)
            : build2DCollapseFromLogical(dstRank);
        auto collapsed = builder.create<memref::CollapseShapeOp>(
            copyOp.getLoc(), dst, reassoc);
        copyOp->setOperand(1, collapsed.getResult());
        LLVM_DEBUG(llvm::dbgs() << " Collapsed copy dst: "
                     << dstType << " -> " << collapsed.getType() << "\n");
      }
    }

    LLVM_DEBUG(llvm::dbgs() << " Fixed " << linalgOpsToFix.size()
                 << " linalg ops and " << copiesToFix.size()
                 << " copies for rank mismatches\n");
  }
  
  /// Find LayoutInfo for a value by checking all possible mappings
  LayoutInfo* findLayoutInfo(Value val, IRMapping &valueMapping,
                             DenseMap<Value, LayoutInfo*> &valueMap) {
    // Direct lookup
    if (valueMap.count(val))
      return valueMap[val];
    
    // Check if val is mapped and lookup the mapped value
    if (Value mapped = valueMapping.lookupOrNull(val)) {
      if (valueMap.count(mapped))
        return valueMap[mapped];
    }
    
    // Reverse lookup: find original value that maps to val
    for (auto &[origVal, layoutPtr] : valueMap) {
      if (valueMapping.lookupOrNull(origVal) == val) {
        return layoutPtr;
      }
    }
    
    return nullptr;
  }
  
private:
  /// Find all SBUF memrefs that need layout legalization
  ///
  /// Algorithm:
  /// 1. Walk entire function, collect all SBUF memref.alloc ops (rank >= 2)
  /// 2. For each alloc, trace uses through subview chains to ALL linalg ops
  /// 3. Collect operand shapes as candidate tile sizes
  /// 4. Filter out invalid tile sizes (dim0 > 128) 
  /// 5. Verify all valid tile sizes are the same
  /// 6. Create LayoutInfo with the validated tile size
  SmallVector<LayoutInfo> findSbufTensorsToLegalize(func::FuncOp func) {
    SmallVector<LayoutInfo> results;
    
    // Step 1: Collect all SBUF memref.alloc ops (rank >= 2) in the entire function
    SmallVector<memref::AllocOp> sbufAllocs;
    func.walk([&](memref::AllocOp allocOp) {
      auto memrefType = allocOp.getType();
      if (!isSbuf(memrefType.getMemorySpace()))
        return;  // Not SBUF
      if (memrefType.getRank() < 2)
        return;  // Scalars or 1D not supported
      
      sbufAllocs.push_back(allocOp);
      LLVM_DEBUG(llvm::dbgs() << " Found SBUF alloc: " << memrefType << "\n");
    });
    
    // Step 2-6: For each SBUF alloc, determine tile sizes and create LayoutInfo
    for (auto allocOp : sbufAllocs) {
      auto memrefType = allocOp.getType();
      auto origShape = memrefType.getShape();
      int64_t R = memrefType.getRank();

      llvm::errs() << "[LegalizeLayout] Processing SBUF alloc: " << memrefType
                   << " at " << allocOp.getLoc() << "\n";

      // Step 2-3: Trace uses to linalg ops, collect operand shapes as tile sizes
      SmallVector<std::tuple<linalg::LinalgOp, unsigned, SmallVector<int64_t>>> linalgUses;
      traceToLinalgOperands(allocOp.getResult(), linalgUses);

      // Extract just the tile sizes from the results
      SmallVector<SmallVector<int64_t>> tileSizes;
      for (auto &[op, idx, tileShape] : linalgUses) {
        tileSizes.push_back(tileShape);
      }

      if (tileSizes.empty()) {
        llvm::errs() << "  -> Skipping (no linalg uses)\n";
        continue;
      }

      // Step 4: Filter out tile sizes that match the alloc's full shape
      // (these are full-buffer writes like linalg.transpose, not tile accesses)
      // and tile sizes with dim0 > 128 (invalid partition dim).
      SmallVector<SmallVector<int64_t>> validTileSizes;
      SmallVector<int64_t> origShapeVec(origShape.begin(), origShape.end());
      for (auto &tile : tileSizes) {
        if (SmallVector<int64_t>(tile) == origShapeVec) {
          LLVM_DEBUG({
            llvm::dbgs() << "   Filtered out full-buffer tile [";
            llvm::interleave(tile, llvm::dbgs(), "x");
            llvm::dbgs() << "] (matches alloc shape)\n";
          });
        } else if (tile[0] > MAX_PARTITION_DIM) {
          LLVM_DEBUG({
            llvm::dbgs() << "   Filtered out invalid tile [";
            llvm::interleave(tile, llvm::dbgs(), "x");
            llvm::dbgs() << "] (dim0 > 128)\n";
          });
        } else {
          validTileSizes.push_back(tile);
        }
      }

      if (validTileSizes.empty()) {
        llvm::errs() << "  -> Skipping (already at tile size)\n";
        continue;
      }

      // Log all discovered tiles for debugging
      llvm::errs() << "  All tiles found (" << tileSizes.size() << " total, "
                   << validTileSizes.size() << " valid):\n";
      for (auto &tile : tileSizes) {
        llvm::errs() << "    [";
        llvm::interleave(tile, llvm::errs(), ",");
        bool isFiltered = (SmallVector<int64_t>(tile) == origShapeVec) ||
                          (!tile.empty() && tile[0] > MAX_PARTITION_DIM);
        llvm::errs() << "]" << (isFiltered ? " (filtered)" : "") << "\n";
      }

      // Step 5: Verify all valid tile sizes are the same
      auto &refTile = validTileSizes[0];
      for (size_t i = 1; i < validTileSizes.size(); ++i) {
        if (validTileSizes[i] != refTile) {
          llvm::errs() << "[LegalizeLayout] Error: SBUF alloc " << memrefType
                       << " has inconsistent tile sizes: [";
          llvm::interleave(refTile, llvm::errs(), ",");
          llvm::errs() << "] vs [";
          llvm::interleave(validTileSizes[i], llvm::errs(), ",");
          llvm::errs() << "]\n";
          hasError = true;
          return results;
        }
      }

      // Validate rank match and divisibility for all dims
      if ((int64_t)refTile.size() != R) {
        llvm::errs() << "[LegalizeLayout] Error: SBUF alloc " << memrefType
                     << " tile rank " << refTile.size()
                     << " does not match alloc rank " << R << ". Tile: [";
        llvm::interleave(refTile, llvm::errs(), ",");
        llvm::errs() << "]\n";
        hasError = true;
        return results;
      }
      for (int64_t i = 0; i < R; i++) {
        if (origShape[i] % refTile[i] != 0) {
          llvm::errs() << "[LegalizeLayout] Error: SBUF alloc " << memrefType
                       << " dim " << i << " size " << origShape[i]
                       << " not divisible by tile size " << refTile[i]
                       << ". Full tile: [";
          llvm::interleave(refTile, llvm::errs(), ",");
          llvm::errs() << "]\n";
          hasError = true;
          return results;
        }
      }

      // Step 6: Compute numBlocks; skip if all 1 (already tile-sized)
      SmallVector<int64_t> numBlocks;
      for (int64_t i = 0; i < R; i++)
        numBlocks.push_back(origShape[i] / refTile[i]);

      if (llvm::all_of(numBlocks, [](int64_t n) { return n == 1; })) {
        llvm::errs() << "  -> Skipping (numBlocks all 1, already tile-sized)\n";
        continue;
      }

      // Validate: middle tile sizes (dims 1..R-2) must all be 1.
      // The physical shape formula [tileSize[0], numBlocks..., tileSize[R-1]]
      // only stores the first and last tile sizes.  Non-unit middle tiles
      // (e.g. tile=[1,128,64] for a 4x128x64 alloc from multi-head attention)
      // cannot be represented by the physical layout.  These allocs have their
      // partition dim NOT at dim 0 (it's a batch dim) and don't need
      // legalization — they're accessed tile-by-tile via subviews in loops.
      if (R > 2) {
        bool middleTilesUnit = true;
        for (int64_t i = 1; i < R - 1; i++) {
          if (refTile[i] != 1) {
            middleTilesUnit = false;
            break;
          }
        }
        if (!middleTilesUnit) {
          if (numBlocks[0] > 1) {
            // Multi-partition SBUF alloc with non-unit middle tiles:
            // the vector engine cannot selectively address individual
            // partitions, and the physical layout format can't represent
            // non-unit middle tiles.  Convert to SharedHbm so DMA can
            // handle the per-partition access.
            llvm::errs() << "  -> Converting " << memrefType
                         << " to SharedHbm (non-unit middle tile, "
                         << "numBlocks[0]=" << numBlocks[0] << ")\n";
            auto hbmMemSpace = nkipy::MemSpaceEnumAttr::get(
                allocOp.getContext(), nkipy::MemSpaceEnum::SharedHbm);
            auto newType = MemRefType::get(
                memrefType.getShape(), memrefType.getElementType(),
                memrefType.getLayout(), hbmMemSpace);
            OpBuilder builder(allocOp);
            builder.setInsertionPointAfter(allocOp);
            auto newAlloc = builder.create<memref::AllocOp>(
                allocOp.getLoc(), newType, allocOp.getAlignmentAttr());

            // Recreate subview users with updated memspace
            SmallVector<memref::SubViewOp> subviews;
            for (auto *user : allocOp.getResult().getUsers())
              if (auto sv = dyn_cast<memref::SubViewOp>(user))
                subviews.push_back(sv);
            for (auto sv : subviews) {
              OpBuilder svBuilder(sv);
              svBuilder.setInsertionPointAfter(sv);
              auto newSv = svBuilder.create<memref::SubViewOp>(
                  sv.getLoc(), newAlloc.getResult(),
                  sv.getMixedOffsets(), sv.getMixedSizes(),
                  sv.getMixedStrides());
              sv.replaceAllUsesWith(newSv.getResult());
              sv.erase();
            }
            allocOp.replaceAllUsesWith(newAlloc.getResult());
            allocOp.erase();
          } else {
            llvm::errs() << "  -> Skipping " << memrefType
                         << " (non-unit middle tile [";
            llvm::interleave(refTile, llvm::errs(), ",");
            llvm::errs() << "], not supported by physical layout)\n";
          }
          continue;
        }
      }

      LayoutInfo info;
      info.originalValue = allocOp.getResult();
      info.origShape = SmallVector<int64_t>(origShape.begin(), origShape.end());
      info.tileSize = refTile;
      info.numBlocks = numBlocks;
      results.push_back(info);
    }
    
    return results;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeLayoutPass() {
  return std::make_unique<NkipyLegalizeLayoutPass>();
}

} // namespace nkipy
} // namespace mlir
