//===- InsertSpillReload.cpp - Insert spill/reload for SBUF pressure ====//
//
// This pass analyzes per-partition SBUF memory pressure and inserts spill
// (SBUF→HBM) and reload (HBM→SBUF) operations when capacity is exceeded.
//
// Runs after legalize-layout so SBUF allocs are in physical layout
// [partTile, numBlocks..., freeTile].  The per-partition size is
// total_size / shape[0] (partTile), matching getSbufPartitionUsableSize.
//
// Algorithm:
// 1. Collect all SBUF allocations and compute their sizes
// 2. Perform liveness analysis to find peak memory pressure points
// 3. At high-pressure points, select victims to spill using a heuristic
// 4. Insert memref.copy operations for spill/reload
//
// These copies are lowered to nisa.dma_copy in the LinalgToNisa pass.
//
// Limitations:
// - Only analyzes the function entry block. SBUF pressure that arises
//   exclusively inside a loop body (e.g., a loop-local alloc that exceeds
//   capacity only within its iteration) is not detected. Full loop-body
//   analysis would require per-block traversal and loop-carried liveness.
//   The preferred solution for loop-body pressure is tiling (reduce the working
//   set per iteration) combined with multi-buffering (overlap DMA and compute
//   by keeping only the current and next tile in SBUF simultaneously).
//
//===----------------------------------------------------------------------===//

#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "nkipy/Transforms/HardwareConstants.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "insert-spill-reload"

using namespace mlir;
using nkipy::getNkipyMemSpace;

namespace mlir {
namespace nkipy {

namespace {

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

// Spill heuristic strategy
enum class SpillHeuristic {
  FARTHEST_NEXT_USE,  // Belady's MIN (optimal)
  LRU,                // Least recently used
  SIZE_BASED,         // Largest first
};

//===----------------------------------------------------------------------===//
// Helper: Compute memref size in bytes
//===----------------------------------------------------------------------===//

// Returns the per-partition size of an SBUF memref in bytes.
//
// After legalize-layout, SBUF allocs have physical shape
// [partTile, numBlocks..., freeTile].  Each of the 128 hardware partitions
// holds total_size / partTile bytes, so we divide by shape[0].
static std::optional<int64_t> computePerPartitionSize(MemRefType type) {
  if (!type.hasStaticShape())
    return std::nullopt;

  auto shape = type.getShape();
  if (shape.empty() || shape[0] == 0)
    return std::nullopt;

  int64_t numElements = type.getNumElements() / shape[0];
  unsigned elementBits = type.getElementTypeBitWidth();
  int64_t elementBytes = (elementBits + 7) / 8;

  return numElements * elementBytes;
}

//===----------------------------------------------------------------------===//
// Allocation Info
//===----------------------------------------------------------------------===//

struct AllocationInfo {
  memref::AllocOp allocOp;
  Value value;
  int64_t sizeBytes;
  Operation *firstUse = nullptr;
  Operation *lastUse = nullptr;
  bool isSpilled = false;
  Value spillSlot;  // HBM buffer for spilled data
};

//===----------------------------------------------------------------------===//
// Liveness Analysis (Simplified)
//===----------------------------------------------------------------------===//

class SimpleLivenessAnalysis {
public:
  explicit SimpleLivenessAnalysis(Block *block) : block(block) {
    analyze();
  }

  // Get the first operation that uses this value
  Operation *getFirstUse(Value val) const {
    auto it = uses.find(val);
    if (it == uses.end() || it->second.empty())
      return nullptr;
    return it->second.front();
  }

  // Get the last operation that uses this value
  Operation *getLastUse(Value val) const {
    auto it = uses.find(val);
    if (it == uses.end() || it->second.empty())
      return nullptr;
    return it->second.back();
  }

  // Get the first use of val that comes strictly AFTER point in the block
  Operation *getNextUseAfter(Value val, Operation *point) const {
    auto it = uses.find(val);
    if (it == uses.end())
      return nullptr;
    for (Operation *use : it->second) {
      if (point->isBeforeInBlock(use))
        return use;
    }
    return nullptr;
  }

  // Get the last use of val that comes strictly BEFORE point in the block
  Operation *getLastUseBefore(Value val, Operation *point) const {
    auto it = uses.find(val);
    if (it == uses.end())
      return nullptr;
    Operation *result = nullptr;
    for (Operation *use : it->second) {
      if (use->isBeforeInBlock(point))
        result = use;
    }
    return result;
  }

  // Check if value is live at a given operation
  bool isLive(Value val, Operation *op) const {
    auto first = getFirstUse(val);
    auto last = getLastUse(val);
    if (!first || !last)
      return false;

    // Check if op is between [first, last] in the block
    if (op->getBlock() != block)
      return false;

    return !op->isBeforeInBlock(first) && !last->isBeforeInBlock(op);
  }

private:
  Block *block;
  DenseMap<Value, SmallVector<Operation *>> uses;

  void analyze() {
    for (Operation &op : *block) {
      // Record uses of all operands
      for (Value operand : op.getOperands()) {
        uses[operand].push_back(&op);
      }

      // Recursively analyze nested regions (e.g., loop bodies).
      // Record the ancestor op directly in `block` rather than the nested op
      // itself — isBeforeInBlock requires both ops to be in the same block.
      op.walk([&](Operation *nestedOp) {
        if (nestedOp == &op)
          return;
        for (Value operand : nestedOp->getOperands()) {
          if (operand.getParentBlock() != block)
            continue;
          // Walk up to find the direct child of `block`.
          Operation *ancestor = nkipy::getAncestorInBlock(nestedOp, block);
          // Avoid duplicate entries for the same ancestor.
          if (uses[operand].empty() || uses[operand].back() != ancestor)
            uses[operand].push_back(ancestor);
        }
      });
    }
  }
};

//===----------------------------------------------------------------------===//
// Memory Pressure Tracking
//===----------------------------------------------------------------------===//

struct PressurePoint {
  Operation *op;
  int64_t sbufUsageBytes;
  SmallVector<AllocationInfo *> liveAllocs;
};

static SmallVector<PressurePoint>
computeMemoryPressure(Block *block, ArrayRef<AllocationInfo> allocs) {
  SimpleLivenessAnalysis liveness(block);
  SmallVector<PressurePoint> pressurePoints;

  for (Operation &op : *block) {
    PressurePoint point;
    point.op = &op;
    point.sbufUsageBytes = 0;

    // Check which allocations are live at this point
    for (auto &alloc : allocs) {
      if (liveness.isLive(alloc.value, &op)) {
        point.sbufUsageBytes += alloc.sizeBytes;
        point.liveAllocs.push_back(const_cast<AllocationInfo *>(&alloc));
      }
    }

    pressurePoints.push_back(point);
  }

  return pressurePoints;
}

//===----------------------------------------------------------------------===//
// Spill Decision: Pick victims based on heuristic
//===----------------------------------------------------------------------===//

static SmallVector<AllocationInfo *>
selectSpillVictims(const PressurePoint &point, int64_t capacityBytes,
                   SpillHeuristic heuristic, SimpleLivenessAnalysis &liveness) {
  SmallVector<AllocationInfo *> victims;

  // Compute effective pressure from unspilled live allocs only.  Already-spilled
  // allocs are excluded because they were selected at an earlier pressure point
  // and their contribution has already been accounted for.
  int64_t livePressure = 0;
  SmallVector<AllocationInfo *> candidates;
  for (AllocationInfo *a : point.liveAllocs) {
    if (!a->isSpilled) {
      livePressure += a->sizeBytes;
      candidates.push_back(a);
    }
  }

  if (livePressure <= capacityBytes)
    return victims;  // Effective pressure within capacity

  int64_t excessBytes = livePressure - capacityBytes;
  int64_t spilledBytes = 0;

  switch (heuristic) {
  case SpillHeuristic::SIZE_BASED:
    // Spill largest allocations first
    llvm::sort(candidates, [](AllocationInfo *a, AllocationInfo *b) {
      return a->sizeBytes > b->sizeBytes;
    });
    break;

  case SpillHeuristic::LRU:
    // Spill least recently used: the candidate whose last use before this
    // pressure point is earliest (i.e., most stale).
    llvm::sort(candidates, [&](AllocationInfo *a, AllocationInfo *b) {
      auto aLast = liveness.getLastUseBefore(a->value, point.op);
      auto bLast = liveness.getLastUseBefore(b->value, point.op);
      if (!aLast) return true;   // Never used before this point → most stale
      if (!bLast) return false;
      return aLast->isBeforeInBlock(bLast);  // Earlier last use → more stale
    });
    break;

  case SpillHeuristic::FARTHEST_NEXT_USE:
    // Belady's MIN: spill the value whose next use AFTER this pressure point
    // is farthest away (optimal page-replacement policy).
    llvm::sort(candidates, [&](AllocationInfo *a, AllocationInfo *b) {
      auto aNext = liveness.getNextUseAfter(a->value, point.op);
      auto bNext = liveness.getNextUseAfter(b->value, point.op);
      if (!aNext) return true;   // No future use → spill first
      if (!bNext) return false;
      return !bNext->isBeforeInBlock(aNext);  // Farther next use → spill first
    });
    break;
  }

  // Select victims until we free enough space
  for (AllocationInfo *candidate : candidates) {
    if (spilledBytes >= excessBytes)
      break;
    victims.push_back(candidate);
    spilledBytes += candidate->sizeBytes;
  }

  return victims;
}

//===----------------------------------------------------------------------===//
// Spill/Reload Insertion
//===----------------------------------------------------------------------===//

static Value createSpillSlot(AllocationInfo &alloc, OpBuilder &builder) {
  auto sbufType = cast<MemRefType>(alloc.value.getType());

  // Create HBM memory space attribute
  auto hbmMemSpace = nkipy::MemSpaceEnumAttr::get(
      builder.getContext(), nkipy::MemSpaceEnum::Hbm);

  // Create HBM type with same shape/element type
  auto hbmType = MemRefType::get(sbufType.getShape(), sbufType.getElementType(),
                                  sbufType.getLayout(), hbmMemSpace);

  // Insert HBM allocation after SBUF allocation
  builder.setInsertionPointAfter(alloc.allocOp);
  auto spillSlot =
      builder.create<memref::AllocOp>(alloc.allocOp.getLoc(), hbmType);

  LLVM_DEBUG(llvm::dbgs() << " Created HBM spill slot for "
               << alloc.value << " (size: " << alloc.sizeBytes << " bytes)\n");

  return spillSlot.getResult();
}

static void insertSpill(AllocationInfo &alloc, Operation *insertAfter,
                        OpBuilder &builder) {
  builder.setInsertionPointAfter(insertAfter);
  builder.create<memref::CopyOp>(insertAfter->getLoc(), alloc.value,
                                  alloc.spillSlot);

  LLVM_DEBUG(llvm::dbgs() << " Inserted spill (SBUF→HBM) after "
               << *insertAfter << "\n");
}

static void insertReload(AllocationInfo &alloc, Operation *insertBefore,
                         OpBuilder &builder) {
  builder.setInsertionPoint(insertBefore);
  builder.create<memref::CopyOp>(insertBefore->getLoc(), alloc.spillSlot,
                                  alloc.value);

  LLVM_DEBUG(llvm::dbgs() << " Inserted reload (HBM→SBUF) before "
               << *insertBefore << "\n");
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct InsertSpillReloadPass
    : public InsertSpillReloadBase<InsertSpillReloadPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  // Resolve the SBUF capacity to use:
  //  - sbufCapacityOverride >= 0  → use override directly (for testing)
  //  - otherwise                  → query getSbufPartitionUsableSize(target)
  std::optional<int64_t> resolveSbufCapacity(func::FuncOp func) const {
    if (sbufCapacityOverride >= 0)
      return sbufCapacityOverride;

    StringRef targetStr = target.empty() ? StringRef("trn2") : StringRef(target);
    auto size = nkipy::getSbufPartitionUsableSize(targetStr);
    if (!size) {
      func.emitError("insert-spill-reload: unknown target '") << targetStr << "'";
      return std::nullopt;
    }
    return *size;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Resolve SBUF capacity
    auto capacityOpt = resolveSbufCapacity(func);
    if (!capacityOpt)
      return signalPassFailure();
    int64_t capacityBytes = *capacityOpt;

    LLVM_DEBUG(llvm::dbgs() << " Processing function: " << func.getName()
                 << " (SBUF capacity: " << capacityBytes << " bytes)\n");

    // Phase 1: Collect SBUF allocations
    SmallVector<AllocationInfo> sbufAllocs;
    func.walk([&](memref::AllocOp allocOp) {
      auto memSpace = getNkipyMemSpace(allocOp.getType());
      if (memSpace && *memSpace == nkipy::MemSpaceEnum::Sbuf) {
        AllocationInfo info;
        info.allocOp = allocOp;
        info.value = allocOp.getResult();

        auto sizeOpt = computePerPartitionSize(allocOp.getType());
        if (!sizeOpt) {
          LLVM_DEBUG(llvm::dbgs() << " Warning: Skipping dynamic-shaped "
                          "SBUF allocation\n");
          return;
        }
        info.sizeBytes = *sizeOpt;

        sbufAllocs.push_back(info);

        LLVM_DEBUG(llvm::dbgs() << " Found SBUF alloc: " << info.value
                     << " (" << info.sizeBytes << " bytes)\n");
      }
    });

    if (sbufAllocs.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No SBUF allocations found\n");
      return;
    }

    // Calculate total SBUF usage
    int64_t totalSbufBytes = 0;
    for (const auto &alloc : sbufAllocs) {
      totalSbufBytes += alloc.sizeBytes;
    }

    LLVM_DEBUG(llvm::dbgs() << " Total SBUF usage: " << totalSbufBytes
                 << " bytes (capacity: " << capacityBytes << " bytes)\n");

    if (totalSbufBytes <= capacityBytes) {
      LLVM_DEBUG(llvm::dbgs() << " SBUF usage within capacity, no "
                      "spilling needed\n");
      return;
    }

    // Phase 2: Analyze each block (function body, loop bodies)
    Block &entryBlock = func.getBody().front();
    SimpleLivenessAnalysis liveness(&entryBlock);

    // Compute liveness for each allocation
    for (auto &alloc : sbufAllocs) {
      alloc.firstUse = liveness.getFirstUse(alloc.value);
      alloc.lastUse = liveness.getLastUse(alloc.value);
    }

    // Phase 3: Compute memory pressure at each program point
    auto pressurePoints = computeMemoryPressure(&entryBlock, sbufAllocs);

    // Find peak pressure
    int64_t peakPressure = 0;
    for (auto &point : pressurePoints)
      peakPressure = std::max(peakPressure, point.sbufUsageBytes);

    if (peakPressure <= capacityBytes) {
      LLVM_DEBUG(llvm::dbgs() << " Peak pressure within capacity (due "
                      "to non-overlapping lifetimes)\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << " Peak SBUF pressure: " << peakPressure
                 << " bytes\n");

    // Phase 4: Select spill victims at ALL over-capacity pressure points.
    // Pressure points are visited in program order. Once a victim is marked
    // isSpilled, subsequent pressure points see reduced effective pressure
    // (the victim's bytes are excluded), so each alloc is selected at most once.
    SmallVector<std::pair<AllocationInfo *, Operation *>> toSpill;
    for (auto &point : pressurePoints) {
      auto victims = selectSpillVictims(point, capacityBytes,
                                        SpillHeuristic::FARTHEST_NEXT_USE, liveness);
      for (AllocationInfo *v : victims) {
        if (!v->isSpilled) {
          v->isSpilled = true;
          toSpill.push_back({v, point.op});
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << " Selected " << toSpill.size()
                 << " total victims to spill\n");

    // Phase 5: Insert spill/reload for each (victim, spillPoint) pair.
    OpBuilder builder(func.getContext());

    for (auto [victim, spillPoint] : toSpill) {
      // Create HBM spill slot
      victim->spillSlot = createSpillSlot(*victim, builder);

      LLVM_DEBUG(llvm::dbgs() << " Spilling " << victim->value
                   << " at " << *spillPoint << "\n");

      // Collect uses that come after spillPoint, including uses inside nested
      // regions (e.g., loop bodies).  For each user, walk up the op-parent
      // chain until we reach spillPoint's block, then check ordering.
      // Do this BEFORE inserting the spill so the new memref.copy is not
      // counted as a "use after spill".
      SmallVector<Operation *> usesAfterSpill;
      for (Operation *user : victim->value.getUsers()) {
        Operation *ancestor = user;
        while (ancestor->getBlock() != spillPoint->getBlock())
          ancestor = ancestor->getParentOp();
        if (spillPoint->isBeforeInBlock(ancestor))
          usesAfterSpill.push_back(ancestor);
      }
      // Sort and deduplicate: multiple uses inside the same nested region
      // all map to the same ancestor op.
      llvm::sort(usesAfterSpill, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      usesAfterSpill.erase(
          std::unique(usesAfterSpill.begin(), usesAfterSpill.end()),
          usesAfterSpill.end());

      // Insert spill after spillPoint
      insertSpill(*victim, spillPoint, builder);

      // Insert a single reload before the first use after the spill
      if (!usesAfterSpill.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "Found " << usesAfterSpill.size()
                     << " uses after spill, inserting reload before first use: "
                     << *usesAfterSpill.front() << "\n");
        insertReload(*victim, usesAfterSpill.front(), builder);
      } else {
        LLVM_DEBUG(llvm::dbgs() << " No uses after spill for "
                     << victim->value << " (dead after spill)\n");
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Pass completed\n");
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createInsertSpillReloadPass() {
  return std::make_unique<InsertSpillReloadPass>();
}

} // namespace nkipy
} // namespace mlir
