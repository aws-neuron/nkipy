
#ifndef NKIPY_TRANSFORMS_PASSES_H
#define NKIPY_TRANSFORMS_PASSES_H

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<func::FuncOp>> createAnnotateMemorySpacePass();
std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeReshapePass();
std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizePartitionDimPass();
std::unique_ptr<OperationPass<func::FuncOp>> createAssignLinalgOpIdsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createInferLayoutPass();
std::unique_ptr<OperationPass<ModuleOp>> createKnobDrivenTilingPass();
std::unique_ptr<OperationPass<ModuleOp>> createApplyAndStripTransformsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createEliminateUninitializedCopiesPass();
std::unique_ptr<OperationPass<func::FuncOp>> createEliminateSameMemSpaceCopyPass();
std::unique_ptr<OperationPass<func::FuncOp>> createInsertSpillReloadPass();
std::unique_ptr<OperationPass<func::FuncOp>> createInsertMemRefDeallocPass();
std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeLoopStepPass();
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeLayoutPass();
std::unique_ptr<OperationPass<func::FuncOp>> createSimplifyLinalgPass();
std::unique_ptr<OperationPass<ModuleOp>> createPrepareArithmeticPass();
std::unique_ptr<OperationPass<ModuleOp>> createRemoveRedundantZeroFillPass();
std::unique_ptr<OperationPass<func::FuncOp>> createInlineNkipyReferencePass();


/// Registers all transformation passes
void registerNkipyPasses();

} // namespace nkipy
} // namespace mlir

#endif // NKIPY_TRANSFORMS_PASSES_H
