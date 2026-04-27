//===- ApplyAndStripTransforms.cpp - Run + strip transform sequence -------===//
//
// This pass runs the transform dialect's @__transform_main named sequence on
// the enclosing module (same semantics as upstream --transform-interpreter)
// and then erases the transform module (the NamedSequenceOp and the
// `transform.with_named_sequence` module attribute).
//
// Motivation: after tiling, nothing downstream consumes the transform block,
// but it stays in the IR all the way until prepare-for-nki. The Python-side
// linalg→NISA phase needs to parse the IR through upstream MLIR bindings,
// which don't know about our custom transform op `transform.nkipy.promote_tensor`
// (it lives in our own dialect). Stripping the transform module right after
// interpretation gives the Python phase clean IR.
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct ApplyAndStripTransformsPass
    : public PassWrapper<ApplyAndStripTransformsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyAndStripTransformsPass)

  StringRef getArgument() const final { return "apply-and-strip-transforms"; }

  StringRef getDescription() const final {
    return "Apply @__transform_main named sequence, then erase the transform "
           "module (transform ops + with_named_sequence attr)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Locate the entry point sequence. If no transform module exists (e.g.,
    // kernel had no tilable ops), nothing to do.
    transform::TransformOpInterface entry =
        transform::detail::findTransformEntryPoint(
            module, /*module=*/ModuleOp(),
            transform::TransformDialect::kTransformEntryPointSymbolName);
    if (entry) {
      if (failed(transform::applyTransformNamedSequence(
              module, entry, /*transformModule=*/ModuleOp(),
              transform::TransformOptions()))) {
        module->emitError()
            << "[apply-and-strip-transforms] transform interpretation failed";
        return signalPassFailure();
      }
    }

    // Erase all top-level transform ops (NamedSequenceOps etc.) regardless of
    // whether we found an entry point — if an empty sequence got generated, it
    // still needs to go.
    SmallVector<Operation *> toErase;
    for (Operation &op : module.getBody()->getOperations()) {
      if (isa<transform::TransformDialect>(op.getDialect()))
        toErase.push_back(&op);
    }
    for (Operation *op : toErase)
      op->erase();

    if (module->hasAttr("transform.with_named_sequence"))
      module->removeAttr("transform.with_named_sequence");
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<ModuleOp>> createApplyAndStripTransformsPass() {
  return std::make_unique<ApplyAndStripTransformsPass>();
}

} // namespace nkipy
} // namespace mlir
