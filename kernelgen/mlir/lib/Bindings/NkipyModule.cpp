#include "nkipy/Transforms/Passes.h"
#include "nkipy-c/Dialect/Registration.h"
#include "nkipy-c/Dialect/Dialects.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// #include "mlir/Dialect/PDL/IR/PDLOps.h"
// #include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"


namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

using namespace mlir;
using namespace mlir::python;
using namespace nkipy;

// Helper to extract MlirContext from capsules with different package prefixes
static MlirContext extractContextFromCapsule(PyObject *capsule) {
  MlirContext context;
  
  // Try standard mlir.ir.Context capsule name
  void *ptr = PyCapsule_GetPointer(capsule, "mlir.ir.Context._CAPIPtr");
  if (ptr) {
    context.ptr = ptr;
    return context;
  }
  
  // Clear error and try NKI's package prefix
  PyErr_Clear();
  ptr = PyCapsule_GetPointer(capsule, "nki.compiler._internal.ir.Context._CAPIPtr");
  if (ptr) {
    context.ptr = ptr;
    return context;
  }
  
  // Return null context if neither worked
  context.ptr = nullptr;
  return context;
}

NB_MODULE(_nkipy, m) {
  m.doc() = "Nkipy Python Native Extension";

  // register passes
  nkipyMlirRegisterAllPasses();

  auto nkipy_m = m.def_submodule("nkipy");

  nkipy_m.def(
      "register_dialect",
      [](nb::handle contextObj) {
        // Get the _CAPIPtr attribute from any Context object (mlir or nki)
        MlirContext context;
        if (contextObj.is_none()) {
          context = mlirContextCreate();
        } else {
          // Get the _CAPIPtr capsule attribute
          PyObject *capsule = PyObject_GetAttrString(contextObj.ptr(), MLIR_PYTHON_CAPI_PTR_ATTR);
          if (!capsule) {
            throw nb::type_error("Expected an MLIR Context object with _CAPIPtr attribute");
          }
          // Try both mlir.ir.Context and nki.compiler._internal.ir.Context capsule names
          context = extractContextFromCapsule(capsule);
          Py_DECREF(capsule);
          if (mlirContextIsNull(context)) {
            throw nb::type_error("Invalid MLIR Context capsule - expected mlir.ir.Context or nki.compiler._internal.ir.Context");
          }
        }

        // Register all dialects including Transform and extensions (nkipy transform ops)
        nkipyMlirRegisterAllDialects(context);

        // Register and load the nkipy dialect
        MlirDialectHandle nkipy = mlirGetDialectHandle__nkipy__();
        mlirDialectHandleRegisterDialect(nkipy, context);
        mlirDialectHandleLoadDialect(nkipy, context);
      },
      nb::arg("context") = nb::none(),
      "Register the nkipy dialect with the given context");

  // Apply transform to a design.
  nkipy_m.def("apply_passes", [](MlirModule &mlir_mod) {
    ModuleOp module = unwrap(mlir_mod);

    // Simplify the loop structure after the transform.
    PassManager pm(module.getContext());
    pm.addNestedPass<func::FuncOp>(
        mlir::affine::createSimplifyAffineStructuresPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(module)))
      throw nb::value_error("failed to apply the post-transform optimization");
  });

  // Utility pass APIs - COMMENTED OUT: Requires MLIRNkipyPasses
  // nkipy_m.def("memref_dce", &memRefDCE);

  // NOTE: Pass functions removed from Python bindings
  // This pass requires NISA dialect which has global constructors that cause segfaults in Python
  // Use nkipy-opt CLI tool instead via subprocess (see nkipy_kernelgen/transforms/nkipy_opt.py)

}
