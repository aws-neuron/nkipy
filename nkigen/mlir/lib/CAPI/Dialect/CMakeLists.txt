
add_mlir_public_c_api_library(MLIRNkipyCAPI
  Dialects.cpp
  NkipyAttributes.cpp
  Registration.cpp
  # ${PROJECT_SOURCE_DIR}/lib/Transforms/Passes.cpp  # COMMENTED: Pass registration not needed, use nkipy-opt CLI

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR
  MLIRSupport
  MLIRNkipy
  MLIRNkipyTransformOps
  # MLIRNkipyPasses  # COMMENTED: Contains NISA dependencies, use nkipy-opt CLI instead
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRAffineDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPDLDialect
  MLIRTransformDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTransforms
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRMemRefTransforms
  MLIRFuncTransforms
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  )
