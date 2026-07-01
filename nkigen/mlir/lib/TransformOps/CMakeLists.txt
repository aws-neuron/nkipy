add_mlir_library(MLIRNkipyTransformOps
    NkipyTransformOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/nkipy/TransformOps

    DEPENDS
    MLIRNkipyTransformOpsIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRTransformDialect
    MLIRTransformDialectInterfaces
    MLIRIR
)
