add_mlir_library(MLIRNkipyPasses
    Passes.cpp
    AnnotateMemorySpace.cpp
    CanonicalizeReshape.cpp
    CanonicalizePartitionDim.cpp
    AssignLinalgOpIds.cpp
    ApplyAndStripTransforms.cpp
    InferLayout.cpp
    KnobDrivenTiling.cpp
    KnobDrivenFusion.cpp
    EliminateUninitializedCopies.cpp
    EliminateSameMemSpaceCopy.cpp
    InsertSpillReload.cpp
    InsertMemRefDealloc.cpp
    CanonicalizeLoopStep.cpp
    LegalizeLayout.cpp
    SimplifyLinalg.cpp
    CanonicalizeLinalgForNisa/Arithmetic.cpp
    CanonicalizeLinalgForNisa/MatmulPrep.cpp
    InlineNkipyReference.cpp
    OpClassification.cpp
    IRHelpers.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/nkipy

    DEPENDS
    MLIRNkipyPassesIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRMemRefDialect
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRDestinationStyleOpInterface
    MLIRTilingInterface
    MLIRTransformDialect
    MLIRTransformDialectTransforms
    MLIRSCFDialect
    MLIRSCFUtils
    MLIRNkipyTransformOps
)
