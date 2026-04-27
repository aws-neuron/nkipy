set(LLVM_TARGET_DEFINITIONS NkipyTransformOps.td)
mlir_tablegen(NkipyTransformOps.h.inc -gen-op-decls)
mlir_tablegen(NkipyTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRNkipyTransformOpsIncGen)
add_dependencies(mlir-headers MLIRNkipyTransformOpsIncGen)

# Fix namespace issue in generated files (older MLIR uses mlir:: instead of ::mlir::)
add_custom_command(
    TARGET MLIRNkipyTransformOpsIncGen POST_BUILD
    COMMAND sed -i 's/std::optional<mlir::Attribute>/std::optional<::mlir::Attribute>/g'
        ${CMAKE_CURRENT_BINARY_DIR}/NkipyTransformOps.h.inc
    COMMAND sed -i 's/, mlir::Attribute/, ::mlir::Attribute/g'
        ${CMAKE_CURRENT_BINARY_DIR}/NkipyTransformOps.h.inc
    COMMAND sed -i 's/std::optional<mlir::Attribute>/std::optional<::mlir::Attribute>/g'
        ${CMAKE_CURRENT_BINARY_DIR}/NkipyTransformOps.cpp.inc
    COMMAND sed -i 's/, mlir::Attribute/, ::mlir::Attribute/g'
        ${CMAKE_CURRENT_BINARY_DIR}/NkipyTransformOps.cpp.inc
    COMMENT "Fixing namespace qualifications in generated transform ops"
)
