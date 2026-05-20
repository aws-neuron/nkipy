get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIROptLib
        MLIRNkipy
        MLIRNkipyPasses
        MLIRNkipyTransformOps
        )

add_llvm_executable(nkipy-opt nkipy-opt.cpp)

llvm_update_compile_flags(nkipy-opt)
target_link_libraries(nkipy-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(nkipy-opt)
