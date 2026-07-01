include(AddMLIRPython)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(NKIPY_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nkipy")
set(NKIPY_MLIR_PYTHON_PACKAGES_DIR "${PROJECT_BINARY_DIR}/tools/nkipy")
set(MLIR_PYTHON_SOURCE_DIR "${MLIR_MAIN_SRC_DIR}/lib/Bindings")
set(NKIPY_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings")

include_directories(${MLIR_PYTHON_SOURCE_DIR})

# Use the system MLIR package prefix to ensure capsule compatibility
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(NkipyMLIRPythonSources)
declare_mlir_python_sources(NkipyMLIRPythonExtensions)

declare_mlir_python_sources(NkipyMLIRPythonSources.Dialects
  ROOT_DIR "${NKIPY_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT NkipyMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT NkipyMLIRPythonSources.Dialects
  ROOT_DIR "${NKIPY_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/NkipyBinding.td
  SOURCES
    dialects/nkipy.py
    dialects/_ods_common.py
    exceptions.py
    __init__.py
  DIALECT_NAME nkipy
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(NkipyMLIRPythonExtensions.Main
  MODULE_NAME _nkipy
  ADD_TO_PARENT NkipyMLIRPythonExtensions
  ROOT_DIR "/"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    ${NKIPY_PYTHON_SOURCE_DIR}/NkipyModule.cpp
    ${NKIPY_PYTHON_SOURCE_DIR}/NkipyAttributes.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIDebug
    MLIRNkipyCAPI
  PRIVATE_LINK_LIBS
    MLIRPass
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRAffineDialect
    LLVMSupport
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

# Only build NKIPy custom dialect extensions, not full MLIR bindings
# This avoids nanobind type conflicts with system mlir/nki.compiler._internal
set(_source_components
  NkipyMLIRPythonSources
  NkipyMLIRPythonExtensions
  # MLIRPythonSources - REMOVED: Use system mlir package instead
  # MLIRPythonExtension.RegisterEverything - REMOVED: Use system mlir package instead
)

add_mlir_python_common_capi_library(NkipyMLIRAggregateCAPI
  INSTALL_COMPONENT NkipyMLIRPythonModules
  INSTALL_DESTINATION _mlir
  OUTPUT_DIRECTORY "${NKIPY_MLIR_PYTHON_PACKAGES_DIR}/_mlir"
  RELATIVE_INSTALL_ROOT "../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    ${_source_components}
)

add_mlir_python_modules(NkipyMLIRPythonModules
  ROOT_PREFIX "${NKIPY_MLIR_PYTHON_PACKAGES_DIR}/_mlir"
  INSTALL_PREFIX "_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    NkipyMLIRAggregateCAPI
  )
