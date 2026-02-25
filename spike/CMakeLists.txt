cmake_minimum_required(VERSION 3.15...3.27)
project(spike)

# Module name is always _spike (matches NB_MODULE in source)
set(RUNTIME_MODULE_NAME "_spike")

# Find nanobind (required for nanobind_add_module macro)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)

# Find libnrt.so and nrt.h
set(NEURON_ROOT "/opt/aws/neuron" CACHE PATH "Path to Neuron installation")
find_library(LIBNRT_LOCATION
    NAMES libnrt.so
    HINTS "${NEURON_ROOT}/lib/" $ENV{NEURON_ROOT}/lib/
    PATH_SUFFIXES nrt
)
find_path(NRT_INCLUDE_DIR
    NAMES nrt/nrt.h
    HINTS "${NEURON_ROOT}/include/" $ENV{NEURON_ROOT}/include/
    PATH_SUFFIXES nrt
)

if(NOT LIBNRT_LOCATION OR NOT NRT_INCLUDE_DIR)
    message(STATUS "Skipping spike runtime build (NRT library or headers not found)")
    return()
endif()

message(STATUS "Found NRT library: ${LIBNRT_LOCATION}")
message(STATUS "Found NRT headers: ${NRT_INCLUDE_DIR}")
set(NRT_LIBRARY ${LIBNRT_LOCATION})

# Source files
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nrt_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_set.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/spike.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sys_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/debug_handler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/python_bindings.cpp
)

# Create the nanobind module
nanobind_add_module(${RUNTIME_MODULE_NAME} ${SOURCES})

# Link NRT library and dependencies
target_link_libraries(${RUNTIME_MODULE_NAME} PRIVATE ${NRT_LIBRARY})
target_include_directories(${RUNTIME_MODULE_NAME} PRIVATE ${NRT_INCLUDE_DIR})

# Set rpath so module finds libnrt.so at runtime without LD_LIBRARY_PATH
get_filename_component(NRT_LIB_DIR ${NRT_LIBRARY} DIRECTORY)
set_target_properties(${RUNTIME_MODULE_NAME} PROPERTIES
    BUILD_RPATH "${NRT_LIB_DIR}"
    INSTALL_RPATH "${NRT_LIB_DIR}"
)

# Compiler-specific options
target_compile_definitions(${RUNTIME_MODULE_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Mark nanobind includes as SYSTEM to suppress warnings from third-party headers
get_target_property(NB_INCLUDE_DIRS nanobind-static INTERFACE_INCLUDE_DIRECTORIES)
if(NB_INCLUDE_DIRS)
    target_include_directories(${RUNTIME_MODULE_NAME} SYSTEM PRIVATE ${NB_INCLUDE_DIRS})
endif()

# Generate stub file automatically (output directly to source package directory)
nanobind_add_stub(
  ${RUNTIME_MODULE_NAME}_stub
  MODULE ${RUNTIME_MODULE_NAME}
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/spike/${RUNTIME_MODULE_NAME}.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:${RUNTIME_MODULE_NAME}>
  DEPENDS ${RUNTIME_MODULE_NAME}
  MARKER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/spike/py.typed
)
# Install the native module
install(TARGETS ${RUNTIME_MODULE_NAME} DESTINATION spike)
