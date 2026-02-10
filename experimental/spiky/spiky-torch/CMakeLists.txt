cmake_minimum_required(VERSION 3.20)
project(spiky_torch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Discover torch include/lib paths via Python to avoid depending on TorchConfig.cmake.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import os, torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PY_DIR
)
set(TORCH_INCLUDE_DIRS
  "${TORCH_PY_DIR}/include"
  "${TORCH_PY_DIR}/include/torch/csrc/api/include"
)
set(TORCH_LIB_DIR "${TORCH_PY_DIR}/lib")

# Find libnrt.so and nrt headers.
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
if(NOT (LIBNRT_LOCATION AND NRT_INCLUDE_DIR))
  message(FATAL_ERROR "NRT library/headers not found (expected under /opt/aws/neuron).")
endif()

pybind11_add_module(_C
  cpp/src/python_bindings.cc
  cpp/src/spiky_abi.cc
  cpp/src/spiky_bindings.cc
  cpp/src/spiky_device.cc
  cpp/src/spiky_allocator.cc
  cpp/src/spiky_hooks.cc
  cpp/src/spiky_storage_impl.cc
  cpp/src/spiky_tensor_impl.cc
  cpp/src/spiky_guard_impl.cc
)

target_include_directories(_C PRIVATE cpp/include)
target_include_directories(_C PRIVATE ${NRT_INCLUDE_DIR})
target_include_directories(_C PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_directories(_C PRIVATE ${TORCH_LIB_DIR})
target_link_libraries(_C PRIVATE torch_python torch torch_cpu c10 ${LIBNRT_LOCATION})
target_compile_definitions(_C PRIVATE -DSPKY_TORCH_BUILD)

if(UNIX AND NOT APPLE)
  target_link_libraries(_C PRIVATE dl)
endif()

get_filename_component(NRT_LIB_DIR ${LIBNRT_LOCATION} DIRECTORY)
set_target_properties(_C PROPERTIES
  # Resolve torch libs at runtime from the installed torch wheel location:
  # spiky/torch/_C.so -> ../../torch/lib
  BUILD_RPATH "\$ORIGIN/../../torch/lib;${NRT_LIB_DIR}"
  INSTALL_RPATH "\$ORIGIN/../../torch/lib;${NRT_LIB_DIR}"
)

install(TARGETS _C DESTINATION spiky/torch)
