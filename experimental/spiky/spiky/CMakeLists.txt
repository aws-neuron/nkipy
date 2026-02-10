cmake_minimum_required(VERSION 3.20)
project(spiky LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find libnrt.so and nrt headers (Trainium environment).
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

if(LIBNRT_LOCATION AND NRT_INCLUDE_DIR)
  message(STATUS "Found NRT library: ${LIBNRT_LOCATION}")
  message(STATUS "Found NRT headers: ${NRT_INCLUDE_DIR}")
  set(HAVE_NRT_LIBRARY TRUE)
else()
  message(FATAL_ERROR "NRT library/headers not found (expected under /opt/aws/neuron).")
endif()

add_library(spiky_core STATIC
  cpp/src/runtime.cc
  cpp/src/pool.cc
  cpp/src/model.cc
  cpp/src/device_tensor.cc
  cpp/src/executor.cc
)
target_include_directories(spiky_core PUBLIC cpp/include)
target_compile_definitions(spiky_core PRIVATE -DSPKY_CORE_BUILD)

target_include_directories(spiky_core PUBLIC ${NRT_INCLUDE_DIR})
target_link_libraries(spiky_core PRIVATE ${LIBNRT_LOCATION})

pybind11_add_module(_spiky cpp/src/python_bindings.cc)
target_link_libraries(_spiky PRIVATE spiky_core)
target_include_directories(_spiky PRIVATE cpp/include)

if(UNIX AND NOT APPLE)
  target_link_libraries(_spiky PRIVATE dl)
endif()

if(HAVE_NRT_LIBRARY)
  # Make the extension find libnrt.so without requiring LD_LIBRARY_PATH.
  get_filename_component(NRT_LIB_DIR ${LIBNRT_LOCATION} DIRECTORY)
  set_target_properties(_spiky PROPERTIES
    BUILD_RPATH "${NRT_LIB_DIR}"
    INSTALL_RPATH "${NRT_LIB_DIR}"
  )
endif()

install(TARGETS _spiky DESTINATION spiky)
