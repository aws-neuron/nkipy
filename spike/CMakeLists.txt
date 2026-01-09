cmake_minimum_required(VERSION 3.15...3.27)
project(spike)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

find_package(nanobind CONFIG REQUIRED)

# Include directories
include_directories(src/include)

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

if(LIBNRT_LOCATION AND NRT_INCLUDE_DIR)
    message(STATUS "Found NRT library: ${LIBNRT_LOCATION}")
    message(STATUS "Found NRT headers: ${NRT_INCLUDE_DIR}")
    set(HAVE_NRT_LIBRARY TRUE)
    set(NRT_LIBRARY ${LIBNRT_LOCATION})
else()
    message(WARNING "NRT library or headers not found. Building without NRT support for development.")
    set(HAVE_NRT_LIBRARY FALSE)
endif()

# Source files
set(SOURCES
    src/nrt_wrapper.cpp
    src/tensor.cpp
    src/tensor_set.cpp
    src/model.cpp
    src/spike.cpp
    src/python_bindings.cpp
)

# Create the nanobind module
nanobind_add_module(_spike ${SOURCES})

# Link libraries and include directories conditionally
if(HAVE_NRT_LIBRARY)
    target_link_libraries(_spike PRIVATE ${NRT_LIBRARY})
    target_include_directories(_spike PRIVATE ${NRT_INCLUDE_DIR})
    target_compile_definitions(_spike PRIVATE HAVE_NRT_LIBRARY)
    
    # This allows the module to find libnrt.so at runtime without LD_LIBRARY_PATH
    get_filename_component(NRT_LIB_DIR ${NRT_LIBRARY} DIRECTORY)
    set_target_properties(_spike PROPERTIES
        BUILD_RPATH "${NRT_LIB_DIR}"
        INSTALL_RPATH "${NRT_LIB_DIR}"
    )
endif()

# Compiler-specific options
target_compile_definitions(_spike PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Generate stub file automatically (output directly to source package directory)
nanobind_add_stub(
  _spike_stub
  MODULE _spike
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/spike/_spike.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_spike>
  DEPENDS _spike
  MARKER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/spike/py.typed
)

# Format the generated stub file with ruff for consistency
add_custom_command(
  TARGET _spike_stub POST_BUILD
  COMMAND ${Python_EXECUTABLE} -m ruff format ${CMAKE_CURRENT_SOURCE_DIR}/src/spike/_spike.pyi
  COMMENT "Formatting _spike.pyi with ruff"
  VERBATIM
)

# Install the module
install(TARGETS _spike DESTINATION spike)
