// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <iostream>
#include <sstream>

#include "model.h"
#include "spike.h"
#include "tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace spike;

NB_MODULE(_spike, m) {
  m.doc() = "NKIPy Spike Runtime C++ bindings";

  // Exception types
  nb::exception<SpikeRuntimeError> nkipy_runtime_error(m, "SpikeRuntimeError",
                                                       PyExc_RuntimeError);
  nb::exception<SpikeError>(m, "SpikeError", nkipy_runtime_error);
  nb::exception<NrtError>(m, "NrtError", nkipy_runtime_error);
  // TODO: figure out a way to expose error_code to Python...
  // below is an attempt but failed
  // nb::cpp_function_def<NrtError>(&NrtError::error_code, nb::scope(nrt_error),
  // nb::name("error_code"), nb::is_method());

  // BenchmarkResult struct
  nb::class_<BenchmarkResult>(m, "BenchmarkResult")
      .def_ro("mean_ms", &BenchmarkResult::mean_ms,
              "Mean execution time in milliseconds")
      .def_ro("min_ms", &BenchmarkResult::min_ms,
              "Minimum execution time in milliseconds")
      .def_ro("max_ms", &BenchmarkResult::max_ms,
              "Maximum execution time in milliseconds")
      .def_ro("std_dev_ms", &BenchmarkResult::std_dev_ms,
              "Standard deviation in milliseconds")
      .def_ro("iterations", &BenchmarkResult::iterations,
              "Number of benchmark iterations")
      .def_ro("warmup_iterations", &BenchmarkResult::warmup_iterations,
              "Number of warmup iterations")
      .def("__repr__", [](const BenchmarkResult &br) {
        return "BenchmarkResult(mean=" + std::to_string(br.mean_ms) +
               "ms, "
               "min=" +
               std::to_string(br.min_ms) +
               "ms, "
               "max=" +
               std::to_string(br.max_ms) +
               "ms, "
               "std_dev=" +
               std::to_string(br.std_dev_ms) +
               "ms, "
               "iterations=" +
               std::to_string(br.iterations) +
               ", "
               "warmup_iterations=" +
               std::to_string(br.warmup_iterations) + ")";
      });

  // TensorMetadata struct
  nb::class_<TensorMetadata>(m, "TensorMetadata")
      .def_ro("size", &TensorMetadata::size, "Tensor size in bytes")
      .def_ro("dtype", &TensorMetadata::dtype, "Data type as string")
      .def_ro("shape", &TensorMetadata::shape, "Tensor shape as list")
      .def("__repr__", [](const TensorMetadata &tm) {
        std::string shape_str = "[";
        for (size_t i = 0; i < tm.shape.size(); ++i) {
          if (i > 0)
            shape_str += ", ";
          shape_str += std::to_string(tm.shape[i]);
        }
        shape_str += "]";
        return "TensorMetadata(size=" + std::to_string(tm.size) + ", dtype='" +
               tm.dtype + "', shape=" + shape_str + ")";
      });

  // ModelTensorInfo struct
  nb::class_<ModelTensorInfo>(m, "ModelTensorInfo")
      .def_ro("inputs", &ModelTensorInfo::inputs, "Input tensor metadata")
      .def_ro("outputs", &ModelTensorInfo::outputs, "Output tensor metadata")
      .def("__repr__", [](const ModelTensorInfo &mti) {
        return "ModelTensorInfo(inputs=" + std::to_string(mti.inputs.size()) +
               " tensors, outputs=" + std::to_string(mti.outputs.size()) +
               " tensors)";
      });

  // NrtTensor class
  nb::class_<NrtTensor>(m, "NrtTensor")
      .def_prop_ro("core_id", &NrtTensor::get_core_id, "Logical NeuronCore ID")
      .def_prop_ro("size", &NrtTensor::get_size, "Tensor size in bytes")
      .def_prop_ro("name", &NrtTensor::get_name, "Tensor name");

  // NrtModel class
  nb::class_<NrtModel>(m, "NrtModel")
      .def_prop_ro("neff_path", &NrtModel::get_neff_path, "NEFF file path")
      .def_prop_ro("core_id", &NrtModel::get_core_id, "Core ID")
      .def_prop_ro("rank_id", &NrtModel::get_rank_id, "Rank ID")
      .def_prop_ro("world_size", &NrtModel::get_world_size, "World size")
      .def_prop_ro("is_collective", &NrtModel::get_is_collective,
                   "Is collective model")
      .def("__repr__", &NrtModel::to_string);

  // Spike class - uses keep_alive for safe shared ownership with tensors/models
  nb::class_<Spike>(m, "Spike")
      .def(nb::init<int>(), "verbose_level"_a = 0,
           "Initialize Spike with verbose level")

      // Static methods
      .def_static("get_visible_neuron_core_count",
                  &Spike::get_visible_neuron_core_count,
                  "Get the number of visible NeuronCores")

      // Core runtime methods
      .def("close", &Spike::close, "Close the NRT runtime")

      // Keep Spike alive for models that depend on it
      .def("load_model", &Spike::load_model, "neff_file"_a, "core_id"_a = 0,
           "cc_enabled"_a = false, "rank_id"_a = 0, "world_size"_a = 1,
           nb::keep_alive<0, 1>(), "Load a model from NEFF file")

      .def("unload_model", &Spike::unload_model, "model"_a, "Unload a model")

      // Execution methods
      .def("execute", &Spike::execute, "model"_a, "inputs"_a, "outputs"_a,
           nb::arg("ntff_name") = nb::none(), "save_trace"_a = false,
           nb::call_guard<nb::gil_scoped_release>(), // releasing the GIL to
                                                     // allow multiple cores
                                                     // to execute (enable CC)
           "Execute a model with given inputs and outputs")

      .def("benchmark", &Spike::benchmark, "model"_a, "inputs"_a, "outputs"_a,
           "warmup_iterations"_a = 1, "benchmark_iterations"_a = 1,
           "Benchmark a model execution")

      // Keep Spike alive for tensors that depend on it
      .def("allocate_tensor", &Spike::allocate_tensor, "size"_a,
           "core_id"_a = 0, nb::arg("name") = nb::none(),
           nb::keep_alive<0, 1>(), "Allocate a tensor on device")

      .def("slice_from_tensor", &Spike::slice_from_tensor, "source"_a,
           "offset"_a = 0, "size"_a = 0, nb::arg("name") = nb::none(),
           nb::keep_alive<0, 1>(), "Create a tensor slice from another tensor")

      .def("free_tensor", &Spike::free_tensor, "tensor"_a, "Free a tensor")

      // Tensor I/O methods
      .def(
          "tensor_write",
          [](Spike &self, NrtTensor &tensor, const nb::bytes &data,
             size_t offset) -> void {
            self.tensor_write(tensor, data.data(), data.size(), offset);
          },
          "tensor"_a, "data"_a, "offset"_a = 0, "Write bytes data to tensor")

      .def(
          "tensor_read",
          [](Spike &self, const NrtTensor &tensor, size_t offset,
             size_t size) -> nb::bytes {
            auto result = self.tensor_read(tensor, offset, size);
            return nb::bytes(result.data(), result.size());
          },
          "tensor"_a, "offset"_a = 0, "size"_a = 0,
          "Read data from tensor as bytes")

      .def(
          "tensor_write_from_pybuffer",
          [](Spike &self, NrtTensor &tensor, nb::object buffer,
             size_t offset) -> void {
            // Get buffer info using Python C API
            Py_buffer py_buffer;
            if (PyObject_GetBuffer(buffer.ptr(), &py_buffer, PyBUF_SIMPLE) !=
                0) {
              throw std::runtime_error(
                  "Object does not support buffer protocol");
            }

            try {
              self.tensor_write_from_pybuffer(tensor, py_buffer.buf,
                                              py_buffer.len, offset);
            } catch (...) {
              PyBuffer_Release(&py_buffer);
              throw;
            }

            PyBuffer_Release(&py_buffer);
          },
          "tensor"_a, "buffer"_a, "offset"_a = 0,
          "Write data from Python buffer protocol object (bytes, bytearray, "
          "memoryview, etc.) to tensor")

      .def(
          "tensor_read_to_pybuffer",
          [](Spike &self, const NrtTensor &tensor, nb::object buffer,
             size_t offset, size_t size) -> void {
            // Get buffer info using Python C API
            Py_buffer py_buffer;
            if (PyObject_GetBuffer(buffer.ptr(), &py_buffer, PyBUF_WRITABLE) !=
                0) {
              throw std::runtime_error(
                  "Object does not support writable buffer protocol");
            }

            try {
              // Read data from tensor
              auto data = self.tensor_read(tensor, offset, size);

              // Check if buffer is large enough
              if (py_buffer.len < static_cast<Py_ssize_t>(data.size())) {
                throw std::runtime_error("Buffer too small for tensor data");
              }

              // Copy data to buffer
              std::memcpy(py_buffer.buf, data.data(), data.size());
            } catch (...) {
              PyBuffer_Release(&py_buffer);
              throw;
            }

            PyBuffer_Release(&py_buffer);
          },
          "tensor"_a, "buffer"_a, "offset"_a = 0, "size"_a = 0,
          "Read data from tensor to Python buffer protocol object (bytearray, "
          "memoryview, etc.)")

      // Model introspection
      .def("get_tensor_info", &Spike::get_tensor_info, "model"_a,
           "Get tensor information for a model");
}
