#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <iostream>
#include <sstream>

#include "model.h"
#include "spike.h"
#include "sys_trace.h"
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

  // SysTraceGuard — RAII wrapper for NRT system trace capture.
  // Exposed as a context manager for Python use.
  nb::class_<SysTraceGuard>(m, "SystemTraceSession")
      .def(nb::init<std::optional<uint32_t>>(), nb::arg("core_id") = nb::none(),
           "Create a trace session. Trace the given core_id or if core_id is "
           "omitted, traces all visible NeuronCores. ")
      .def("stop", &SysTraceGuard::stop,
           "Stop tracing. Called automatically on __exit__.")
      .def("fetch_events_json", &SysTraceGuard::fetch_events_json,
           "Fetch events as JSON string, consumes events from the system trace "
           "ring buffer")
      .def("drain_events", &SysTraceGuard::drain_events,
           "Discard events in the buffer")
      .def(
          "__enter__",
          [](SysTraceGuard &self) -> SysTraceGuard & { return self; },
          nb::rv_policy::reference)
      .def("__exit__", [](SysTraceGuard &self, nb::args) { self.stop(); });

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
      .def_prop_ro("name", &NrtTensor::get_name, "Tensor name")
      .def_prop_ro("va", &NrtTensor::get_va,
                   "CPU-accessible virtual address of device HBM memory");

  // NrtModel class
  nb::class_<NrtModel>(m, "NrtModel")
      .def_prop_ro("neff_path", &NrtModel::get_neff_path, "NEFF file path")
      .def_prop_ro("core_id", &NrtModel::get_core_id, "Core ID")
      .def_prop_ro("rank_id", &NrtModel::get_rank_id, "Rank ID")
      .def_prop_ro("world_size", &NrtModel::get_world_size, "World size")
      .def_prop_ro("is_collective", &NrtModel::get_is_collective,
                   "Is collective model")
      .def("__repr__", &NrtModel::to_string);

  // NrtTensorSet class
  nb::class_<NrtTensorSet>(m, "NrtTensorSet");

  // NonBlock result structures
  nb::class_<NonBlockTensorReadResult>(m, "NonBlockTensorReadResult")
      .def_ro("id", &NonBlockTensorReadResult::id)
      .def_ro("data", &NonBlockTensorReadResult::data)
      .def_ro("err", &NonBlockTensorReadResult::err);

  nb::class_<NonBlockTensorWriteResult>(m, "NonBlockTensorWriteResult")
      .def_ro("id", &NonBlockTensorWriteResult::id)
      .def_ro("err", &NonBlockTensorWriteResult::err);

  nb::class_<NonBlockExecResult>(m, "NonBlockExecResult")
      .def_ro("id", &NonBlockExecResult::id)
      .def_ro("err", &NonBlockExecResult::err);

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

      // Nonblocking operations
      .def("init_nonblock", &Spike::init_nonblock,
           "Initialize for nonblocking operations")

      .def(
          "tensor_read_nonblock",
          [](Spike &self, std::shared_ptr<const NrtTensor> tensor,
             size_t offset, size_t size) {
            return self.tensor_read_nonblock(std::move(tensor), offset, size);
          },
          "tensor"_a, "offset"_a = 0, "size"_a = 0,
          "Read data from tensor as bytes nonblockingly")

      .def(
          "tensor_read_nonblock",
          [](Spike &self, std::shared_ptr<const NrtTensor> tensor,
             nb::ndarray<> dest, size_t offset, size_t size) {
            return self.tensor_read_nonblock(std::move(tensor), std::move(dest),
                                             offset, size);
          },
          "tensor"_a, "dest"_a, "offset"_a = 0, "size"_a = 0,
          "Read data from tensor into the provided destination nonblockingly")

      .def(
          "tensor_write_nonblock",
          [](Spike &self, std::shared_ptr<NrtTensor> tensor, nb::bytes data_obj,
             size_t offset) {
            return self.tensor_write_nonblock(std::move(tensor),
                                              std::move(data_obj), offset);
          },
          "tensor"_a, "data"_a, "offset"_a = 0,
          "Write bytes data to tensor nonblockingly")

      .def(
          "tensor_write_nonblock",
          [](Spike &self, std::shared_ptr<NrtTensor> tensor,
             nb::ndarray<> data_obj, size_t offset) {
            return self.tensor_write_nonblock(std::move(tensor),
                                              std::move(data_obj), offset);
          },
          "tensor"_a, "data"_a, "offset"_a = 0,
          "Write ndarray data to tensor nonblockingly")

      .def(
          "tensor_write_nonblock",
          [](Spike &self, std::shared_ptr<NrtTensor> tensor, int64_t data,
             size_t size, size_t offset) {
            return self.tensor_write_nonblock(
                std::move(tensor), reinterpret_cast<const void *>(data), size,
                offset);
          },
          "tensor"_a, "data"_a, "size"_a, "offset"_a = 0,
          "Write raw pointer data to tensor nonblockingly")

      .def("tensor_write_nonblock_batched_prepare",
           &Spike::tensor_write_nonblock_batched_prepare, "tensors"_a,
           "data_objs"_a, "offsets"_a = std::nullopt,
           "Prepare a batched tensor write")

      .def("tensor_write_nonblock_batched_start",
           &Spike::tensor_write_nonblock_batched_start, "batch_id"_a,
           "Start a prepared batched tensor write")

      .def("tensor_read_nonblock_batched_prepare",
           &Spike::tensor_read_nonblock_batched_prepare, "tensors"_a, "dests"_a,
           "offsets"_a = std::nullopt, "sizes"_a = std::nullopt,
           "Prepare a batched tensor read")

      .def("tensor_read_nonblock_batched_start",
           &Spike::tensor_read_nonblock_batched_start, "batch_id"_a,
           "Start a prepared batched tensor read")

      .def("execute_nonblock", &Spike::execute_nonblock, "model"_a, "inputs"_a,
           "outputs"_a, nb::arg("ntff_name") = nb::none(),
           "save_trace"_a = false,
           "Execute a model with given inputs and outputs nonblockingly")

      .def("try_poll", &Spike::try_poll, "Try to poll for nonblocking results")

      .def(
          "create_tensor_set",
          [](Spike &self,
             const std::unordered_map<
                 std::string, std::shared_ptr<const NrtTensor>> &tensor_map) {
            return self.create_tensor_set(tensor_map);
          },
          "tensors"_a, "Create a tensor set with the tensors")

      // Wrap existing NRT objects (for interop with external code)
      .def(
          "wrap_model",
          [](Spike &self, int64_t ptr) {
            return self.wrap_model(reinterpret_cast<nrt_model_t *>(ptr));
          },
          "ptr"_a, "Wrap an existing NRT model pointer")

      .def(
          "wrap_tensor",
          [](Spike &self, int64_t ptr) {
            return self.wrap_tensor(reinterpret_cast<nrt_tensor_t *>(ptr));
          },
          "ptr"_a, "Wrap an existing NRT tensor pointer")

      .def(
          "wrap_tensor_set",
          [](Spike &self, int64_t ptr) {
            return self.wrap_tensor_set(
                reinterpret_cast<nrt_tensor_set_t *>(ptr));
          },
          "ptr"_a, "Wrap an existing NRT tensor set pointer")

      // Model introspection
      .def("get_tensor_info", &Spike::get_tensor_info, "model"_a,
           "Get tensor information for a model");
}
