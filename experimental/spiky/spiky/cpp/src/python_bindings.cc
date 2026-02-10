// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "spiky/device_tensor.h"
#include "spiky/executor.h"
#include "spiky/pool.h"
#include "spiky/runtime.h"
#include "spiky/torch_abi.h"

namespace py = pybind11;

namespace spiky {
namespace {

// ========= Torch ABI implementation =========

bool abi_is_initialized() { return Runtime::Global().IsInitialized(); }
bool abi_is_shutting_down() { return Runtime::Global().IsShuttingDown(); }

SpikyTorchAllocation abi_alloc(int device, size_t size) {
  nrt_tensor_t* t = DeviceMemoryPool::Global().Acquire(size, device, "spiky_torch");
  void* va = nrt_tensor_get_va(t);
  if (!va) {
    DeviceMemoryPool::Global().Release(t);
    throw std::runtime_error("spiky: failed to get VA for torch allocation");
  }
  return SpikyTorchAllocation{va, static_cast<void*>(t), device, size};
}

void abi_release(void* ctx) noexcept {
  auto* t = static_cast<nrt_tensor_t*>(ctx);
  DeviceMemoryPool::Global().Release(t);
}

nrt_tensor_t* abi_find(void* data) noexcept {
  return DeviceMemoryPool::Global().FindByDataPtr(data, nullptr);
}

bool abi_copy(void* dst, const void* src, size_t size) noexcept {
  return DeviceMemoryPool::Global().CopyByDataPtr(dst, src, size);
}

void abi_empty_cache() noexcept { DeviceMemoryPool::Global().ClearCache(); }
size_t abi_cached_blocks() noexcept { return DeviceMemoryPool::Global().CachedBlocks(); }

SpikyTorchABI g_abi{
    /*abi_version=*/1,
    &abi_is_initialized,
    &abi_is_shutting_down,
    &abi_alloc,
    &abi_release,
    &abi_find,
    &abi_copy,
    &abi_empty_cache,
    &abi_cached_blocks,
};

py::capsule MakeTorchCapsule() {
  return py::capsule(&g_abi, kSpikyTorchCapsuleName);
}

// ========= Helpers =========

DLDataType DTypeFromBuffer(const py::buffer_info& info) {
  // Best-effort mapping from PEP3118 format + itemsize.
  // We treat unknown as float32.
  const char* fmt = info.format.c_str();
  size_t itemsize = static_cast<size_t>(info.itemsize);
  // Common cases for numpy:
  // - float32: "f" itemsize 4
  // - float64: "d" itemsize 8
  // - int32: "i" itemsize 4
  // - int64: "q" itemsize 8
  // - uint8: "B" itemsize 1
  if (info.format == "f" && itemsize == 4) return DLDataType{kDLFloat, 32, 1};
  if (info.format == "e" && itemsize == 2) return DLDataType{kDLFloat, 16, 1};
  if (info.format == "d" && itemsize == 8) return DLDataType{kDLFloat, 64, 1};
  if (info.format == "i" && itemsize == 4) return DLDataType{kDLInt, 32, 1};
  if (info.format == "q" && itemsize == 8) return DLDataType{kDLInt, 64, 1};
  if (info.format == "B" && itemsize == 1) return DLDataType{kDLUInt, 8, 1};
  if (info.format == "H" && itemsize == 2) return DLDataType{kDLUInt, 16, 1};
  if (info.format == "I" && itemsize == 4) return DLDataType{kDLUInt, 32, 1};
  if (info.format == "Q" && itemsize == 8) return DLDataType{kDLUInt, 64, 1};
  (void)fmt;
  return DLDataType{kDLFloat, 32, 1};
}

struct DLTensorHolder {
  DLTensor dl{};
  std::vector<int64_t> shape;
};

std::vector<DLTensorHolder> MakeDLTensors(py::sequence seq) {
  std::vector<DLTensorHolder> holders;
  holders.reserve(seq.size());
  for (auto item : seq) {
    py::buffer b = py::reinterpret_borrow<py::buffer>(item);
    py::buffer_info info = b.request();

    DLTensorHolder h;
    h.shape.reserve(info.ndim);
    for (auto d : info.shape) h.shape.push_back(static_cast<int64_t>(d));

    h.dl.data = info.ptr;
    h.dl.device = DLDevice{kDLCPU, 0};
    h.dl.ndim = static_cast<int>(h.shape.size());
    h.dl.dtype = DTypeFromBuffer(info);
    h.dl.shape = h.shape.data();
    h.dl.strides = nullptr;
    h.dl.byte_offset = 0;
    holders.push_back(std::move(h));
  }
  return holders;
}

}  // namespace
}  // namespace spiky

PYBIND11_MODULE(_spiky, m) {
  using namespace spiky;

  m.doc() = "spiky core";

  m.def("init", [](int64_t device_id) { Runtime::Global().Init(device_id); Engine::Global().Init(device_id); },
        py::arg("device_id") = 0);
  m.def("close", []() { Engine::Global().Shutdown(); Runtime::Global().Close(); });
  m.def("is_initialized", []() { return Runtime::Global().IsInitialized(); });
  m.def("device_count", []() { return Runtime::Global().DeviceCount(); });

  m.def("_get_torch_abi", []() { return MakeTorchCapsule(); });

  m.def("empty_cache", []() { DeviceMemoryPool::Global().ClearCache(); });
  m.def("get_cached_blocks", []() { return DeviceMemoryPool::Global().CachedBlocks(); });

  m.def("register_bundle", [](std::map<int64_t, std::string> bucket_to_neff,
                             std::map<int64_t, int64_t> dynamic_specs,
                             bool cc_enabled, int64_t rank_id, int64_t world_size) {
    BundleSpec spec;
    spec.bucket_to_neff = std::move(bucket_to_neff);
    spec.dynamic_specs = std::move(dynamic_specs);
    spec.cc_enabled = cc_enabled;
    spec.rank_id = static_cast<uint32_t>(rank_id);
    spec.world_size = static_cast<uint32_t>(world_size);
    return Engine::Global().RegisterBundle(spec);
  },
  py::arg("bucket_to_neff"), py::arg("dynamic_specs"),
  py::arg("cc_enabled") = false, py::arg("rank_id") = 0, py::arg("world_size") = 1);
  m.def("unregister_bundle", [](int64_t bundle_id) { Engine::Global().UnregisterBundle(bundle_id); });
  m.def("select_bucket", [](int64_t bundle_id, int64_t actual_len) { return Engine::Global().SelectBucket(bundle_id, actual_len); });

  m.def("execute_bundle",
        [](int64_t bundle_id, int64_t bucket_size, py::sequence inputs,
           bool pad_on_device, bool keep_outputs_on_device, bool unpad_outputs, int64_t actual_len,
           bool save_trace, std::string ntff_name) {
          auto holders = spiky::MakeDLTensors(inputs);
          std::vector<DLTensor*> dls;
          dls.reserve(holders.size());
          for (auto& h : holders) dls.push_back(&h.dl);
          auto outs = Engine::Global().Execute(bundle_id, bucket_size, dls,
                                               pad_on_device, keep_outputs_on_device, unpad_outputs, actual_len,
                                               save_trace, ntff_name);

          if (keep_outputs_on_device) {
            py::list out_list;
            for (auto& t : outs) out_list.append(py::cast(std::move(t)));
            return out_list;
          }

          // Host outputs: convert to numpy arrays (one D2H read per output), then
          // return the device buffers to the pool as the C++ objects go out of scope.
          auto numpy_dtype_from_str = [](const std::string& s) -> py::object {
            if (s == "float32") return py::dtype::of<float>();
            if (s == "float16") return py::dtype("float16");
            if (s == "int8") return py::dtype("int8");
            if (s == "uint8") return py::dtype("uint8");
            if (s == "int16") return py::dtype("int16");
            if (s == "uint16") return py::dtype("uint16");
            if (s == "int32") return py::dtype("int32");
            if (s == "uint32") return py::dtype("uint32");
            if (s == "int64") return py::dtype("int64");
            if (s == "uint64") return py::dtype("uint64");
            if (s == "bfloat16") {
              try {
                py::module_ m = py::module_::import("ml_dtypes");
                return m.attr("bfloat16");
              } catch (...) {
                throw std::runtime_error("spiky: bfloat16 host output requires ml_dtypes");
              }
            }
            throw std::runtime_error("spiky: unsupported dtype for host output: " + s);
          };

          py::list host_out;
          for (auto& t : outs) {
            py::object dt = numpy_dtype_from_str(t.dtype());
            py::dtype npdt(dt);
            std::vector<py::ssize_t> shape;
            shape.reserve(t.shape().size());
            for (int64_t d : t.shape()) shape.push_back(static_cast<py::ssize_t>(d));
            py::array arr(npdt, shape);
            t.read_to(arr.mutable_data(), static_cast<size_t>(arr.nbytes()), 0);
            host_out.append(std::move(arr));
          }
          return host_out;
        },
        py::arg("bundle_id"), py::arg("bucket_size"), py::arg("inputs"),
        py::arg("pad_on_device") = true,
        py::arg("keep_outputs_on_device") = true,
        py::arg("unpad_outputs") = true,
        py::arg("actual_len") = 0,
        py::arg("save_trace") = false,
        py::arg("ntff_name") = std::string(""));

  m.def("execute_pipelined",
        [](int64_t bundle_id, int64_t bucket_size, py::sequence inputs, py::sequence next_inputs,
           bool pad_on_device, bool keep_outputs_on_device, bool unpad_outputs, int64_t actual_len,
           bool save_trace, std::string ntff_name) {
          auto holders = spiky::MakeDLTensors(inputs);
          auto nholders = spiky::MakeDLTensors(next_inputs);
          std::vector<DLTensor*> dls;
          std::vector<DLTensor*> ndls;
          dls.reserve(holders.size());
          ndls.reserve(nholders.size());
          for (auto& h : holders) dls.push_back(&h.dl);
          for (auto& h : nholders) ndls.push_back(&h.dl);
          auto outs = Engine::Global().ExecutePipelined(bundle_id, bucket_size, dls, ndls,
                                                        pad_on_device, keep_outputs_on_device, unpad_outputs, actual_len,
                                                        save_trace, ntff_name);
          if (keep_outputs_on_device) {
            py::list out_list;
            for (auto& t : outs) out_list.append(py::cast(std::move(t)));
            return out_list;
          }

          auto numpy_dtype_from_str = [](const std::string& s) -> py::object {
            if (s == "float32") return py::dtype::of<float>();
            if (s == "float16") return py::dtype("float16");
            if (s == "int8") return py::dtype("int8");
            if (s == "uint8") return py::dtype("uint8");
            if (s == "int16") return py::dtype("int16");
            if (s == "uint16") return py::dtype("uint16");
            if (s == "int32") return py::dtype("int32");
            if (s == "uint32") return py::dtype("uint32");
            if (s == "int64") return py::dtype("int64");
            if (s == "uint64") return py::dtype("uint64");
            if (s == "bfloat16") {
              try {
                py::module_ m = py::module_::import("ml_dtypes");
                return m.attr("bfloat16");
              } catch (...) {
                throw std::runtime_error("spiky: bfloat16 host output requires ml_dtypes");
              }
            }
            throw std::runtime_error("spiky: unsupported dtype for host output: " + s);
          };

          py::list host_out;
          for (auto& t : outs) {
            py::object dt = numpy_dtype_from_str(t.dtype());
            py::dtype npdt(dt);
            std::vector<py::ssize_t> shape;
            shape.reserve(t.shape().size());
            for (int64_t d : t.shape()) shape.push_back(static_cast<py::ssize_t>(d));
            py::array arr(npdt, shape);
            t.read_to(arr.mutable_data(), static_cast<size_t>(arr.nbytes()), 0);
            host_out.append(std::move(arr));
          }
          return host_out;
        },
        py::arg("bundle_id"), py::arg("bucket_size"), py::arg("inputs"), py::arg("next_inputs"),
        py::arg("pad_on_device") = true,
        py::arg("keep_outputs_on_device") = true,
        py::arg("unpad_outputs") = true,
        py::arg("actual_len") = 0,
        py::arg("save_trace") = false,
        py::arg("ntff_name") = std::string(""));

  m.def("flush_pipeline", [](int64_t bundle_id) { Engine::Global().FlushPipeline(bundle_id); });

  m.def("get_stats", []() { return Engine::Global().GetStats(); });
  m.def("reset_stats", []() { Engine::Global().ResetStats(); });

  py::class_<RuntimeStats>(m, "RuntimeStats")
      .def_readonly("total_executions", &RuntimeStats::total_executions)
      .def_readonly("total_compile_count", &RuntimeStats::total_compile_count)
      .def_readonly("bucket_hits", &RuntimeStats::bucket_hits)
      .def_readonly("total_pad_ratio_sum", &RuntimeStats::total_pad_ratio_sum)
      .def_readonly("total_execution_time_ms", &RuntimeStats::total_execution_time_ms)
      .def_readonly("min_execution_time_ms", &RuntimeStats::min_execution_time_ms)
      .def_readonly("max_execution_time_ms", &RuntimeStats::max_execution_time_ms)
      .def_readonly("total_h2d_time_ms", &RuntimeStats::total_h2d_time_ms)
      .def_readonly("total_nrt_exec_time_ms", &RuntimeStats::total_nrt_exec_time_ms)
      .def_readonly("total_d2h_time_ms", &RuntimeStats::total_d2h_time_ms);

  py::class_<DeviceTensor>(m, "DeviceTensor")
      .def_property_readonly("nrt_ptr", &DeviceTensor::nrt_ptr)
      .def_property_readonly("device", &DeviceTensor::device)
      .def_property_readonly("size_bytes", &DeviceTensor::size_bytes)
      .def_property_readonly("shape", &DeviceTensor::shape)
      .def_property_readonly("dtype", &DeviceTensor::dtype)
      .def("read_to_bytes", &DeviceTensor::read_to_bytes);

  // Placeholder Bundle class for API completeness (currently bundle ids are returned).
  py::class_<Bundle>(m, "Bundle");
}
