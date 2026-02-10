// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include "spiky_torch/abi.h"
#include "spiky_torch/allocator.h"
#include "spiky_torch/device.h"
#include "spiky_torch/hooks.h"

namespace py = pybind11;

namespace spiky_torch {
void register_backend_name();
}  // namespace spiky_torch

PYBIND11_MODULE(_C, m) {
  m.doc() = "spiky.torch: PyTorch device integration for spiky (device name nkipy)";

  // Ensure ABI is available once module is imported.
  m.def("_load_abi", []() { spiky_torch::LoadABIFromPython(); });

  m.def("_register_backend_name", []() {
    spiky_torch::LoadABIFromPython();
    spiky_torch::register_backend_name();
  });

  m.def("_register_hooks", []() {
    spiky_torch::LoadABIFromPython();
    spiky_torch::register_hooks();
  });

  m.def("_register_allocator", []() {
    spiky_torch::LoadABIFromPython();
    // Allocator is registered via StorageImpl using c10; nothing else required here.
    (void)spiky_torch::allocator::get();
  });

  m.def("_device_count", []() -> int64_t { return static_cast<int64_t>(spiky_torch::device::device_count()); });
  m.def("_current_device", []() -> int64_t { return static_cast<int64_t>(spiky_torch::device::current_device()); });
  m.def("_set_device", [](int64_t d) { spiky_torch::device::set_device(static_cast<int>(d)); });
  m.def("_is_available", []() -> bool { return spiky_torch::device::is_available(); });

  m.def("_empty_cache", []() { spiky_torch::allocator::empty_cache(); });
  m.def("_get_cached_blocks", []() -> int64_t { return static_cast<int64_t>(spiky_torch::allocator::get_cached_blocks()); });
}

