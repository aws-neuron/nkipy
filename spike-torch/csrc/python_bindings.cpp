// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include "include/spike_allocator.h"
#include "include/spike_device.h"
#include "include/spike_hooks.h"
#include "include/spike_torch.h"

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "spike-torch: PyTorch device integration for spike runtime";

  // Device registration
  m.def("_register_spike_device", &spike_torch::register_spike_device,
        "Register spike as a device type in PyTorch");

  m.def("_register_spike_hooks", &spike_torch::register_spike_hooks,
        "Register PrivateUse1HooksInterface for spike device");

  // Device functions
  m.def(
      "_device_count",
      []() -> int64_t {
        return static_cast<int64_t>(spike_torch::device::device_count());
      },
      "Get the number of spike devices");

  m.def(
      "_current_device",
      []() -> int64_t {
        return static_cast<int64_t>(spike_torch::device::current_device());
      },
      "Get the current spike device");

  m.def(
      "_set_device",
      [](int64_t device) { spike_torch::device::set_device(device); },
      "Set the current spike device", py::arg("device"));

  m.def(
      "_is_available",
      []() -> bool { return spike_torch::device::is_available(); },
      "Check if spike devices are available");

  // Memory management
  m.def(
      "_empty_cache", []() { spike_torch::allocator::empty_cache(); },
      "Empty the spike memory cache");

  m.def(
      "_get_cached_blocks",
      []() -> int64_t {
        return static_cast<int64_t>(spike_torch::allocator::get_cached_blocks());
      },
      "Get the number of cached memory blocks");

  m.def(
      "_clear_allocator_state",
      []() { spike_torch::allocator::clear_allocator_state(); },
      "Clear spike-torch allocator cache and registry");

  m.def(
      "_mark_allocator_ready",
      []() { spike_torch::allocator::mark_allocator_ready(); },
      "Mark allocator ready after NRT re-initialization");

  m.def(
      "_get_tensor_info",
      [](uintptr_t data_ptr) -> py::tuple {
        void *ptr = reinterpret_cast<void *>(data_ptr);
        auto info = spike_torch::allocator::get_tensor_info(ptr);
        if (!info.tensor) {
          throw std::runtime_error("No tensor found for data pointer");
        }
        return py::make_tuple(
            reinterpret_cast<uintptr_t>(info.tensor),  // nrt_tensor_t*
            static_cast<int64_t>(info.size),
            static_cast<int64_t>(info.device)  // core_id
        );
      },
      "Get (nrt_ptr, size, core_id) for a data pointer",
      py::arg("data_ptr"));
}
