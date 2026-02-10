// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spiky_torch/allocator.h"

#include "spiky_torch/abi.h"
#include "spiky_torch/device.h"

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>

#include <mutex>
#include <stdexcept>

namespace spiky_torch {
namespace allocator {
namespace {

void spiky_deleter(void* ctx) noexcept {
  if (!ctx) return;
  const SpikyTorchABI* abi = nullptr;
  try {
    abi = GetABI();
  } catch (...) {
    return;
  }
  if (abi->is_shutting_down && abi->is_shutting_down()) return;
  abi->release(ctx);
}

class SpikyAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t size) override {
    const SpikyTorchABI* abi = GetABI();
    if (!abi->is_initialized()) {
      throw std::runtime_error("spiky_torch: runtime not initialized (call spiky.init() first)");
    }
    if (size == 0) {
      return c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0));
    }
    int dev = device::current_device();
    SpikyTorchAllocation a = abi->alloc(dev, size);
    return c10::DataPtr(a.data, a.ctx, &spiky_deleter,
                        c10::Device(c10::DeviceType::PrivateUse1, dev));
  }

  void copy_data(void* dest, const void* src, std::size_t size) const override {
    const SpikyTorchABI* abi = GetABI();
    if (!abi->is_initialized()) {
      throw std::runtime_error("spiky_torch: runtime not initialized");
    }
    if (!abi->copy_by_ptr(dest, src, size)) {
      throw std::runtime_error("spiky_torch: copy_data failed (unknown pointers or cross-device)");
    }
  }
};

std::once_flag g_init;
SpikyAllocator* g_alloc = nullptr;

SpikyAllocator* getAllocator() {
  std::call_once(g_init, []() { g_alloc = new SpikyAllocator(); });
  return g_alloc;
}

}  // namespace

c10::Allocator* get() { return getAllocator(); }

void empty_cache() {
  const SpikyTorchABI* abi = GetABI();
  abi->empty_cache();
}

size_t get_cached_blocks() {
  const SpikyTorchABI* abi = GetABI();
  return abi->cached_blocks();
}

bool copy_tensor_data(void* dest, const void* src, size_t size) noexcept {
  try {
    const SpikyTorchABI* abi = GetABI();
    return abi->copy_by_ptr(dest, src, size);
  } catch (...) {
    return false;
  }
}

}  // namespace allocator
}  // namespace spiky_torch
