// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "include/spike_allocator.h"

#include <c10/core/DeviceType.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "include/spike_device.h"

extern "C" {
#include <nrt/nrt.h>
}

namespace {
// Flag to indicate NRT is shutting down. When true, the deleter skips NRT calls
// to avoid errors when tensors are garbage collected after nrt_close().
std::atomic<bool> g_nrt_shutting_down{false};
} // namespace

namespace spike_torch {
namespace allocator {

namespace {

// Global registry to track allocations (DataPtr deleter cannot capture state)
struct AllocatorRegistry {
  struct AllocationInfo {
    nrt_tensor_t *tensor;
    size_t size;
    int device;
  };

  std::unordered_map<void *, AllocationInfo> allocations;
  std::mutex mutex;

  static AllocatorRegistry &getInstance() {
    static AllocatorRegistry instance;
    return instance;
  }

  void add(void *ptr, nrt_tensor_t *tensor, size_t size, int device) {
    std::lock_guard<std::mutex> lock(mutex);
    allocations[ptr] = {tensor, size, device};
  }

  AllocationInfo remove(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    if (it != allocations.end()) {
      AllocationInfo info = it->second;
      allocations.erase(it);
      return info;
    }
    return {nullptr, 0, -1};
  }

  nrt_tensor_t *find(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    return (it != allocations.end()) ? it->second.tensor : nullptr;
  }

  AllocationInfo find_info(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    if (it != allocations.end()) {
      return it->second;
    }
    return {nullptr, 0, -1};
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex);
    allocations.clear();
  }
};

// Simple memory pool that caches allocated blocks for reuse
class SimpleMemoryPool {
  struct Block {
    nrt_tensor_t *tensor;
    void *data_ptr;
    size_t size;
    std::chrono::steady_clock::time_point last_used;

    Block(nrt_tensor_t *t, void *ptr, size_t s)
        : tensor(t), data_ptr(ptr), size(s),
          last_used(std::chrono::steady_clock::now()) {}
  };

  std::vector<Block> blocks_;
  mutable std::mutex mutex_;

public:
  // Try to find a cached block of exact size
  std::pair<nrt_tensor_t *, void *> get_block(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find exact size match
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
                           [size](const Block &b) { return b.size == size; });

    if (it != blocks_.end()) {
      // Found a match - remove from cache and return
      nrt_tensor_t *tensor = it->tensor;
      void *data_ptr = it->data_ptr;
      blocks_.erase(it);
      return {tensor, data_ptr};
    }

    return {nullptr, nullptr};
  }

  // Add a freed block to the cache
  void add_block(nrt_tensor_t *tensor, void *data_ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_.emplace_back(tensor, data_ptr, size);
  }

  // Clear all cached blocks
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &block : blocks_) {
      nrt_tensor_free(&block.tensor);
    }
    blocks_.clear();
  }

  size_t cached_blocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_.size();
  }
};

// Per-device memory pools
std::vector<SimpleMemoryPool> *memory_pools = nullptr;
std::mutex pools_init_mutex;

// Get or initialize the memory pools
std::vector<SimpleMemoryPool> &getMemoryPools() {
  std::lock_guard<std::mutex> lock(pools_init_mutex);
  if (!memory_pools) {
    int count = device::device_count();
    if (count <= 0) {
      throw std::runtime_error(
          "No Neuron devices available. Cannot initialize memory pools.");
    }
    memory_pools = new std::vector<SimpleMemoryPool>(count);
  }
  return *memory_pools;
}

// Get pool for specific device
SimpleMemoryPool &getDevicePool(int device) {
  auto &pools = getMemoryPools();
  if (device < 0 || device >= static_cast<int>(pools.size())) {
    throw std::runtime_error("Invalid device index: " + std::to_string(device));
  }
  return pools[device];
}

// Static deleter function for DataPtr
void spike_deleter(void *ctx) noexcept {
  auto *t = static_cast<nrt_tensor_t *>(ctx);
  if (!t) {
    return;
  }

  // If NRT is shutting down, skip all NRT calls to avoid errors.
  // The runtime is already closed or closing, so we just abandon the tensor.
  if (g_nrt_shutting_down.load(std::memory_order_acquire)) {
    return;
  }

  void *ptr = nrt_tensor_get_va(t);
  if (!ptr) {
    nrt_tensor_free(&t);
    return;
  }

  AllocatorRegistry::AllocationInfo info{nullptr, 0, -1};
  try {
    info = AllocatorRegistry::getInstance().remove(ptr);
  } catch (...) {
    info = {nullptr, 0, -1};
  }

  if (info.tensor && info.size > 0 && info.device >= 0) {
    try {
      auto &device_pool = getDevicePool(info.device);
      device_pool.add_block(info.tensor, ptr, info.size);
      return;
    } catch (...) {
      // Fall through to free below.
    }
  }

  // Never throw from a deleter: free best-effort.
  nrt_tensor_free(&t);
}

class SpikeAllocator : public c10::Allocator {
public:
  SpikeAllocator() = default;

  c10::DataPtr allocate(size_t size) override {
    // Handle zero-size allocation
    if (size == 0) {
      return c10::DataPtr(nullptr,
                          c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    auto dev = device::current_device();

    // First check the device-specific cache
    auto &device_pool = getDevicePool(dev);
    auto [cached_tensor, cached_ptr] = device_pool.get_block(size);
    if (cached_tensor != nullptr) {
      AllocatorRegistry::getInstance().add(cached_ptr, cached_tensor, size,
                                           dev);
      return c10::DataPtr(cached_ptr, cached_tensor, &spike_deleter,
                          c10::Device(c10::DeviceType::PrivateUse1, dev));
    }

    // No cached block - allocate new
    nrt_tensor_t *tensor = nullptr;
    NRT_STATUS status = nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, dev,
                                            size, nullptr, &tensor);

    // Handle OOM by clearing cache
    if (status == NRT_RESOURCE && device_pool.cached_blocks() > 0) {
      device_pool.clear();
      status = nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, dev, size,
                                   nullptr, &tensor);
    }

    if (status != NRT_SUCCESS || !tensor) {
      std::string error_msg = "Failed to allocate Neuron tensor of size " +
                              std::to_string(size) + " bytes";
      if (status == NRT_RESOURCE) {
        error_msg += " (out of memory)";
      }
      error_msg += ". Status: " + std::to_string(status);
      throw std::runtime_error(error_msg);
    }

    void *data = nrt_tensor_get_va(tensor);
    if (!data) {
      nrt_tensor_free(&tensor);
      throw std::runtime_error("Failed to get tensor data pointer");
    }

    AllocatorRegistry::getInstance().add(data, tensor, size, dev);

    return c10::DataPtr(data, tensor, &spike_deleter,
                        c10::Device(c10::DeviceType::PrivateUse1, dev));
  }

  void copy_data(void *dest, const void *src, std::size_t size) const override {
    auto &registry = AllocatorRegistry::getInstance();

    nrt_tensor_t *src_tensor = registry.find(const_cast<void *>(src));
    nrt_tensor_t *dst_tensor = registry.find(dest);

    if (src_tensor && dst_tensor) {
      NRT_STATUS status = nrt_tensor_copy(src_tensor, 0, dst_tensor, 0, size);
      if (status != NRT_SUCCESS) {
        throw std::runtime_error("nrt_tensor_copy failed");
      }
    } else {
      throw std::runtime_error(
          "SpikeAllocator::copy_data called with unknown pointers. "
          "Both source and destination must be allocated by this allocator.");
    }
  }
};

// Global singleton instance
std::once_flag allocator_init_flag;
SpikeAllocator *allocator_instance = nullptr;

SpikeAllocator *getAllocator() {
  std::call_once(allocator_init_flag,
                 []() { allocator_instance = new SpikeAllocator(); });
  return allocator_instance;
}

} // anonymous namespace

c10::Allocator *get() { return getAllocator(); }

void empty_cache() {
  auto &pools = getMemoryPools();
  for (auto &pool : pools) {
    pool.clear();
  }
}

size_t get_cached_blocks() {
  size_t total = 0;
  auto &pools = getMemoryPools();
  for (const auto &pool : pools) {
    total += pool.cached_blocks();
  }
  return total;
}

void clear_allocator_state() {
  // Clear cached blocks (best-effort).
  {
    std::lock_guard<std::mutex> lock(pools_init_mutex);
    if (memory_pools) {
      for (auto &pool : *memory_pools) {
        pool.clear();
      }
      delete memory_pools;
      memory_pools = nullptr;
    }
  }

  // Clear registry regardless of whether pools were initialized.
  AllocatorRegistry::getInstance().clear();

  // Mark NRT as shutting down. Any tensors garbage collected after this point
  // will have their deleters skip NRT calls to avoid errors.
  g_nrt_shutting_down.store(true, std::memory_order_release);
}

void mark_allocator_ready() {
  // Reset the shutdown flag when NRT is re-initialized.
  // Called after get_spike_singleton() to indicate NRT is ready for operations.
  g_nrt_shutting_down.store(false, std::memory_order_release);
}

bool copy_tensor_data(void *dest, const void *src, size_t size) {
  auto &registry = AllocatorRegistry::getInstance();

  nrt_tensor_t *src_tensor = registry.find(const_cast<void *>(src));
  nrt_tensor_t *dst_tensor = registry.find(dest);

  if (src_tensor && dst_tensor) {
    NRT_STATUS status = nrt_tensor_copy(src_tensor, 0, dst_tensor, 0, size);
    return status == NRT_SUCCESS;
  }
  return false;
}

nrt_tensor_t *find_tensor(void *ptr) {
  return AllocatorRegistry::getInstance().find(ptr);
}

nrt_tensor_t *get_tensor_from_context(void *ctx) {
  return static_cast<nrt_tensor_t *>(ctx);
}

TensorInfo get_tensor_info(void *ptr) {
  auto info = AllocatorRegistry::getInstance().find_info(ptr);
  return {info.tensor, info.size, info.device};
}

} // namespace allocator
} // namespace spike_torch
