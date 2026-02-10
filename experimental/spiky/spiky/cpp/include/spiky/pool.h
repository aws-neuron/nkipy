#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>

extern "C" {
#include <nrt/nrt.h>
}

namespace spiky {

struct PoolStats {
  uint64_t used_bytes{0};
  uint64_t cached_bytes{0};
  uint64_t total_bytes{0};
  uint64_t allocation_count{0};
  uint64_t reuse_count{0};
  uint64_t cache_hit_count{0};
  uint64_t cache_miss_count{0};
};

// Shared best-fit pool + pointer registry for both executor and torch backend.
class DeviceMemoryPool {
 public:
  static DeviceMemoryPool& Global();

  // Acquire a tensor of at least `size` bytes on a given device/core.
  nrt_tensor_t* Acquire(size_t size, int device, const char* name = nullptr);

  // Return to cache. Never throws; shutdown-safe (no-op when NRT is shutting down).
  void Release(nrt_tensor_t* tensor) noexcept;

  // Find tensor by host-visible VA pointer (as used by torch DataPtr::get()).
  // Supports pointers into a tensor (e.g. torch views with storage_offset).
  // Returns the base tensor handle and optionally the byte offset into the tensor.
  nrt_tensor_t* FindByDataPtr(void* data, size_t* byte_offset = nullptr) const;

  // For torch hooks: copy between two registered device pointers.
  bool CopyByDataPtr(void* dst, const void* src, size_t size) const noexcept;

  // Cache management.
  void ClearCache() noexcept;
  size_t CachedBlocks() const noexcept;
  PoolStats GetStats() const noexcept;

  // Zero tensor cache for device-side padding.
  nrt_tensor_t* GetZeroTensor(size_t min_size, int device);

 private:
  DeviceMemoryPool() = default;

  struct Block {
    nrt_tensor_t* tensor{nullptr};
    void* va{nullptr};
    size_t size{0};
    int device{-1};
  };

  // Best-fit cache: key=size, value=block.
  std::multimap<size_t, Block> free_blocks_;
  std::unordered_map<nrt_tensor_t*, Block> in_use_by_tensor_;
  // Base VA -> block (exact match). For range lookup, we also keep an ordered map.
  std::unordered_map<void*, Block> in_use_by_ptr_;
  std::map<uintptr_t, Block> in_use_by_addr_;

  mutable std::mutex mutex_;
  PoolStats stats_;

  std::unordered_map<int, nrt_tensor_t*> zero_tensors_;
  std::unordered_map<int, size_t> zero_tensor_sizes_;
};

}  // namespace spiky
