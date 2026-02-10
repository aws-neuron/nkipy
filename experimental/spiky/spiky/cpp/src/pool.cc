#include "spiky/pool.h"

#include "spiky/runtime.h"

#include <vector>

extern "C" {
#include <nrt/nrt_status.h>
}

namespace spiky {

DeviceMemoryPool& DeviceMemoryPool::Global() {
  static DeviceMemoryPool pool;
  return pool;
}

static inline void* GetVA(nrt_tensor_t* t) {
  return t ? nrt_tensor_get_va(t) : nullptr;
}

nrt_tensor_t* DeviceMemoryPool::Acquire(size_t size, int device, const char* name) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!Runtime::Global().IsInitialized()) {
    throw std::runtime_error("spiky: NRT is not initialized (call spiky.init() first)");
  }

  // Best-fit: smallest block >= size, matching requested device.
  auto it = free_blocks_.lower_bound(size);
  while (it != free_blocks_.end() && it->second.device != device) {
    ++it;
  }
  if (it != free_blocks_.end()) {
    Block block = it->second;
    free_blocks_.erase(it);
    in_use_by_tensor_[block.tensor] = block;
    in_use_by_ptr_[block.va] = block;
    in_use_by_addr_[reinterpret_cast<uintptr_t>(block.va)] = block;

    stats_.used_bytes += block.size;
    stats_.cached_bytes -= block.size;
    stats_.reuse_count++;
    if (block.size == size) stats_.cache_hit_count++;
    else stats_.cache_miss_count++;
    return block.tensor;
  }

  nrt_tensor_t* tensor = nullptr;
  NRT_STATUS status = nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, device, size, name, &tensor);
  if (status == NRT_RESOURCE && !free_blocks_.empty()) {
    // Emergency clear cache and retry once.
    for (auto& kv : free_blocks_) {
      if (kv.second.tensor) nrt_tensor_free(&kv.second.tensor);
    }
    free_blocks_.clear();
    stats_.total_bytes -= stats_.cached_bytes;
    stats_.cached_bytes = 0;
    status = nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, device, size, name, &tensor);
  }
  if (status != NRT_SUCCESS || !tensor) {
    throw std::runtime_error("spiky: nrt_tensor_allocate failed, status=" + std::to_string(status));
  }

  void* va = GetVA(tensor);
  if (!va) {
    nrt_tensor_free(&tensor);
    throw std::runtime_error("spiky: nrt_tensor_get_va failed");
  }

  Block block{tensor, va, size, device};
  in_use_by_tensor_[tensor] = block;
  in_use_by_ptr_[va] = block;
  in_use_by_addr_[reinterpret_cast<uintptr_t>(va)] = block;

  stats_.used_bytes += size;
  stats_.total_bytes += size;
  stats_.allocation_count++;
  return tensor;
}

void DeviceMemoryPool::Release(nrt_tensor_t* tensor) noexcept {
  if (!tensor) return;
  if (Runtime::Global().IsShuttingDown()) {
    return;  // Shutdown-safe: abandon.
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = in_use_by_tensor_.find(tensor);
  Block block;
  if (it != in_use_by_tensor_.end()) {
    block = it->second;
    in_use_by_tensor_.erase(it);
    in_use_by_ptr_.erase(block.va);
    in_use_by_addr_.erase(reinterpret_cast<uintptr_t>(block.va));
  } else {
    // Best-effort: reconstruct.
    block.tensor = tensor;
    block.va = GetVA(tensor);
    block.size = 0;
    block.device = -1;
    // If we can't track size/device, just free.
    nrt_tensor_free(&tensor);
    return;
  }

  free_blocks_.insert({block.size, block});
  stats_.used_bytes -= block.size;
  stats_.cached_bytes += block.size;
}

nrt_tensor_t* DeviceMemoryPool::FindByDataPtr(void* data, size_t* byte_offset) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!data) return nullptr;

  // Fast path: exact base match.
  auto it = in_use_by_ptr_.find(data);
  if (it != in_use_by_ptr_.end()) {
    if (byte_offset) *byte_offset = 0;
    return it->second.tensor;
  }

  // Range lookup: find the block with the greatest base <= data.
  uintptr_t addr = reinterpret_cast<uintptr_t>(data);
  auto ub = in_use_by_addr_.upper_bound(addr);
  if (ub == in_use_by_addr_.begin()) return nullptr;
  --ub;
  const Block& b = ub->second;
  uintptr_t base = reinterpret_cast<uintptr_t>(b.va);
  if (addr < base) return nullptr;
  uintptr_t end = base + static_cast<uintptr_t>(b.size);
  if (addr >= end) return nullptr;
  if (byte_offset) *byte_offset = static_cast<size_t>(addr - base);
  return b.tensor;
}

bool DeviceMemoryPool::CopyByDataPtr(void* dst, const void* src, size_t size) const noexcept {
  if (!dst || !src) return false;
  if (Runtime::Global().IsShuttingDown()) return false;

  size_t src_off = 0;
  size_t dst_off = 0;
  nrt_tensor_t* src_t = FindByDataPtr(const_cast<void*>(src), &src_off);
  nrt_tensor_t* dst_t = FindByDataPtr(dst, &dst_off);
  if (!src_t || !dst_t) return false;

  // Validate device match and bounds.
  int src_dev = -1;
  int dst_dev = -1;
  size_t src_size = 0;
  size_t dst_size = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it_s = in_use_by_tensor_.find(src_t);
    auto it_d = in_use_by_tensor_.find(dst_t);
    if (it_s == in_use_by_tensor_.end() || it_d == in_use_by_tensor_.end()) return false;
    src_dev = it_s->second.device;
    dst_dev = it_d->second.device;
    src_size = it_s->second.size;
    dst_size = it_d->second.size;
  }
  if (src_dev != dst_dev) return false;
  if (src_off + size > src_size) return false;
  if (dst_off + size > dst_size) return false;

  NRT_STATUS status = nrt_tensor_copy(src_t, src_off, dst_t, dst_off, size);
  return status == NRT_SUCCESS;
}

void DeviceMemoryPool::ClearCache() noexcept {
  if (Runtime::Global().IsShuttingDown()) return;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& kv : free_blocks_) {
    if (kv.second.tensor) nrt_tensor_free(&kv.second.tensor);
  }
  stats_.total_bytes -= stats_.cached_bytes;
  stats_.cached_bytes = 0;
  free_blocks_.clear();

  for (auto& kv : zero_tensors_) {
    if (kv.second) nrt_tensor_free(&kv.second);
  }
  zero_tensors_.clear();
  zero_tensor_sizes_.clear();
}

size_t DeviceMemoryPool::CachedBlocks() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return free_blocks_.size();
}

PoolStats DeviceMemoryPool::GetStats() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

nrt_tensor_t* DeviceMemoryPool::GetZeroTensor(size_t min_size, int device) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!Runtime::Global().IsInitialized()) {
    throw std::runtime_error("spiky: NRT is not initialized (call spiky.init() first)");
  }

  auto it = zero_tensors_.find(device);
  if (it != zero_tensors_.end()) {
    auto sz_it = zero_tensor_sizes_.find(device);
    if (sz_it != zero_tensor_sizes_.end() && sz_it->second >= min_size) {
      return it->second;
    }
    if (it->second) nrt_tensor_free(&it->second);
    zero_tensors_.erase(it);
    zero_tensor_sizes_.erase(device);
  }

  constexpr size_t kChunk = 1 * 1024 * 1024;
  size_t alloc_size = ((min_size + kChunk - 1) / kChunk) * kChunk;

  nrt_tensor_t* t = nullptr;
  NRT_STATUS status = nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, device, alloc_size, "spiky_zero", &t);
  if (status != NRT_SUCCESS || !t) {
    throw std::runtime_error("spiky: failed to allocate zero tensor");
  }

  std::vector<char> zeros(alloc_size, 0);
  status = nrt_tensor_write(t, zeros.data(), 0, alloc_size);
  if (status != NRT_SUCCESS) {
    nrt_tensor_free(&t);
    throw std::runtime_error("spiky: failed to init zero tensor");
  }

  zero_tensors_[device] = t;
  zero_tensor_sizes_[device] = alloc_size;
  return t;
}

}  // namespace spiky
