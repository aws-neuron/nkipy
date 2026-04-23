#pragma once

#include "define.h"
#include "rdma_context.h"

class MemoryAllocator {
 public:
  explicit MemoryAllocator() : page_size_(sysconf(_SC_PAGESIZE)) {
    if (page_size_ <= 0) {
      page_size_ = 4096;  // fallback to 4KB
    }
  }

  ~MemoryAllocator() = default;

  std::shared_ptr<RegMemBlock> allocate(
      size_t size, MemoryType type,
      std::shared_ptr<RdmaContext> const ctx = nullptr) {
    void* addr = nullptr;
    struct ibv_mr* mr = nullptr;

    if (type == MemoryType::GPU) {
      throw std::runtime_error("GPU memory allocation is not supported");
    } else {  // MemoryType::HOST
      addr = allocateHost(size);
    }

    if (!addr) {
      throw std::runtime_error("Failed to allocate memory");
    }

    // Create RegMemBlock with custom deleter
    auto deleter = [this, type](RegMemBlock* block) {
      if (block) {
        if (block->addr) deallocateRaw(block->addr, type);
        delete block;
      }
    };

    auto block = new RegMemBlock(addr, size, type);
    if (ctx) {
      mr = ctx->regMem(addr, size);

      if (!mr) {
        deallocateRaw(addr, type);
        throw std::runtime_error("Failed to register memory with RDMA");
      }
      block->setMRByContextID(ctx->getContextID(), mr);
    }

    return std::shared_ptr<RegMemBlock>(block, deleter);
  }

 private:
  long page_size_;

  void* allocateHost(size_t size) {
    void* addr = nullptr;
    int ret = posix_memalign(&addr, page_size_, size);
    if (ret != 0) {
      std::cerr << "posix_memalign failed with error: " << ret << std::endl;
      return nullptr;
    }
    memset(addr, 0, size);  // Initialize memory to zero
    return addr;
  }

  // Deallocate memory without RDMA deregistration
  void deallocateRaw(void* addr, MemoryType type) {
    if (!addr) return;

    free(addr);
  }
};
