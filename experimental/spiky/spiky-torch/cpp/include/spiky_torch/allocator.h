#pragma once

#include <cstddef>

namespace c10 {
class Allocator;
}

namespace spiky_torch {
namespace allocator {

c10::Allocator* get();
void empty_cache();
size_t get_cached_blocks();
bool copy_tensor_data(void* dest, const void* src, size_t size) noexcept;

}  // namespace allocator
}  // namespace spiky_torch
