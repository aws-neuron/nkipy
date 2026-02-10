// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {
#include <nrt/nrt.h>
}

namespace spiky {

// Stable C ABI for spiky_torch. Passed as a PyCapsule.
struct SpikyTorchAllocation {
  void* data;
  void* ctx;       // nrt_tensor_t*
  int device;
  size_t size;
};

struct SpikyTorchABI {
  uint32_t abi_version;

  bool (*is_initialized)();
  bool (*is_shutting_down)();

  // Allocate NRT tensor and return {data, ctx}. Throws via C++ exceptions in core;
  // spiky_torch side should catch and rethrow as TORCH_CHECK.
  SpikyTorchAllocation (*alloc)(int device, size_t size);

  // Release ctx (nrt_tensor_t*) back to shared pool. Must be noexcept.
  void (*release)(void* ctx) noexcept;

  // Find nrt_tensor_t* by data pointer (for hooks/copy).
  nrt_tensor_t* (*find_tensor)(void* data) noexcept;

  // Device-to-device copy by data pointers (same device only; returns false on failure).
  bool (*copy_by_ptr)(void* dst, const void* src, size_t size) noexcept;

  // Cache control.
  void (*empty_cache)() noexcept;
  size_t (*cached_blocks)() noexcept;
};

// Capsule name for PyCapsule import.
inline constexpr const char* kSpikyTorchCapsuleName = "spiky.torch_abi";

}  // namespace spiky

