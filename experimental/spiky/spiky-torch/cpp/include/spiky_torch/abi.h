// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

// This header is shared between spiky core and spiky_torch via the PyCapsule ABI.
// We duplicate the struct layout here (keep in sync with spiky/torch_abi.h).

extern "C" {
#include <nrt/nrt.h>
}

namespace spiky_torch {

struct SpikyTorchAllocation {
  void* data;
  void* ctx;  // nrt_tensor_t*
  int device;
  size_t size;
};

struct SpikyTorchABI {
  uint32_t abi_version;
  bool (*is_initialized)();
  bool (*is_shutting_down)();
  SpikyTorchAllocation (*alloc)(int device, size_t size);
  void (*release)(void* ctx) noexcept;
  nrt_tensor_t* (*find_tensor)(void* data) noexcept;
  bool (*copy_by_ptr)(void* dst, const void* src, size_t size) noexcept;
  void (*empty_cache)() noexcept;
  size_t (*cached_blocks)() noexcept;
};

inline constexpr const char* kSpikyTorchCapsuleName = "spiky.torch_abi";

const SpikyTorchABI* GetABI();
void LoadABIFromPython();

}  // namespace spiky_torch

