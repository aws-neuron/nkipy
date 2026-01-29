// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <c10/util/Optional.h>

namespace spike_torch {

// SpikeHooksInterface implements the PrivateUse1HooksInterface to enable
// autograd support for spike tensors. This interface is required for
// gradient computation and other autograd operations.
class SpikeHooksInterface : public at::PrivateUse1HooksInterface {
public:
  SpikeHooksInterface() = default;

  ~SpikeHooksInterface() override = default;

  // Initialize the spike runtime if needed
  void init() const override;

  // Check if a pointer is pinned memory
  // Only spike-allocated device pointers are considered pinned
  bool isPinnedPtr(const void *data) const override;

  // Check if spike runtime is available / has primary context
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override;

  // Get the pinned memory allocator
  c10::Allocator *getPinnedMemoryAllocator() const override {
    return at::getHostAllocator(at::kPrivateUse1);
  }

  // Resize storage for spike tensors
  void resizePrivateUse1Bytes(const c10::Storage &storage,
                              size_t new_bytes) const override;

  // Check if spike runtime is available
  bool isAvailable() const;
};

// Get the singleton instance of SpikeHooksInterface
at::PrivateUse1HooksInterface *get_spike_hooks();

// Explicitly register the hooks interface with PyTorch
void register_spike_hooks();

} // namespace spike_torch
