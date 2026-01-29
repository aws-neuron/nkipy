// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/StorageImpl.h>
#include <nrt/nrt.h>

namespace spike_torch {

// Custom StorageImpl for spike tensors
// Stores nrt_tensor_t* in the DataPtr context for NRT operations
struct SpikeStorageImpl : public c10::StorageImpl {
  SpikeStorageImpl(use_byte_size_t use_byte_size, size_t size_bytes,
                   at::DataPtr data_ptr, at::Allocator *allocator,
                   bool resizable);

  ~SpikeStorageImpl() override = default;

  void release_resources() override;

  // Get the underlying NRT tensor from the DataPtr context
  nrt_tensor_t *nrt_tensor() const {
    auto ctx = data_ptr().get_context();
    return ctx ? static_cast<nrt_tensor_t *>(ctx) : nullptr;
  }
};

} // namespace spike_torch
