// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "include/spike_storage_impl.h"

namespace spike_torch {

SpikeStorageImpl::SpikeStorageImpl(use_byte_size_t use_byte_size,
                                   size_t size_bytes, at::DataPtr data_ptr,
                                   at::Allocator *allocator, bool resizable)
    : c10::StorageImpl(use_byte_size, size_bytes, std::move(data_ptr),
                       allocator, resizable) {
  // The nrt_tensor is stored in the DataPtr context
  // We'll retrieve it when needed through the public API
}

void SpikeStorageImpl::release_resources() {
  // The DataPtr deleter will handle freeing the nrt_tensor
  // We just need to call the base class implementation
  StorageImpl::release_resources();
}

} // namespace spike_torch
