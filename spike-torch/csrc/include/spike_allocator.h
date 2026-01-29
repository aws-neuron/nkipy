// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/Allocator.h>
#include <nrt/nrt.h>

namespace spike_torch {
namespace allocator {

// Get the singleton spike allocator instance
c10::Allocator *get();

// Empty the memory cache (release cached but unused memory)
void empty_cache();

// Get the number of cached blocks across all devices
size_t get_cached_blocks();

// Clear allocator state (cache + allocation registry).
// Intended for spike_torch.reset() so subsequent runs can reinitialize cleanly.
// Also sets a shutdown flag so deleters skip NRT calls after this point.
void clear_allocator_state();

// Reset the shutdown flag after NRT is re-initialized.
// Call this after get_spike_singleton() to indicate NRT is ready for operations.
void mark_allocator_ready();

// Copy data between spike tensors
// Returns true if successful, false if tensors not found
bool copy_tensor_data(void *dest, const void *src, size_t size);

// Find the nrt_tensor for a given data pointer
// Returns the nrt_tensor_t* if found, nullptr otherwise
nrt_tensor_t *find_tensor(void *ptr);

// Get the nrt_tensor from a DataPtr context
// The context is the nrt_tensor_t* stored in the DataPtr
nrt_tensor_t *get_tensor_from_context(void *ctx);

// Tensor info struct for Python binding
struct TensorInfo {
  nrt_tensor_t *tensor;
  size_t size;
  int device;
};

// Get full tensor info (tensor ptr, size, device) for a data pointer
// Returns TensorInfo with tensor=nullptr if not found
TensorInfo get_tensor_info(void *ptr);

} // namespace allocator
} // namespace spike_torch
