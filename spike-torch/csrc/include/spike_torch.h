// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Main header for spike-torch
// Includes all necessary headers for spike PyTorch integration

#include "spike_device.h"
#include "spike_guard_impl.h"
#include "spike_tensor_impl.h"
#include "spike_storage_impl.h"
#include "spike_allocator.h"
#include "spike_hooks.h"

namespace spike_torch {

// Register spike as a PyTorch device
// This should be called once during module initialization
void register_spike_device();

} // namespace spike_torch
