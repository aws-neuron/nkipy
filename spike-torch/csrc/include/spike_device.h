// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace spike_torch {
namespace device {

// Get the current spike device index (thread-local)
int current_device();

// Set the current spike device (thread-local)
void set_device(int device);

// Get the total number of available spike devices
int device_count();

// Check if spike devices are available
bool is_available();

} // namespace device
} // namespace spike_torch
