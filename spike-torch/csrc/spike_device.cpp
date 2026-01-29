// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "include/spike_device.h"

#include <stdexcept>
#include <string>

extern "C" {
#include <nrt/nrt.h>
}

namespace spike_torch {
namespace device {

namespace {
// Thread-local current device index
thread_local int current_device_index = 0;
} // namespace

int current_device() { return current_device_index; }

void set_device(int device) {
  // Validate device index
  if (device < 0) {
    throw std::invalid_argument("Device index must be non-negative, got " +
                                std::to_string(device));
  }

  int count = device_count();
  if (device >= count) {
    throw std::invalid_argument("Device index " + std::to_string(device) +
                                " is out of range. Valid range is 0 to " +
                                std::to_string(count - 1));
  }

  current_device_index = device;
}

int device_count() {
  uint32_t count = 0;
  NRT_STATUS status = nrt_get_visible_nc_count(&count);
  if (status != NRT_SUCCESS) {
    return 0;
  }
  return static_cast<int>(count);
}

bool is_available() { return device_count() > 0; }

} // namespace device
} // namespace spike_torch
