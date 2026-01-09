// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef SPIKE_SRC_INCLUDE_NRT_WRAPPER_H
#define SPIKE_SRC_INCLUDE_NRT_WRAPPER_H

#include "error.h"

#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_profile.h>

#include <cstdint>
#include <string>

namespace spike {

// RAII wrapper for NRT runtime
class NrtRuntime {
public:
  NrtRuntime(const std::string &fw_version = "spike",
             const std::string &fal_version = "0.1.0", bool owned = true);
  ~NrtRuntime();

  // Non-copyable, movable
  NrtRuntime(const NrtRuntime &) = delete;
  NrtRuntime &operator=(const NrtRuntime &) = delete;
  NrtRuntime(NrtRuntime &&) = default;
  NrtRuntime &operator=(NrtRuntime &&) = default;

  static uint32_t get_visible_nc_count();

private:
  bool initialized_;
  bool owned_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_NRT_WRAPPER_H
