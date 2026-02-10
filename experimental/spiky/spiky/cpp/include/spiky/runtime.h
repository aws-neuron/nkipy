// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace spiky {

// Explicit NRT lifecycle: init/close are refcounted.
class Runtime {
 public:
  static Runtime& Global();

  void Init(int64_t device_id = 0);
  void Close();

  bool IsInitialized() const;
  int64_t DeviceId() const;
  int64_t DeviceCount() const;

  // For torch deleters and shutdown-safe behavior.
  void MarkShuttingDown();
  void MarkReady();
  bool IsShuttingDown() const;

 private:
  Runtime() = default;
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  void InitImpl(int64_t device_id);
  void CloseImpl();

  int64_t device_id_{0};
  int64_t refcount_{0};
  bool initialized_{false};
};

}  // namespace spiky

