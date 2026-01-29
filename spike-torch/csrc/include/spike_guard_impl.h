// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "spike_device.h"

namespace spike_torch {

// DeviceGuard implementation for spike device
// This enables device context management in PyTorch
struct SpikeGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  SpikeGuardImpl() = default;

  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
    c10::Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      device::set_device(d.index());
    }
    return old_device;
  }

  c10::Device getDevice() const override {
    return c10::Device(c10::DeviceType::PrivateUse1, device::current_device());
  }

  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
    device::set_device(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    device::set_device(d.index());
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
    // Neuron doesn't have async streams - return default stream
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    // Neuron doesn't have streams - just return the input stream
    return s;
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return static_cast<c10::DeviceIndex>(device::device_count());
  }

  // Event-related methods (not supported for Neuron)
  void record(void **event, const c10::Stream &stream,
              const c10::DeviceIndex device_index,
              const c10::EventFlag flag) const override {
    TORCH_CHECK(false, "Spike backend doesn't support events");
  }

  void block(void *event, const c10::Stream &stream) const override {
    TORCH_CHECK(false, "Spike backend doesn't support events");
  }

  bool queryEvent(void *event) const override {
    TORCH_CHECK(false, "Spike backend doesn't support events");
  }

  void destroyEvent(void *event,
                    const c10::DeviceIndex device_index) const noexcept override {
    // No-op
  }
};

} // namespace spike_torch
