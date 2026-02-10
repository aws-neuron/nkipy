// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>

namespace spiky_torch {

struct SpikyGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override;
  c10::Device exchangeDevice(c10::Device device) const override;
  c10::Device getDevice() const override;
  void setDevice(c10::Device device) const override;
  void uncheckedSetDevice(c10::Device device) const noexcept override;
  c10::Stream getStream(c10::Device device) const noexcept override;
  c10::Stream exchangeStream(c10::Stream stream) const noexcept override;
  c10::DeviceIndex deviceCount() const noexcept override;
};

}  // namespace spiky_torch

