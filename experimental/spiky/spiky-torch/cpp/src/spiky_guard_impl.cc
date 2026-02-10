#include "spiky_torch/guard_impl.h"

#include "spiky_torch/device.h"

namespace spiky_torch {

c10::DeviceType SpikyGuardImpl::type() const { return c10::DeviceType::PrivateUse1; }

c10::Device SpikyGuardImpl::exchangeDevice(c10::Device device_) const {
  auto old = getDevice();
  setDevice(device_);
  return old;
}

c10::Device SpikyGuardImpl::getDevice() const {
  return c10::Device(c10::DeviceType::PrivateUse1, device::current_device());
}

void SpikyGuardImpl::setDevice(c10::Device device_) const {
  device::set_device(device_.index());
}

void SpikyGuardImpl::uncheckedSetDevice(c10::Device device_) const noexcept {
  try {
    device::set_device(device_.index());
  } catch (...) {
  }
}

c10::Stream SpikyGuardImpl::getStream(c10::Device device_) const noexcept {
  return c10::Stream(c10::Stream::DEFAULT, device_);
}

c10::Stream SpikyGuardImpl::exchangeStream(c10::Stream stream) const noexcept {
  return stream;
}

c10::DeviceIndex SpikyGuardImpl::deviceCount() const noexcept {
  return static_cast<c10::DeviceIndex>(device::device_count());
}

}  // namespace spiky_torch

