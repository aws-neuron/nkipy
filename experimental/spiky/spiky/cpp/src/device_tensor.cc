// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spiky/device_tensor.h"

#include "spiky/pool.h"
#include "spiky/runtime.h"

#include <stdexcept>

namespace spiky {

DeviceTensor::DeviceTensor(nrt_tensor_t* t, int device, size_t size_bytes,
                           std::vector<int64_t> shape, std::string dtype)
    : tensor_(t), device_(device), size_bytes_(size_bytes),
      shape_(std::move(shape)), dtype_(std::move(dtype)) {}

DeviceTensor::~DeviceTensor() {
  if (tensor_) {
    DeviceMemoryPool::Global().Release(tensor_);
    tensor_ = nullptr;
  }
}

DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept {
  *this = std::move(other);
}

DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept {
  if (this != &other) {
    if (tensor_) DeviceMemoryPool::Global().Release(tensor_);
    tensor_ = other.tensor_;
    device_ = other.device_;
    size_bytes_ = other.size_bytes_;
    shape_ = std::move(other.shape_);
    dtype_ = std::move(other.dtype_);
    other.tensor_ = nullptr;
  }
  return *this;
}

std::vector<uint8_t> DeviceTensor::read_to_bytes() const {
  if (!tensor_) return {};
  if (!Runtime::Global().IsInitialized()) {
    throw std::runtime_error("spiky: runtime not initialized");
  }
  std::vector<uint8_t> out(size_bytes_);
  if (size_bytes_ == 0) return out;
  NRT_STATUS status = nrt_tensor_read(tensor_, out.data(), 0, size_bytes_);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error("spiky: nrt_tensor_read failed");
  }
  return out;
}

void DeviceTensor::read_to(void* dst, size_t nbytes, size_t byte_offset) const {
  if (!tensor_) return;
  if (!Runtime::Global().IsInitialized()) {
    throw std::runtime_error("spiky: runtime not initialized");
  }
  if (nbytes == 0) return;
  if (byte_offset + nbytes > size_bytes_) {
    throw std::runtime_error("spiky: read_to out of bounds");
  }
  NRT_STATUS status = nrt_tensor_read(tensor_, dst, byte_offset, nbytes);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error("spiky: nrt_tensor_read failed");
  }
}

}  // namespace spiky
