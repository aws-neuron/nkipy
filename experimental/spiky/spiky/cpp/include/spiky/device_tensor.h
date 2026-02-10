#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

extern "C" {
#include <nrt/nrt.h>
}

namespace spiky {

// Python-visible wrapper for an NRT device tensor.
// Owned by the shared pool; destructor returns it to the pool.
class DeviceTensor {
 public:
  DeviceTensor() = default;
  DeviceTensor(nrt_tensor_t* t, int device, size_t size_bytes,
               std::vector<int64_t> shape, std::string dtype);
  ~DeviceTensor();

  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  DeviceTensor(DeviceTensor&& other) noexcept;
  DeviceTensor& operator=(DeviceTensor&& other) noexcept;

  uintptr_t nrt_ptr() const { return reinterpret_cast<uintptr_t>(tensor_); }
  int device() const { return device_; }
  int64_t size_bytes() const { return static_cast<int64_t>(size_bytes_); }
  const std::vector<int64_t>& shape() const { return shape_; }
  const std::string& dtype() const { return dtype_; }

  // Copy to CPU bytes.
  std::vector<uint8_t> read_to_bytes() const;

  // Read device bytes directly into a host buffer.
  void read_to(void* dst, size_t nbytes, size_t byte_offset = 0) const;

 private:
  friend class DeviceMemoryPool;

  nrt_tensor_t* tensor_{nullptr};
  int device_{-1};
  size_t size_bytes_{0};
  std::vector<int64_t> shape_;
  std::string dtype_;
};

}  // namespace spiky
