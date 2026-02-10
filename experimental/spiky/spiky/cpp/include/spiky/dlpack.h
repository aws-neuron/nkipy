#pragma once

// Minimal DLPack definitions used by spiky. This avoids depending on an external
// dlpack header in the build environment.

#include <cstdint>

// NOLINTBEGIN(readability-identifier-naming)
enum DLDeviceType : int32_t {
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLOpenCL = 4,
  kDLVulkan = 7,
  kDLMetal = 8,
  kDLVPI = 9,
  kDLROCM = 10,
  kDLExtDev = 12,
};

struct DLDevice {
  DLDeviceType device_type;
  int32_t device_id;
};

enum DLDataTypeCode : uint8_t {
  kDLInt = 0,
  kDLUInt = 1,
  kDLFloat = 2,
  kDLBfloat = 4,
};

struct DLDataType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct DLTensor {
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
};
// NOLINTEND(readability-identifier-naming)

