#ifndef SPIKE_SRC_INCLUDE_TENSOR_FORMAT_H
#define SPIKE_SRC_INCLUDE_TENSOR_FORMAT_H

#include "dtype.h"
#include "float_converter.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace spike {

struct TensorFormatOptions {
  uint32_t edge_items = 3;
  uint32_t threshold = 1000;
  uint32_t precision = 4;
};

class TensorFormatter {
public:
  explicit TensorFormatter(TensorFormatOptions opts = {}) : opts_(opts) {}

  std::string format(const void *data, size_t data_size,
                     const std::string &dtype_str,
                     const std::vector<uint64_t> &shape) const {
    auto dtype_opt = DType::from(dtype_str);
    if (!dtype_opt) {
      return "Unsupported data type: " + dtype_str;
    }
    auto dtype = *dtype_opt;

    // Calculate total logical elements
    size_t total_elements = 1;
    for (auto dim : shape) {
      total_elements *= dim;
    }

    // Drop trailing dimensions of size 1 for display
    std::vector<uint64_t> display_shape = shape;
    while (display_shape.size() > 1 && display_shape.back() == 1) {
      display_shape.pop_back();
    }

    // Verify data size
    size_t storage_units = (total_elements + dtype.elements_per_unit - 1) /
                           dtype.elements_per_unit;
    size_t expected_bytes = storage_units * dtype.element_size;
    if (data_size < expected_bytes) {
      return "Data size mismatch: expected " + std::to_string(expected_bytes) +
             " bytes, got " + std::to_string(data_size) + " bytes";
    }

    // Scalar: empty shape
    if (display_shape.empty()) {
      return formatElement(data, 0, dtype);
    }

    std::string result;
    formatArray(static_cast<const uint8_t *>(data), display_shape, 0,
                total_elements, dtype, result);
    return result;
  }

private:
  void formatArray(const uint8_t *data, const std::vector<uint64_t> &shape,
                   size_t dim, size_t total_elements, const DType &dtype,
                   std::string &result) const {
    result += "[";
    size_t dim_size = shape[dim];

    if (dim == shape.size() - 1) {
      // Innermost dimension: format elements
      bool use_ellipsis =
          total_elements > opts_.threshold && dim_size > 2 * opts_.edge_items;
      if (use_ellipsis) {
        for (size_t i = 0; i < opts_.edge_items; ++i) {
          if (i > 0)
            result += ", ";
          result += formatElement(data, i, dtype);
        }
        result += ", ..., ";
        for (size_t i = dim_size - opts_.edge_items; i < dim_size; ++i) {
          if (i > dim_size - opts_.edge_items)
            result += ", ";
          result += formatElement(data, i, dtype);
        }
      } else {
        for (size_t i = 0; i < dim_size; ++i) {
          if (i > 0)
            result += ", ";
          result += formatElement(data, i, dtype);
        }
      }
    } else {
      // Outer dimension: recurse
      size_t stride = 1;
      for (size_t d = dim + 1; d < shape.size(); ++d) {
        stride *= shape[d];
      }
      size_t stride_bytes = strideBytes(stride, dtype);

      bool use_ellipsis =
          total_elements > opts_.threshold && dim_size > 2 * opts_.edge_items;
      std::string separator = ",\n" + std::string(dim + 1, ' ');

      if (use_ellipsis) {
        for (size_t i = 0; i < opts_.edge_items; ++i) {
          if (i > 0)
            result += separator;
          formatArray(data + i * stride_bytes, shape, dim + 1, total_elements,
                      dtype, result);
        }
        result += separator + "...";
        for (size_t i = dim_size - opts_.edge_items; i < dim_size; ++i) {
          result += separator;
          formatArray(data + i * stride_bytes, shape, dim + 1, total_elements,
                      dtype, result);
        }
      } else {
        for (size_t i = 0; i < dim_size; ++i) {
          if (i > 0)
            result += separator;
          formatArray(data + i * stride_bytes, shape, dim + 1, total_elements,
                      dtype, result);
        }
      }
    }
    result += "]";
  }

  static size_t strideBytes(size_t num_elements, const DType &dtype) {
    if (dtype.elements_per_unit > 1) {
      size_t units = (num_elements + dtype.elements_per_unit - 1) /
                     dtype.elements_per_unit;
      return units * dtype.element_size;
    }
    return num_elements * dtype.element_size;
  }

  std::string formatElement(const void *base, size_t index,
                            const DType &dtype) const {
    const auto *data = static_cast<const uint8_t *>(base);

    switch (dtype.kind) {
    case DTypeKind::INT8: {
      int8_t val;
      std::memcpy(&val, data + index * 1, 1);
      return std::to_string(val);
    }
    case DTypeKind::INT16: {
      int16_t val;
      std::memcpy(&val, data + index * 2, 2);
      return std::to_string(val);
    }
    case DTypeKind::INT32: {
      int32_t val;
      std::memcpy(&val, data + index * 4, 4);
      return std::to_string(val);
    }
    case DTypeKind::INT64: {
      int64_t val;
      std::memcpy(&val, data + index * 8, 8);
      return std::to_string(val);
    }
    case DTypeKind::UINT8: {
      uint8_t val;
      std::memcpy(&val, data + index * 1, 1);
      return std::to_string(val);
    }
    case DTypeKind::UINT16: {
      uint16_t val;
      std::memcpy(&val, data + index * 2, 2);
      return std::to_string(val);
    }
    case DTypeKind::UINT32: {
      uint32_t val;
      std::memcpy(&val, data + index * 4, 4);
      return std::to_string(val);
    }
    case DTypeKind::UINT64: {
      uint64_t val;
      std::memcpy(&val, data + index * 8, 8);
      return std::to_string(val);
    }
    case DTypeKind::FLOAT32: {
      float val;
      std::memcpy(&val, data + index * 4, 4);
      return formatFloat(val);
    }
    case DTypeKind::FLOAT16: {
      uint16_t raw;
      std::memcpy(&raw, data + index * 2, 2);
      float val = unpun_cast<float>(fp16_to_fp32(raw));
      return formatFloat(val);
    }
    case DTypeKind::BFLOAT16: {
      uint16_t raw;
      std::memcpy(&raw, data + index * 2, 2);
      float val = unpun_cast<float>(bfp16_to_fp32(raw));
      return formatFloat(val);
    }
    case DTypeKind::FLOAT8E3M4: {
      uint8_t raw = data[index];
      float val = unpun_cast<float>(cast_float8e3_to_fp32(raw));
      return formatFloat(val);
    }
    case DTypeKind::FLOAT8E4M3: {
      uint8_t raw = data[index];
      float val = unpun_cast<float>(cast_float8e4_to_fp32(raw));
      return formatFloat(val);
    }
    case DTypeKind::FLOAT8E4M3FN: {
      uint8_t raw = data[index];
      float val = unpun_cast<float>(cast_float8e4m3fn_to_fp32(raw));
      return formatFloat(val);
    }
    case DTypeKind::FLOAT8E5M2: {
      uint8_t raw = data[index];
      float val = unpun_cast<float>(cast_float8e5_to_fp32(raw));
      return formatFloat(val);
    }
    case DTypeKind::FLOAT32R: {
      uint32_t raw;
      std::memcpy(&raw, data + index * 4, 4);
      float val = unpun_cast<float>(fp32r_to_fp32(raw));
      return formatFloat(val);
    }
    case DTypeKind::FLOAT4E2M1FN_X4: {
      // Packed: 4 elements per 32-bit unit
      size_t unit_index = index / 4;
      size_t sub_index = index % 4;
      uint32_t packed;
      std::memcpy(&packed, data + unit_index * 4, 4);
      float out[4];
      cast_float4e2m1fn_x4_to_fp32(packed, out);
      return formatFloat(out[sub_index]);
    }
    case DTypeKind::FLOAT8E4M3FN_X4: {
      size_t unit_index = index / 4;
      size_t sub_index = index % 4;
      uint32_t packed;
      std::memcpy(&packed, data + unit_index * 4, 4);
      float out[4];
      cast_float8e4m3fn_x4_to_fp32(packed, out);
      return formatFloat(out[sub_index]);
    }
    case DTypeKind::FLOAT8E5M2_X4: {
      size_t unit_index = index / 4;
      size_t sub_index = index % 4;
      uint32_t packed;
      std::memcpy(&packed, data + unit_index * 4, 4);
      float out[4];
      cast_float8e5m2_x4_to_fp32(packed, out);
      return formatFloat(out[sub_index]);
    }
    }
    return "?";
  }

  std::string formatFloat(float val) const {
    if (val == 0.0f) {
      // Distinguish +0 and -0
      std::string zeros(opts_.precision, '0');
      if (std::signbit(val))
        return "-0." + zeros;
      return "0." + zeros;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", opts_.precision, val);
    return buf;
  }

  TensorFormatOptions opts_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_TENSOR_FORMAT_H
