#ifndef SPIKE_SRC_INCLUDE_DTYPE_H
#define SPIKE_SRC_INCLUDE_DTYPE_H

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace spike {

// Macro defining all supported dtypes: (name, element_size_bytes,
// elements_per_unit) For packed types, elements_per_unit > 1 (e.g.,
// FLOAT8E4M3FN_X4 has 4 elements per 4-byte unit). For non-packed types,
// elements_per_unit == 1.
#define DTYPE_ELEMS(X)                                                         \
  X(INT8, 1, 1)                                                                \
  X(INT16, 2, 1)                                                               \
  X(INT32, 4, 1)                                                               \
  X(INT64, 8, 1)                                                               \
  X(UINT8, 1, 1)                                                               \
  X(UINT16, 2, 1)                                                              \
  X(UINT32, 4, 1)                                                              \
  X(UINT64, 8, 1)                                                              \
  X(FLOAT16, 2, 1)                                                             \
  X(FLOAT32, 4, 1)                                                             \
  X(BFLOAT16, 2, 1)                                                            \
  X(FLOAT8E3M4, 1, 1)                                                          \
  X(FLOAT8E4M3, 1, 1)                                                          \
  X(FLOAT8E4M3FN, 1, 1)                                                        \
  X(FLOAT8E5M2, 1, 1)                                                          \
  X(FLOAT32R, 4, 1)                                                            \
  X(FLOAT4E2M1FN_X4, 4, 4)                                                     \
  X(FLOAT8E4M3FN_X4, 4, 4)                                                     \
  X(FLOAT8E5M2_X4, 4, 4)

enum class DTypeKind {
#define DTYPE_ENUM(name, size, elems) name,
  DTYPE_ELEMS(DTYPE_ENUM)
#undef DTYPE_ENUM
};

struct DType {
  DTypeKind kind;
  uint32_t element_size;      // bytes per storage unit
  uint32_t elements_per_unit; // logical elements per storage unit

  uint32_t bytes_per_element() const {
    return element_size / elements_per_unit;
  }

  static std::optional<DType> from(std::string_view name) {
    // Map lowercase NRT dtype strings to DType values
#define DTYPE_FROM(dtype_name, size, elems)                                    \
  if (name == to_nrt_string(DTypeKind::dtype_name))                            \
    return DType{DTypeKind::dtype_name, size, elems};
    DTYPE_ELEMS(DTYPE_FROM)
#undef DTYPE_FROM
    return std::nullopt;
  }

  static constexpr const char *to_nrt_string(DTypeKind kind) {
    switch (kind) {
    case DTypeKind::INT8:
      return "int8";
    case DTypeKind::INT16:
      return "int16";
    case DTypeKind::INT32:
      return "int32";
    case DTypeKind::INT64:
      return "int64";
    case DTypeKind::UINT8:
      return "uint8";
    case DTypeKind::UINT16:
      return "uint16";
    case DTypeKind::UINT32:
      return "uint32";
    case DTypeKind::UINT64:
      return "uint64";
    case DTypeKind::FLOAT16:
      return "float16";
    case DTypeKind::FLOAT32:
      return "float32";
    case DTypeKind::BFLOAT16:
      return "bfloat16";
    case DTypeKind::FLOAT8E3M4:
      return "float8e3m4";
    case DTypeKind::FLOAT8E4M3:
      return "float8e4m3";
    case DTypeKind::FLOAT8E4M3FN:
      return "float8e4m3fn";
    case DTypeKind::FLOAT8E5M2:
      return "float8e5m2";
    case DTypeKind::FLOAT32R:
      return "float32r";
    case DTypeKind::FLOAT4E2M1FN_X4:
      return "float4e2m1fn_x4";
    case DTypeKind::FLOAT8E4M3FN_X4:
      return "float8e4m3fn_x4";
    case DTypeKind::FLOAT8E5M2_X4:
      return "float8e5m2_x4";
    }
    return "unknown";
  }

  std::string name() const { return to_nrt_string(kind); }
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_DTYPE_H
