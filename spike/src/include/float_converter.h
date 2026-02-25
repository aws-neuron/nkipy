// Single-value float conversions for TensorFormatter (debug stream display).
//
// Converts reduced-precision types from NRT debug stream bytes to float32 for
// human-readable output.

#ifndef SPIKE_SRC_INCLUDE_FLOAT_CONVERTER_H
#define SPIKE_SRC_INCLUDE_FLOAT_CONVERTER_H

#include <cstdint>
#include <cstring>

namespace spike {

template <typename To, typename From> static inline To unpun_cast(From val) {
  static_assert(sizeof(To) == sizeof(From), "sizes must match");
  To result;
  std::memcpy(&result, &val, sizeof(To));
  return result;
}

// float16 (IEEE 754 half-precision) -> float32
// Based on the standard bit-manipulation algorithm (same as
// Eigen::half_to_float).
static inline uint32_t fp16_to_fp32(uint16_t val) {
  uint32_t sign = (static_cast<uint32_t>(val) & 0x8000) << 16;
  uint32_t o_bits = (val & 0x7fff) << 13;
  uint32_t exp = o_bits & (0x7c00 << 13);
  o_bits += (127 - 15) << 23; // exponent rebias

  if (exp == (0x7c00 << 13)) {
    // Inf/NaN
    o_bits += (128 - 16) << 23;
  } else if (exp == 0) {
    // Zero/subnormal
    o_bits += 1 << 23;
    // Renormalize by subtracting the magic number
    float magic = unpun_cast<float>(static_cast<uint32_t>(113 << 23));
    o_bits = unpun_cast<uint32_t>(unpun_cast<float>(o_bits) - magic);
  }

  return sign | o_bits;
}

// bfloat16 -> float32 (left-shift 16 bits into upper half of float32)
static inline uint32_t bfp16_to_fp32(uint16_t val) {
  return static_cast<uint32_t>(val) << 16;
}

// float8 E3M4 -> float32
static inline uint32_t cast_float8e3_to_fp32(uint8_t val) {
  uint32_t sign = (static_cast<uint32_t>(val) >> 7) & 1;
  uint32_t exp = (static_cast<uint32_t>(val) >> 4) & 0x7;
  uint32_t mant = static_cast<uint32_t>(val) & 0xF;

  if (exp == 0 && mant == 0)
    return sign << 31;
  // All exp==max patterns are special (matches neuron_dtypes upconv_fp_to_fp32)
  if (exp == 0x7) {
    if (mant == 0)
      return (sign << 31) | 0x7F800000; // ±inf
    return 0x7FC00000;                  // NaN (quiet, positive)
  }

  if (exp == 0) {
    // Subnormal: find leading 1
    while ((mant & 0x8) == 0) {
      mant <<= 1;
      exp--;
    }
    mant &= 0x7; // remove implicit 1
    exp++;
  }

  // E3M4: bias=3, float32 bias=127
  uint32_t fp32_exp =
      static_cast<uint32_t>(static_cast<int32_t>(exp) - 3 + 127);
  uint32_t fp32_mant = mant << (23 - 4);
  return (sign << 31) | (fp32_exp << 23) | fp32_mant;
}

// float8 E4M3 -> float32
static inline uint32_t cast_float8e4_to_fp32(uint8_t val) {
  uint32_t sign = (static_cast<uint32_t>(val) >> 7) & 1;
  uint32_t exp = (static_cast<uint32_t>(val) >> 3) & 0xF;
  uint32_t mant = static_cast<uint32_t>(val) & 0x7;

  if (exp == 0 && mant == 0)
    return sign << 31;
  // All exp==max patterns are special (matches neuron_dtypes upconv_fp_to_fp32)
  if (exp == 0xF) {
    if (mant == 0)
      return (sign << 31) | 0x7F800000; // ±inf
    return 0x7FC00000;                  // NaN (quiet, positive)
  }

  if (exp == 0) {
    while ((mant & 0x4) == 0) {
      mant <<= 1;
      exp--;
    }
    mant &= 0x3;
    exp++;
  }

  // E4M3: bias=7, float32 bias=127
  uint32_t fp32_exp =
      static_cast<uint32_t>(static_cast<int32_t>(exp) - 7 + 127);
  uint32_t fp32_mant = mant << (23 - 3);
  return (sign << 31) | (fp32_exp << 23) | fp32_mant;
}

// float8 E4M3FN -> float32 (same exponent layout as E4M3, no inf, NaN=0x7F)
static inline uint32_t cast_float8e4m3fn_to_fp32(uint8_t val) {
  uint32_t sign = (static_cast<uint32_t>(val) >> 7) & 1;
  uint32_t exp = (static_cast<uint32_t>(val) >> 3) & 0xF;
  uint32_t mant = static_cast<uint32_t>(val) & 0x7;

  if (exp == 0 && mant == 0)
    return sign << 31;
  if (exp == 0xF && mant == 0x7)
    return 0x7FC00000; // NaN (sign-less)

  if (exp == 0) {
    while ((mant & 0x4) == 0) {
      mant <<= 1;
      exp--;
    }
    mant &= 0x3;
    exp++;
  }

  uint32_t fp32_exp =
      static_cast<uint32_t>(static_cast<int32_t>(exp) - 7 + 127);
  uint32_t fp32_mant = mant << (23 - 3);
  return (sign << 31) | (fp32_exp << 23) | fp32_mant;
}

// float8 E5M2 -> float32
static inline uint32_t cast_float8e5_to_fp32(uint8_t val) {
  uint32_t sign = (static_cast<uint32_t>(val) >> 7) & 1;
  uint32_t exp = (static_cast<uint32_t>(val) >> 2) & 0x1F;
  uint32_t mant = static_cast<uint32_t>(val) & 0x3;

  if (exp == 0 && mant == 0)
    return sign << 31;
  if (exp == 0x1F)
    return (sign << 31) | 0x7F800000 | (mant << 21); // inf or NaN

  if (exp == 0) {
    while ((mant & 0x2) == 0) {
      mant <<= 1;
      exp--;
    }
    mant &= 0x1;
    exp++;
  }

  // E5M2: bias=15, float32 bias=127
  uint32_t fp32_exp =
      static_cast<uint32_t>(static_cast<int32_t>(exp) - 15 + 127);
  uint32_t fp32_mant = mant << (23 - 2);
  return (sign << 31) | (fp32_exp << 23) | fp32_mant;
}

// float32r (reduced-range float32) -> float32
// float32r has the same format as float32 but with reduced exponent range.
// The conversion is identity since the bit pattern is the same.
static inline uint32_t fp32r_to_fp32(uint32_t val) { return val; }
static inline uint32_t fp32_to_fp32r(uint32_t val) { return val; }

// Packed types: 4 elements packed into 32 bits (byte-padded, 1 element per
// byte).
//
// FLOAT4E2M1FN_X4: Assumes SBUF byte-padded layout where each 4-bit float4
// occupies a full byte (bits [3:0] = sign|exp|mantissa, bits [7:4] unused).
static inline void cast_float4e2m1fn_x4_to_fp32(uint32_t packed, float *out) {
  for (int i = 0; i < 4; ++i) {
    uint8_t bits = (packed >> (i * 8)) & 0xFF;
    uint32_t sign = (static_cast<uint32_t>(bits) >> 3) & 1;
    uint32_t exp = (static_cast<uint32_t>(bits) >> 1) & 0x3;
    uint32_t mant = static_cast<uint32_t>(bits) & 0x1;

    uint32_t fp32;
    if (exp == 0 && mant == 0) {
      fp32 = sign << 31;
    } else if (exp == 0) {
      // Subnormal: value = (-1)^sign * 0.5
      fp32 = (sign << 31) | (126 << 23); // 0.5
    } else {
      // E2M1: bias=1
      uint32_t fp32_exp =
          static_cast<uint32_t>(static_cast<int32_t>(exp) - 1 + 127);
      uint32_t fp32_mant = mant << (23 - 1);
      fp32 = (sign << 31) | (fp32_exp << 23) | fp32_mant;
    }
    out[i] = unpun_cast<float>(fp32);
  }
}

// FLOAT8E4M3FN_X4: 4 float8e4m3fn values packed in 32 bits
static inline void cast_float8e4m3fn_x4_to_fp32(uint32_t packed, float *out) {
  for (int i = 0; i < 4; ++i) {
    uint8_t byte = (packed >> (i * 8)) & 0xFF;
    out[i] = unpun_cast<float>(cast_float8e4m3fn_to_fp32(byte));
  }
}

// FLOAT8E5M2_X4: 4 float8e5m2 values packed in 32 bits
static inline void cast_float8e5m2_x4_to_fp32(uint32_t packed, float *out) {
  for (int i = 0; i < 4; ++i) {
    uint8_t byte = (packed >> (i * 8)) & 0xFF;
    out[i] = unpun_cast<float>(cast_float8e5_to_fp32(byte));
  }
}

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_FLOAT_CONVERTER_H
