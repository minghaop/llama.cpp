//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "Quantization.hpp"

namespace genie {
namespace quantization {

// Convert string to enum
DataType toDataType(const std::string& typeStr) {
  if (typeStr == "QNN_DATATYPE_UFIXED_POINT_4") return DataType::UFixed4;
  if (typeStr == "QNN_DATATYPE_SFIXED_POINT_8") return DataType::SFixed8;
  if (typeStr == "QNN_DATATYPE_SFIXED_POINT_16") return DataType::SFixed16;
  if (typeStr == "QNN_DATATYPE_UFIXED_POINT_8") return DataType::UFixed8;
  if (typeStr == "QNN_DATATYPE_UFIXED_POINT_16") return DataType::UFixed16;
  if (typeStr == "QNN_DATATYPE_FLOAT_16") return DataType::Float16;
  if (typeStr == "QNN_DATATYPE_FLOAT_32") return DataType::Float32;
  return DataType::Unknown;
}

// fixed-point to fixed-point
template <typename SrcT, typename DstT>
struct RequantConvert {
  static void run(const void* src,
                  size_t srcStartAddress,
                  void* dst,
                  size_t dstStartAddress,
                  size_t length,
                  double requantScale,
                  int32_t requantOffset,
                  double /*srcScale*/,
                  int32_t /*srcOffset*/) {
    const SrcT* s = static_cast<const SrcT*>(src) + srcStartAddress;
    DstT* d       = static_cast<DstT*>(dst);
    for (size_t i = 0; i < length; ++i) {
      const double val       = static_cast<double>(s[i]) * requantScale + requantOffset;
      d[dstStartAddress + i] = static_cast<DstT>(std::round(val));
    }
  }
};

// uint4 to ufixed-point
template <typename DstT>
struct RequantConvert<uint4_t, DstT> {
  static void run(const void* src,
                  size_t srcStartAddress,
                  void* dst,
                  size_t dstStartAddress,
                  size_t length,
                  double requantScale,
                  int32_t requantOffset,
                  double /*srcScale*/,
                  int32_t /*srcOffset*/) {
    const uint8_t* s = static_cast<const uint8_t*>(src) + srcStartAddress;
    DstT* d          = static_cast<DstT*>(dst);
    for (size_t j = 0; j < length / 2; ++j) {
      uint8_t high = (s[j] >> 4) & 0x0F;
      uint8_t low  = s[j] & 0x0F;
      d[dstStartAddress + 2 * j] =
          static_cast<DstT>(std::round(requantScale * low + requantOffset));
      d[dstStartAddress + 2 * j + 1] =
          static_cast<DstT>(std::round(requantScale * high + requantOffset));
    }
  }
};

// fixed-point to float
template <typename SrcT>
struct RequantConvert<SrcT, float> {
  static void run(const void* src,
                  size_t srcStartAddress,
                  void* dst,
                  size_t dstStartAddress,
                  size_t length,
                  double /*requantScale*/,
                  int32_t /*requantOffset*/,
                  double srcScale,
                  int32_t srcOffset) {
    const SrcT* s = static_cast<const SrcT*>(src) + srcStartAddress;
    float* d      = static_cast<float*>(dst);
    for (size_t i = 0; i < length; ++i) {
      // d = (s + srcOffset) * srcScale
      d[dstStartAddress + i] =
          static_cast<float>((static_cast<double>(s[i]) + srcOffset) * srcScale);
    }
  }
};

// float to fixed-point
template <typename DstT>
struct RequantConvert<float, DstT> {
  static void run(const void* src,
                  size_t srcStartAddress,
                  void* dst,
                  size_t dstStartAddress,
                  size_t length,
                  double scale,
                  int32_t offset,
                  double /*srcScale*/,
                  int32_t /*srcOffset*/) {
    const float* s = static_cast<const float*>(src) + srcStartAddress;
    DstT* d        = static_cast<DstT*>(dst);
    for (size_t i = 0; i < length; ++i) {
      // val = s/scale - offset
      const double val       = static_cast<double>(s[i]) / scale - offset;
      d[dstStartAddress + i] = static_cast<DstT>(std::round(val));
    }
  }
};

// float to float
template <>
struct RequantConvert<float, float> {
  static void run(const void* src,
                  size_t srcStartAddress,
                  void* dst,
                  size_t dstStartAddress,
                  size_t length,
                  double /*scale*/,
                  int32_t /*offset*/,
                  double /*srcScale*/,
                  int32_t /*srcOffset*/) {
    std::memcpy(static_cast<float*>(dst) + dstStartAddress,
                static_cast<const float*>(src) + srcStartAddress,
                length * sizeof(float));
  }
};

// Unified entry point: generic function template
template <typename SrcT, typename DstT>
void requantConvertFunc(const void* src,
                        size_t srcStartAddress,
                        void* dst,
                        size_t dstStartAddress,
                        size_t length,
                        double requantScale,
                        int32_t requantOffset,
                        double srcScale,
                        int32_t srcOffset) {
  RequantConvert<SrcT, DstT>::run(src,
                                  srcStartAddress,
                                  dst,
                                  dstStartAddress,
                                  length,
                                  requantScale,
                                  requantOffset,
                                  srcScale,
                                  srcOffset);
}

// Static mapping table initialization
static const std::map<TypePair, ConvertFunc>& getConverterMap() {
  static const std::map<TypePair, ConvertFunc> m = {
      // SFixed8
      {TypePair(DataType::SFixed8, DataType::SFixed8), &requantConvertFunc<int8_t, int8_t>},
      {TypePair(DataType::SFixed8, DataType::SFixed16), &requantConvertFunc<int8_t, int16_t>},
      {TypePair(DataType::SFixed8, DataType::Float32), &requantConvertFunc<int8_t, float>},
      // SFixed16
      {TypePair(DataType::SFixed16, DataType::SFixed8), &requantConvertFunc<int16_t, int8_t>},
      {TypePair(DataType::SFixed16, DataType::SFixed16), &requantConvertFunc<int16_t, int16_t>},
      {TypePair(DataType::SFixed16, DataType::Float32), &requantConvertFunc<int16_t, float>},
      // UFixed8
      {TypePair(DataType::UFixed8, DataType::UFixed8), &requantConvertFunc<uint8_t, uint8_t>},
      {TypePair(DataType::UFixed8, DataType::UFixed16), &requantConvertFunc<uint8_t, uint16_t>},
      {TypePair(DataType::UFixed8, DataType::Float32), &requantConvertFunc<uint8_t, float>},
      // UFixed16
      {TypePair(DataType::UFixed16, DataType::UFixed8), &requantConvertFunc<uint16_t, uint8_t>},
      {TypePair(DataType::UFixed16, DataType::UFixed16), &requantConvertFunc<uint16_t, uint16_t>},
      {TypePair(DataType::UFixed16, DataType::Float32), &requantConvertFunc<uint16_t, float>},
      // Float32
      {TypePair(DataType::Float32, DataType::UFixed8), &requantConvertFunc<float, uint8_t>},
      {TypePair(DataType::Float32, DataType::UFixed16), &requantConvertFunc<float, uint16_t>},
      {TypePair(DataType::Float32, DataType::SFixed8), &requantConvertFunc<float, int8_t>},
      {TypePair(DataType::Float32, DataType::SFixed16), &requantConvertFunc<float, int16_t>},
      {TypePair(DataType::Float32, DataType::Float32), &requantConvertFunc<float, float>},
      // UFixed4
      {TypePair(DataType::UFixed4, DataType::UFixed8), &requantConvertFunc<uint4_t, uint8_t>},
      {TypePair(DataType::UFixed4, DataType::UFixed16), &requantConvertFunc<uint4_t, uint16_t>},
  };
  return m;
}

void requantize(const void* src,
                const size_t srcOffset,
                const Encoding& srcEncoding,
                void* dst,
                size_t dstOffset,
                const Encoding& dstEncoding,
                const size_t length) {
  double requantScale   = srcEncoding.scale / dstEncoding.scale;
  int32_t requantOffset = srcEncoding.offset * requantScale - dstEncoding.offset;

  const auto& converterMap = getConverterMap();
  TypePair tp{srcEncoding.dataType, dstEncoding.dataType};
  auto it = converterMap.find(tp);
  if (it == converterMap.end()) {
    throw std::runtime_error("Unsupported requantization operation");
  }
  // Unified call
  it->second(src,
             srcOffset,
             dst,
             dstOffset,
             length,
             requantScale,
             requantOffset,
             srcEncoding.scale,
             srcEncoding.offset);
}

}  // namespace quantization
}  // namespace genie