//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <functional>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace genie {
namespace quantization {


//For uint4 embedding requantization overloading
struct uint4_t {};

// Enum for supported data types
enum class DataType { UFixed4, SFixed8, SFixed16, UFixed8, UFixed16, Float16, Float32, Unknown };

DataType toDataType(const std::string& typeStr);

typedef struct {
  double scale;
  int32_t offset;
  DataType dataType;
} Encoding;

// Type pair struct for mapping
struct TypePair {
  DataType src;
  DataType dst;
  TypePair(DataType s, DataType d) : src(s), dst(d) {}
  bool operator<(const TypePair& other) const {
    return std::tie(src, dst) < std::tie(other.src, other.dst);
  }
};

// Unified conversion function signature
using ConvertFunc = std::function<void(
    const void*, size_t, void*, size_t, size_t, double, int32_t, double, int32_t)>;

void requantize(const void* src,
                const size_t srcOffset,
                const Encoding& srcEncoding,
                void* dst,
                size_t dstOffset,
                const Encoding& dstEncoding,
                const size_t length);

template <typename FromType, typename ToType>
void requantize(const FromType src,
                const Encoding& srcEncoding,
                ToType& dst,
                const Encoding& dstEncoding) {
  requantize(&src, 0, srcEncoding, &dst, 0, dstEncoding, 1);
}

}  // namespace quantization
}  // namespace genie
