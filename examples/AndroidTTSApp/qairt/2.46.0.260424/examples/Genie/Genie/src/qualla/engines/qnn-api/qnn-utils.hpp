//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#ifdef _MSC_VER
#pragma warning(disable : 4068)
#endif

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "QnnApiUtils.hpp"
#include "QnnInterface.h"
#include "qualla/detail/utils.hpp"
#include "qualla/env.hpp"

namespace qnn {
namespace tools {
namespace netrun {
enum class PerfProfile;
}
}  // namespace tools
}  // namespace qnn

namespace qualla {
namespace QnnUtils {
struct DataType {
  DataType() = default;
  DataType(const Qnn_Tensor_t* tensor) : _dtype(QNN_TENSOR_GET_DATA_TYPE(tensor)) {}
  DataType(Qnn_DataType_t dtype) : _dtype(dtype){};

  // Enable switch and comparisons
  constexpr operator Qnn_DataType_t() const { return _dtype; }

  uint32_t bw() const;
  int type() const;
  int32_t val() const;
  const char* str() const;

 private:
  Qnn_DataType_t _dtype{QNN_DATATYPE_UNDEFINED};
};

bool writeRawData(void* tensorData, size_t tensorSize, const std::filesystem::path& path);
bool readRawData(void* tensorData, size_t tensorSize, const std::filesystem::path& path);

struct Dims {
  uint32_t batch    = 1;
  uint32_t height   = 1;
  uint32_t width    = 1;
  uint32_t channel  = 1;
  uint32_t bitwidth = 0;

  Dims() = default;
  Dims(uint32_t _height, uint32_t _width, uint32_t _channel, uint32_t _bitwidth);
  Dims(const std::vector<uint32_t>& dims, uint32_t _bitwidth, uint32_t _batchSize = 1);

  Dims T() const;

  bool operator==(const Dims& rhs) const;
  bool operator!=(const Dims& rhs) const;
  uint32_t operator[](size_t idx) const;

  size_t getNumElements() const;
  size_t getSize() const;
  size_t getAlignedSize() const;
  uint32_t getMaxDim() const;
  std::vector<uint32_t> getVector() const;
};

struct QuantParam {
  double scale   = 1.0;
  int32_t offset = 0;

  QuantParam() = default;
  QuantParam(double _scale, int32_t _offset) : scale(_scale), offset(_offset) {}
};

struct Tensor {
  Qnn_Tensor_t* tensor = nullptr;
  std::string name;
  Dims dims;
  std::vector<QuantParam> quantParam;
  DataType dtype;

  Tensor() = default;
  Tensor(Qnn_Tensor_t* tensor, uint32_t batchSize = 1);
};

// Maps tensor name to QnnUtils::Tensor<Qnn_Tensor_t* tensor, dims, quantparams>
typedef std::map<std::string, Tensor> TensorMap;

static inline uint8_t sat_round(const uint16_t x) {
  const uint16_t rounded   = x + 0x80;              // add 0.5
  const uint16_t corrected = std::max(rounded, x);  // catch unsigned wrap around
  const uint16_t shifted   = corrected >> 8;        // divide by 256
  return static_cast<uint8_t>(shifted);             // to 8-bit
}

static inline void downcast_u16_to_u8(uint8_t* dest, const uint16_t* src, size_t nmemb) {
  for (size_t i = 0; i < nmemb; i++) dest[i] = sat_round(src[i]);
}

template <typename FloatType, typename IntType>
static inline IntType quantize(const FloatType val, int32_t offset, double scale) {
  return static_cast<IntType>(val / scale - offset);
}

template <typename FloatType, typename IntType>
static inline void quantizeTensorPtr(
    FloatType* tensor_float, IntType* tensor_quant, int32_t offset, double scale, size_t nmemb) {
  double quantizedValue;
  uint64_t offsetTemp = static_cast<uint64_t>(-offset);

  if constexpr (std::is_signed<IntType>::value) {
    // Use bit shifting instead of pow for better performance
    double maxQuantizedValue = (1LL << (sizeof(IntType) * 8 - 1)) - 1;
    double minQuantizedValue = -(1LL << (sizeof(IntType) * 8 - 1));

    PRAGMA_LOOP_VECTORIZE
    for (size_t i = 0; i < nmemb; ++i) {
      quantizedValue = (tensor_float[i] / scale) - offsetTemp;
      if (quantizedValue < minQuantizedValue)
        quantizedValue = minQuantizedValue;
      else if (quantizedValue > maxQuantizedValue)
        quantizedValue = maxQuantizedValue;
      tensor_quant[i] = static_cast<IntType>(quantizedValue);
    }
  } else {
    // Use bit shifting instead of pow for better performance
    double trueBitWidthMax = (1ULL << (sizeof(IntType) * 8)) - 1;
    double encodingMin     = (0 - static_cast<IntType>(offsetTemp)) * scale;
    double encodingMax     = (trueBitWidthMax - offsetTemp) * scale;
    double encodingRange   = encodingMax - encodingMin;

    PRAGMA_LOOP_VECTORIZE
    for (size_t i = 0; i < nmemb; ++i) {
      quantizedValue = (trueBitWidthMax * (tensor_float[i] - encodingMin) / encodingRange);
      if (quantizedValue < 0)
        quantizedValue = 0;
      else if (quantizedValue > trueBitWidthMax)
        quantizedValue = trueBitWidthMax;

      tensor_quant[i] = static_cast<IntType>(quantizedValue);
    }
  }
}

template <typename FloatType, typename IntType>
static inline void perWidthQuantizeTensorPtr(FloatType* tensor_float,
                                             IntType* tensor_quant,
                                             std::vector<QnnUtils::QuantParam>& quantParam,
                                             uint32_t height,
                                             uint32_t width,
                                             uint32_t channel) {
  for (uint32_t h = 0; h < height; h++) {
    for (uint32_t w = 0; w < width; w++) {
      double scale   = quantParam[w].scale;
      int32_t offset = quantParam[w].offset;

      PRAGMA_LOOP_VECTORIZE
      for (uint32_t c = 0; c < channel; c++) {
        uint32_t i      = (h * width * channel) + (w * channel) + c;
        double val      = tensor_float[i];
        tensor_quant[i] = static_cast<IntType>(val / scale - offset);
      }
    }
  }
}

// String parser that returns a list of upto N numbers from the string
template <size_t N>
std::array<uint16_t, N> parseNumberFromString(const std::string& name) {
  std::array<uint16_t, N> parsed_numbers = {0};  // Fixed-size array with template parameter

  size_t n_found  = 0;  // Count of numbers found
  bool in_number  = false;
  uint16_t number = 0;
  for (const char& ch : name) {
    if (ch >= '0' && ch <= '9') {
      in_number = true;
      number    = static_cast<uint16_t>(number * 10 + (ch - '0'));
    } else if (in_number) {
      parsed_numbers[n_found++] = number;
      in_number                 = false;
      number                    = 0;            // Reset number after pushing to the array
      if (n_found >= N) return parsed_numbers;  // Early exit if we've found N numbers
    }
  }

  // Add the last number if the string ends with a number
  if (in_number) parsed_numbers[n_found] = number;
  return parsed_numbers;
}

void getQuantParamString(const std::vector<QuantParam>& quantParam,
                         std::string& scale_string,
                         std::string& offset_string);

qnn::tools::netrun::PerfProfile quallaToQnnPerformanceProfile(
    qualla::PerformanceProfile perfProfile);

qualla::PerformanceProfile qnnToQuallaPerformanceProfile(
    qnn::tools::netrun::PerfProfile perfProfile);

inline uint32_t parseLayerIndex(const std::string& name) {
  const auto [layer_idx, head_idx] = parseNumberFromString<2>(name);
  return static_cast<uint32_t>(layer_idx) << 16 | static_cast<uint32_t>(head_idx);
}

// Utility function to replace the 'oldSub' substring in 'str' with 'newSub' substring
std::string replaceSubstring(const std::string& str,
                             const std::string& oldSub,
                             const std::string& newSub);

//  Utility function to match any prefix from a given set
bool matchPrefixAny(const std::string& s, const std::unordered_set<std::string>& prefixes);

//  Utility function to match any prefix from a given set and return prefix
std::string getPrefix(const std::string& s, const std::unordered_set<std::string>& prefixes);

// Utility function to identify key/value tensors
inline bool isKVTensor(const std::string& s) {
  // Check if tensor name contains *key* or *value* AND ends with _in or _out
  return (s.ends_with("_in") || s.ends_with("_out")) &&
         (s.find("key") != std::string::npos || s.find("value") != std::string::npos);
}

}  // namespace QnnUtils
}  // namespace qualla
