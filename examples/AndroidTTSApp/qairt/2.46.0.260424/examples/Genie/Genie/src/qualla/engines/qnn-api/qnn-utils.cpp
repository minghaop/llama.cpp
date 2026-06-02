//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "QnnApi.hpp"
#include "QnnTypeUtils.hpp"
#include "qnn-utils.hpp"

namespace fs = std::filesystem;

namespace qualla {
namespace QnnUtils {
//-----------------------------------------------
// DataType
//-----------------------------------------------
uint32_t DataType::bw() const {
  // Alternate implementation for bw() = lambda x: (10 * ((x & 0xf0)>>4) + (x & 0xf)) // 8
  return aiswutility::getDataTypeContainerSize(_dtype);
}

int DataType::type() const { return _dtype >> 8; }

int32_t DataType::val() const { return static_cast<int32_t>(_dtype); }

const char* DataType::str() const { return aiswutility::dataTypeToString(_dtype); }

bool writeRawData(void* data, size_t size, const fs::path& path) {
  auto p = path.parent_path();
  if (!fs::exists(p) && !fs::create_directories(p)) return false;
  std::ofstream f(path, std::ofstream::binary);
  f.write(reinterpret_cast<char*>(data), static_cast<long>(size));
  f.close();
  return true;
}
bool readRawData(void* data, size_t size, const fs::path& path) {
  if (fs::file_size(path) != size) {
    throw std::runtime_error(fmt::format("file size doesnot match: {} size {}, buf-size {}",
                                         path.string(),
                                         fs::file_size(path),
                                         size));
  }
  std::ifstream f(path, std::ifstream::binary);
  f.read(reinterpret_cast<char*>(data), static_cast<long>(size));
  f.close();
  return true;
}

//-----------------------------------------------
// Dims
//-----------------------------------------------

Dims::Dims(uint32_t _height, uint32_t _width, uint32_t _channel, uint32_t _bitwidth)
    : height(_height), width(_width), channel(_channel), bitwidth(_bitwidth) {}

Dims::Dims(const std::vector<uint32_t>& dims, uint32_t _bitwidth, uint32_t _batchSize)
    : Dims(dims.at(1), dims.at(2), dims.at(3), _bitwidth) {
    if (_batchSize == 1) {
      // Hack to mix batch dimension
      height = (dims.at(0) != 1 && dims.at(1) == 1) ? dims.at(0) : height;
      batch  = (dims.at(0) > 1 && dims.at(1) != 1) ? dims.at(0) : batch;
    } else if (_batchSize > 1) {
      batch = dims.at(0);
      height = dims.at(1);
    }
}

Dims Dims::T() const { return Dims(width, height, channel, bitwidth); }

bool Dims::operator==(const Dims& rhs) const {
  return (height == rhs.height) && (width == rhs.width) && (channel == rhs.channel) &&
         (bitwidth == rhs.bitwidth);
}

bool Dims::operator!=(const Dims& rhs) const { return !(*this == rhs); }

uint32_t Dims::operator[](size_t idx) const {
  if (idx >= 4) {
    throw std::out_of_range("Dimensions index out-of-range");
  }
  return getVector()[idx];
}

size_t Dims::getNumElements() const { return height * width * channel; }

size_t Dims::getSize() const { return getNumElements() * batch * bitwidth; }

size_t Dims::getAlignedSize() const {
  size_t size = getSize();
  if ((size & uint64_t{7}) != uint64_t{0}) {
    size += (uint64_t{8} - (size & uint64_t{7}));
  }
  return size;
}

uint32_t Dims::getMaxDim() const { return std::max({height, width, channel}); }

std::vector<uint32_t> Dims::getVector() const { return {batch, height, width, channel}; }

//-----------------------------------------------
// Tensor
//-----------------------------------------------
Tensor::Tensor(Qnn_Tensor_t* _tensor, uint32_t batchSize)
    : tensor(_tensor),
      name(QNN_TENSOR_GET_NAME(_tensor)),
      dtype(QNN_TENSOR_GET_DATA_TYPE(_tensor)) {
  // Populate tensor dimensions
  const uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
  std::vector<uint32_t> tensorDims(std::max(rank, 4u), 1);
  std::copy_n(QNN_TENSOR_GET_DIMENSIONS(tensor), rank, tensorDims.end() - rank);

  dims = Dims(tensorDims, aiswutility::getDataTypeContainerSize(QNN_TENSOR_GET_DATA_TYPE(tensor)), batchSize);

  // Populate tensor quant params
  if (aiswutility::isQuantizedDataType(QNN_TENSOR_GET_DATA_TYPE(tensor))) {
    auto quantParams = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
    switch (quantParams.quantizationEncoding) {
      case Qnn_QuantizationEncoding_t::QNN_QUANTIZATION_ENCODING_SCALE_OFFSET: {
        float scale    = quantParams.scaleOffsetEncoding.scale;
        int32_t offset = quantParams.scaleOffsetEncoding.offset;
        quantParam.emplace_back(scale, offset);
        break;
      }
      case Qnn_QuantizationEncoding_t::QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
        auto encodingStruct = quantParams.axisScaleOffsetEncoding;
        for (uint32_t n = 0; n < encodingStruct.numScaleOffsets; n++) {
          auto scaleOffset = encodingStruct.scaleOffset[n];
          quantParam.emplace_back(scaleOffset.scale, scaleOffset.offset);
        }
        break;
      }
      default: {
        QNN_ERROR("Unsupported quantization encoding type!");
        quantParam.emplace_back(0, 0);
        break;
      }
    }
  } else {
    quantParam.emplace_back(0, 0);
  }
}

void getQuantParamString(const std::vector<QuantParam>& quantParam,
                         std::string& scale_string,
                         std::string& offset_string) {
  std::ostringstream scales_s;
  std::ostringstream offsets_s;
  for (size_t i = 0; i < quantParam.size(); i++) {
    if (i != 0) {
      scales_s << ", ";
      offsets_s << ", ";
    }
    scales_s << std::fixed << std::setprecision(20) << quantParam[i].scale;
    offsets_s << quantParam[i].offset;
  }
  scale_string  = scales_s.str();
  offset_string = offsets_s.str();
}

using QuallaPerfProfile = qualla::PerformanceProfile;
using QnnNetRunPerfProfile = qnn::tools::netrun::PerfProfile;

qnn::tools::netrun::PerfProfile quallaToQnnPerformanceProfile(QuallaPerfProfile perfProfile) {
  static const std::unordered_map<QuallaPerfProfile, QnnNetRunPerfProfile> quallaToQnnPerformanceProfile = {
    {QuallaPerfProfile::PERFORMANCE_BURST,                      QnnNetRunPerfProfile::BURST},
    {QuallaPerfProfile::PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE, QnnNetRunPerfProfile::SUSTAINED_HIGH_PERFORMANCE},
    {QuallaPerfProfile::PERFORMANCE_HIGH_PERFORMANCE,           QnnNetRunPerfProfile::HIGH_PERFORMANCE},
    {QuallaPerfProfile::PERFORMANCE_BALANCED,                   QnnNetRunPerfProfile::BALANCED},
    {QuallaPerfProfile::PERFORMANCE_LOW_BALANCED,               QnnNetRunPerfProfile::LOW_BALANCED},
    {QuallaPerfProfile::PERFORMANCE_HIGH_POWER_SAVER,           QnnNetRunPerfProfile::HIGH_POWER_SAVER},
    {QuallaPerfProfile::PERFORMANCE_POWER_SAVER,                QnnNetRunPerfProfile::POWER_SAVER},
    {QuallaPerfProfile::PERFORMANCE_LOW_POWER_SAVER,            QnnNetRunPerfProfile::LOW_POWER_SAVER},
    {QuallaPerfProfile::PERFORMANCE_EXTREME_POWER_SAVER,        QnnNetRunPerfProfile::EXTREME_POWER_SAVER}
  };

  // Translate qualla profile to QNN perf profiles
  auto itr = quallaToQnnPerformanceProfile.find(perfProfile);
  return (itr != quallaToQnnPerformanceProfile.end()) ? itr->second : QnnNetRunPerfProfile::BALANCED;
}

qualla::PerformanceProfile qnnToQuallaPerformanceProfile(QnnNetRunPerfProfile perfProfile) {
  static const std::unordered_map<QnnNetRunPerfProfile, QuallaPerfProfile> qnnToQuallaPerformanceProfile = {
    {QnnNetRunPerfProfile::BURST,                      QuallaPerfProfile::PERFORMANCE_BURST},
    {QnnNetRunPerfProfile::SUSTAINED_HIGH_PERFORMANCE, QuallaPerfProfile::PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE},
    {QnnNetRunPerfProfile::HIGH_PERFORMANCE,           QuallaPerfProfile::PERFORMANCE_HIGH_PERFORMANCE},
    {QnnNetRunPerfProfile::BALANCED,                   QuallaPerfProfile::PERFORMANCE_BALANCED},
    {QnnNetRunPerfProfile::LOW_BALANCED,               QuallaPerfProfile::PERFORMANCE_LOW_BALANCED},
    {QnnNetRunPerfProfile::HIGH_POWER_SAVER,           QuallaPerfProfile::PERFORMANCE_HIGH_POWER_SAVER},
    {QnnNetRunPerfProfile::POWER_SAVER,                QuallaPerfProfile::PERFORMANCE_POWER_SAVER},
    {QnnNetRunPerfProfile::LOW_POWER_SAVER,            QuallaPerfProfile::PERFORMANCE_LOW_POWER_SAVER},
    {QnnNetRunPerfProfile::EXTREME_POWER_SAVER,        QuallaPerfProfile::PERFORMANCE_EXTREME_POWER_SAVER}
  };

  // Translate QNN profile to qualla perf profiles
  auto itr = qnnToQuallaPerformanceProfile.find(perfProfile);
  return (itr != qnnToQuallaPerformanceProfile.end()) ? itr->second : QuallaPerfProfile::PERFORMANCE_BALANCED;
}

std::string replaceSubstring(const std::string& str,
                             const std::string& oldSub,
                             const std::string& newSub) {
  std::string modified = str;
  size_t pos           = modified.find(oldSub);
  if (pos != std::string::npos) modified.replace(pos, oldSub.length(), newSub);
  return modified;
};

//  Utility function to match any prefix from a given set
bool matchPrefixAny(const std::string& str, const std::unordered_set<std::string>& prefixes) {
  for (const std::string& prefix : prefixes) {
    if (str.starts_with(prefix)) {
      return true;
    }
  }
  return false;
}

std::string getPrefix(const std::string& s, const std::unordered_set<std::string>& prefixes) {
  for (const std::string& prefix : prefixes) {
    if (s.starts_with(prefix)) {
      return prefix;
    }
  }
  return "";
}

}  // namespace QnnUtils
}  // namespace qualla
