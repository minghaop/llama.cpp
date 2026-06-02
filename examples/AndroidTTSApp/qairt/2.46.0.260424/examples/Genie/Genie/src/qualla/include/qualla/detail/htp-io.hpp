//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "qualla/env.hpp"

namespace qualla {

enum ExternalBufferType { PERSISTENT, REGULAR, BUFFER_UNDEFINED };

enum DataFillPolicy { EXTERNAL, INTERNAL, POLICY_UNDEFINED };

enum DataType {
  CHAR,
  UINT_8,
  INT_8,
  UINT_16,
  INT_16,
  UINT_32,
  INT32,
  UINT64,
  INT64,
  FLOAT32,
  DOUBLE,
  TYPE_UNDEFINED
};

inline DataType getStringToDataType(const std::string& datatype) {
  static std::unordered_map<std::string, DataType> s_stringToDataTypeMap{
      {"char", DataType::CHAR},
      {"uint8_t", DataType::UINT_8},
      {"int8_t", DataType::INT_8},
      {"uint16_t", DataType::UINT_16},
      {"int16_t", DataType::INT_16},
      {"uint32_t", DataType::UINT_32},
      {"int32_t", DataType::INT32},
      {"uint64_t", DataType::UINT64},
      {"int64_t", DataType::INT64},
      {"float", DataType::FLOAT32},
      {"float32", DataType::FLOAT32},
      {"double", DataType::DOUBLE}};

  if (s_stringToDataTypeMap.count(datatype) == 0) return DataType::TYPE_UNDEFINED;
  return s_stringToDataTypeMap[datatype];
}

inline ExternalBufferType getExternalBufferTypeFromString(const std::string& externalBufferType) {
  static std::unordered_map<std::string, ExternalBufferType> s_stringToExternalBufferType{
      {"persistent", ExternalBufferType::PERSISTENT}, {"regular", ExternalBufferType::REGULAR}};

  if (s_stringToExternalBufferType.count(externalBufferType) == 0)
    return ExternalBufferType::BUFFER_UNDEFINED;
  return s_stringToExternalBufferType[externalBufferType];
}

class QuantParams {
 public:
  QuantParams(nlohmann::json& data) {
    using qc = qualla::Config;
    m_scale  = qc::optional<uint32_t>(data, "scale", 0);
    m_offset = qc::optional<double>(data, "offset", 0.0);
  }

 private:
  uint32_t m_scale;
  double m_offset;
};

class DataConfig {
 public:
  DataConfig(nlohmann::json& dataJson) {
    m_dataLength = 1;
    if (!dataJson.contains("dimensions"))
      throw std::runtime_error("dimensions not found in the data config.");
    for (auto& val : dataJson["dimensions"]) {
      m_shape.push_back(val.get<uint32_t>());
      m_dataLength *= m_shape.back();
    }
    if (dataJson.contains("data-type"))
      m_dataType = getStringToDataType(dataJson["data-type"].get<std::string>());
    if (m_dataType == DataType::TYPE_UNDEFINED) {
      throw std::runtime_error("Encountered undefined dataType during i/o data config.");
    }
    if (dataJson.contains("quant-params")) {
      m_quantParams = std::make_unique<QuantParams>(dataJson["quant-params"]);
    }
  }

 public:
  std::vector<uint32_t> m_shape;
  size_t m_dataLength{0};
  DataType m_dataType{DataType::TYPE_UNDEFINED};
  // Assume that data uses channel quantization
  std::unique_ptr<QuantParams> m_quantParams;
};

struct InferenceStep;

class HtpIO {
 public:
  HtpIO(ExternalBufferType bufferType = ExternalBufferType::REGULAR,
        DataFillPolicy fillPolicy     = DataFillPolicy::INTERNAL)
      : m_bufferType(bufferType), m_fillPolicy(fillPolicy) {
    m_initialized = true;
  }

  virtual void populateTensor(const InferenceStep& curStep) = 0;

  virtual ~HtpIO() = default;

 protected:
  // for persistent buffers only use the m_data, this doesn't build data copy, just holds the
  // pointer to user supplied data
  bool m_initialized              = false;
  ExternalBufferType m_bufferType = ExternalBufferType::BUFFER_UNDEFINED;
  DataFillPolicy m_fillPolicy     = DataFillPolicy::POLICY_UNDEFINED;
};
}  // namespace qualla
