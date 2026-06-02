//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#ifndef ACCUMULATOR_HPP
#define ACCUMULATOR_HPP
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Exception.hpp"
#include "GenieCommon.h"
#include "InjectiveConnector.hpp"

namespace genie {

class Accumulator {
 public:
  Accumulator(size_t bufferSize = 0, size_t embeddingDimension = 0);
  bool reset();
  bool flush();
  bool append(uint8_t* data, size_t dataSize);
  bool append(void* src,
              std::string srcDataType,
              double srcScale,
              int32_t srcOffset,
              size_t numElements,
              uint32_t tokenNum);
  void* getData();
  size_t getDataSize();
  std::string& getDataType() { return dataType; }
  double& getScale() { return scale; }
  int32_t& getOffset() { return offset; }
  float getByteWidth() { return byteWidth; }

  uint32_t getTokenNum() { return embeddingTokenNum; }
  std::vector<uint32_t> getVisionParam() { return visionParam; }
  void setVisionParam(uint32_t visionPos, uint32_t temporal, uint32_t height, uint32_t width);

  void setEncoding(std::string dType,
                   double generatorScale,
                   int32_t generatorOffset,
                   float generatorByteWidth);

  std::shared_ptr<qualla::InputSequenceTracker> getSequenceTracker() { return m_sequenceTracker; }

  size_t m_embeddingDimension{0};
  std::string dataType{"QNN_DATATYPE_FLOAT_32"};
  double scale{1.0};
  int32_t offset{0};
  float byteWidth{4};
  std::vector<uint8_t> embeddingsBuffer;
  uint32_t embeddingTokenNum{0};
  std::shared_ptr<qualla::InputSequenceTracker> m_sequenceTracker;
  //[visionPos, temporal, height, width, ...]
  std::vector<uint32_t> visionParam;
};
}  // namespace genie
#endif  // ACCUMULATOR_HPP
