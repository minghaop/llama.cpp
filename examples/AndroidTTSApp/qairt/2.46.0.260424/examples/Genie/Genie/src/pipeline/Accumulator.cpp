//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>

#include "Accumulator.hpp"
#include "QnnTypes.h"
#include "Quantization.hpp"

using namespace genie;

using QnnDataType = quantization::DataType;

Accumulator::Accumulator(size_t bufferSize, size_t embeddingDimension) {
  embeddingsBuffer.reserve(bufferSize);
  m_embeddingDimension = embeddingDimension;
  m_sequenceTracker    = std::make_shared<qualla::InputSequenceTracker>();
}

bool Accumulator::append(uint8_t* data, size_t dataSize) {
  embeddingsBuffer.insert(embeddingsBuffer.end(), data, data + dataSize);
  m_sequenceTracker->index += dataSize / m_embeddingDimension;
  return true;
}

bool Accumulator::append(void* src,
                         std::string srcDataType,
                         double srcScale,
                         int32_t srcOffset,
                         size_t numElements,
                         uint32_t tokenNum) {
  const size_t dstBufferOffset = embeddingsBuffer.size() / byteWidth;
  const size_t embeddingSize   = numElements * byteWidth;
  embeddingsBuffer.resize(embeddingsBuffer.size() + embeddingSize);
  void* dst = embeddingsBuffer.data();

  const quantization::Encoding srcEncoding{
      srcScale, srcOffset, quantization::toDataType(srcDataType)};
  const quantization::Encoding dstEncoding{scale, offset, quantization::toDataType(dataType)};
  quantization::requantize(src, 0, srcEncoding, dst, dstBufferOffset, dstEncoding, numElements);

  m_sequenceTracker->index += tokenNum;
  embeddingTokenNum += tokenNum;
  return true;
}

bool Accumulator::reset() {
  flush();
  m_sequenceTracker->index = 0;     // sync with _n_past = 0.
  return true;
}

bool Accumulator::flush() {
  embeddingsBuffer.clear();
  embeddingTokenNum = 0;
  visionParam.clear();
  return true;
}

void* Accumulator::getData() { return embeddingsBuffer.data(); }

size_t Accumulator::getDataSize() { return embeddingsBuffer.size(); }

void Accumulator::setEncoding(std::string dType,
                              double generatorScale,
                              int32_t generatorOffset,
                              float generatorByteWidth) {
  byteWidth = generatorByteWidth;
  offset    = generatorOffset;
  scale     = generatorScale;
  dataType  = dType;
}

void Accumulator::setVisionParam(uint32_t visionPos,
                                 uint32_t temporal,
                                 uint32_t height,
                                 uint32_t width) {
  visionParam.push_back(visionPos);
  visionParam.push_back(temporal);
  visionParam.push_back(height);
  visionParam.push_back(width);
}
