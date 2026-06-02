//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "QnnTypes.h"

namespace aiswutility {

#define DATA_FMT_ALIGN(__sz, __align)                       \
  (((unsigned int)(__align) & ((unsigned int)(__align)-1))  \
       ? ((((__sz) + (__align)-1) / (__align)) * (__align)) \
       : (((__sz) + (__align)-1) & (~((unsigned int)(__align)-1))))

#if !defined(__arm__)
uint32_t calculateByteLength(const std::vector<uint32_t>& dims, const Qnn_DataType_t dataType);
uint32_t calculateByteLength(const std::vector<uint32_t>& dims,
                             const Qnn_DataType_t dataType,
                             bool& success);
#endif
size_t calculateByteLength(const std::vector<size_t>& dims, const Qnn_DataType_t dataType);
size_t calculateByteLength(const std::vector<size_t>& dims,
                           const Qnn_DataType_t dataType,
                           bool& success);

// There are some usages where half byte types aren't accepted. So there are two utils. One that
// accept half byte datatypes and one that doesn't
#if !defined(__arm__)
float calculateByteExactLength(const std::vector<uint32_t>& dims, const Qnn_DataType_t dataType);
uint32_t calculateElementCount(const std::vector<uint32_t>& dims);
#endif
float calculateByteExactLength(const std::vector<size_t>& dims, const Qnn_DataType_t dataType);
size_t calculateElementCount(const std::vector<size_t>& dims);

/**
 * @brief A function to compute the maximum amount of memory required to fully contain a tensor.
 *
 * @param[in] dims .
 *
 * @param[out] dataFormat .
 *
 * @return bufferSize
 *         - 0 if format is not valid
 *         - uint32_t value if format is valid
 */
uint32_t getBufferSize(const uint32_t* dims, const uint32_t dataFormat);

uint64_t queryTensorSize(Qnn_Tensor_t tensor);

template <class T>
std::vector<T> createNormalDistBuffer(uint32_t numValues,
                                      float mean   = 0.0f,
                                      float stddev = 1.0f,
                                      T min        = std::numeric_limits<T>::lowest(),
                                      T max        = std::numeric_limits<T>::max(),
                                      int seed     = 0) {
  std::vector<T> buffer(numValues, 0);

  if (seed == 0) {
    constexpr size_t magic = 0x9e3779b9;
    seed                   = std::hash<int>{}(__LINE__) + magic;
  }

  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(mean, stddev);

  // To ensure that the value is within 6 standard deviations of the mean, we clamp the min/max to 6
  // stddev.
  max = static_cast<T>(std::min(static_cast<float>(max), (mean + 6 * stddev)));
  min = static_cast<T>(std::max(static_cast<float>(min), (mean - 6 * stddev)));

  for (size_t i = 0; i < buffer.size(); i++) {
    float temp = dist(gen);
    buffer[i]  = std::min(std::max(static_cast<T>(temp), min), max);
  }

  return buffer;
}

template <class T>
std::vector<T> createUniformDistBuffer(uint32_t numValues,
                                       T min    = std::numeric_limits<T>::lowest(),
                                       T max    = std::numeric_limits<T>::max(),
                                       int seed = 0) {
  std::vector<T> buffer(numValues, 0);

  if (seed == 0) {
    constexpr size_t magic = 0x9e3779b9;
    seed                   = std::hash<int>{}(__LINE__) + magic;
  }

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(min, max);

  for (size_t i = 0; i < buffer.size(); i++) {
    buffer[i] = std::min(std::max(static_cast<T>(dist(gen)), min), max);
  }

  return buffer;
}

template <class T>
std::vector<T> createNormalDistBuffer(const std::vector<uint32_t>& dims,
                                      float mean   = 0.0f,
                                      float stddev = 1.0f,
                                      T min        = std::numeric_limits<T>::lowest(),
                                      T max        = std::numeric_limits<T>::max(),
                                      int seed     = 0) {
  const uint32_t numValues =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>());
  return createNormalDistBuffer<T>(numValues, mean, stddev, min, max, seed);
}

template <class T>
std::vector<T> createUniformDistBuffer(const std::vector<uint32_t>& dims,
                                       T min    = std::numeric_limits<T>::lowest(),
                                       T max    = std::numeric_limits<T>::max(),
                                       int seed = 0) {
  const uint32_t numValues =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>());
  return createUniformDistBuffer<T>(numValues, min, max, seed);
}

template <class T>
void createNormalDistBuffer(void* buffer,
                            uint32_t numValues,
                            float mean   = 0.0f,
                            float stddev = 1.0f,
                            T min        = std::numeric_limits<T>::lowest(),
                            T max        = std::numeric_limits<T>::max(),
                            int seed     = 0) {
  auto vec = createNormalDistBuffer<T>(numValues, mean, stddev, min, max, seed);
  for (uint32_t i = 0; i < numValues; ++i) {
    reinterpret_cast<T*>(buffer)[i] = vec[i];
  }
}

template <class T>
void createUniformDistBuffer(void* buffer,
                             uint32_t numValues,
                             T min    = std::numeric_limits<T>::lowest(),
                             T max    = std::numeric_limits<T>::max(),
                             int seed = 0) {
  auto vec = createUniformDistBuffer<T>(numValues, min, max, seed);
  for (uint32_t i = 0; i < numValues; ++i) {
    reinterpret_cast<T*>(buffer)[i] = vec[i];
  }
}

bool isDlcBuffer(const uint8_t* buffer, size_t size);

bool GetFileDataAsBuffer(const std::string& filePath, std::vector<std::uint8_t>& buffer);

}  // namespace aiswutility