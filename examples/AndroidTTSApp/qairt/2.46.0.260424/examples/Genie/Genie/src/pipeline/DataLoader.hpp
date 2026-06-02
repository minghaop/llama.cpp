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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Exception.hpp"
#include "GenieCommon.h"
#include "Node.hpp"
#include "qualla/detail/tensor.hpp"

namespace genie {
namespace pipeline {

class DataLoader {
 public:
  DataLoader();

  ~DataLoader();

  // Data collection methods - makes a copy of the data
  void addTensorData(const std::string& tensorName,
                     const void* data,
                     size_t dataSize,
                     const std::string& dataType,
                     double scale,
                     int32_t offset,
                     const std::vector<uint32_t>& dimensions);

  // Access training data
  qualla::TrainingData& getTrainingData();

  // Clear collected data
  void clearTrainingData();

  // Batch management
  void setBatchSize(size_t batchSize);
  void setMaxIterations(size_t maxIterations);

  // Get current indices for a specific tensor
  size_t getCurrentIteration(const std::string& tensorName) const;
  size_t getCurrentBatchIndex(const std::string& tensorName) const;

 private:
  // Convert raw data to qualla::Tensor with proper data type
  qualla::Tensor createTensor(const void* data,
                              size_t dataSize,
                              const std::string& dataType,
                              double scale,
                              int32_t offset,
                              const std::vector<uint32_t>& dimensions);

  // Map string data type to qualla::TensorDataType
  qualla::TensorDataType mapDataType(const std::string& dataType);

  // Calculate element size based on data type
  size_t getElementSize(qualla::TensorDataType dataType);

  // Per-tensor index tracking structure
  struct TensorIndices {
    size_t currentIteration  = 0;
    size_t currentBatchIndex = 0;
  };

  qualla::TrainingData m_trainingData;
  std::unordered_map<std::string, TensorIndices> m_tensorIndices;  // Per-tensor tracking
  size_t m_batchSize     = 1;
  size_t m_maxIterations = 1;
};

}  // namespace pipeline
}  // namespace genie
