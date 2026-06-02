//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cstring>

#include "DataLoader.hpp"
#include "Exception.hpp"

using namespace genie;

pipeline::DataLoader::DataLoader() {}

pipeline::DataLoader::~DataLoader() { clearTrainingData(); }

void pipeline::DataLoader::addTensorData(const std::string& tensorName,
                                         const void* data,
                                         size_t dataSize,
                                         const std::string& dataType,
                                         double scale,
                                         int32_t offset,
                                         const std::vector<uint32_t>& dimensions) {
  if (data == nullptr || dataSize == 0) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL,
                    "DataLoader::addTensorData received null or empty data");
  }

  // Create tensor from raw data (makes a copy)
  qualla::Tensor tensor = createTensor(data, dataSize, dataType, scale, offset, dimensions);

  // Initialize per-tensor indices if this is a new tensor
  if (m_tensorIndices.find(tensorName) == m_tensorIndices.end()) {
    m_tensorIndices[tensorName] = TensorIndices();
  }

  // Initialize structure if needed - use dynamic sizing
  if (m_trainingData.find(tensorName) == m_trainingData.end()) {
    m_trainingData[tensorName] = std::vector<std::vector<qualla::Tensor>>();
    m_trainingData[tensorName].reserve(m_maxIterations);
  }

  // Get current indices for this tensor
  TensorIndices& indices = m_tensorIndices[tensorName];

  // Ensure we have enough iterations allocated
  if (indices.currentIteration >= m_trainingData[tensorName].size()) {
    m_trainingData[tensorName].resize(indices.currentIteration + 1);
    m_trainingData[tensorName][indices.currentIteration].reserve(m_batchSize);
  }

  // Add tensor to current batch for this tensor
  m_trainingData[tensorName][indices.currentIteration].push_back(tensor);

  // Advance batch index for this tensor
  indices.currentBatchIndex++;

  // If we've filled a batch, move to next iteration
  if (indices.currentBatchIndex >= m_batchSize) {
    indices.currentBatchIndex = 0;
    indices.currentIteration++;
  }
}

qualla::Tensor pipeline::DataLoader::createTensor(const void* data,
                                                  size_t dataSize,
                                                  const std::string& dataType,
                                                  double scale,
                                                  int32_t offset,
                                                  const std::vector<uint32_t>& dimensions) {
  // Map data type
  qualla::TensorDataType tensorDataType = mapDataType(dataType);
  size_t elementSize                    = getElementSize(tensorDataType);
  size_t numElements                    = dataSize / elementSize;

  // Create quantization parameters
  qualla::TensorQuantizationParams qparams = {scale, offset};
  // user_owned = false means the Tensor will manage the memory
  qualla::Tensor tensor(
      const_cast<void*>(data), tensorDataType, qparams, numElements, dimensions, false);

  return tensor;
}

qualla::TensorDataType pipeline::DataLoader::mapDataType(const std::string& dataType) {
  if (dataType == "QNN_DATATYPE_FLOAT_32" || dataType == "float32") {
    return qualla::TENSOR_DATATYPE_FLOAT_32;
  } else if (dataType == "QNN_DATATYPE_FLOAT_16" || dataType == "float16") {
    return qualla::TENSOR_DATATYPE_FLOAT_POINT_16;
  } else if (dataType == "QNN_DATATYPE_UFIXED_POINT_8" || dataType == "uint8") {
    return qualla::TENSOR_DATATYPE_UFIXED_POINT_8;
  } else if (dataType == "QNN_DATATYPE_UFIXED_POINT_16" || dataType == "uint16") {
    return qualla::TENSOR_DATATYPE_UFIXED_POINT_16;
  } else if (dataType == "QNN_DATATYPE_INT_32" || dataType == "int32") {
    return qualla::TENSOR_DATATYPE_INT_32;
  } else if (dataType == "QNN_DATATYPE_INT_64" || dataType == "int64") {
    return qualla::TENSOR_DATATYPE_INT_64;
  } else {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Unsupported data type: " + dataType);
  }
}

size_t pipeline::DataLoader::getElementSize(qualla::TensorDataType dataType) {
  switch (dataType) {
    case qualla::TENSOR_DATATYPE_UFIXED_POINT_8:
      return 1;
    case qualla::TENSOR_DATATYPE_FLOAT_POINT_16:
    case qualla::TENSOR_DATATYPE_UFIXED_POINT_16:
      return 2;
    case qualla::TENSOR_DATATYPE_INT_64:
      return 8;
    case qualla::TENSOR_DATATYPE_FLOAT_32:
    case qualla::TENSOR_DATATYPE_INT_32:
    default:
      return 4;
  }
}

qualla::TrainingData& pipeline::DataLoader::getTrainingData() {
  return m_trainingData;
}

void pipeline::DataLoader::clearTrainingData() {
  m_trainingData.clear();
  m_tensorIndices.clear();  // Clear per-tensor indices
}

void pipeline::DataLoader::setBatchSize(size_t batchSize) { m_batchSize = batchSize; }

void pipeline::DataLoader::setMaxIterations(size_t maxIterations) {
  m_maxIterations = maxIterations;
  // Resize existing data structures if needed
  for (auto& [name, iterations] : m_trainingData) {
    iterations.resize(maxIterations);
  }
}

size_t pipeline::DataLoader::getCurrentIteration(const std::string& tensorName) const {
  auto it = m_tensorIndices.find(tensorName);
  if (it != m_tensorIndices.end()) {
    return it->second.currentIteration;
  }
  return 0;
}

size_t pipeline::DataLoader::getCurrentBatchIndex(const std::string& tensorName) const {
  auto it = m_tensorIndices.find(tensorName);
  if (it != m_tensorIndices.end()) {
    return it->second.currentBatchIndex;
  }
  return 0;
}
