//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cstring>
#include <fstream>
#include <iostream>

#include "IOTensor.hpp"
#include "QnnTypeMacros.hpp"
#include "qualla/detail/Log.hpp"

#ifdef _WIN32
#define __strdup _strdup
#else
#define __strdup strdup
#endif

namespace qualla {
// Classs Invocations
IOTensor::IOTensor(BufferType bufferAllocIn, QNN_INTERFACE_VER_TYPE* qnnInterface)
    : IOBuffer(bufferAllocIn, qnnInterface) {}

IOTensor::~IOTensor() {}

// Allocates and Registers.
// Setup details for Qnn_Tensor_t for execution
// based on information in TensorWrapper provided by model.so.
bool IOTensor::setupTensors(Qnn_Tensor_t** tensors,
                            std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                            uint32_t tensorCount,
                            Qnn_Tensor_t* tensorWrappers,
                            std::unordered_map<std::string, size_t>& tensorsSize,
                            Qnn_ContextHandle_t contextHandle,
                            bool skipBufferAllocation) {
  if (nullptr == tensorWrappers) {
    QNN_ERROR("tensorWrappers is nullptr");
    return false;
  }
  if (0 == tensorCount) {
    QNN_DEBUG("tensor count is 0. Nothing to setup.");
    return true;
  }

  *tensors = reinterpret_cast<Qnn_Tensor_t*>(calloc(1, tensorCount * sizeof(Qnn_Tensor_t)));
  if (nullptr == *tensors) {
    QNN_ERROR("mem alloc failed for *tensors");
    return false;
  }

  auto returnStatus = true;

  size_t totalBufferSize = 0;
  uint64_t allocIdx      = 0;
  if (m_bufferType == BufferType::SHARED_BUFFER) {
    // Calculate the total size of the tensors
    for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
      auto wrapperTensorName = std::string(QNN_TENSOR_GET_NAME(tensorWrappers[tensorIdx]));
      totalBufferSize += tensorsSize[wrapperTensorName];
    }
    QNN_DEBUG("Calculated total size %zu", totalBufferSize);

    if (!skipBufferAllocation) {
      void* memPointer = nullptr;
      int32_t fd       = -1;

      // Allocate the buffer of this size
      allocIdx   = m_allocator->allocate(totalBufferSize);
      fd         = m_allocator->getFd(allocIdx);
      memPointer = m_allocator->getBuffer(allocIdx);
      if (memPointer) {
        QNN_DEBUG("Successfully allocated a buffer of size %zu, pointer %p, fd %d",
                  totalBufferSize,
                  memPointer,
                  fd);
      } else {
        QNN_ERROR("Not able to allocate buffer of size %zu, fd %d", totalBufferSize, fd);
        return false;
      }
    }
  }

  uint64_t offset = 0;

  for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
    Qnn_Tensor_t wrapperTensor = tensorWrappers[tensorIdx];
    auto wrapperTensorName     = std::string(QNN_TENSOR_GET_NAME(tensorWrappers[tensorIdx]));
    if (true == returnStatus) {
      (*tensors)[tensorIdx] = QNN_TENSOR_INIT;
      returnStatus          = deepCopyQnnTensorInfo(((*tensors) + tensorIdx), &wrapperTensor);
    }
    if (true == returnStatus) {
      size_t tensorDataSize = tensorsSize[wrapperTensorName];
      if (m_bufferType == BufferType::SHARED_BUFFER) {
        if (!skipBufferAllocation) {
          returnStatus = m_register->mapFusedTensorBuffer(
              ((*tensors) + tensorIdx), allocIdx, offset, contextHandle, tensorDataSize);
          offset += tensorDataSize;
        }
      } else {
        if (!skipBufferAllocation) {
          returnStatus = m_register->allocateTensorBuffer(((*tensors) + tensorIdx), tensorDataSize);
        }
      }
    }
    if (true != returnStatus) {
      QNN_ERROR("Failure in setupTensors, cleaning up resources");
      tearDownTensors(*tensors, tensorIdx);
      *tensors = nullptr;
      QNN_ERROR("Failure in setupTensors, done cleaning up resources");
      return false;
    } else {
      tensorNameToTensorPointer.insert({wrapperTensorName, ((*tensors) + tensorIdx)});
      // QNN_DEBUG("allocateBuffer successful");
    }
  }

  return returnStatus;
}

// Setup details for all input tensors for graph execution.
bool IOTensor::setupInputTensors(Qnn_Tensor_t** inputs,
                                 std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                                 const qnn_wrapper_api::GraphInfo_t& graphInfo,
                                 std::unordered_map<std::string, size_t>& inputTensorsSize,
                                 Qnn_ContextHandle_t contextHandle,
                                 bool skipBufferAllocation) {
  if (true != setupTensors(inputs,
                           tensorNameToTensorPointer,
                           graphInfo.numInputTensors,
                           (graphInfo.inputTensors),
                           inputTensorsSize,
                           contextHandle,
                           skipBufferAllocation)) {
    QNN_ERROR("Failure in setupInputTensors, cleaning up resources");
    if (nullptr != *inputs) {
      QNN_DEBUG("cleaning up input tensors");
      tearDownTensors(*inputs, graphInfo.numInputTensors);
      *inputs = nullptr;
    }
    QNN_ERROR("Failure in setupInputTensors, done cleaning up resources");

    return false;
  }

  return true;
}

// Setup details for all output tensors for graph execution.
bool IOTensor::setupOutputTensors(Qnn_Tensor_t** outputs,
                                  std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                                  const qnn_wrapper_api::GraphInfo_t& graphInfo,
                                  std::unordered_map<std::string, size_t>& outputTensorsSize,
                                  Qnn_ContextHandle_t contextHandle,
                                  bool skipBufferAllocation) {
  if (true != setupTensors(outputs,
                           tensorNameToTensorPointer,
                           graphInfo.numOutputTensors,
                           (graphInfo.outputTensors),
                           outputTensorsSize,
                           contextHandle,
                           skipBufferAllocation)) {
    QNN_ERROR("Failure in setupOutputTensors, cleaning up resources");
    if (nullptr != *outputs) {
      QNN_DEBUG("cleaning up output tensors");
      tearDownTensors(*outputs, graphInfo.numOutputTensors);
      *outputs = nullptr;
    }
    QNN_ERROR("Failure in setupOutputTensors, done cleaning up resources");

    return false;
  }

  return true;
}

// Setup details for Qnn_Tensor_t for execution.
// Reuse same memory handle for KV input and output tensor.
bool IOTensor::setupTensorWithSharedBuffers(
    Qnn_Tensor_t** tensors,
    std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
    uint32_t tensorCount,
    Qnn_Tensor_t* tensorWrappers,
    std::unordered_map<std::string, size_t>& tensorsSize,
    std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap) {
  if (nullptr == tensorWrappers) {
    QNN_ERROR("tensorWrappers is nullptr");
    return false;
  }

  if (0 == tensorCount) {
    QNN_DEBUG("tensor count is 0. Nothing to setup.");
    return true;
  }

  *tensors = reinterpret_cast<Qnn_Tensor_t*>(calloc(1, tensorCount * sizeof(Qnn_Tensor_t)));
  if (nullptr == *tensors) {
    QNN_ERROR("mem alloc failed for *tensors");
    return false;
  }

  bool returnStatus = true;
  for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
    Qnn_Tensor_t wrapperTensor = tensorWrappers[tensorIdx];
    auto wrapperTensorName     = std::string(QNN_TENSOR_GET_NAME(tensorWrappers[tensorIdx]));
    if (true == returnStatus) {
      (*tensors)[tensorIdx] = QNN_TENSOR_INIT;
      returnStatus          = deepCopyQnnTensorInfo(((*tensors) + tensorIdx), &wrapperTensor);
    }
    if (true == returnStatus) {
      if (sharedTensorMap.find(wrapperTensorName) == sharedTensorMap.end()) {
        QNN_DEBUG("IoTensor :: Create Buffer for Tensor %s", wrapperTensorName.c_str());
        size_t tensorDataSize = tensorsSize[wrapperTensorName];
        returnStatus = m_register->allocateTensorBuffer(((*tensors) + tensorIdx), tensorDataSize);
      } else {
        std::string inputName = QNN_TENSOR_GET_NAME(sharedTensorMap[wrapperTensorName]);
        QNN_DEBUG("IoTensor :: Reuse Buffer for Tensor %s", wrapperTensorName.c_str());
        returnStatus =
            m_register->useSameMemory(((*tensors) + tensorIdx), sharedTensorMap[wrapperTensorName]);
      }
    }
    if (true != returnStatus) {
      QNN_ERROR("Failure in setupTensors, cleaning up resources");
      tearDownTensors(*tensors, static_cast<uint32_t>(tensorIdx));
      *tensors = nullptr;
      QNN_ERROR("Failure in setupTensors, done cleaning up resources");
      break;
    } else {
      tensorNameToTensorPointer.insert({wrapperTensorName, ((*tensors) + tensorIdx)});
    }
  }
  return returnStatus;
}

bool IOTensor::mapFusedBufferOffset(
    qnn_wrapper_api::GraphInfo_t* graphInfo,
    Qnn_ContextHandle_t contextHandle,
    const std::map<std::string, std::tuple<int, size_t, size_t>>& graphAllocs) {
  std::lock_guard lk(_tmp_lock);  // READ COMMENT IN IOTensor.hpp _tmp_lock

  bool ret = true;
  for (bool mode : {true, false}) {
    Qnn_Tensor_t* tensorBank = (mode) ? graphInfo->inputTensors : graphInfo->outputTensors;
    uint32_t numTensors      = (mode) ? graphInfo->numInputTensors : graphInfo->numOutputTensors;

    for (size_t tidx = 0; tidx < numTensors; tidx++) {
      Qnn_Tensor_t& tensorWrapper = tensorBank[tidx];
      Qnn_Tensor_t* tensor        = &tensorWrapper;
      std::string tensorName      = std::string(QNN_TENSOR_GET_NAME(tensorWrapper));

      if (!graphAllocs.contains(tensorName)) continue;
      auto& [allocIdx, offset, size] = graphAllocs.at(tensorName);
      ret &= m_register->mapFusedTensorBuffer(
          tensor, static_cast<size_t>(allocIdx), offset, contextHandle, size);
    }
  }

  return ret;
}

bool IOTensor::mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                                    uint64_t allocIdx,
                                    size_t offset,
                                    Qnn_ContextHandle_t ctx,
                                    size_t tensorDatasize) {
  return m_register->mapFusedTensorBuffer(tensor, allocIdx, offset, ctx, tensorDatasize);
}

// Clean up all tensors related data after execution.
bool IOTensor::tearDownTensors(Qnn_Tensor_t* tensors, uint32_t tensorCount) {
  if (nullptr != tensors) {
    QNN_DEBUG("cleaning up resources for tensors");
    for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
      Qnn_Tensor_t* tensor = &tensors[tensorIdx];

      // Free tensor name
      const char* tensorName = QNN_TENSOR_GET_NAME(tensor);
      if (nullptr != tensorName) {
        free(static_cast<void*>(const_cast<char*>(tensorName)));
        QNN_TENSOR_SET_NAME(tensor, nullptr);
      }

      // Free tensor dimensions
      uint32_t* tensorDims = QNN_TENSOR_GET_DIMENSIONS(tensor);
      if (nullptr != tensorDims) {
        free(tensorDims);
        QNN_TENSOR_SET_DIMENSIONS(tensor, nullptr);
      }

      // Free tensor quantization encodings
      Qnn_QuantizeParams_t qParams = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
      if (qParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        if (qParams.axisScaleOffsetEncoding.scaleOffset != nullptr) {
          free(qParams.axisScaleOffsetEncoding.scaleOffset);
          qParams.axisScaleOffsetEncoding.scaleOffset = nullptr;
          QNN_TENSOR_SET_QUANT_PARAMS(tensor, qParams);
        }
      }

      if (m_bufferType == BufferType::SHARED_BUFFER) {
        m_register->deregisterTensorFusedBuffer(tensor);
      } else {
        m_register->freeTensorBuffer(tensor);
      }
    }
    free(tensors);
    tensors = nullptr;
  }
  return true;
}

// Clean up all tensors after execution.
bool IOTensor::tearDownTensors(std::vector<Qnn_Tensor_t*>& tensors, uint32_t numTensors) {
  for (Qnn_Tensor_t* tensor : tensors) {
    tearDownTensors(tensor, numTensors);
  }

  return true;
}

bool IOTensor::tearDownTensors(std::vector<Qnn_Tensor_t>& tensors) {
  return tearDownTensors(tensors.data(), static_cast<uint32_t>(tensors.size()));
}

// Clean up all tensors after execution.
bool IOTensor::tearDownTensors(std::unordered_map<std::string, Qnn_Tensor_t*>& tensors,
                               std::unordered_map<std::string, uint32_t>& tensorCountMap) {
  for (auto& tensor : tensors) {
    tearDownTensors(tensor.second, tensorCountMap[tensor.first]);
  }

  return true;
}

// Clean up all tensors after execution.
bool IOTensor::tearDownTensors(std::vector<std::unordered_map<std::string, Qnn_Tensor_t*>>& tensors,
                               std::unordered_map<std::string, uint32_t>& tensorCountMap) {
  for (auto& tensor : tensors) {
    tearDownTensors(tensor, tensorCountMap);
  }

  return true;
}

bool IOTensor::tearDownTensors(const qnn_wrapper_api::GraphInfo_t* graphInfo) {
  bool status = true;
  if (!tearDownTensors(graphInfo->inputTensors, graphInfo->numInputTensors)) {
    status = false;
    QNN_ERROR("Failed to tear down input tensors for graph %s", graphInfo->graphName);
  }

  if (!tearDownTensors(graphInfo->outputTensors, graphInfo->numOutputTensors)) {
    status = false;
    QNN_ERROR("Failed to tear down output tensors for graph %s", graphInfo->graphName);
  }
  return status;
}

bool IOTensor::deepCopyQnnTensorInfo(Qnn_Tensor_t* dst, Qnn_Tensor_t* src) {
  if (nullptr == dst || nullptr == src) {
    QNN_ERROR("Received nullptr for one of the tensor: src=%p dst=%p", src, dst);
    return false;
  }

  // Set tensor.version before using QNN_TENSOR_SET macros, as they require the version to be set
  // to correctly assign values
  dst->version = src->version;

  // Deepcopy tensor primitives
  const char* tensorName = QNN_TENSOR_GET_NAME(src);
  QNN_TENSOR_SET_NAME(dst, tensorName ? __strdup(tensorName) : nullptr);
  QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));

  // Deepcopy tensor dimensions
  QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
  QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);
  if (QNN_TENSOR_GET_RANK(src) > 0) {
    const size_t tensorDimsSize = QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t);
    auto tensorDims = reinterpret_cast<uint32_t*>(malloc(tensorDimsSize));
    if (tensorDims) {
      QNN_TENSOR_SET_DIMENSIONS(dst, tensorDims);
      memcpy(tensorDims, QNN_TENSOR_GET_DIMENSIONS(src), tensorDimsSize);
    }
    else {
      QNN_ERROR("Failed to allocate memory for tensor dimensions: name=%s",
                tensorName ? tensorName : "???");
      return false;
    }
  }

  // Deepcopy tensor quantization params
  const Qnn_QuantizeParams_t qParamsSrc = QNN_TENSOR_GET_QUANT_PARAMS(src);
  Qnn_QuantizeParams_t& qParamsDst = dst->v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  qParamsDst.encodingDefinition    = qParamsSrc.encodingDefinition;
  switch(qParamsSrc.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET: {
      qParamsDst.quantizationEncoding = qParamsSrc.quantizationEncoding;
      qParamsDst.scaleOffsetEncoding  = qParamsSrc.scaleOffsetEncoding;
      break;
    }
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
      qParamsDst.quantizationEncoding = qParamsSrc.quantizationEncoding;
      qParamsDst.axisScaleOffsetEncoding.axis = qParamsSrc.axisScaleOffsetEncoding.axis;
      const size_t numScaleOffsets = qParamsDst.axisScaleOffsetEncoding.numScaleOffsets
                                   = qParamsSrc.axisScaleOffsetEncoding.numScaleOffsets;
      if (numScaleOffsets > 0) {
        qParamsDst.axisScaleOffsetEncoding.scaleOffset =
            reinterpret_cast<Qnn_ScaleOffset_t*>(malloc(numScaleOffsets * sizeof(Qnn_ScaleOffset_t)));
        if (qParamsDst.axisScaleOffsetEncoding.scaleOffset) {
          for (size_t idx = 0; idx < numScaleOffsets; idx++) {
            qParamsDst.axisScaleOffsetEncoding.scaleOffset[idx].scale =
                qParamsSrc.axisScaleOffsetEncoding.scaleOffset[idx].scale;
            qParamsDst.axisScaleOffsetEncoding.scaleOffset[idx].offset =
                qParamsSrc.axisScaleOffsetEncoding.scaleOffset[idx].offset;
          }
        }
        else {
          QNN_ERROR("Failed to allocate memory for tensor quantization encodings: name=%s",
                    tensorName ? tensorName : "???");
          return false;
        }
      }
      break;
    }
    default:
      qParamsDst.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
      break;
  }

  return true;
}

};  // namespace qualla