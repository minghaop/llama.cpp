//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "QnnApiUtils.hpp"
#include "QnnCommon.h"
#include "QnnTypes.h"
#include "QnnWrapperUtils.hpp"
#include "qualla/IOBuffer.hpp"

namespace qualla {

class IOTensor : public IOBuffer {
 public:
  IOTensor(BufferType bufferAllocIn             = BufferType::DEFAULT,
           QNN_INTERFACE_VER_TYPE* qnnInterface = nullptr);

  virtual ~IOTensor();

  IOTensor(std::shared_ptr<IOTensor> ioTensor) : IOBuffer(*ioTensor.get()) {}

  bool setupInputTensors(Qnn_Tensor_t** inputs,
                         std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                         const qnn_wrapper_api::GraphInfo_t& graphInfo,
                         std::unordered_map<std::string, size_t>& inputTensorsSize,
                         Qnn_ContextHandle_t contextHandle,
                         bool skipBufferAllocation = false);

  bool setupOutputTensors(Qnn_Tensor_t** outputs,
                          std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                          const qnn_wrapper_api::GraphInfo_t& graphInfo,
                          std::unordered_map<std::string, size_t>& outputTensorsSize,
                          Qnn_ContextHandle_t contextHandle,
                          bool skipBufferAllocation = false);

  bool setupTensorWithSharedBuffers(
      Qnn_Tensor_t** outputs,
      std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
      uint32_t tensorCount,
      Qnn_Tensor_t* tensorWrappers,
      std::unordered_map<std::string, size_t>& tensorsSize,
      std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap);

  bool tearDownTensors(Qnn_Tensor_t* tensors, uint32_t tensorCount);
  bool tearDownTensors(std::vector<Qnn_Tensor_t*>& tensors, uint32_t tensorCount);
  bool tearDownTensors(std::vector<Qnn_Tensor_t>& tensors);
  bool tearDownTensors(std::unordered_map<std::string, Qnn_Tensor_t*>& tensors,
                       std::unordered_map<std::string, uint32_t>& tensorCountMap);
  bool tearDownTensors(std::vector<std::unordered_map<std::string, Qnn_Tensor_t*>>& tensors,
                       std::unordered_map<std::string, uint32_t>& tensorCountMap);
  bool tearDownTensors(const qnn_wrapper_api::GraphInfo_t* graphInfo);
  bool mapFusedBufferOffset(
      qnn_wrapper_api::GraphInfo_t* graphInfo,
      Qnn_ContextHandle_t contextHandle,
      const std::map<std::string, std::tuple<int, size_t, size_t>>& graphAllocs);
  bool mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                            uint64_t allocIdx,
                            size_t offset,
                            Qnn_ContextHandle_t ctx,
                            size_t tensorDatasize);

  void randomFn() override { return; }

 private:
  // There seems to be a race condition in mapFusedBufferOffset because we are
  // calling it from multiple threads. Maybe memRegister/memDeRegister is not thread-safe
  // Until I figure this out, adding a temporary lock here. TODO: Fix and remove this!
  std::mutex _tmp_lock;

  bool deepCopyQnnTensorInfo(Qnn_Tensor_t* dest, Qnn_Tensor_t* src);

  bool setupTensors(Qnn_Tensor_t** tensors,
                    std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                    uint32_t tensorCount,
                    Qnn_Tensor_t* tensorsInfo,
                    std::unordered_map<std::string, size_t>& tensorsSize,
                    Qnn_ContextHandle_t contextHandle,
                    bool skipBufferAllocation = false);
};

}  // namespace qualla
