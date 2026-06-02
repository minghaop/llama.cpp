//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "QnnTypes.h"

class IBufferRegs {
 public:
  virtual ~IBufferRegs() {}
  IBufferRegs() {}
  virtual bool initialize()                               = 0;
  virtual void* getBuffer(Qnn_Tensor_t* tensor)           = 0;
  virtual int getFd(Qnn_Tensor_t* tensor)                 = 0;
  virtual size_t getOffset(Qnn_Tensor_t* tensor)          = 0;
  virtual size_t getBufferSize(Qnn_Tensor_t* tensor)      = 0;
  virtual size_t getTotalBufferSize(Qnn_Tensor_t* tensor) = 0;

  virtual bool allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) = 0;
  virtual bool freeTensorBuffer(Qnn_Tensor_t* tensor)                            = 0;
  virtual bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src)              = 0;
  virtual bool useExternalMemory(Qnn_Tensor_t* dest, void* extMem)               = 0;

  virtual bool mapFusedTensorBuffer(Qnn_Tensor_t* tensor,
                                    uint64_t alloc_idx,
                                    size_t offset,
                                    Qnn_ContextHandle_t ctx,
                                    size_t tensorDatasize) = 0;

  virtual bool deregisterTensorFusedBuffer(Qnn_Tensor_t* tensor)                                = 0;
  virtual bool registerTensor(Qnn_Tensor_t* tensor, uint64_t allocIdx)                          = 0;
  virtual bool deregisterTensor(Qnn_Tensor_t* tensor)                                           = 0;
  virtual bool mapTensorBuffer(Qnn_Tensor_t* tensor, uint64_t alloc_idx, size_t tensorDatasize) = 0;

  // Functions to sync memory buffers for Read/Write using DmaBuf.
  virtual bool beforeWriteToBuffer(Qnn_Tensor_t* /*tensor*/) { return false; };
  virtual bool afterWriteToBuffer(Qnn_Tensor_t* /*tensor*/) { return false; };
  virtual bool beforeReadFromBuffer(Qnn_Tensor_t* /*tensor*/) { return false; };
  virtual bool afterReadFromBuffer(Qnn_Tensor_t* /*tensor*/) { return false; };
};
