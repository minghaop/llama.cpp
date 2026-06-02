//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <stdlib.h>

#include <memory>
#include <unordered_set>

#include "qualla/detail/buffer/Allocator/ClientAllocator.hpp"
#include "qualla/detail/buffer/Registration/IBufferRegs.hpp"

class ClientRegs final : public IBufferRegs {
 public:
  ClientRegs(std::shared_ptr<ClientAllocator> clientAllocator);
  // Disable copy constructors, r-value referencing, etc
  ClientRegs(const ClientRegs&)            = delete;
  ClientRegs& operator=(const ClientRegs&) = delete;
  ClientRegs(ClientRegs&&)                 = delete;
  ClientRegs& operator=(ClientRegs&&)      = delete;
  ~ClientRegs();
  bool initialize() override;

  void* getBuffer(Qnn_Tensor_t* tensor) override;
  int getFd(Qnn_Tensor_t* tensor) override;
  size_t getOffset(Qnn_Tensor_t* /*tensor*/) override { return 0; }
  size_t getBufferSize(Qnn_Tensor_t* tensor) override;
  size_t getTotalBufferSize(Qnn_Tensor_t* tensor) override { return getBufferSize(tensor); }

  bool allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) override;
  bool freeTensorBuffer(Qnn_Tensor_t* tensor) override;
  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) override;
  bool useExternalMemory(Qnn_Tensor_t* dest, void* extMem) override;
  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src, int /*offset*/) {
    return useSameMemory(dest, src);
  }

  bool deregisterTensorFusedBuffer(Qnn_Tensor_t* /*tensor*/) override { return false; }
  bool mapFusedTensorBuffer(Qnn_Tensor_t* /*tensor*/,
                            uint64_t /*allocIdx*/,
                            size_t /*offset*/,
                            Qnn_ContextHandle_t /*ctx*/,
                            size_t /*size*/) override;

  bool registerTensor(Qnn_Tensor_t* tensor, uint64_t allocIdx) override;
  bool deregisterTensor(Qnn_Tensor_t* tensor) override;
  bool mapTensorBuffer(Qnn_Tensor_t* tensor, uint64_t allocIdx, size_t tensorDatasize) override;

 private:
  std::shared_ptr<ClientAllocator> m_clientAllocator;                 // memory managing class.
  std::unordered_map<Qnn_Tensor_t*, uint64_t> m_tensorToAllocIdxMap;  // {  Tensor --> AllocIdx }
  std::unordered_map<uint64_t, std::unordered_set<Qnn_Tensor_t*>>
      m_allocIdxToTensorsMap;                            // { AllocIdx --> set<Qnn_Tensor_t*>}
  std::unordered_set<Qnn_Tensor_t*> m_extBufferTensors;  // external buffers where we need to skip
};
