//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "QnnInterface.h"
#include "qualla/detail/buffer/Allocator/DmaAllocator.hpp"
#include "qualla/detail/buffer/Registration/IBufferRegs.hpp"

class DmaRegs final : public IBufferRegs {
 public:
  DmaRegs(Qnn_ContextHandle_t contextHandle,
          QNN_INTERFACE_VER_TYPE* qnnInterface,
          std::shared_ptr<DmaAllocator> dmaAllocator);
  // Disable copy constructors, r-value referencing, etc
  DmaRegs(const DmaRegs&)            = delete;
  DmaRegs& operator=(const DmaRegs&) = delete;
  DmaRegs(DmaRegs&&)                 = delete;
  DmaRegs& operator=(DmaRegs&&)      = delete;
  bool initialize() override;
  ~DmaRegs();

  void* getBuffer(Qnn_Tensor_t* tensor) override;
  int getFd(Qnn_Tensor_t* tensor) override;
  size_t getOffset(Qnn_Tensor_t* tensor) override;
  size_t getBufferSize(Qnn_Tensor_t* tensor) override;
  size_t getTotalBufferSize(Qnn_Tensor_t* tensor) override;

  // To allocate and free memory based on the tensor
  bool freeTensorBuffer(Qnn_Tensor_t* tensor) override;
  bool allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) override;
  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) override;
  bool useExternalMemory(Qnn_Tensor_t* /*dest*/, void* /*extMem*/) override;

  // To register and deregister fused buffers
  bool deregisterTensorFusedBuffer(Qnn_Tensor_t* /*tensor*/) override;
  bool mapFusedTensorBuffer(Qnn_Tensor_t* /*tensor*/,
                            uint64_t /*alloc_idx*/,
                            size_t /*offset*/,
                            Qnn_ContextHandle_t /*ctx*/,
                            size_t /*tensorDatasize*/) override;

  // To register and deregister fused buffers
  bool deregisterTensor(Qnn_Tensor_t* tensor) override;
  bool registerTensor(Qnn_Tensor_t* tensor, uint64_t allocIdx) override;
  bool mapTensorBuffer(Qnn_Tensor_t* tensor, uint64_t allocIdx, size_t tensorDatasize) override;

  bool beforeWriteToBuffer(Qnn_Tensor_t* tensor) override;
  bool afterWriteToBuffer(Qnn_Tensor_t* tensor) override;
  bool beforeReadFromBuffer(Qnn_Tensor_t* tensor) override;
  bool afterReadFromBuffer(Qnn_Tensor_t* tensor) override;

 private:
  QNN_INTERFACE_VER_TYPE* m_qnnInterface;
  Qnn_ContextHandle_t m_contextHandle;

  std::shared_ptr<DmaAllocator> m_dmaAllocator;
  std::unordered_map<Qnn_Tensor_t*, uint64_t> m_tensorToAllocIdxMap;  //  {  Tensor --> AllocIdx }
  std::unordered_map<uint64_t, std::unordered_set<Qnn_Tensor_t*>>
      m_allocIdxToTensorsMap;  // { AllocIdx --> set<Qnn_Tensor_t*>}
};
