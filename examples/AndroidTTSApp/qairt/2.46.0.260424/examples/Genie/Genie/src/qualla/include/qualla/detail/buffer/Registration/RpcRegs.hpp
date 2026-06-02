//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <unordered_set>

#include "QnnInterface.h"
#include "qualla/detail/buffer/Allocator/RpcAllocator.hpp"
#include "qualla/detail/buffer/Registration/IBufferRegs.hpp"

struct RpcBufferData {
  int fd;
  void* memPointer;
  size_t size;
  size_t totalBufferSize;
  size_t offset;
  uint64_t allocIdx;

  RpcBufferData() : fd(-1), memPointer(nullptr), size(0), allocIdx(0) {}
  RpcBufferData(int fdIn, void* memPointerIn, size_t sizeIn, uint64_t allocIdxIn)
      : fd(fdIn), memPointer(memPointerIn), size(sizeIn), allocIdx(allocIdxIn) {}
  RpcBufferData(int fdIn,
                void* memPointerIn,
                size_t sizeIn,
                size_t totalBufferSizeIn,
                size_t offsetIn,
                uint64_t allocIdxIn)
      : fd(fdIn),
        memPointer(memPointerIn),
        size(sizeIn),
        totalBufferSize(totalBufferSizeIn),
        offset(offsetIn),
        allocIdx(allocIdxIn) {}
};

class RpcRegs final : public IBufferRegs {
 public:
  RpcRegs(Qnn_ContextHandle_t contextHandle,
          QNN_INTERFACE_VER_TYPE* qnnInterface,
          std::shared_ptr<RpcAllocator> rpcAllocator);
  // Disable copy constructors, r-value referencing, etc
  RpcRegs(const RpcRegs&)            = delete;
  RpcRegs& operator=(const RpcRegs&) = delete;
  RpcRegs(RpcRegs&&)                 = delete;
  RpcRegs& operator=(RpcRegs&&)      = delete;
  ~RpcRegs();

  bool initialize() override;

  // To access internal memory per tensor
  void* getBuffer(Qnn_Tensor_t* tensor) override;
  int getFd(Qnn_Tensor_t* tensor) override;
  size_t getOffset(Qnn_Tensor_t* tensor) override;
  size_t getBufferSize(Qnn_Tensor_t* tensor) override;
  size_t getTotalBufferSize(Qnn_Tensor_t* tensor) override;

  // To allocate and free memory based on the tensor
  bool allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) override;
  bool freeTensorBuffer(Qnn_Tensor_t* tensor) override;
  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) override;
  bool useExternalMemory(Qnn_Tensor_t* dest, void* extMem) override;

  // To register and deregister fused buffers
  bool deregisterTensorFusedBuffer(Qnn_Tensor_t* tensor) override;
  bool registerFusedTensors(Qnn_Tensor_t* tensor,
                            RpcBufferData* rpcBufferData,
                            Qnn_ContextHandle_t contextHandle);
  bool mapFusedTensorBuffer(Qnn_Tensor_t* tensor,
                            uint64_t alloc_idx,
                            size_t offset,
                            Qnn_ContextHandle_t ctx,
                            size_t tensorDatasize) override;

  // To register and deregister buffers
  bool deregisterTensor(Qnn_Tensor_t* tensor) override;
  bool registerTensor(Qnn_Tensor_t* tensor, uint64_t allocIdx) override;
  bool mapTensorBuffer(Qnn_Tensor_t* tensor, uint64_t alloc_idx, size_t tensorDatasize) override;

 private:
  RpcBufferData* getRpcMemTensorData(Qnn_Tensor_t* tensor);
  uint64_t getAllocIdx(Qnn_Tensor_t* tensor);

  QNN_INTERFACE_VER_TYPE* m_qnnInterface;
  Qnn_ContextHandle_t m_contextHandle;

  // Helpful data structures.
  std::shared_ptr<RpcAllocator> m_rpcAllocator{nullptr};
  std::unordered_map<Qnn_Tensor_t*, uint64_t> m_tensorToAllocIdxMap;  //  {  Tensor --> AllocIdx }
  std::unordered_map<Qnn_MemHandle_t, RpcBufferData*>
      m_memHandleToRpcBufferData;  //  {  Tensor --> RpcBufferData }
  std::unordered_map<uint64_t, std::unordered_set<Qnn_Tensor_t*>>
      m_allocIdxToTensorsMap;  // { AllocIdx --> set<Qnn_Tensor_t*>}
  std::map<std::tuple<int, size_t, Qnn_ContextHandle_t>, Qnn_Tensor_t*>
      m_memConfigList;  // { <fd, offset, contextHandle> --> Qnn_tensor*}
};
