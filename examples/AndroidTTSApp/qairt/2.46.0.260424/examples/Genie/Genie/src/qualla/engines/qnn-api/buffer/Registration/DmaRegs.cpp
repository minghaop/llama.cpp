//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <dlfcn.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>

#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>

#include "QnnMem.h"
#include "QnnTypeMacros.hpp"
#include "qualla/detail/Log.hpp"
#include "qualla/detail/buffer/Registration/DmaRegs.hpp"

DmaRegs::DmaRegs(Qnn_ContextHandle_t contextHandle,
                 QNN_INTERFACE_VER_TYPE* qnnInterface,
                 std::shared_ptr<DmaAllocator> dmaAllocator)
    : m_qnnInterface(qnnInterface), m_contextHandle(contextHandle), m_dmaAllocator(dmaAllocator) {}

bool DmaRegs::initialize() { return m_dmaAllocator->initialize(); }

DmaRegs::~DmaRegs() {
  for (auto it = m_tensorToAllocIdxMap.begin(); it != m_tensorToAllocIdxMap.end();) {
    auto nxt = std::next(it);
    if (true != deregisterTensor(it->first)) {
      QNN_ERROR("Failed to deregister tensor.");
    }
    it = nxt;
  }
  m_tensorToAllocIdxMap.clear();
  m_allocIdxToTensorsMap.clear();
}

void* DmaRegs::getBuffer(Qnn_Tensor_t* tensor) {
  if (tensor == nullptr) {
    QNN_ERROR("getBuffer : Couldn't find tensor %p", tensor);
    return nullptr;
  }
  uint64_t allocIdx = m_tensorToAllocIdxMap[tensor];
  return m_dmaAllocator->getBuffer(allocIdx);
}

int DmaRegs::getFd(Qnn_Tensor_t* tensor) {
  if (tensor == nullptr) {
    QNN_ERROR("getFd : Couldn't find tensor %p", tensor);
    return -1;
  }
  uint64_t allocIdx = m_tensorToAllocIdxMap[tensor];
  return m_dmaAllocator->getFd(allocIdx);
}

size_t DmaRegs::getOffset(Qnn_Tensor_t* tensor) {
  if (tensor == nullptr) {
    QNN_ERROR("getOffset : Couldn't find tensor %p", tensor);
    return 0;
  }
  return 0;
}

size_t DmaRegs::getBufferSize(Qnn_Tensor_t* tensor) {
  if (tensor == nullptr) {
    QNN_ERROR("getBufferSize : Couldn't find tensor %p", tensor);
    return 0;
  }
  uint64_t allocIdx = m_tensorToAllocIdxMap[tensor];
  return m_dmaAllocator->getBufferSize(allocIdx);
};

size_t DmaRegs::getTotalBufferSize(Qnn_Tensor_t* tensor) {
  if (tensor == nullptr) {
    QNN_ERROR("getTotalBufferSize : Couldn't find tensor %p", tensor);
    return 0;
  }
  uint64_t allocIdx = m_tensorToAllocIdxMap[tensor];
  return m_dmaAllocator->getTotalBufferSize(allocIdx);
}

bool DmaRegs::registerTensor(Qnn_Tensor_t* tensor, uint64_t allocIdx) {
  if (!tensor) {
    QNN_ERROR("DmaRegs: Received nullptr for tensor");
    return false;
  }
  if (allocIdx == UINT64_MAX) {
    QNN_ERROR("DmaRegs: Received invalid allocation Id.");
    return false;
  }
  Qnn_MemDescriptor_t memDescriptor = {
      {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
      QNN_TENSOR_GET_DATA_TYPE(tensor),
      QNN_MEM_TYPE_DMA_BUF,
      {.dmaBufInfo = {m_dmaAllocator->getFd(allocIdx), m_dmaAllocator->getBuffer(allocIdx)}}};
  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
  QNN_TENSOR_SET_MEM_HANDLE(tensor, nullptr);
  Qnn_MemHandle_t memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);

  if (QNN_SUCCESS !=
      m_qnnInterface->memRegister(m_contextHandle, &memDescriptor, 1, &(memHandle))) {
    QNN_ERROR("DmaRegs: Failure to register ion memory with the backend");
    return false;
  }
  QNN_DEBUG(
      "DmaRegs: Memregister successful with handle %p for DMA buffer with size: %zu and "
      "fd %d",
      memHandle,
      m_dmaAllocator->getBufferSize(allocIdx),
      m_dmaAllocator->getFd(allocIdx));
  QNN_TENSOR_SET_MEM_HANDLE(tensor, memHandle);
  return true;
}

bool DmaRegs::mapTensorBuffer(Qnn_Tensor_t* tensor, uint64_t allocIdx, size_t /*tensorDatasize*/) {
  if (!tensor) {
    QNN_ERROR("DmaRegs: Received nullptr for tensor");
    return false;
  }

  if (m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("DmaRegs: Tensor already registered");
    return false;
  }

  if (true != registerTensor(tensor, allocIdx)) {
    const char* tname = QNN_TENSOR_GET_NAME(tensor);
    QNN_ERROR("DmaRegs: Tensor %s registration failed with the backend", tname);
    m_dmaAllocator->freeBuffer(allocIdx);
    return false;
  }
  m_tensorToAllocIdxMap[tensor] = allocIdx;
  m_allocIdxToTensorsMap[allocIdx].insert(tensor);
  return true;
}

bool DmaRegs::allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) {
  if (!tensor) {
    QNN_ERROR("DmaRegs: Received nullptr for tensor");
    return false;
  }

  if (m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("DmaRegs: Tensor already allocated");
    return false;
  }

  uint64_t allocIdx = m_dmaAllocator->allocate(tensorDataSize);

  if (true != registerTensor(tensor, allocIdx)) {
    const char* tname = QNN_TENSOR_GET_NAME(tensor);
    QNN_ERROR("DmaRegs: Tensor %s registration failed with the backend", tname);
    m_dmaAllocator->freeBuffer(allocIdx);
    return false;
  }
  m_tensorToAllocIdxMap[tensor] = allocIdx;
  m_allocIdxToTensorsMap[allocIdx].insert(tensor);
  return true;
}

bool DmaRegs::deregisterTensor(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("DmaRegs: Received nullptr for tensor");
    return false;
  }

  if (!m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("DmaRegs: Failed to deregister, tensor not registered");
    return false;
  }
  auto allocIdx = m_tensorToAllocIdxMap[tensor];
  if (!m_allocIdxToTensorsMap.contains(allocIdx)) {
    QNN_ERROR("DmaRegs: Failed to deregister, tensor not registered");
    return false;
  }
  if (m_allocIdxToTensorsMap[allocIdx].size() == 1) {
    auto memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
    if (QNN_SUCCESS != m_qnnInterface->memDeRegister(&memHandle, 1)) {
      QNN_ERROR("Failed to deregister ion memory with the backend");
      return false;
    }
    QNN_TENSOR_SET_MEM_HANDLE(tensor, nullptr);
    QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_UNDEFINED);
  }

  if (m_allocIdxToTensorsMap[allocIdx].contains(tensor)) {
    m_allocIdxToTensorsMap[allocIdx].erase(tensor);
  }
  if (m_allocIdxToTensorsMap[allocIdx].empty()) {
    m_allocIdxToTensorsMap.erase(allocIdx);
  }
  m_tensorToAllocIdxMap.erase(tensor);
  return true;
}

bool DmaRegs::freeTensorBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("DmaRegs: Received nullptr for tensor");
    return false;
  }
  if (!m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("Tensor is not registered with the backend.");
    return false;
  }

  auto tensorAllocIdx = m_tensorToAllocIdxMap[tensor];
  if (true != deregisterTensor(tensor)) {
    QNN_ERROR("Tensor is failed to deregister.");
    return false;
  }

  if (!m_allocIdxToTensorsMap.contains(tensorAllocIdx)) {
    m_dmaAllocator->freeBuffer(tensorAllocIdx);
  }

  return true;
}

bool DmaRegs::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
  if (nullptr == dest || nullptr == src) {
    QNN_ERROR("DmaRegs: Received nullptr");
    return false;
  }
  if (!m_tensorToAllocIdxMap.contains(src)) {
    QNN_ERROR("DmaRegs: Src Tensor not found");
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSOR_GET_MEM_TYPE(src));
  QNN_TENSOR_SET_MEM_HANDLE(dest, QNN_TENSOR_GET_MEM_HANDLE(src));
  m_tensorToAllocIdxMap[dest] = m_tensorToAllocIdxMap[src];
  m_allocIdxToTensorsMap[m_tensorToAllocIdxMap[src]].insert(dest);
  return true;
}

bool DmaRegs::useExternalMemory(Qnn_Tensor_t* /*dest*/, void* /*extMem*/) {
  QNN_WARN("External Memory not supported!!");
  return false;
}

// To register and deregister fused buffers
bool DmaRegs::deregisterTensorFusedBuffer(Qnn_Tensor_t* /*tensor*/) {
  QNN_WARN("Fused Buffers not supported\n");
  return false;
}

bool DmaRegs::mapFusedTensorBuffer(Qnn_Tensor_t* /*tensor*/,
                                   uint64_t /*alloc_idx*/,
                                   size_t /*offset*/,
                                   Qnn_ContextHandle_t /*ctx*/,
                                   size_t /*tensorDatasize*/) {
  QNN_WARN("Fused Buffers not supported\n");
  return false;
}

bool DmaRegs::beforeWriteToBuffer(Qnn_Tensor_t* tensor) {
  if (m_dmaAllocator->isIon()) {
    return true;
  }
  if (!tensor) {
    QNN_WARN("beforeWriteToBuffer: received a null pointer to a tensor");
    return false;
  }
  if (!m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("beforeWriteToBuffer: Tensor not found with address = %p", tensor);
    return false;
  }
  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
  auto ioctlReturnValue        = ioctl(getFd(tensor), DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    QNN_ERROR(
        "beforeWriteToBuffer: Error preparing the cache for buffer writes."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}

bool DmaRegs::afterWriteToBuffer(Qnn_Tensor_t* tensor) {
  if (m_dmaAllocator->isIon()) {
    return true;
  }
  if (!tensor) {
    QNN_WARN("afterWriteToBuffer: received a null pointer to a tensor");
    return false;
  }
  if (!m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("afterWriteToBuffer: Tensor not found with address = %p", tensor);
    return false;
  }
  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
  auto ioctlReturnValue        = ioctl(getFd(tensor), DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    QNN_ERROR(
        "afterWriteToBuffer: Error close the cache after buffer writing."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}

bool DmaRegs::beforeReadFromBuffer(Qnn_Tensor_t* tensor) {
  if (m_dmaAllocator->isIon()) {
    return true;
  }
  if (!tensor) {
    QNN_WARN("beforeReadFromBuffer: received a null pointer to a tensor");
    return false;
  }
  if (!m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("beforeReadFromBuffer: Tensor not found with address = %p", tensor);
    return false;
  }
  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
  auto ioctlReturnValue        = ioctl(getFd(tensor), DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    QNN_ERROR(
        "beforeReadFromBuffer: Error preparing the cache for buffer reading."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}

bool DmaRegs::afterReadFromBuffer(Qnn_Tensor_t* tensor) {
  if (m_dmaAllocator->isIon()) {
    return true;
  }
  if (!tensor) {
    QNN_WARN("afterReadFromBuffer: received a null pointer to a tensor");
    return false;
  }
  if (!m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("afterReadFromBuffer: Tensor not found with address = %p", tensor);
    return false;
  }

  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
  auto ioctlReturnValue        = ioctl(getFd(tensor), DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    QNN_ERROR(
        "afterReadFromBuffer: Error closing the cache after buffer reading."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}
