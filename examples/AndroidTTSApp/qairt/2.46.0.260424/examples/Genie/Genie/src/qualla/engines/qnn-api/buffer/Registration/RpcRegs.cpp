//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "QnnHtpMem.h"
#include "QnnMem.h"
#include "QnnTypeMacros.hpp"
#include "dlwrap.hpp"
#include "qualla/detail/Log.hpp"
#include "qualla/detail/buffer/Registration/RpcRegs.hpp"

#if 1
#define TRACE_MEMORY_ALLOC QNN_DEBUG
#else
#define TRACE_MEMORY_ALLOC(fmt, ...)
#endif

RpcRegs::RpcRegs(Qnn_ContextHandle_t contextHandle,
                 QNN_INTERFACE_VER_TYPE* qnnInterface,
                 std::shared_ptr<RpcAllocator> rpcAllocator)
    : m_qnnInterface(qnnInterface), m_contextHandle(contextHandle), m_rpcAllocator(rpcAllocator) {
  (void)m_contextHandle;
}

bool RpcRegs::initialize() { return m_rpcAllocator->initialize(); }

RpcRegs::~RpcRegs() {
  for (auto it = m_tensorToAllocIdxMap.begin(); it != m_tensorToAllocIdxMap.end();) {
    auto nxt = std::next(it);
    if (true != deregisterTensor(it->first)) {
      QNN_ERROR("Failed to deregister tensor.");
    }
    it = nxt;
  }
  // clean all maps for safe side, though these must have been clean by now.
  m_tensorToAllocIdxMap.clear();
  m_allocIdxToTensorsMap.clear();
  m_memHandleToRpcBufferData.clear();
  m_memConfigList.clear();
}

RpcBufferData* RpcRegs::getRpcMemTensorData(Qnn_Tensor_t* tensor) {
  if (tensor == nullptr) return nullptr;
  Qnn_MemHandle_t mem_handle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  if (mem_handle == nullptr || !m_memHandleToRpcBufferData.contains(mem_handle)) return nullptr;
  return m_memHandleToRpcBufferData[mem_handle];
}

void* RpcRegs::getBuffer(Qnn_Tensor_t* tensor) {
  RpcBufferData* data = getRpcMemTensorData(tensor);
  if (data == nullptr) {
    QNN_ERROR("getBuffer : Couldn't find tensor %p", tensor);
    return nullptr;
  }
  return data->memPointer;
}

int RpcRegs::getFd(Qnn_Tensor_t* tensor) {
  RpcBufferData* data = getRpcMemTensorData(tensor);
  if (data == nullptr) {
    QNN_ERROR("getFd : Couldn't find tensor %p", tensor);
    return -1;
  }
  return data->fd;
}

size_t RpcRegs::getOffset(Qnn_Tensor_t* tensor) {
  RpcBufferData* data = getRpcMemTensorData(tensor);
  if (data == nullptr) {
    QNN_ERROR("getOffset : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->offset;
}

size_t RpcRegs::getBufferSize(Qnn_Tensor_t* tensor) {
  RpcBufferData* data = getRpcMemTensorData(tensor);
  if (data == nullptr) {
    QNN_ERROR("getBufferSize : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->size;
};

size_t RpcRegs::getTotalBufferSize(Qnn_Tensor_t* tensor) {
  RpcBufferData* data = getRpcMemTensorData(tensor);
  if (data == nullptr) {
    QNN_ERROR("getTotalBufferSize : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->totalBufferSize;
}

uint64_t RpcRegs::getAllocIdx(Qnn_Tensor_t* tensor) {
  RpcBufferData* data = getRpcMemTensorData(tensor);
  if (data == nullptr) {
    QNN_ERROR("getAllocIdx : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->allocIdx;
}

bool RpcRegs::registerTensor(Qnn_Tensor_t* tensor, uint64_t allocIdx) {
  if (!tensor) {
    QNN_ERROR("RpcRegs: Received nullptr for tensor");
    return false;
  }
  if (allocIdx == UINT64_MAX) {
    QNN_ERROR("RpcRegs: Received invalid allocation Id.");
    return false;
  }
  Qnn_MemDescriptor_t memDescriptor = {
      {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
      QNN_TENSOR_GET_DATA_TYPE(tensor),
      QNN_MEM_TYPE_ION,
      {{-1}}};
  int curFd                = m_rpcAllocator->getFd(allocIdx);
  memDescriptor.ionInfo.fd = curFd;
  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
  QNN_TENSOR_SET_MEM_HANDLE(tensor, nullptr);

  Qnn_MemHandle_t memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  if (QNN_SUCCESS !=
      m_qnnInterface->memRegister(m_contextHandle, &memDescriptor, 1, &(memHandle))) {
    const char* tname = QNN_TENSOR_GET_NAME(tensor);
    QNN_ERROR("memRegister fail %s (ctx=%p fd=%d)", tname, m_contextHandle, curFd);
    return false;
  }
  QNN_TENSOR_SET_MEM_HANDLE(tensor, memHandle);
  return true;
}

bool RpcRegs::registerFusedTensors(Qnn_Tensor_t* tensor,
                                   RpcBufferData* rpcBufferData,
                                   Qnn_ContextHandle_t contextHandle) {
  if (!tensor) {
    QNN_ERROR("RpcRegs: Received nullptr for tensor");
    return false;
  }
  if (!rpcBufferData) {
    QNN_ERROR("RpcRegs: Received invalid RPC buffer data.");
    return false;
  }
  // Register a new memHandle based on function arguments

  QnnMemHtp_Descriptor_t htp_mem_desciptor = {
      QNN_HTP_MEM_SHARED_BUFFER, rpcBufferData->totalBufferSize, {0}};
  htp_mem_desciptor.sharedBufferConfig.fd     = rpcBufferData->fd;
  htp_mem_desciptor.sharedBufferConfig.offset = rpcBufferData->offset;

  Qnn_MemDescriptor_t mem_descriptor = {
      {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
      QNN_TENSOR_GET_DATA_TYPE(tensor),
      QNN_MEM_TYPE_CUSTOM,
      {{-1}}};
  mem_descriptor.customInfo = &htp_mem_desciptor;

  Qnn_MemHandle_t mem_handle = nullptr;
  Qnn_ErrorHandle_t ret =
      m_qnnInterface->memRegister(contextHandle, &mem_descriptor, 1, &mem_handle);
  if (ret != QNN_SUCCESS) {
    const char* tname = QNN_TENSOR_GET_NAME(tensor);
    QNN_ERROR("%-20s (ctx=%p fd=%d offset=%zu), totalBufSize: %zu",
              tname,
              contextHandle,
              rpcBufferData->fd,
              rpcBufferData->offset,
              rpcBufferData->totalBufferSize);
    QNN_ERROR("memRegister ERROR(%lu)", static_cast<unsigned long>(ret));
    return false;
  }
  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
  QNN_TENSOR_SET_MEM_HANDLE(tensor, mem_handle);
  return true;
}

bool RpcRegs::allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensor");
    return false;
  }

  if (m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("Tensor already allocated");
    return false;
  }

  auto allocIdx = m_rpcAllocator->allocate(tensorDataSize);
  if (allocIdx == UINT64_MAX) {
    QNN_ERROR("Rpc mem_alloc failure");
    return false;
  }

  const char* tname            = QNN_TENSOR_GET_NAME(tensor);
  void* memPointer             = m_rpcAllocator->getBuffer(allocIdx);
  int fd                       = m_rpcAllocator->getFd(allocIdx);
  RpcBufferData* rpcBufferData = new RpcBufferData(fd, memPointer, tensorDataSize, allocIdx);
  if (true != registerTensor(tensor, allocIdx)) {
    QNN_ERROR("Failed to register fused tensor buffer for %s for fd: %u", tname, fd);
    delete rpcBufferData;
    m_rpcAllocator->freeBuffer(allocIdx);
    return false;
  }
  Qnn_MemHandle_t memHandle             = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  m_memHandleToRpcBufferData[memHandle] = rpcBufferData;
  return true;
}

bool RpcRegs::mapTensorBuffer(Qnn_Tensor_t* tensor, uint64_t allocIdx, size_t tensorDataSize) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensor");
    return false;
  }

  if (m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("Tensor already mapped");
    return false;
  }

  const char* tname            = QNN_TENSOR_GET_NAME(tensor);
  void* memPointer             = m_rpcAllocator->getBuffer(allocIdx);
  int fd                       = m_rpcAllocator->getFd(allocIdx);
  RpcBufferData* rpcBufferData = new RpcBufferData(fd, memPointer, tensorDataSize, allocIdx);
  if (true != registerTensor(tensor, allocIdx)) {
    QNN_ERROR("Failed to register fused tensor buffer for %s for fd: %u", tname, fd);
    delete rpcBufferData;
    m_rpcAllocator->freeBuffer(allocIdx);
    return false;
  }
  Qnn_MemHandle_t memHandle             = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  m_memHandleToRpcBufferData[memHandle] = rpcBufferData;
  m_tensorToAllocIdxMap[tensor]         = allocIdx;
  m_allocIdxToTensorsMap[allocIdx].insert(tensor);
  return true;
}

bool RpcRegs::mapFusedTensorBuffer(Qnn_Tensor_t* tensor,
                                   uint64_t allocIdx,
                                   size_t offset,
                                   Qnn_ContextHandle_t contextHandle,
                                   size_t tensorDatasize) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensor");
    return false;
  }

  const char* tname = QNN_TENSOR_GET_NAME(tensor);
  int curFd         = m_rpcAllocator->getFd(allocIdx);
  // Check if tensor already has a memHandle assigned
  Qnn_MemHandle_t curMemHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  if (curMemHandle != nullptr) {
    // Check if memHandle is already identical to requested buffer and offset
    RpcBufferData* curRpcBufferData = getRpcMemTensorData(tensor);
    if (curRpcBufferData == nullptr) return false;
    if (curRpcBufferData->fd == curFd && curRpcBufferData->offset == offset) {
      return true;
    }

    // updated offset, deregister previous mem_handle
    tensorDatasize = (tensorDatasize == 0) ? curRpcBufferData->size : tensorDatasize;
    deregisterTensor(tensor);
  } else {
    // For inital tensors, we need to check if the tensor can re-use a memHandle
    // from another tensor in the same context
    auto memConfig = std::make_tuple(curFd, offset, contextHandle);
    if (m_memConfigList.contains(memConfig)) {
      auto& parentTensor              = m_memConfigList[memConfig];
      Qnn_MemHandle_t parentMemHandle = QNN_TENSOR_GET_MEM_HANDLE(parentTensor);
      QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
      QNN_TENSOR_SET_MEM_HANDLE(tensor, parentMemHandle);
      TRACE_MEMORY_ALLOC("%-20s : Mapping to memHandle %p", tname, parentMemHandle);
      return true;
    }
  }

  void* memPointer             = m_rpcAllocator->getBuffer(allocIdx);
  size_t totalBufferSize       = m_rpcAllocator->getBufferSize(allocIdx);
  RpcBufferData* rpcBufferData = new RpcBufferData(curFd,
                                                   reinterpret_cast<uint8_t*>(memPointer) + offset,
                                                   tensorDatasize,
                                                   totalBufferSize,
                                                   offset,
                                                   allocIdx);
  if (true != registerFusedTensors(tensor, rpcBufferData, contextHandle)) {
    QNN_ERROR("Failed to register fused tensor buffer for %s for fd: %u", tname, curFd);
    delete rpcBufferData;
    return false;
  }
  Qnn_MemHandle_t memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  TRACE_MEMORY_ALLOC("%-20s (ctx=%p fd=%d offset=%zu) memPointer=%p memHandle=%p",
                     tname,
                     contextHandle,
                     curFd,
                     offset,
                     (static_cast<uint8_t*>(memPointer) + offset),
                     curMemHandle);
  m_memHandleToRpcBufferData[memHandle] = rpcBufferData;
  m_tensorToAllocIdxMap[tensor]         = allocIdx;
  m_allocIdxToTensorsMap[allocIdx].insert(tensor);
  if (curMemHandle == nullptr) {  // Cache memory config for initial memRegisters only
    m_memConfigList[std::make_tuple(curFd, offset, contextHandle)] = tensor;
  }
  return true;
}

bool RpcRegs::freeTensorBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensor");
    return false;
  }

  if (!m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("Tensor is not registered with the backend.");
    return false;
  }

  auto tensorAllocIdx = m_tensorToAllocIdxMap[tensor];
  if (true != deregisterTensor(tensor)) {
    QNN_ERROR("Failed to deregister tensor.");
    return false;
  }

  if (m_allocIdxToTensorsMap[tensorAllocIdx].empty()) {
    m_rpcAllocator->freeBuffer(tensorAllocIdx);
    m_allocIdxToTensorsMap.erase(tensorAllocIdx);
  }

  return true;
}

bool RpcRegs::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
  if (nullptr == dest || nullptr == src) {
    QNN_ERROR("Received nullptr");
    return false;
  }

  if (m_tensorToAllocIdxMap.contains(src)) {
    QNN_ERROR("Src Tensor not found");
    return false;
  }

  if (false == freeTensorBuffer(dest)) {
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSOR_GET_MEM_TYPE(src));
  QNN_TENSOR_SET_MEM_HANDLE(dest, QNN_TENSOR_GET_MEM_HANDLE(src));
  m_tensorToAllocIdxMap[dest] = m_tensorToAllocIdxMap[src];
  m_allocIdxToTensorsMap[m_tensorToAllocIdxMap[src]].insert(dest);

  return true;
}

bool RpcRegs::useExternalMemory(Qnn_Tensor_t* /*dest*/, void* /*extMem*/) {
  QNN_ERROR("We don't support external memory feature for shared buffers yet!");
  return false;
}

bool RpcRegs::deregisterTensor(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensors");
    return false;
  }
  auto memHandle                  = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  const char* tname               = QNN_TENSOR_GET_NAME(tensor);
  RpcBufferData* curRpcBufferData = getRpcMemTensorData(tensor);
  if (curRpcBufferData == nullptr) {
    QNN_ERROR("Failed to get RPC buffer data for tensor %s", tname);
    return false;
  }
  TRACE_MEMORY_ALLOC("memDeRegister %-20s (fd=%d offset=%lu) memHandle=%p",
                     tname,
                     curRpcBufferData->fd,
                     curRpcBufferData->offset,
                     memHandle);
  if (m_memHandleToRpcBufferData.contains(memHandle)) m_memHandleToRpcBufferData.erase(memHandle);
  if (QNN_SUCCESS != m_qnnInterface->memDeRegister(&memHandle, 1)) {
    QNN_ERROR("Failed to deregister ion memory with the backend for %s", tname);
    return false;
  }
  QNN_TENSOR_SET_MEM_HANDLE(tensor, nullptr);
  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_UNDEFINED);
  auto allocIdx = curRpcBufferData->allocIdx;
  if (m_allocIdxToTensorsMap[allocIdx].contains(tensor)) {
    m_allocIdxToTensorsMap[allocIdx].erase(tensor);
  }
  delete curRpcBufferData;
  return true;
}

bool RpcRegs::deregisterTensorFusedBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensors");
    return false;
  }
  if (!m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("Tensor is not registered with the backend.");
    return false;
  }
  if (true != deregisterTensor(tensor)) {
    QNN_ERROR("Failed to deregister tensor.");
    return false;
  }
  if (m_tensorToAllocIdxMap.contains(tensor)) {
    m_tensorToAllocIdxMap.erase(tensor);
  }
  return true;
}
