//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "QnnTypeMacros.hpp"
#include "qualla/detail/Log.hpp"
#include "qualla/detail/buffer/Registration/ClientRegs.hpp"

ClientRegs::ClientRegs(std::shared_ptr<ClientAllocator> clientAllocator)
    : m_clientAllocator(clientAllocator) {}

ClientRegs::~ClientRegs() {
  for (auto it = m_tensorToAllocIdxMap.begin(); it != m_tensorToAllocIdxMap.end();) {
    auto nxt = std::next(it);
    if (true != deregisterTensor(it->first)) {
      QNN_ERROR("Failed to deregister tensor.");
    }
    it = nxt;
  }
  m_tensorToAllocIdxMap.clear();
  m_extBufferTensors.clear();
};

void* ClientRegs::getBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor || !m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_WARN("getBuffer: received a null pointer to a tensor");
    return nullptr;
  }
  return QNN_TENSOR_GET_CLIENT_BUF(tensor).data;
}

size_t ClientRegs::getBufferSize(Qnn_Tensor_t* tensor) {
  if (!tensor || !m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_WARN("getBufferSize: received a null pointer to a tensor");
    return 0;
  }
  return QNN_TENSOR_GET_CLIENT_BUF(tensor).dataSize;
}

int ClientRegs::getFd(Qnn_Tensor_t* /*tensor*/) {
  QNN_WARN("getFd: This is not ION memory");
  return -1;
};

bool ClientRegs::initialize() { return m_clientAllocator->initialize(); }

bool ClientRegs::registerTensor(Qnn_Tensor_t* tensor, uint64_t allocIdx) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensors");
    return false;
  }
  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_RAW);
  Qnn_ClientBuffer_t clientBuffer;
  clientBuffer.data     = m_clientAllocator->getBuffer(allocIdx);
  clientBuffer.dataSize = m_clientAllocator->getBufferSize(allocIdx);
  QNN_TENSOR_SET_CLIENT_BUF(tensor, clientBuffer);
  m_tensorToAllocIdxMap[tensor] = allocIdx;
  return true;
}

bool ClientRegs::deregisterTensor(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensors");
    return false;
  }
  Qnn_ClientBuffer_t temp({nullptr, 0u});
  QNN_TENSOR_SET_CLIENT_BUF(tensor, temp);
  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_UNDEFINED);
  if (!m_tensorToAllocIdxMap.contains(tensor)) {
    return true;
  }
  uint64_t allocIdx = m_tensorToAllocIdxMap[tensor];
  if (m_allocIdxToTensorsMap[allocIdx].contains(tensor)) {
    m_allocIdxToTensorsMap[allocIdx].erase(tensor);
    if (m_allocIdxToTensorsMap[allocIdx].empty()) {
      m_allocIdxToTensorsMap.erase(allocIdx);
    }
  }
  m_tensorToAllocIdxMap.erase(tensor);
  return true;
}

bool ClientRegs::allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) {
  uint64_t allocIdx = m_clientAllocator->allocate(tensorDataSize);
  if (true != registerTensor(tensor, allocIdx)) {
    QNN_ERROR("mem registration failed for the clientBuffer");
    return false;
  }
  return true;
}

/**
 * Implements a fallback for fused mappings on ClientBuffers.
 *
 * Similar to RPC fused tensors, tensors can be mapped to an offset
 * within a shared buffer. However, the tensors cannot be registered
 * with QNN; we do not gain the same performance advantage as RPC fused buffers.
 */
bool ClientRegs::mapFusedTensorBuffer(Qnn_Tensor_t* tensor,
                                      uint64_t allocIdx,
                                      size_t offset,
                                      Qnn_ContextHandle_t contextHandle [[maybe_unused]],
                                      size_t tensorDatasize) {
  if (!mapTensorBuffer(tensor, allocIdx, tensorDatasize)) {
    return false;
  }

  Qnn_ClientBuffer_t clientBuffer = QNN_TENSOR_GET_CLIENT_BUF(tensor);
  clientBuffer.data               = static_cast<uint8_t*>(clientBuffer.data) + offset;
  clientBuffer.dataSize           = tensorDatasize;
  QNN_TENSOR_SET_CLIENT_BUF(tensor, clientBuffer);

  return true;
}

bool ClientRegs::mapTensorBuffer(Qnn_Tensor_t* tensor,
                                 uint64_t allocIdx,
                                 size_t /*tensorDatasize*/) {
  if (true != registerTensor(tensor, allocIdx)) {
    QNN_ERROR("mem registration failed for the clientBuffer");
    return false;
  }
  m_allocIdxToTensorsMap[allocIdx].insert(tensor);
  return true;
}

bool ClientRegs::freeTensorBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensors");
    return false;
  }
  if (m_extBufferTensors.contains(tensor)) {
    QNN_DEBUG("Tensor is using external memory with the backend.");
    return true;
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
  if (m_allocIdxToTensorsMap[tensorAllocIdx].empty()) {
    m_clientAllocator->freeBuffer(tensorAllocIdx);
    m_allocIdxToTensorsMap.erase(tensorAllocIdx);
  }
  return true;
}

bool ClientRegs::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
  if (nullptr == dest || nullptr == src) {
    QNN_ERROR("Received nullptr");
    return false;
  }

  if (false == freeTensorBuffer(dest)) {
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSOR_GET_MEM_TYPE(src));
  QNN_TENSOR_SET_CLIENT_BUF(dest, QNN_TENSOR_GET_CLIENT_BUF(src));
  m_tensorToAllocIdxMap[dest] = m_tensorToAllocIdxMap[src];
  m_allocIdxToTensorsMap[m_tensorToAllocIdxMap[src]].insert(dest);

  return true;
}

bool ClientRegs::useExternalMemory(Qnn_Tensor_t* dest, void* extMem) {
  if (nullptr == dest || nullptr == extMem) {
    QNN_ERROR("Received nullptr");
    return false;
  }

  Qnn_ClientBuffer_t clientBuffer;
  clientBuffer.data     = extMem;
  clientBuffer.dataSize = QNN_TENSOR_GET_CLIENT_BUF(dest).dataSize;
  if (false == freeTensorBuffer(dest)) {
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSORMEMTYPE_RAW);
  QNN_TENSOR_SET_CLIENT_BUF(dest, clientBuffer);
  m_extBufferTensors.insert(dest);
  return true;
}
