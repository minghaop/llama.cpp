//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <vector>

#include "QnnTypeMacros.hpp"
#include "qualla/detail/Log.hpp"
#include "qualla/detail/buffer/Allocator/ClientAllocator.hpp"

#define INVALID_FD 0

ClientAllocator::ClientAllocator(std::shared_ptr<Estimator> estimator, bool allowSharedBuffers)
    : m_estimator(estimator), m_sharedBuffer(allowSharedBuffers) {}

ClientAllocator::~ClientAllocator() {
  for (auto it = m_buffers.begin(); it != m_buffers.end();) {
    auto nxt = std::next(it);
    freeBuffer(it->first);
    it = nxt;
  }
  m_buffers.clear();
}

bool ClientAllocator::initialize() { return true; }

void* ClientAllocator::getBuffer(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("ClientAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return nullptr;
  }
  auto ptr = m_buffers[allocIdx];
  if (!ptr) {
    QNN_WARN("ClientAllocator: getBuffer Failed");
    return nullptr;
  }
  return ptr->buffer;
}

int ClientAllocator::getFd(uint64_t /*allocIdx*/) { return m_fd; }

size_t ClientAllocator::getTotalBufferSize(uint64_t allocIdx) { return getBufferSize(allocIdx); }

size_t ClientAllocator::getBufferSize(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("ClientAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return 0;
  }
  auto ptr = m_buffers[allocIdx];
  if (ptr == nullptr) {
    QNN_ERROR("ClientAllocator: getBufferSize Failed");
    return 0;
  }
  return ptr->bufferSize;
}

std::unordered_map<std::string, std::pair<uint64_t, size_t>>&
ClientAllocator::getTensorAllocInfo() {
  return m_tensorAllocInfo;
}

void* ClientAllocator::allocateBuffer(uint64_t bufferSize, int32_t* fd) {
  if (bufferSize == 0) {
    return nullptr;
  }
  void* buffer = malloc(bufferSize);
  if (nullptr == buffer) {
    return nullptr;
  }
  *fd                    = INVALID_FD;
  ClientBufferData* data = new ClientBufferData(buffer, bufferSize);
  return static_cast<void*>(data);
}

bool ClientAllocator::allocateSharedBuffer(std::unordered_map<std::string, size_t>& tensorAllocMap,
                                           uint32_t& allocIdx) {
  size_t allocSize = 0;
  // Accumulate tensor sizes and map their offsets within the current allocation index.
  for (auto& [tensor_name, tensor_size] : tensorAllocMap) {
    m_tensorAllocInfo[tensor_name] = std::make_pair(allocIdx, allocSize);
    allocSize += tensor_size;
  }
  int fd     = -1;
  void* data = allocateBuffer(allocSize, &fd);
  if (data == nullptr || fd == -1) {
    QNN_ERROR("ClientAllocator: allocation failed for the chunk size: %zu, fd: %d", allocSize, fd);
    return false;
  }
  m_buffers[allocIdx] = static_cast<ClientBufferData*>(data);
  m_lastAllocIdx      = allocIdx;
  allocIdx++;
  return true;
}

bool ClientAllocator::allocateBuffers() {
  if (!m_estimator) {
    QNN_ERROR("ClientAllocator: Estimator is null");
    return false;
  }
  // Client Buffers don't work based on fd, it is used as an imaginary index for pointer
  uint32_t allocIdx = m_lastAllocIdx + 1;
  for (auto& [_, tensors] : m_estimator->getEstimations()) {
    if (m_sharedBuffer) {
      if (!allocateSharedBuffer(tensors, allocIdx)) {
        return false;
      }
      continue;
    }
    // Since it is not working with Accumulated chunk sizes, will rely on adding individual length
    // memory.
    for (auto& [tensor_name, tensor_size] : tensors) {
      int fd     = -1;
      void* data = allocateBuffer(tensor_size, &fd);
      if (data == nullptr || fd == -1) {
        QNN_ERROR("ClientAllocator: mem alloc for tensor %s and fd %d.", tensor_name.c_str(), fd);
        return false;
      }
      m_tensorAllocInfo[tensor_name] = std::make_pair(allocIdx, tensor_size);
      m_buffers[allocIdx]            = static_cast<ClientBufferData*>(data);
      m_lastAllocIdx                 = allocIdx;
      allocIdx++;
    }
  }
  m_fd = INVALID_FD;
  return true;
}

uint64_t ClientAllocator::allocate(uint64_t bufferSize) {
  int fd     = -1;
  void* data = allocateBuffer(bufferSize, &fd);
  if (data == nullptr || fd == -1) {
    QNN_ERROR("ClientAllocator: mem alloc for buffer size %zu and fd %d.",
              static_cast<size_t>(bufferSize),
              fd);
    return 0;
  }
  m_lastAllocIdx++;
  m_buffers[m_lastAllocIdx] = static_cast<ClientBufferData*>(data);
  return m_lastAllocIdx;
}

void ClientAllocator::freeBuffer(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("ClientAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return;
  }
  auto clientBufferData = m_buffers[allocIdx];
  if (clientBufferData == nullptr) {
    m_buffers.erase(allocIdx);
    return;
  }
  free(clientBufferData->buffer);
  delete m_buffers[allocIdx];
  m_buffers.erase(allocIdx);
}
