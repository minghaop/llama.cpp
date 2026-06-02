//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <vector>

#include "dlwrap.hpp"
#include "PAL/DynamicLoading.hpp"
#include "qualla/detail/Log.hpp"
#include "qualla/detail/buffer/Allocator/RpcAllocator.hpp"

#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS  1

#if 1
#define TRACE_MEMORY_ALLOC QNN_DEBUG
#else
#define TRACE_MEMORY_ALLOC(fmt, ...)
#endif

RpcAllocator::RpcAllocator(std::shared_ptr<Estimator> estimator, uint32_t dataAlignmentSize)
    : m_libCdspRpc(nullptr),
      m_rpcMemAlloc(nullptr),
      m_rpcMemFree(nullptr),
      m_rpcMemToFd(nullptr),
      m_dataAlignmentSize(dataAlignmentSize),
      m_estimator(estimator) {}

bool RpcAllocator::initialize() {
  // On Android, 32-bit and 64-bit libcdsprpc.so can be found at /vendor/lib and /vendor/lib64
  // respectively. On Windows, it's installed into something like this
  //      c:\Windows\System32\DriverStore\FileRepository\qcnspmcdm8380.inf_arm64_30b9cc995571de6a\libcdsprpc.dll
#ifdef _WIN32
  const char* dsprpc_so = "libcdsprpc.dll";
#else
  const char* dsprpc_so = "libcdsprpc.so";
#endif

  m_libCdspRpc = pal::dynamicloading::dlOpen(dsprpc_so, pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
  if (nullptr == m_libCdspRpc) {
    const char* msg = pal::dynamicloading::dlError();
    QNN_ERROR("Unable to load backend. dlerror(): %s", msg ? msg : "Unknown error");
    return false;
  }
  m_rpcMemAlloc = reinterpret_cast<RpcMemAllocFn_t>(pal::dynamicloading::dlSym(m_libCdspRpc, "rpcmem_alloc"));
  m_rpcMemFree  = reinterpret_cast<RpcMemFreeFn_t>(pal::dynamicloading::dlSym(m_libCdspRpc, "rpcmem_free"));
  m_rpcMemToFd  = reinterpret_cast<RpcMemToFdFn_t>(pal::dynamicloading::dlSym(m_libCdspRpc, "rpcmem_to_fd"));
  if (nullptr == m_rpcMemAlloc || nullptr == m_rpcMemFree || nullptr == m_rpcMemToFd) {
    const char* msg = pal::dynamicloading::dlError();
    QNN_ERROR("Unable to access symbols in libcdsprpc. dlerror(): %s", msg ? msg : "Unknown error");
    return false;
  }

  return true;
}

RpcAllocator::~RpcAllocator() {
  for (auto it = m_buffers.begin(); it != m_buffers.end();) {
    auto nxt = std::next(it);
    freeBuffer(it->first);
    it = nxt;
  }
  m_buffers.clear();
  if (m_libCdspRpc) {
    QNN_DEBUG("Closing libcdsprpc.so handle");
    pal::dynamicloading::dlClose(m_libCdspRpc);
  }
}

void* RpcAllocator::getBuffer(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("RpcAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return nullptr;
  }
  auto ptr = m_buffers[allocIdx];
  if (!ptr) {
    QNN_WARN("RpcAllocator: getBuffer failed");
    return nullptr;
  }
  return ptr->memPointer;
}

int RpcAllocator::getFd(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("RpcAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return -1;
  }
  auto ptr = m_buffers[allocIdx];
  if (ptr == nullptr) {
    QNN_ERROR("RpcAllocator: getFd failed");
    return -1;
  }
  return ptr->fd;
}

size_t RpcAllocator::getBufferSize(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("RpcAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return 0;
  }
  auto ptr = m_buffers[allocIdx];
  if (ptr == nullptr) {
    QNN_ERROR("RpcAllocator: getBufferSize failed");
    return 0;
  }
  return ptr->size;
};

size_t RpcAllocator::getTotalBufferSize(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("RpcAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return 0;
  }
  auto ptr = m_buffers[allocIdx];
  if (ptr == nullptr) {
    QNN_ERROR("RpcAllocator: getTotalBufferSize failed");
    return 0;
  }
  return ptr->totalBufferSize;
}

std::unordered_map<std::string, std::pair<uint64_t, size_t>>& RpcAllocator::getTensorAllocInfo() {
  return m_tensorAllocInfo;
}

void* RpcAllocator::allocateBuffer(uint64_t bufferSize, int32_t* fd) {
  if (m_libCdspRpc == nullptr) {
    QNN_ERROR("RpcAllocator not initialized for fused buffer");
    return nullptr;
  }

  size_t alignedBufferSize = align(bufferSize, m_dataAlignmentSize);
  QNN_DEBUG("%s: m_dataAlignmentSize=%u, alignedBufferSize=%zu, original bufferSize=%zu",
            __func__,
            m_dataAlignmentSize,
            alignedBufferSize,
            static_cast<size_t>(bufferSize));
  auto memPointer = m_rpcMemAlloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, alignedBufferSize);

  if (!memPointer) {
    QNN_ERROR("Not able to allocate fused buffer of size: %zu", static_cast<size_t>(bufferSize));
    return nullptr;
  }

  QNN_DEBUG("Successfully allocated fused buffer at %p with size %zu",
            memPointer,
            static_cast<size_t>(bufferSize));

  if ((*fd = m_rpcMemToFd(memPointer)) == -1) {
    QNN_ERROR("Not able to get fd for the fused buffer of size: %zu",
              static_cast<size_t>(bufferSize));
    return nullptr;
  }

  RpcMem* data = new RpcMem(*fd, memPointer, bufferSize, alignedBufferSize);

  QNN_DEBUG("Retrieved fd %d for pointer %p", *fd, memPointer);
  return static_cast<void*>(data);
}

bool RpcAllocator::allocateBuffers() {
  if (!m_estimator) {
    QNN_ERROR("RpcAllocator: Estimator is null");
    return false;
  }
  uint64_t allocIdx = m_lastAllocIdx + 1, numChunks = 0;
  size_t totalAllocSize = 0;
  for (auto& [_, tensors] : m_estimator->getEstimations()) {
    // Calculate total allocation chunk size
    size_t allocSize = 0;
    for (const auto& [tensor_name, tensor_size] : tensors) {
      m_tensorAllocInfo[tensor_name] = std::make_pair(allocIdx, allocSize);
      allocSize += tensor_size;
    }

    // Allocate chunk for this unique context set
    if (allocSize <= 0) {
      QNN_ERROR("Unexpected chunk size detected. Please re-check IO allocations");
      return false;
    }
    int fd     = -1;
    void* data = allocateBuffer(allocSize, &fd);
    if (data == nullptr || fd == -1) {
      QNN_ERROR(
          "RpcAllocator: mem allocation failed for the chunk size: %zu, fd: %d", allocSize, fd);
      return false;
    }
    m_buffers[allocIdx] = static_cast<RpcMem*>(data);
    m_lastAllocIdx      = allocIdx;
    totalAllocSize += allocSize;
    allocIdx++;
    numChunks++;
  }

  QNN_INFO("Allocated total size = %zu across %zu buffers",
           totalAllocSize,
           static_cast<size_t>(numChunks));
  return true;
}

uint64_t RpcAllocator::allocate(uint64_t bufferSize) {
  int fd     = -1;
  void* data = allocateBuffer(bufferSize, &fd);
  if (data == nullptr || fd == -1) {
    QNN_ERROR("RpcAllocator: mem allocation failed for the chunk size: %zu, fd: %d",
              static_cast<size_t>(bufferSize),
              fd);
    return 0;
  }
  m_lastAllocIdx++;
  m_buffers[m_lastAllocIdx] = static_cast<RpcMem*>(data);
  return m_lastAllocIdx;
}

void RpcAllocator::freeBuffer(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("RpcAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return;
  }
  auto rpcBufferData = m_buffers[allocIdx];
  if (rpcBufferData == nullptr) {
    m_buffers.erase(allocIdx);
    return;
  }
  if (!rpcBufferData->memPointer) {
    QNN_ERROR("RpcAllocator: Nullptr received for memory with fd :%d", rpcBufferData->fd);
  }
  m_rpcMemFree(rpcBufferData->memPointer);
  delete rpcBufferData;
  m_buffers.erase(allocIdx);
}
