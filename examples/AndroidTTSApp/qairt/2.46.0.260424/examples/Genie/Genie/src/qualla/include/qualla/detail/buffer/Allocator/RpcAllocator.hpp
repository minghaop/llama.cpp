//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>

#include "qualla/detail/buffer/Allocator/IBufferAlloc.hpp"
#include "qualla/detail/buffer/Estimator.hpp"

#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS  1
#define RPC_FUSED_BUFFERS     0
#define RPC_NON_FUSED_BUFFERS 1

inline uint64_t align(uint64_t size, uint32_t dataAlignmentSize) {
  return dataAlignmentSize
             ? (size + dataAlignmentSize - 1) & ~(static_cast<uint64_t>(dataAlignmentSize) - 1)
             : size;
}

typedef void* (*RpcMemAllocFn_t)(int, uint32_t, int);
typedef void (*RpcMemFreeFn_t)(void*);
typedef int (*RpcMemToFdFn_t)(void*);

struct RpcMem {
  int fd{-1};
  void* memPointer{nullptr};
  size_t size{0};
  size_t totalBufferSize{0};

  RpcMem() = default;
  RpcMem(int fdIn, void* memPointerIn, size_t sizeIn)
      : fd(fdIn), memPointer(memPointerIn), size(sizeIn) {}
  RpcMem(int fdIn, void* memPointerIn, size_t sizeIn, size_t totalBufferSizeIn)
      : fd(fdIn), memPointer(memPointerIn), size(sizeIn), totalBufferSize(totalBufferSizeIn) {}
};

class RpcAllocator final : public IBufferAlloc {
 public:
  RpcAllocator(std::shared_ptr<Estimator> estimator, uint32_t dataAlignmentSize = 0);
  // Disable copy constructors, r-value referencing, etc
  RpcAllocator(const RpcAllocator&)            = delete;
  RpcAllocator& operator=(const RpcAllocator&) = delete;
  RpcAllocator(RpcAllocator&&)                 = delete;
  RpcAllocator& operator=(RpcAllocator&&)      = delete;
  ~RpcAllocator();

  bool initialize() override;

  void* allocateBuffer(uint64_t bufferSize, int32_t* fd) override;
  bool allocateBuffers() override;

  uint64_t allocate(uint64_t bufferSize) override;

  void freeBuffer(uint64_t allocIdx) override;

  void* getBuffer(uint64_t allocIdx) override;
  int getFd(uint64_t allocIdx) override;
  size_t getBufferSize(uint64_t allocIdx) override;
  size_t getTotalBufferSize(uint64_t allocIdx) override;

  std::unordered_map<std::string, std::pair<uint64_t, size_t>>& getTensorAllocInfo() override;

 private:
  // IbufferAllocator
  // Pointer to the dlopen'd libcdsprpc.so shared library which contains
  // rpcmem_alloc, rpcmem_free, rpcmem_to_fd APIs
  void* m_libCdspRpc{nullptr};
  // Function pointer to rpcmem_alloc
  RpcMemAllocFn_t m_rpcMemAlloc{nullptr};
  // Function pointer to rpcmem_free
  RpcMemFreeFn_t m_rpcMemFree{nullptr};
  // Function pointer to rpcmem_to_fd
  RpcMemToFdFn_t m_rpcMemToFd{nullptr};
  uint32_t m_dataAlignmentSize{0};

  std::shared_ptr<Estimator> m_estimator{nullptr};
  uint64_t m_lastAllocIdx{0};
  std::unordered_map<std::string, std::pair<uint64_t, size_t>> m_tensorAllocInfo;
  std::unordered_map<uint64_t, RpcMem*> m_buffers;  // { allocIdx --> RpcBufferData}
};
