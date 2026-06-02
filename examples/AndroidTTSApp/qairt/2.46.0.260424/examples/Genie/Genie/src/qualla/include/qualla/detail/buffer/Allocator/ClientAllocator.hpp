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

#include "qualla/detail/buffer/Allocator/IBufferAlloc.hpp"
#include "qualla/detail/buffer/Estimator.hpp"

struct ClientBufferData {
  void* buffer;
  size_t bufferSize;

  ClientBufferData() : buffer(nullptr), bufferSize(0) {}
  ClientBufferData(void* data, size_t dataSize) : buffer(data), bufferSize(dataSize) {}
};

class ClientAllocator final : public IBufferAlloc {
 public:
  ClientAllocator(std::shared_ptr<Estimator> estimator, bool allowSharedBuffers = false);
  // Disable copy constructors, r-value referencing, etc
  ClientAllocator(const ClientAllocator&)            = delete;
  ClientAllocator& operator=(const ClientAllocator&) = delete;
  ClientAllocator(ClientAllocator&&)                 = delete;
  ClientAllocator& operator=(ClientAllocator&&)      = delete;
  ~ClientAllocator();

  bool initialize() override;

  void* allocateBuffer(uint64_t bufferSize, int32_t* fd) override;
  bool allocateBuffers() override;

  uint64_t allocate(uint64_t bufferSize) override;

  void freeBuffer(uint64_t allocIdx) override;

  void* getBuffer(uint64_t allocIdx) override;
  int getFd(uint64_t /*allocIdx*/) override;
  size_t getBufferSize(uint64_t allocIdx) override;
  size_t getTotalBufferSize(uint64_t allocIdx) override;

  std::unordered_map<std::string, std::pair<uint64_t, size_t>>& getTensorAllocInfo() override;

 private:
  /** Allocates a single buffer for mimicking RPC shared buffers. Used for testing purposes. */
  bool allocateSharedBuffer(std::unordered_map<std::string, size_t>& tensorAllocMap,
                            uint32_t& allocIdx);
  uint64_t m_lastAllocIdx{0};
  std::shared_ptr<Estimator> m_estimator;
  std::unordered_map<uint64_t, ClientBufferData*> m_buffers;
  std::unordered_map<std::string, std::pair<uint64_t, size_t>> m_tensorAllocInfo;
  int m_fd{-1};  // Default fd
  bool m_sharedBuffer{false};
};
