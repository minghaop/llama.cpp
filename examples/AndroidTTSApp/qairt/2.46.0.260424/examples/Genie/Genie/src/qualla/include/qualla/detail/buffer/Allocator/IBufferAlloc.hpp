//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <string>
#include <unordered_map>

class IBufferAlloc {
 public:
  // ctor
  IBufferAlloc() {}

  // dtor
  virtual ~IBufferAlloc() {}

  virtual bool initialize()                                                                  = 0;
  virtual void* allocateBuffer(uint64_t bufferSize, int32_t* fd)                             = 0;
  virtual bool allocateBuffers()                                                             = 0;
  virtual std::unordered_map<std::string, std::pair<uint64_t, size_t>>& getTensorAllocInfo() = 0;
  virtual uint64_t allocate(uint64_t bufferSize)                                             = 0;
  virtual void freeBuffer(uint64_t allocIdx)                                                 = 0;
  virtual void* getBuffer(uint64_t allocIdx)                                                 = 0;
  virtual int getFd(uint64_t allocIdx)                                                       = 0;
  virtual size_t getBufferSize(uint64_t allocIdx)                                            = 0;
  virtual size_t getTotalBufferSize(uint64_t allocIdx)                                       = 0;
};
