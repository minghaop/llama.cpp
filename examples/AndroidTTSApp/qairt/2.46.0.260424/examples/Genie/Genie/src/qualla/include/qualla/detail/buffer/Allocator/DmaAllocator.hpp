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

typedef void* (*DmaBufCreateFn_t)();
typedef int (*DmaBufAllocFn_t)(void*, const char*, size_t, unsigned int, size_t);
typedef void (*DmaBufDeinitFn_t)(void*);

// Definitions obtained from Adreno OpenCL ION SDK example
#define ION_SYSTEM_HEAP_ID           25
#define ION_HEAP(bit)                (1 << (bit))
#define ION_CL_DEVICE_PAGE_SIZE_QCOM 4096

typedef int (*IonOpenFn_t)();
typedef int (*IonAllocFd_t)(int, size_t, size_t, unsigned int, unsigned int, int*);
typedef int (*IonCloseFn_t)(int);

struct DmaBufferData {
  void* dmaBufferAllocator;
  int ionAllocatorFd;
  int fd;
  void* memPointer;
  size_t totalBufferSize;

  DmaBufferData()
      : dmaBufferAllocator(nullptr),
        ionAllocatorFd(-1),
        fd(-1),
        memPointer(nullptr),
        totalBufferSize(0) {}
  DmaBufferData(
      void* bufferAllocatorIn, int ionAllocatorFdIn, int fdIn, void* memPointerIn, size_t sizeIn)
      : dmaBufferAllocator(bufferAllocatorIn),
        ionAllocatorFd(ionAllocatorFdIn),
        fd(fdIn),
        memPointer(memPointerIn),
        totalBufferSize(sizeIn) {}
};

class DmaAllocator final : public IBufferAlloc {
 public:
  DmaAllocator(std::shared_ptr<Estimator> estimator);
  // Disable copy constructors, r-value referencing, etc
  DmaAllocator(const DmaAllocator&)            = delete;
  DmaAllocator& operator=(const DmaAllocator&) = delete;
  DmaAllocator(DmaAllocator&&)                 = delete;
  DmaAllocator& operator=(DmaAllocator&&)      = delete;
  ~DmaAllocator();

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

  bool isIon();

 private:
  bool m_useIonMemHandle = false;
  // Pointer to the dlopen'd libdmabufheap.so shared library which contains
  // dmaBufCreate, dmaBufAlloc, dmaBufDeinit
  void* m_libDmaBufHeapHandle = nullptr;
  DmaBufCreateFn_t m_dmaBufCreate{nullptr};
  DmaBufAllocFn_t m_dmaBufAlloc{nullptr};
  DmaBufDeinitFn_t m_dmaBufDeinit{nullptr};

  // Pointer to the dlopen'd libion.so shared library which contains
  // ionOpenFn, ionAllocFn, ionCloseFn
  void* m_libIonMemHandle = nullptr;
  IonOpenFn_t m_ionOpenFn{nullptr};
  IonAllocFd_t m_ionAllocFn{nullptr};
  IonCloseFn_t m_ionCloseFn{nullptr};

  std::shared_ptr<Estimator> m_estimator;
  uint64_t m_lastAllocIdx{0};
  std::unordered_map<std::string, std::pair<uint64_t, size_t>> m_tensorAllocInfo;
  std::unordered_map<uint64_t, DmaBufferData*> m_buffers;  // { allocIdx -> DmaBufferData }
};
