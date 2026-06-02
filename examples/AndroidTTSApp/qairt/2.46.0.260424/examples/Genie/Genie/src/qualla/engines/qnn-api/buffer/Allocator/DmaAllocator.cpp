//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <dlfcn.h>
#include <fcntl.h>
#ifndef __QNX__
 #include <linux/dma-buf.h>
#endif
#include <pthread.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "PAL/DynamicLoading.hpp"
#include "qualla/detail/Log.hpp"
#include "qualla/detail/buffer/Allocator/DmaAllocator.hpp"

DmaAllocator::DmaAllocator(std::shared_ptr<Estimator> estimator)
    : m_libDmaBufHeapHandle(nullptr),
      m_dmaBufCreate(nullptr),
      m_dmaBufAlloc(nullptr),
      m_dmaBufDeinit(nullptr),
      m_libIonMemHandle(nullptr),
      m_ionOpenFn(nullptr),
      m_ionAllocFn(nullptr),
      m_ionCloseFn(nullptr),
      m_estimator(estimator) {}

bool DmaAllocator::initialize() {
  const char* dmaHeapLocation = "/dev/dma_heap";
  const char* ionLocation     = "/dev/ion";
  if (std::filesystem::exists(dmaHeapLocation)) {
    QNN_DEBUG("Using DmaBuf Allocator");
    m_useIonMemHandle = false;
  } else if (std::filesystem::exists(ionLocation)) {
    QNN_DEBUG("Using ION Allocator");
    m_useIonMemHandle = true;
  } else {
    QNN_ERROR("Zero Copy Memory Not Supported");
    return false;
  }

  if (m_useIonMemHandle) {
    // On Android, 32-bit and 64-bit libion.so can be found at /system/lib and /system/lib64
    //  respectively.
    const std::string defaultLibPaths[] = {"libion.so"};
    for (const auto& path : defaultLibPaths) {
      m_libIonMemHandle = pal::dynamicloading::dlOpen(path.c_str(), pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
      if (m_libIonMemHandle != nullptr) {
        break;
      }
    }
    if (nullptr == m_libIonMemHandle) {
      const char* msg = pal::dynamicloading::dlError();
      QNN_ERROR("Unable to load memory allocator. dlerror(): %s", msg ? msg : "Unknown error");
      return false;
    }
    m_ionOpenFn = reinterpret_cast<IonOpenFn_t>(pal::dynamicloading::dlSym(m_libIonMemHandle, "ion_open"));
    if (nullptr == m_ionOpenFn) {
      const char* msg = pal::dynamicloading::dlError();
      QNN_ERROR("Unable to access symbols in libion. dlerror(): %s", msg ? msg : "Unknown error");
      return false;
    }
    m_ionAllocFn = reinterpret_cast<IonAllocFd_t>(pal::dynamicloading::dlSym(m_libIonMemHandle, "ion_alloc_fd"));
    if (nullptr == m_ionAllocFn) {
      const char* msg = pal::dynamicloading::dlError();
      QNN_ERROR("Unable to access symbols in libion. dlerror(): %s", msg ? msg : "Unknown error");
      return false;
    }
    m_ionCloseFn = reinterpret_cast<IonCloseFn_t>(pal::dynamicloading::dlSym(m_libIonMemHandle, "ion_close"));
    if (nullptr == m_ionCloseFn) {
      const char* msg = pal::dynamicloading::dlError();
      QNN_ERROR("Unable to access symbols in libion. dlerror(): %s", msg ? msg : "Unknown error");
      return false;
    }
    return true;
  }

  // On Android, 32-bit and 64-bit libdmaBufheap.so can be found at /system/lib and /system/lib64
  //  respectively.
  const std::string defaultLibPaths[] = {"libdmabufheap.so", "libdmabufheap.so.0"};
  for (const auto& path : defaultLibPaths) {
    m_libDmaBufHeapHandle = pal::dynamicloading::dlOpen(path.c_str(), pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
    if (m_libDmaBufHeapHandle != nullptr) {
      break;
    }
  }

  if (nullptr == m_libDmaBufHeapHandle) {
    const char* msg = pal::dynamicloading::dlError();
    QNN_ERROR("Unable to load memory allocator. dlerror(): %s", msg ? msg : "Unknown error");
    return false;
  }

  m_dmaBufCreate = reinterpret_cast<DmaBufCreateFn_t>(
      pal::dynamicloading::dlSym(m_libDmaBufHeapHandle, "CreateDmabufHeapBufferAllocator"));
  m_dmaBufAlloc =
      reinterpret_cast<DmaBufAllocFn_t>(pal::dynamicloading::dlSym(m_libDmaBufHeapHandle, "DmabufHeapAlloc"));
  m_dmaBufDeinit = reinterpret_cast<DmaBufDeinitFn_t>(
      pal::dynamicloading::dlSym(m_libDmaBufHeapHandle, "FreeDmabufHeapBufferAllocator"));

  if (nullptr == m_dmaBufCreate || nullptr == m_dmaBufAlloc || nullptr == m_dmaBufDeinit) {
    const char* msg = pal::dynamicloading::dlError();
    QNN_ERROR("Unable to access symbols in libdmaBufheap. dlerror(): %s", msg ? msg : "Unknown error");
    return false;
  }
  return true;
}

DmaAllocator::~DmaAllocator() {
  for (auto it = m_buffers.begin(); it != m_buffers.end();) {
    auto nxt = std::next(it);
    freeBuffer(it->first);
    it = nxt;
  }
  m_buffers.clear();
  if (m_libDmaBufHeapHandle) {
    pal::dynamicloading::dlClose(m_libDmaBufHeapHandle);
    m_libDmaBufHeapHandle = nullptr;
  }
  if (m_libIonMemHandle) {
    pal::dynamicloading::dlClose(m_libIonMemHandle);
    m_libIonMemHandle = nullptr;
  }
}

void* DmaAllocator::getBuffer(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("DmaAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return nullptr;
  }
  auto ptr = m_buffers[allocIdx];
  if (!ptr) {
    QNN_WARN("DmaAllocator: getBuffer failed");
    return nullptr;
  }
  return ptr->memPointer;
}

int DmaAllocator::getFd(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("DmaAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return -1;
  }
  auto ptr = m_buffers[allocIdx];
  if (ptr == nullptr) {
    QNN_ERROR("DmaAllocator: getFd failed");
    return -1;
  }
  return ptr->fd;
}

size_t DmaAllocator::getBufferSize(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("DmaAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return 0;
  }
  auto ptr = m_buffers[allocIdx];
  if (ptr == nullptr) {
    QNN_ERROR("DmaAllocator: getBufferSize failed");
    return 0;
  }
  return ptr->totalBufferSize;
};

size_t DmaAllocator::getTotalBufferSize(uint64_t allocIdx) { return getBufferSize(allocIdx); }

bool DmaAllocator::isIon() { return m_useIonMemHandle; }

std::unordered_map<std::string, std::pair<uint64_t, size_t>>& DmaAllocator::getTensorAllocInfo() {
  return m_tensorAllocInfo;
}

void* DmaAllocator::allocateBuffer(uint64_t bufferSize, int32_t* fd) {
  int ionAllocatorFd       = -1;
  void* dmaBufferAllocator = nullptr;
  if (m_useIonMemHandle) {
    if (m_libIonMemHandle == nullptr) {
      QNN_ERROR("DmaAllocator not initialized");
      return nullptr;
    }

    ionAllocatorFd = m_ionOpenFn();
    if (ionAllocatorFd == -1) {
      QNN_ERROR("DmaAllocator: nullptr returned for ion_open().");
      return nullptr;
    }

    int errorCode = m_ionAllocFn(ionAllocatorFd,
                                 bufferSize,
                                 ION_CL_DEVICE_PAGE_SIZE_QCOM,
                                 ION_HEAP(ION_SYSTEM_HEAP_ID),
                                 0,
                                 fd);
    if (errorCode < 0) {
      QNN_ERROR("DmaBufAlloc returned a invalid file descriptor = %d", static_cast<int>(*fd));
      return nullptr;
    }
  } else {
    if (m_libDmaBufHeapHandle == nullptr) {
      QNN_ERROR("DmaAllocator not initialized");
      return nullptr;
    }

    dmaBufferAllocator = m_dmaBufCreate();
    if (dmaBufferAllocator == nullptr) {
      QNN_ERROR("DmaAllocator: nullptr returned for CreateDmabufHeapBufferAllocator().");
      return nullptr;
    }

    *fd = m_dmaBufAlloc(dmaBufferAllocator, "qcom,system", bufferSize, 0, 0);
    if ((*fd) < 0) {
      QNN_ERROR("DmaBufAlloc returned a invalid file descriptor = %d", static_cast<int>(*fd));
      return nullptr;
    }
  }

  void* memPointer = mmap(nullptr, bufferSize, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
  if (MAP_FAILED == memPointer) {
    QNN_ERROR("DmaAllocator: Unable to open file returned by DmaBufAlloc with mmap");
    return nullptr;
  }

  DmaBufferData* data =
      new DmaBufferData(dmaBufferAllocator, ionAllocatorFd, *fd, memPointer, bufferSize);

  return static_cast<void*>(data);
}

bool DmaAllocator::allocateBuffers() {
  if (!m_estimator) {
    QNN_ERROR("DmaAllocator: Estimator is null");
    return false;
  }

  uint32_t allocIdx = m_lastAllocIdx + 1;
  for (auto& [_, tensors] : m_estimator->getEstimations()) {
    // Since it is not working with Accumulated chunk sizes, will rely on adding individual length
    // memory.
    for (auto& [tensor_name, tensor_size] : tensors) {
      int fd     = -1;
      void* data = allocateBuffer(tensor_size, &fd);
      if (data == nullptr || fd == -1) {
        QNN_ERROR("DmaAllocator: mem alloc failed for tensor %s and fd %d.", tensor_name.c_str(), fd);
        return false;
      }
      m_tensorAllocInfo[tensor_name] = std::make_pair(allocIdx, tensor_size);
      m_buffers[allocIdx]            = static_cast<DmaBufferData*>(data);
      m_lastAllocIdx                 = allocIdx;
      allocIdx++;
    }
  }
  return true;
}

uint64_t DmaAllocator::allocate(uint64_t bufferSize) {
  int fd     = -1;
  void* data = allocateBuffer(bufferSize, &fd);
  if (data == nullptr || fd == -1) {
    QNN_ERROR("DmaAllocator: mem alloc failed for buffer of size %zu and fd %d.",
              static_cast<size_t>(bufferSize),
              fd);
    return 0;
  }
  m_lastAllocIdx++;
  m_buffers[m_lastAllocIdx] = static_cast<DmaBufferData*>(data);
  return m_lastAllocIdx;
}

void DmaAllocator::freeBuffer(uint64_t allocIdx) {
  if (!m_buffers.contains(allocIdx)) {
    QNN_ERROR("DmaAllocator: Invalid alloc Idx: %zu", static_cast<size_t>(allocIdx));
    return;
  }
  auto dmaBufferData = m_buffers[allocIdx];
  if (dmaBufferData == nullptr) {
    m_buffers.erase(allocIdx);
    return;
  }
  if (!dmaBufferData->memPointer) {
    QNN_ERROR("DmaBufAllocator: Nullptr recieved for memory with fd :%d", dmaBufferData->fd);
  }
  if (munmap(dmaBufferData->memPointer, dmaBufferData->totalBufferSize)) {
    QNN_ERROR("DmaAllocator: Unmap failed for memory with fd %d", dmaBufferData->fd);
  }
  if (m_useIonMemHandle) {
    close(dmaBufferData->fd);
    m_ionCloseFn(dmaBufferData->ionAllocatorFd);
  }
  else {
    if (m_dmaBufDeinit) {
      m_dmaBufDeinit(dmaBufferData->dmaBufferAllocator);
    }
    else {
      QNN_ERROR("DmaAllocator: DmaBuf Deinit function pointer is null");
    }
  }
  delete dmaBufferData;
  m_buffers.erase(allocIdx);
}
