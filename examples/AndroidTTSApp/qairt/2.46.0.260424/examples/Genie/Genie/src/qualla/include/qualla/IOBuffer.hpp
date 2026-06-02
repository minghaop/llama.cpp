//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "QnnContext.h"
#include "QnnInterface.h"
#include "qualla/detail/buffer/Allocator/IBufferAlloc.hpp"
#include "qualla/detail/buffer/Estimator.hpp"
#include "qualla/detail/buffer/Registration/IBufferRegs.hpp"

namespace qualla {

static uint64_t s_ioCounter{0};

enum class IOEVENT : uint8_t {
  NO_EVENT                = 0,
  ALLOCATE_EVENT          = 1,
  REGISTER_EVENT          = 2,
  ALLOCATE_REGISTER_EVENT = 3
};

static std::unordered_map<IOEVENT, std::string> ioEventMap = {
  {IOEVENT::NO_EVENT, "IO_NO_EVENT"},
  {IOEVENT::ALLOCATE_EVENT, "IO_ALLOCATE_EVENT"},
  {IOEVENT::REGISTER_EVENT, "IO_REGISTER_EVENT"},
  {IOEVENT::ALLOCATE_REGISTER_EVENT, "IO_ALLOCATE_REGISTER_EVENT"}
};

enum class BufferType : uint8_t {
  INVALID       = 0,
  DEFAULT       = 1,  // malloc based allocator
  SHARED_BUFFER = 2,  // shared buffer allocator; actual allocator depends on the platform
  DMABUF        = 3,  // dma buffer allocator
};

class IOBuffer {
 public:
  IOBuffer(BufferType bufferAlloc, QNN_INTERFACE_VER_TYPE* qnnInterface);

  // for HTP
  // IoBuffer --> IoTensor -->KVManager (Except Encoder)
  // IoBuffer --> IoTensor (Encoder)
  virtual ~IOBuffer();

  // Copy constructor and copy-assignment:
  IOBuffer(const IOBuffer& other)
      : m_name(other.m_name + "_copy"),  // Or decide a better naming scheme
        m_initialize(other.m_initialize),
        m_event(other.m_event),
        m_bufferType(other.m_bufferType),
        m_dataAlignmentSize(other.m_dataAlignmentSize),
        m_qnnInterface(other.m_qnnInterface),
        m_contextHandle(other.m_contextHandle),
        m_allocator(other.m_allocator),  // shared_ptr, so shares ownership
        m_register(other.m_register),    // shared_ptr, so shares ownership
        m_estimator(other.m_estimator)   // shared_ptr, so shares ownership
  {}

  IOBuffer& operator=(const IOBuffer& other) {
    if (this != &other) {
      m_name              = other.m_name + "_copy";  // Or another naming approach
      m_initialize        = other.m_initialize;
      m_event             = other.m_event;
      m_bufferType        = other.m_bufferType;
      m_dataAlignmentSize = other.m_dataAlignmentSize;
      m_qnnInterface      = other.m_qnnInterface;
      m_contextHandle     = other.m_contextHandle;
      m_allocator         = other.m_allocator;
      m_register          = other.m_register;
      m_estimator         = other.m_estimator;
    } else {
      m_event = IOEVENT::NO_EVENT;
    }
    return *this;
  }

  // Delete move constructor
  IOBuffer(IOBuffer&&)            = delete;
  IOBuffer& operator=(IOBuffer&&) = delete;

  BufferType getBufferAllocType();
  bool isInitialize();
  bool setEvent(IOEVENT event);
  virtual void randomFn(){};

  bool initialize(Qnn_ContextHandle_t contextHandle    = nullptr,
                  uint32_t dataAlignmentSize           = 0,
                  std::shared_ptr<Estimator> estimator = nullptr);
  bool initializeAllocator();
  bool initializeRegistrar();
  void deRegisterAll();

  void* getBuffer(Qnn_Tensor_t* tensor);
  void* getBuffer(uint64_t allocIdx);

  int getFd(Qnn_Tensor_t* tensor);
  int getFd(uint64_t allocIdx);

  size_t getOffset(Qnn_Tensor_t* tensor);
  size_t getOffset(uint64_t allocIdx);

  size_t getBufferSize(Qnn_Tensor_t* tensor);
  size_t getBufferSize(uint64_t allocIdx);

  size_t getTotalBufferSize(Qnn_Tensor_t* tensor);
  size_t getTotalBufferSize(uint64_t allocIdx);

  void* allocateTensorFusedBuffer(uint64_t bufferSize, int32_t* fd);
  uint64_t allocate(uint64_t tensorDataSize);
  bool allocateBuffers();

  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src);
  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src, int offset);

  bool useExternalMemory(Qnn_Tensor_t* dest, void* extMem);

  std::unordered_map<std::string, std::pair<uint64_t, size_t>>& getAllocInfo();

  // Functions to sync memory buffers for Read/Write using DmaBuf.
  bool beforeWriteToBuffer(Qnn_Tensor_t* tensor);
  bool afterWriteToBuffer(Qnn_Tensor_t* tensor);
  bool beforeReadFromBuffer(Qnn_Tensor_t* tensor);
  bool afterReadFromBuffer(Qnn_Tensor_t* tensor);

  std::string m_name{};
  bool m_initialize{false};
  IOEVENT m_event{IOEVENT::NO_EVENT};
  BufferType m_bufferType{BufferType::INVALID};
  uint32_t m_dataAlignmentSize{0};

  QNN_INTERFACE_VER_TYPE* m_qnnInterface{nullptr};
  Qnn_ContextHandle_t m_contextHandle{nullptr};

  std::shared_ptr<IBufferAlloc> m_allocator;
  std::shared_ptr<IBufferRegs> m_register;
  std::shared_ptr<Estimator> m_estimator;
};

}  // namespace qualla
