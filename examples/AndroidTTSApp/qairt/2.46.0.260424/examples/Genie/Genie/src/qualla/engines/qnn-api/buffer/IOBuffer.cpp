//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#if !defined(_WIN32) && !defined(__QNX__)
#include "qualla/detail/buffer/Registration/DmaRegs.hpp"
#endif
#include "qualla/IOBuffer.hpp"
#include "qualla/detail/Log.hpp"
#include "qualla/detail/buffer/Registration/ClientRegs.hpp"
#include "qualla/detail/buffer/Registration/RpcRegs.hpp"

namespace qualla {

IOBuffer::IOBuffer(BufferType bufferAlloc, QNN_INTERFACE_VER_TYPE* qnnInterface)
    : m_initialize(false),
      m_event(IOEVENT::ALLOCATE_REGISTER_EVENT),
      m_bufferType(bufferAlloc),
      m_qnnInterface(qnnInterface) {
  m_name = "iobuffer" + std::to_string(s_ioCounter++);
}

bool IOBuffer::initialize(Qnn_ContextHandle_t contextHandle,
                          uint32_t dataAlignmentSize,
                          std::shared_ptr<Estimator> estimator) {
  m_contextHandle     = contextHandle;
  m_dataAlignmentSize = dataAlignmentSize;
  m_estimator         = estimator;
  if (true != initializeAllocator()) {
    QNN_ERROR("Failed to initialize buffer manager allocator");
    return false;
  }
  if (true != initializeRegistrar()) {
    QNN_ERROR("Failed to initialize buffer manager registrar");
    return false;
  }
  return true;
}

bool IOBuffer::initializeAllocator() {
  if (m_bufferType == BufferType::SHARED_BUFFER) {
#if !defined(QUALLA_ENGINE_QNN_HTP)
    return false;
#elif defined(__linux__) && defined(__x86_64__)
    m_allocator = std::shared_ptr<IBufferAlloc>(new ClientAllocator(m_estimator, true));
#else
    m_allocator = std::shared_ptr<IBufferAlloc>(new RpcAllocator(m_estimator, m_dataAlignmentSize));
#endif  // QUALLA_ENGINE_QNN_HTP
  } else if (m_bufferType == BufferType::DMABUF) {
#if defined(_WIN32) || !defined(QUALLA_ENGINE_QNN_GPU) || defined(__QNX__)
    return false;
#else
    m_allocator = std::shared_ptr<IBufferAlloc>(new DmaAllocator(m_estimator));
#endif  // QUALLA_ENGINE_QNN_GPU
  } else {
    m_allocator = std::shared_ptr<IBufferAlloc>(new ClientAllocator(m_estimator));
  }
  return true;
}

bool IOBuffer::initializeRegistrar() {
  if (m_bufferType == BufferType::SHARED_BUFFER) {
#if !defined(QUALLA_ENGINE_QNN_HTP)
    return false;
#elif defined(__linux__) && defined(__x86_64__)
    m_register  = std::shared_ptr<IBufferRegs>(
        new ClientRegs(std::dynamic_pointer_cast<ClientAllocator>(m_allocator)));
#else
    m_register  = std::shared_ptr<IBufferRegs>(new RpcRegs(
        m_contextHandle, m_qnnInterface, std::dynamic_pointer_cast<RpcAllocator>(m_allocator)));
#endif  // QUALLA_ENGINE_QNN_HTP
  } else if (m_bufferType == BufferType::DMABUF) {
#if defined(_WIN32) || !defined(QUALLA_ENGINE_QNN_GPU) || defined(__QNX__)
    return false;
#else
    m_register = std::shared_ptr<IBufferRegs>(new DmaRegs(
        m_contextHandle, m_qnnInterface, std::dynamic_pointer_cast<DmaAllocator>(m_allocator)));
#endif  // QUALLA_ENGINE_QNN_GPU
  } else {
    m_register = std::shared_ptr<IBufferRegs>(
        new ClientRegs(std::dynamic_pointer_cast<ClientAllocator>(m_allocator)));
  }
  if (true != m_register->initialize()) {
    return false;
  }
  m_initialize = true;
  return true;
}

IOBuffer::~IOBuffer() {
  // de allocates the memory.
  // De registration should have been completed by now, don't do it again.
  QNN_DEBUG("Destructing %s", m_name.c_str());
  m_allocator.reset();
  m_estimator.reset();
  m_event = IOEVENT::ALLOCATE_REGISTER_EVENT;
}

void IOBuffer::deRegisterAll() {
  QNN_DEBUG("Trying DeRegistration. %s", m_name.c_str());
  m_register.reset();
  if (!m_register) m_event = IOEVENT::REGISTER_EVENT;
}

BufferType IOBuffer::getBufferAllocType() { return m_bufferType; }

bool IOBuffer::isInitialize() { return m_initialize; }

bool IOBuffer::setEvent(IOEVENT event) {
  if (!m_initialize) {
    return false;
  }
  m_event = event;
  return true;
}

void* IOBuffer::getBuffer(Qnn_Tensor_t* tensor) { return m_register->getBuffer(tensor); }
void* IOBuffer::getBuffer(uint64_t allocIdx) { return m_allocator->getBuffer(allocIdx); }

int IOBuffer::getFd(Qnn_Tensor_t* tensor) { return m_register->getFd(tensor); }
int IOBuffer::getFd(uint64_t allocIdx) { return m_allocator->getFd(allocIdx); }

size_t IOBuffer::getOffset(Qnn_Tensor_t* tensor) { return m_register->getOffset(tensor); }
size_t IOBuffer::getOffset(uint64_t /*allocIdx*/) { return 0; }

size_t IOBuffer::getBufferSize(Qnn_Tensor_t* tensor) { return m_register->getBufferSize(tensor); }
size_t IOBuffer::getBufferSize(uint64_t allocIdx) { return m_allocator->getBufferSize(allocIdx); }

size_t IOBuffer::getTotalBufferSize(Qnn_Tensor_t* tensor) {
  return m_register->getTotalBufferSize(tensor);
}
size_t IOBuffer::getTotalBufferSize(uint64_t allocIdx) {
  return m_allocator->getTotalBufferSize(allocIdx);
}

void* IOBuffer::allocateTensorFusedBuffer(uint64_t bufferSize, int32_t* fd) {
  uint64_t allocIdx = m_allocator->allocate(bufferSize);
  *fd               = m_allocator->getFd(allocIdx);
  void* memPointer  = m_allocator->getBuffer(allocIdx);
  return memPointer;
}

uint64_t IOBuffer::allocate(uint64_t tensorDataSize) {
  return m_allocator->allocate(tensorDataSize);
}

bool IOBuffer::allocateBuffers() { return m_allocator->allocateBuffers(); }

bool IOBuffer::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
  return m_register->useSameMemory(dest, src);
}

bool IOBuffer::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src, int /*offset*/) {
  return m_register->useSameMemory(dest, src);
}

bool IOBuffer::useExternalMemory(Qnn_Tensor_t* dest, void* extMem) {
  return m_register->useExternalMemory(dest, extMem);
}

bool IOBuffer::beforeWriteToBuffer(Qnn_Tensor_t* tensor) {
  return m_register->beforeWriteToBuffer(tensor);
}

bool IOBuffer::afterWriteToBuffer(Qnn_Tensor_t* tensor) {
  return m_register->afterWriteToBuffer(tensor);
}

bool IOBuffer::beforeReadFromBuffer(Qnn_Tensor_t* tensor) {
  return m_register->beforeReadFromBuffer(tensor);
}

bool IOBuffer::afterReadFromBuffer(Qnn_Tensor_t* tensor) {
  return m_register->afterReadFromBuffer(tensor);
}

std::unordered_map<std::string, std::pair<uint64_t, size_t>>& IOBuffer::getAllocInfo() {
  return m_allocator->getTensorAllocInfo();
}

}  // namespace qualla
