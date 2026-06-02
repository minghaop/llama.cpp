//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#pragma once

#include "QnnInterface.h"
#include "LPAI/QnnLpaiMem.h"
#include "System/QnnSystemInterface.h"

#define DEFAULT_MEM_TYPE QNN_LPAI_MEM_TYPE_DDR

typedef struct qnn_sample_app_context {
  const char* backendLibPath;
  const char* qnnSystemLibPath;
  const char* contextBinaryPath;

  void* backendLibHandle;
  void* qnnSystemLibHandle;

  QNN_INTERFACE_VER_TYPE lpaiInterface;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface;

  Qnn_BackendHandle_t backendHandle;
  Qnn_ContextHandle_t contextHandle;
  QnnSystemContext_Handle_t qnnSystemCtxHandle;
  Qnn_GraphHandle_t graphHandle;

  void* contextBinaryBuffer;
  uint32_t contextBinaryBufferSize;

  QnnSystemContext_GraphInfo_t* graphInfo;

  void* scratchBuffer;
  void* persistentBuffer;

  Qnn_Tensor_t* inputs;
  Qnn_Tensor_t* outputs;

  uint32_t numInputs;
  uint32_t numOutputs;
} qnn_sample_app_context;

#define QNN_SAMPLE_APP_CONTEXT_INIT                                       \
  {                                                                       \
    NULL,                                   /* backendLibPath */          \
        NULL,                               /* qnnSystemLibPath */        \
        NULL,                               /* contextBinaryPath */       \
        NULL,                               /* backendLibHandle */        \
        NULL,                               /* qnnSystemLibHandle */      \
        QNN_INTERFACE_VER_TYPE_INIT,        /* lpaiInterface */           \
        QNN_SYSTEM_INTERFACE_VER_TYPE_INIT, /* qnnSystemInterface */      \
        NULL,                               /* backendHandle */           \
        NULL,                               /* contextHandle */           \
        NULL,                               /* qnnSystemCtxHandle */      \
        NULL,                               /* graphHandle */             \
        NULL,                               /* contextBinaryBuffer */     \
        0u,                                 /* contextBinaryBufferSize */ \
        NULL,                               /* graphInfo */               \
        NULL,                               /* scratchBuffer */           \
        NULL,                               /* persistentBuffer */        \
        NULL,                               /* inputs */                  \
        NULL,                               /* outputs */                 \
        0u,                                 /* numInputs */               \
        0u                                  /* numOutputs */              \
  }

int parse_args(int argc, char* argv[], qnn_sample_app_context* appCtx);

void* allocate_memory(size_t size, QnnLpaiMem_MemType_t memType);

void* allocate_aligned_memory(uint64_t startAddrAlignment,
                              uint64_t sizeAlignment,
                              size_t size,
                              QnnLpaiMem_MemType_t memType);

void free_memory(void* memory_ptr, QnnLpaiMem_MemType_t memType);

void free_aligned_memory(void* memory_ptr, QnnLpaiMem_MemType_t memType);

int allocate_tensors(Qnn_Tensor_t* qnnTensorArray,
                     Qnn_Tensor_t** qnnTensorArrayNew,
                     uint32_t numTensors,
                     QnnLpaiMem_MemType_t memType,
                     uint32_t startAddrAlignment,
                     uint32_t sizeAlignment);

void free_tensors(Qnn_Tensor_t* qnnTensors, uint32_t numTensors, QnnLpaiMem_MemType_t memType);

int config_memory(qnn_sample_app_context* appCtx);

void cleanup_memory(qnn_sample_app_context* appCtx);