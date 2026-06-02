//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "LPAI/QnnLpaiCommon.h"
#include "PAL/DynamicLoading.hpp"
#include "QnnSampleApp.hpp"
#include "QnnSampleAppConfigs.hpp"
#include "QnnSampleAppHelpers.hpp"

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                          uint32_t* numProviders);

typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
    const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

#define MAKE_MULTIPLE(i, m) (((i) % (m)) ? ((i) + (m) - ((i) % (m))) : (i))
#define MAX_ALIGNMENT 256
#define PAD_TO_VALUE(addr, alignment) ((((size_t)addr) + (alignment - 1)) & ~(alignment - 1))
#define IS_ALIGNED(addr, alignment)   ((((size_t)addr) & (alignment - 1)) == 0)

void print_help() {
  printf("QNN LPAI Sample App\n");
  printf(
      "--retrieve_context <path-to-context-binary>              path to qnn context binary file\n");
  printf("--backend <path-to-backendlib>                           path to LPAI backend lib\n");
  printf("--systemlib <path-to-qnnsystemlib>                       path to libQnnSystem.so\n");
  printf("-h                                                       print the help menu\n");
}

int parse_args(int argc, char* argv[], qnn_sample_app_context* appCtx) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0) {
      print_help();
      continue;
    } else if (strcmp(argv[i], "--retrieve_context") == 0 && i < (argc - 1)) {
      appCtx->contextBinaryPath = argv[++i];
      continue;
    } else if (strcmp(argv[i], "--backend") == 0 && (i < argc - 1)) {
      appCtx->backendLibPath = argv[++i];
      continue;
    } else if (strcmp(argv[i], "--systemlib") == 0 && (i < argc - 1)) {
      appCtx->qnnSystemLibPath = argv[++i];
      continue;
    } else {
      printf("Invalid argument %s\n", argv[i]);
      return -1;
    }
  }
  if (!appCtx->backendLibPath || !appCtx->qnnSystemLibPath || !appCtx->contextBinaryPath) {
    printf("Missing required argument\n");
    return -1;
  }
  return 0;
}

// default memory allocation/free implementation
// user is responsible for implementing island memory allocation/free
void* allocate_memory(size_t size, QnnLpaiMem_MemType_t memType) { return malloc(size); }

/**
 * @brief Allocates memory aligned to a specified address boundary.
 *
 * This function allocates a block of memory that is aligned to `startAddrAlignment` bytes.
 * It also ensures the allocated size is rounded up to the nearest multiple of `sizeAlignment`.
 * The original pointer returned by malloc is stored just before the aligned address,
 * allowing proper deallocation via `free_aligned_memory`.
 *
 * @param startAddrAlignment Alignment requirement for the returned address (must be ≤
 * MAX_ALIGNMENT).
 * @param sizeAlignment      Alignment requirement for the size of the allocation.
 * @param size               Requested size of the memory block.
 * @param memType            Memory type refer to QnnLpaiMem_MemType_t
 *
 * @return Pointer to aligned memory block, or NULL if allocation fails or alignment is invalid.
 *
 * @example
 *     void* ptr = allocate_aligned_memory(64, 16, 100, QNN_LPAI_MEM_TYPE_DDR);
 *     if (ptr) {
 *         // Use aligned memory
 *         ...
 *         free_aligned_memory(ptr, QNN_LPAI_MEM_TYPE_DDR);
 *     }
 */
void* allocate_aligned_memory(uint64_t startAddrAlignment,
                              uint64_t sizeAlignment,
                              size_t size,
                              QnnLpaiMem_MemType_t memType) {
  // `alignment` must be less than or equal to 256 since pAllocOffset is one byte
  if (startAddrAlignment > MAX_ALIGNMENT) {
    return NULL;
  }

  // Allocate extra space: alignment + sizeof(void*) to store original pointer
  size_t total_size = MAKE_MULTIPLE(size, sizeAlignment) + startAddrAlignment + sizeof(void*);
  void* raw         = allocate_memory(total_size, memType);
  if (!raw) return NULL;

  // Move past the space for storing the original pointer
  uintptr_t raw_addr     = (uintptr_t)raw + sizeof(void*);
  uintptr_t aligned_addr = PAD_TO_VALUE(raw_addr, startAddrAlignment);

  // Store the original pointer just before the aligned address
  void** storage = (void**)(aligned_addr - sizeof(void*));
  *storage       = raw;

  return (void*)aligned_addr;
}

void free_memory(void* memory_ptr, QnnLpaiMem_MemType_t memType) { free(memory_ptr); }

/**
 * @brief Frees memory previously allocated with `allocate_aligned_memory`.
 *
 * This function retrieves the original pointer stored just before the aligned address
 * and frees it using `free()`. This is necessary because the aligned pointer returned
 * to the user is offset from the actual pointer returned by `malloc`.
 *
 * @param memory_ptr Pointer to the aligned memory block returned by `allocate_aligned_memory`.
 * @param memType    Memory type refer to QnnLpaiMem_MemType_t.
 */
void free_aligned_memory(void* memory_ptr, QnnLpaiMem_MemType_t memType) {
  if (memory_ptr != NULL) {
    // Retrieve the original pointer stored just before the aligned address
    void** storage = (void**)((uintptr_t)memory_ptr - sizeof(void*));
    free_memory(*storage, memType);
  }
}

int allocate_tensors(Qnn_Tensor_t* qnnTensorArray,
                     Qnn_Tensor_t** qnnTensorArrayNew,
                     uint32_t numTensors,
                     QnnLpaiMem_MemType_t memType,
                     uint32_t startAddrAlignment,
                     uint32_t sizeAlignment) {
  if (qnnTensorArray == NULL || qnnTensorArrayNew == NULL) {
    printf("Invalid tensor array\n");
    return -1;
  }
  /* Allocate new tensor array */
  *qnnTensorArrayNew = (Qnn_Tensor_t*)allocate_memory(sizeof(Qnn_Tensor_t) * numTensors, memType);
  if (*qnnTensorArrayNew == NULL) {
    printf("Fail to allocate new tensor array\n");
    return -1;
  }
  memcpy(*qnnTensorArrayNew, qnnTensorArray, sizeof(Qnn_Tensor_t) * numTensors);

  /* allocate for qnn data buffer */
  Qnn_Tensor_t* qnnTensorsNew = *qnnTensorArrayNew;
  for (uint32_t i = 0; i < numTensors; i++) {
    Qnn_Tensor_t* qnnTensor = &qnnTensorsNew[i];
    qnnTensor->v1.memType   = QNN_TENSORMEMTYPE_RAW;
    int dataSize            = qnnApp_calculate_tensor_size(qnnTensor->v1);
    if (dataSize == -1) {
      printf("Fail to calculate tensor size for tensor\n");
      return -1;
    }
    qnnTensor->v1.clientBuf.data =
        allocate_aligned_memory(startAddrAlignment, sizeAlignment, dataSize, memType);
    if (qnnTensor->v1.clientBuf.data == NULL) {
      printf("Fail to malloc for tensor\n");
      return -1;
    }
    qnnTensor->v1.clientBuf.dataSize = dataSize;
  }

  return 0;
}

void free_tensors(Qnn_Tensor_t* qnnTensors, uint32_t numTensors, QnnLpaiMem_MemType_t memType) {
  for (uint32_t i = 0; i < numTensors; i++) {
    Qnn_Tensor_t qnnTensor = qnnTensors[i];
    if (qnnTensor.v1.clientBuf.data) {
      free_aligned_memory(qnnTensor.v1.clientBuf.data, memType);
    }
  }
  free_memory(qnnTensors, memType);
}

int config_memory(qnn_sample_app_context* appCtx) {
  uint32_t scratchSize = 0, persistentSize = 0;
  Qnn_ErrorHandle_t error = qnnApp_graphGetPropertyScratchMemSize(
      appCtx->graphHandle, &appCtx->lpaiInterface, &scratchSize);
  error |= qnnApp_graphGetPropertyPersistentMemSize(
      appCtx->graphHandle, &appCtx->lpaiInterface, &persistentSize);
  if (error != QNN_SUCCESS) {
    printf("Fail to query memory requirements\n");
    return -1;
  }
  /* Allocate and set scratch memory */
  if (scratchSize) {
    appCtx->scratchBuffer = allocate_memory(scratchSize, DEFAULT_MEM_TYPE);
    if (!appCtx->scratchBuffer) {
      printf("Fail to allocate scratch buffer\n");
      return -1;
    }
    error = qnnApp_graphSetConfigScratchMem(appCtx->graphHandle,
                                            &appCtx->lpaiInterface,
                                            scratchSize,
                                            appCtx->scratchBuffer,
                                            DEFAULT_MEM_TYPE);
    if (error != QNN_SUCCESS) {
      printf("Fail to set scratch buffer\n");
      return -1;
    }
  }
  /* Allocate and set persistent memory */
  if (persistentSize) {
    appCtx->persistentBuffer = allocate_memory(persistentSize, DEFAULT_MEM_TYPE);
    if (!appCtx->persistentBuffer) {
      printf("Fail to allocate persistent buffer\n");
      return -1;
    }
    error = qnnApp_graphSetConfigPersistentMem(appCtx->graphHandle,
                                               &appCtx->lpaiInterface,
                                               persistentSize,
                                               appCtx->persistentBuffer,
                                               DEFAULT_MEM_TYPE);
    if (error != QNN_SUCCESS) {
      printf("Fail to set persistent buffer\n");
      return -1;
    }
  }
  return 0;
}

void cleanup_memory(qnn_sample_app_context* appCtx) {
  /* Free context binary buffer */
  if (appCtx->contextBinaryBuffer) {
    free_aligned_memory(appCtx->contextBinaryBuffer, DEFAULT_MEM_TYPE);
  }
  /* Free rt memory buffers */
  if (appCtx->scratchBuffer) {
    free_memory(appCtx->scratchBuffer, DEFAULT_MEM_TYPE);
  }
  if (appCtx->persistentBuffer) {
    free_memory(appCtx->persistentBuffer, DEFAULT_MEM_TYPE);
  }
  if (appCtx->inputs) {
    free_tensors(appCtx->inputs, appCtx->numInputs, DEFAULT_MEM_TYPE);
  }
  if (appCtx->outputs) {
    free_tensors(appCtx->outputs, appCtx->numOutputs, DEFAULT_MEM_TYPE);
  }
}

#define GET_GRAPH_INFOS(binaryInfo)                                  \
  ((binaryInfo)->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2 \
       ? (binaryInfo)->contextBinaryInfoV2.graphs                    \
       : (binaryInfo)->contextBinaryInfoV1.graphs)

#define GET_GRAPH_INPUT_TENSORS(graphInfo)                        \
  (graphInfo)->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2 \
      ? (graphInfo)->graphInfoV2.graphInputs                      \
      : (graphInfo)->graphInfoV1.graphInputs;

#define GET_GRAPH_OUTPUT_TENSORS(graphInfo)                       \
  (graphInfo)->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2 \
      ? (graphInfo)->graphInfoV2.graphOutputs                     \
      : (graphInfo)->graphInfoV1.graphOutputs;

#define GET_GRAPH_NUM_INPUT_TENSORS(graphInfo)                    \
  (graphInfo)->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2 \
      ? (graphInfo)->graphInfoV2.numGraphInputs                   \
      : (graphInfo)->graphInfoV1.numGraphInputs;

#define GET_GRAPH_NUM_OUTPUT_TENSORS(graphInfo)                   \
  (graphInfo)->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2 \
      ? (graphInfo)->graphInfoV2.numGraphOutputs                  \
      : (graphInfo)->graphInfoV1.numGraphOutputs;

int qnnApp_getLpaiInterfaces(const char* backendLibPath,
                             void** backendLibHandle,
                             QNN_INTERFACE_VER_TYPE* lpaiInterface) {
  /* Load backend library */
  *backendLibHandle = pal::dynamicloading::dlOpen(
      backendLibPath, pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_GLOBAL);
  if (*backendLibHandle == NULL) {
    printf("Fail to load backend library\n");
    return -1;
  }
  /* Load the function to retrieve providers */
  QnnInterfaceGetProvidersFn_t getProviders =
      (QnnInterfaceGetProvidersFn_t)pal::dynamicloading::dlSym(*backendLibHandle,
                                                               "QnnInterface_getProviders");
  if (getProviders == NULL) {
    printf("Fail to load QnnInterface_getProviders\n");
    return -1;
  }
  /* Retrieve providers */
  const QnnInterface_t** interfaceProviders = NULL;
  uint32_t numProviders                     = 0;
  Qnn_ErrorHandle_t error                   = getProviders(&interfaceProviders, &numProviders);
  if (error) {
    printf("Fail to retrieve providers\n");
    return -1;
  }
  if (interfaceProviders == NULL) {
    printf("Failed to get interface providers: null interface providers received.\n");
    return -1;
  }
  if (numProviders == 0) {
    printf("Failed to get interface providers: 0 interface providers.\n");
    return -1;
  }
  // Loop through all available interface providers and pick the one that suits the current API
  // version
  bool foundValidInterface = false;
  for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
    if (interfaceProviders[pIdx]->backendId == QNN_BACKEND_ID_LPAI) {
      *lpaiInterface      = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
      foundValidInterface = true;
      break;
    }
  }
  if (!foundValidInterface) {
    printf("Fail to find valid LPAI interface\n");
    return -1;
  }
  return 0;
}

int qnnApp_getQnnSystemInterface(const char* qnnSystemLibPath,
                                 void** qnnSystemLibHandle,
                                 QNN_SYSTEM_INTERFACE_VER_TYPE* qnnSystemInterface) {
  *qnnSystemLibHandle = pal::dynamicloading::dlOpen(
      qnnSystemLibPath, pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_GLOBAL);
  if (*qnnSystemLibHandle == NULL) {
    printf("Fail to load qnn system library\n");
    return -1;
  }
  /* Load the function to retrieve providers */
  QnnSystemInterfaceGetProvidersFn_t getProviders =
      (QnnSystemInterfaceGetProvidersFn_t)pal::dynamicloading::dlSym(
          *qnnSystemLibHandle, "QnnSystemInterface_getProviders");
  if (getProviders == NULL) {
    printf("Fail to load QnnSystemInterface_getProviders\n");
    return -1;
  }
  /* Retrieve providers */
  const QnnSystemInterface_t** interfaceProviders = NULL;
  uint32_t numProviders                           = 0;
  Qnn_ErrorHandle_t error = getProviders(&interfaceProviders, &numProviders);
  if (error) {
    printf("Fail to retrieve providers\n");
    return -1;
  }
  if (interfaceProviders == NULL) {
    printf("Failed to get interface providers: null interface providers received.\n");
    return -1;
  }
  if (numProviders == 0) {
    printf("Failed to get interface providers: 0 interface providers.\n");
    return -1;
  }
  *qnnSystemInterface = interfaceProviders[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
  return 0;
}

int qnnApp_loadContextBinary(const char* contextBinaryFilePath,
                             void* contextBinaryBuffer,
                             uint32_t contextBinaryBufferSize) {
  FILE* file = fopen(contextBinaryFilePath, "rb");
  if (!file) {
    printf("Fail to open context binary file\n");
    return -1;
  }
  int size = fread(contextBinaryBuffer, 1, contextBinaryBufferSize, file);
  if (size != contextBinaryBufferSize) {
    printf("Fail to read context binary file\n");
    fclose(file);
    return -1;
  }
  fclose(file);
  return 0;
}

int qnnApp_getGraphInfo(QNN_SYSTEM_INTERFACE_VER_TYPE* qnnSystemInterface,
                        QnnSystemContext_Handle_t qnnSystemCtxHandle,
                        void* contextBinaryBuffer,
                        uint32_t contextBinaryBufferSize,
                        QnnSystemContext_GraphInfo_t** graphInfo) {
  /* Extract QNN binaryInfo */
  const QnnSystemContext_BinaryInfo_t* binaryInfo;
  Qnn_ContextBinarySize_t binaryInfoSize;
  Qnn_ErrorHandle_t error = qnnSystemInterface->systemContextGetBinaryInfo(qnnSystemCtxHandle,
                                                                           contextBinaryBuffer,
                                                                           contextBinaryBufferSize,
                                                                           &binaryInfo,
                                                                           &binaryInfoSize);
  if (error != QNN_SUCCESS) {
    printf("Received error when getting binary info\n");
    return -1;
  }
  /* Extract graphInfo from QNN binaryInfo, assume only one graph in the context */
  QnnSystemContext_GraphInfo_t* graphInfos = GET_GRAPH_INFOS(binaryInfo);
  *graphInfo                               = &(graphInfos[0]);
  return 0;
}

int qnnApp_getGraphIO(QnnSystemContext_GraphInfo_t* graphInfo,
                      Qnn_Tensor_t** inputs,
                      Qnn_Tensor_t** outputs,
                      uint32_t* numInputs,
                      uint32_t* numOutputs) {
  *inputs     = GET_GRAPH_INPUT_TENSORS(graphInfo);
  *outputs    = GET_GRAPH_OUTPUT_TENSORS(graphInfo);
  *numInputs  = GET_GRAPH_NUM_INPUT_TENSORS(graphInfo);
  *numOutputs = GET_GRAPH_NUM_OUTPUT_TENSORS(graphInfo);
  return 0;
}

int qnnApp_freeLpaiBackend(QNN_INTERFACE_VER_TYPE* lpaiInterface,
                           QNN_SYSTEM_INTERFACE_VER_TYPE* qnnSystemInterface,
                           void* backendLibHandle,
                           void* qnnSystemLibHandle,
                           Qnn_BackendHandle_t backendHandle,
                           Qnn_ContextHandle_t contextHandle,
                           QnnSystemContext_Handle_t qnnSystemCtxHandle) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  int ret                 = 0;
  /* Free context */
  if (contextHandle) {
    error = lpaiInterface->contextFree(contextHandle, NULL);
    if (error != QNN_SUCCESS) {
      printf("Received error when freeing context\n");
      ret = -1;
    }
  }
  /* Free Backend */
  if (backendHandle) {
    error = lpaiInterface->backendFree(backendHandle);
    if (error != QNN_SUCCESS) {
      printf("Received error when freeing backend\n");
      ret = -1;
    }
  }
  /* Free qnn system context */
  if (qnnSystemCtxHandle) {
    error = qnnSystemInterface->systemContextFree(qnnSystemCtxHandle);
    if (error != QNN_SUCCESS) {
      printf("Received error when freeing qnn system context\n");
      ret = -1;
    }
  }
  /* Unload backend lib */
  if (backendLibHandle) {
    pal::dynamicloading::dlClose(backendLibHandle);
  }
  /* Unload qnn system context lib */
  if (qnnSystemLibHandle) {
    pal::dynamicloading::dlClose(qnnSystemLibHandle);
  }
  return ret;
}

int qnnApp_get_file_size(const char* filePath) {
  FILE* file = fopen(filePath, "rb");
  if (!file) {
    printf("Could not open file\n");
    return -1;
  }
  int ret = fseek(file, 0L, SEEK_END);
  if (ret == -1) {
    printf("Fail to search file\n");
    fclose(file);
    return -1;
  }

  int size = ftell(file);
  fclose(file);
  return size;
}

static long get_datatype_size(Qnn_DataType_t dataType) {
  switch (dataType) {
    case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      return sizeof(int8_t);
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return sizeof(int16_t);
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return sizeof(int32_t);
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
      return sizeof(int64_t);
    default:
      printf("INVALID DATATYPE %d\n", dataType);
      return -1;
  }
}

long qnnApp_calculate_tensor_size(Qnn_TensorV1_t tensor) {
  long size = get_datatype_size(tensor.dataType);
  if (size == -1) {
    return -1;
  }
  for (uint32_t i = 0; i < tensor.rank; i++) {
    size *= tensor.dimensions[i];
  }
  return size;
}