//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#pragma once

#include <stdbool.h>

#include "QnnInterface.h"
#include "LPAI/QnnLpaiGraph.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"


/**
 * Get lpai nonisland and island interface
 *
 * @param [in] backendLibPath Path to backend lib
 *
 * @param [out] backendLibHandle Backend lib handle
 *
 * @param [out] lpaiInterface LPAI nonisland interface
 *
 * Return 0 on success, return -1 on failure
 */
int qnnApp_getLpaiInterfaces(const char* backendLibPath,
                             void** backendLibHandle,
                             QNN_INTERFACE_VER_TYPE* lpaiInterface);

/**
 * Get qnn system interface
 *
 * @param [in] qnnSystemLibPath Path to qnn system lib
 *
 * @param [out] qnnSystemLibHandle Qnn system lib handle
 *
 * @param [out] qnnSystemInterface Qnn system interface
 *
 * Return 0 on success, return -1 on failure
 */
int qnnApp_getQnnSystemInterface(const char* qnnSystemLibPath,
                                 void** qnnSystemLibHandle,
                                 QNN_SYSTEM_INTERFACE_VER_TYPE* qnnSystemInterface);

/**
 * Load context binary file
 *
 * @param [in] contextBinaryFilePath Path to context binary file
 *
 * @param [in] contextBinaryBuffer Context binary buffer
 *
 * @param [in] contextBinaryBufferSize Context binary buffer size
 *
 * Return 0 on success, return -1 on failure
 */
int qnnApp_loadContextBinary(const char* contextBinaryFilePath,
                             void* contextBinaryBuffer,
                             uint32_t contextBinaryBufferSize);

/**
 * Retrieve graph info from context binary
 *
 * @param [in] qnnSystemInterface Qnn system interface
 *
 * @param [in] qnnSystemCtxHandle Qnn system context handle
 *
 * @param [in] contextBinaryBuffer Context binary buffer
 *
 * @param [in] contextBinaryBufferSize Context binary buffer size
 *
 * @param [out] graphInfo Qnn graph info
 *
 * Return 0 on success, return -1 on failure
 */
int qnnApp_getGraphInfo(QNN_SYSTEM_INTERFACE_VER_TYPE* qnnSystemInterface,
                        QnnSystemContext_Handle_t qnnSystemCtxHandle,
                        void* contextBinaryBuffer,
                        uint32_t contextBinaryBufferSize,
                        QnnSystemContext_GraphInfo_t** graphInfo);

/**
 * Retrieve graph io info from graphInfo
 *
 * @param [in] graphInfo Qnn graph info
 *
 * @param [out] inputs Array of input tensors
 *
 * @param [out] outputs Array of output tensors
 *
 * @param [out] numInputs Number of input tensors
 *
 * @param [out] numOutputs Number of output tensors
 *
 * Return 0 on success, return -1 on failure
 */
int qnnApp_getGraphIO(QnnSystemContext_GraphInfo_t* graphInfo,
                      Qnn_Tensor_t** inputs,
                      Qnn_Tensor_t** outputs,
                      uint32_t* numInputs,
                      uint32_t* numOutputs);

/**
 * Free backend resources
 *
 * Return 0 on success, return -1 on failure
 */
int qnnApp_freeLpaiBackend(QNN_INTERFACE_VER_TYPE* lpaiInterface,
                           QNN_SYSTEM_INTERFACE_VER_TYPE* qnnSystemInterface,
                           void* backendLibHandle,
                           void* qnnSystemLibHandle,
                           Qnn_BackendHandle_t backendHandle,
                           Qnn_ContextHandle_t contextHandle,
                           QnnSystemContext_Handle_t qnnSystemCtxHandle);

int qnnApp_get_file_size(const char* filePath);

long qnnApp_calculate_tensor_size(Qnn_TensorV1_t tensor);