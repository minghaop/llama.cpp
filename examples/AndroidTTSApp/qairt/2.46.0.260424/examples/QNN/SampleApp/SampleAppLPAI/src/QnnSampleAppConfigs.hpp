//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#pragma once

#include <stdbool.h>

#include "QnnGraph.h"
#include "QnnInterface.h"
#include "LPAI/QnnLpaiBackend.h"
#include "LPAI/QnnLpaiGraph.h"
#include "LPAI/QnnLpaiMem.h"

/**
 * Set scratch memory
 *
 * @param [in] graphHandle Graph Handle
 *
 * @param [in] lpaiInterface LPAI interface
 *
 * @param [in] scratchSize scratch memory size
 *
 * @param [in] scratchBuffer scratch buffer
 *
 * @param [in] memType scratch buffer memory type
 *
 * Return 0 on success, return -1 on failure
 */
Qnn_ErrorHandle_t qnnApp_graphSetConfigScratchMem(Qnn_GraphHandle_t graphHandle,
                                                  QNN_INTERFACE_VER_TYPE* lpaiInterface,
                                                  uint32_t scratchSize,
                                                  void* scratchBuffer,
                                                  QnnLpaiMem_MemType_t memType);

/**
 * Set persistent memory
 *
 * @param [in] graphHandle Graph Handle
 *
 * @param [in] lpaiInterface LPAI interface
 *
 * @param [in] persistentSize persistent memory size
 *
 * @param [in] persistentBuffer persistent buffer
 *
 * @param [in] memType persistent buffer memory type
 *
 * Return 0 on success, return -1 on failure
 */
Qnn_ErrorHandle_t qnnApp_graphSetConfigPersistentMem(Qnn_GraphHandle_t graphHandle,
                                                     QNN_INTERFACE_VER_TYPE* lpaiInterface,
                                                     uint32_t persistentSize,
                                                     void* persistentBuffer,
                                                     QnnLpaiMem_MemType_t memType);

/**
 * Set perf configs
 *
 * @param [in] graphHandle Graph Handle
 *
 * @param [in] lpaiInterface LPAI interface
 *
 * @param [in] fps number of frames to second to be processed
 *
 * @param [in] ftrtRatio faster than real time processing request : multiples of 0.1 factors
 *
 * @param [in] isRealTime indicate if a client is real-time
 *
 * Return 0 on success, return -1 on failure
 */
Qnn_ErrorHandle_t qnnApp_graphSetConfigPerf(Qnn_GraphHandle_t graphHandle,
                                            QNN_INTERFACE_VER_TYPE* lpaiInterface,
                                            uint32_t fps,
                                            uint32_t ftrtRatio,
                                            bool isRealTime);

/**
 * Set core affinity configs
 *
 * @param [in] graphHandle Graph Handle
 *
 * @param [in] lpaiInterface LPAI interface
 *
 * @param [in] coreAffinity soft/hard
 *
 * @param [in] coreSel bit mask, each bit corresponds to a core, set the bit to use the core
 *
 * Return 0 on success, return -1 on failure
 */
Qnn_ErrorHandle_t qnnApp_graphSetConfigCoreAffinity(Qnn_GraphHandle_t graphHandle,
                                                    QNN_INTERFACE_VER_TYPE* lpaiInterface,
                                                    QnnLpaiGraph_CoreAffinityType_t coreAffinity,
                                                    uint32_t coreSel);

/**
 * Set graph priority config
 *
 * @param [in] graphHandle Graph Handle
 *
 * @param [in] lpaiInterface LPAI interface
 *
 * @param [in] priority graph priority
 *
 * Return 0 on success, return -1 on failure
 */
Qnn_ErrorHandle_t qnnApp_graphSetConfigPriority(Qnn_GraphHandle_t graphHandle,
                                                QNN_INTERFACE_VER_TYPE* lpaiInterface,
                                                Qnn_Priority_t priority);

/**
 * Get scratch memory size requirement
 *
 * @param [in] graphHandle Graph Handle
 *
 * @param [in] lpaiInterface LPAI interface
 *
 * @param [out] scratchSize scratch memory size requirement
 *
 * Return 0 on success, return -1 on failure
 */
Qnn_ErrorHandle_t qnnApp_graphGetPropertyScratchMemSize(Qnn_GraphHandle_t graphHandle,
                                                        QNN_INTERFACE_VER_TYPE* lpaiInterface,
                                                        uint32_t* scratchSize);

/**
 * Get persistent memory size requirement
 *
 * @param [in] graphHandle Graph Handle
 *
 * @param [in] lpaiInterface LPAI interface
 *
 * @param [out] persistentSize persistent memory size requirement
 *
 * Return 0 on success, return -1 on failure
 */
Qnn_ErrorHandle_t qnnApp_graphGetPropertyPersistentMemSize(Qnn_GraphHandle_t graphHandle,
                                                           QNN_INTERFACE_VER_TYPE* lpaiInterface,
                                                           uint32_t* persistentSize);

/**
 * Get buffer alignment requirements
 *
 * @param [in] backendHandle Backend Handle
 *
 * @param [in] lpaiInterface LPAI interface
 *
 * @param [out] startAddrAlignment start address alignment requirement
 *
 * @param [out] sizeAlignment size alignment requirement
 *
 * Return 0 on success, return -1 on failure
 */
Qnn_ErrorHandle_t qnnApp_backendGetPropertyAlignmentReq(Qnn_BackendHandle_t backendHandle,
                                                        QNN_INTERFACE_VER_TYPE* lpaiInterface,
                                                        uint32_t* startAddrAlignment,
                                                        uint32_t* sizeAlignment);

/**
 * Check if persistent binary is required
 *
 * @param [in] backendHandle Backend Handle
 *
 * @param [in] lpaiInterface LPAI interface
 *
 * @param [out] requirePersistentBinary persistent binary requirement
 *
 * Return 0 on success, return -1 on failure
 */
Qnn_ErrorHandle_t qnnApp_backendGetPropertyPersistentBinary(Qnn_BackendHandle_t backendHandle,
                                                            QNN_INTERFACE_VER_TYPE* lpaiInterface,
                                                            bool* requirePersistentBinary);
