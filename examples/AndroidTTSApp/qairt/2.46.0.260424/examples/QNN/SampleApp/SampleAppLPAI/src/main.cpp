//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <stdio.h>
#include <string.h>

#include "LPAI/QnnLpaiMem.h"
#include "QnnSampleApp.hpp"
#include "QnnSampleAppConfigs.hpp"
#include "QnnSampleAppHelpers.hpp"

#define GET_GRAPH_NAME(graphInfo)                                 \
  (graphInfo)->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2 \
      ? (graphInfo)->graphInfoV2.graphName                        \
      : (graphInfo)->graphInfoV1.graphName;

int main(int argc, char* argv[]) {
  qnn_sample_app_context appCtx = QNN_SAMPLE_APP_CONTEXT_INIT;

  int ret = parse_args(argc, argv, &appCtx);
  if (ret) {
    return ret;
  }

  uint32_t buffer_start_addr_alignment = 1;
  uint32_t buffer_size_alignment       = 1;

  do {
    /*
     * Get LPAI interfaces
     */
    ret = qnnApp_getLpaiInterfaces(
        appCtx.backendLibPath, &appCtx.backendLibHandle, &appCtx.lpaiInterface);
    if (ret) {
      break;
    }

    /* Get QNN system interface */
    ret = qnnApp_getQnnSystemInterface(
        appCtx.qnnSystemLibPath, &appCtx.qnnSystemLibHandle, &appCtx.qnnSystemInterface);
    if (ret) {
      break;
    }

    /* Create backend */
    Qnn_ErrorHandle_t error = appCtx.lpaiInterface.backendCreate(NULL, NULL, &appCtx.backendHandle);
    if (error != QNN_SUCCESS) {
      printf("Received error when creating backend\n");
      ret = -1;
      break;
    }

    /* Query buffer alignment requirements */
    error = qnnApp_backendGetPropertyAlignmentReq(appCtx.backendHandle,
                                                  &appCtx.lpaiInterface,
                                                  &buffer_start_addr_alignment,
                                                  &buffer_size_alignment);
    if (error != QNN_SUCCESS) {
      printf("Received error when querying buffer alignment requirements\n");
      ret = -1;
      break;
    }

    /* Allocate memory for context binary */
    appCtx.contextBinaryBufferSize = qnnApp_get_file_size(appCtx.contextBinaryPath);
    if (appCtx.contextBinaryBufferSize == 0) {
      printf("Invalid context binary file\n");
      ret = -1;
      break;
    }
    /* Context binary buffer must be buffer_start_addr_alignment aligned */
    appCtx.contextBinaryBuffer = allocate_aligned_memory(buffer_start_addr_alignment,
                                                         buffer_size_alignment,
                                                         appCtx.contextBinaryBufferSize,
                                                         DEFAULT_MEM_TYPE);
    if (!appCtx.contextBinaryBuffer) {
      printf("Failed to allocate context binary buffer\n");
      ret = -1;
      break;
    }
    /* Load context binary buffer */
    ret = qnnApp_loadContextBinary(
        appCtx.contextBinaryPath, appCtx.contextBinaryBuffer, appCtx.contextBinaryBufferSize);
    if (ret) {
      printf("Failed to load context binary\n");
      break;
    }

    /* Create context from serialized binary */
    QnnContext_Config_t persistentBinaryConfig;
    persistentBinaryConfig.option             = QNN_CONTEXT_CONFIG_PERSISTENT_BINARY;
    persistentBinaryConfig.isPersistentBinary = 0;
    QnnContext_Config_t* contextConfigPtrs[2] = {0};  // contextConfigPtrs[1] is nullptr
    contextConfigPtrs[0]                      = &persistentBinaryConfig;
    /* Query if persistent binary is required, if no, context binary buffer can be freed after
     * contextCreateFromBinary */
    /* if yes, context binary buffer need to be persistent until contextFree, also need to pass the
     * persistent binary context config to contextCreateFromBinary */
    bool requirePersistentBinary = false;
    error                        = qnnApp_backendGetPropertyPersistentBinary(
        appCtx.backendHandle, &appCtx.lpaiInterface, &requirePersistentBinary);
    if (error != QNN_SUCCESS) {
      printf("Received error when querying persistent binary requirement\n");
      ret = -1;
      break;
    }
    if (requirePersistentBinary) {
      persistentBinaryConfig.isPersistentBinary = 1;
    }
    error = appCtx.lpaiInterface.contextCreateFromBinary(
        appCtx.backendHandle,
        NULL,
        requirePersistentBinary ? (const QnnContext_Config_t**)(contextConfigPtrs) : NULL,
        appCtx.contextBinaryBuffer,
        appCtx.contextBinaryBufferSize,
        &appCtx.contextHandle,
        NULL);
    if (error != QNN_SUCCESS) {
      printf("Received error when creating context\n");
      ret = -1;
      break;
    }

    /* Create qnn system context */
    error = appCtx.qnnSystemInterface.systemContextCreate(&appCtx.qnnSystemCtxHandle);
    if (error != QNN_SUCCESS) {
      printf("Received error when creating qnn system context\n");
      ret = -1;
      break;
    }

    /* Retrieve graph info from context binary */
    ret = qnnApp_getGraphInfo(&appCtx.qnnSystemInterface,
                              appCtx.qnnSystemCtxHandle,
                              appCtx.contextBinaryBuffer,
                              appCtx.contextBinaryBufferSize,
                              &appCtx.graphInfo);
    if (ret) {
      printf("Failed to get graph info\n");
      break;
    }

    /* Free context binary buffer if it does not need to be persistent */
    if (!requirePersistentBinary) {
      free_aligned_memory(appCtx.contextBinaryBuffer, DEFAULT_MEM_TYPE);
      appCtx.contextBinaryBuffer = NULL;
    }

    /* Retrieve graph from context */
    /* Recommended to use QnnSystemTypeMacros.hpp */
    const char* graphName = GET_GRAPH_NAME(appCtx.graphInfo);
    error =
        appCtx.lpaiInterface.graphRetrieve(appCtx.contextHandle, graphName, &appCtx.graphHandle);
    if (error != QNN_SUCCESS) {
      printf("Received error when retrieving graph\n");
      ret = -1;
      break;
    }

    /* Query, allocate and set scratch and persistent memory */
    ret = config_memory(&appCtx);
    if (ret) {
      printf("Failed to config scratch/persistent memory\n");
      break;
    }

    // default perf config setting
    uint32_t fps       = 10;
    uint32_t ftrtRatio = 0;
    bool isRealTime    = false;
    /* Set perf configs */
    error = qnnApp_graphSetConfigPerf(
        appCtx.graphHandle, &appCtx.lpaiInterface, fps, ftrtRatio, isRealTime);
    if (error != QNN_SUCCESS) {
      printf("Failed to set perf configs\n");
      ret = -1;
      break;
    }

    // default core affinity setting
    QnnLpaiGraph_CoreAffinityType_t coreAffinity = QNN_LPAI_GRAPH_CORE_AFFINITY_SOFT;
    uint32_t coreSel                             = 0;
    /* Set core affinity configs */
    error = qnnApp_graphSetConfigCoreAffinity(
        appCtx.graphHandle, &appCtx.lpaiInterface, coreAffinity, coreSel);
    if (error != QNN_SUCCESS) {
      printf("Failed to set core affinity configs\n");
      ret = -1;
      break;
    }

    // default priority config setting
    Qnn_Priority_t priority = QNN_PRIORITY_DEFAULT;
    /* Set priority config */
    error = qnnApp_graphSetConfigPriority(appCtx.graphHandle, &appCtx.lpaiInterface, priority);
    if (error != QNN_SUCCESS) {
      printf("Failed to set priority configs\n");
      ret = -1;
      break;
    }

    /* Finalize graph */
    error = appCtx.lpaiInterface.graphFinalize(appCtx.graphHandle, NULL, NULL);
    if (error != QNN_SUCCESS) {
      printf("Received error when finalizing graph\n");
      ret = -1;
      break;
    }

    /* Retrieve graph io info from graphInfo */
    ret = qnnApp_getGraphIO(
        appCtx.graphInfo, &appCtx.inputs, &appCtx.outputs, &appCtx.numInputs, &appCtx.numOutputs);
    if (ret) {
      printf("Failed to get graph IO info\n");
      break;
    }
    // inputTensors and outputTensors retrieved from graph info is located in ddr
    // and the memory is managed by qnnSystemCtx
    // user need to allocate their own inputTensors and outputTensors array so that
    // they can reside in other mem pool (e.g., tcm), and user need to manage their
    // allcoated memory
    Qnn_Tensor_t* inputTensorsNew  = NULL;
    Qnn_Tensor_t* outputTensorsNew = NULL;
    ret                            = allocate_tensors(appCtx.inputs,
                           &inputTensorsNew,
                           appCtx.numInputs,
                           DEFAULT_MEM_TYPE,
                           buffer_start_addr_alignment,
                           buffer_size_alignment);
    ret |= allocate_tensors(appCtx.outputs,
                            &outputTensorsNew,
                            appCtx.numOutputs,
                            DEFAULT_MEM_TYPE,
                            buffer_start_addr_alignment,
                            buffer_size_alignment);
    appCtx.inputs  = inputTensorsNew;
    appCtx.outputs = outputTensorsNew;
    if (ret) {
      printf("Failed to allocate graph IO\n");
      break;
    }

    do {
      /* Execute graph */
      /* If graphExecute returns QNN_GRAPH_ERROR_EARLY_TERMINATION, output is not ready and more
       * input frames are needed*/
      error = appCtx.lpaiInterface.graphExecute(appCtx.graphHandle,
                                                appCtx.inputs,
                                                appCtx.numInputs,
                                                appCtx.outputs,
                                                appCtx.numOutputs,
                                                NULL,
                                                NULL);
    } while (error == QNN_GRAPH_ERROR_EARLY_TERMINATION);
    if (error != QNN_SUCCESS) {
      printf("Received error when executing graph \n");
      ret = -1;
      break;
    }
  } while (0);

  qnnApp_freeLpaiBackend(&appCtx.lpaiInterface,
                         &appCtx.qnnSystemInterface,
                         appCtx.backendLibHandle,
                         appCtx.qnnSystemLibHandle,
                         appCtx.backendHandle,
                         appCtx.contextHandle,
                         appCtx.qnnSystemCtxHandle);

  // memory for context binary buffer can only be freed after context is freed
  cleanup_memory(&appCtx);

  return ret;
}