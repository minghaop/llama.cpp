//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#include "QnnSampleApp.hpp"
#include "QnnSampleAppConfigs.hpp"
#include "QnnSampleAppHelpers.hpp"

Qnn_ErrorHandle_t qnnApp_graphSetConfigScratchMem(Qnn_GraphHandle_t graphHandle,
                                                  QNN_INTERFACE_VER_TYPE *lpaiInterface,
                                                  uint32_t scratchSize,
                                                  void *scratchBuffer,
                                                  QnnLpaiMem_MemType_t memType) {
  // Create LPAI config
  QnnLpaiGraph_Mem_t lpaiGraphMem;
  lpaiGraphMem.memType = memType;
  lpaiGraphMem.size    = scratchSize;
  lpaiGraphMem.addr    = scratchBuffer;
  // Create QNN LPAI custom config
  QnnLpaiGraph_CustomConfig_t customGraphCfg;
  customGraphCfg.option = QNN_LPAI_GRAPH_SET_CFG_SCRATCH_MEM;
  customGraphCfg.config = &lpaiGraphMem;
  // Create QNN config
  QnnGraph_Config_t graphConfig;
  graphConfig.option                 = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graphConfig.customConfig           = &customGraphCfg;
  QnnGraph_Config_t *graphCfgPtrs[2] = {0};  // graphPropPtrs[1] is nullptr
  graphCfgPtrs[0]                    = &graphConfig;
  return lpaiInterface->graphSetConfig(graphHandle, (const QnnGraph_Config_t **)graphCfgPtrs);
}

Qnn_ErrorHandle_t qnnApp_graphSetConfigPersistentMem(Qnn_GraphHandle_t graphHandle,
                                                     QNN_INTERFACE_VER_TYPE *lpaiInterface,
                                                     uint32_t persistentSize,
                                                     void *persistentBuffer,
                                                     QnnLpaiMem_MemType_t memType) {
  // Create LPAI config
  QnnLpaiGraph_Mem_t lpaiGraphMem;
  lpaiGraphMem.memType = memType;
  lpaiGraphMem.size    = persistentSize;
  lpaiGraphMem.addr    = persistentBuffer;
  // Create QNN LPAI custom config
  QnnLpaiGraph_CustomConfig_t customGraphCfg;
  customGraphCfg.option = QNN_LPAI_GRAPH_SET_CFG_PERSISTENT_MEM;
  customGraphCfg.config = &lpaiGraphMem;
  // Create QNN config
  QnnGraph_Config_t graphConfig;
  graphConfig.option                 = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graphConfig.customConfig           = &customGraphCfg;
  QnnGraph_Config_t *graphCfgPtrs[2] = {0};  // graphPropPtrs[1] is nullptr
  graphCfgPtrs[0]                    = &graphConfig;
  return lpaiInterface->graphSetConfig(graphHandle, (const QnnGraph_Config_t **)graphCfgPtrs);
}

Qnn_ErrorHandle_t qnnApp_graphSetConfigPerf(Qnn_GraphHandle_t graphHandle,
                                            QNN_INTERFACE_VER_TYPE *lpaiInterface,
                                            uint32_t fps,
                                            uint32_t ftrtRatio,
                                            bool isRealTime) {
  // Create LPAI config
  QnnLpaiGraph_PerfCfg_t lpaiGraphPerfCfg;
  // Default to DDR, other memory types can be set as well
  lpaiGraphPerfCfg.fps        = fps;
  lpaiGraphPerfCfg.ftrtRatio  = ftrtRatio;
  lpaiGraphPerfCfg.clientType = isRealTime ? QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_REAL_TIME
                                           : QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_NON_REAL_TIME;
  // Create QNN LPAI custom config
  QnnLpaiGraph_CustomConfig_t customGraphCfg;
  customGraphCfg.option = QNN_LPAI_GRAPH_SET_CFG_PERF_CFG;
  customGraphCfg.config = &lpaiGraphPerfCfg;
  // Create QNN config
  QnnGraph_Config_t graphConfig;
  graphConfig.option                 = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graphConfig.customConfig           = &customGraphCfg;
  QnnGraph_Config_t *graphCfgPtrs[2] = {0};  // graphPropPtrs[1] is nullptr
  graphCfgPtrs[0]                    = &graphConfig;
  return lpaiInterface->graphSetConfig(graphHandle, (const QnnGraph_Config_t **)graphCfgPtrs);
}

Qnn_ErrorHandle_t qnnApp_graphSetConfigCoreAffinity(Qnn_GraphHandle_t graphHandle,
                                                    QNN_INTERFACE_VER_TYPE *lpaiInterface,
                                                    QnnLpaiGraph_CoreAffinityType_t coreAffinity,
                                                    uint32_t coreSel) {
  // Create LPAI config
  QnnLpaiGraph_CoreAffinity_t lpaiGraphCoreAffinityCfg;
  lpaiGraphCoreAffinityCfg.affinity      = coreAffinity;
  lpaiGraphCoreAffinityCfg.coreSelection = coreSel;
  // Create QNN LPAI custom config
  QnnLpaiGraph_CustomConfig_t customGraphCfg;
  customGraphCfg.option = QNN_LPAI_GRAPH_SET_CFG_CORE_AFFINITY;
  customGraphCfg.config = &lpaiGraphCoreAffinityCfg;
  // Create QNN config
  QnnGraph_Config_t graphConfig;
  graphConfig.option                 = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graphConfig.customConfig           = &customGraphCfg;
  QnnGraph_Config_t *graphCfgPtrs[2] = {0};  // graphPropPtrs[1] is nullptr
  graphCfgPtrs[0]                    = &graphConfig;
  return lpaiInterface->graphSetConfig(graphHandle, (const QnnGraph_Config_t **)graphCfgPtrs);
}

Qnn_ErrorHandle_t qnnApp_graphSetConfigPriority(Qnn_GraphHandle_t graphHandle,
                                                QNN_INTERFACE_VER_TYPE *lpaiInterface,
                                                Qnn_Priority_t priority) {
  // Create QNN config
  QnnGraph_Config_t graphConfig;
  graphConfig.option                 = QNN_GRAPH_CONFIG_OPTION_PRIORITY;
  graphConfig.priority               = priority;
  QnnGraph_Config_t *graphCfgPtrs[2] = {0};  // graphPropPtrs[1] is nullptr
  graphCfgPtrs[0]                    = &graphConfig;
  return lpaiInterface->graphSetConfig(graphHandle, (const QnnGraph_Config_t **)graphCfgPtrs);
}

Qnn_ErrorHandle_t qnnApp_graphGetPropertyScratchMemSize(Qnn_GraphHandle_t graphHandle,
                                                        QNN_INTERFACE_VER_TYPE *lpaiInterface,
                                                        uint32_t *scratchSize) {
  // Create QNN LPAI custom property
  QnnLpaiGraph_CustomProperty_t customGraphProp;
  customGraphProp.option   = QNN_LPAI_GRAPH_GET_PROP_SCRATCH_MEM_SIZE;
  customGraphProp.property = scratchSize;
  // Create QNN property
  QnnGraph_Property_t graphProp;
  graphProp.option                      = QNN_GRAPH_PROPERTY_OPTION_CUSTOM;
  graphProp.customProperty              = &customGraphProp;
  QnnGraph_Property_t *graphPropPtrs[2] = {0};  // graphPropPtrs[1] is nullptr
  graphPropPtrs[0]                      = &graphProp;
  return lpaiInterface->graphGetProperty(graphHandle, graphPropPtrs);
}

Qnn_ErrorHandle_t qnnApp_graphGetPropertyPersistentMemSize(Qnn_GraphHandle_t graphHandle,
                                                           QNN_INTERFACE_VER_TYPE *lpaiInterface,
                                                           uint32_t *persistentSize) {
  // Create QNN LPAI custom property
  QnnLpaiGraph_CustomProperty_t customGraphProp;
  customGraphProp.option   = QNN_LPAI_GRAPH_GET_PROP_PERSISTENT_MEM_SIZE;
  customGraphProp.property = persistentSize;
  // Create QNN property
  QnnGraph_Property_t graphProp;
  graphProp.option                      = QNN_GRAPH_PROPERTY_OPTION_CUSTOM;
  graphProp.customProperty              = &customGraphProp;
  QnnGraph_Property_t *graphPropPtrs[2] = {0};  // graphPropPtrs[1] is nullptr
  graphPropPtrs[0]                      = &graphProp;
  return lpaiInterface->graphGetProperty(graphHandle, graphPropPtrs);
}

Qnn_ErrorHandle_t qnnApp_backendGetPropertyAlignmentReq(Qnn_BackendHandle_t backendHandle,
                                                        QNN_INTERFACE_VER_TYPE *lpaiInterface,
                                                        uint32_t *startAddrAlignment,
                                                        uint32_t *sizeAlignment) {
  QnnLpaiBackend_BufferAlignmentReq_t bufferAlignmentReq;
  // Create QNN LPAI backend custom property
  QnnLpaiBackend_CustomProperty_t customBackendProp;
  customBackendProp.option   = QNN_LPAI_BACKEND_GET_PROP_ALIGNMENT_REQ;
  customBackendProp.property = &bufferAlignmentReq;
  // Create QNN property
  QnnBackend_Property_t backendProp;
  backendProp.option                        = QNN_BACKEND_PROPERTY_OPTION_CUSTOM;
  backendProp.customProperty                = &customBackendProp;
  QnnBackend_Property_t *backendPropPtrs[2] = {0};  // backendPropPtrs[1] is nullptr
  backendPropPtrs[0]                        = &backendProp;
  Qnn_ErrorHandle_t error = lpaiInterface->backendGetProperty(backendHandle, backendPropPtrs);
  if (!error) {
    *startAddrAlignment = bufferAlignmentReq.startAddrAlignment;
    *sizeAlignment      = bufferAlignmentReq.sizeAlignment;
  }
  return error;
}

Qnn_ErrorHandle_t qnnApp_backendGetPropertyPersistentBinary(Qnn_BackendHandle_t backendHandle,
                                                            QNN_INTERFACE_VER_TYPE *lpaiInterface,
                                                            bool *requirePersistentBinary) {
  // Create QNN LPAI backend custom property
  QnnLpaiBackend_CustomProperty_t customBackendProp;
  customBackendProp.option   = QNN_LPAI_BACKEND_GET_PROP_REQUIRE_PERSISTENT_BINARY;
  customBackendProp.property = requirePersistentBinary;
  // Create QNN property
  QnnBackend_Property_t backendProp;
  backendProp.option                        = QNN_BACKEND_PROPERTY_OPTION_CUSTOM;
  backendProp.customProperty                = &customBackendProp;
  QnnBackend_Property_t *backendPropPtrs[2] = {0};  // backendPropPtrs[1] is nullptr
  backendPropPtrs[0]                        = &backendProp;
  Qnn_ErrorHandle_t error = lpaiInterface->backendGetProperty(backendHandle, backendPropPtrs);
  return error;
}