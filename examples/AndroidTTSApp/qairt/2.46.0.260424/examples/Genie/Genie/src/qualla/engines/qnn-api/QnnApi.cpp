//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#if defined(LINUX_OE_HOST) || defined(LINUX_OPENWRT_HOST)
#include <fcntl.h>
#include <unistd.h>
#endif  // LINUX_OE_HOST || LINUX_OPENWRT_HOST

#include <chrono>
#include <filesystem>
#include <set>
#include <sstream>
#if defined(__GNUC__) && !defined(__clang__)
#include <cstring>
#endif  // defined(__GNUC__) && !defined(__clang__)

#ifndef _WIN32
#include <sys/mman.h>
#endif  // _WIN32

#include "MmappedFile/MmappedFile.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnTypeMacros.hpp"
#include "QnnTypeUtils.hpp"
#include "dlwrap.hpp"
#ifdef SPILLFILL
#ifdef QUALLA_ENGINE_QNN_HTP
#include "QnnHtpCommon.h"
#include "QnnHtpContext.h"
#endif  // QUALLA_ENGINE_QNN_HTP
#endif  // SPILLFILL

#include "Exception.hpp"
#include "QnnApi.hpp"
#include "Trace.hpp"

static LogCallback& getUserLogCallback() {
  static LogCallback s_userLogCallback = nullptr;
  return s_userLogCallback;
}

void emptyLogCallback(const char* /*fmt*/,
                      QnnLog_Level_t /*level*/,
                      uint64_t /*timestamp*/,
                      va_list /*args*/){
    // intentionally empty
};

QnnApi::~QnnApi() {
  QNN_DEBUG("Freeing Graphs");
  if (true != freeGraphs()) {
    QNN_DEBUG("Could not free Graphs");
  }

  if (m_scorer && !freeGraphInfo(m_scorer)) {
    QNN_DEBUG("Could not free scorer graph");
  }

  // Free context if not already done
  if (m_isContextCreated) {
    QNN_DEBUG("Freeing Context");
    if (true != freeContext()) {
      QNN_DEBUG("Could not free context");
    }
  }

  if (m_profileBackendHandle) {
    QNN_DEBUG("Freeing profile handle");
    if (QNN_PROFILE_NO_ERROR != m_qnnInterface.profileFree(m_profileBackendHandle))
      QNN_ERROR("Could not free QNN HTP backend profile handle.");
  }

  QNN_DEBUG("Freeing Device");
  if (m_isDeviceCreated) {
    if (true != freeDevice()) {
      QNN_ERROR("Device Free failure");
    }
  }

  // Terminate backend
  if (m_isBackendInitialized) {
    QNN_DEBUG("Terminating Backend");
    if (true != terminateBackend()) {
      QNN_DEBUG("Could not terminate backend");
    }
  }

  QNN_DEBUG("Terminating Logging");
  if (m_isLogInitialized) {
    terminateLogging();
  }
  m_isLogInitialized = false;

  // Free backend extensions before closing the backend library handle
  // to avoid accessing freed library during BackendExtensions destruction
  if (m_backendExtensions) {
    m_backendExtensions.reset();
  }

  if (m_backendLibraryHandle) {
    QNN_DEBUG("Closing Backend Lib Handle");
    pal::dynamicloading::dlClose(m_backendLibraryHandle);
  }

  if (m_libModelHandle) {
    QNN_DEBUG("Closing Model Lib Handle");
    pal::dynamicloading::dlClose(m_libModelHandle);
  }
}

bool QnnApi::getContextConfigs(ConfigList<QnnContext_Config_t>& configList,
                               bool graphSwitching,
                               const std::vector<std::string>& execSelectGraphs,
                               bool loadSelectGraphs) {
  if (loadSelectGraphs && !execSelectGraphs.empty()) {
    configList.add(std::make_unique<QnnContextConfig>(ContextEnableGraphsConfig(execSelectGraphs)));
  }

  if (graphSwitching) {
    configList.add(std::make_unique<QnnContextConfig>(ContextMemoryLimitHintConfig(1024)));
    configList.add(std::make_unique<QnnContextConfig>(ContextPersistentBinaryConfig(true)));
  }

  return true;
}

bool QnnApi::setGraphConfigsBeforeExecute(Qnn_GraphHandle_t graphHandle,
                                          QnnGraph_Config_t** graphConfigs,
                                          uint32_t configCount) {
  if (!graphConfigs || configCount == 0u) {
    QNN_ERROR("No graph configs to set");
    return false;
  }

  std::vector<const QnnGraph_Config_t*> graphConfigsPointers(configCount + 1, nullptr);
  for (size_t idx = 0u; idx < configCount; idx++) {
    graphConfigsPointers[idx] = graphConfigs[idx];
  }
  if (QNN_SUCCESS != m_qnnInterface.graphSetConfig(graphHandle, graphConfigsPointers.data())) {
    QNN_ERROR("Failed to set graph configs.");
    return false;
  }

  return true;
}

bool QnnApi::getQnnInterface(std::string backendPath) {
  QnnInterfaceGetProvidersFn_t getInterfaceProviders{nullptr};

  m_backendLibraryHandle =
      m_resourceManager->dlOpen(backendPath.c_str(), pal::dynamicloading::DL_NOW);
  if (nullptr == m_backendLibraryHandle) {
    QNN_ERROR("Unable to load backend. dlerror(): %s", pal::dynamicloading::dlError());
    return false;
  }

  // Get QNN Interface
  getInterfaceProviders = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
      pal::dynamicloading::dlSym(m_backendLibraryHandle, "QnnInterface_getProviders"));
  if (nullptr == getInterfaceProviders) {
    return false;
  }

  uint32_t numProviders{0};
  QnnInterface_t** interfaceProviders{nullptr};
  if (QNN_SUCCESS != getInterfaceProviders(const_cast<const QnnInterface_t***>(&interfaceProviders),
                                           &numProviders)) {
    QNN_ERROR("Failed to get interface providers.");
    return false;
  }

  if (nullptr == interfaceProviders) {
    QNN_ERROR("Failed to get interface providers: null interface providers received.");
    return false;
  }
  if (0u == numProviders) {
    QNN_ERROR("Failed to get interface providers: 0 interface providers.");
    return false;
  }

  bool foundValidInterface{false};
  for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
    const Qnn_ApiVersion_t& apiVersion = interfaceProviders[pIdx]->apiVersion;
    if ((QNN_API_VERSION_MAJOR == apiVersion.coreApiVersion.major) &&
        (QNN_API_VERSION_MINOR <= apiVersion.coreApiVersion.minor)) {
      foundValidInterface = true;
      m_qnnInterface      = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
      m_backendId         = interfaceProviders[pIdx]->backendId;
      break;
    }
  }

  if (!foundValidInterface) {
    QNN_ERROR("Unable to find a compatible QNN API interface.");
    QNN_ERROR("Expected API version %u.%u.%u or later",
              static_cast<unsigned int>(QNN_API_VERSION_MAJOR),
              static_cast<unsigned int>(QNN_API_VERSION_MINOR),
              static_cast<unsigned int>(QNN_API_VERSION_PATCH));
    std::stringstream availableVersions;
    for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
      const Qnn_ApiVersion_t& apiVersion = interfaceProviders[pIdx]->apiVersion;
      availableVersions << apiVersion.coreApiVersion.major << "." << apiVersion.coreApiVersion.minor
                        << "." << apiVersion.coreApiVersion.patch << ", ";
    }
    // Remove trailing comma
    availableVersions.seekp(-2, std::ios_base::cur);
    availableVersions << '\0';
    QNN_ERROR("Available API versions: %s", availableVersions.str().c_str());
    m_backendLibraryHandle = nullptr;
    return false;
  }

  return true;
}

bool QnnApi::loadModel(std::string model_path) {
  const char* dlsym_error;

  pal::dynamicloading::dlError();
  m_libModelHandle = pal::dynamicloading::dlOpen(model_path.c_str(), pal::dynamicloading::DL_NOW);
  if (nullptr == m_libModelHandle) {
    QNN_ERROR("Unable to load model. dlerror(): %s", pal::dynamicloading::dlError());
    return false;
  }

  // Currently model Prefix is fixed. If model was prepared with
  // custom prefix, we need to change this.
  std::string modelPrefix = "QnnModel";

  std::string modelPrepareFunc = modelPrefix + "_composeGraphs";
  m_composeGraphsFnHandle      = reinterpret_cast<ComposeGraphsFnHandleType_t>(
      pal::dynamicloading::dlSym(m_libModelHandle, modelPrepareFunc.c_str()));
  dlsym_error = pal::dynamicloading::dlError();
  if (nullptr == m_composeGraphsFnHandle) {
    m_composeGraphsFnHandle           = nullptr;
    std::string genaiModelPrepareFunc = "QnnModel_GenAI_composeGraphs";
    m_genaiComposeGraphsFnHandle      = reinterpret_cast<GenAIComposeGraphsFnHandleType_t>(
        pal::dynamicloading::dlSym(m_libModelHandle, genaiModelPrepareFunc.c_str()));
    dlsym_error = pal::dynamicloading::dlError();
    if (nullptr == m_genaiComposeGraphsFnHandle) {
      QNN_ERROR("Did not find QnnModel_GenAI_composeGraphs function: %s", dlsym_error);
      return false;
    }
  }

  std::string modelFreeFunc = modelPrefix + "_freeGraphsInfo";
  m_freeGraphInfoFnHandle   = reinterpret_cast<FreeGraphInfoFnHandleType_t>(
      pal::dynamicloading::dlSym(m_libModelHandle, modelFreeFunc.c_str()));
  dlsym_error = pal::dynamicloading::dlError();
  if (nullptr == m_freeGraphInfoFnHandle) {
    QNN_ERROR("Did not find QnnModel_freeGraphsInfo function: %s", dlsym_error);
    return false;
  }

  return true;
}

void userLogCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
  uint32_t i    = static_cast<uint32_t>(level);
  auto callback = getUserLogCallback();
  callback(fmt, i, timestamp, args);
}

bool QnnApi::initializeLogging(const QnnLog_Level_t& logLevel,
                               bool debug_qnn,
                               LogCallback userCallback) {
  // initialize logging in the backend
  if (nullptr != m_qnnInterface.logCreate) {
    QnnLog_Callback_t logCallback;
    if (userCallback) {
      LogCallback& s_userLogCallback = getUserLogCallback();
      s_userLogCallback              = userCallback;
    }
    if (debug_qnn) {
      logCallback = userLogCallback;
    } else {
      logCallback = emptyLogCallback;
    }
    QNN_DEBUG("Initializing logging in the backend. Callback: [%p], Log Level: [%d]",
              logCallback,
              static_cast<int32_t>(logLevel));
    if (QNN_SUCCESS != m_qnnInterface.logCreate(logCallback, logLevel, &m_logHandle)) {
      QNN_WARN("Unable to initialize logging in the backend.");
    }
    m_isLogInitialized = true;
  } else {
    QNN_WARN("Logging not available in the backend.");
    return true;
  }

  return true;
}

void QnnApi::terminateLogging() {
  // Terminate logging in the backend
  if (nullptr != m_qnnInterface.logFree && nullptr != m_logHandle) {
    if (QNN_SUCCESS != m_qnnInterface.logFree(m_logHandle)) {
      QNN_WARN("Unable to terminate logging in the backend.");
    }
  }
}

bool QnnApi::initializeBackendExtensions(BackendExtensionsConfigs backendExtensionsConfig,
                                         bool debug_qnn,
                                         QnnLog_Level_t qnnLogLevel) {
  if (backendExtensionsConfig.sharedLibraryPath.empty() &&
      backendExtensionsConfig.configFilePath.empty()) {
    // Backend extensions are not in use, return success.
    return true;
  }
  try {
    m_backendExtensions.reset(new BackendExtensions(backendExtensionsConfig,
                                                    m_backendLibraryHandle,
                                                    debug_qnn,
                                                    debug_qnn ? userLogCallback : emptyLogCallback,
                                                    qnnLogLevel,
                                                    m_resourceManager));
  } catch (const std::exception& e) {
    const char* msg = e.what();
    QNN_WARN("%s", msg);
    m_backendExtensions = nullptr;
    return false;
  }

  if (nullptr == m_backendExtensions) {
    QNN_ERROR("Unable to create backend extensions object.");
    return false;
  }

  return true;
}

// Initialize a QnnBackend.
bool QnnApi::initializeBackend() {
  GENIE_TRACE();
  if (nullptr == m_qnnInterface.backendCreate) {
    QNN_ERROR("BackendCreate API is not supported for this backend");
    return false;
  }

  QnnBackend_Config_t** customConfigs{nullptr};
  uint32_t customConfigCount{0};
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeBackendInitialize(&customConfigs,
                                                                   &customConfigCount)) {
      QNN_ERROR("Extensions Failure in beforeBackendInitialize()");
      return false;
    }
  }
  QnnBackend_Config_t** allBackendConfigs{nullptr};
  if ((m_backendConfigCount + customConfigCount) > 0) {
    allBackendConfigs = reinterpret_cast<QnnBackend_Config_t**>(
        calloc((m_backendConfigCount + customConfigCount + 1), sizeof(QnnBackend_Config_t*)));
    if (nullptr == allBackendConfigs) {
      QNN_ERROR("Could not allocate memory for allBackendConfigs");
      return false;
    }
    for (size_t cnt = 0; cnt < m_backendConfigCount; cnt++) {
      allBackendConfigs[cnt] = m_backendConfigs[cnt];
    }
    for (size_t cnt = 0; cnt < customConfigCount; cnt++) {
      allBackendConfigs[cnt + m_backendConfigCount] = customConfigs[cnt];
    }
  }

  Qnn_ErrorHandle_t errCode = m_qnnInterface.backendCreate(
      m_logHandle, const_cast<const QnnBackend_Config_t**>(allBackendConfigs), &m_backendHandle);
  if (QNN_SUCCESS != errCode) {
    QNN_ERROR("Could not initialize backend due to error = %lu",
              static_cast<unsigned long>(errCode));
    if (allBackendConfigs) {
      free(allBackendConfigs);
    }
    return false;
  }
  QNN_DEBUG("Initialize Backend Returned Status = %lu", static_cast<unsigned long>(errCode));

  m_isBackendInitialized = true;
  if (allBackendConfigs) {
    free(allBackendConfigs);
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterBackendInitialize()) {
      QNN_ERROR("Extensions Failure in afterBackendInitialize()");
      return false;
    }
  }

  return true;
}

// Terminate the backend after done.
bool QnnApi::terminateBackend() {
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeBackendTerminate()) {
      QNN_ERROR("Extensions Failure in beforeBackendTerminate()");
      return false;
    }
  }
  // Terminate backend
  if (m_isBackendInitialized && nullptr != m_qnnInterface.backendFree) {
    QNN_DEBUG("Freeing backend");
    if (QNN_BACKEND_NO_ERROR != m_qnnInterface.backendFree(m_backendHandle)) {
      QNN_ERROR("Could not free backend");
    }
  }
  m_isBackendInitialized = false;

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterBackendTerminate()) {
      QNN_ERROR("Extensions Failure in afterBackendTerminate()");
      return false;
    }
  }

  return true;
}

bool QnnApi::createDevice() {
  GENIE_TRACE();
  QnnDevice_Config_t** deviceConfigs{nullptr};
  uint32_t configCount{0};
  uint32_t socModel{0};
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeCreateDevice(
            &deviceConfigs, &configCount, socModel)) {
      QNN_ERROR("Extensions Failure in beforeCreateDevice()");
      return false;
    }
  }
  std::vector<const QnnDevice_Config_t*> deviceConfigPointers(configCount + 1, nullptr);
  for (size_t idx = 0u; idx < configCount; idx++) {
    deviceConfigPointers[idx] = deviceConfigs[idx];
  }
  if (nullptr != m_qnnInterface.deviceCreate) {
    auto qnnStatus =
        m_qnnInterface.deviceCreate(m_logHandle, deviceConfigPointers.data(), &m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus) {
      if (QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE == qnnStatus) {
        QNN_WARN("Device feature unsupported");
      } else {
        QNN_ERROR("Failed to create device: %lu", static_cast<unsigned long>(qnnStatus));
        return false;
      }
    }
  }
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterCreateDevice()) {
      QNN_ERROR("Extensions Failure in afterCreateDevice()");
      return false;
    }
  }
  return true;
}

bool QnnApi::freeDevice() {
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeFreeDevice()) {
      QNN_ERROR("Extensions Failure in beforeFreeDevice()");
      return false;
    }
  }
  if (nullptr != m_qnnInterface.deviceFree) {
    auto qnnStatus = m_qnnInterface.deviceFree(m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus) {
      if (QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE == qnnStatus) {
        QNN_WARN("Device feature unsupported");
      } else {
        QNN_ERROR("Failed to free device: %lu", static_cast<unsigned long>(qnnStatus));
        return false;
      }
    }
  }
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterFreeDevice()) {
      QNN_ERROR("Extensions Failure in afterfreeDevice()");
      return false;
    }
  }
  return true;
}

// Create a Context in a backend.
bool QnnApi::createContext() {
  QnnContext_Config_t** customConfigs{nullptr};
  uint32_t customConfigCount{0};
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeContextCreate(&customConfigs,
                                                               &customConfigCount)) {
      QNN_ERROR("Extensions Failure in beforeContextCreate()");
      return false;
    }
  }

  ContextConfigList configList = ContextConfigList::fromArray(customConfigs, customConfigCount);

  if (true != getContextConfigs(configList)) {
    QNN_ERROR("Couldn't populate context configs");
    return false;
  }

  Qnn_ContextHandle_t contextHandle{nullptr};
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnInterface.contextCreate(m_backendHandle,
                                   nullptr,
                                   static_cast<const QnnContext_Config_t**>(configList),
                                   &contextHandle)) {
    QNN_ERROR("Could not create context");
    return false;
  }

  m_contextVec.push_back(contextHandle);
  m_isContextCreated = true;

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterContextCreate()) {
      QNN_ERROR("Extensions Failure in afterContextCreate()");
      return false;
    }
  }

  return true;
}

// Free context after done.
bool QnnApi::freeContext() {
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeContextFree(m_contextVec)) {
      QNN_ERROR("Extensions Failure in beforeContextFree()");
      return false;
    }
  }
  for (const auto& context : m_contextVec) {
    if (context && (QNN_CONTEXT_NO_ERROR != m_qnnInterface.contextFree(context, nullptr))) {
      QNN_ERROR("Could not free context");
      return false;
    }
  }
  m_isContextCreated = false;

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterContextFree()) {
      QNN_ERROR("Extensions Failure in afterContextFree()");
      return false;
    }
  }

  return true;
}

// Calls composeGraph function in QNN's model.so.
// composeGraphs is supposed to populate graph related
// information in graphsInfo and graphsCount.
// m_debug is the option supplied to composeGraphs to
// say that all intermediate tensors including output tensors
// are expected to be read by the app.
bool QnnApi::composeGraphs(std::vector<GraphConfigs> graphConfigs) {
  qnn_wrapper_api::GraphConfigInfo_t** customConfigs{nullptr};
  uint32_t customConfigGraphsCount{0};
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeComposeGraphs(&customConfigs,
                                                               &customConfigGraphsCount)) {
      QNN_ERROR("Extensions Failure in beforeComposeGraphs()");
      return false;
    }
  }

  std::map<std::string, std::vector<QnnGraph_Config_t*>> graphConfigsPointers;
  if (!graphConfigs.empty()) {
    for (auto const& inputGraphConfig : graphConfigs) {
      // Only reset the memory for this graph, if it has not previously been populated with
      // something
      if (graphConfigsPointers.find(inputGraphConfig.graphName) == graphConfigsPointers.end()) {
        graphConfigsPointers[inputGraphConfig.graphName] = std::vector<QnnGraph_Config_t*>();
        graphConfigsPointers[inputGraphConfig.graphName].reserve(s_graphConfigsReserveCount);
      }
      if (inputGraphConfig.priorityPresent) {
        QnnGraph_Config_t* newGraphConfig =
            reinterpret_cast<QnnGraph_Config_t*>(malloc(sizeof(QnnGraph_Config_t)));
        newGraphConfig->option   = QNN_GRAPH_CONFIG_OPTION_PRIORITY;
        newGraphConfig->priority = inputGraphConfig.priority;
        graphConfigsPointers[inputGraphConfig.graphName].push_back(newGraphConfig);
      }
    }
  }

  if (customConfigs != nullptr && customConfigGraphsCount > 0) {
    for (size_t gIdx = 0; gIdx < customConfigGraphsCount; gIdx++) {
      auto configPtr = customConfigs[gIdx]->graphConfigs;
      if (*configPtr &&
          (!customConfigs[gIdx]->graphName || strlen(customConfigs[gIdx]->graphName) == 0)) {
        QNN_ERROR("Graph configs specified without a graph name in the backend extensions.");
        return false;
      }
      if (customConfigs[gIdx]->graphName && strlen(customConfigs[gIdx]->graphName) > 0 &&
          *configPtr) {
        if (graphConfigsPointers.find(customConfigs[gIdx]->graphName) ==
            graphConfigsPointers.end()) {
          graphConfigsPointers[customConfigs[gIdx]->graphName] = std::vector<QnnGraph_Config_t*>();
          graphConfigsPointers[customConfigs[gIdx]->graphName].reserve(s_graphConfigsReserveCount);
        }
        while (*configPtr) {
          graphConfigsPointers[customConfigs[gIdx]->graphName].push_back(
              const_cast<QnnGraph_Config_t*>(*configPtr));
          configPtr++;
        }
      }
    }
  }

  auto graphConfigsInfo = reinterpret_cast<qnn_wrapper_api::GraphConfigInfo_t**>(
      calloc(graphConfigsPointers.size(), sizeof(qnn_wrapper_api::GraphConfigInfo_t*)));
  size_t graphIdx{0};
  for (auto const& graphConfig : graphConfigsPointers) {
    if (graphConfigsInfo && graphConfig.second.size() > 0) {
      graphConfigsInfo[graphIdx] = reinterpret_cast<qnn_wrapper_api::GraphConfigInfo_t*>(
          malloc(sizeof(qnn_wrapper_api::GraphConfigInfo_t)));
      graphConfigsInfo[graphIdx]->graphName    = const_cast<char*>(graphConfig.first.c_str());
      graphConfigsInfo[graphIdx]->graphConfigs = reinterpret_cast<const QnnGraph_Config_t**>(
          calloc(graphConfig.second.size() + 1, sizeof(QnnGraph_Config_t*)));
      for (size_t cnt = 0; cnt < graphConfig.second.size(); cnt++) {
        graphConfigsInfo[graphIdx]->graphConfigs[cnt] = graphConfig.second[cnt];
      }
    }
    graphIdx++;
  }

  int status = m_composeGraphsFnHandle(
      m_backendHandle,
      m_qnnInterface,
      m_contextVec[0],
      const_cast<const qnn_wrapper_api::GraphConfigInfo_t**>(graphConfigsInfo),
      graphConfigsPointers.size(),
      &m_graphsInfo,
      &m_graphsCount,
      m_debugModeRequested,
      nullptr,
      QnnLog_Level_t::QNN_LOG_LEVEL_VERBOSE);

  if (graphConfigsInfo) {
    for (size_t gIdx = 0; gIdx < graphConfigsPointers.size(); gIdx++) {
      if (graphConfigsInfo[gIdx]) {
        if (graphConfigsInfo[gIdx]->graphConfigs) {
          free(graphConfigsInfo[gIdx]->graphConfigs);
          graphConfigsInfo[gIdx]->graphConfigs = nullptr;
          graphConfigsInfo[gIdx]->graphName    = nullptr;
        }
        free(graphConfigsInfo[gIdx]);
        graphConfigsInfo[gIdx] = nullptr;
      }
    }
    free(graphConfigsInfo);
  }

  for (auto const& graphConfig : graphConfigsPointers) {
    for (size_t cnt = 0; cnt < graphConfig.second.size(); cnt++) {
      if (graphConfig.second[cnt]) {
        free(graphConfig.second[cnt]);
      }
    }
    // graphConfig.second.clear();
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterComposeGraphs()) {
      QNN_ERROR("Extensions Failure in afterComposeGraphs()");
      return false;
    }
  }

  if (0 != status) {
    QNN_ERROR("Failed in composeGraphs()");
    return false;
  }

  // For now, we only handle 1 graph for this framework.
  if (m_graphsCount != 1) {
    QNN_ERROR("Only one graph is supported by framework");
    return false;
  }

  return true;
}

bool QnnApi::composeGraphs(std::vector<GraphConfigs> /*graphConfigs*/,
                           uint32_t* inputDim,
                           uint32_t inputRank,
                           uint32_t* outputDim,
                           uint32_t outputRank,
                           uint32_t* kvDim,
                           uint32_t kvRank,
                           uint32_t* kvScaleDim,
                           Qnn_Param_t* params,
                           uint32_t numParams) {
  std::string model_name = "qnn_model";
  qnn_wrapper_api::ModelError status =
      m_genaiComposeGraphsFnHandle(m_backendHandle,
                                   m_qnnInterface,
                                   m_contextVec[0],
                                   nullptr,
                                   0,
                                   inputDim,
                                   inputRank,
                                   outputDim,
                                   outputRank,
                                   kvDim,
                                   kvRank,
                                   kvScaleDim,
                                   params,
                                   numParams,
                                   model_name.c_str(),
                                   &m_graphsInfo,
                                   &m_graphsCount,
                                   m_debugModeRequested,
                                   nullptr,
                                   QnnLog_Level_t::QNN_LOG_LEVEL_VERBOSE);

  m_graphCountPerContext.push_back(m_graphsCount);

  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    m_contextMap[m_graphsInfo[graphIdx]] = m_contextVec.back();
    m_graphNameToContextIdx[std::string(m_graphsInfo[graphIdx]->graphName)] =
        static_cast<uint32_t>(m_contextVec.size() - 1);
  }

  if (status == qnn_wrapper_api::MODEL_NO_ERROR) {
    return true;
  }

  return false;
}

bool QnnApi::finalizeCpuGraphs() {
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeGraphFinalize()) {
      QNN_ERROR("Extensions Failure in beforeGraphFinalize()");
      return false;
    }
  }

  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    if (QNN_GRAPH_NO_ERROR !=
        m_qnnInterface.graphFinalize(m_graphsInfo[graphIdx]->graph, nullptr, nullptr)) {
      return false;
    }

    if (m_profileBackendHandle) {
      extractBackendProfilingInfo(m_profileBackendHandle);
    }
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterGraphFinalize()) {
      QNN_ERROR("Extensions Failure in afterGraphFinalize()");
      return false;
    }
  }

  return true;
}

bool QnnApi::finalizeGraphs() {
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeGraphFinalize()) {
      QNN_ERROR("Extensions Failure in beforeGraphFinalize()");
      return false;
    }
  }

  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    if (QNN_GRAPH_NO_ERROR !=
        m_qnnInterface.graphFinalize(m_graphsInfo[graphIdx]->graph, nullptr, nullptr)) {
      return false;
    }

    if (m_profileBackendHandle) {
      extractBackendProfilingInfo(m_profileBackendHandle);
    }
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterGraphFinalize()) {
      QNN_ERROR("Extensions Failure in afterGraphFinalize()");
      return false;
    }
  }

  return true;
}

bool QnnApi::freeGraphs() {
  freeGraphsInfo(&m_graphsInfo, m_graphsCount);
  if (m_graphsInfo) {
    free(m_graphsInfo);
  }
  m_graphsInfo  = nullptr;
  m_graphsCount = 0;
  return true;
}

bool QnnApi::mapAndGetContextBinaryInfo(const bool useMmap,
                                        std::shared_ptr<uint8_t>& buffer,
                                        const std::string& binaryPath,
                                        const uint64_t bufferSize,
                                        const size_t contextIdx,
                                        const bool graphSwitching,
                                        QnnSystemContext_Handle_t sysCtxHandle,
                                        const QnnSystemContext_BinaryInfo_t** binaryInfo) {
  GENIE_TRACE();
  if (useMmap) {
    if (!m_resourceManager->isDLCRecord(binaryPath)) {
#if defined(LINUX_OE_HOST) || defined(LINUX_OPENWRT_HOST)
      // Read binary file with mmapped syscall
      int fd      = open(binaryPath.c_str(), O_RDONLY);
      void* mmbuf = mmap(nullptr, bufferSize, PROT_READ, MAP_PRIVATE, fd, 0);
      close(fd);
      if (madvise(mmbuf, bufferSize, MADV_NOHUGEPAGE)) {
        QNN_WARN("Failed to advise OS on memory usage");
      }

      buffer = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(mmbuf),
                                        [bufferSize](uint8_t* ptr) { munmap(ptr, bufferSize); });
#else
      // Memory mapped binary allocation
      auto mmf = std::make_shared<mmapped::File>(binaryPath);
      if (!(*mmf)) {
        QNN_ERROR("Failed to allocate memory mapped region for context index = %zu", contextIdx);
      }

#if !defined(_WIN32) && !defined(__QNXNTO__)
      // Note: There is no Windows-equivalent of madvise
      if (!mmf->adviseRange(0, bufferSize, MADV_NOHUGEPAGE)) {
        QNN_ERROR("Failed to advise OS on memory usage err: %s", strerror(errno));
        return false;
      }
#endif  // !_WIN32 && !__QNXNTO__

      // Note: Following custom deallocator is necessary to tie the lifespan of 'mmf' to that of
      // 'buffer'
      buffer = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(mmf->mutableData()),
                                        [mmf](uint8_t* /*ptr*/) mutable { mmf.reset(); });
#endif  // LINUX_OE_HOST || LINUX_OPENWRT_HOST
    } else {
      if (!m_resourceManager->getBuffer(binaryPath, buffer, bufferSize)) {
        return false;
      }
    }
  } else {
    // DLC is not supported in non-mmap mode.
    if (m_resourceManager->isDLCRecord(binaryPath)) {
      QNN_ERROR("DLC is not supported in non-mmap mode");
      return false;
    }
    // Regular binary allocation
    buffer = std::shared_ptr<uint8_t>(new uint8_t[bufferSize], std::default_delete<uint8_t[]>());

    if (!buffer) {
      QNN_ERROR("Failed to allocate memory for context index = %zu", contextIdx);
      return false;
    }
    if (!m_resourceManager->getBuffer(binaryPath, buffer, bufferSize)) {
      QNN_ERROR("Failed to read binary data for context index = %zu", contextIdx);
      return false;
    }
  }

  if (graphSwitching) {
    // When graph switching is enabled, buffer should be kept all the way until QnnApi class EOL
    m_persistentContextBins.push_back(buffer);
  }

  Qnn_ContextBinarySize_t binaryInfoSize{0};
  if (QNN_SUCCESS != m_resourceManager->getQnnSystemInterface().systemContextGetBinaryInfo(
                         sysCtxHandle,
                         static_cast<void*>(buffer.get()),
                         bufferSize,
                         binaryInfo,
                         &binaryInfoSize)) {
    QNN_ERROR("Failed to get context binary info for context index = %zu", contextIdx);
    return false;
  }

  return true;
}

bool QnnApi::parseIOTensorsAndAccumulate() {
  GENIE_TRACE();
  for (size_t gIdx = 0; gIdx < m_graphsCount; gIdx++) {
    const auto& graphInfo = m_graphsInfo[gIdx];
    for (bool io : {true, false}) {
      uint32_t numTensors = io ? graphInfo->numInputTensors : graphInfo->numOutputTensors;
      auto tensorWrappers = io ? graphInfo->inputTensors : graphInfo->outputTensors;
      for (size_t tensorIdx = 0; tensorIdx < numTensors; tensorIdx++) {
        Qnn_Tensor_t& tensor   = tensorWrappers[tensorIdx];
        std::string tensorName = QNN_TENSOR_GET_NAME(tensor);

        if (QNN_TENSOR_GET_DIMENSIONS(tensor) == nullptr || QNN_TENSOR_GET_RANK(tensor) == 0) {
          QNN_ERROR("Couldn't get tensor shape : %s", tensorName.c_str());
          return false;
        }

        m_graphIdxToIOMap[gIdx][tensorName] = qualla::QnnUtils::Tensor(&tensor);
      }
    }
  }
  // Maps tensorName to context bitVector, each bit representing a context the tensor exists in
  std::unordered_map<std::string, CtxBitVector> tensorCtxMap;
  // Maps a ContextHandle to a one-hot encoded bitVector (e.g. 1, 2, 4, ...)
  std::unordered_map<uint32_t, CtxBitVector> ctx_to_hash;
  std::unordered_map<size_t, CtxBitVector> ctxToHash;
  // Iterate over all tensors in all GraphVariants to figure out allocations
  for (size_t gIdx = 0; gIdx < m_graphsCount; gIdx++) {
    const auto& graphInfo = m_graphsInfo[gIdx];
    auto variantType      = m_graphVariantTypeMap[graphInfo->graphName];
    // Map the context handle to a hashed bitVector
    size_t curContextHandle = m_graphIdxToContextIdx[gIdx];
    if (!ctxToHash.contains(curContextHandle)) {
      ctxToHash[curContextHandle] = 1 << ctxToHash.size();
    }
    for (auto& [tname, tspec] : m_graphIdxToIOMap[gIdx]) {
      size_t size           = tspec.dims.getAlignedSize();
      CtxBitVector tcontext = ctxToHash[curContextHandle];

      // Check if it's LoRA enabled model
      if (!m_loraWeightEnabled && tname.find("lora") != std::string::npos)
        m_loraWeightEnabled = true;
      // Check if graph has lmhead weight input
      if (!m_lmHeadWeightInput && tname.compare("weight") == 0) m_lmHeadWeightInput = true;

      // Allocate KV Tensors as in+out
      if (qualla::QnnUtils::matchPrefixAny(tname, m_cacheGroupPrefixes) &&
          qualla::QnnUtils::isKVTensor(tname)) {
        if (tname.ends_with("_in")) continue;  // kv_in is processed along with kv_out

        // For kv_out, add the size of kv_in as well
        const std::string tname_in = tname.substr(0, tname.rfind('_')).append("_in");

        if (variantType != GraphType::DECODER_PREFILL) {
          if (m_graphIdxToIOMap[gIdx].count(tname_in)) {
            size += m_graphIdxToIOMap[gIdx][tname_in].dims.getAlignedSize();
          }
        } else {
          size_t kv_in_channel;
          if (m_cacheGroupUseScatter[qualla::QnnUtils::getPrefix(tname_in, m_cacheGroupPrefixes)]) {
            kv_in_channel =
                m_cacheGroupCtxSize[qualla::QnnUtils::getPrefix(tname_in, m_cacheGroupPrefixes)];
          } else {
            size_t ctx_size =
                m_cacheGroupCtxSize[qualla::QnnUtils::getPrefix(tname_in, m_cacheGroupPrefixes)];
            size_t kv_out_channel =
                (tname.rfind("key") != std::string::npos ? tspec.dims.channel : tspec.dims.width);
            if (ctx_size < kv_out_channel) {
              QNN_ERROR("Invalid KV dimension: context size %zu is less than channel dimension %zu",
                        ctx_size,
                        kv_out_channel);
              return false;
            }
            kv_in_channel = ctx_size - kv_out_channel;
          }
          size +=
              (kv_in_channel * tspec.dims.batch * tspec.dims.height *
               (tname.rfind("key") != std::string::npos ? tspec.dims.width : tspec.dims.channel));
          size *= tspec.dims.bitwidth;
        }
      }

      if (tensorCtxMap.contains(tname)) {  // For duplicate tensor names, link them
        CtxBitVector context_bitvec = tensorCtxMap.at(tname);
        size                        = std::max(m_contextAllocMap[context_bitvec][tname], size);
        if ((context_bitvec & tcontext) == 0)  // Set of contexts needs to be updated
          m_contextAllocMap[context_bitvec].erase(tname);

        tcontext |= context_bitvec;
      }

      m_contextAllocMap[tcontext][tname] = size;
      tensorCtxMap[tname]                = tcontext;
    }
    // Cleanup is essential in case of very large number of splits
    for (auto it = m_contextAllocMap.cbegin(); it != m_contextAllocMap.cend();)
      it = (it->second.empty()) ? m_contextAllocMap.erase(it) : ++it;
  }
#if QNN_IO_TENSOR_DEBUG
  for (auto& [bitvector, nameMap] : m_contextAllocMap) {
    for (auto [tname, size] : nameMap)
      QNN_DEBUG("Context: %d Tensor name: %s Tensor size: %zu", bitvector, tname.c_str(), size);
  }
#endif
  m_estimator = std::make_shared<Estimator>(m_contextAllocMap);
  return true;
}

bool QnnApi::registerTensorsWithBackend(size_t graphIdx) {
  std::string graphName = m_graphsInfo[graphIdx]->graphName;
  GraphType variantType = m_graphVariantTypeMap[graphName];
  std::map<std::string, std::tuple<int, size_t, size_t>> graph_allocs;
  for (auto& [tname, tspec] : m_graphIdxToIOMap[graphIdx]) {
    if (qualla::QnnUtils::matchPrefixAny(tname, m_cacheGroupPrefixes) && tname.ends_with("_in")) {
      continue;  // Process past_key/value_Inputs along with the outputs
    }
    auto& [alloc_idx, offset] = m_tensorAllocInfo->at(tname);

    size_t kv_offset = 0;
    size_t size      = tspec.dims.getAlignedSize();
    if (qualla::QnnUtils::matchPrefixAny(tname, m_cacheGroupPrefixes) &&
        qualla::QnnUtils::isKVTensor(tname)) {
      std::string in_name = tname.substr(0, tname.rfind("_")).append("_in");
      if (variantType != GraphType::DECODER_PREFILL) {
        if (m_graphIdxToIOMap[graphIdx].count(in_name)) {
          auto kv_in            = m_graphIdxToIOMap[graphIdx][in_name];
          kv_offset             = kv_in.dims.getAlignedSize();
          graph_allocs[in_name] = {alloc_idx, offset, kv_offset};
        }
      } else {
        auto kv_out = m_graphIdxToIOMap[graphIdx][tname];
        size_t kv_in_channel;
        if (m_cacheGroupUseScatter[qualla::QnnUtils::getPrefix(in_name, m_cacheGroupPrefixes)]) {
          kv_in_channel =
              m_cacheGroupCtxSize[qualla::QnnUtils::getPrefix(in_name, m_cacheGroupPrefixes)];
        } else {
          size_t ctx_size =
              m_cacheGroupCtxSize[qualla::QnnUtils::getPrefix(in_name, m_cacheGroupPrefixes)];
          size_t kv_out_channel =
              (tname.rfind("key") != std::string::npos ? kv_out.dims.channel : kv_out.dims.width);
          if (ctx_size < kv_out_channel) {
            QNN_ERROR("Invalid KV dimension: context size %zu is less than channel dimension %zu",
                      ctx_size,
                      kv_out_channel);
            return false;
          }
          kv_in_channel = ctx_size - kv_out_channel;
        }
        kv_offset =
            (kv_in_channel * kv_out.dims.batch * kv_out.dims.height *
             (tname.rfind("key") != std::string::npos ? kv_out.dims.width : kv_out.dims.channel));
        graph_allocs[in_name] = {alloc_idx, offset, kv_offset};
      }
    }
    graph_allocs[tname] = {alloc_idx, offset + kv_offset, size};
  }
  auto& curContextHandle = m_contextVec[m_graphIdxToContextIdx[graphIdx]];
  if (!m_ioTensor->mapFusedBufferOffset(m_graphsInfo[graphIdx], curContextHandle, graph_allocs)) {
    QNN_ERROR("Error mapping tensor to allocation buffers");
    return false;
  }

#if QNN_IO_TENSOR_DEBUG
  for (auto [tname, data] : graph_allocs) {
    QNN_DEBUG("Tensor Name: %s Alloc Idx: %d Tensor Offset: %zu Tensor Size: %zu",
              tname.c_str(),
              get<0>(data),
              get<1>(data),
              get<2>(data));
  }
#endif

  return true;
}

#ifdef QUALLA_ENGINE_QNN_HTP
bool QnnApi::createFromBinaryHtp(std::vector<std::string> cachedBinariesPathVec,
                                 size_t spillFillBufferSize,
                                 uint64_t mmap_budget,
                                 bool graphSwitching,
                                 const std::vector<std::string>& execSelectGraphs,
                                 bool loadSelectGraphs,
                                 bool skipLoraValidation) {
  GENIE_TRACE();
  // Let backendExtensions populate configs
  QnnContext_Config_t** customConfigs{nullptr};
  uint32_t customConfigCount{0};
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeCreateFromBinary(&customConfigs,
                                                                  &customConfigCount)) {
      QNN_ERROR("Extensions Failure in beforeCreateFromBinary()");
      return false;
    }
  }

  // baseConfigList holds configs that are common to all contexts.
  ContextConfigList baseConfigList = ContextConfigList::fromArray(customConfigs, customConfigCount);

  // Reading Binary Buffer and storing for later use during Deserialization
  std::vector<std::shared_ptr<uint8_t>> bufferVec(cachedBinariesPathVec.size());
  // Stores sizes of all the Binary Buffers
  std::vector<uint64_t> allBuffSizes(cachedBinariesPathVec.size());
  // Stores graphs per Contexts

  for (size_t contextIdx = 0; contextIdx < cachedBinariesPathVec.size(); contextIdx++) {
    auto _start = std::chrono::steady_clock::now();  // context Loading start

    uint64_t bufferSize{0};
    std::shared_ptr<uint8_t>& buffer{bufferVec[contextIdx]};
    // read serialized binary into a byte buffer
    bufferSize               = m_resourceManager->getBufferSize(cachedBinariesPathVec[contextIdx]);
    allBuffSizes[contextIdx] = bufferSize;
    if (0 == bufferSize) {
      QNN_ERROR("Received path to an empty file for context index = %zu. Nothing to deserialize.",
                contextIdx);
      return false;
    }

    // Inspect binary info
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS !=
        m_resourceManager->getQnnSystemInterface().systemContextCreate(&sysCtxHandle)) {
      QNN_ERROR("Could not create system handle for context index = %zu", contextIdx);
      return false;
    }

    const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
    if (!mapAndGetContextBinaryInfo(m_mmapContextBins,
                                    buffer,
                                    cachedBinariesPathVec[contextIdx],
                                    bufferSize,
                                    contextIdx,
                                    graphSwitching,
                                    sysCtxHandle,
                                    &binaryInfo)) {
      QNN_ERROR("Failed to map context Binary for contextIdx: %zu", contextIdx);
      return false;
    }

    auto _stop     = std::chrono::steady_clock::now();  // context Loading stop
    auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count();
    static_cast<void>(_duration);
    QNN_DEBUG("Loading contexts[%lu] took: %lld us", contextIdx, _duration);
    m_resourceManager->getQnnSystemInterface().systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;
  }

  // Iterate over all the tensors across the graphs Info and build info about the IO space it is
  // requiring.
  if (false == parseIOTensorsAndAccumulate()) {
    QNN_ERROR("Error in parsing the IO tensor info for all context binaries");
    return false;
  }

  // Spill-Fill configuration
  Qnn_ContextHandle_t first_contextHandle{nullptr};

  if (true !=
      getContextConfigs(baseConfigList, graphSwitching, execSelectGraphs, loadSelectGraphs)) {
    QNN_ERROR("Couldn't populate context configs");
    return false;
  }

  // I/O estimation configuration
  bool ioMemEstimationEnable = true;
#if defined(__linux__) && defined(__x86_64__)
  ioMemEstimationEnable = false;
#elif defined(__QNX__) || (defined(__aarch64__) && defined(__linux__))
  const QnnDevice_PlatformInfo_t* platformInfo{nullptr};
  if (nullptr != m_qnnInterface.deviceGetPlatformInfo) {
    auto qnnStatus = m_qnnInterface.deviceGetPlatformInfo(nullptr, &platformInfo);
    if (QNN_SUCCESS != qnnStatus) {
      QNN_ERROR("Failed to get platform info.");
      return false;
    }
  }
  if (platformInfo->v1.hwDevices->v1.numCores > 1) {
    ioMemEstimationEnable = false;
  }
#endif /* #if defined (__QNX__) || (defined(__aarch64__) && \
          defined(__linux__)) */
  if (ioMemEstimationEnable) {
    QnnHtpContext_CustomConfig_t ioMemEstimation;
    ioMemEstimation.option          = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
    ioMemEstimation.ioMemEstimation = true;
    baseConfigList.add(std::make_unique<ContextCustomHtpConfig>(ioMemEstimation));
  }
  if (mmap_budget > 0) {
    QnnHtpContext_CustomConfig_t customConfigReadBudget;
    customConfigReadBudget.option = QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET;
    customConfigReadBudget.fileReadMemoryBudgetInMb = mmap_budget;
    baseConfigList.add(std::make_unique<ContextCustomHtpConfig>(customConfigReadBudget));
  }

  if (skipLoraValidation) {
    QnnHtpContext_CustomConfig_t customConfigSkipLoraValidation;
    customConfigSkipLoraValidation.option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION;
    customConfigSkipLoraValidation.skipValidationOnBinarySection = true;
    baseConfigList.add(std::make_unique<ContextCustomHtpConfig>(customConfigSkipLoraValidation));
  }
  size_t graphIdx = 0;
  for (size_t contextIdx = 0; contextIdx < cachedBinariesPathVec.size(); contextIdx++) {
    if (nullptr == m_qnnInterface.contextCreateFromBinary) {
      QNN_ERROR("contextCreateFromBinaryFnHandle is nullptr for context index = %zu", contextIdx);
      freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }

    Qnn_ContextHandle_t contextHandle{nullptr};

    ContextConfigList configList(baseConfigList);
    if (spillFillBufferSize > 0) {
      QnnHtpContext_CustomConfig_t customConfigSF;
      customConfigSF.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
      QnnHtpContext_GroupRegistration_t groupInfo;
      if (contextIdx == 0) {
        groupInfo.firstGroupHandle = 0x0;
      } else {
        groupInfo.firstGroupHandle = first_contextHandle;
      }
      groupInfo.maxSpillFillBuffer     = spillFillBufferSize;
      customConfigSF.groupRegistration = groupInfo;
      configList.add(std::make_unique<ContextCustomHtpConfig>(customConfigSF));
    }

    const QnnContext_Config_t** contextConfigs =
        static_cast<const QnnContext_Config_t**>(configList);

    auto start = std::chrono::steady_clock::now();  // context Deserialization starts

    auto errCode = m_qnnInterface.contextCreateFromBinary(
        m_backendHandle,
        m_deviceHandle,
        contextConfigs,
        const_cast<const void*>(static_cast<void*>(bufferVec[contextIdx].get())),
        allBuffSizes[contextIdx],
        &contextHandle,
        nullptr  // profile handle
    );

    auto stop     = std::chrono::steady_clock::now();  // context Deserialization stops
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    static_cast<void>(duration);
    QNN_DEBUG("Initializing context[%lu] with %zu graphs took: %lld us",
              contextIdx,
              static_cast<size_t>(m_graphCountPerContext[contextIdx]),
              duration);
    if (contextIdx == 0 && true != allocateAll()) {  // allocate
      QNN_ERROR("Failed to allocate memory for IO tensors.");
      freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }

    if (errCode != QNN_SUCCESS) {
      QNN_ERROR("Could not create context from binary for context index = %zu : err %lu",
                contextIdx,
                static_cast<unsigned long>(errCode));
      freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }

    // Clearing buffer which is deseralized to reduce Memory footprint
    bufferVec[contextIdx].reset();

    if (m_profileBackendHandle) {
      extractBackendProfilingInfo(m_profileBackendHandle);
    }

    m_contextVec.push_back(contextHandle);
    m_contextIdxToHandle[contextIdx] = contextHandle;
    for (size_t n_graph = 0; n_graph < m_graphCountPerContext[contextIdx]; n_graph++) {
      qnn_wrapper_api::GraphInfo_t* cur_graph = m_graphsInfo[graphIdx];
      m_contextMap[cur_graph]                 = contextHandle;

      if (nullptr == m_qnnInterface.graphRetrieve) {
        QNN_ERROR("graphRetrieveFnHandle is nullptr.");
        freeGraphsInfo(&m_graphsInfo, m_graphsCount);
        return false;
      }

      if (!m_graphsInfo ||
          QNN_SUCCESS != m_qnnInterface.graphRetrieve(
                             contextHandle, cur_graph->graphName, &(cur_graph->graph))) {
        QNN_ERROR("Unable to retrieve graph handle for graph index = %zu", graphIdx);
        freeGraphsInfo(&m_graphsInfo, m_graphsCount);
        return false;
      }

      // Register all the Tensors per graph.
      if (false == registerTensorsWithBackend(graphIdx)) {
        QNN_ERROR("Unable to MemRegister IO Tensors for graph index = %zu", graphIdx);
        freeGraphsInfo(&m_graphsInfo, m_graphsCount);
        return false;
      }
      graphIdx++;
    }

    if (spillFillBufferSize > 0 && contextIdx == 0) {
      first_contextHandle = contextHandle;
    }
  }

  m_isContextCreated = true;

  QNN_DEBUG("Initialized %zu graphs from %lu contexts",
            static_cast<size_t>(m_graphsCount),
            cachedBinariesPathVec.size());

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterCreateFromBinary()) {
      QNN_ERROR("Extensions Failure in afterCreateFromBinary()");
      return false;
    }
  }

  return true;
}
#endif  // QUALLA_ENGINE_QNN_HTP

bool QnnApi::checkCapabilityOfCreateAsync(bool& propRet) {
  if (nullptr == m_qnnInterface.propertyHasCapability) {
    QNN_ERROR("propertyHasCapability is nullptr.......");
    return false;
  }
  if (QNN_PROPERTY_SUPPORTED == m_qnnInterface.propertyHasCapability(
                                    QNN_PROPERTY_CONTEXT_SUPPORT_CREATE_FROM_BINARY_LIST_ASYNC)) {
    propRet = true;
  } else {
    propRet = false;
  }
  return true;
}

bool freeContextParams(QnnContext_Params_t** contextParamsList, size_t numParams) {
  if (contextParamsList == nullptr || *contextParamsList == nullptr) {
    return false;
  }
  for (size_t i = 0; i < numParams; i++) {
    if (nullptr != contextParamsList[i]) {
      delete contextParamsList[i];
    }
  }
  return true;
}

void QnnApi::contextNotifyFn(Qnn_ContextHandle_t context,
                             Qnn_GraphHandle_t graph,
                             const char* graph_name,
                             QnnContext_createFromBinaryAsyncNotifyType_t completeType,
                             void* notifyParam,
                             Qnn_ErrorHandle_t /*status*/) {
  std::pair<QnnApi*, uint32_t>* pair = reinterpret_cast<std::pair<QnnApi*, uint32_t>*>(notifyParam);
  QnnApi* QnnApi                     = pair->first;
  uint32_t contextId                 = pair->second;

  if (completeType ==
      QnnContext_createFromBinaryAsyncNotifyType_t::QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT) {
    QnnApi->updateContext(context, contextId);
  } else if (completeType ==
             QnnContext_createFromBinaryAsyncNotifyType_t::QNN_CONTEXT_NOTIFY_TYPE_GRAPH_INIT) {
    QnnApi->updateQnnApiGraphsandContextsInfo(graph_name, graph, contextId);
  }
}

#ifdef QUALLA_ENGINE_QNN_HTP
bool QnnApi::createFromBinaryListAsyncHtp(std::vector<std::string> cachedBinariesPathVec,
                                          size_t /*spillFillBufferSize*/,
                                          uint64_t mmap_budget,
                                          bool graphSwitching,
                                          const std::vector<std::string>& execSelectGraphs,
                                          bool loadSelectGraphs,
                                          bool skipLoraValidation,
                                          bool lazyIOInitialization) {
  GENIE_TRACE();
  auto _start = std::chrono::steady_clock::now();

  // Let backendExtensions populate configs
  QnnContext_Config_t** customConfigs{nullptr};
  uint32_t customConfigCount{0};
  std::map<std::string, std::tuple<QnnContext_Config_t**, uint32_t>> contextKeyToCustomConfigsMap;
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeCreateContextsFromBinaryList(
            &contextKeyToCustomConfigsMap, &customConfigs, &customConfigCount)) {
      QNN_ERROR("Extensions Failure in beforeCreateContextsFromBinaryList()");
      return false;
    }
  }

  // groupConfigList holds configs that are common to all contexts.
  ContextConfigList groupConfigList =
      ContextConfigList::fromArray(customConfigs, customConfigCount);
  const QnnContext_Config_t** groupConfigs =
      static_cast<const QnnContext_Config_t**>(groupConfigList);

  // contextConfigList contains "per-context" configs provided to the context params lists
  ContextConfigList contextConfigList;
  if (true !=
      getContextConfigs(contextConfigList, graphSwitching, execSelectGraphs, loadSelectGraphs)) {
    QNN_ERROR("Couldn't populate context configs");
    return false;
  }

  if (mmap_budget > 0) {
    QnnHtpContext_CustomConfig_t customConfigReadBudget;
    customConfigReadBudget.option = QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET;
    customConfigReadBudget.fileReadMemoryBudgetInMb = mmap_budget;
    contextConfigList.add(std::make_unique<ContextCustomHtpConfig>(customConfigReadBudget));
  }

  if (skipLoraValidation) {
    QnnHtpContext_CustomConfig_t customConfigSkipLoraValidation;
    customConfigSkipLoraValidation.option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION;
    customConfigSkipLoraValidation.skipValidationOnBinarySection = true;
    contextConfigList.add(std::make_unique<ContextCustomHtpConfig>(customConfigSkipLoraValidation));
  }

  const QnnContext_Config_t** contextConfigs =
      static_cast<const QnnContext_Config_t**>(contextConfigList);

  std::vector<QnnContext_Params_t*> contextParamsList(cachedBinariesPathVec.size() + 1, nullptr);
  std::vector<std::shared_ptr<uint8_t>> bufferVec(cachedBinariesPathVec.size());

  for (size_t contextIdx = 0; contextIdx < cachedBinariesPathVec.size(); contextIdx++) {
    uint64_t bufferSize{0};
    std::shared_ptr<uint8_t>& buffer{bufferVec[contextIdx]};
    // read serialized binary into a byte buffer
    bufferSize = m_resourceManager->getBufferSize(cachedBinariesPathVec[contextIdx]);
    if (0 == bufferSize) {
      QNN_ERROR("Received path to an empty file for context index = %zu. Nothing to deserialize.",
                contextIdx);
      return false;
    }

    // Inspect binary info
    QnnSystemContext_Handle_t sysCtxHandle = nullptr;
    if (QNN_SUCCESS !=
        m_resourceManager->getQnnSystemInterface().systemContextCreate(&sysCtxHandle)) {
      QNN_ERROR("Could not create system handle for context index = %zu", contextIdx);
      return false;
    }
    const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
    if (!mapAndGetContextBinaryInfo(m_mmapContextBins,
                                    buffer,
                                    cachedBinariesPathVec[contextIdx],
                                    bufferSize,
                                    contextIdx,
                                    graphSwitching,
                                    sysCtxHandle,
                                    &binaryInfo)) {
      QNN_ERROR("Failed to map context Binary.");
      return false;
    }
    m_resourceManager->getQnnSystemInterface().systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    if (m_profileBackendHandle) {
      extractBackendProfilingInfo(m_profileBackendHandle);
    }
    // Passing class QnnApi pointer into callback funtion(notifyFn)
    std::pair<QnnApi*, uint32_t>* notifyParam =
        new std::pair<QnnApi*, uint32_t>(this, static_cast<size_t>(contextIdx));

    QnnContext_Params_t* contextParam = new QnnContext_Params_t{
        .version = QNN_CONTEXT_PARAMS_VERSION_1,
        .v1      = QnnContext_ParamsV1_t{contextConfigs,
                                    const_cast<const void*>(static_cast<void*>(buffer.get())),
                                    bufferSize,
                                    nullptr,
                                    QnnApi::contextNotifyFn,
                                    static_cast<void*>(notifyParam)}};

    contextParamsList[contextIdx] = contextParam;
    auto _stop                    = std::chrono::steady_clock::now();
    auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count();
    static_cast<void>(_duration);
    QNN_DEBUG("Loading contexts[%lu] took: %lld us", contextIdx, _duration);
  }
  if (nullptr == m_qnnInterface.contextCreateFromBinaryListAsync) {
    QNN_ERROR("contextCreateFromBinaryListAsyncFnHandle is nullptr");
    freeContextParams(contextParamsList.data(), cachedBinariesPathVec.size());
    return false;
  }
  auto start   = std::chrono::steady_clock::now();
  auto errCode = m_qnnInterface.contextCreateFromBinaryListAsync(
      m_backendHandle,
      m_deviceHandle,
      const_cast<const QnnContext_Params_t**>(contextParamsList.data()),
      groupConfigs,
      nullptr);
  auto stop     = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  static_cast<void>(duration);
  QNN_DEBUG("Initializing %lu context took: %lld us", cachedBinariesPathVec.size(), duration);

  // Explicitly free the context binary buffers. This ensures that the lifecycle
  // of the buffers outlasts the API call where their raw pointers are referenced.
  for (auto contextBinaryBuffer : bufferVec) {
    QNN_DEBUG("Freeing context binary buffer @%p", contextBinaryBuffer.get());
    contextBinaryBuffer.reset();
  }
  // Iterate over all the tensors across the graphs Info and build info about the IO space it is
  // requiring.
  if (false == parseIOTensorsAndAccumulate()) {
    QNN_ERROR("Error in parsing the IO tensor info for all context binaries");
    return false;
  }
  if (errCode != QNN_SUCCESS) {
    QNN_ERROR("Could not create context from binary List Async for context, err %lu",
              static_cast<unsigned long>(errCode));
    freeContextParams(contextParamsList.data(), cachedBinariesPathVec.size());
    return false;
  }
  if (!lazyIOInitialization) {
    if (true != allocateAll()) {
      QNN_ERROR("Failed to Allocate the buffers");
      return false;
    }

    if (true != registerAll()) {
      QNN_ERROR("Failed to Register the buffers");
      return false;
    }
  }
  // set graphInfo in m_graphsInfo
  size_t graphIdx = 0;
  for (size_t i = 0; i < m_graphCountPerContext.size(); i++) {
    for (uint32_t graphCount = 0; graphCount < m_graphCountPerContext[i]; graphCount++) {
      m_contextMap[m_graphsInfo[graphIdx]] = m_contextIdxToHandle[i];
      graphIdx++;
    }
  }
  m_isContextCreated = true;

  if (true != freeContextParams(contextParamsList.data(), cachedBinariesPathVec.size())) {
    QNN_ERROR("Couldn't free context params list");
    return false;
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterCreateContextsFromBinaryList()) {
      QNN_ERROR("Extensions Failure in afterCreateContextsFromBinaryList()");
      return false;
    }
  }
  return true;
}
#endif  // QUALLA_ENGINE_QNN_HTP

bool QnnApi::populateGraphBinaryInfo(std::vector<std::string> cachedBinariesPathVec,
                                     bool /*graphSwitching*/,
                                     bool useMmap) {
  GENIE_TRACE();
  std::vector<uint64_t> allBuffSizes(cachedBinariesPathVec.size());
  std::vector<std::shared_ptr<uint8_t>> bufferVec(cachedBinariesPathVec.size());
  for (size_t contextIdx = 0; contextIdx < cachedBinariesPathVec.size(); contextIdx++) {
    auto _start = std::chrono::steady_clock::now();  // context Loading start

    uint64_t bufferSize{0};
    std::shared_ptr<uint8_t>& buffer{bufferVec[contextIdx]};
    uint32_t graphsCount{0};
    // read serialized binary into a byte buffer
    bufferSize               = m_resourceManager->getBufferSize(cachedBinariesPathVec[contextIdx]);
    allBuffSizes[contextIdx] = bufferSize;
    if (0 == bufferSize) {
      QNN_ERROR("Received path to an empty file for context index = %zu. Nothing to deserialize.",
                contextIdx);
      return false;
    }
    // Inspect binary info
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS !=
        m_resourceManager->getQnnSystemInterface().systemContextCreate(&sysCtxHandle)) {
      QNN_ERROR("Could not create system handle for context index = %zu", contextIdx);
      return false;
    }
    const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
    if (!mapAndGetContextBinaryInfo(useMmap,
                                    buffer,
                                    cachedBinariesPathVec[contextIdx],
                                    bufferSize,
                                    contextIdx,
                                    false,
                                    sysCtxHandle,
                                    &binaryInfo)) {
      QNN_ERROR("Failed to map context Binary for contextIdx: %zu", contextIdx);
      return false;
    }
    qnn_wrapper_api::GraphInfo_t** graphsInfo{nullptr};
    if (!copyMetadataToGraphsInfo(binaryInfo, graphsInfo, graphsCount)) {
      QNN_ERROR("Failed to copy metadata for graph index = %zu", contextIdx);
      freeGraphsInfo(&graphsInfo, graphsCount);
      if (contextIdx > 0) freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }
    m_graphCountPerContext.push_back(graphsCount);
    if (m_graphsInfo == nullptr) {
      m_graphsInfo = reinterpret_cast<qnn_wrapper_api::GraphInfo_t**>(
          calloc(graphsCount, sizeof(qnn_wrapper_api::GraphInfo_t*)));
    } else {
      m_graphsInfo = reinterpret_cast<qnn_wrapper_api::GraphInfo_t**>(
          realloc(m_graphsInfo,
                  sizeof(qnn_wrapper_api::GraphInfo_t*) *
                      size_t(std::accumulate(
                          m_graphCountPerContext.begin(), m_graphCountPerContext.end(), 0))));
    }
    auto _stop     = std::chrono::steady_clock::now();  // context Loading stop
    auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count();
    static_cast<void>(_duration);
    QNN_DEBUG("Populating Graph binary info[%lu] took: %lld us", contextIdx, _duration);
    for (size_t gIdx = 0; gIdx < graphsCount; gIdx++) {
      std::string graph_name                = graphsInfo[gIdx]->graphName;
      m_graphsInfo[m_graphsCount]           = graphsInfo[gIdx];
      m_graphNameToInfo[graph_name]         = graphsInfo[gIdx];
      m_graphNameToContextIdx[graph_name]   = static_cast<uint32_t>(contextIdx);
      m_graphIdxToContextIdx[m_graphsCount] = static_cast<uint32_t>(contextIdx);
      m_graphsCount++;
    }
    if (graphsInfo) {
      free(graphsInfo);
      graphsInfo = nullptr;
    }
    m_resourceManager->getQnnSystemInterface().systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;
    // Clearing buffer which is deseralized to reduce Memory footprint
    bufferVec[contextIdx].reset();
  }
  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    m_graphNameToIndex[m_graphsInfo[graphIdx]->graphName] = graphIdx;
  }
  return true;
}

#ifdef QUALLA_ENGINE_QNN_HTP
bool QnnApi::initializeHtp(std::string backendPath,
                           std::vector<std::string> modelPathOrCachedBinaryPathVec,
                           BackendExtensionsConfigs backendExtensionsConfig,
                           std::vector<GraphConfigs> graphConfigs,
                           bool loadFromCachedBinary,
                           std::string /*systemLibraryPath*/,
                           bool debugModeRequested,
                           size_t spillFillBufferSize,
                           bool mmapContextBins,
                           bool asyncInit,
                           uint64_t mmap_budget,
                           bool debug_qnn,
                           bool graphSwitching,
                           const std::vector<std::string>& execSelectGraphs,
                           bool loadSelectGraphs,
                           bool skipLoraValidation,
                           bool lazyIOInitialization,
                           uint32_t logLevel,
                           LogCallback inLogCallBack) {
  GENIE_TRACE();
  if (modelPathOrCachedBinaryPathVec.size() > 1 && false == loadFromCachedBinary) {
    QNN_ERROR(
        "Currently only 1 model file is supported for this framework! \
            Although multiple context files are supported!");
    return false;
  }

  m_mmapContextBins = mmapContextBins;

  // Setting up Debug mode
  m_debugModeRequested = debugModeRequested;
  if (m_debugModeRequested) {
    QNN_WARN("Warning: Debug mode set to true.");
  }

  // Initialize the QNN run time
  if (false == getQnnInterface(backendPath)) {
    QNN_ERROR("Qnn getQnnInterface FAILED!");
    return false;
  }

  if (!loadFromCachedBinary) {
    if (false == loadModel(modelPathOrCachedBinaryPathVec[0])) {
      QNN_ERROR("Loading model FAILED!");
      return false;
    }
  }

  QnnLog_Level_t qnnLogLevel = static_cast<QnnLog_Level_t>(logLevel);
  if (false == initializeLogging(qnnLogLevel, debug_qnn, inLogCallBack)) {
    QNN_ERROR("Unable to Initialize logging in backend");
    return false;
  }

  if (!backendExtensionsConfig.configFilePath.empty() &&
      false == initializeBackendExtensions(backendExtensionsConfig, debug_qnn, qnnLogLevel)) {
    QNN_WARN("Failure in initializing backend extensions.");
  }

  if (false == initializeBackend()) {
    QNN_ERROR("Qnn initializeBackend FAILED!");
    return false;
  }
  if (false == createDevice()) {
    QNN_ERROR("Device Creation failure");
    m_isDeviceCreated = false;
    return false;
  } else {
    m_isDeviceCreated = true;
  }
  if (!loadFromCachedBinary) {
    if (false == createContext()) {
      QNN_ERROR("Qnn createContext FAILED!");
      return false;
    }
    if (false == composeGraphs(graphConfigs)) {
      QNN_ERROR("composeGraphs FAILED!");
      return false;
    }
    if (false == finalizeGraphs()) {
      QNN_ERROR("finalizeGraphs FAILED!");
      return false;
    }
  } else {
    bool cfb_ret         = false;
    bool asyncCapability = false;
    if (asyncInit == true) {
      if (!checkCapabilityOfCreateAsync(asyncCapability)) {
        QNN_ERROR("Capabilty checked failed");
        return false;
      }
      asyncInit = asyncCapability && asyncInit;
    }

    if (asyncInit == true) {
      QNN_INFO("Using create From Binary List Async");
      cfb_ret = createFromBinaryListAsyncHtp(modelPathOrCachedBinaryPathVec,
                                             spillFillBufferSize,
                                             mmap_budget,
                                             graphSwitching,
                                             execSelectGraphs,
                                             loadSelectGraphs,
                                             skipLoraValidation,
                                             lazyIOInitialization);
      if (cfb_ret == false) {
        QNN_ERROR("Create From Binary List Async FAILED!");
        return false;
      }
    } else {
      QNN_INFO("Using create From Binary");
      cfb_ret = createFromBinaryHtp(modelPathOrCachedBinaryPathVec,
                                    spillFillBufferSize,
                                    mmap_budget,
                                    graphSwitching,
                                    execSelectGraphs,
                                    loadSelectGraphs,
                                    skipLoraValidation);
      if (false == cfb_ret) {
        QNN_ERROR("Create From Binary FAILED!");
        return false;
      }
    }
  }

#if NSP_LOG_LEVEL > 1
  for (const auto& graphNameIndex : m_graphNameToIndex) {
    QNN_DEBUG("Found Graph name %s corresponding to index %zu",
              graphNameIndex.first.c_str(),
              graphNameIndex.second);
  }

  fprintf(stderr, "context_handles = [");
  for (auto ctx_handle : m_contextVec) fprintf(stderr, "%p, ", ctx_handle);
  fprintf(stderr, "]\n");
#endif
  return true;
}
#endif  // QUALLA_ENGINE_QNN_HTP

bool QnnApi::initializeCpu(std::string backendPath,
                           std::string modelPath,
                           std::vector<GraphConfigs> graphConfigs,
                           uint32_t* inputDim,
                           uint32_t inputRank,
                           uint32_t* outputDim,
                           uint32_t outputRank,
                           uint32_t* kvDim,
                           uint32_t kvRank,
                           uint32_t* kvScaleDim,
                           Qnn_Param_t* params,
                           uint32_t numParams,
                           bool debugModeRequested,
                           bool debug_qnn,
                           uint32_t logLevel,
                           LogCallback inLogCallBack) {
  GENIE_TRACE();
  // Setting up Debug mode
  m_debugModeRequested = debugModeRequested;
  if (m_debugModeRequested) {
    QNN_WARN("Warning: Debug mode set to true.");
  }

  // Initialize the QNN run time
  if (false == getQnnInterface(backendPath)) {
    QNN_ERROR("Qnn getQnnInterface FAILED!");
    return false;
  }

  QnnLog_Level_t qnnLogLevel = static_cast<QnnLog_Level_t>(logLevel);
  if (false == initializeLogging(qnnLogLevel, debug_qnn, inLogCallBack)) {
    QNN_ERROR("Unable to Initialize logging in backend");
  }

  if (false == initializeBackend()) {
    QNN_ERROR("Qnn initializeBackend FAILED!");
    return false;
  }

  // CPU does not support createDevice.
  m_isDeviceCreated = false;

  if (false == loadModel(modelPath)) {
    QNN_ERROR("Loading model FAILED!");
    return false;
  }

  if (false == createContext()) {
    QNN_ERROR("Qnn createContext FAILED!");
    return false;
  }

  if (false == composeGraphs(graphConfigs,
                             inputDim,
                             inputRank,
                             outputDim,
                             outputRank,
                             kvDim,
                             kvRank,
                             kvScaleDim,
                             params,
                             numParams)) {
    QNN_ERROR("composeGraphs FAILED!");
    return false;
  }

  if (false == finalizeCpuGraphs()) {
    QNN_ERROR("finalizeGraphs FAILED!");
    return false;
  }

  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    m_graphNameToIndex[m_graphsInfo[graphIdx]->graphName] = graphIdx;
  }
#if NSP_LOG_LEVEL > 1
  for (const auto& graphNameIndex : m_graphNameToIndex) {
    QNN_DEBUG("Found Graph name %s corresponding to index %zu",
              graphNameIndex.first.c_str(),
              graphNameIndex.second);
  }
#endif
  return true;
}

bool QnnApi::graphExecute(Qnn_Tensor_t* input,
                          Qnn_Tensor_t* output,
                          std::string graphName,
                          std::map<std::string, std::pair<double, uint16_t>>& timeLogs) {
  qnn_wrapper_api::GraphInfo_t* graph_info = m_graphsInfo[m_graphNameToIndex[graphName]];
  return graphExecute(graph_info, input, output, timeLogs);
}

bool QnnApi::graphExecute(qnn_wrapper_api::GraphInfo_t* graph_info,
                          const Qnn_Tensor_t* input,
                          Qnn_Tensor_t* output,
                          std::map<std::string, std::pair<double, uint16_t>>& timeLogs) {
  GENIE_TRACE();
  std::string graphName = graph_info->graphName;
  QnnGraph_Config_t** customGraphConfigs{nullptr};
  uint32_t configCount{0};
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeExecute(
            graphName.c_str(), &customGraphConfigs, &configCount)) {
      QNN_ERROR("Extensions Failure in beforeExecute()");
      return false;
    }
    if (customGraphConfigs) {
      if (true !=
          setGraphConfigsBeforeExecute(graph_info->graph, customGraphConfigs, configCount)) {
        QNN_ERROR("Failure in setGraphConfigsBeforeExecute()");
        return false;
      }
    }
  }

  // if (true != boostPerformance()) {
  //     QNN_ERROR("Couldn't boost the performance");
  //     return false;
  // }

  Qnn_ErrorHandle_t ret = QNN_GRAPH_NO_ERROR;
  try {
#if NSP_LOG_LEVEL > 1
    auto start = std::chrono::steady_clock::now();
#endif

    ret = m_qnnInterface.graphExecute(graph_info->graph,
                                      input,
                                      graph_info->numInputTensors,
                                      output,
                                      graph_info->numOutputTensors,
                                      m_profileBackendHandle,
                                      nullptr);
#if NSP_LOG_LEVEL > 1
    auto stop     = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    QNN_DEBUG("graphExecute[%s] took: %lld us", graphName.c_str(), duration);
#endif
#if NSP_LOG_LEVEL > 6
    timeLogs[graphName].first += static_cast<double>(duration);
    timeLogs[graphName].second++;
#endif

  } catch (const std::exception& e) {
    QNN_ERROR("ERROR executing inference ret=%zu err=%s", static_cast<size_t>(ret), e.what());
  } catch (...) {
    QNN_ERROR("ERROR executing inference ret=%zu", static_cast<size_t>(ret));
  }

  if (m_profileBackendHandle) {
    extractBackendProfilingInfo(m_profileBackendHandle, timeLogs, graphName);
  }

  // if (true != resetPerformance()) {
  //     QNN_ERROR("Couldn't reset the performance");
  //     return false;
  // }

  if (ret != QNN_GRAPH_NO_ERROR) {
    QNN_ERROR("Failed to execute graph. Error %zu", static_cast<size_t>(ret));
    return false;
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterExecute()) {
      QNN_ERROR("Extensions Failure in afterExecute()");
      return false;
    }
  }

  return true;
}

bool QnnApi::extractBackendProfilingInfo(
    Qnn_ProfileHandle_t profileHandle,
    std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
    std::string graphName) {
  if (nullptr == m_profileBackendHandle) {
    QNN_ERROR("QNN HTP Profile handle is nullptr; may not be initialized.");
    return false;
  }
  const QnnProfile_EventId_t* profileEvents{nullptr};
  uint32_t numEvents{0};
  if (QNN_PROFILE_NO_ERROR !=
      m_qnnInterface.profileGetEvents(profileHandle, &profileEvents, &numEvents)) {
    QNN_ERROR("Failure in QNN HTP profile get events.");
    return false;
  }
  QNN_DEBUG("ProfileEvents: [%p], numEvents: [%d]", profileEvents, numEvents);
  for (size_t event = 0; event < numEvents; event++) {
    extractProfilingEvent(*(profileEvents + event), timeLogs, graphName);
    extractProfilingSubEvents(*(profileEvents + event), timeLogs, graphName);
  }
  return true;
}

bool QnnApi::extractProfilingSubEvents(QnnProfile_EventId_t profileEventId,
                                       std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
                                       std::string graphName) {
  const QnnProfile_EventId_t* profileSubEvents{nullptr};
  uint32_t numSubEvents{0};
  if (QNN_PROFILE_NO_ERROR !=
      m_qnnInterface.profileGetSubEvents(profileEventId, &profileSubEvents, &numSubEvents)) {
    QNN_ERROR("Failure in QNN HTP profile get sub events.");
    return false;
  }
  QNN_DEBUG("ProfileSubEvents: [%p], numSubEvents: [%d]", profileSubEvents, numSubEvents);
  for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
    extractProfilingEvent(*(profileSubEvents + subEvent), timeLogs, graphName);
    extractProfilingSubEvents(*(profileSubEvents + subEvent), timeLogs, graphName);
  }
  return true;
}

bool QnnApi::extractProfilingEvent(QnnProfile_EventId_t profileEventId,
                                   std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
                                   std::string graphName) {
  QnnProfile_EventData_t eventData;
  if (QNN_PROFILE_NO_ERROR != m_qnnInterface.profileGetEventData(profileEventId, &eventData)) {
    QNN_ERROR("Failure in profile get event type.");
    return false;
  }

  QNN_DEBUG(
      "Event Info - Event Type: [%d], Event Value: [%lu], Event Identifier: [%s], Event Unit: [%d]",
      eventData.type,
      eventData.value,
      eventData.identifier,
      eventData.unit);
#if NSP_LOG_LEVEL > 6
  timeLogs[graphName + "_" + eventData.identifier].first += static_cast<double>(eventData.value);
  timeLogs[graphName + "_" + eventData.identifier].second++;
#else
  static_cast<void>(timeLogs);
  static_cast<void>(graphName);
#endif

  return true;
}

bool QnnApi::extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle) {
  if (nullptr == m_profileBackendHandle) {
    QNN_ERROR("QNN HTP Profile handle is nullptr; may not be initialized.");
    return false;
  }
  const QnnProfile_EventId_t* profileEvents{nullptr};
  uint32_t numEvents{0};
  if (QNN_PROFILE_NO_ERROR !=
      m_qnnInterface.profileGetEvents(profileHandle, &profileEvents, &numEvents)) {
    QNN_ERROR("Failure in QNN HTP profile get events.");
    return false;
  }
  QNN_DEBUG("ProfileEvents: [%p], numEvents: [%d]", profileEvents, numEvents);
  for (size_t event = 0; event < numEvents; event++) {
    extractProfilingEvent(*(profileEvents + event));
    extractProfilingSubEvents(*(profileEvents + event));
  }
  return true;
}

bool QnnApi::extractProfilingSubEvents(QnnProfile_EventId_t profileEventId) {
  const QnnProfile_EventId_t* profileSubEvents{nullptr};
  uint32_t numSubEvents{0};
  if (QNN_PROFILE_NO_ERROR !=
      m_qnnInterface.profileGetSubEvents(profileEventId, &profileSubEvents, &numSubEvents)) {
    QNN_ERROR("Failure in QNN HTP profile get sub events.");
    return false;
  }
  QNN_DEBUG("ProfileSubEvents: [%p], numSubEvents: [%d]", profileSubEvents, numSubEvents);
  for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
    extractProfilingEvent(*(profileSubEvents + subEvent));
    extractProfilingSubEvents(*(profileSubEvents + subEvent));
  }
  return true;
}

bool QnnApi::extractProfilingEvent(QnnProfile_EventId_t profileEventId) {
  QnnProfile_EventData_t eventData;
  if (QNN_PROFILE_NO_ERROR != m_qnnInterface.profileGetEventData(profileEventId, &eventData)) {
    QNN_ERROR("Failure in profile get event type.");
    return false;
  }

  QNN_DEBUG(
      "Event Info - Event Type: [%d], Event Value: [%lu], Event Identifier: [%s], Event Unit: [%d]",
      eventData.type,
      eventData.value,
      eventData.identifier,
      eventData.unit);

  return true;
}

bool QnnApi::applyBinarySection(const std::string& graphName, const std::string& binSectionPath) {
  size_t graphId = m_graphNameToIndex[graphName];
  return applyBinarySection(graphId, binSectionPath);
}

bool QnnApi::applyBinarySection(size_t graphId, const std::string& binSectionPath) {
  // assumption splitNum  from 0
  QNN_DEBUG("QnnApi::applyBinarySection %zu ", graphId);

  if (graphId >= m_graphsCount) {
    QNN_ERROR(
        "Passed split %zu base Model graphcount %zu", graphId, static_cast<size_t>(m_graphsCount));
    return false;
  }

  uint64_t bufferSize{0};
  std::shared_ptr<uint8_t> buffer{nullptr};

  bufferSize = m_resourceManager->getBufferSize(binSectionPath);
  buffer     = std::shared_ptr<uint8_t>(new uint8_t[bufferSize]);
  if (!m_resourceManager->readBinaryFromFile(binSectionPath, buffer, bufferSize)) {
    QNN_ERROR("Failed to read binary data for context index = %zu", graphId);
    return false;
  }

  // beforeContextApplyBinarySection
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeContextApplyBinarySection()) {
      QNN_ERROR("Extensions Failure in beforeContextApplyBinarySection() graph = %zu", graphId);
      return false;
    }
  }

  // contextApplyBinarySection
  if (nullptr != m_qnnInterface.contextApplyBinarySection) {
    QnnContext_Buffer_t qnnBuffer;
    qnnBuffer.version               = QNN_CONTEXT_BUFFER_VERSION_1;
    qnnBuffer.v1.memType            = QNN_CONTEXTMEMTYPE_RAW;
    qnnBuffer.v1.binaryBuf.dataSize = bufferSize;
    qnnBuffer.v1.binaryBuf.data     = static_cast<void*>(buffer.get());
    size_t contextId                = m_graphIdxToContextIdx[graphId];

    auto contextHandle = m_contextVec[contextId];
    auto graphHandle   = m_graphsInfo[graphId]->graph;
    if (contextHandle == nullptr || graphHandle == nullptr) {
      QNN_ERROR(" contexthandle or graph handle is null for patch no = %zu", graphId);
      return false;
    }

    Qnn_ErrorHandle_t errorCode =
        m_qnnInterface.contextApplyBinarySection(contextHandle,
                                                 graphHandle,
                                                 QNN_CONTEXT_SECTION_UPDATABLE,
                                                 &qnnBuffer,
                                                 nullptr,   // profile handle is null
                                                 nullptr);  // singal handle is null
    if (errorCode != QNN_SUCCESS) {
      QNN_ERROR("Could not apply patch for graph = %zu errocode = %zu",
                static_cast<size_t>(graphId),
                static_cast<size_t>(errorCode));
      return false;
    }
  } else {
    QNN_ERROR("contextApplyBinarySection interface not supported!!");
    return false;
  }

  // afterContextApplyBinarySection
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterContextApplyBinarySection()) {
      QNN_ERROR("Extensions Failure in afterContextApplyBinarySection() graph = %zu", graphId);
      return false;
    }
  }

  return true;
}

bool QnnApi::applyBinarySection(size_t binIdx,
                                const std::string& binSectionPath,
                                bool useMmap,
                                bool graphSwitch,
                                std::string& lazyLora,
                                bool weightSharedLora) {
  // assumption splitNum  from 0
  QNN_DEBUG("QnnApi::applyBinarySection %zu ", binIdx);

  uint32_t numAdapterGraph = 0;
  if (nullptr == m_qnnInterface.contextApplyBinarySection) {
    QNN_ERROR("contextApplyBinarySection Interface not suported!!");
    return false;
  }
  if (binIdx >= m_graphsCount) {
    QNN_ERROR(
        "Passed split %zu base Model graphcount %zu", binIdx, static_cast<size_t>(m_graphsCount));
    return false;
  }

  uint64_t bufferSize{0};
  std::shared_ptr<uint8_t> buffer{nullptr};

  bufferSize = m_resourceManager->getBufferSize(binSectionPath);

  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  if (QNN_SUCCESS !=
      m_resourceManager->getQnnSystemInterface().systemContextCreate(&sysCtxHandle)) {
    QNN_ERROR("Could not create system handle for context index = %zu", binIdx);
    return false;
  }
  Qnn_ContextBinarySize_t binaryInfoSize{0};

  if (m_adapterNameToBuffer[binSectionPath]) {
    buffer = m_adapterNameToBuffer[binSectionPath];
    if (QNN_SUCCESS != m_resourceManager->getQnnSystemInterface().systemContextGetBinaryInfo(
                           sysCtxHandle,
                           static_cast<void*>(buffer.get()),
                           bufferSize,
                           &binaryInfo,
                           &binaryInfoSize)) {
      QNN_ERROR("Failed to get context binary info for context index = %zu", binIdx);
      return false;
    }
  } else {
    if (!mapAndGetContextBinaryInfo(useMmap,
                                    buffer,
                                    binSectionPath,
                                    bufferSize,
                                    binIdx,
                                    graphSwitch,
                                    sysCtxHandle,
                                    &binaryInfo)) {
      QNN_ERROR("Failed to map context Binary for contextIdx: %zu", binIdx);
      return false;
    }
    m_adapterNameToBuffer[binSectionPath] = buffer;
  }

  numAdapterGraph = getNumGraphInBinary(binaryInfo);
  m_resourceManager->getQnnSystemInterface().systemContextFree(sysCtxHandle);
  sysCtxHandle = nullptr;

  if (numAdapterGraph <= 0) {
    QNN_ERROR(" numAdapterGraph is <=0 ");
    return false;
  }

  size_t graphId = 0;
  for (size_t idx = 0; idx < numAdapterGraph; idx++) {
    graphId          = numAdapterGraph * binIdx + idx;
    size_t contextId = m_graphIdxToContextIdx[graphId];

    auto contextHandle = m_contextVec[contextId];
    auto graphHandle   = m_graphsInfo[graphId]->graph;
    if (contextHandle == nullptr || graphHandle == nullptr) {
      QNN_ERROR("Contexthandle or graph handle is null for patch no = %zu ", graphId);
      return false;
    }

    // beforeContextApplyBinarySection
    if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
      if (!m_backendExtensions->interface()->beforeContextApplyBinarySection()) {
        QNN_ERROR("Extensions Failure in beforeContextApplyBinarySection() graph = %zu", graphId);
        return false;
      }
    }

    QnnContext_Buffer_t qnnBuffer;
    qnnBuffer.version               = QNN_CONTEXT_BUFFER_VERSION_1;
    qnnBuffer.v1.memType            = QNN_CONTEXTMEMTYPE_RAW;
    qnnBuffer.v1.binaryBuf.dataSize = bufferSize;
    qnnBuffer.v1.binaryBuf.data     = static_cast<void*>(buffer.get());

    if ((graphSwitch && lazyLora == "lazy") || weightSharedLora) {
      // Cache info for deferred call during execute
      m_adapterCache[graphHandle] = std::make_tuple(contextHandle, qnnBuffer, graphId, false);
    } else {
      // contextApplyBinarySection
      Qnn_ErrorHandle_t errorCode =
          m_qnnInterface.contextApplyBinarySection(contextHandle,
                                                   graphHandle,
                                                   QNN_CONTEXT_SECTION_UPDATABLE,
                                                   &qnnBuffer,
                                                   nullptr,   // profile handle is null
                                                   nullptr);  // signal handle is null
      if (errorCode != QNN_SUCCESS) {
        QNN_ERROR("Could not Apply Patch for graph = %zu error code = %zu ",
                  graphId,
                  static_cast<size_t>(errorCode));
        return false;
      }
    }

    // afterContextApplyBinarySection
    if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
      if (!m_backendExtensions->interface()->afterContextApplyBinarySection()) {
        QNN_ERROR("Extensions Failure in afterContextApplyBinarySection() graph = %zu", graphId);
        return false;
      }
    }
  }

  if (updateIOEncodings(buffer, bufferSize, numAdapterGraph * binIdx) == false) {
    QNN_ERROR("qnn-htp: Adapter updateIOEncodings failed");
    return false;
  }

  return true;
}

bool QnnApi::setPerfProfile(qualla::PerformanceProfile& perfProfile) {
  qnn::tools::netrun::PerfProfile qnnPerfProfile =
      qualla::QnnUtils::quallaToQnnPerformanceProfile(perfProfile);
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (qnnPerfProfile != m_perfProfile)
      m_backendExtensions->interface()->setPerfProfile(qnnPerfProfile);
  }
  m_perfProfile = qnnPerfProfile;
  return true;
}

qualla::PerformanceProfile QnnApi::getPerfProfile() {
  return qualla::QnnUtils::qnnToQuallaPerformanceProfile(m_perfProfile);
}

bool QnnApi::applyCachedAdapter(Qnn_GraphHandle_t graphHandle) {
  auto contextHandle = std::get<0>(m_adapterCache[graphHandle]);
  auto qnnBuffer     = std::get<1>(m_adapterCache[graphHandle]);
  size_t graphId     = std::get<2>(m_adapterCache[graphHandle]);

  // beforeContextApplyBinarySection
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeContextApplyBinarySection()) {
      QNN_ERROR("Extensions Failure in beforeContextApplyBinarySection() graph = %zu", graphId);
      return false;
    }
  }

  // contextApplyBinarySection
  if (nullptr != m_qnnInterface.contextApplyBinarySection) {
    Qnn_ErrorHandle_t errorCode =
        m_qnnInterface.contextApplyBinarySection(contextHandle,
                                                 graphHandle,
                                                 QNN_CONTEXT_SECTION_UPDATABLE,
                                                 &qnnBuffer,
                                                 nullptr,   // profile handle is null
                                                 nullptr);  // signal handle is null
    if (errorCode != QNN_SUCCESS) {
      QNN_ERROR("Could not apply patch for graph = %zu error code = %zu ",
                graphId,
                static_cast<size_t>(errorCode));
      return false;
    }
  }

  // afterContextApplyBinarySection
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterContextApplyBinarySection()) {
      QNN_ERROR("Extensions Failure in afterContextApplyBinarySection() graph = %zu", graphId);
      return false;
    }
  }

  std::get<3>(m_adapterCache[graphHandle]) = true;
  return true;
}

bool QnnApi::updateSharedAdapterForGraph(Qnn_GraphHandle_t graphHandle) {
  std::lock_guard<std::mutex> guard{m_updateSharedAdapterMutex};
  if (m_currentAdapterGraphHandle != graphHandle) {
    if (!applyCachedAdapter(graphHandle)) {
      QNN_ERROR(
          "Could not apply weight-shared cached lora adapter.Requires an adapter to be applied "
          "before execution");
      return false;
    }
    m_currentAdapterGraphHandle = graphHandle;
  }
  return true;
}

bool QnnApi::updateIOEncodings(std::shared_ptr<uint8_t>& buffer,
                               size_t bufferSize,
                               size_t graphIdx) {
  QNN_DEBUG("Applying adapter Encodings");
  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  if (QNN_SUCCESS !=
      m_resourceManager->getQnnSystemInterface().systemContextCreate(&sysCtxHandle)) {
    QNN_ERROR("Could not create system handle for context index = %zu", graphIdx);
    return false;
  }
  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  Qnn_ContextBinarySize_t binaryInfoSize{0};
  if (QNN_SUCCESS != m_resourceManager->getQnnSystemInterface().systemContextGetBinaryInfo(
                         sysCtxHandle,
                         static_cast<void*>(buffer.get()),
                         bufferSize,
                         &binaryInfo,
                         &binaryInfoSize)) {
    QNN_ERROR("Failed to get context binary info for context index = %zu", graphIdx);
    return false;
  }

  uint32_t graphIdxU32 = static_cast<uint32_t>(graphIdx);
  if (!updateMetaDataToGraphsInfo(binaryInfo, m_graphsInfo, graphIdxU32)) {
    QNN_ERROR("Failed to copy metadata for graph index = %zu", graphIdx);
    return false;
  }
  m_resourceManager->getQnnSystemInterface().systemContextFree(sysCtxHandle);
  sysCtxHandle = nullptr;
  QNN_DEBUG(" updateIOEncodings success ");
  return true;
}

bool QnnApi::createFromBinaryGpu(std::vector<std::string> cachedBinariesPathVec) {
  auto _start = std::chrono::steady_clock::now();

  for (size_t contextIdx = 0; contextIdx < cachedBinariesPathVec.size(); contextIdx++) {
    uint64_t bufferSize{0};
    std::shared_ptr<uint8_t> buffer{nullptr};
    uint32_t graphsCount{0};
    // read serialized binary into a byte buffer
    bufferSize = m_resourceManager->getBufferSize(cachedBinariesPathVec[contextIdx]);
    if (0 == bufferSize) {
      QNN_ERROR("Received path to an empty file for context index = %zu. Nothing to deserialize.",
                contextIdx);
      return false;
    }

    // Inspect binary info
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS !=
        m_resourceManager->getQnnSystemInterface().systemContextCreate(&sysCtxHandle)) {
      QNN_ERROR("Could not create system handle for context index = %zu", contextIdx);
      return false;
    }

    const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
    bool useMmap           = true;
    const bool graphSwitch = false;

    if (!mapAndGetContextBinaryInfo(useMmap,
                                    buffer,
                                    cachedBinariesPathVec[contextIdx],
                                    bufferSize,
                                    contextIdx,
                                    graphSwitch,
                                    sysCtxHandle,
                                    &binaryInfo)) {
      QNN_ERROR("Failed to map context Binary for contextIdx: %zu", contextIdx);
      return false;
    }

    qnn_wrapper_api::GraphInfo_t** graphsInfo{nullptr};
    if (!copyMetadataToGraphsInfo(binaryInfo, graphsInfo, graphsCount)) {
      QNN_ERROR("Failed to copy metadata for graph index = %zu", contextIdx);
      freeGraphsInfo(&graphsInfo, graphsCount);
      if (contextIdx > 0) freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }

    m_graphCountPerContext.push_back(graphsCount);
    if (m_graphsInfo == nullptr) {
      m_graphsInfo = reinterpret_cast<qnn_wrapper_api::GraphInfo_t**>(
          calloc(graphsCount, sizeof(qnn_wrapper_api::GraphInfo_t*)));
    } else {
      m_graphsInfo = reinterpret_cast<qnn_wrapper_api::GraphInfo_t**>(
          realloc(m_graphsInfo,
                  sizeof(qnn_wrapper_api::GraphInfo_t*) *
                      size_t(std::accumulate(
                          m_graphCountPerContext.begin(), m_graphCountPerContext.end(), 0))));
    }
    m_resourceManager->getQnnSystemInterface().systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    if (nullptr == m_qnnInterface.contextCreateFromBinary) {
      QNN_ERROR("contextCreateFromBinaryFnHandle is nullptr for context index = %zu", contextIdx);
      freeGraphsInfo(&graphsInfo, graphsCount);
      if (contextIdx > 0) freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }
    Qnn_ContextHandle_t contextHandle{nullptr};
    auto _stop     = std::chrono::steady_clock::now();
    auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count();
    static_cast<void>(_duration);
    QNN_DEBUG("Loading contexts[%lu] took: %lld us", contextIdx, _duration);

    auto start = std::chrono::steady_clock::now();

    auto errCode = m_qnnInterface.contextCreateFromBinary(
        m_backendHandle,
        m_deviceHandle,
        nullptr,
        const_cast<const void*>(static_cast<void*>(buffer.get())),
        bufferSize,
        &contextHandle,
        nullptr  // profile handle

    );

    if (errCode != QNN_SUCCESS) {
      QNN_ERROR("Could not create context from binary for context index = %zu : err %zu",
                contextIdx,
                static_cast<size_t>(errCode));
      freeGraphsInfo(&graphsInfo, graphsCount);
      if (contextIdx > 0) freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }

    auto stop     = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    static_cast<void>(duration);
    QNN_DEBUG("Initializing context[%lu] with %u graphs took: %lld us",
              contextIdx,
              graphsCount,
              duration);

    for (size_t n_graph = 0; n_graph < graphsCount; n_graph++) {
      // Allocate inputTensors and outputTensors
      qnn_wrapper_api::GraphInfo_t* cur_graph = graphsInfo[n_graph];

      m_graphsInfo[m_graphsCount++] = cur_graph;
      m_contextMap[cur_graph]       = contextHandle;
    }
    m_contextVec.push_back(contextHandle);
  }

  m_isContextCreated = true;

  QNN_DEBUG("Initialized %u graphs from %lu contexts", m_graphsCount, cachedBinariesPathVec.size());

  if (nullptr == m_qnnInterface.graphRetrieve) {
    QNN_ERROR("graphRetrieveFnHandle is nullptr.");
    freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    return false;
  }

  size_t graphIdx = 0;
  for (size_t i = 0; i < m_graphCountPerContext.size(); i++) {
    for (uint32_t graphCount = 0; graphCount < m_graphCountPerContext[i]; graphCount++) {
      if (!m_graphsInfo ||
          QNN_SUCCESS != m_qnnInterface.graphRetrieve(m_contextVec[i],
                                                      m_graphsInfo[graphIdx]->graphName,
                                                      &(m_graphsInfo[graphIdx]->graph))) {
        QNN_ERROR("Unable to retrieve graph handle for graph index = %zu", graphIdx);
        freeGraphsInfo(&m_graphsInfo, m_graphsCount);
        return false;
      }
      graphIdx++;
    }
  }

  return true;
}

bool QnnApi::initializeGpu(std::string backendPath,
                           std::vector<std::string> modelPathOrCachedBinaryPath,
                           bool debug_qnn,
                           uint32_t logLevel,
                           LogCallback inLogCallBack) {
  GENIE_TRACE();
  if (modelPathOrCachedBinaryPath.size() != 1) {
    QNN_ERROR("Multiple Files not supported for now!!");
    return false;
  }

  if (false == getQnnInterface(backendPath)) {
    QNN_ERROR("Qnn getQnnInterface FAILED!");
    return false;
  }

  QnnLog_Level_t qnnLogLevel = static_cast<QnnLog_Level_t>(logLevel);
  if (false == initializeLogging(qnnLogLevel, debug_qnn, inLogCallBack)) {
    QNN_ERROR("Unable to Initialize logging in backend");
    return false;
  }

  // Initialize Backend
  if (false == initializeBackend()) {
    QNN_ERROR("Qnn initializeBackend FAILED!");
    return false;
  }

  if (false == createFromBinaryGpu(modelPathOrCachedBinaryPath)) {
    QNN_ERROR("Create From Binary FAILED!");
    return false;
  }

  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    m_graphNameToIndex[m_graphsInfo[graphIdx]->graphName] = graphIdx;
  }
  QNN_DEBUG("Model Initialized");

  return true;
}

bool QnnApi::setOemKey(const std::string& oemKey) {
  if (nullptr == m_qnnInterface.propertyHasCapability) {
    QNN_ERROR("propertyHasCapability is nullptr.");
    return false;
  }

  if (m_qnnInterface.propertyHasCapability(QNN_PROPERTY_BACKEND_SUPPORT_PLATFORM_OPTIONS) !=
      QNN_PROPERTY_SUPPORTED) {
    QNN_ERROR("Backend does not support QNN_PROPERTY_BACKEND_SUPPORT_PLATFORM_OPTIONS");
  }

  if (nullptr == m_qnnInterface.backendSetConfig) {
    QNN_ERROR("backendSetConfig is nullptr.");
    return false;
  }

  QnnBackend_Config_t backendConfig           = QNN_BACKEND_CONFIG_INIT;
  std::string oem_string                      = "oem:" + oemKey;
  backendConfig.option                        = QNN_BACKEND_CONFIG_OPTION_PLATFORM;
  backendConfig.platformOption                = oem_string.c_str();
  const QnnBackend_Config_t* backendConfigs[] = {&backendConfig, nullptr};

  Qnn_ErrorHandle_t err = m_qnnInterface.backendSetConfig(
      m_backendHandle, const_cast<const QnnBackend_Config_t**>(backendConfigs));
  if (QNN_SUCCESS != err) {
    QNN_ERROR("backendSetConfig for OEM key failed.");
    return false;
  }
  return true;
}

bool QnnApi::setExecutionPriority(Qnn_Priority_t priority) {
  if (nullptr == m_qnnInterface.propertyHasCapability) {
    QNN_ERROR("propertyHasCapability is nullptr.");
    return false;
  }

  if (m_qnnInterface.propertyHasCapability(QNN_PROPERTY_CONTEXT_SUPPORT_CONFIGURATION) !=
      QNN_PROPERTY_SUPPORTED) {
    QNN_ERROR("Backend does not support QNN_PROPERTY_CONTEXT_SUPPORT_CONFIGURATION");
  }

  if (nullptr == m_qnnInterface.contextSetConfig) {
    QNN_ERROR("contextSetConfig is nullptr.");
    return false;
  }

  QnnContext_Config_t contextConfig           = QNN_CONTEXT_CONFIG_INIT;
  contextConfig.option                        = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
  contextConfig.priority                      = priority;
  const QnnContext_Config_t* contextConfigs[] = {&contextConfig, nullptr};

  for (auto ctxtHandle : m_contextVec) {
    Qnn_ErrorHandle_t err = m_qnnInterface.contextSetConfig(
        ctxtHandle, const_cast<const QnnContext_Config_t**>(contextConfigs));
    if (QNN_SUCCESS != err) {
      QNN_ERROR("contextSetConfig for priority failed.");
      return false;
    }
  }

  return true;
}

static std::string dataFormatToString(const Qnn_TensorDataFormat_t format) {
  switch (format) {
    case QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER:
      return "QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER";
    case QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT:
      return "QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT";
    default:
      return std::to_string(format);
  }
}

bool QnnApi::initializeScorer(
    const std::string& scorerPath,
    const std::map<uint32_t, std::array<std::tuple<int, size_t>, 2>>& scorerAllocs,
    std::map<uint32_t, uint8_t*>& scorerMemptrs,
    size_t expectedContextLength,
    Qnn_TensorDataFormat_t expectedCacheFormat) {
  // Current design assumes a scorer network that takes in n_layer anchors + n_layer keys
  // The output is n_layer scores, where score = anchor @ key

  // Load the model
  uint64_t scorerSize = m_resourceManager->getBufferSize(scorerPath);
  if (scorerSize == 0) {
    QNN_ERROR("Scorer file %s couldn't be read, or is empty", scorerPath.c_str());
    return false;
  }

  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  if (QNN_SUCCESS !=
      m_resourceManager->getQnnSystemInterface().systemContextCreate(&sysCtxHandle)) {
    QNN_ERROR("Could not create system handle for scorer");
    return false;
  }

  std::shared_ptr<uint8_t> buffer;
  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  if (!mapAndGetContextBinaryInfo(m_mmapContextBins,
                                  buffer,
                                  scorerPath,
                                  scorerSize,
                                  m_contextVec.size(),
                                  false,
                                  sysCtxHandle,
                                  &binaryInfo)) {
    QNN_ERROR("Failed to map context Binary for scorer");
    return false;
  };

  uint32_t graphsCount{0};
  qnn_wrapper_api::GraphInfo_t** graphsInfo{nullptr};
  if (!copyMetadataToGraphsInfo(binaryInfo, graphsInfo, graphsCount)) {
    QNN_ERROR("Failed to copy metadata for scorer");
    return false;
  }

  if (QNN_SUCCESS != m_resourceManager->getQnnSystemInterface().systemContextFree(sysCtxHandle)) {
    QNN_ERROR("Could not free system context object");
    return false;
  }

  Qnn_ContextHandle_t contextHandle{nullptr};
  Qnn_ErrorHandle_t errCode = m_qnnInterface.contextCreateFromBinary(
      m_backendHandle,
      m_deviceHandle,
      nullptr,
      const_cast<const void*>(static_cast<void*>(buffer.get())),
      scorerSize,
      &contextHandle,
      nullptr);
  if (errCode != QNN_SUCCESS) {
    QNN_ERROR("Couldn't initialize scorer %s : err %lu",
              scorerPath.c_str(),
              static_cast<unsigned long>(errCode));
    return false;
  };

  m_contextVec.push_back(contextHandle);
  buffer.reset();

  m_scorer = graphsInfo[0];
  errCode  = m_qnnInterface.graphRetrieve(contextHandle, m_scorer->graphName, &(m_scorer->graph));
  if (errCode != QNN_SUCCESS) {
    QNN_ERROR("Unable to retrieve scorer graph handle");
    return false;
  }

  std::map<std::string, std::tuple<int, size_t, size_t>> scorerAllocMap;
  for (size_t idx = 0; idx < m_scorer->numInputTensors; idx++) {
    auto tensor = qualla::QnnUtils::Tensor(&m_scorer->inputTensors[idx]);

    // Parse the layer index based on the layer name. We expect anchor_0_in and keys_0_in
    std::string tname(tensor.tensor->v1.name);  // Use std::string for bounds checking
    uint32_t index = static_cast<uint32_t>(qualla::QnnUtils::parseNumberFromString<1>(tname)[0])
                     << 16;

    // Map the anchor and past_key tensor to use the same buffer as the LLM model
    if (tname.starts_with("anchor")) {
      const auto& [alloc_idx, offset] = scorerAllocs.at(index)[0];
      scorerAllocMap[tname]           = {alloc_idx, offset, tensor.dims.getAlignedSize()};
      // m_ioTensor->useSameMemory(&tensor, scorerTensors[index][0]);
    } else if (tname.starts_with("key") || tname.starts_with("past")) {
      if (tensor.tensor->v1.dataFormat != expectedCacheFormat) {
        QNN_ERROR("Scorer network KV dataFormat does not match the model. Expected %s, found %s",
                  dataFormatToString(expectedCacheFormat).c_str(),
                  dataFormatToString(tensor.tensor->v1.dataFormat).c_str());
        return false;
      }
      const auto& [alloc_idx, offset] = scorerAllocs.at(index)[1];
      scorerAllocMap[tname]           = {alloc_idx, offset, tensor.dims.getAlignedSize()};
      // m_ioTensor->useSameMemory(&tensor, scorerTensors[index][1]);
    }
  }
  m_ioTensor->mapFusedBufferOffset(m_scorer, contextHandle, scorerAllocMap);

  // Score tensor outputs need to be allocated
  size_t totalSize = 0;
  std::map<uint32_t, std::tuple<size_t, size_t, Qnn_Tensor_t*>> scoreTensorOffsets;
  for (size_t idx = 0; idx < m_scorer->numOutputTensors; idx++) {
    Qnn_Tensor_t& tensor = m_scorer->outputTensors[idx];

    // Parse the layer index based on the layer name. We expect anchor_0_in and keys_0_in
    std::string tname(tensor.v1.name);  // Use std::string for readability
    uint32_t index = static_cast<uint32_t>(qualla::QnnUtils::parseNumberFromString<1>(tname)[0])
                     << 16;

    auto scoreTensor        = qualla::QnnUtils::Tensor(&tensor);
    const size_t scoreSize  = scoreTensor.dims.getAlignedSize();
    const size_t scoreCount = scoreTensor.dims.channel;

    if (scoreCount != expectedContextLength) {
      QNN_ERROR(
          "Error validating scoring network. Expected %zu scores, but network produces %zu scores.",
          expectedContextLength,
          scoreCount);
      return false;
    }

    scoreTensorOffsets[index] = {scoreSize, totalSize, &tensor};
    totalSize += scoreSize;
  }

  // Allocate buffer for scores
  uint64_t allocIdx    = m_ioTensor->allocate(totalSize);
  uint8_t* scoreMemptr = static_cast<uint8_t*>(m_ioTensor->getBuffer(allocIdx));

  // Register and accumulate set of all score buffers
  for (auto& [index, scoreTensorOffset] : scoreTensorOffsets) {
    auto& [alloc_size, allocOffset, tensor] = scoreTensorOffset;

    scorerMemptrs[index] = scoreMemptr + allocOffset;
    if (!m_ioTensor->mapFusedBufferOffset(
            tensor, allocIdx, allocOffset, contextHandle, alloc_size)) {
      QNN_ERROR(
          "Error registering output tensor %s for scorer %s", tensor->v1.name, m_scorer->graphName);
      return false;
    }
  }

  return true;
}

bool QnnApi::executeScorer() {
  GENIE_TRACE();
  QNN_DEBUG("Executing scorer %s", m_scorer->graphName);
#if NSP_LOG_LEVEL > 1
  auto start = std::chrono::steady_clock::now();
#endif

  std::map<std::string, std::pair<double, uint16_t>> timeLogs;
  if (!graphExecute(m_scorer, m_scorer->inputTensors, m_scorer->outputTensors, timeLogs)) {
    QNN_ERROR("Error executing scorer network");
    return false;
  }

#if NSP_LOG_LEVEL > 1
  auto stop     = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  QNN_DEBUG("graphExecute[%s] took: %lld us\n", m_scorer->graphName, duration);
#endif
  return true;
}

void QnnApi::setIOTensor(std::shared_ptr<qualla::IOTensor> ioTensor) { m_ioTensor = ioTensor; }

std::shared_ptr<qualla::IOTensor> QnnApi::getIOTensor() { return m_ioTensor; }

void QnnApi::setKVDim(uint32_t kvDim) { m_kvDim = kvDim; }

void QnnApi::setContextSize(size_t ctxSize) { m_ctxSize = ctxSize; }

void QnnApi::setKVUpdateMethod(KVManagerMode kvUpdateMethod) { m_kvUpdateMethod = kvUpdateMethod; }

std::unordered_map<std::string, std::pair<uint64_t, size_t>>* QnnApi::getTensorAllocInfo() {
  return m_tensorAllocInfo;
}

void QnnApi::setDataAlignmentSize(uint32_t dataAlignmentSize) {
  m_dataAlignmentSize = dataAlignmentSize;
}

void QnnApi::setCacheGroupPrefixes(std::unordered_set<std::string>& prefixList) {
  m_cacheGroupPrefixes = prefixList;
}

bool QnnApi::getLmHeadWeightInputEnabled() { return m_lmHeadWeightInput; }

bool QnnApi::getLoraWeightEnabled() { return m_loraWeightEnabled; }

QNN_INTERFACE_VER_TYPE* QnnApi::getQnnInterfaceVer() { return &m_qnnInterface; }

qnn_wrapper_api::GraphInfo_t**& QnnApi::getGraphsInfo() { return m_graphsInfo; }

uint32_t QnnApi::getGraphsCount() { return m_graphsCount; }

std::vector<uint32_t> QnnApi::getGraphCountPerContext() { return m_graphCountPerContext; }

std::vector<Qnn_ContextHandle_t>& QnnApi::getContexts() { return m_contextVec; }

Qnn_ContextHandle_t QnnApi::getContexts(qnn_wrapper_api::GraphInfo_t* const graph) {
  return m_contextMap.at(graph);
};

void QnnApi::updateContext(Qnn_ContextHandle_t context, uint32_t contextId) {
  std::lock_guard<std::mutex> lock(m_updateCallbackMutex);
  m_contextVec.push_back(context);
  m_contextIdxToHandle[contextId] = context;
}

void QnnApi::updateQnnApiGraphsandContextsInfo(std::string graphName,
                                               Qnn_GraphHandle_t graph,
                                               uint32_t contextId) {
  std::lock_guard<std::mutex> lock(m_updateCallbackMutex);
  m_graphNameToInfo[graphName]->graph = graph;
  m_graphNameToContextIdx[graphName]  = contextId;
  // m_graphsCount++;
}

bool QnnApi::allocateAll() {
  if (!m_ioTensor->isInitialize() &&
      true != m_ioTensor->initialize(
                  m_contextIdxToHandle[0], m_dataAlignmentSize, m_estimator)) {  // initialize
    QNN_ERROR("Qnn-Api: failure to initialize IOTensor");
    return false;
  }
  // Calculate total allocation sizes and offset of each tensor within its allocated buffer
  if (m_ioTensor->allocateBuffers() == false) {
    QNN_ERROR("Qnn-Api: Failed to allocate the Memory across the context buffers.");
    return false;
  }

  m_tensorAllocInfo = &m_ioTensor->getAllocInfo();
  QNN_DEBUG("Allocation Finished.");
  return true;
}

bool QnnApi::registerAll() {
  m_tensorAllocInfo = &m_ioTensor->getAllocInfo();  // always update the allocation Info
  for (uint32_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    // Register all the Tensors per graph.
    if (false == registerTensorsWithBackend(graphIdx)) {
      QNN_ERROR("Unable to MemRegister IO Tensors for graph index = %zu",
                static_cast<size_t>(graphIdx));
      freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }
  }
  QNN_DEBUG("Completed Registration of the Tensors.");
  return true;
}

void QnnApi::resetAdapterGraphHandleState() {
  std::lock_guard<std::mutex> guard{m_updateSharedAdapterMutex};
  m_currentAdapterGraphHandle = nullptr;
  QNN_DEBUG("Reset Current AdapterGraphHandle.");
}

bool QnnApi::clearAdapterBuffer(const std::string& binSectionPath) {
  if (m_adapterNameToBuffer.count(binSectionPath) > 0) {
    m_adapterNameToBuffer.erase(binSectionPath);
    QNN_DEBUG("Cleared adapter buffer for: %s", binSectionPath.c_str());
    return true;
  }
  return false;
}
