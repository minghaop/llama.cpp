//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <functional>
#include <memory>
#include <mutex>

#include "BackendExtensions.hpp"
#include "MmappedFile/MmappedFile.hpp"
#include "QnnConfig.hpp"
#ifdef QUALLA_ENGINE_QNN_HTP
#include "QnnHtpDevice.h"
#include "QnnHtpPerfInfrastructure.h"
#endif  // QUALLA_ENGINE_QNN_HTP
#include "QnnWrapperUtils.hpp"
#include "ResourceManager.hpp"
#include "Traceable.hpp"
#include "buffer/IOTensor.hpp"
#include "config/ConfigList.hpp"
#include "config/QnnContextConfig.hpp"
#include "qnn-utils.hpp"

#define QNN_IO_TENSOR_DEBUG 0

using qualla::QnnUtils::QuantParam;
using ContextConfigList = ConfigList<QnnContext_Config_t>;
using LogCallback =
    std::function<void(const char* fmt, uint32_t level, uint64_t timestamp, va_list args)>;

enum class KVManagerMode {
  POINTER_SHIFT = 0x0,
  SHIFT_CONCAT  = 0x1,
  SMART_MASK    = 0x2,
  NATIVE_KV     = 0x3
};

enum class GraphType { NONE, DEFAULT, LUT, DECODER, DECODER_PREFILL, LMHEAD, IMAGE_ENCODER };

void registerUserCb(LogCallback l);

void userLogCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args);

class QnnApi : public genie::profiling::Traceable {
 public:
  typedef uint32_t CtxBitVector;

  // ctor
  QnnApi(std::shared_ptr<genie::profiling::TraceLogger> traceLogger,
         std::shared_ptr<genie::ResourceManager> resourceManager)
      : Traceable(traceLogger), m_resourceManager(std::move(resourceManager)) {
    if (!m_resourceManager) {
      // Provide fallback: create a default ResourceManager
      try {
        QNN_INFO("Null ResourceManager passed. Creating a default ResourceManager.");
        m_resourceManager = std::make_shared<genie::ResourceManager>();
      } catch (const std::exception& e) {
        QNN_ERROR("QnnApi: Failed to initialize ResourceManager: %s", e.what());
        throw std::runtime_error("QnnApi: ResourceManager initialization failed");
      }
    }
  };

  // dtor
  ~QnnApi();

  bool freeGraphs();

  static QnnApi& getInstance();

  static void contextNotifyFn(Qnn_ContextHandle_t context,
                              Qnn_GraphHandle_t graph,
                              const char* graph_name,
                              QnnContext_createFromBinaryAsyncNotifyType_t completeType,
                              void* notifyParam,
                              Qnn_ErrorHandle_t status);

  bool createFromBinaryGpu(std::vector<std::string> cachedBinariesPathVec);
#ifdef QUALLA_ENGINE_QNN_HTP
  bool createFromBinaryHtp(std::vector<std::string> cachedBinariesPathVec,
                           size_t spillFillBufferSize                       = 0,
                           uint64_t mmap_budget                             = 0,
                           bool graphSwitching                              = false,
                           const std::vector<std::string>& execSelectGraphs = {},
                           bool loadSelectGraphs                            = false,
                           bool skipLoraValidation                          = false);

  bool createFromBinaryListAsyncHtp(std::vector<std::string> cachedBinariesPathVec,
                                    size_t spillFillBufferSize                       = 0,
                                    uint64_t mmap_budget                             = 0,
                                    bool graphSwitching                              = false,
                                    const std::vector<std::string>& execSelectGraphs = {},
                                    bool loadSelectGraphs                            = false,
                                    bool skipLoraValidation                          = false,
                                    bool lazyIOInitialization                        = false);

  bool initializeHtp(std::string backendPath,
                     std::vector<std::string> modelPathOrCachedBinaryPathVec,
                     BackendExtensionsConfigs backendExtensionsConfig,
                     std::vector<GraphConfigs> graphConfigs           = {},
                     bool loadFromCachedBinary                        = false,
                     std::string systemLibraryPath                    = "",
                     bool debugModeRequested                          = false,
                     size_t spillFillBufferSize                       = 0,
                     bool mmapContextBins                             = false,
                     bool asyncInit                                   = true,
                     uint64_t mmap_budget                             = 0,
                     bool debug_qnn                                   = false,
                     bool graphSwitching                              = false,
                     const std::vector<std::string>& execSelectGraphs = {},
                     bool loadSelectGraphs                            = false,
                     bool skipLoraValidation                          = false,
                     bool lazyIOInitialization                        = false,
                     uint32_t logLevel                                = 1,
                     LogCallback inLogCallBack                        = nullptr);
#endif  // QUALLA_ENGINE_QNN_HTP
  bool initializeGpu(std::string backendPath,
                     std::vector<std::string> modelPathOrCachedBinaryPath,
                     bool debug_qnn            = false,
                     uint32_t logLevel         = 1,
                     LogCallback inLogCallBack = nullptr);

  bool initializeCpu(std::string backendPath,
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
                     bool debug_qnn            = false,
                     uint32_t logLevel         = 1,
                     LogCallback inLogCallBack = nullptr);

  bool populateGraphBinaryInfo(std::vector<std::string> modelPathOrCachedBinaryPathVec,
                               bool graphSwitching = false,
                               bool useMmap        = false);

  void setIOTensor(std::shared_ptr<qualla::IOTensor> ioTensor);

  std::shared_ptr<qualla::IOTensor> getIOTensor();

  std::shared_ptr<qualla::IOTensor> getIOTensorBufferMgr();

  void setKVDim(uint32_t kvDim);

  void setContextSize(size_t ctxSize);

  void setKVUpdateMethod(KVManagerMode kvUpdateMethod);

  void setDataAlignmentSize(uint32_t dataAlignmentSize);

  std::unordered_map<std::string, std::pair<uint64_t, size_t>>* getTensorAllocInfo();

  bool allocateAll();

  bool registerAll();

  void setEstimator(
      std::unordered_map<CtxBitVector, std::unordered_map<std::string, size_t>>& ctxAllocMap);

  void setCacheGroupPrefixes(std::unordered_set<std::string>& prefixList);

  void setGraphVariantType(std::unordered_map<std::string, GraphType>& graphVariantTypeMap) {
    m_graphVariantTypeMap = graphVariantTypeMap;
  };

  void setCacheGroupCtxSize(std::map<std::string, size_t>& cacheGroupCtxSize) {
    m_cacheGroupCtxSize = cacheGroupCtxSize;
  };

  void setCacheGroupUseScatter(std::map<std::string, bool>& cacheGroupUseScatter) {
    m_cacheGroupUseScatter = cacheGroupUseScatter;
  };

  bool getLmHeadWeightInputEnabled();

  bool getLoraWeightEnabled();

  bool graphExecute(Qnn_Tensor_t* input,
                    Qnn_Tensor_t* output,
                    std::string graphName,
                    std::map<std::string, std::pair<double, uint16_t>>& timeLogs);
  bool graphExecute(qnn_wrapper_api::GraphInfo_t* graph_info,
                    const Qnn_Tensor_t* input,
                    Qnn_Tensor_t* output,
                    std::map<std::string, std::pair<double, uint16_t>>& timeLogs);

  bool applyBinarySection(size_t binIdx,
                          const std::string& binSectionPath,
                          bool useMmap,
                          bool graphSwitch,
                          std::string& lazyLora,
                          bool weightSharedLora);

  bool applyBinarySection(size_t graphIdx, const std::string& binSectionPath);

  bool applyBinarySection(const std::string& graphName, const std::string& binSectionPath);

  bool applyCachedAdapter(Qnn_GraphHandle_t graphHandle);

  bool clearAdapterBuffer(const std::string& binSectionPath);

  bool setPerfProfile(qualla::PerformanceProfile& perfProfile);

  qualla::PerformanceProfile getPerfProfile();

  QNN_INTERFACE_VER_TYPE* getQnnInterfaceVer();

  qnn_wrapper_api::GraphInfo_t**& getGraphsInfo();

  uint32_t getGraphsCount();

  std::vector<uint32_t> getGraphCountPerContext();

  std::vector<Qnn_ContextHandle_t>& getContexts();

  Qnn_ContextHandle_t getContexts(qnn_wrapper_api::GraphInfo_t* const graph);

  void updateContext(Qnn_ContextHandle_t context, uint32_t contextId);

  void updateQnnApiGraphsandContextsInfo(std::string graphName,
                                         Qnn_GraphHandle_t graph,
                                         uint32_t contextId);

  bool updateIOEncodings(std::shared_ptr<uint8_t>& buffer, size_t bufferSize, size_t graphIdx);

  bool setOemKey(const std::string& oemKey);

  bool setExecutionPriority(Qnn_Priority_t priority);

  // KeyDiff - scoring network. Initialize and execute for each ctx_size and each layer
  bool initializeScorer(
      const std::string& scorerPath,
      const std::map<uint32_t, std::array<std::tuple<int, size_t>, 2>>& scorerAllocs,
      std::map<uint32_t, uint8_t*>& scorerMemptrs,
      size_t expectedContextLength,
      Qnn_TensorDataFormat_t expectedCacheFormat);

  bool executeScorer();

  virtual const char* getTraceNamespace() const override { return "QnnApi"; }
  void setDlcPath(std::string dlcPath);

  std::string getDlcPath();

  bool updateSharedAdapterForGraph(Qnn_GraphHandle_t graphHandle);

  void resetAdapterGraphHandleState();
  // Lazy LoRA variables
  std::unordered_map<Qnn_GraphHandle_t,
                     std::tuple<Qnn_ContextHandle_t, QnnContext_Buffer_t, size_t, bool>>
      m_adapterCache;
  // Lora Weight Sharing
  Qnn_GraphHandle_t m_currentAdapterGraphHandle{nullptr};

 private:
  //---------------------------------------------------------------
  // Private functions
  //---------------------------------------------------------------
  // Context configs
  bool getContextConfigs(ContextConfigList& configList,
                         bool graphSwitching                              = false,
                         const std::vector<std::string>& execSelectGraphs = {},
                         bool loadSelectGraphs                            = false);

  bool freeContextConfigs(QnnContext_Config_t** contextConfigs, uint32_t contextConfigCount);

  bool setGraphConfigsBeforeExecute(Qnn_GraphHandle_t graphHandle,
                                    QnnGraph_Config_t** graphConfigs,
                                    uint32_t configCount);

  // QNN Interface
  bool getQnnInterface(std::string backendPath);

  bool loadModel(std::string modelPath);

  bool initializeLogging(const QnnLog_Level_t& logLevel,
                         bool debug_qnn,
                         LogCallback userCallback = nullptr);

  void terminateLogging();

  bool initializeBackendExtensions(BackendExtensionsConfigs backendExtensionsConfig,
                                   bool debug_qnn,
                                   QnnLog_Level_t qnnLogLevel);

  // Backend
  bool initializeBackend();

  bool terminateBackend();

  // Device
  bool createDevice();

  bool freeDevice();

  // Context
  bool createContext();

  bool freeContext();

  // Graph
  bool composeGraphs(std::vector<GraphConfigs> graphConfigs);

  bool composeGraphs(std::vector<GraphConfigs> graphConfigs,
                     uint32_t* inputDim,
                     uint32_t inputRank,
                     uint32_t* outputDim,
                     uint32_t outputRank,
                     uint32_t* kvDim,
                     uint32_t kvRank,
                     uint32_t* kvScaleDim,
                     Qnn_Param_t* params,
                     uint32_t numParams);

  bool mapAndGetContextBinaryInfo(const bool useMmap,
                                  std::shared_ptr<uint8_t>& buffer,
                                  const std::string& binaryPath,
                                  const uint64_t bufferSize,
                                  const size_t contextIdx,
                                  const bool graphSwitching,
                                  QnnSystemContext_Handle_t sysCtxHandle,
                                  const QnnSystemContext_BinaryInfo_t** binaryInfo);

  bool parseIOTensorsAndAccumulate();

  bool registerTensorsWithBackend(size_t graphIdx);

  bool finalizeGraphs();

  bool finalizeCpuGraphs();

  bool checkCapabilityOfCreateAsync(bool& propRet);

  // Profiling event extraction helpers
  bool extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle,
                                   std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
                                   std::string graphName);

  bool extractProfilingSubEvents(QnnProfile_EventId_t profileEventId,
                                 std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
                                 std::string graphName);

  bool extractProfilingEvent(QnnProfile_EventId_t profileEventId,
                             std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
                             std::string graphName);

  bool extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);

  bool extractProfilingSubEvents(QnnProfile_EventId_t profileEventId);

  bool extractProfilingEvent(QnnProfile_EventId_t profileEventId);

  //---------------------------------------------------------------
  // Private members
  //---------------------------------------------------------------
  // Default number of graphs to assume space for during init
  static const uint32_t s_graphConfigsReserveCount = 16;

  // Model vars
  typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                            uint32_t* numProviders);
  typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
      const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

  // Graph Related Function Handle Types
  typedef qnn_wrapper_api::ModelError_t (*ComposeGraphsFnHandleType_t)(
      Qnn_BackendHandle_t,
      QNN_INTERFACE_VER_TYPE,
      Qnn_ContextHandle_t,
      const qnn_wrapper_api::GraphConfigInfo_t**,
      const uint32_t,
      qnn_wrapper_api::GraphInfo_t***,
      uint32_t*,
      bool,
      QnnLog_Callback_t,
      QnnLog_Level_t);

  typedef qnn_wrapper_api::ModelError_t (*GenAIComposeGraphsFnHandleType_t)(
      Qnn_BackendHandle_t,
      QNN_INTERFACE_VER_TYPE,
      Qnn_ContextHandle_t,
      const qnn_wrapper_api::GraphConfigInfo_t**,
      const uint32_t,
      uint32_t* inputDim,
      uint32_t inputRank,
      uint32_t* outputDim,
      uint32_t outputRank,
      uint32_t* kvDim,
      uint32_t kvRank,
      uint32_t* kvScaleDim,
      Qnn_Param_t* params,
      uint32_t numParam,
      const char* modelName,
      qnn_wrapper_api::GraphInfo_t***,
      uint32_t*,
      bool,
      QnnLog_Callback_t,
      QnnLog_Level_t);

  typedef qnn_wrapper_api::ModelError_t (*FreeGraphInfoFnHandleType_t)(
      qnn_wrapper_api::GraphInfo_t***, uint32_t);

  void* m_libModelHandle{nullptr};
  void* m_backendHandle{nullptr};
  void* m_backendLibraryHandle{nullptr};
  uint32_t m_backendId{0};

  // QNN Handles
  QNN_INTERFACE_VER_TYPE m_qnnInterface;
  std::unique_ptr<BackendExtensions> m_backendExtensions{nullptr};
  ComposeGraphsFnHandleType_t m_composeGraphsFnHandle{nullptr};
  GenAIComposeGraphsFnHandleType_t m_genaiComposeGraphsFnHandle{nullptr};
  FreeGraphInfoFnHandleType_t m_freeGraphInfoFnHandle{nullptr};
  Qnn_LogHandle_t m_logHandle{nullptr};
  Qnn_DeviceHandle_t m_deviceHandle{nullptr};
  Qnn_ProfileHandle_t m_profileBackendHandle{nullptr};
  QnnBackend_Config_t** m_backendConfigs{nullptr};
  uint32_t m_backendConfigCount{0};
  qnn::tools::netrun::PerfProfile m_perfProfile = qnn::tools::netrun::PerfProfile::HIGH_PERFORMANCE;
#ifdef QUALLA_ENGINE_QNN_HTP
  QnnHtpDevice_PerfInfrastructure_t* m_perfInfra{nullptr};
#endif
  uint32_t m_powerConfigId = 1;

  // Graphs and contexts
  uint32_t m_graphsCount{0};  // Total graph count
  std::vector<uint32_t> m_graphCountPerContext;
  std::vector<Qnn_ContextHandle_t> m_contextVec;
  std::unordered_map<qnn_wrapper_api::GraphInfo*, Qnn_ContextHandle_t> m_contextMap;
  qnn_wrapper_api::GraphInfo_t** m_graphsInfo{nullptr};
  std::unordered_map<std::string, size_t> m_graphNameToIndex;
  std::unordered_map<std::string, qnn_wrapper_api::GraphInfo*> m_graphNameToInfo;
  std::unordered_map<std::string, size_t> m_graphNameToContextIdx;
  std::unordered_map<size_t, Qnn_ContextHandle_t> m_contextIdxToHandle;
  std::mutex m_updateCallbackMutex;
  std::unordered_map<std::string, GraphType> m_graphVariantTypeMap;
  std::map<std::string, size_t> m_cacheGroupCtxSize;
  std::map<std::string, bool> m_cacheGroupUseScatter;
  // Useful Structure for IO Estimation
  std::shared_ptr<Estimator> m_estimator;
  // m_graphIdxToIOMap: stores {GraphId -> IOTensorMap}
  std::unordered_map<size_t, qualla::QnnUtils::TensorMap> m_graphIdxToIOMap;
  // m_contextAllocMap: stores {Translated ContextId -> {Tensor name, size}}
  std::unordered_map<CtxBitVector, std::unordered_map<std::string, size_t>> m_contextAllocMap;
  // m_tensorAllocInfo: stores {Tensor name -> (allocIdx, offset)}
  std::unordered_map<std::string, std::pair<uint64_t, size_t>>* m_tensorAllocInfo;
  // m_graphIdxToContextIdx: stores {Graph Idx -> Context Idx}
  std::unordered_map<size_t, size_t> m_graphIdxToContextIdx;
  // m_adapterNameToBuffer: stores {LoRA Adapter name -> raw data}
  std::unordered_map<std::string, std::shared_ptr<uint8_t>> m_adapterNameToBuffer;
  std::mutex m_updateSharedAdapterMutex;
  // Useful Structure for IO Esimtation
  std::shared_ptr<qualla::IOTensor> m_ioTensor;
  size_t m_ctxSize{0};
  uint32_t m_kvDim{0};
  uint32_t m_dataAlignmentSize{0};
  bool m_loraWeightEnabled{false};
  bool m_lmHeadWeightInput{false};
  KVManagerMode m_kvUpdateMethod{KVManagerMode::POINTER_SHIFT};
  std::unordered_set<std::string> m_cacheGroupPrefixes;

  // For KeyDiff, keep track of the scorer network graph
  qnn_wrapper_api::GraphInfo* m_scorer{nullptr};

  // Logistics variables
  bool m_isContextCreated{false};
  bool m_isBackendInitialized{false};
  bool m_isDeviceCreated{false};
  bool m_isLogInitialized{false};

  // Debug variables
  bool m_debugModeRequested;
  bool m_debugQnn{false};

  // Memory variables
  bool m_mmapContextBins;
  std::vector<std::shared_ptr<uint8_t>> m_persistentContextBins;
  std::shared_ptr<genie::ResourceManager> m_resourceManager;
};
