//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>

#include "Accumulator.hpp"
#include "Dlc.hpp"
#include "Embedding.hpp"
#include "EncoderDecoder.hpp"
#include "Engine.hpp"
#include "GenieDlc.h"
#include "GenieNode.h"
#include "GeniePipeline.h"

namespace qualla {
class ModelIOAdaptor;
}

namespace genie {
namespace pipeline {

enum ExecutionStrategy {
  GENIE_EXECUTION_STRATEGY_EXPLICIT  = 0x01,
  GENIE_EXECUTION_STRATEGY_IMPLICIT  = 0x02,
  GENIE_EXECUTION_STARTEGY_UNDEFINED = 0xFF
};

class Pipeline;
class Node {
 public:
  class Config {
   public:
    static GenieNodeConfig_Handle_t add(std::shared_ptr<Config> config);
    static std::shared_ptr<Config> get(GenieNodeConfig_Handle_t handle);
    static void remove(GenieNodeConfig_Handle_t handle);
    Config(const char* configStr);
    Config(std::shared_ptr<Dlc> dlc, const char* useCaseName, const char* configStr = nullptr);

    void createConfigFromDlc(const char* useCaseName, const char* configPath = nullptr);

    nlohmann::json& getJson() { return m_config; };
    void bindProfiler(std::shared_ptr<Profiler> profiler);
    void unbindProfiler();
    std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();
    void bindLogger(std::shared_ptr<genie::log::Logger> log);
    void unbindLogger();
    std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();
    std::shared_ptr<Dlc>& getDlc();

   private:
    static qnn::util::HandleManager<Config>& getManager();

     nlohmann::json m_config;
    std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
    std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
    std::shared_ptr<Dlc> m_dlc;
  };

  static std::shared_ptr<Node> createNode(std::shared_ptr<Config> config,
                                          std::shared_ptr<ProfileStat> profileStat,
                                          std::shared_ptr<genie::log::Logger> logger = nullptr);
  Node( nlohmann::json config);
  // Node Utilities
  static GenieNode_Handle_t add(std::shared_ptr<Node> node);
  static std::shared_ptr<Node> get(GenieNode_Handle_t handle);

  void bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler);
  void unbindProfiler();
  std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();

  void bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& log);
  void unbindLogger();
  std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();
  std::string getName() const;

  static void remove(GenieNode_Handle_t handle);
  virtual int32_t bindPipeline(Pipeline& pipeline);
  virtual int32_t execute(void* userData,
                          std::shared_ptr<ProfileStat> profileStat,
                          nlohmann::json executionConfig = {}) = 0;

  virtual int32_t train();
  virtual int32_t saveLora(std::string loraAdapterName,
                           std::string engine,
                           std::shared_ptr<ProfileStat> profileStat);

  bool isTypeGenerator() { return m_typeGenerator; };
  size_t getMaxIterations();
  size_t getBatchSize();
  bool isConnected() { return m_isConnected; };
  bool hasWildcard() { return m_hasWildcardConnection; };
  std::string getType() { return m_type; };
  virtual int32_t save(const std::string&);
  virtual int32_t restore(const std::string&);
  virtual void reset(){};
  virtual int32_t setPriority(std::string engine, const GeniePipeline_Priority_t priority);
  virtual int32_t setOemkey(const std::string& oemKey);

  virtual int32_t applyLora(std::string loraAdapterName,
                            std::string engine,
                            std::shared_ptr<ProfileStat> profileStat);
  virtual int32_t applyLoraStrength(std::string tensorName, std::string engine, float alpha);

  virtual GenieEngine_Handle_t getEngineHandle(const std::string& engineRole,
                                               std::shared_ptr<ProfileStat> profileStat);
  virtual int32_t bindEngine(const std::string& engineRole,
                             std::shared_ptr<Engine> engine,
                             std::shared_ptr<ProfileStat> profileStat);
  virtual GenieSampler_Handle_t getSamplerHandle();
  virtual GenieTokenizer_Handle_t getTokenizerHandle();

  // Input Modality setters
  virtual int32_t setTextInputData(GenieNode_IOName_t nodeIOName,
                                   const char* txt,
                                   std::shared_ptr<ProfileStat> profileStat);
  virtual int32_t setEmbeddingInputData(GenieNode_IOName_t nodeIOName,
                                        const void* embedding,
                                        size_t embeddingSize,
                                        std::shared_ptr<ProfileStat> profileStat);
  virtual int32_t setImageInputData(GenieNode_IOName_t nodeIOName,
                                    const void* imageData,
                                    size_t imageSize,
                                    std::shared_ptr<ProfileStat> profileStat);

  // Input setter based on the buffers
  virtual int32_t setInputBufferData(GenieNode_IOName_t nodeIOName,
                                     const void* data,
                                     size_t dataSize,
                                     nlohmann::json dataConfig,
                                     std::shared_ptr<ProfileStat> profileStat);

  // Get Output buffer data
  virtual int32_t getOutputBufferData(GenieNode_IOName_t nodeIOName,
                                      nlohmann::json& ioConfig,
                                      GenieNode_IOCallback_t callback,
                                      const void* userData,
                                      std::shared_ptr<ProfileStat>);

  // Output Modality setters
  virtual int32_t setTextOutputCallback(GenieNode_IOName_t nodeIOName,
                                        GenieNode_TextOutput_Callback_t callback);
  virtual int32_t setEmbeddingOutputCallback(GenieNode_IOName_t nodeIOName,
                                             GenieNode_EmbeddingOutputCallback_t callback);
  void markConnected() { m_isConnected = true; };
  void markWildcard() { m_hasWildcardConnection = true; }
  const ExecutionStrategy& getExecutionStrategyFromString(const std::string& strategy);
  const qualla::LayerType& getLayerTypeFromNodeIO(const GenieNode_IOName_t& ioName);
  virtual ~Node();

  virtual void registerModelAdaptor(std::shared_ptr<qualla::ModelIOAdaptor> /*adaptor*/) {
    throw Exception(GENIE_STATUS_ERROR_INVALID_ARGUMENT,
                    "Model Adaptors are unsupported on this node.");
  }

  ExecutionStrategy m_executionStrategy = ExecutionStrategy::GENIE_EXECUTION_STARTEGY_UNDEFINED;

 protected:
  std::string m_type     = "inference";
  bool m_typeGenerator   = false;
  bool m_crossAttention  = false;
  size_t m_maxIterations = 0;
  size_t m_batchSize     = 0;
  Pipeline* m_pipeline   = nullptr;
  nlohmann::json m_config;

 private:
  static qnn::util::HandleManager<Node>& getManager();

  std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
  std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
  static std::atomic<std::uint32_t> s_nameCounter;
  std::string m_name;
  bool m_isConnected           = false;
  bool m_hasWildcardConnection = false;
};

}  // namespace pipeline

}  // namespace genie
