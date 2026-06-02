//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <atomic>
#include <functional>
#include <memory>

#include "Dlc.hpp"
#include "Engine.hpp"
#include "GenieDialog.h"
#include "GenieDlc.h"
#include "GenieEngine.h"
#include "GenieNode.h"
#include "LogUtils.hpp"
#include "Logger.hpp"
#include "Profile.hpp"
#include "Sampler.hpp"
#include "Tokenizer.hpp"
#include "Util.hpp"
#include "Util/HandleManager.hpp"
#include "nlohmann/json.hpp"
#include "qualla/DialogCallback.hpp"
#include "qualla/ModelIOAdaptor.hpp"
#include "qualla/dialog.hpp"
#include "qualla/env.hpp"

namespace genie {

class Dialog {
 public:
  class Config {
   public:
    static GenieDialogConfig_Handle_t add(std::shared_ptr<Config> config);
    static std::shared_ptr<Config> get(GenieDialogConfig_Handle_t handle);
    static void remove(GenieDialogConfig_Handle_t handle);

    Config(const char* inputStr);
    Config(std::shared_ptr<Dlc> dlc, const char* useCaseName, const char* configStr = nullptr);

    void createConfigFromDlc(const char* useCaseName, const char* configPath = nullptr);
    void createConfigFromJson(const char* configStr);

    nlohmann::json& getJson();
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

  static void verifyAndUpdateConfig(nlohmann::json& config);
  static void updateDialogConfigForKVShare(nlohmann::json& config);
  static void validateDialogConfig(const nlohmann::json& config);
  static void translateDialogConfig(const nlohmann::json& genieConfig,
                                    nlohmann::json& quallaConfig);

  static GenieDialog_Handle_t add(std::shared_ptr<Dialog> dialog);
  static std::shared_ptr<Dialog> get(GenieDialog_Handle_t handle);
  static void remove(GenieDialog_Handle_t handle);

  static GenieSampler_Handle_t getSamplerHandle(std::shared_ptr<genie::Dialog> dialog);
  static GenieTokenizer_Handle_t getTokenizerHandle(std::shared_ptr<genie::Dialog> dialog);
  static void getStandaloneEnginesConfig(nlohmann::json& config,
                                         nlohmann::json& standaloneEnginesConfig);

  qualla::DialogCallback dialogCallback;

  void initDialog(nlohmann::json config,
                  std::shared_ptr<qualla::Env> env,
                  std::shared_ptr<ProfileStat> profileStat,
                  std::shared_ptr<genie::log::Logger> logger = nullptr,
                  std::shared_ptr<genie::Profiler> profiler  = nullptr);

  Dialog(nlohmann::json& config,
         std::shared_ptr<ProfileStat> profileStat,
         std::shared_ptr<genie::log::Logger> logger = nullptr,
         std::shared_ptr<genie::Profiler> profiler  = nullptr,
         std::shared_ptr<qualla::Env> env           = nullptr);
  Dialog(std::shared_ptr<Config> config,
         std::shared_ptr<ProfileStat> profileStat,
         std::shared_ptr<genie::log::Logger> logger = nullptr);
  ~Dialog();

  std::string getName();

  Dialog(const Dialog&)            = delete;
  Dialog& operator=(const Dialog&) = delete;
  Dialog(Dialog&&)                 = delete;
  Dialog& operator=(Dialog&&)      = delete;

  int32_t query(const char* queryStr,
                GenieDialog_SentenceCode_t sentenceCode,
                GenieDialog_QueryCallback_t callback,
                const void* userData,
                std::shared_ptr<ProfileStat> profileStat);

  int32_t query(const char* queryStr,
                GenieNode_TextOutput_SentenceCode_t sentenceCode,
                GenieNode_TextOutput_Callback_t callback,
                const void* userData,
                std::shared_ptr<ProfileStat> profileStat);

  int32_t query(const std::vector<const char*>& queryStr_vec,
                GenieDialog_SentenceCode_t sentenceCode,
                GenieDialog_QueryCallback_t callback,
                const void** userData,
                std::shared_ptr<ProfileStat> profileStat);

  int32_t embeddingQuery(const void* embeddings,
                         const uint32_t embeddingsSize,
                         GenieDialog_SentenceCode_t sentenceCode,
                         GenieDialog_TokenToEmbeddingCallback_t t2eCallback,
                         GenieDialog_QueryCallback_t callback,
                         const void* userData,
                         std::shared_ptr<ProfileStat> profileStat);

  int32_t embeddingQuery(const void* embeddings,
                         const uint32_t embeddingsSize,
                         GenieNode_TextOutput_SentenceCode_t sentenceCode,
                         GenieNode_TextOutput_Callback_t callback,
                         const void* userData,
                         std::shared_ptr<ProfileStat> profileStat);

  int32_t tokenQuery(const uint32_t* tokens,
                     const uint32_t sizeInputTokens,
                     GenieDialog_SentenceCode_t sentenceCode,
                     GenieDialog_TokenQueryCallback_t callback,
                     const void* userData,
                     std::shared_ptr<ProfileStat> profileStat);

  int32_t embeddingQuery(const void* embeddings,
                         const uint32_t embeddingsSize,
                         GenieDialog_SentenceCode_t sentenceCode,
                         GenieDialog_TokenToEmbeddingCallback_t t2eCallback,
                         GenieDialog_TokenQueryCallback_t callback,
                         const void* userData,
                         std::shared_ptr<ProfileStat> profileStat);

  std::vector<float> train(qualla::TrainingData& trainingData);

  bool setCrossAttentionHiddenStates(const void* crossAttentionEmbeddings,
                                     const uint32_t embeddingsSize);
  int32_t save(const std::string&);
  int32_t restore(const std::string&);
  void reset();

  int32_t signalAction(GenieDialog_Action_t action);

  void setStopSequence(const char* newStopSeqs);

  int32_t applyLora(std::string loraAdapterName,
                    std::string engineRole,
                    std::shared_ptr<ProfileStat> profileStat);
  int32_t applyLoraStrength(std::string tensorName, std::string engineRole, float alpha);

  int32_t releaseLoraMemory(std::string lora_adapter_name, std::string engineRole);

  int32_t setPriority(std::string engineRole, const GenieDialog_Priority_t priority);
  int32_t setOemkey(const std::string& oemKey);

  void bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler);
  void unbindProfiler();
  std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();

  void bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& log);
  void unbindLogger();
  std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();

  GenieEngine_Handle_t getEngineHandle(const std::string& engineRole,
                                       std::shared_ptr<ProfileStat> profileStat);
  int32_t bindEngine(const std::string& engineRole,
                     std::shared_ptr<Engine> engine,
                     std::shared_ptr<ProfileStat> profileStat);

  int32_t getDimensions(qualla::LayerType layerType,
                        std::vector<uint32_t>& dimensions);

  int32_t getInputQuantParam(std::string& dataType,
                             double& scale,
                             int32_t& offset,
                             size_t& byteWidth,
                             bool isCrossAttention = false);

  void setPerformancePolicy(const Genie_PerformancePolicy_t policy);
  const Genie_PerformancePolicy_t& getPerformancePolicy();
  void setMaxNumTokens(const uint32_t maxNumTokens);

  bool setVisionParam(const std::vector<uint32_t>& visionParam);

  uint32_t getContextOccupancy() const { return m_quallaDialog->getContextOccupancy(); }
  std::string getAppliedLoraAdapter(const std::string& engineRole = "primary") const;

  void registerModelAdaptor(std::shared_ptr<qualla::ModelIOAdaptor> adaptor);

  size_t getEmbeddingLength() const {
    return static_cast<size_t>(m_quallaDialog->getEmbeddingLength());
  };

 protected:
  // Protected visibility for genie-ppl-run & perplexity APIs
  std::unique_ptr<qualla::Dialog> m_quallaDialog;
  friend class Accuracy;

 private:
  static qnn::util::HandleManager<Dialog>& getManager();

  uint32_t m_tokenLimit{UINT32_MAX};
  std::atomic<bool> m_abort{false};
  std::atomic<bool> m_pause{false};
  std::atomic<uint32_t> m_activeQuery{0};
  static std::atomic<std::uint32_t> s_nameCounter;
  std::vector<std::pair<std::string, size_t>> m_sharedEngineKeys;
  bool m_sharedEngine{false};
  std::string m_name;
  GenieSampler_Handle_t m_samplerHandle;
  GenieTokenizer_Handle_t m_tokenizerHandle;
  Genie_PerformancePolicy_t m_performancePolicy;
  std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
  std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
};

}  // namespace genie
