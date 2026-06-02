//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <atomic>
#include <memory>

#include "Dlc.hpp"
#include "Engine.hpp"
#include "GenieDlc.h"
#include "GenieEmbedding.h"
#include "LogUtils.hpp"
#include "Logger.hpp"
#include "Profile.hpp"
#include "Util.hpp"
#include "Util/HandleManager.hpp"
#include "nlohmann/json.hpp"
#include "qualla/ModelIOAdaptor.hpp"
#include "qualla/encoder.hpp"
#include "qualla/env.hpp"

namespace genie {

class Embedding {
 public:
  class Config {
   public:
    static GenieEmbeddingConfig_Handle_t add(std::shared_ptr<Config> config);
    static std::shared_ptr<Config> get(GenieEmbeddingConfig_Handle_t handle);
    static void remove(GenieEmbeddingConfig_Handle_t handle);

    Config(const char* configStr);
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

  static GenieEmbedding_Handle_t add(std::shared_ptr<Embedding> embedding);
  static std::shared_ptr<Embedding> get(GenieEmbedding_Handle_t handle);
  static void remove(GenieEmbedding_Handle_t handle);

  static void verifyAndUpdateConfig(nlohmann::json& config);
  static void validateEmbeddingConfig(const nlohmann::json& config, bool validateTextEncoder);
  static void translateEmbeddingConfig(const nlohmann::json& genieConfig,
                                       nlohmann::json& quallaConfig);

  void initEmbedding(nlohmann::json& config,
                     std::shared_ptr<qualla::Env> env,
                     std::shared_ptr<ProfileStat> profileStat,
                     std::shared_ptr<genie::log::Logger> logger = nullptr);

  Embedding(nlohmann::json& config,
            std::shared_ptr<ProfileStat> profileStat,
            std::shared_ptr<genie::log::Logger> logger = nullptr,
            std::shared_ptr<qualla::Env> env           = nullptr);

  Embedding(std::shared_ptr<Config> config,
            std::shared_ptr<ProfileStat> profileStat,
            std::shared_ptr<genie::log::Logger> logger = nullptr);

  Embedding(const Embedding&)            = delete;
  Embedding& operator=(const Embedding&) = delete;
  Embedding(Embedding&&)                 = delete;
  Embedding& operator=(Embedding&&)      = delete;

  int32_t applyLora(std::string loraAdapterName,
                    std::string engineRole,
                    std::shared_ptr<ProfileStat> profileStat);
  int32_t applyLoraStrength(std::string tensorName, std::string engineRole, float alpha);

  bool encode(const char* queryStr,
              std::vector<uint8_t>& outputEmbedding,
              std::shared_ptr<ProfileStat> profileStat);

  int32_t generate(const char* queryStr,
                   GenieEmbedding_GenerateCallback_t callback,
                   const void* userData,
                   std::shared_ptr<ProfileStat> profileStat);

  int32_t encode(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
                 std::vector<uint8_t>& outputEmbedding,
                 std::shared_ptr<ProfileStat> profileStat);

  void bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler);
  void unbindProfiler();
  std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();
  void bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& log);
  void unbindLogger();
  std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();
  std::string getName();
  std::string getType();
  int32_t getInputNames(std::unordered_set<std::string>& inputTensorNames);
  int32_t getOutputDimensions(std::vector<uint32_t>& dimensions);
  int32_t getOutputQuantParam(std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              float& byteWidth);
  int32_t getDimensions(qualla::LayerType layerType, std::vector<uint32_t>& dimensions);
  int32_t getQuantParam(qualla::LayerType layerType,
                        std::string& dataType,
                        double& scale,
                        int32_t& offset,
                        size_t& byteWidth);
  void setPerformancePolicy(const Genie_PerformancePolicy_t policy);
  const Genie_PerformancePolicy_t& getPerformancePolicy();

  void registerModelAdaptor(std::shared_ptr<qualla::ModelIOAdaptor> adaptor);

 private:
  static qnn::util::HandleManager<Embedding>& getManager();

  std::unique_ptr<qualla::Encoder> m_quallaEmbedding;
  static std::atomic<std::uint32_t> s_nameCounter;
  std::string m_name;
  std::string m_type = "text";
  std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
  std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
  Genie_PerformancePolicy_t m_performancePolicy;
};

}  // namespace genie
