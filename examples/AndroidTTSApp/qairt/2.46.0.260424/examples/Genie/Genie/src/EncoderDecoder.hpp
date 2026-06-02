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

#include "Engine.hpp"
#include "GenieNode.h"
#include "LogUtils.hpp"
#include "Logger.hpp"
#include "Profile.hpp"
#include "Util/HandleManager.hpp"
#include "qualla/detail/tensor.hpp"
#include "qualla/encoder-decoder.hpp"

namespace genie {

class EncoderDecoder {
 public:
  class Config {
   public:
    static GenieNodeConfig_Handle_t add(std::shared_ptr<Config> config);
    static std::shared_ptr<Config> get(GenieNodeConfig_Handle_t handle);
    static void remove(GenieNodeConfig_Handle_t handle);

    Config(const char* configStr);
    nlohmann::json& getJson();
    void bindProfiler(std::shared_ptr<Profiler> profiler);
    void unbindProfiler();
    std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();
    void bindLogger(std::shared_ptr<genie::log::Logger> log);
    void unbindLogger();
    std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();

   private:
    static qnn::util::HandleManager<Config>& getManager();

    nlohmann::json m_config;
    std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
    std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
  };

  static GenieNode_Handle_t add(std::shared_ptr<EncoderDecoder> encoderDecoder);
  static std::shared_ptr<EncoderDecoder> get(GenieNode_Handle_t handle);
  static void remove(GenieNode_Handle_t handle);

  static void validateEncoderDecoderConfig(const nlohmann::json& config);
  static void translateEncoderDecoderConfig(const nlohmann::json& genieConfig,
                                            nlohmann::json& quallaConfig);

  void initEncoderDecoder(nlohmann::json& config,
                          std::shared_ptr<ProfileStat> profileStat,
                          std::shared_ptr<genie::log::Logger> logger = nullptr,
                          std::shared_ptr<qualla::Env> env           = nullptr);

  EncoderDecoder(nlohmann::json& config,
                 std::shared_ptr<ProfileStat> profileStat,
                 std::shared_ptr<genie::log::Logger> logger = nullptr,
                 std::shared_ptr<qualla::Env> env           = nullptr);

  EncoderDecoder(std::shared_ptr<Config> config,
                 std::shared_ptr<ProfileStat> profileStat,
                 std::shared_ptr<genie::log::Logger> logger = nullptr);

  EncoderDecoder(const EncoderDecoder&)            = delete;
  EncoderDecoder& operator=(const EncoderDecoder&) = delete;
  EncoderDecoder(EncoderDecoder&&)                 = delete;
  EncoderDecoder& operator=(EncoderDecoder&&)      = delete;

  std::vector<float> train(qualla::TrainingData& training_data);

  int32_t saveLora(std::string loraAdapterName,
                   std::string engineRole,
                   std::shared_ptr<ProfileStat> profileStat);

  void bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler);
  void unbindProfiler();
  std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();
  void bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& log);
  void unbindLogger();
  std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();
  std::string getName();
  std::string getType();
  int32_t getDimensions(qualla::LayerType layerType, std::vector<uint32_t>& dimensions);
  int32_t getQuantParam(qualla::LayerType layerType,
                        std::string& dataType,
                        double& scale,
                        int32_t& offset,
                        size_t& byteWidth);
  void setPerformancePolicy(const Genie_PerformancePolicy_t policy);
  const Genie_PerformancePolicy_t& getPerformancePolicy();

 private:
  static qnn::util::HandleManager<EncoderDecoder>& getManager();

  std::unique_ptr<qualla::EncoderDecoder> m_quallaEncoderDecoder;
  static std::atomic<std::uint32_t> s_nameCounter;
  std::string m_name;
  std::string m_type = "unet";
  std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
  std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
  Genie_PerformancePolicy_t m_performancePolicy;
};

}  // namespace genie
