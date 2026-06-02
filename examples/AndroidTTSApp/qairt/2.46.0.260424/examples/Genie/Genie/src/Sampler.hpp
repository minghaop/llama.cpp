//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>
#include <vector>

#include "nlohmann/json.hpp"
#include "Context.hpp"
#include "GenieSampler.h"
#include "Util/HandleManager.hpp"
#include "qualla/env.hpp"
#include "qualla/sampler.hpp"

namespace genie {

class Sampler {
 public:
  class SamplerConfig {
   public:
    static GenieSamplerConfig_Handle_t add(std::shared_ptr<SamplerConfig> config);

    static std::shared_ptr<SamplerConfig> get(GenieSamplerConfig_Handle_t handle);

    static void remove(GenieSamplerConfig_Handle_t handle);

    static void validateSamplerConfig(const nlohmann::json& config);

    static void translateSamplerConfig(const nlohmann::json& genieConfig,
                                       nlohmann::json& quallaConfig);

    void validateStandaloneSamplerConfig(const nlohmann::json& config);

    SamplerConfig(const char* configStr);

    void setParam(const std::string& keyStr, const std::string& valueStr);

    nlohmann::json getSamplerJson() const;
    nlohmann::json getJson() const;

   private:
    static qnn::util::HandleManager<SamplerConfig>& getManager();
    nlohmann::json m_standaloneSamplerConfig{};
    bool m_isStandaloneSampler{false};
    nlohmann::json m_samplerConfig;
  };

  static GenieSampler_Handle_t add(std::shared_ptr<Sampler> sampler);
  static std::shared_ptr<Sampler> get(GenieSampler_Handle_t handle);
  static void remove(GenieSampler_Handle_t handle);
  static void registerCallback(const char* name, GenieSampler_ProcessCallback_t samplerCallback);
  static void registerUserDataCallback(const char* name,
                                       GenieSampler_UserDataCallback_t samplerCallback,
                                       const void* userData);

  Sampler(const std::string name,
          std::unordered_map<std::string, std::shared_ptr<qualla::Sampler>> quallaSamplers);

  Sampler(std::shared_ptr<SamplerConfig> config);

  void applyConfig(nlohmann::json samplerConfigJson);

  void sampleData(const void* data,
                  size_t dataSize,
                  const char* dataConfigJson,
                  GenieSampler_Callback_t callback,
                  const void* userData);

  const nlohmann::json& getJson();

 private:
  static qnn::util::HandleManager<Sampler>& getManager();
  bool m_isStandaloneSampler{false};
  std::string m_name;
  std::shared_ptr<Context> m_context;
  std::shared_ptr<qualla::Env> m_env;
  static std::atomic<std::uint32_t> s_nameCounter;
  std::unordered_map<std::string, std::shared_ptr<qualla::Sampler>>
      m_quallaSamplers;  // {role}->{sampler} mappings
};

}  // namespace genie
