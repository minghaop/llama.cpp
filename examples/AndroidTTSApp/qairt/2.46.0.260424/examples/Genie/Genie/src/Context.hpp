//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#include <memory>

#include "Exception.hpp"
#include "qualla/context.hpp"
#include "qualla/env.hpp"

namespace genie {

class Context {
 public:
  class ContextConfig {
   public:
    ContextConfig(nlohmann::json config) : m_config(config){};
    nlohmann::json& getJson();
    static void validateContextConfig(const nlohmann::json& config);
    static void translateContextConfig(const nlohmann::json& genieConfig, nlohmann::json& quallaConfig);

   private:
    nlohmann::json m_config;
  };

  Context(const nlohmann::json& config, std::shared_ptr<qualla::Env> env);
  std::shared_ptr<qualla::Context> getQuallaContext();
  ~Context();

 private:
  std::string m_name;
  std::shared_ptr<qualla::Context> m_quallaContext;
  std::shared_ptr<qualla::Env> m_env;
  static std::atomic<std::uint32_t> s_nameCounter;
};

}  // namespace genie
