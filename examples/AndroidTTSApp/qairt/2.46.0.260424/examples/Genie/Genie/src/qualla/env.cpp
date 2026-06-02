//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <stdexcept>

#include "qualla/env.hpp"

namespace fs = std::filesystem;

namespace qualla {

static uint64_t s_nameCounter{0};

Env::Env(const nlohmann::json& conf, std::shared_ptr<genie::ResourceManager> resourceManager)
    : m_resourceManager(std::move(resourceManager)) {
  if (!m_resourceManager) {
    throw std::invalid_argument("Env requires a non-null ResourceManager");
  }
  _path.models = fs::path();
  _path.cache  = fs::path();

  if (conf.contains("path")) {
    const nlohmann::json& p = conf["path"];

    if (p.contains("models"))
      _path.models = fs::path(p["models"].get<std::string>()).make_preferred();
    if (p.contains("cache")) _path.cache = fs::path(p["cache"].get<std::string>()).make_preferred();
  }
  _name = "env" + std::to_string(s_nameCounter++);
}

Env::~Env() {}

bool Env::update(std::shared_ptr<Env> env) {
  if (_name == env->getName()) return true;
  _name     = env->getName();
  m_loggers = env->getLogger();
  _path     = env->getPath();
  return true;
}

std::shared_ptr<Env> Env::create(std::shared_ptr<genie::ResourceManager> resourceManager,
                                 const nlohmann::json& conf) {
  return std::make_shared<Env>(conf, std::move(resourceManager));
}

std::shared_ptr<Env> Env::create(std::shared_ptr<genie::ResourceManager> resourceManager,
                                 std::istream& json_stream) {
  return create(std::move(resourceManager), nlohmann::json::parse(json_stream));
}

std::shared_ptr<Env> Env::create(std::shared_ptr<genie::ResourceManager> resourceManager,
                                 const std::string& json_str) {
  return create(std::move(resourceManager), nlohmann::json::parse(json_str));
}

}  // namespace qualla
