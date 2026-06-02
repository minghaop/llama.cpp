//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <filesystem>
#include <memory>
#include <unordered_set>

#include "GenieLog.h"
#include "LogUtils.hpp"
#include "Logger.hpp"
#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/state.hpp"

namespace genie {
class ResourceManager;
}  // namespace genie

// The following macros can only be used in classes or functions that have access to a
// qualla::Env pointer/smart-pointer specifically named "_env"
#define __ERROR(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __WARN(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_WARN, fmt::format(__fmt, ##__VA_ARGS__))
#define __INFO(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_INFO, fmt::format(__fmt, ##__VA_ARGS__))
#define __KPIS(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __DEBUG(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __KVTRACE(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace qualla {

enum PerformanceProfile {
  PERFORMANCE_BURST                      = 10,
  PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE = 20,
  PERFORMANCE_HIGH_PERFORMANCE           = 30,
  PERFORMANCE_BALANCED                   = 40,
  PERFORMANCE_LOW_BALANCED               = 50,
  PERFORMANCE_HIGH_POWER_SAVER           = 60,
  PERFORMANCE_POWER_SAVER                = 70,
  PERFORMANCE_LOW_POWER_SAVER            = 80,
  PERFORMANCE_EXTREME_POWER_SAVER        = 90,
};

enum InputType { TOKENS = 0x01, EMBEDDINGS = 0x02, PIXELS = 0x03, UNKNOWN = 0xFF };

class Env : public State {
 public:
  QUALLA_API Env(const nlohmann::json& conf,
                 std::shared_ptr<genie::ResourceManager> resourceManager);
  QUALLA_API ~Env();

  struct Path {
    std::filesystem::path models;
    std::filesystem::path cache;
  };

  const Path& path() const { return _path; }

  std::shared_ptr<genie::log::Logger> logger() {
    if (m_loggers.size() > 0) {
      return *m_loggers.begin();
    } else {
      return nullptr;
    }
  }

  std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger() { return m_loggers; }
  Path getPath() { return _path; }
  std::string getName() { return _name; }
  bool update(std::shared_ptr<Env> env);

  void bindLogger(std::shared_ptr<genie::log::Logger>& logger) { m_loggers.insert(logger); }

  QUALLA_API static std::shared_ptr<Env> create(
      std::shared_ptr<genie::ResourceManager> resourceManager, const nlohmann::json& conf = {});
  std::shared_ptr<genie::ResourceManager> getResourceManager() const { return m_resourceManager; }
  QUALLA_API static std::shared_ptr<Env> create(
      std::shared_ptr<genie::ResourceManager> resourceManager, std::istream& json_stream);
  QUALLA_API static std::shared_ptr<Env> create(
      std::shared_ptr<genie::ResourceManager> resourceManager, const std::string& json_str);

 private:
  std::string _name;
  Path _path;
  std::unordered_set<std::shared_ptr<genie::log::Logger>> m_loggers;
  std::shared_ptr<genie::ResourceManager> m_resourceManager;
};

}  // namespace qualla
