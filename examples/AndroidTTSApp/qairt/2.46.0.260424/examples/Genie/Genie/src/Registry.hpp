//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "Engine.hpp"
#include "nlohmann/json.hpp"

namespace genie {

class Registry {
 public:
  // Static function to get keys from registry
  static std::vector<std::pair<std::string, size_t>> getKeysFromRegistry(
      const nlohmann::json& config);

  // Static function to get engine from registry
  static std::unordered_map<std::string, std::shared_ptr<qualla::Engine>> getEngineFromRegistry(
      const std::vector<std::pair<std::string, size_t>>& keys,
      std::shared_ptr<ProfileStat> profileStat,
      std::shared_ptr<genie::log::Logger> logger,
      std::shared_ptr<qualla::Env> env);

  // Static function to delete engines from registry
  static void deleteEnginesFromRegistry(const std::vector<std::pair<std::string, size_t>>& keys);

 private:
  // Helper class to generate a unique key
  class EngineKey {
   public:
    size_t createKey(const nlohmann::json& engineConfig);
    // delete all constructors and destructor
    EngineKey()                            = default;
    ~EngineKey()                           = default;
    EngineKey(const EngineKey&)            = delete;
    EngineKey(EngineKey&&)                 = delete;
    EngineKey& operator=(const EngineKey&) = delete;
    EngineKey& operator=(EngineKey&&)      = delete;
  };

  class RegEngine {
   public:
    RegEngine(const nlohmann::json& engineConfig,
              std::shared_ptr<ProfileStat> profileStat,
              std::shared_ptr<genie::log::Logger> logger,
              std::shared_ptr<qualla::Env> env);
    ~RegEngine();

    // Delete all other constructors
    RegEngine()                            = delete;
    RegEngine(const RegEngine&)            = delete;
    RegEngine(RegEngine&&)                 = delete;
    RegEngine& operator=(const RegEngine&) = delete;
    RegEngine& operator=(RegEngine&&)      = delete;

    void incrementUseCount();
    void decrementUseCount();
    std::string getRole() const;
    uint32_t getUseCount() const;

    std::shared_ptr<genie::Engine> getGenieEngine() const;

   private:
    std::shared_ptr<genie::Engine> m_engine;
    std::string m_role;
    std::atomic<uint32_t> m_useCount;
  };

  static std::unordered_map<size_t, std::shared_ptr<RegEngine>>& getKeyToEngineMap();
  static std::unordered_map<size_t, nlohmann::json>& getKeyToConfigMap();
  static std::mutex& getMutexRegistryFn();
};

}  // namespace genie