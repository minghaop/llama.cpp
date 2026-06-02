//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "Registry.hpp"
using namespace genie;

std::unordered_map<size_t, std::shared_ptr<Registry::RegEngine>>& Registry::getKeyToEngineMap() {
  static std::unordered_map<size_t, std::shared_ptr<Registry::RegEngine>> s_keyEngineMap;
  return s_keyEngineMap;
}

std::unordered_map<size_t, nlohmann::json>& Registry::getKeyToConfigMap() {
  static std::unordered_map<size_t, nlohmann::json> s_keyToConfigMap;
  return s_keyToConfigMap;
}

std::mutex& Registry::getMutexRegistryFn() {
  static std::mutex s_mutexRegistryFn;
  return s_mutexRegistryFn;
}

size_t Registry::EngineKey::createKey(const nlohmann::json& engineConfig) {
  // Generate a unique key based on the engine config
  size_t engineKey = 0;
  if (engineConfig["backend"].contains("QnnHtp")) {
    std::string allowAsyncInit =
        std::to_string(static_cast<bool>(engineConfig["backend"]["QnnHtp"]["allow-async-init"]));
    std::string kvDim =
        std::to_string(static_cast<uint32_t>(engineConfig["backend"]["QnnHtp"]["kv-dim"]));

    std::string models;
    for (const auto& model : engineConfig["model"]["binary"]["ctx-bins"]) {
      models += model;
    }
    std::string key = allowAsyncInit + kvDim + models;
    engineKey       = std::hash<std::string>{}(key);
  } else if (engineConfig["backend"].contains("QnnGenAiTransformer")) {
    std::string kv_quantization = std::to_string(false);
    if (engineConfig["backend"]["QnnGenAiTransformer"].contains("kv-quantization")) {
      kv_quantization = std::to_string(
          static_cast<bool>(engineConfig["backend"]["QnnGenAiTransformer"]["kv-quantization"]));
    }
    std::string model = engineConfig["model"]["library"]["model-bin"];

    std::string key = model + kv_quantization;
    engineKey       = std::hash<std::string>{}(key);
  }
  return engineKey;
}

std::vector<std::pair<std::string, size_t>> Registry::getKeysFromRegistry(
    const nlohmann::json& config) {
  std::lock_guard<std::mutex> lock(getMutexRegistryFn());

  auto& keyToConfigMap = getKeyToConfigMap();
  std::vector<std::pair<std::string, size_t>> keys;
  for (const auto& engineConfig : config["shared-engines"]) {
    EngineKey engKey;
    size_t key = engKey.createKey(engineConfig["standalone-engine"]["engine"]);
    std::string role =
        Engine::changeRole(engineConfig["standalone-engine"]["engine"].contains("role")
                               ? engineConfig["standalone-engine"]["engine"]["role"]
                               : "primary");
    keyToConfigMap[key] = engineConfig;
    keys.push_back({role, key});
  }
  return keys;
}

std::unordered_map<std::string, std::shared_ptr<qualla::Engine>> Registry::getEngineFromRegistry(
    const std::vector<std::pair<std::string, size_t>>& keys,
    std::shared_ptr<ProfileStat> profileStat,
    std::shared_ptr<genie::log::Logger> logger,
    std::shared_ptr<qualla::Env> env) {
  std::lock_guard<std::mutex> lock(getMutexRegistryFn());

  std::unordered_map<std::string, std::shared_ptr<qualla::Engine>> result;
  auto& keyToConfigMap = getKeyToConfigMap();
  auto& keyToEngineMap = getKeyToEngineMap();
  for (const auto& [role, key] : keys) {
    if (!keyToEngineMap.contains(key)) {
      // Create a new engine and add it to the map
      keyToEngineMap[key] =
          std::make_shared<RegEngine>(keyToConfigMap[key], profileStat, logger, env);
    }
    keyToEngineMap[key]->incrementUseCount();
    result[role] = keyToEngineMap[key]->getGenieEngine()->getEngine();
  }
  return result;
}

void Registry::deleteEnginesFromRegistry(const std::vector<std::pair<std::string, size_t>>& keys) {
  std::lock_guard<std::mutex> lock(getMutexRegistryFn());

  auto& keyToConfigMap = getKeyToConfigMap();
  auto& keyToEngineMap = getKeyToEngineMap();
  for (const auto& [_, key] : keys) {
    if (keyToEngineMap.contains(key)) {
      keyToEngineMap[key]->decrementUseCount();
      if (keyToEngineMap[key]->getUseCount() == 1) {
        // Destroy the engine object
        keyToEngineMap.erase(key);
        keyToConfigMap.erase(key);
      }
    }
  }
}

Registry::RegEngine::RegEngine(const nlohmann::json& engineConfig,
                               std::shared_ptr<ProfileStat> profileStat,
                               std::shared_ptr<genie::log::Logger> logger,
                               std::shared_ptr<qualla::Env> env) {
  // Initialize the engine
  auto engineConfigPtr = std::make_shared<genie::Engine::EngineConfig>(engineConfig.dump().c_str());
  m_engine             = std::make_shared<genie::Engine>(engineConfigPtr, profileStat, logger, env);
  m_role               = (engineConfig["standalone-engine"]["engine"].contains("role"))
                             ? Engine::changeRole(engineConfig["standalone-engine"]["engine"]["role"])
                             : Engine::changeRole("primary");
  m_useCount           = 1;
}

Registry::RegEngine::~RegEngine() {
  // Clean up resources
}

void Registry::RegEngine::incrementUseCount() { ++m_useCount; }

void Registry::RegEngine::decrementUseCount() { --m_useCount; }

uint32_t Registry::RegEngine::getUseCount() const { return m_useCount; }

std::string Registry::RegEngine::getRole() const { return m_role; }

std::shared_ptr<genie::Engine> Registry::RegEngine::getGenieEngine() const { return m_engine; }