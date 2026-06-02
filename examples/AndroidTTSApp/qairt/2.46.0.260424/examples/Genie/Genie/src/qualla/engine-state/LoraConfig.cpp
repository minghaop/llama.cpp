//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <qualla/detail/Log.hpp>
#include <qualla/detail/utils.hpp>

#include "ResourceManager.hpp"
#include "qualla/LoraConfig.hpp"

namespace qualla {

LoraAdapter::LoraAdapter(nlohmann::json& config,
                         LoraConfigType configType,
                         std::string baseDir,
                         std::shared_ptr<Env> env)
    : _env(env) {
  // Init adapter case
  m_adapterName     = config["adapter-name"];
  m_alphaTensorName = config["alpha-tensor-name"];
  __DEBUG("LoraAdapter-new: {} config {}", m_adapterName, config.dump());

  for (auto& val : config["alpha-tensor-value"]) {
    m_alphaTensorVal.push_back(val.get<float>());
  }
  bool skipDefaultAlphaVal = config["alpha-tensor-value"].size() == config["alphas"].size();
  if (!skipDefaultAlphaVal) {
    m_alphaTensorVal.clear();
  }
  for (auto& alpha : config["alphas"]) {
    m_alphaTensorList.push_back(alpha.get<std::string>());
    if (!skipDefaultAlphaVal) {
      m_alphaTensorVal.push_back(1.0f);
    }
  }
  if (config.contains("metadata-dlc")) {
    m_metadataDlc = config["metadata-dlc"].get<std::string>();
  }
  if (configType == LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE) {
    auto resourceManager = env->getResourceManager();
    for (auto& curSection : config["bin-sections"]) {
      auto binSection = utils::trim(curSection);
      if (binSection.empty())  // empty is treated as skip lora case
      {
        m_binList.push_back(binSection);
        continue;
      }
      // DLC record paths (record://) are resolved via ResourceManager, not the filesystem
      if (resourceManager->isDLCRecord(binSection)) {
        if (resourceManager->getBufferSize(binSection) == 0) {
          __ERROR("LoRA: Can't access Lora binsection DLC record : {}", binSection);
          throw std::runtime_error("LoRA: Can't access adapter DLC record : " + binSection);
        }
        m_binList.push_back(binSection);
        continue;
      }
      // validate and early error out
      fs::path binPath = fs::path(binSection);
      if (binPath.is_relative()) binPath = baseDir / fs::path(binPath);
      if (!fs::is_regular_file(binPath)) {
        __ERROR("LoRA: Can't access Lora binsection adapter : {}", binPath.string());
        throw std::runtime_error("LoRA: Can't access adapter file : " + binPath.string());
      }
      m_binList.push_back(binPath.string());
    }
  } else if (configType == LoraConfigType::LORA_INPUT_WEIGHT_ENABLE) {
    m_weightPath = utils::trim(config["path"]);
    if (m_weightPath.empty()) {
      __ERROR("LoRA: Weight path is empty after trimming");
      throw std::runtime_error("LoRA: Weight path cannot be empty");
    }
  }
}

void LoraAdapter::addGroupInfo(std::string& groupName, std::vector<std::string>& quantBinsList) {
  m_groupName    = groupName;
  m_quantBinList = quantBinsList;
}

LoraConfig::LoraConfig(Config& config, std::shared_ptr<Env> env) : _env(env) {
  // base Directory
  std::string baseDir = config.optional<std::string>("binsection-basedir", "");
  // Default LoRA is Disabled
  m_loraConfigType = LoraConfigType::LORA_DISABLE;
  switch (config.optional<uint8_t>("lora-version", 0)) {
    case 0:
      m_loraConfigType = LoraConfigType::LORA_DISABLE;
      break;
    case 1:
      m_loraConfigType = LoraConfigType::LORA_INPUT_WEIGHT_ENABLE;
      break;
    case 2:
      m_loraConfigType = LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE;
      break;
    default:
      throw std::runtime_error("Lora Version Undefined.");
  }

  // LoRA adapter Config
  auto adapterConfig = config.optional<nlohmann::json>("lora", {});
  if (!adapterConfig.empty() && adapterConfig.is_array()) {
    for (auto& lc : adapterConfig) {
      auto adapter = std::make_shared<LoraAdapter>(lc, m_loraConfigType, baseDir, _env);
      m_loraAdapterList[adapter->m_adapterName] = adapter;
    }
  }
  auto groupConfig = config.optional<nlohmann::json>("group", {});

  if (!groupConfig.empty()) {
    __DEBUG("LoRA config is group enabled");
  } else {
    __DEBUG("LoRA config is group disabled");
  }

  if (!groupConfig.empty() && groupConfig.is_array()) {
    // mark the following as invalid in the config validation, assume it is valid here.
    // if (_params.lazy_lora == "lazy") {
    //   throw std::runtime_error("qnn-htp: Grouped Lora cannot be used with lazy-lora config.");
    // }
    if (m_loraConfigType == LoraConfigType::LORA_INPUT_WEIGHT_ENABLE) {
      throw std::runtime_error("LoRA: Grouped Lora cannot be used for LoraV1");
    }
    for (auto& curConfig : groupConfig) {
      // Init group adapter case
      std::string groupName = Config::mandatory<std::string>(curConfig, "name");
      __DEBUG("LoraGroupConfig-new: {} config {}", groupName, curConfig.dump());
      std::vector<std::string> quantBinSection;
      auto resourceManager = _env->getResourceManager();
      for (auto& curSection : curConfig["quant-bin-sections"]) {
        auto binSection = utils::trim(curSection);
        if (binSection.empty())  // empty is treated as skip lora case
        {
          quantBinSection.push_back(binSection);
          continue;
        }
        // DLC record paths (record://) are resolved via ResourceManager, not the filesystem
        if (resourceManager->isDLCRecord(binSection)) {
          if (resourceManager->getBufferSize(binSection) == 0) {
            __ERROR("LoRA: Can't access Lora quantbinsection DLC record : {}", binSection);
            throw std::runtime_error("LoRA: Can't find adapter DLC record : " + binSection);
          }
          quantBinSection.push_back(binSection);
          continue;
        }
        // validate and early error out
        fs::path binPath = fs::path(binSection);
        if (binPath.is_relative()) binPath = baseDir / fs::path(binPath);
        if (!fs::is_regular_file(binPath)) {
          __ERROR("LoRA: Can't access Lora quantbinsection adapter : {}", binPath.string());
          throw std::runtime_error("LoRA: Can't find adapter file : " + binPath.string());
        }
        quantBinSection.push_back(binPath.string());
      }
      // apply info to all groups
      for (auto& member : curConfig["members"]) {
        auto memberName = member.get<std::string>();
        if (!m_loraAdapterList.contains(memberName)) {
          __ERROR("Wrong {} member is specified for lora group {}", memberName, groupName);
          throw std::runtime_error("LoRA: Invalid member '" + memberName + "' in group '" +
                                   groupName);
        }
        m_loraAdapterList[memberName]->addGroupInfo(groupName, quantBinSection);
      }
    }
  }

  if (!m_loraAdapterList.empty()) {
    m_alphaTensorName = m_loraAdapterList.begin()->second->m_alphaTensorName;
    // Cached alpha Val list
    for (auto& [_, loraAdapter] : m_loraAdapterList) {
      if (loraAdapter && !loraAdapter->m_alphaTensorVal.empty()) {
        for (uint32_t idx = 0; idx < static_cast<uint32_t>(loraAdapter->m_alphaTensorVal.size());
             idx++) {
          m_cachedLoraAlphaVal[loraAdapter->m_alphaTensorList[idx]] =
              loraAdapter->m_alphaTensorVal[idx];
        }
      }
    }
  }
}

LoraConfig& LoraConfig::operator=(const LoraConfig& other) {
  if (this != &other) {
    m_appliedAdapter     = other.m_appliedAdapter;
    m_alphaTensorName    = other.m_alphaTensorName;
    m_loraConfigType     = other.m_loraConfigType;
    m_event              = LoraEventType::APPLY_EVENT;
    m_loraAdapterList    = other.m_loraAdapterList;
    m_cachedLoraAlphaVal = other.m_cachedLoraAlphaVal;
  } else {
    m_event = LoraEventType::NO_EVENT;
  }
  return *this;
}

std::shared_ptr<LoraAdapter> LoraConfig::getAppliedAdapter() {
  if (m_appliedAdapter.empty()) return {};
  return m_loraAdapterList[m_appliedAdapter];
}

std::shared_ptr<LoraAdapter> LoraConfig::getAdapter(const std::string& name) {
  if (!m_loraAdapterList.contains(name)) return {};
  return m_loraAdapterList[name];
}

void LoraConfig::updateAppliedAdapterName(const std::string& name) {
  m_appliedAdapter = name;
  m_event          = LoraEventType::APPLY_EVENT;
}

std::string LoraConfig::getAppliedAdapterName() { return m_appliedAdapter; }

bool LoraConfig::hasAlpha(const std::string& name) { return m_cachedLoraAlphaVal.contains(name); }

std::string LoraConfig::getAlphaTensorName() { return m_alphaTensorName; }

LoraConfigType LoraConfig::getLoraConfigType() { return m_loraConfigType; }

LoraEventType LoraConfig::getEventType() { return m_event; }

float LoraConfig::getCachedAlphaVal(const std::string& name) {
  if (!hasAlpha(name)) return 0.0f;
  return m_cachedLoraAlphaVal[name];
}

void LoraConfig::updateCacheAlphaVal(const std::string& name, const float val) {
  if (!hasAlpha(name)) return;
  m_cachedLoraAlphaVal[name] = val;
}

}  // namespace qualla
