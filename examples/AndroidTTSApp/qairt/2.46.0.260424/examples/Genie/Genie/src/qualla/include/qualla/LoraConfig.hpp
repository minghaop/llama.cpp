//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "detail/config.hpp"
#include "env.hpp"

namespace qualla {

enum class LoraConfigType : uint8_t {
  LORA_DISABLE               = 0,
  LORA_INPUT_WEIGHT_ENABLE   = 1,
  LORA_ADAPTER_WEIGHT_ENABLE = 2
};

enum class LoraEventType : uint8_t { NO_EVENT = 0, APPLY_EVENT = 1 };

namespace fs = std::filesystem;

class LoraAdapter {
 public:
  LoraAdapter(nlohmann::json& config,
              LoraConfigType configType = LoraConfigType::LORA_DISABLE,
              std::string baseDir       = "",
              std::shared_ptr<Env> env  = nullptr);

  void addGroupInfo(std::string& groupName, std::vector<std::string>& quantBinsList);

  LoraAdapter()  = delete;
  ~LoraAdapter() = default;

  // Copy constructor and copy-assignment
  LoraAdapter(const LoraAdapter& other) = delete;

  LoraAdapter& operator=(const LoraAdapter& other) = delete;

  // delete move and assignment
  LoraAdapter(LoraAdapter&& orig)            = delete;
  LoraAdapter& operator=(LoraAdapter&& orig) = delete;

  // members
  std::string m_adapterName;                   // name of this adapter or weight
  std::string m_groupName;                     // LoRA group name
  std::vector<std::string> m_binList;          // list of bins to apply
  std::vector<std::string> m_quantBinList;     // list of quant bins to apply
  std::string m_weightPath;                    // path to weight directory
  std::string m_alphaTensorName;               // name of alpha Tensor
  std::vector<std::string> m_alphaTensorList;  // name of group of alpha Tensor
  std::vector<float> m_alphaTensorVal;         // all the alpha values
  std::string m_metadataDlc;                   // DLC metadata, used for updating V3 adapter
  std::shared_ptr<Env> _env;
};

class LoraConfig {
 public:
  LoraConfig(Config& config, std::shared_ptr<Env> env);
  LoraConfig()  = delete;
  ~LoraConfig() = default;

  // copy constructor and assignment operator
  LoraConfig(const LoraConfig& other) = default;
  LoraConfig& operator=(const LoraConfig& other);

  // move constructor and assignment operator
  LoraConfig(LoraConfig&& other)            = delete;
  LoraConfig& operator=(LoraConfig&& other) = delete;

  // Methods
  std::shared_ptr<LoraAdapter> getAppliedAdapter();
  std::shared_ptr<LoraAdapter> getAdapter(const std::string& name);
  void updateAppliedAdapterName(const std::string& name);
  std::string getAppliedAdapterName();
  bool hasAlpha(const std::string& name);
  std::string getAlphaTensorName();
  LoraConfigType getLoraConfigType();
  LoraEventType getEventType();
  float getCachedAlphaVal(const std::string& name);
  void updateCacheAlphaVal(const std::string& name, const float val);

 private:
  // members
  std::string m_appliedAdapter    = "";  // empty corresponds to none is applied
  std::string m_alphaTensorName   = "";  // assume all adapters use same alpha Tensor name
  LoraConfigType m_loraConfigType = LoraConfigType::LORA_DISABLE;
  LoraEventType m_event           = LoraEventType::NO_EVENT;
  std::unordered_map<std::string, std::shared_ptr<LoraAdapter>>
      m_loraAdapterList;                                        // Map of {name , LoRA adapter}
  std::unordered_map<std::string, float> m_cachedLoraAlphaVal;  // cached alpha values
  std::shared_ptr<Env> _env;
};

}  // namespace qualla