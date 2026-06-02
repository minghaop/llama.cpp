//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "Config.hpp"
#ifdef QUALLA_ENGINE_QNN_HTP
#include "QnnHtpCommon.h"
#include "QnnHtpContext.h"
#else
#include "QnnContext.h"
#endif  // QUALLA_ENGINE_QNN_HTP
#include "QnnTypes.h"

class QnnContextConfig : public GenericConfig<QnnContext_Config_t> {
 public:
  QnnContextConfig(const QnnContext_Config_t config) : GenericConfig<QnnContext_Config_t>(config){};
  QnnContextConfig()          = default;
  virtual ~QnnContextConfig() = default;

  virtual void serialize(nlohmann::ordered_json& json) const override;
};

class ContextEnableGraphsConfig final : public QnnContextConfig {
 private:
  std::vector<std::string> m_enableGraphs;
  std::vector<const char*> m_enableGraphsPtr;

 public:
  ContextEnableGraphsConfig(std::vector<std::string> enableGraphs) : m_enableGraphs(enableGraphs) {
    for (std::string& graphName : m_enableGraphs) {
      m_enableGraphsPtr.push_back(graphName.c_str());
    }
    m_enableGraphsPtr.push_back(nullptr);

    m_config.option       = QNN_CONTEXT_CONFIG_ENABLE_GRAPHS;
    m_config.enableGraphs = m_enableGraphsPtr.data();
  };
};

class ContextOemStringConfig final : public QnnContextConfig {
 private:
  std::string m_oemString;

 public:
  ContextOemStringConfig(std::string oemString) : m_oemString(oemString) {
    m_config.option    = QNN_CONTEXT_CONFIG_OPTION_OEM_STRING;
    m_config.oemString = m_oemString.c_str();
  };
};

class ContextMemoryLimitHintConfig final : public QnnContextConfig {
 public:
  ContextMemoryLimitHintConfig(uint64_t memoryLimitHint) {
    m_config.option          = QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT;
    m_config.memoryLimitHint = memoryLimitHint;
  };
};

class ContextPersistentBinaryConfig final : public QnnContextConfig {
 public:
  ContextPersistentBinaryConfig(bool isPersistentBinary) {
    m_config.option             = QNN_CONTEXT_CONFIG_PERSISTENT_BINARY;
    m_config.isPersistentBinary = isPersistentBinary;
  };
};
#ifdef QUALLA_ENGINE_QNN_HTP
class ContextCustomHtpConfig final : public QnnContextConfig {
 private:
  QnnHtpContext_CustomConfig_t m_customConfig;

 public:
  ContextCustomHtpConfig(QnnHtpContext_CustomConfig_t customConfig) : m_customConfig(customConfig) {
    m_config.option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    m_config.customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&m_customConfig);
  };

  virtual void serialize(nlohmann::ordered_json& json) const override;
};
#endif  // QUALLA_ENGINE_QNN_HTP
