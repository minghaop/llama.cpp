//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#include <exception>
#include <set>

#include "Exception.hpp"
#include "Macro.hpp"
#include "ResourceManager.hpp"
#include "Sampler.hpp"
#include "qualla/detail/Log.hpp"
#if ENABLE_DEBUG_LOGS
#include <iostream>
#endif

using namespace genie;

//=============================================================================
// Sampler functions
//=============================================================================

std::atomic<std::uint32_t> Sampler::s_nameCounter{0u};

qnn::util::HandleManager<Sampler>& Sampler::getManager() {
  static qnn::util::HandleManager<Sampler> s_manager;
  return s_manager;
}

GenieSampler_Handle_t Sampler::add(std::shared_ptr<Sampler> sampler) {
  return reinterpret_cast<GenieSampler_Handle_t>(getManager().add(sampler));
}

std::shared_ptr<Sampler> Sampler::get(GenieSampler_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Sampler::remove(GenieSampler_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

Sampler::Sampler(const std::string name,
                 std::unordered_map<std::string, std::shared_ptr<qualla::Sampler>> quallaSamplers)
    : m_name(name), m_quallaSamplers(quallaSamplers) {
  m_isStandaloneSampler = false;
}

Sampler::Sampler(std::shared_ptr<SamplerConfig> config) {
  auto configJson = config->getJson();
  nlohmann::json quallaSamplerConfig;
  SamplerConfig::translateSamplerConfig(configJson["standalone-sampler"], quallaSamplerConfig);
  m_name    = "sampler" + std::to_string(s_nameCounter.fetch_add(1u));
  m_env     = qualla::Env::create(std::make_shared<ResourceManager>(), nlohmann::json{});
  m_context = std::make_shared<Context>(configJson["standalone-sampler"], m_env);
  auto sampler =
      qualla::Sampler::create(m_context->getQuallaContext(), quallaSamplerConfig["sampler"]);
  if (!sampler) {
    throw Exception(GENIE_STATUS_ERROR_MEM_ALLOC, "Could not create a standalone sampler object");
  }
  m_quallaSamplers["primary"] = sampler;
  m_isStandaloneSampler       = true;
}

void Sampler::sampleData(const void* data,
                         size_t dataSize,
                         const char* dataConfigJson,
                         GenieSampler_Callback_t callback,
                         const void* userData) {
  if (m_quallaSamplers.empty()) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "No qualla sampler instances available");
  }

  // Delegate to qualla sampler's sampleData implementation
  const std::vector<int32_t> tokens =
      m_quallaSamplers["primary"]->sampleData(data, dataSize, dataConfigJson);

  callback(static_cast<uint32_t>(tokens.size()), tokens.data(), userData);
}

void Sampler::applyConfig(nlohmann::json samplerConfigJson) {
  // Loop through the live qualla sampler instances and update the parameters
  for (auto& [_, quallaSampler] : m_quallaSamplers) {
    quallaSampler->applyConfig(samplerConfigJson);
  }
}

void Sampler::registerCallback(const char* name, GenieSampler_ProcessCallback_t samplerCallback) {
  QNN_WARN("This API will soon be deprecated in favor of GenieSampler_registerUserDataCallback");
  std::string funcCbName = std::string(name);
  qualla::Sampler::registerProcessCallBack(funcCbName, samplerCallback);
}

void Sampler::registerUserDataCallback(const char* name,
                                       GenieSampler_UserDataCallback_t samplerCallback,
                                       const void* userData) {
  std::string funcCbName = std::string(name);
  qualla::Sampler::registerUserDataCallBack(funcCbName, samplerCallback, userData);
}

//=============================================================================
// Sampler::SamplerConfig functions
//=============================================================================
qnn::util::HandleManager<Sampler::SamplerConfig>& Sampler::SamplerConfig::getManager() {
  static qnn::util::HandleManager<Sampler::SamplerConfig> s_manager;
  return s_manager;
}

GenieSamplerConfig_Handle_t Sampler::SamplerConfig::add(
    std::shared_ptr<Sampler::SamplerConfig> config) {
  return reinterpret_cast<GenieSamplerConfig_Handle_t>(getManager().add(config));
}

std::shared_ptr<Sampler::SamplerConfig> Sampler::SamplerConfig::get(
    GenieSamplerConfig_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Sampler::SamplerConfig::remove(GenieSamplerConfig_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

Sampler::SamplerConfig::SamplerConfig(const char* configStr) {
  nlohmann::json config;
  {
    std::set<nlohmann::json> keys;

    auto callback = [&keys](
                        int depth, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
      if ((depth == 1) && (event == nlohmann::json::parse_event_t::key)) {
        if (keys.count(parsed) > 0) {
          throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                          "Multiple sampler config key: " + parsed.dump());
        }
        keys.insert(parsed);
      }
      return true;
    };

    config = nlohmann::json::parse(configStr, callback);
  }

  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "config is not an object");
  }

  if (!config.contains("sampler") && !config.contains("standalone-sampler")) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing field: sampler or standalone-sampler");
  }

  if (config.contains("standalone-sampler")) m_isStandaloneSampler = true;

  // component is used in the "ENFORCE" macros
  std::string component = "sampler";
  if (m_isStandaloneSampler) {
    component = "standalone-sampler";
  }

  for (auto& item : config.items()) {
    if (item.key() == "sampler") {
      JSON_ENFORCE_OBJECT();
      validateSamplerConfig(item.value());
    } else if (item.key() == "standalone-sampler") {
      JSON_ENFORCE_OBJECT();
      validateStandaloneSamplerConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown " + component + " config key: " + item.key());
    }
  }

  if (m_isStandaloneSampler) {  // store two reference to allow legacy flow in newer flow
    m_standaloneSamplerConfig = config;
    m_samplerConfig           = config["standalone-sampler"];
  } else {
    m_samplerConfig = config;
  }
}

void Sampler::SamplerConfig::setParam(const std::string& keyStr, const std::string& valueStr) {
  if (!keyStr.empty()) {
    // Case 1: Only the parameter mentioned in keyStr is to be updated by valueStr
    std::set<std::string> validParams = {"seed",
                                         "top-p",
                                         "top-k",
                                         "temp",
                                         "type",
                                         "callback-name",
                                         "penalize-last-n",
                                         "repetition-penalty",
                                         "presence-penalty",
                                         "frequency-penalty"};
    if (std::find(validParams.begin(), validParams.end(), keyStr) == validParams.end()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Invalid key obtained: " + keyStr);
    }
    try {
      if (keyStr == "seed")
        m_samplerConfig["sampler"]["seed"] = std::stoi(valueStr);
      else if (keyStr == "top-p")
        m_samplerConfig["sampler"]["top-p"] = std::stof(valueStr);
      else if (keyStr == "top-k")
        m_samplerConfig["sampler"]["top-k"] = std::stof(valueStr);
      else if (keyStr == "temp")
        m_samplerConfig["sampler"]["temp"] = std::stof(valueStr);
      else if (keyStr == "type")
        m_samplerConfig["sampler"]["type"] = valueStr;
      else if (keyStr == "callback-name")
        m_samplerConfig["sampler"]["callback-name"] = valueStr;
      else if (keyStr == "penalize-last-n")
        m_samplerConfig["sampler"]["token-penalty"]["penalize-last-n"] = std::stoi(valueStr);
      else if (keyStr == "repetition-penalty")
        m_samplerConfig["sampler"]["token-penalty"]["repetition-penalty"] = std::stof(valueStr);
      else if (keyStr == "presence-penalty")
        m_samplerConfig["sampler"]["token-penalty"]["presence-penalty"] = std::stof(valueStr);
      else if (keyStr == "frequency-penalty")
        m_samplerConfig["sampler"]["token-penalty"]["frequency-penalty"] = std::stof(valueStr);
    } catch (const std::invalid_argument&) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Invalid value obtained: " + valueStr + " for key: " + keyStr);
    }
  } else {
    // Case 2: User has passed entire json as a string in valueStr

    if (valueStr.empty())
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Both keyStr and valueStr cannot be empty");

    nlohmann::json config = nlohmann::json::parse(valueStr);
    if (!config.contains("sampler")) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing field: sampler");
    }

    // component is used in the "ENFORCE" macros
    const std::string component = "sampler";
    for (auto& item : config.items()) {
      if (item.key() == "sampler") {
        JSON_ENFORCE_OBJECT();
        validateSamplerConfig(item.value());
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                        "Unknown sampler config key: " + item.key());
      }
    }

    m_samplerConfig["sampler"]["seed"] = qualla::Config::optional<int32_t>(
        config["sampler"], "seed", m_samplerConfig["sampler"]["seed"]);
    m_samplerConfig["sampler"]["temp"] = qualla::Config::optional<float>(
        config["sampler"], "temp", m_samplerConfig["sampler"]["temp"]);
    m_samplerConfig["sampler"]["top-k"] = qualla::Config::optional<size_t>(
        config["sampler"], "top-k", m_samplerConfig["sampler"]["top-k"]);
    m_samplerConfig["sampler"]["top-p"] = qualla::Config::optional<float>(
        config["sampler"], "top-p", m_samplerConfig["sampler"]["top-p"]);
    m_samplerConfig["sampler"]["version"] = qualla::Config::optional<int32_t>(
        config["sampler"], "version", m_samplerConfig["sampler"]["version"]);
    if (config["sampler"].contains("type"))
      m_samplerConfig["sampler"]["type"] = config["sampler"]["type"];
    if (config["sampler"].contains("callback-name"))
      m_samplerConfig["sampler"]["callback-name"] = config["sampler"]["callback-name"];

    if (config["sampler"].contains("token-penalty")) {
      if (config["sampler"]["token-penalty"].contains("penalize-last-n")) {
        m_samplerConfig["sampler"]["token-penalty"]["penalize-last-n"] =
            config["sampler"]["token-penalty"]["penalize-last-n"];
      }
      if (config["sampler"]["token-penalty"].contains("repetition-penalty")) {
        m_samplerConfig["sampler"]["token-penalty"]["repetition-penalty"] =
            config["sampler"]["token-penalty"]["repetition-penalty"];
      }
      if (config["sampler"]["token-penalty"].contains("presence-penalty")) {
        m_samplerConfig["sampler"]["token-penalty"]["presence-penalty"] =
            config["sampler"]["token-penalty"]["presence-penalty"];
      }
      if (config["sampler"]["token-penalty"].contains("frequency-penalty")) {
        m_samplerConfig["sampler"]["token-penalty"]["frequency-penalty"] =
            config["sampler"]["token-penalty"]["frequency-penalty"];
      }
    }
  }
}

static void validateTokenPenaltyConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "token-penalty config is not an object");
  }

  const std::set<std::string> mandatoryFields{"version"};

  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing token-penalty field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  const std::string component = "token-penalty";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(
            GENIE_STATUS_ERROR_JSON_VALUE,
            "Invalid token-penalty config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "penalize-last-n") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "repetition-penalty") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "presence-penalty") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "frequency-penalty") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown token-penalty config key: " + item.key());
    }
  }
}
void Sampler::SamplerConfig::validateSamplerConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "sampler config is not an object");
  }

  const std::set<std::string> mandatoryFields{"version"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing sampler field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  const std::string component = "sampler";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid sampler config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "seed") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "temp") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "top-k") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "top-p") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "greedy") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "callback-name") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "token-penalty") {
      validateTokenPenaltyConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown sampler config key: " + item.key());
    }
  }

  // For custom sampler, ensure type = "custom" and callback-name is specified
  if (config.contains("callback-name") && config.contains("type") && config["type"] != "custom") {
    throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                    "callback-name specified but type is set to: " + config["type"].dump() +
                        " Type must be custom");
  }

  if (config.contains("type") && config["type"] == "custom" && !config.contains("callback-name")) {
    throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                    "callback-name not specified but type is set to custom");
  }

  if ((config.contains("type") && config["type"] == "custom") &&
      (config.contains("temp") || config.contains("top-p") || config.contains("top-k") ||
       config.contains("greedy"))) {
    throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                    "Provided keys are not compatible with custom sampler type.");
  }
}

void Sampler::SamplerConfig::validateStandaloneSamplerConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "standalone-sampler config is not an object");
  }

  const std::set<std::string> mandatoryFields{"version", "sampler", "context"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing standalone-sampler field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  const std::string component = "standalone-sampler";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid sampler config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "sampler") {
      JSON_ENFORCE_OBJECT();
      validateSamplerConfig(item.value());
    } else if (item.key() == "context") {
      JSON_ENFORCE_OBJECT();
      Context::ContextConfig::validateContextConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown standalone-sampler config key: " + item.key());
    }
  }
}

void Sampler::SamplerConfig::translateSamplerConfig(const nlohmann::json& genieConfig,
                                                    nlohmann::json& quallaConfig) {
  if (genieConfig.contains("sampler")) {
    quallaConfig["sampler"]["type"] = "basic";
    if (genieConfig["sampler"].contains("seed")) {
      quallaConfig["sampler"]["seed"] = genieConfig["sampler"]["seed"];
    }
    if (genieConfig["sampler"].contains("temp")) {
      quallaConfig["sampler"]["temp"] = genieConfig["sampler"]["temp"];
    }
    if (genieConfig["sampler"].contains("type")) {
      quallaConfig["sampler"]["type"] = genieConfig["sampler"]["type"];
    }
    if (genieConfig["sampler"].contains("callback-name")) {
      quallaConfig["sampler"]["callback-name"] = genieConfig["sampler"]["callback-name"];
    }
    quallaConfig["sampler"]["role"] = "primary";
    if (genieConfig["sampler"].contains("top-k")) {
      quallaConfig["sampler"]["top-k"] = genieConfig["sampler"]["top-k"];
    }
    if (genieConfig["sampler"].contains("top-p")) {
      quallaConfig["sampler"]["top-p"] = genieConfig["sampler"]["top-p"];
    }
    if (genieConfig["sampler"].contains("greedy")) {
      quallaConfig["sampler"]["greedy"] = genieConfig["sampler"]["greedy"];
    }
    if (genieConfig["sampler"].contains("seed")) {
      quallaConfig["sampler"]["seed"] = genieConfig["sampler"]["seed"];
    }
    if (genieConfig["sampler"].contains("token-penalty")) {
      if (genieConfig["sampler"]["token-penalty"].contains("penalize-last-n")) {
        quallaConfig["sampler"]["token-penalty"]["penalize-last-n"] =
            genieConfig["sampler"]["token-penalty"]["penalize-last-n"];
      }
      if (genieConfig["sampler"]["token-penalty"].contains("repetition-penalty")) {
        quallaConfig["sampler"]["token-penalty"]["repetition-penalty"] =
            genieConfig["sampler"]["token-penalty"]["repetition-penalty"];
      }
      if (genieConfig["sampler"]["token-penalty"].contains("presence-penalty")) {
        quallaConfig["sampler"]["token-penalty"]["presence-penalty"] =
            genieConfig["sampler"]["token-penalty"]["presence-penalty"];
      }
      if (genieConfig["sampler"]["token-penalty"].contains("frequency-penalty")) {
        quallaConfig["sampler"]["token-penalty"]["frequency-penalty"] =
            genieConfig["sampler"]["token-penalty"]["frequency-penalty"];
      }
    }
  }
}

nlohmann::json Sampler::SamplerConfig::getSamplerJson() const { return m_samplerConfig; }

nlohmann::json Sampler::SamplerConfig::getJson() const { return m_standaloneSamplerConfig; }
