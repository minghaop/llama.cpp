//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <exception>
#include <fstream>
#include <set>
#include <sstream>

#include "EncoderDecoder.hpp"
#include "Exception.hpp"
#include "Macro.hpp"
#include "ResourceManager.hpp"
#include "nlohmann/json.hpp"
#include "qualla/detail/utils.hpp"
#include "qualla/env.hpp"

using namespace genie;

#ifdef _WIN32
static constexpr const char* libPrefix = "";
static constexpr const char* libSuffix = ".dll";
#else
static constexpr const char* libPrefix = "lib";
static constexpr const char* libSuffix = ".so";
#endif

inline std::string getLibName(std::string baseName) { return libPrefix + baseName + libSuffix; }

//=============================================================================
// EncoderDecoder::Config functions
//=============================================================================
qnn::util::HandleManager<EncoderDecoder::Config>& EncoderDecoder::Config::getManager() {
  static qnn::util::HandleManager<EncoderDecoder::Config> s_manager;
  return s_manager;
}

GenieNodeConfig_Handle_t EncoderDecoder::Config::add(
    std::shared_ptr<EncoderDecoder::Config> config) {
  return reinterpret_cast<GenieNodeConfig_Handle_t>(getManager().add(config));
}

std::shared_ptr<EncoderDecoder::Config> EncoderDecoder::Config::get(
    GenieNodeConfig_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void EncoderDecoder::Config::remove(GenieNodeConfig_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

EncoderDecoder::Config::Config(const char* configStr) {
  nlohmann::json config;

  {
    std::set<nlohmann::json> keys;

    auto callback = [&keys](
                        int depth, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
      if ((depth == 1) && (event == nlohmann::json::parse_event_t::key)) {
        if (keys.count(parsed) > 0) {
          throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                          "Multiple encoder-decoder config key: " + parsed.dump());
        }
        keys.insert(parsed);
      }
      return true;
    };

    config = nlohmann::json::parse(configStr, callback);
  }

  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Encoder-decoder config is not an object");
  }

  std::set<std::string> mandatoryFields{"encoder-decoder"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing encoder-decoder field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "encoder-decoder";

  for (auto& item : config.items()) {
    if (item.key() == "encoder-decoder") {
      JSON_ENFORCE_OBJECT();
      validateEncoderDecoderConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown encoder-decoder config key: " + item.key());
    }
  }
  m_config = config;
}

nlohmann::json& EncoderDecoder::Config::getJson() { return m_config; }

void EncoderDecoder::Config::bindProfiler(std::shared_ptr<Profiler> profiler) {
  if (!profiler) return;
  profiler->incrementUseCount();
  m_profiler.insert(profiler);
}

void EncoderDecoder::Config::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& EncoderDecoder::Config::getProfiler() {
  return m_profiler;
}

void EncoderDecoder::Config::bindLogger(std::shared_ptr<genie::log::Logger> logger) {
  if (!logger) return;
  logger->incrementUseCount();
  m_logger.insert(logger);
}

void EncoderDecoder::Config::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& EncoderDecoder::Config::getLogger() {
  return m_logger;
}

static void validateLoraAdapterConfig(const nlohmann::json& config,
                                      LORA_VERSION& specifiedLoraVersion) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "lora adapter config is not an object");
  }
  const std::set<std::string> mandatoryFields{"version", "name"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing lora adapter field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  const std::string component        = "lora adapter";
  LORA_VERSION configuredLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_UNDEFINED;
  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid lora config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "name") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "alphas") {
      JSON_ENFORCE_ARRAY();
      configuredLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V2;  // alphas occurs with V2 and V3
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE, "alphas must be an array of strings");
        }
      }
    } else if (item.key() == "metadata-dlc") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "bin-sections") {
      JSON_ENFORCE_ARRAY();
      configuredLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V2;  // Adapter occurs with V2 and V3
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                          "bin-sections must be an array of strings");
        }
      }
    } else if (item.key() == "path") {
      configuredLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V1;  // Weights are V1
      JSON_ENFORCE_STRING();
      // Note:all directory validations will done by NSP engine
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown lora adapter config key: " + item.key());
    }
  }

  if (specifiedLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V1 &&
      (configuredLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V2 ||
       configuredLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V3)) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                    "LoRA Adapters must be used with lora version: 2 or 3");
  } else if ((specifiedLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V2 ||
              specifiedLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V3) &&
             configuredLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V1) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                    "LoRA Weights must be used with lora version: 1");
  } else if (configuredLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_UNDEFINED) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Invalid lora config.");
  }
}

static void validateLoraConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "lora config is not an object");
  }

  const std::set<std::string> mandatoryFields{"version", "adapters"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing lora field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  const std::string component       = "lora";
  LORA_VERSION specifiedLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V2;  // Default is loraV2
  if (config.find("lora-version") != config.end()) {
    switch (static_cast<uint8_t>(config["lora-version"])) {
      case 1:
        specifiedLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V1;
        break;
      case 2:
        specifiedLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V2;
        break;
      case 3:
        specifiedLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V3;
        break;
      default:
        specifiedLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_UNDEFINED;
        break;
    }
  }

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid lora config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "alpha-tensor-name") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "adapters") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        validateLoraAdapterConfig(elem, specifiedLoraVersion);
      }
    } else if (item.key() == "lora-version") {  // Optional
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown lora config key: " + item.key());
    }
  }
  if (specifiedLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_UNDEFINED) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                    "Unsupported lora version: " + to_string(config["lora-version"]));
  }
}

static void validateModelCheckpointConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "checkpoint config is not an object");
  }

  // component is used in the "ENFORCE" macros
  std::string component = "checkpoint";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
            "Invalid checkpoint config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "model-basedir") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "output-dir") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "cache-dir") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "model-json-path") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "clipping-map-path") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "trainable-params-path") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "data-type") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "lora") {
      JSON_ENFORCE_OBJECT();
      validateLoraConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
          "Unknown checkpoint config key: " + item.key());
    }
  }
}

static void validateModelConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "model config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "type"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing model field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "model";

  std::string type;
  // bool binary = false;
  nlohmann::json binaryConfig;
  // bool library = false;
  nlohmann::json libraryConfig;
  bool checkpoint = false;
  nlohmann::json checkpointConfig;
  nlohmann::json positionalEncodingConfig;
  // bool positionalEncoding = false;
  nlohmann::json visionParamConfig;

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid model config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
      type = item.value().get<std::string>();
      if (type == "checkpoint") {
        checkpoint = true;
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid model config: unsupported type: " + item.value().dump());
      }
    } else if (item.key() == "checkpoint") {
      JSON_ENFORCE_OBJECT();
      checkpointConfig = item.value();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown model config key: " + item.key());
    }
  }

  if (checkpoint) {
    if (!checkpointConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing binary model config");
    }
    validateModelCheckpointConfig(checkpointConfig);
  } else {
    if (checkpointConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "checkpoint model config for incorrect model type: " + type);
    }
  }
}

static void validateTrainingConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "training-config is not an object");
  }

  // component is used in the "ENFORCE" macros
  std::string component = "training-config";

  // Mandatory fields
  std::set<std::string> mandatoryFields{
      "random-seed", "torch-num-threads", "torch-num-inter-op-threads"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing training-config field: " + field);
    }
  }

  for (auto& item : config.items()) {
    if (item.key() == "random-seed") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "torch-num-threads") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "torch-num-inter-op-threads") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "max-iterations") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "batch-size") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "softmax-topk") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "gclip-max-norm") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "enable-logit-masking") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "enable-clear-torch-op-cache") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "loss-type") {
      JSON_ENFORCE_STRING();
      std::string lossType                           = item.value().get<std::string>();
      const std::set<std::string> supportedLossTypes = {"mse", "cross_entropy"};
      if (supportedLossTypes.find(lossType) == supportedLossTypes.end()) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE, "Unsupported loss-type: " + lossType);
      }
    } else if (item.key() == "grad-scaler") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown training-config key: " + item.key());
    }
  }
}

static void validateOptimizerConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "optimizer-config is not an object");
  }

  // component is used in the "ENFORCE" macros
  std::string component = "optimizer-config";

  for (auto& item : config.items()) {
    if (item.key() == "opt-name") {
      JSON_ENFORCE_STRING();
      std::string optName                             = item.value().get<std::string>();
      const std::set<std::string> supportedOptimizers = {"SGD", "AdamW"};
      if (supportedOptimizers.find(optName) == supportedOptimizers.end()) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Unsupported optimizer opt-name: " + optName);
      }
    } else if (item.key() == "learning-rate") {
      JSON_ENFORCE_NUMERIC();
      double lr = item.value().get<double>();
      if (lr <= 0.0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "learning-rate must be positive, got: " + std::to_string(lr));
      }
    } else if (item.key() == "wd" || item.key() == "weight-decay") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "momentum") {
      JSON_ENFORCE_NUMERIC();
      double momentum = item.value().get<double>();
      if (momentum < 0.0 || momentum > 1.0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "momentum must be between 0 and 1, got: " + std::to_string(momentum));
      }
    } else if (item.key() == "beta_1" || item.key() == "beta-1") {
      JSON_ENFORCE_NUMERIC();
      double beta1 = item.value().get<double>();
      if (beta1 < 0.0 || beta1 > 1.0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "beta_1 must be between 0 and 1, got: " + std::to_string(beta1));
      }
    } else if (item.key() == "beta_2" || item.key() == "beta-2") {
      JSON_ENFORCE_NUMERIC();
      double beta2 = item.value().get<double>();
      if (beta2 < 0.0 || beta2 > 1.0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "beta_2 must be between 0 and 1, got: " + std::to_string(beta2));
      }
    } else if (item.key() == "eps" || item.key() == "epsilon") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown optimizer-config key: " + item.key());
    }
  }
}

static void validateBackendConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "backend config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "type"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing backend field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "backend";

  std::string type;
  // bool htp = false;
  nlohmann::json htpConfig;
  // bool genai = false;
  nlohmann::json genaiConfig;

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid backend config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
      type = item.value().get<std::string>();
      if (type == "libtorch") {
        // Placeholder to validate libTorch parameters
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid backend config: unsupported type: " + item.value().dump());
      }
    } else if (item.key() == "device") {
      JSON_ENFORCE_STRING();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown backend config key: " + item.key());
    }
  }
}

//=============================================================================
// Scheduler::Config functions
//=============================================================================

static void validateSchedulerConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "scheduler config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "path"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing scheduler field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "scheduler";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid scheduler config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "path") {
      JSON_ENFORCE_STRING();
      // Note: the existence of this file is checked by qualla
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown scheduler config key: " + item.key());
    }
  }
}

static void translateSchedulerConfig(const nlohmann::json& genieConfig,
                                     nlohmann::json& quallaConfig) {
  quallaConfig["scheduler"]["path"] = genieConfig["scheduler"]["path"];
}

//=============================================================================
// Engine::Config functions
//=============================================================================

static void validateEngineConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "engine config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "backend", "model"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing engine field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "engine";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid engine config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "backend") {
      JSON_ENFORCE_OBJECT();
      validateBackendConfig(item.value());
    } else if (item.key() == "mode") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "model") {
      JSON_ENFORCE_OBJECT();
      validateModelConfig(item.value());
    } else if (item.key() == "training-config") {
      JSON_ENFORCE_OBJECT();
      validateTrainingConfig(item.value());
    } else if (item.key() == "optimizer-config") {
      JSON_ENFORCE_OBJECT();
      validateOptimizerConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown engine config key: " + item.key());
    }
  }
}

//=============================================================================
// EncoderDecoder validation and translation functions
//=============================================================================

void EncoderDecoder::validateEncoderDecoderConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Encoder-decoder config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "type"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing encoder-decoder field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "encoder-decoder";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(
            GENIE_STATUS_ERROR_JSON_VALUE,
            "Invalid encoder-decoder config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
      const std::set<std::string> supportedTypes = {"unet"};
      if (std::find(supportedTypes.begin(), supportedTypes.end(), std::string(item.value())) ==
          supportedTypes.end()) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Unknown encoder-decoder type: " + std::string(item.value()));
      }
    } else if (item.key() == "engine") {
      JSON_ENFORCE_OBJECT();
      validateEngineConfig(config["engine"]);
    } else if (item.key() == "scheduler") {
      JSON_ENFORCE_OBJECT();
      validateSchedulerConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown encoder-decoder config key: " + item.key());
    }
  }
}

static void translateLoraConfig(const nlohmann::json& genieLoraConfig,
                                nlohmann::json& quallaLoraConfig) {
  if (genieLoraConfig.contains("role")) quallaLoraConfig["role"] = genieLoraConfig["role"];
  quallaLoraConfig["lora-version"] = static_cast<uint8_t>(LORA_VERSION::GENIE_LORA_VERSION_V2);
  if (genieLoraConfig.contains("lora-version") && genieLoraConfig["lora-version"] == 1) {
    quallaLoraConfig["lora-version"] = genieLoraConfig["lora-version"];
  }
  if (genieLoraConfig.contains("metadata-dlc")) {
    quallaLoraConfig["metadata-dlc"] = genieLoraConfig["metadata-dlc"];
  }
  for (size_t i = 0; i < genieLoraConfig["adapters"].size(); i++) {
    quallaLoraConfig["lora"][i]["adapter-name"]      = genieLoraConfig["adapters"][i]["name"];
    quallaLoraConfig["lora"][i]["alpha-tensor-name"] = "";
    if (genieLoraConfig.contains("alpha-tensor-name")) {
      quallaLoraConfig["lora"][i]["alpha-tensor-name"] = genieLoraConfig["alpha-tensor-name"];
    }
    quallaLoraConfig["lora"][i]["alphas"] = nlohmann::json::array();
    if (genieLoraConfig["adapters"][i].contains("alphas")) {
      quallaLoraConfig["lora"][i]["alphas"] = genieLoraConfig["adapters"][i]["alphas"];
    } else {
      if (genieLoraConfig.contains("alpha-tensor-name")) {
        quallaLoraConfig["lora"][i]["alphas"].emplace_back(genieLoraConfig["alpha-tensor-name"]);
      }
    }
    quallaLoraConfig["lora"][i]["alpha-tensor-value"] = nlohmann::json::array();
    quallaLoraConfig["lora"][i]["binsection-basedir"] = "";
    if (genieLoraConfig.contains("lora-version") && genieLoraConfig["lora-version"] == 1) {
      quallaLoraConfig["lora"][i]["path"] = genieLoraConfig["adapters"][i]["path"];
    } else {
      quallaLoraConfig["lora"][i]["bin-sections"] = genieLoraConfig["adapters"][i]["bin-sections"];
    }
    if (genieLoraConfig["adapters"][i].contains("metadata-dlc")) {
      quallaLoraConfig["lora"][i]["metadata-dlc"] = genieLoraConfig["adapters"][i]["metadata-dlc"];
    }
  }
  if (genieLoraConfig.contains("groups")) {
    for (size_t i = 0; i < genieLoraConfig["groups"].size(); i++) {
      quallaLoraConfig["group"][i]["name"]               = genieLoraConfig["groups"][i]["name"];
      quallaLoraConfig["group"][i]["members"]            = genieLoraConfig["groups"][i]["members"];
      quallaLoraConfig["group"][i]["binsection-basedir"] = "";
      quallaLoraConfig["group"][i]["quant-bin-sections"] =
          genieLoraConfig["groups"][i]["quant-bin-sections"];
    }
  }
}

static void translateEngineConfig(const nlohmann::json& genieEngineConfig,
                                  nlohmann::json& quallaEngineConfig) {
  if (genieEngineConfig["version"] == 1) {
    if (genieEngineConfig["backend"]["type"] == "libtorch") {
      quallaEngineConfig["type"] = "libtorch";
    }

    if (genieEngineConfig["backend"].contains("device")) {
      quallaEngineConfig["training_config"]["device"] = genieEngineConfig["backend"]["device"];
    }

    if (genieEngineConfig["model"]["type"] == "checkpoint") {
      if (genieEngineConfig["model"]["checkpoint"].contains("model-basedir")) {
        quallaEngineConfig["training_config"]["model_basedir"] =
            genieEngineConfig["model"]["checkpoint"]["model-basedir"];
      }
      if (genieEngineConfig["model"]["checkpoint"].contains("lora")) {
        translateLoraConfig(genieEngineConfig["model"]["checkpoint"]["lora"],
                            quallaEngineConfig["loraConfig"]);
      }
      if (genieEngineConfig["model"]["checkpoint"].contains("output-dir")) {
        quallaEngineConfig["training_config"]["output_dir"] =
            genieEngineConfig["model"]["checkpoint"]["output-dir"];
      }
      if (genieEngineConfig["model"]["checkpoint"].contains("cache-dir")) {
        quallaEngineConfig["training_config"]["cache_dir"] =
            genieEngineConfig["model"]["checkpoint"]["cache-dir"];
      }
      if (genieEngineConfig["model"]["checkpoint"].contains("model-json-path")) {
        quallaEngineConfig["training_config"]["model_json_path"] =
            genieEngineConfig["model"]["checkpoint"]["model-json-path"];
      }
      if (genieEngineConfig["model"]["checkpoint"].contains("clipping-map-path")) {
        quallaEngineConfig["training_config"]["clipping_map_path"] =
            genieEngineConfig["model"]["checkpoint"]["clipping-map-path"];
      }
      if (genieEngineConfig["model"]["checkpoint"].contains("lora")) {
        quallaEngineConfig["training_config"]["clipping_map_path"] =
            genieEngineConfig["model"]["checkpoint"]["clipping-map-path"];
      }
      if (genieEngineConfig["model"]["checkpoint"].contains("trainable-params-path")) {
        quallaEngineConfig["training_config"]["trainable_params_path"] =
            genieEngineConfig["model"]["checkpoint"]["trainable-params-path"];
      }
      if (genieEngineConfig["model"]["checkpoint"].contains("data-type")) {
        quallaEngineConfig["training_config"]["dtype"] =
            genieEngineConfig["model"]["checkpoint"]["data-type"];
      }
    }

    // Translate training-config
    if (genieEngineConfig.contains("training-config")) {
      const auto& tc = genieEngineConfig["training-config"];

      if (tc.contains("random-seed")) {
        quallaEngineConfig["training_config"]["random_seed"] = tc["random-seed"];
      }
      if (tc.contains("torch-num-threads")) {
        quallaEngineConfig["training_config"]["torch_num_threads"] = tc["torch-num-threads"];
      }
      if (tc.contains("torch-num-inter-op-threads")) {
        quallaEngineConfig["training_config"]["torch_num_inter_op_threads"] =
            tc["torch-num-inter-op-threads"];
      }
      if (tc.contains("max-iterations")) {
        quallaEngineConfig["training_config"]["max_iterations"] = tc["max-iterations"];
      }
      if (tc.contains("batch-size")) {
        quallaEngineConfig["training_config"]["batch_size"] = tc["batch-size"];
      }
      if (tc.contains("softmax-topk")) {
        quallaEngineConfig["training_config"]["softmax_topk"] = tc["softmax-topk"];
      }
      if (tc.contains("gclip-max-norm")) {
        quallaEngineConfig["training_config"]["gclip_max_norm"] = tc["gclip-max-norm"];
      }
      if (tc.contains("enable-logit-masking")) {
        quallaEngineConfig["training_config"]["enable_logit_masking"] = tc["enable-logit-masking"];
      }
      if (tc.contains("loss-type")) {
        quallaEngineConfig["training_config"]["loss_type"] = tc["loss-type"];
      }
      if (tc.contains("grad-scaler")) {
        quallaEngineConfig["training_config"]["grad_scaler"] = tc["grad-scaler"];
      }
      if (tc.contains("target-tensor-name")) {
        quallaEngineConfig["training_config"]["target_tensor_name"] = tc["target-tensor-name"];
      }
    }

    // Translate optimizer-config
    if (genieEngineConfig.contains("optimizer-config")) {
      const auto& oc = genieEngineConfig["optimizer-config"];

      if (oc.contains("opt-name")) {
        quallaEngineConfig["optimizer_config"]["opt_name"] = oc["opt-name"];
      }
      if (oc.contains("learning-rate")) {
        quallaEngineConfig["optimizer_config"]["learning_rate"] = oc["learning-rate"];
      }
      if (oc.contains("wd")) {
        quallaEngineConfig["optimizer_config"]["weight_decay"] = oc["wd"];
      } else if (oc.contains("weight-decay")) {
        quallaEngineConfig["optimizer_config"]["weight_decay"] = oc["weight-decay"];
      }
      if (oc.contains("momentum")) {
        quallaEngineConfig["optimizer_config"]["momentum"] = oc["momentum"];
      }
      if (oc.contains("beta_1")) {
        quallaEngineConfig["optimizer_config"]["beta_1"] = oc["beta_1"];
      } else if (oc.contains("beta-1")) {
        quallaEngineConfig["optimizer_config"]["beta_1"] = oc["beta-1"];
      }
      if (oc.contains("beta_2")) {
        quallaEngineConfig["optimizer_config"]["beta_2"] = oc["beta_2"];
      } else if (oc.contains("beta-2")) {
        quallaEngineConfig["optimizer_config"]["beta_2"] = oc["beta-2"];
      }
      if (oc.contains("eps")) {
        quallaEngineConfig["optimizer_config"]["eps"] = oc["eps"];
      }
    }
  }
}

void EncoderDecoder::translateEncoderDecoderConfig(const nlohmann::json& genieConfig,
                                                   nlohmann::json& quallaConfig) {
  if (genieConfig.contains("type")) {
    quallaConfig["type"] = genieConfig["type"];
    if (quallaConfig["type"] == "unet") {
      quallaConfig["engine"]["training_config"]["target_tensor_name"] = "noise_input";
    }
  }

  if (genieConfig.contains("engine")) {
    translateEngineConfig(genieConfig["engine"], quallaConfig["engine"]);
  }

  if (genieConfig.contains("scheduler")) {
    translateSchedulerConfig(genieConfig, quallaConfig);
  }
}

//=============================================================================
// EncoderDecoder functions
//=============================================================================
std::atomic<std::uint32_t> EncoderDecoder::s_nameCounter{0u};

qnn::util::HandleManager<EncoderDecoder>& EncoderDecoder::getManager() {
  static qnn::util::HandleManager<EncoderDecoder> s_manager;
  return s_manager;
}

GenieNode_Handle_t EncoderDecoder::add(std::shared_ptr<EncoderDecoder> encoderDecoder) {
  return reinterpret_cast<GenieNode_Handle_t>(getManager().add(encoderDecoder));
}

std::shared_ptr<EncoderDecoder> EncoderDecoder::get(GenieNode_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void EncoderDecoder::remove(GenieNode_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void EncoderDecoder::initEncoderDecoder(nlohmann::json& config,
                                        std::shared_ptr<ProfileStat> /*profileStat*/,
                                        std::shared_ptr<genie::log::Logger> logger,
                                        std::shared_ptr<qualla::Env> env) {
  std::shared_ptr<qualla::Env> envUsed = nullptr;
  if (!env) {
    envUsed = qualla::Env::create(std::make_shared<ResourceManager>(), nlohmann::json{});
  } else {
    envUsed = env;
  }

  if (logger) envUsed->bindLogger(logger);

  nlohmann::json quallaConfig;
  translateEncoderDecoderConfig(config["encoder-decoder"], quallaConfig);

  m_name                 = "encoder-decoder" + std::to_string(s_nameCounter.fetch_add(1u));
  m_quallaEncoderDecoder = qualla::EncoderDecoder::create(envUsed, m_name, quallaConfig);

  if (!m_quallaEncoderDecoder) {
    throw Exception(GENIE_STATUS_ERROR_MEM_ALLOC, "Could not create an encoder-decoder object");
  }

  // Get type from config
  if (config["encoder-decoder"].contains("type")) {
    m_type = config["encoder-decoder"]["type"];
  }

  // Note: KPIs tracking for EncoderDecoder will be added when available in qualla
  // qualla::EncoderDecoder::KPIs kpis = m_quallaEncoderDecoder->kpis();
  // if (profileStat)
  //   profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_EMBEDDING_CREATE, kpis);
}

EncoderDecoder::EncoderDecoder(nlohmann::json& config,
                               std::shared_ptr<ProfileStat> /*profileStat*/,
                               std::shared_ptr<genie::log::Logger> logger,
                               std::shared_ptr<qualla::Env> env) {
  initEncoderDecoder(config, nullptr, logger, env);
}

EncoderDecoder::EncoderDecoder(std::shared_ptr<Config> config,
                               std::shared_ptr<ProfileStat> /*profileStat*/,
                               std::shared_ptr<genie::log::Logger> logger) {
  initEncoderDecoder(config->getJson(), nullptr, logger);
}

std::vector<float> EncoderDecoder::train(qualla::TrainingData& training_data) {
  return m_quallaEncoderDecoder->train(training_data);
}

int32_t EncoderDecoder::saveLora(std::string loraAdapterName,
                                 std::string engineRole,
                                 std::shared_ptr<ProfileStat> /*profileStat*/) {
  std::string role = Engine::changeRole(engineRole);
  bool status      = m_quallaEncoderDecoder->saveLoraAdapter(loraAdapterName, role);
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
}

void EncoderDecoder::bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler) {
  for (auto it : profiler) {
    it->incrementUseCount();
    m_profiler.insert(it);
  }
}

void EncoderDecoder::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

void EncoderDecoder::bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& logger) {
  for (auto it : logger) {
    it->incrementUseCount();
    m_logger.insert(it);
    m_quallaEncoderDecoder->getEnv()->bindLogger(it);
  }
}

void EncoderDecoder::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& EncoderDecoder::getLogger() {
  return m_logger;
}

std::unordered_set<std::shared_ptr<Profiler>>& EncoderDecoder::getProfiler() { return m_profiler; }

std::string EncoderDecoder::getName() { return m_name; }

std::string EncoderDecoder::getType() { return m_type; }

int32_t EncoderDecoder::getDimensions(qualla::LayerType layerType,
                                      std::vector<uint32_t>& dimensions) {
  if (!m_quallaEncoderDecoder) {
    return GENIE_STATUS_ERROR_GENERAL;
  }

  try {
    // Use the new generic method with "encoder_hidden_states" as the default input tensor name
    m_quallaEncoderDecoder->get_dimensions(dimensions, layerType);
    return GENIE_STATUS_SUCCESS;
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
}

int32_t EncoderDecoder::getQuantParam(qualla::LayerType layerType,
                                      std::string& dataType,
                                      double& scale,
                                      int32_t& offset,
                                      size_t& byteWidth) {
  if (!m_quallaEncoderDecoder) {
    return GENIE_STATUS_ERROR_GENERAL;
  }

  try {
    // Use the new generic method with "encoder_hidden_states" as the default input tensor name
    m_quallaEncoderDecoder->get_tensorQuantParam(dataType, scale, offset, byteWidth, layerType);
    return GENIE_STATUS_SUCCESS;
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
}

void EncoderDecoder::setPerformancePolicy(const Genie_PerformancePolicy_t policy) {
  // TODO: Implement when qualla::EncoderDecoder supports performance policy
  // m_quallaEncoderDecoder->setPerformancePolicy(static_cast<qualla::PerformanceProfile>(policy));
  m_performancePolicy = policy;
}

const Genie_PerformancePolicy_t& EncoderDecoder::getPerformancePolicy() {
  // TODO: Implement when qualla::EncoderDecoder supports performance policy
  // m_performancePolicy =
  //     static_cast<Genie_PerformancePolicy_t>(m_quallaEncoderDecoder->getPerformancePolicy());
  return m_performancePolicy;
}
