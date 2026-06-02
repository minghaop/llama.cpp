//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <exception>
#include <set>

#include "Engine.hpp"
#include "Exception.hpp"
#include "Macro.hpp"
#include "ResourceManager.hpp"
#include "qualla/detail/utils.hpp"

using namespace genie;

inline std::string getLibName(const std::string& baseName) {
#ifdef _WIN32
  return baseName + ".dll";  // Windows shared lib
#else
  return "lib" + baseName + ".so";  // Linux shared lib
#endif  // _WIN32
}

const std::string& Engine::changeRole(const std::string& role) {
  static const std::unordered_map<std::string, std::string> s_roleMap{{"primary", "primary"},
                                                                      {"target", "primary"},
                                                                      {"secondary", "secondary"},
                                                                      {"draft", "secondary"}};

  auto it = s_roleMap.find(role);
  if (it == s_roleMap.end()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unrecognized engine role: " + role);
  }

  return it->second;
}

//=============================================================================
// Engine functions
//=============================================================================

static qnn::util::HandleManager<Engine>& getEngineManager() {
  static qnn::util::HandleManager<Engine> s_manager;
  return s_manager;
}

GenieEngine_Handle_t Engine::add(std::shared_ptr<Engine> engine) {
  return reinterpret_cast<GenieEngine_Handle_t>(getEngineManager().add(engine));
}

std::shared_ptr<Engine> Engine::get(GenieEngine_Handle_t handle) {
  return getEngineManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Engine::remove(GenieEngine_Handle_t handle) {
  getEngineManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

//=============================================================================
// Engine::EngineConfig functions
//=============================================================================

static qnn::util::HandleManager<Engine::EngineConfig>& getEngineConfigManager() {
  static qnn::util::HandleManager<Engine::EngineConfig> s_manager;
  return s_manager;
}

std::atomic<std::uint32_t> Engine::s_nameCounter{0u};

GenieEngineConfig_Handle_t Engine::EngineConfig::add(std::shared_ptr<Engine::EngineConfig> config) {
  return reinterpret_cast<GenieEngineConfig_Handle_t>(getEngineConfigManager().add(config));
}

std::shared_ptr<Engine::EngineConfig> Engine::EngineConfig::get(GenieEngineConfig_Handle_t handle) {
  return getEngineConfigManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Engine::EngineConfig::remove(GenieEngineConfig_Handle_t handle) {
  getEngineConfigManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Engine::EngineConfig::bindLogger(std::shared_ptr<genie::log::Logger> logger) {
  if (!logger) return;
  logger->incrementUseCount();
  m_logger.insert(logger);
}

void Engine::EngineConfig::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& Engine::EngineConfig::getLogger() {
  return m_logger;
}

void Engine::EngineConfig::bindProfiler(std::shared_ptr<Profiler> profiler) {
  if (!profiler) return;
  if (profiler->m_traceLogger) {
    throw Exception(
        GENIE_STATUS_ERROR_INVALID_ARGUMENT,
        "Engine config does not currently support binding profilers with tracing enabled.");
  }
  profiler->incrementUseCount();
  m_profiler.insert(profiler);
}

void Engine::EngineConfig::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& Engine::EngineConfig::getProfiler() {
  return m_profiler;
}

//=============================================================================
// Backend::Config functions
//=============================================================================

static inline bool position_dim_set = false;
static inline bool rope_theta_set   = false;

static void validateBackendHtpConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "QnnHtp config is not an object");
  }

  std::set<std::string> mandatoryFields{
      "version", "spill-fill-bufsize", "mmap-budget", "use-mmap", "cpu-mask", "poll"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing QnnHtp field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "QnnHtp";
  bool graph_switching  = false;
  bool lazy_lora        = false;
  bool weightSharedLora = false;
  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid QnnHtp config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "spill-fill-bufsize") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "mmap-budget") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "use-mmap") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "pos-id-dim") {
      position_dim_set = true;
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "shared-engine") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "cpu-mask") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "poll") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "kv-dim") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "batch-size") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "kv-update-method") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "allow-async-init") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "rope-theta") {
      rope_theta_set = true;
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "enable-graph-switching") {
      JSON_ENFORCE_BOOLEAN();
      graph_switching = item.value();
    } else if (item.key() == "graph-switching-lora-policy") {
      JSON_ENFORCE_STRING();
      if (item.value() != "lazy" && item.value() != "eager") {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid QnnHtp config. graph-switching-lora-policy option must either be "
                        "lazy or eager");
      }
      if (item.value() == "lazy") {
        lazy_lora = true;
      }
    } else if (item.key() == "weight-shared-lora") {
      JSON_ENFORCE_BOOLEAN();
      weightSharedLora = item.value();
    } else if (item.key() == "skip-lora-validation") {
      JSON_ENFORCE_BOOLEAN();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown QnnHtp config key: " + item.key());
    }
  }
  if (!graph_switching && lazy_lora) {
    throw Exception(
        GENIE_STATUS_ERROR_JSON_VALUE,
        "Invalid QnnHtp config. Lazy LoRA application policy requires graph switching enabled");
  }
  if (weightSharedLora && lazy_lora) {
    throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                    "Invalid QnnHtp config. lazyLora and weightSharedLora cannot be both enabled");
  }
}

static void validateBackendGenaiConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "QnnGenAiTransformer config is not an object");
  }

  std::set<std::string> mandatoryFields{"version"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Missing QnnGenAiTransformer field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "QnnGenAiTransformer";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(
            GENIE_STATUS_ERROR_JSON_VALUE,
            "Invalid QnnGenAiTransformer config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "use-mmap") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "kv-quantization") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "n-logits") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-layer") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-embd") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-heads") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-kv-heads") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "model-input") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "shared-engine") {
      JSON_ENFORCE_BOOLEAN();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown QnnGenAiTransformer config key: " + item.key());
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
  bool htp = false;
  nlohmann::json htpConfig;
  bool genai = false;
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
      if (type == "QnnHtp") {
        htp = true;
      } else if (type == "QnnGenAiTransformer") {
        genai = true;
      } else if (type != "QnnGpu") {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid backend config: unsupported type: " + item.value().dump());
      }
    } else if (item.key() == "extensions") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "QnnHtp") {
      JSON_ENFORCE_OBJECT();
      htpConfig = item.value();
    } else if (item.key() == "QnnGenAiTransformer") {
      JSON_ENFORCE_OBJECT();
      genaiConfig = item.value();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown backend config key: " + item.key());
    }
  }

  if (htp) {
    if (!htpConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing QnnHtp engine config");
    }
    validateBackendHtpConfig(htpConfig);
  } else {
    if (htpConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "QnnHtp backend config for incorrect backend type: " + type);
    }
  }

  if (genai) {
    if (!genaiConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing QnnGenAiTransformer engine config");
    }
    validateBackendGenaiConfig(genaiConfig);
  } else {
    if (genaiConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "QnnGenAiTransformer backend config for incorrect backend type: " + type);
    }
  }
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

static void validateModelBinaryConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "binary config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "ctx-bins"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing binary field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "binary";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid binary config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "ctx-bins") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE, "ctx-bins must be an array of strings");
        }
      }
    } else if (item.key() == "lora") {
      JSON_ENFORCE_OBJECT();
      validateLoraConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown binary config key: " + item.key());
    }
  }
}

static void validateModelLibraryConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "library config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "model-bin"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing library field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "library";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid library config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "model-bin") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "lora") {
      JSON_ENFORCE_OBJECT();
      validateLoraConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown library config key: " + item.key());
    }
  }
}

static void validateRopeScalingConfig(const nlohmann::json& config) {
  // component is used in the "ENFORCE" macros
  std::string component = "rope-scaling";
  if (config.is_object()) {
    std::string ropeType;
    for (auto& item : config.items()) {
      if (item.key() == "rope-type") {
        JSON_ENFORCE_STRING();
        ropeType = item.value().get<std::string>();
        if (ropeType != "llama3" && ropeType != "default" && ropeType != "longrope" &&
            ropeType != "qwen2vl-mrope" && ropeType != "qwen3vl-mrope" && ropeType != "yarnrope" &&
            ropeType != "huanyuan") {
          throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Rope type not supported" + ropeType);
        }
      } else if (item.key() == "factor" || item.key() == "low-freq-factor" ||
                 item.key() == "high-freq-factor" ||
                 item.key() == "original-max-position-embeddings" || item.key() == "time-step" ||
                 item.key() == "spatial-merge-size" || item.key() == "dim" ||
                 item.key() == "extrapolation-factor" || item.key() == "attn-factor" ||
                 item.key() == "beta-fast" || item.key() == "beta-slow") {
        JSON_ENFORCE_NUMERIC();
      } else if (item.key() == "short-factor" || item.key() == "long-factor" ||
                 item.key() == "mrope-section") {
        JSON_ENFORCE_ARRAY();
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                        "Rope scaling parameter not supported " + item.key());
      }
    }
  }
}

static void validatePositionalEncodingConfig(const nlohmann::json& config) {
  // component is used in the "ENFORCE" macros
  std::string component = "positional-encoding";
  nlohmann::json ropeScalingConfig;
  if (config.is_object()) {
    for (auto& item : config.items()) {
      if (item.key() == "type") {
        std::string positionEncodingType = item.value().get<std::string>();
        if (positionEncodingType != "rope" && positionEncodingType != "absolute" &&
            positionEncodingType != "alibi") {
          throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "positional-encoding type not supported");
        }
      } else if (item.key() == "rope-dim") {
        JSON_ENFORCE_NUMERIC();
      } else if (item.key() == "rope-theta") {
        JSON_ENFORCE_NUMERIC();
      } else if (item.key() == "rope-scaling") {
        JSON_ENFORCE_OBJECT();
        ropeScalingConfig = item.value();
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                        "Unknown positional encoding config key: " + item.key());
      }
    }
  }
  if (position_dim_set) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                    "Specify one config from pos-id-dim and positional-encoding");
  }
  if (rope_theta_set) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                    "Specify one config from rope-theta and positional-encoding");
  }
  if (ropeScalingConfig.is_object()) {
    validateRopeScalingConfig(ropeScalingConfig);
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
  bool binary = false;
  nlohmann::json binaryConfig;
  bool library = false;
  nlohmann::json libraryConfig;
  nlohmann::json positionalEncodingConfig;
  bool positionalEncoding = false;

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
      if (type == "binary") {
        binary = true;
      } else if (type == "library") {
        library = true;
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid model config: unsupported type: " + item.value().dump());
      }
    } else if (item.key() == "binary") {
      JSON_ENFORCE_OBJECT();
      binaryConfig = item.value();
    } else if (item.key() == "library") {
      JSON_ENFORCE_OBJECT();
      libraryConfig = item.value();
    } else if (item.key() == "positional-encoding") {
      JSON_ENFORCE_OBJECT();
      positionalEncodingConfig = item.value();
      positionalEncoding       = true;
    } else if (item.key() == "draft-token-map") {
      JSON_ENFORCE_STRING();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown model config key: " + item.key());
    }
  }

  if (binary) {
    if (!binaryConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing binary model config");
    }
    validateModelBinaryConfig(binaryConfig);
  } else {
    if (binaryConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "binary model config for incorrect model type: " + type);
    }
  }

  if (library) {
    if (!libraryConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing library model config");
    }
    validateModelLibraryConfig(libraryConfig);
  } else {
    if (libraryConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "library model config for incorrect model type: " + type);
    }
  }

  if (positionalEncoding) {
    if (!positionalEncodingConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing Positional encoding config");
    }
    validatePositionalEncodingConfig(positionalEncodingConfig);
  } else {
    if (positionalEncodingConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Positional encoding config for incorrect model type: " + type);
    }
  }
}

static void validateKeyDiffConfig(const nlohmann::json& config) {
  // component is used in the "ENFORCE" macros
  std::string component = "keydiff";

  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "keydiff config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "scoring-network", "update-frequency"};

  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing keydiff field: " + field);
    }
  }

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid keydiff config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "scoring-network") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "update-frequency") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "anchor-alpha") {
      JSON_ENFORCE_NUMERIC();
      double anchorAlpha = item.value().get<double>();
      if (anchorAlpha <= 0.0 || anchorAlpha >= 1.0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid keydiff config: anchor-alpha is not in the range (0.0, 1.0): " +
                            item.value().dump());
      }
    }
  }
}

static void validateSlidingWindowConfig(const nlohmann::json& config) {
  // component is used in the "ENFORCE" macros
  std::string component = "sliding-window";
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "sliding-window config is not an object");
  }
  std::set<std::string> mandatoryFields{"version", "window-size"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing sliding-window field: " + field);
    }
  }
  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(
            GENIE_STATUS_ERROR_JSON_VALUE,
            "Invalid sliding-window config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "window-size") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown sliding-window config key: " + item.key());
    }
  }
}

static void validateLongContextConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "longcontext config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "type"};

  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing engine field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "longcontext";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid longcontext config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
      if (item.value() != "keydiff" && item.value() != "sliding-window") {
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                        "Unknown value: for longcontext config key: " + item.key());
      }
    } else if (item.key() == "reserved-tokens") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "keydiff") {
      validateKeyDiffConfig(item.value());
    } else if (item.key() == "sliding-window") {
      validateSlidingWindowConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown longcontext config key: " + item.key());
    }
  }
}

static void validateEngineConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "engine config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "backend", "model", "n-threads"};

  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing engine field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "engine";
  // All different roles that are defined in the engine config
  std::unordered_set<std::string> roleValues{"draft", "target", "primary", "secondary"};
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
    } else if (item.key() == "model") {
      JSON_ENFORCE_OBJECT();
      validateModelConfig(item.value());
    } else if (item.key() == "n-threads") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "role") {
      JSON_ENFORCE_STRING();
      if (roleValues.count(item.value()) == 0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                        "Unknown value: for engine config key: " + item.key());
      }
    } else if (item.key() == "longcontext") {
      validateLongContextConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown engine config key: " + item.key());
    }
  }
}

static void validateContextConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "context config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "bos-token", "eos-token", "size", "n-vocab"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing context field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "context";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid context config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "bos-token") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "eos-token") {
      JSON_ENFORCE_ARRAY_OR_NUMERIC();
    } else if (item.key() == "eot-token") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "size") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-vocab") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "draft-n-vocab") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "pad-token") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-embd") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown context config key: " + item.key());
    }
  }
}

static void translateContextConfig(const nlohmann::json& genieConfig,
                                   nlohmann::json& quallaConfig) {
  if (genieConfig.contains("bos-token")) {
    quallaConfig["context"]["bos-token"] = genieConfig["bos-token"];
  }
  if (genieConfig.contains("eos-token")) {
    quallaConfig["context"]["eos-token"] = genieConfig["eos-token"];
  }
  if (genieConfig.contains("eot-token")) {
    quallaConfig["context"]["eot-token"] = genieConfig["eot-token"];
  }
  if (genieConfig.contains("size")) {
    quallaConfig["context"]["size"] = genieConfig["size"];
  }
  if (genieConfig.contains("n-vocab")) {
    quallaConfig["context"]["n-vocab"] = genieConfig["n-vocab"];
  }
  if (genieConfig.contains("draft-n-vocab")) {
    quallaConfig["context"]["draft-n-vocab"] = genieConfig["draft-n-vocab"];
  }
  if (genieConfig.contains("pad-token")) {
    quallaConfig["context"]["pad-token"] = genieConfig["pad-token"];
  }
  if (genieConfig.contains("n-embd")) {
    quallaConfig["context"]["n-embd"] = genieConfig["n-embd"];
  }
}

static void translateEmbeddingConfig(const nlohmann::json& genieConfig,
                                     nlohmann::json& quallaConfig) {
  quallaConfig["context"]["n-embd"] = genieConfig["size"];
  if (genieConfig.contains("datatype")) {
    std::string dataType = "QNN_DATATYPE_UNDEFINED";
    if (genieConfig["datatype"] == "float32") {
      dataType = "QNN_DATATYPE_FLOAT_32";
    } else if (genieConfig["datatype"] == "native") {
      dataType = "QNN_DATATYPE_UNDEFINED";
    } else if (genieConfig["datatype"] == "ufixed8") {
      dataType = "QNN_DATATYPE_UFIXED_POINT_8";
    } else if (genieConfig["datatype"] == "ufixed16") {
      dataType = "QNN_DATATYPE_UFIXED_POINT_16";
    } else if (genieConfig["datatype"] == "sfixed8") {
      dataType = "QNN_DATATYPE_SFIXED_POINT_8";
    } else if (genieConfig["datatype"] == "sfixed16") {
      dataType = "QNN_DATATYPE_SFIXED_POINT_16";
    } else if (genieConfig["datatype"] == "ufixed4") {
      dataType = "QNN_DATATYPE_UFIXED_POINT_4";
    }
    quallaConfig["context"]["embedding-datatype"] = dataType;
  }
  if (genieConfig.contains("quant-param")) {
    quallaConfig["context"]["quant-param"]["scale"]  = genieConfig["quant-param"]["scale"];
    quallaConfig["context"]["quant-param"]["offset"] = genieConfig["quant-param"]["offset"];
  }
}

//=============================================================================
// Embedding::Config functions
//=============================================================================

static void validateEmbeddingConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "embedding config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "size"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing embedding field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "embedding";
  bool lutPathSet       = false;
  bool isTypeLut        = false;
  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid embedding config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "size") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
      const std::set<std::string> supportedTypes = {"lut", "callback"};
      if (std::find(supportedTypes.begin(), supportedTypes.end(), std::string(item.value())) ==
          supportedTypes.end()) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Unknown embedding type: " + std::string(item.value()));
      }
      if (item.value() == "lut") {
        isTypeLut = true;
      }
    } else if (item.key() == "datatype") {
      JSON_ENFORCE_STRING();
      const std::set<std::string> supportedTypes = {
          "float32", "native", "ufixed4", "ufixed8", "ufixed16", "sfixed8", "sfixed16"};
      if (std::find(supportedTypes.begin(), supportedTypes.end(), std::string(item.value())) ==
          supportedTypes.end()) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Unknown embedding datatype: " + std::string(item.value()));
      }
    } else if (item.key() == "lut-path") {
      JSON_ENFORCE_STRING();
      lutPathSet = true;
    } else if (item.key() == "quant-param") {
      JSON_ENFORCE_OBJECT();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown embedding config key: " + item.key());
    }
  }
  if (isTypeLut ^ lutPathSet) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                    "lut-path config option should be used with type lut");
  }
}

static void translateLongContextConfig(const nlohmann::json& genieLongContextConfig,
                                       nlohmann::json& quallaLongContextConfig,
                                       const size_t reservedTokens) {
  quallaLongContextConfig["type"]            = genieLongContextConfig["type"];
  quallaLongContextConfig["reserved-tokens"] = reservedTokens;
  if (genieLongContextConfig.contains("reserved-tokens")) {
    size_t sinkTokens = genieLongContextConfig["reserved-tokens"];
    sinkTokens += reservedTokens;
    quallaLongContextConfig["reserved-tokens"] = sinkTokens;
  }
  if (genieLongContextConfig.contains("sliding-window")) {
    const nlohmann::json& genieSlidingWindowConfig = genieLongContextConfig["sliding-window"];
    quallaLongContextConfig["window-size"]         = genieSlidingWindowConfig["window-size"];
  }
  if (genieLongContextConfig.contains("keydiff")) {
    const nlohmann::json& genieKeyDiffConfig    = genieLongContextConfig["keydiff"];
    quallaLongContextConfig["update-frequency"] = genieKeyDiffConfig["update-frequency"];
    quallaLongContextConfig["scoring-network"]  = genieKeyDiffConfig["scoring-network"];
    if (genieKeyDiffConfig.contains("anchor-alpha")) {
      quallaLongContextConfig["anchor-alpha"] = genieKeyDiffConfig["anchor-alpha"];
    }
  }
}

static void translateLoraConfig(const nlohmann::json& genieLoraConfig,
                                nlohmann::json& quallaLoraConfig) {
  if (genieLoraConfig.contains("role"))
    quallaLoraConfig["role"] = Engine::changeRole(genieLoraConfig["role"]);
  quallaLoraConfig["lora-version"] = static_cast<uint8_t>(LORA_VERSION::GENIE_LORA_VERSION_V2);
  if (genieLoraConfig.contains("lora-version") && genieLoraConfig["lora-version"] == 1) {
    quallaLoraConfig["lora-version"] = genieLoraConfig["lora-version"];
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
                                  nlohmann::json& quallaEngineConfig,
                                  const size_t reservedTokens) {
  if (genieEngineConfig["version"] == 1) {
    if (genieEngineConfig.contains("role")) {
      quallaEngineConfig["role"] = Engine::changeRole(genieEngineConfig["role"]);
    } else {
      quallaEngineConfig["role"] = Engine::changeRole("primary");
    }

    quallaEngineConfig["n-threads"] = genieEngineConfig["n-threads"];

    if (genieEngineConfig["backend"]["type"] == "QnnHtp") {
      quallaEngineConfig["type"]          = "qnn-htp";
      quallaEngineConfig["backend-lib"]   = getLibName("QnnHtp");
      quallaEngineConfig["mmap-budget"]   = genieEngineConfig["backend"]["QnnHtp"]["mmap-budget"];
      quallaEngineConfig["use-mmap"]      = genieEngineConfig["backend"]["QnnHtp"]["use-mmap"];
      quallaEngineConfig["shared-engine"] = false;
      if (genieEngineConfig["backend"]["QnnHtp"].contains("shared-engine")) {
        quallaEngineConfig["shared-engine"] =
            genieEngineConfig["backend"]["QnnHtp"]["shared-engine"];
      }
      quallaEngineConfig["spill-fill-bufsize"] =
          genieEngineConfig["backend"]["QnnHtp"]["spill-fill-bufsize"];
      if (genieEngineConfig["backend"]["QnnHtp"].contains("pos-id-dim")) {
        quallaEngineConfig["pos-id-dim"] = genieEngineConfig["backend"]["QnnHtp"]["pos-id-dim"];
      }
      quallaEngineConfig["cpumask"] = genieEngineConfig["backend"]["QnnHtp"]["cpu-mask"];
      quallaEngineConfig["poll"]    = genieEngineConfig["backend"]["QnnHtp"]["poll"];
      quallaEngineConfig["kv-dim"]  = genieEngineConfig["backend"]["QnnHtp"]["kv-dim"];
      if (genieEngineConfig["backend"]["QnnHtp"].contains("batch-size")) {
        quallaEngineConfig["batch-size"] = genieEngineConfig["backend"]["QnnHtp"]["batch-size"];
      }
      if (genieEngineConfig["backend"]["QnnHtp"].contains("rope-theta")) {
        quallaEngineConfig["rope-theta"] = genieEngineConfig["backend"]["QnnHtp"]["rope-theta"];
      }
      if (genieEngineConfig["backend"]["QnnHtp"].contains("kv-update-method")) {
        quallaEngineConfig["kv-update-method"] =
            genieEngineConfig["backend"]["QnnHtp"]["kv-update-method"];
      }
      if (genieEngineConfig["backend"]["QnnHtp"].contains("skip-lora-validation")) {
        quallaEngineConfig["skip-lora-validation"] =
            genieEngineConfig["backend"]["QnnHtp"]["skip-lora-validation"];
      }
      // By default, Qualla will default to the async init path.
      // For now, we are forcing async init off unless explicitly
      // specified in the Genie config. It is HTP specific feature only.
      quallaEngineConfig["use-async-Init"] = false;
      if (genieEngineConfig["backend"]["QnnHtp"].contains("allow-async-init")) {
        quallaEngineConfig["use-async-Init"] =
            genieEngineConfig["backend"]["QnnHtp"]["allow-async-init"];
      }
      if (genieEngineConfig["backend"]["QnnHtp"].contains("enable-graph-switching")) {
        quallaEngineConfig["enable-graph-switching"] =
            genieEngineConfig["backend"]["QnnHtp"]["enable-graph-switching"];
      }
      if (genieEngineConfig["backend"]["QnnHtp"].contains("graph-switching-lora-policy")) {
        quallaEngineConfig["graph-switching-lora-policy"] =
            genieEngineConfig["backend"]["QnnHtp"]["graph-switching-lora-policy"];
      }
      if (genieEngineConfig["backend"]["QnnHtp"].contains("weight-shared-lora")) {
        quallaEngineConfig["weight-shared-lora"] =
            genieEngineConfig["backend"]["QnnHtp"]["weight-shared-lora"];
      }
    } else if (genieEngineConfig["backend"]["type"] == "QnnGenAiTransformer") {
      quallaEngineConfig["type"]          = "qnn-cpu";
      quallaEngineConfig["backend-lib"]   = getLibName("QnnGenAiTransformer");
      quallaEngineConfig["shared-engine"] = false;
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("n-logits")) {
        quallaEngineConfig["n_logits"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["n-logits"];
      }
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("shared-engine")) {
        quallaEngineConfig["shared-engine"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["shared-engine"];
      }
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("use-mmap")) {
        quallaEngineConfig["use-mmap"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["use-mmap"];
      }
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("kv-quantization")) {
        quallaEngineConfig["kv-quantization"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["kv-quantization"];
      }
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("n-layer")) {
        quallaEngineConfig["n_layer"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["n-layer"];
      }
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("n-embd")) {
        quallaEngineConfig["n_embd"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["n-embd"];
      }
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("n-heads")) {
        quallaEngineConfig["n_heads"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["n-heads"];
        quallaEngineConfig["n_kv_heads"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["n-heads"];
      }
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("n-kv-heads")) {
        quallaEngineConfig["n_kv_heads"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["n-kv-heads"];
      }
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("model-input")) {
        quallaEngineConfig["model-input"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["model-input"];
      }
    } else if (genieEngineConfig["backend"]["type"] == "QnnGpu") {
      quallaEngineConfig["type"] = "qnn-gpu";
    }

    if (genieEngineConfig["backend"].contains("extensions")) {
      quallaEngineConfig["backend-ext-conf"] = genieEngineConfig["backend"]["extensions"];
    }

    if (genieEngineConfig["model"]["type"] == "binary") {
      nlohmann::json trimmedCtxBins = nlohmann::json::array();
      for (const auto& binPath : genieEngineConfig["model"]["binary"]["ctx-bins"]) {
        trimmedCtxBins.push_back(qualla::utils::trim(binPath));
      }
      quallaEngineConfig["model-list"] = trimmedCtxBins;
      if (genieEngineConfig["model"]["binary"].contains("lora")) {
        quallaEngineConfig["loraConfig"] = nlohmann::json();
        translateLoraConfig(genieEngineConfig["model"]["binary"]["lora"],
                            quallaEngineConfig["loraConfig"]);
      }
    } else if (genieEngineConfig["model"]["type"] == "library") {
      quallaEngineConfig["model"]          = getLibName("QnnGenAiTransformerModel");
      quallaEngineConfig["model-bin-path"] = genieEngineConfig["model"]["library"]["model-bin"];
      if (genieEngineConfig["model"]["library"].contains("lora")) {
        quallaEngineConfig["loraConfig"] = nlohmann::json();
        translateLoraConfig(genieEngineConfig["model"]["library"]["lora"],
                            quallaEngineConfig["loraConfig"]);
      }
    }
    if (genieEngineConfig["model"].contains("positional-encoding")) {
      quallaEngineConfig["positional-encoding"]["type"] =
          genieEngineConfig["model"]["positional-encoding"]["type"];
      if (genieEngineConfig["model"]["positional-encoding"]["type"] == "rope") {
        quallaEngineConfig["positional-encoding"]["rope-dim"] =
            genieEngineConfig["model"]["positional-encoding"]["rope-dim"];
        if (genieEngineConfig["model"]["positional-encoding"].contains("rope-theta")) {
          quallaEngineConfig["positional-encoding"]["rope-theta"] =
              genieEngineConfig["model"]["positional-encoding"]["rope-theta"];
        }
        if (genieEngineConfig["model"]["positional-encoding"].contains("rope-scaling")) {
          if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                  "rope-type")) {
            quallaEngineConfig["positional-encoding"]["rope-scaling"]["rope-type"] =
                genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["rope-type"];
            if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["rope-type"] ==
                "llama3") {
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "factor")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["factor"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["factor"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "low-freq-factor")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["low-freq-factor"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]
                                     ["low-freq-factor"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "high-freq-factor")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["high-freq-factor"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]
                                     ["high-freq-factor"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "original-max-position-embeddings")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]
                                  ["original-max-position-embeddings"] =
                                      genieEngineConfig["model"]["positional-encoding"]
                                                       ["rope-scaling"]
                                                       ["original-max-position-embeddings"];
              }
            }
            if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["rope-type"] ==
                "longrope") {
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "factor")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["factor"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["factor"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "short-factor")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["short-factor"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]
                                     ["short-factor"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "long-factor")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["long-factor"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]
                                     ["long-factor"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "original-max-position-embeddings")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]
                                  ["original-max-position-embeddings"] =
                                      genieEngineConfig["model"]["positional-encoding"]
                                                       ["rope-scaling"]
                                                       ["original-max-position-embeddings"];
              }
            }
            if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["rope-type"] ==
                    "qwen2vl-mrope" ||
                genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["rope-type"] ==
                    "qwen3vl-mrope") {
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "time-step")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["time-step"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["time-step"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "spatial-merge-size")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["spatial-merge-size"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]
                                     ["spatial-merge-size"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "mrope-section")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["mrope-section"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]
                                     ["mrope-section"];
              }
            }
            if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["rope-type"] ==
                "yarnrope") {
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "factor")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["factor"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["factor"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "dim")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["dim"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["dim"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "extrapolation-factor")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["extrapolation-factor"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]
                                     ["extrapolation-factor"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "attn-factor")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["attn-factor"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]
                                     ["attn-factor"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "beta-fast")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["beta-fast"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["beta-fast"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "beta-slow")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["beta-slow"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["beta-slow"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "original-max-position-embeddings")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]
                                  ["original-max-position-embeddings"] =
                                      genieEngineConfig["model"]["positional-encoding"]
                                                       ["rope-scaling"]
                                                       ["original-max-position-embeddings"];
              }
            }
          }
        }
      }
    }
    if (genieEngineConfig["model"].contains("draft-token-map")) {
      quallaEngineConfig["draft-token-map"] = genieEngineConfig["model"]["draft-token-map"];
    }
    if (genieEngineConfig.contains("longcontext")) {
      const nlohmann::json& genieLongContextConfig = genieEngineConfig["longcontext"];
      nlohmann::json quallaLongContextConfig;
      translateLongContextConfig(genieLongContextConfig, quallaLongContextConfig, reservedTokens);
      quallaEngineConfig["longcontext"] = quallaLongContextConfig;
    }
    if (genieEngineConfig.contains("cache-groups")) {
      quallaEngineConfig["cache-groups"] = genieEngineConfig["cache-groups"];
      for (auto& item : quallaEngineConfig["cache-groups"]) {
        if (item.contains("longcontext")) {
          const nlohmann::json& genieLongContextConfig = item["longcontext"];
          nlohmann::json quallaLongContextConfig;
          translateLongContextConfig(
              genieLongContextConfig, quallaLongContextConfig, reservedTokens);
          item["longcontext"] = quallaLongContextConfig;
        }
      }
    }
  }
}

static void translateStandaloneEngineConfigs(const nlohmann::json& genieConfig,
                                             nlohmann::json& quallaContextConfig,
                                             nlohmann::json& quallaEngineConfig) {
  size_t ssdPrefixLength{0};
  translateContextConfig(genieConfig["standalone-engine"]["context"], quallaContextConfig);
  translateEngineConfig(
      genieConfig["standalone-engine"]["engine"], quallaEngineConfig, ssdPrefixLength);
  if (genieConfig["standalone-engine"].contains("buffer-type")) {
    quallaEngineConfig["buffer-type"] = genieConfig["standalone-engine"]["buffer-type"];
  }
  if (genieConfig["standalone-engine"].contains("embedding")) {
    translateEmbeddingConfig(genieConfig["standalone-engine"]["embedding"], quallaContextConfig);
  }
}
void Engine::validateStandaloneEngineConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "standalone-engine config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "context", "engine"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing standalone-engine field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "standalone-engine";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(
            GENIE_STATUS_ERROR_JSON_VALUE,
            "Invalid standalone-engine config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "engine") {
      JSON_ENFORCE_OBJECT();
      validateEngineConfig(item.value());
    } else if (item.key() == "context") {
      JSON_ENFORCE_OBJECT();
      validateContextConfig(item.value());
    } else if (item.key() == "embedding") {
      JSON_ENFORCE_OBJECT();
      validateEmbeddingConfig(item.value());
    } else if (item.key() == "execution-strategy") {
      /* Skip Validation,
         Executor performs validation */
    } else if (item.key() == "buffer-type") {
      JSON_ENFORCE_STRING();
      static const std::unordered_set<std::string> s_bufferTypes{"persistent", "regular"};
      if (!s_bufferTypes.contains(item.value()))
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                        "Unsupported buffer type: " + item.value().dump());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown standalone-engine config key: " + item.key());
    }
  }
}
nlohmann::json& Engine::EngineConfig::getJson() { return m_config; }

Engine::EngineConfig::EngineConfig(const char* configStr) {
  rope_theta_set   = false;
  position_dim_set = false;
  nlohmann::json config;
  {
    std::set<nlohmann::json> keys;

    auto callback = [&keys](
                        int depth, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
      if ((depth == 1) && (event == nlohmann::json::parse_event_t::key)) {
        if (keys.count(parsed) > 0) {
          throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                          "Multiple standalone-engine config key: " + parsed.dump());
        }
        keys.insert(parsed);
      }
      return true;
    };

    config = nlohmann::json::parse(configStr, callback);
  }

  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "standalone-engine config is not an object");
  }
  std::set<std::string> mandatoryFields{"standalone-engine"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing standalone-engine field: " + field);
    }
  }
  // component is used in the "ENFORCE" macros
  std::string component = "standalone-engine";
  for (auto& item : config.items()) {
    if (item.key() == "standalone-engine") {
      JSON_ENFORCE_OBJECT();
      Engine::validateStandaloneEngineConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown standalone-engine config key: " + item.key());
    }
  }

  m_config = config;
}

void Engine::bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& logger) {
  for (auto it : logger) {
    it->incrementUseCount();
    m_logger.insert(it);
    m_env->bindLogger(it);
  }
}

void Engine::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& Engine::getLogger() { return m_logger; }

void Engine::bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler) {
  for (auto it : profiler) {
    if (it->m_traceLogger) {
      throw Exception(GENIE_STATUS_ERROR_INVALID_ARGUMENT,
                      "Engine does not currently support binding profilers with tracing enabled.");
    }
    it->incrementUseCount();
    m_profiler.insert(it);
  }
}

void Engine::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& Engine::getProfiler() { return m_profiler; }

std::string Engine::getName() { return m_name; }

Engine::Engine(const nlohmann::json& config,
               std::shared_ptr<ProfileStat> profileStat,
               std::shared_ptr<genie::log::Logger> logger,
               std::shared_ptr<qualla::Env> env) {
  nlohmann::json quallaEngineConfig;
  nlohmann::json quallaContextConfig;
  translateStandaloneEngineConfigs(config, quallaContextConfig, quallaEngineConfig);
  m_name = "engine" + std::to_string(s_nameCounter.fetch_add(1u));
  if (!env) {
    m_env = qualla::Env::create(std::make_shared<ResourceManager>(), nlohmann::json{});
  } else {
    m_env = env;
  }

  if (logger) m_env->bindLogger(logger);
  m_context      = qualla::Context::create(m_env, m_name, quallaContextConfig["context"]);
  m_quallaEngine = qualla::Engine::create(*m_context, quallaEngineConfig);
  if (!m_quallaEngine) {
    throw Exception(GENIE_STATUS_ERROR_MEM_ALLOC, "Could not create a engine object");
  }
  qualla::Engine::KPIs kpis = m_quallaEngine->kpis();
  if (profileStat) profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_ENGINE_CREATE, kpis);
}

Engine::Engine(std::shared_ptr<EngineConfig>& config,
               std::shared_ptr<ProfileStat> profileStat,
               std::shared_ptr<genie::log::Logger> logger,
               std::shared_ptr<qualla::Env> env)
    : Engine(config->getJson(), profileStat, logger, env) {}

Engine::Engine(std::shared_ptr<qualla::Engine> quallaEngine, const std::string& name)
    : m_name(name), m_quallaEngine(quallaEngine) {}

// These all are just pass through calls to lower engine object.

size_t Engine::setInputBufferData(qualla::LayerType inputLayer,
                                  const void* data,
                                  size_t dataSize,
                                  nlohmann::json& dataConfig) {
  return m_quallaEngine->setInputBufferData(
      inputLayer, const_cast<void*>(data), dataSize, dataConfig);
}

size_t Engine::process(const nlohmann::json& config) {
  // This only allows preset input execution
  return m_quallaEngine->processPresetInput(config);
}

bool Engine::updateKV(uint32_t nPast) { return m_quallaEngine->updateKV(nPast); }

size_t Engine::restore(const std::string& name) { return m_quallaEngine->restore(name); }

size_t Engine::save(const std::string& name) { return m_quallaEngine->save(name); }

bool Engine::getOutputBufferData(qualla::LayerType outputLayer,
                                 const nlohmann::json& outputConfig,
                                 qualla::Engine::OutputCallback outputCallback) {
  return m_quallaEngine->getOutputBufferData(outputLayer, outputConfig, outputCallback);
}

void Engine::reset() { m_quallaEngine->reset(); }
Engine::~Engine() {}

bool Engine::checkIsEngineBound() {
  if (m_quallaEngine->isBound()) {
    std::cout << "Engine is Bounded." << std::endl;
    return true;
  }
  return false;
}
