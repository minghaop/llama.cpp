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

#include "Embedding.hpp"
#include "Exception.hpp"
#include "Macro.hpp"
#include "nlohmann/json.hpp"
#include "qualla/detail/utils.hpp"

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
// Context::Config functions
//=============================================================================

static void validateContextConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "context config is not an object");
  }

  std::set<std::string> mandatoryFields{
      "version", "n-vocab", "ctx-size", "embed-size", "pad-token"};
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
    } else if (item.key() == "n-vocab") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "ctx-size") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "embed-size") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "pad-token") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown context config key: " + item.key());
    }
  }
}

static void translateContextConfig(const nlohmann::json& genieConfig,
                                   nlohmann::json& quallaConfig) {
  quallaConfig["n-vocab"]   = genieConfig["n-vocab"];
  quallaConfig["size"]      = genieConfig["ctx-size"];
  quallaConfig["n-embd"]    = genieConfig["embed-size"];
  quallaConfig["pad-token"] = genieConfig["pad-token"];
}

//=============================================================================
// Tokenizer::Config functions
//=============================================================================

static void validateTokenizerConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "tokenizer config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "path"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing tokenizer field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "tokenizer";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid tokenizer config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "path") {
      JSON_ENFORCE_STRING();
      // Note: the existence of this file is checked by qualla
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown tokenizer config key: " + item.key());
    }
  }
}

static void translateTokenizerConfig(const nlohmann::json& genieConfig,
                                     nlohmann::json& quallaConfig) {
  quallaConfig["tokenizer"] = genieConfig["path"];
}

//=============================================================================
// Backend::Config functions
//=============================================================================

static void validateBackendHtpConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "QnnHtp config is not an object");
  }

  std::set<std::string> mandatoryFields{
      "version", "spill-fill-bufsize", "use-mmap", "pooled-output", "allow-async-init"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing QnnHtp field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "QnnHtp";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid QnnHtp config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "spill-fill-bufsize") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "use-mmap") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "data-alignment-size") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "pooled-output") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "allow-async-init") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "disable-kv-cache") {
      JSON_ENFORCE_BOOLEAN();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown QnnHtp config key: " + item.key());
    }
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
    } else if (item.key() == "n-logits") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-layer") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-embd") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-heads") {
      JSON_ENFORCE_NUMERIC();
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
      } else if (type == "libtorch") {
        // Placeholder to validate libtorch parameters
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid backend config: unsupported type: " + item.value().dump());
      }
    } else if (item.key() == "extensions") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "device") {
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
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing QnnHtp embedding config");
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
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Missing QnnGenAiTransformer embedding config");
    }
    validateBackendGenaiConfig(genaiConfig);
  } else {
    if (genaiConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "QnnGenAiTransformer backend config for incorrect backend type: " + type);
    }
  }
}

//=============================================================================
// Model::Config functions
//=============================================================================

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
    } else if (item.key() == "data-type") {
      JSON_ENFORCE_STRING();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown checkpoint config key: " + item.key());
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

static inline bool visionParam = false;

static void validateRopeScalingConfig(const nlohmann::json& config) {
  // component is used in the "ENFORCE" macros
  std::string component = "rope-scaling";
  if (config.is_object()) {
    std::string ropeType;
    for (auto& item : config.items()) {
      if (item.key() == "rope-type") {
        JSON_ENFORCE_STRING();
        ropeType = item.value().get<std::string>();
        if (ropeType != "qwen2vl") {
          throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Rope type not supported" + ropeType);
        }
        break;
      }
    }
    if (ropeType.empty()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Missing parameter rope-type in Rope scaling");
    }
    for (auto& item : config.items()) {
      if (item.key() == "rope-type") {
        continue;
      }
      if (ropeType == "qwen2vl") {
        if (item.key() == "height" || item.key() == "width") {
          if (visionParam) {
            throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                            "height/width already be set from vision-param.");
          }
          JSON_ENFORCE_NUMERIC();
        } else if (item.key() == "spatial-merge-size" || item.key() == "patch-size" ||
                   item.key() == "window-size") {
          JSON_ENFORCE_NUMERIC();
        } else {
          throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                          "Rope scaling parameter not supported with qwen2vl: " + item.key());
        }
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
        if (positionEncodingType != "rope" && positionEncodingType != "none") {
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
  if (ropeScalingConfig.is_object()) {
    validateRopeScalingConfig(ropeScalingConfig);
  }
}

static void validateVisionParamConfig(const nlohmann::json& config) {
  // component is used in the "ENFORCE" macros
  std::string component = "vision-param";
  if (config.is_object()) {
    for (auto& item : config.items()) {
      if (item.key() == "height") {
        JSON_ENFORCE_NUMERIC();
      } else if (item.key() == "width") {
        JSON_ENFORCE_NUMERIC();
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                        "Unknown vision-param config key: " + item.key());
      }
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
  bool binary = false;
  nlohmann::json binaryConfig;
  bool library = false;
  nlohmann::json libraryConfig;
  bool checkpoint = false;
  nlohmann::json checkpointConfig;
  nlohmann::json positionalEncodingConfig;
  bool positionalEncoding = false;
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
      if (type == "binary") {
        binary = true;
      } else if (type == "library") {
        library = true;
      } else if (type == "checkpoint") {
        checkpoint = true;
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid model config: unsupported type: " + item.value().dump());
      }
    } else if (item.key() == "mask-type") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "cross-attention") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "binary") {
      JSON_ENFORCE_OBJECT();
      binaryConfig = item.value();
    } else if (item.key() == "checkpoint") {
      JSON_ENFORCE_OBJECT();
      checkpointConfig = item.value();
    } else if (item.key() == "library") {
      JSON_ENFORCE_OBJECT();
      libraryConfig = item.value();
    } else if (item.key() == "positional-encoding") {
      JSON_ENFORCE_OBJECT();
      positionalEncodingConfig = item.value();
      positionalEncoding       = true;
    } else if (item.key() == "vision-param") {
      JSON_ENFORCE_OBJECT();
      visionParamConfig = item.value();
      visionParam       = true;
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

  if (checkpoint) {
    if (!checkpointConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing checkpoint model config");
    }
    validateModelCheckpointConfig(checkpointConfig);
  } else {
    if (checkpointConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "checkpoint model config for incorrect model type: " + type);
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

  if (visionParam) {
    if (!visionParamConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing vision param config");
    }
    validateVisionParamConfig(visionParamConfig);
  } else {
    if (visionParamConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Vision param config for incorrect model type: " + type);
    }
  }
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
    } else if (item.key() == "n-threads") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "dlc-path") {
      JSON_ENFORCE_STRING();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown engine config key: " + item.key());
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
                                  nlohmann::json& quallaEngineConfig) {
  if (genieEngineConfig["version"] == 1) {
    if (genieEngineConfig.contains("dlc-path")) {
      quallaEngineConfig["dlc-path"] = genieEngineConfig["dlc-path"];
    }
    if (genieEngineConfig.contains("n-threads"))
      quallaEngineConfig["n-threads"] = genieEngineConfig["n-threads"];

    if (genieEngineConfig["backend"]["type"] == "QnnHtp") {
      quallaEngineConfig["type"]                    = "qnn-htp";
      quallaEngineConfig["model-architecture-type"] = "encoder",
      quallaEngineConfig["backend-lib"]             = getLibName("QnnHtp");
      quallaEngineConfig["use-mmap"] = genieEngineConfig["backend"]["QnnHtp"]["use-mmap"];
      if (genieEngineConfig["backend"]["QnnHtp"].contains("data-alignment-size")) {
        quallaEngineConfig["data-alignment-size"] =
            genieEngineConfig["backend"]["QnnHtp"]["data-alignment-size"];
      }
      quallaEngineConfig["spill-fill-bufsize"] =
          genieEngineConfig["backend"]["QnnHtp"]["spill-fill-bufsize"];
      quallaEngineConfig["pooled-output"] = genieEngineConfig["backend"]["QnnHtp"]["pooled-output"];
      if (genieEngineConfig["backend"]["QnnHtp"].contains("disable-kv-cache")) {
        quallaEngineConfig["disable-kv-cache"] =
            genieEngineConfig["backend"]["QnnHtp"]["disable-kv-cache"];
      }
      if (genieEngineConfig.contains("mode")) {
        quallaEngineConfig["model-type"] = genieEngineConfig["mode"];
      }
      // By default, Qualla will default to the async init path.
      // For now, we are forcing async init off unless explicitly
      // specified in the Genie config. It is HTP specific feature only.
      quallaEngineConfig["use-async-Init"] = false;
      if (genieEngineConfig["backend"]["QnnHtp"].contains("allow-async-init")) {
        quallaEngineConfig["use-async-Init"] =
            genieEngineConfig["backend"]["QnnHtp"]["allow-async-init"];
      }
    } else if (genieEngineConfig["backend"]["type"] == "QnnGenAiTransformer") {
      quallaEngineConfig["type"]         = "qnn-cpu";
      quallaEngineConfig["model-output"] = "embeddings";
      quallaEngineConfig["backend-lib"]  = getLibName("QnnGenAiTransformer");
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("n-logits")) {
        quallaEngineConfig["n_logits"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["n-logits"];
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
      }
    } else if (genieEngineConfig["backend"]["type"] == "libtorch") {
      quallaEngineConfig["type"] = "libtorch";
      if (genieEngineConfig["backend"].contains("device")) {
        quallaEngineConfig["training_config"]["device"] = genieEngineConfig["backend"]["device"];
      }
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
    } else if (genieEngineConfig["model"]["type"] == "checkpoint") {
      if (genieEngineConfig["model"]["checkpoint"].contains("model-basedir")) {
        quallaEngineConfig["training_config"]["model_basedir"] =
            genieEngineConfig["model"]["checkpoint"]["model-basedir"];
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
      if (genieEngineConfig["model"]["checkpoint"].contains("data-type")) {
        quallaEngineConfig["training_config"]["dtype"] =
            genieEngineConfig["model"]["checkpoint"]["data-type"];
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
                "qwen2vl") {
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "height")) {
                quallaEngineConfig["vision-param"]["height"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["height"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "width")) {
                quallaEngineConfig["vision-param"]["width"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["width"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "spatial-merge-size")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["spatial-merge-size"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]
                                     ["spatial-merge-size"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "patch-size")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["patch-size"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["patch-size"];
              }
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "window-size")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["window-size"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]
                                     ["window-size"];
              }
            }
          }
        }
      }
    }
    if (genieEngineConfig["model"].contains("cross-attention")) {
      quallaEngineConfig["cross-attention"] = genieEngineConfig["model"]["cross-attention"];
    }
    if (genieEngineConfig["model"].contains("mask-type")) {
      quallaEngineConfig["mask-type"] = genieEngineConfig["model"]["mask-type"];
    }
    if (genieEngineConfig["model"].contains("vision-param")) {
      if (genieEngineConfig["model"]["vision-param"].contains("height")) {
        quallaEngineConfig["vision-param"]["height"] =
            genieEngineConfig["model"]["vision-param"]["height"];
      }
      if (genieEngineConfig["model"]["vision-param"].contains("width")) {
        quallaEngineConfig["vision-param"]["width"] =
            genieEngineConfig["model"]["vision-param"]["width"];
      }
    }
  }
}

//=============================================================================
// Prompt::Config functions
//=============================================================================

static void validatePromptConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "prompt config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "prompt-template"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing prompt field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "prompt";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid context config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "prompt-template") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE, "prompt tags must be an array of strings");
        }
      }
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown context config key: " + item.key());
    }
  }
}

static void translatePromptConfig(const nlohmann::json& genieConfig, nlohmann::json& quallaConfig) {
  quallaConfig["tags"] = genieConfig["prompt-template"];
}

//=============================================================================
// Embedding::Config functions
//=============================================================================
qnn::util::HandleManager<Embedding::Config>& Embedding::Config::getManager() {
  static qnn::util::HandleManager<Embedding::Config> s_manager;
  return s_manager;
}

GenieEmbeddingConfig_Handle_t Embedding::Config::add(std::shared_ptr<Embedding::Config> config) {
  return reinterpret_cast<GenieEmbeddingConfig_Handle_t>(getManager().add(config));
}

std::shared_ptr<Embedding::Config> Embedding::Config::get(GenieEmbeddingConfig_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Embedding::Config::remove(GenieEmbeddingConfig_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

static void validateLutConfig(const nlohmann::json& config) {
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
  std::string component = "lut";
  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid embedding config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "size") {
      JSON_ENFORCE_NUMERIC();
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
    } else if (item.key() == "quant-param") {
      JSON_ENFORCE_OBJECT();
    } else if (item.key() == "dlc-path") {
      JSON_ENFORCE_STRING();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown embedding config key: " + item.key());
    }
  }
}
void Embedding::validateEmbeddingConfig(const nlohmann::json& config, bool validateTextEncoder) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Embedding config is not an object");
  }

  std::set<std::string> mandatoryFields{"version"};
  if (validateTextEncoder) {
    mandatoryFields.insert("context");
    mandatoryFields.insert("tokenizer");
    mandatoryFields.insert("engine");
  }
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing embedding field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "embedding";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid embedding config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "context") {
      JSON_ENFORCE_OBJECT();
      validateContextConfig(item.value());
    } else if (item.key() == "tokenizer") {
      JSON_ENFORCE_OBJECT();
      validateTokenizerConfig(item.value());
    } else if (item.key() == "prompt") {  // optional parameter
      JSON_ENFORCE_OBJECT();
      validatePromptConfig(item.value());
    } else if (item.key() == "truncate-input") {  // optional parameter
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "engine") {
      JSON_ENFORCE_OBJECT();
      validateEngineConfig(config["engine"]);
    } else if (item.key() == "lut") {
      JSON_ENFORCE_OBJECT();
      validateLutConfig(config["lut"]);
    } else if (item.key() == "dlc-path") {
      JSON_ENFORCE_STRING();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown embedding config key: " + item.key());
    }
  }
}

static void translateLutConfig(const nlohmann::json& lutConfig, nlohmann::json& quallaConfig) {
  quallaConfig["context"]["n-embd"] = lutConfig["size"];

  if (lutConfig.contains("datatype")) {
    std::string dataType = "QNN_DATATYPE_UNDEFINED";
    if (lutConfig["datatype"] == "float32") {
      dataType = "QNN_DATATYPE_FLOAT_32";
    } else if (lutConfig["datatype"] == "native") {
      dataType = "QNN_DATATYPE_UNDEFINED";
    } else if (lutConfig["datatype"] == "ufixed8") {
      dataType = "QNN_DATATYPE_UFIXED_POINT_8";
    } else if (lutConfig["datatype"] == "ufixed16") {
      dataType = "QNN_DATATYPE_UFIXED_POINT_16";
    } else if (lutConfig["datatype"] == "sfixed8") {
      dataType = "QNN_DATATYPE_SFIXED_POINT_8";
    } else if (lutConfig["datatype"] == "sfixed16") {
      dataType = "QNN_DATATYPE_SFIXED_POINT_16";
    } else if (lutConfig["datatype"] == "ufixed4") {
      dataType = "QNN_DATATYPE_UFIXED_POINT_4";
    }
    quallaConfig["context"]["embedding-datatype"] = dataType;
  }
  if (lutConfig.contains("quant-param")) {
    quallaConfig["context"]["quant-param"]["scale"]  = lutConfig["quant-param"]["scale"];
    quallaConfig["context"]["quant-param"]["offset"] = lutConfig["quant-param"]["offset"];
  }

  quallaConfig["lut-path"]  = lutConfig["lut-path"];
  quallaConfig["context"]   = quallaConfig["context"];
  quallaConfig["tokenizer"] = quallaConfig["tokenizer"];
}

void Embedding::translateEmbeddingConfig(const nlohmann::json& genieConfig,
                                         nlohmann::json& quallaConfig) {
  if (genieConfig.contains("context")) {
    translateContextConfig(genieConfig["context"], quallaConfig["context"]);
  }
  if (genieConfig.contains("prompt")) {
    translatePromptConfig(genieConfig["prompt"], quallaConfig["prompt"]);
  }
  if (genieConfig.contains("tokenizer")) {
    translateTokenizerConfig(genieConfig["tokenizer"], quallaConfig);
  }
  if (genieConfig.contains("engine")) {
    translateEngineConfig(genieConfig["engine"], quallaConfig["engine"]);
  }
  if (genieConfig.contains("type")) {
    quallaConfig["type"] = genieConfig["type"];
    if (genieConfig["type"] == "image-encoder") {
      quallaConfig["type"] = "ImageEncoder";
    }
  }
  if (genieConfig.contains("lut")) {
    translateLutConfig(genieConfig["lut"], quallaConfig);
  }
  if (genieConfig.contains(
          "truncate-input")) {  // to allow truncation of input incase it exceeds the context.
    quallaConfig["truncate-input"] = genieConfig["truncate-input"];
  }
}

void Embedding::verifyAndUpdateConfig(nlohmann::json& config) {
  std::set<std::string> mandatoryFields{"embedding"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing embedding field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "embedding";

  for (auto& item : config.items()) {
    if (item.key() == "embedding") {
      JSON_ENFORCE_OBJECT();
      bool validateTextEncoder = true;
      if (item.value().contains("type")) {
        if (item.value()["type"] == "image-encoder" || item.value()["type"] == "lut-encoder") {
          validateTextEncoder = false;
        }
      }
      validateEmbeddingConfig(item.value(), validateTextEncoder);
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown embedding config key: " + item.key());
    }
  }
}

void Embedding::Config::createConfigFromJson(const char* configStr) {
  nlohmann::json config;
  util::getJsonFromStr(configStr, config);
  verifyAndUpdateConfig(config);
  m_config = config;
}

void Embedding::Config::createConfigFromDlc(const char* useCaseName, const char* configStr) {
  std::shared_ptr<const char[]> metadataStr;
  uint64_t metadataBufferSize{};
  m_dlc->getGenieMetadataBuffer(metadataStr, &metadataBufferSize);
  nlohmann::json metadataJson;
  util::getJsonFromStr(metadataStr.get(), metadataJson);
  bool useCaseFound = false;
  for (auto& item : metadataJson.items()) {
    if (item.key() == "use-cases") {
      if (item.value().type() != nlohmann::json::value_t::array) {
        throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "use-cases must be an array");
      }
      for (const auto& useCase : item.value()) {
        if (!useCase.is_object()) continue;
        if (!useCase.contains("name")) {
          throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                                 "name field must be present in a genie config use-case.");
        }
        if (useCaseName == nullptr ||
            strcmp(useCase["name"].get<std::string>().c_str(), useCaseName) == 0) {
          useCaseFound = true;
          if (!useCase.contains("config")) {
            throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                                   "Missing or invalid config field in use-case");
          } else {
            if (!useCase["config"].contains("name")) {
              throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                                     "Missing name field in config of use-case");
            }
          }
          std::string configName = useCase["config"]["name"].get<std::string>();
          if (configName.empty()) {
            throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                                   "Must provide Genie config name for the use case");
          }
          std::string genieConfigRecordName = "genai.artifact." + configName;
          nlohmann::json genieConfigJson;
          util::getJsonFromDlc(m_dlc, genieConfigRecordName, genieConfigJson);
          verifyAndUpdateConfig(genieConfigJson);
          m_config                                    = genieConfigJson;
          m_config["embedding"]["engine"]["dlc-path"] = m_dlc->getPath();

          // If user passes a standalone genie config, overwrite the config in DLC with it
          if (configStr != nullptr) {
            util::overwriteConfig(m_config, configStr);
          }
          break;
        }
      }
    }
  }

  if (!useCaseFound) {
    throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                           "Use case not found: " + std::string(useCaseName));
  }
}

Embedding::Config::Config(const char* configStr) { createConfigFromJson(configStr); }

Embedding::Config::Config(std::shared_ptr<Dlc> dlc,
                          const char* useCaseName,
                          const char* configStr) {
  m_dlc = std::move(dlc);
  createConfigFromDlc(useCaseName, configStr);
}

nlohmann::json& Embedding::Config::getJson() { return m_config; }

void Embedding::Config::bindProfiler(std::shared_ptr<Profiler> profiler) {
  if (!profiler) return;
  profiler->incrementUseCount();
  m_profiler.insert(profiler);
}

void Embedding::Config::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& Embedding::Config::getProfiler() {
  return m_profiler;
}

void Embedding::Config::bindLogger(std::shared_ptr<genie::log::Logger> logger) {
  if (!logger) return;
  logger->incrementUseCount();
  m_logger.insert(logger);
}

void Embedding::Config::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& Embedding::Config::getLogger() {
  return m_logger;
}

std::shared_ptr<Dlc>& Embedding::Config::getDlc() { return m_dlc; }

//=============================================================================
// Embedding functions
//=============================================================================
std::atomic<std::uint32_t> Embedding::s_nameCounter{0u};

qnn::util::HandleManager<Embedding>& Embedding::getManager() {
  static qnn::util::HandleManager<Embedding> s_manager;
  return s_manager;
}

GenieEmbedding_Handle_t Embedding::add(std::shared_ptr<Embedding> embedding) {
  return reinterpret_cast<GenieEmbedding_Handle_t>(getManager().add(embedding));
}

std::shared_ptr<Embedding> Embedding::get(GenieEmbedding_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Embedding::remove(GenieEmbedding_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Embedding::initEmbedding(nlohmann::json& config,
                              std::shared_ptr<qualla::Env> env,
                              std::shared_ptr<ProfileStat> profileStat,
                              std::shared_ptr<genie::log::Logger> logger) {
  if (logger) env->bindLogger(logger);
  nlohmann::json quallaConfig;
  translateEmbeddingConfig(config["embedding"], quallaConfig);
  m_name            = "embedding" + std::to_string(s_nameCounter.fetch_add(1u));
  m_quallaEmbedding = qualla::Encoder::create(env, m_name, quallaConfig);
  if (!m_quallaEmbedding) {
    throw Exception(GENIE_STATUS_ERROR_MEM_ALLOC, "Could not create a embedding object");
  }

  qualla::Encoder::KPIs kpis = m_quallaEmbedding->kpis();
  if (profileStat)
    profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_EMBEDDING_CREATE, kpis);
}

Embedding::Embedding(nlohmann::json& config,
                     std::shared_ptr<ProfileStat> profileStat,
                     std::shared_ptr<genie::log::Logger> logger,
                     std::shared_ptr<qualla::Env> env) {
  if (!env) {
    auto envNew = qualla::Env::create(std::make_shared<ResourceManager>(), nlohmann::json{});
    initEmbedding(config, envNew, profileStat, logger);
  } else {
    initEmbedding(config, env, profileStat, logger);
  }
}

Embedding::Embedding(std::shared_ptr<Config> config,
                     std::shared_ptr<ProfileStat> profileStat,
                     std::shared_ptr<genie::log::Logger> logger) {
  auto env =
      qualla::Env::create(config->getDlc() != nullptr ? config->getDlc()->getResourceManager()
                                                      : std::make_shared<ResourceManager>(),
                          nlohmann::json{});
  initEmbedding(config->getJson(), env, profileStat, logger);
}

void Embedding::bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler) {
  for (auto it : profiler) {
    it->incrementUseCount();
    m_profiler.insert(it);
  }
}

void Embedding::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

void Embedding::bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& logger) {
  for (auto it : logger) {
    it->incrementUseCount();
    m_logger.insert(it);
    m_quallaEmbedding->getEnv()->bindLogger(it);
  }
}

void Embedding::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& Embedding::getLogger() { return m_logger; }

std::unordered_set<std::shared_ptr<Profiler>>& Embedding::getProfiler() { return m_profiler; }

std::string Embedding::getName() { return m_name; }

std::string Embedding::getType() { return m_type; }

int32_t Embedding::applyLora(std::string loraAdapterName,
                             std::string engineRole,
                             std::shared_ptr<ProfileStat> profileStat) {
  bool status =
      m_quallaEmbedding->applyLoraAdapter(loraAdapterName, Engine::changeRole(engineRole));
  if (status) {
    qualla::Encoder::KPIs kpis = m_quallaEmbedding->kpis();
    if (profileStat)
      profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_EMBEDDING_APPLY_LORA, kpis);
  }
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
}

int32_t Embedding::applyLoraStrength(std::string tensorName, std::string engineRole, float alpha) {
  bool status =
      m_quallaEmbedding->applyLoraStrength(tensorName, alpha, Engine::changeRole(engineRole));
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
}

bool Embedding::encode(const char* queryStr,
                       std::vector<uint8_t>& outputEmbedding,
                       std::shared_ptr<ProfileStat> profileStat) {
  std::string query(queryStr);
  std::vector<int32_t> tokenizedResult;
  bool status = m_quallaEmbedding->encode(query, outputEmbedding, tokenizedResult);
  if (status) {
    qualla::Encoder::KPIs kpis = m_quallaEmbedding->kpis();
    if (profileStat)
      profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_EMBEDDING_GENERATE, kpis);
  }
  return status;
}

int32_t Embedding::generate(const char* queryStr,
                            GenieEmbedding_GenerateCallback_t callback,
                            const void* userData,
                            std::shared_ptr<ProfileStat> profileStat) {
  std::vector<uint8_t> outputEmbedding;
  bool status = false;
  status      = encode(queryStr, outputEmbedding, profileStat);
  if (status) {
    std::vector<uint32_t> dimensions;
    m_quallaEmbedding->output_dimensions(dimensions);
    callback(dimensions.data(),
             dimensions.size(),
             reinterpret_cast<float*>(outputEmbedding.data()),
             userData);
  }
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERATE_FAILED);
}

int32_t Embedding::encode(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
                          std::vector<uint8_t>& outputEmbedding,
                          std::shared_ptr<ProfileStat> /*profileStat*/) {
  bool status = false;
  status      = m_quallaEmbedding->encode(inputs, outputEmbedding);
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERATE_FAILED);
}

int32_t Embedding::getInputNames(std::unordered_set<std::string>& inputTensorNames) {
  m_quallaEmbedding->input_names(inputTensorNames);
  return (GENIE_STATUS_SUCCESS);
}

int32_t Embedding::getOutputDimensions(std::vector<uint32_t>& dimensions) {
  m_quallaEmbedding->output_dimensions(dimensions);
  return (GENIE_STATUS_SUCCESS);
}

int32_t Embedding::getOutputQuantParam(std::string& dataType,
                                       double& scale,
                                       int32_t& offset,
                                       float& byteWidth) {
#define DEFAULT_DATATYPE  "QNN_DATATYPE_FLOAT_32"
#define DEFAULT_SCALE     1.0
#define DEFAULT_OFFSET    0
#define DEFAULT_BYTEWIDTH 4
  dataType  = DEFAULT_DATATYPE;
  scale     = DEFAULT_SCALE;
  offset    = DEFAULT_OFFSET;
  byteWidth = DEFAULT_BYTEWIDTH;
  m_quallaEmbedding->outputTensorQuantParam(dataType, scale, offset, byteWidth);
  return (GENIE_STATUS_SUCCESS);
}

int32_t Embedding::getDimensions(qualla::LayerType layerType, std::vector<uint32_t>& dimensions) {
  if (!m_quallaEmbedding) {
    return GENIE_STATUS_ERROR_GENERAL;
  }

  try {
    // Use the new generic method with "encoder_hidden_states" as the default input tensor name
    m_quallaEmbedding->get_dimensions(dimensions, layerType);
    return GENIE_STATUS_SUCCESS;
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
}

int32_t Embedding::getQuantParam(qualla::LayerType layerType,
                                 std::string& dataType,
                                 double& scale,
                                 int32_t& offset,
                                 size_t& byteWidth) {
  if (!m_quallaEmbedding) {
    return GENIE_STATUS_ERROR_GENERAL;
  }

  try {
    // Use the new generic method with "encoder_hidden_states" as the default input tensor name
    m_quallaEmbedding->get_tensorQuantParam(dataType, scale, offset, byteWidth, layerType);
    return GENIE_STATUS_SUCCESS;
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
}

void Embedding::setPerformancePolicy(const Genie_PerformancePolicy_t policy) {
  m_quallaEmbedding->setPerformancePolicy(static_cast<qualla::PerformanceProfile>(policy));
}

const Genie_PerformancePolicy_t& Embedding::getPerformancePolicy() {
  m_performancePolicy =
      static_cast<Genie_PerformancePolicy_t>(m_quallaEmbedding->getPerformancePolicy());
  return m_performancePolicy;
}

void Embedding::registerModelAdaptor(std::shared_ptr<qualla::ModelIOAdaptor> adaptor) {
  m_quallaEmbedding->registerModelAdaptor(adaptor);
}
