//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <exception>
#include <set>
#include <sstream>

#include "Dialog.hpp"
#include "Exception.hpp"
#include "Macro.hpp"
#include "Profile.hpp"
#include "Registry.hpp"
#include "qualla/detail/timer.hpp"
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
  quallaConfig["tokenizer"] = genieConfig["dialog"]["tokenizer"]["path"];
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

static void translateEmbeddingConfig(const nlohmann::json& genieConfig,
                                     nlohmann::json& quallaConfig) {
  if (genieConfig["dialog"].contains("embedding")) {
    quallaConfig["context"]["n-embd"] = genieConfig["dialog"]["embedding"]["size"];

    if (genieConfig["dialog"]["embedding"].contains("datatype")) {
      std::string dataType = "QNN_DATATYPE_UNDEFINED";
      if (genieConfig["dialog"]["embedding"]["datatype"] == "float32") {
        dataType = "QNN_DATATYPE_FLOAT_32";
      } else if (genieConfig["dialog"]["embedding"]["datatype"] == "native") {
        dataType = "QNN_DATATYPE_UNDEFINED";
      } else if (genieConfig["dialog"]["embedding"]["datatype"] == "ufixed8") {
        dataType = "QNN_DATATYPE_UFIXED_POINT_8";
      } else if (genieConfig["dialog"]["embedding"]["datatype"] == "ufixed16") {
        dataType = "QNN_DATATYPE_UFIXED_POINT_16";
      } else if (genieConfig["dialog"]["embedding"]["datatype"] == "sfixed8") {
        dataType = "QNN_DATATYPE_SFIXED_POINT_8";
      } else if (genieConfig["dialog"]["embedding"]["datatype"] == "sfixed16") {
        dataType = "QNN_DATATYPE_SFIXED_POINT_16";
      } else if (genieConfig["dialog"]["embedding"]["datatype"] == "ufixed4") {
        dataType = "QNN_DATATYPE_UFIXED_POINT_4";
      }
      quallaConfig["context"]["embedding-datatype"] = dataType;
    }
    if (genieConfig["dialog"]["embedding"].contains("quant-param")) {
      quallaConfig["context"]["quant-param"]["scale"] =
          genieConfig["dialog"]["embedding"]["quant-param"]["scale"];
      quallaConfig["context"]["quant-param"]["offset"] =
          genieConfig["dialog"]["embedding"]["quant-param"]["offset"];
    }

    // Encoder translation.
    if (genieConfig["dialog"]["embedding"].contains("type")) {
      quallaConfig["encoder"]["type"]      = genieConfig["dialog"]["embedding"]["type"];
      quallaConfig["encoder"]["lut-path"]  = genieConfig["dialog"]["embedding"]["lut-path"];
      quallaConfig["encoder"]["context"]   = quallaConfig["context"];
      quallaConfig["encoder"]["tokenizer"] = quallaConfig["tokenizer"];
    }
  }
}

static inline bool position_dim_set = false;
static inline bool rope_theta_set   = false;

//=============================================================================
// Backend::Config functions
//=============================================================================

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
  bool sharedEngine     = false;
  bool asyncInit        = false;
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
    } else if (item.key() == "data-alignment-size") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "mmap-budget") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "use-mmap") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "pos-id-dim") {
      position_dim_set = true;
      JSON_ENFORCE_NUMERIC();
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
      asyncInit = item.value();
    } else if (item.key() == "rope-theta") {
      rope_theta_set = true;
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "enable-graph-switching") {
      JSON_ENFORCE_BOOLEAN();
      graph_switching = item.value();
    } else if (item.key() == "shared-engine") {
      JSON_ENFORCE_BOOLEAN();
      sharedEngine = item.value();
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
  if (sharedEngine && !asyncInit) {
    throw Exception(
        GENIE_STATUS_ERROR_JSON_VALUE,
        "Invalid QnnHtp config. Engine sharing is only supported with async Init enabled");
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
      static const std::unordered_set<std::string> validBackendTypes = {
        "QnnHtp", "QnnGenAiTransformer", "QnnGpu", "libtorch"
      };

      type = item.value().get<std::string>();
      if (type == "QnnHtp") {
        htp = true;
      } else if (type == "QnnGenAiTransformer") {
        genai = true;
      } else if (!validBackendTypes.contains(type)) {
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
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing QnnHtp dialog config");
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
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing QnnGenAiTransformer dialog config");
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

static void validateLoraGroupConfig(const nlohmann::json& config,
                                    LORA_VERSION& specifiedLoraVersion) {
  if (specifiedLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V1) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                    "LoRA Adapter Groups must be used with lora version: 2 or 3");
  }
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "lora group config is not an object");
  }
  const std::set<std::string> mandatoryFields{"version", "name", "members", "quant-bin-sections"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing lora group field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  const std::string component = "lora adapter group";
  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid lora config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "name") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "members") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE, "members must be an array of strings");
        }
      }
    } else if (item.key() == "quant-bin-sections") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                          "quant-bin-sections must be an array of strings");
        }
      }
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown lora adapter group config key: " + item.key());
    }
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
    } else if (item.key() == "groups") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        validateLoraGroupConfig(elem, specifiedLoraVersion);
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
    const std::string& key = item.key();
    const auto& value      = item.value();

    if (key == "version") {
      JSON_ENFORCE_NUMERIC();
      if (value.get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid checkpoint config: unsupported version: " + value.dump());
      }
    } else if (key == "model-basedir") {
      JSON_ENFORCE_STRING();
    } else if (key == "output-dir") {
      JSON_ENFORCE_STRING();
    } else if (key == "cache-dir") {
      JSON_ENFORCE_STRING();
    } else if (key == "model-json-path") {
      JSON_ENFORCE_STRING();
    } else if (key == "clipping-map-path") {
      JSON_ENFORCE_STRING();
    } else if (key == "trainable-params-path") {
      JSON_ENFORCE_STRING();
    } else if (key == "data-type") {
      JSON_ENFORCE_STRING();
    } else if (key == "lora") {
      JSON_ENFORCE_OBJECT();
      validateLoraConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown checkpoint config key: " + key);
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
            ropeType != "linear" && ropeType != "hunyuan") {
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
            positionEncodingType != "alibi" && positionEncodingType != "none") {
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
  bool checkpoint = false;
  nlohmann::json checkpointConfig;
  bool positionalEncoding = false;
  nlohmann::json positionalEncodingConfig;

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
    } else if (item.key() == "library") {
      JSON_ENFORCE_OBJECT();
      libraryConfig = item.value();
    } else if (item.key() == "checkpoint") {
      JSON_ENFORCE_OBJECT();
      checkpointConfig = item.value();
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

//=============================================================================
// Engine::Config functions
//=============================================================================

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
      if (anchorAlpha < 0.0 || anchorAlpha > 1.0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid keydiff config: anchor-alpha is not in the range [0.0, 1.0]: " +
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
  // component is used in the "ENFORCE" macros
  std::string component = "longcontext";

  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "longcontext config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "type"};

  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing longcontext field: " + field);
    }
  }

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

static void validateOptimizerConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "optimizer-config entry is not an object");
  }

  // component is used in the "ENFORCE" macros
  std::string component = "optimizer-config";

  for (auto& item : config.items()) {
    const std::string& key = item.key();
    const auto& value      = item.value();

    // optimizer name
    if (key == "opt-name") {
      JSON_ENFORCE_STRING();
      std::string optName = value.get<std::string>();
      const std::set<std::string> supportedOptimizers = {"SGD", "AdamW"};
      if (supportedOptimizers.find(optName) == supportedOptimizers.end()) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Unsupported optimizer opt-name: " + optName);
      }
    }
    // learning rate
    else if (key == "learning-rate") {
      JSON_ENFORCE_NUMERIC();
      double lr = value.get<double>();
      if (lr <= 0.0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "learning-rate must be positive, got: " + std::to_string(lr));
      }
    }
    // weight decay
    else if (key == "weight-decay") {
      JSON_ENFORCE_NUMERIC();
    }
    // momentum
    else if (key == "momentum") {
      JSON_ENFORCE_NUMERIC();
      double momentum = value.get<double>();
      if (momentum < 0.0 || momentum > 1.0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "momentum must be between 0 and 1, got: " + std::to_string(momentum));
      }
    }
    // beta 1
    else if (key == "beta-1") {
      JSON_ENFORCE_NUMERIC();
      double beta1 = value.get<double>();
      if (beta1 < 0.0 || beta1 > 1.0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "beta_1 must be between 0 and 1, got: " + std::to_string(beta1));
      }
    }
    // beta 2
    else if (key == "beta-2") {
      JSON_ENFORCE_NUMERIC();
      double beta2 = value.get<double>();
      if (beta2 < 0.0 || beta2 > 1.0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "beta_2 must be between 0 and 1, got: " + std::to_string(beta2));
      }
    }
    // epsilon
    else if (key == "epsilon") {
      JSON_ENFORCE_NUMERIC();
    }
    else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown optimizer-config key: " + key);
    }
  }
}

static void validateTrainingConfig(const nlohmann::json& config) {
  // component is used in the "ENFORCE" macros
  const std::string component = "training-config";

  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "training-config entry is not an object");
  }

  const std::set<std::string> mandatoryFields{
      "random-seed", "torch-num-threads", "torch-num-inter-op-threads"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing training-config entry field: " + field);
    }
  }

  for (auto& item : config.items()) {
    if (item.key() == "random-seed") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "torch-num-threads") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "torch-num-inter-op-threads") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "device") {
      JSON_ENFORCE_STRING();
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
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown training-config key: " + item.key());
    }
  }
}

static void validateCacheGroupConfig(const nlohmann::json& config) {
  // component is used in the "ENFORCE" macros
  std::string component = "cache-groups";

  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "cache-groups entry is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "prefix"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing cache-groups entry field: " + field);
    }
  }

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(
            GENIE_STATUS_ERROR_JSON_VALUE,
            "Invalid cache-groups entry config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "longcontext") {
      validateLongContextConfig(item.value());
    } else if (item.key() == "prefix") {
      JSON_ENFORCE_STRING();
      if (item.value() == "") {
        throw Exception(
            GENIE_STATUS_ERROR_JSON_VALUE,
            "Invalid cache-groups entry config: cache tensor prefix cannot be an empty string.");
      }
    } else if (item.key() == "attention-mask-tensor-name") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "cache-index-tensor-name") {
      JSON_ENFORCE_STRING();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown cache-group config key: " + item.key());
    }
  }
}

static void validateEngineConfig(const nlohmann::json& config, std::string dialogType) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "engine config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "backend", "model", "n-threads"};
  if (dialogType == "spd") {
    mandatoryFields.insert("role");
  } else if (dialogType == "kv-share") {
    mandatoryFields.insert("role");
  }

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
    } else if (item.key() == "model") {
      JSON_ENFORCE_OBJECT();
      validateModelConfig(item.value());
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "n-threads") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "role" && dialogType == "spd") {
      JSON_ENFORCE_STRING();
      if (item.value() != "draft" && item.value() != "target") {
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                        "Unknown value: for engine config key: " + item.key());
      }
    } else if (item.key() == "role" && dialogType == "kv-share") {
      JSON_ENFORCE_STRING();
      if (item.value() != "primary" && item.value() != "secondary") {
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                        "Unknown value: for engine config key: " + item.key());
      }
    } else if (item.key() == "role" && dialogType == "eaglet") {
      JSON_ENFORCE_STRING();
      if (item.value() != "draft" && item.value() != "target") {
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                        "Unknown value: for engine config key: " + item.key());
      }
    } else if (item.key() == "longcontext") {
      validateLongContextConfig(item.value());
    } else if (item.key() == "cache-groups") {
      JSON_ENFORCE_ARRAY();
      if (item.value().size() == 0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE, "cache-groups cannot be an empty array.");
      }
      for (auto& cacheGroup : item.value()) {
        validateCacheGroupConfig(cacheGroup);
      }
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

static void validateMultiEngineConfig(const nlohmann::json& configs, std::string dialogType) {
  if (configs.is_object()) {
    validateEngineConfig(configs, dialogType);
    if (dialogType == "spd") {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "engine config for spd is not an array");
    } else if (dialogType == "kv-share") {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "engine config for kv-share is not an array");
    }
  } else if (configs.is_array() && dialogType == "spd") {
    if (configs.size() != 2) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "engine config for spd contain invalid number of engines");
    }
    bool engineRoleMask[2] = {false, false};
    for (auto& item : configs) {
      validateEngineConfig(item, dialogType);
      if (item["role"] == "draft") {
        engineRoleMask[0] = true;
      } else if (item["role"] == "target") {
        engineRoleMask[1] = true;
      }
    }
    if (!engineRoleMask[0]) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "engine config for spd does not contain draft engine");
    }
    if (!engineRoleMask[1]) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "engine config for spd does not contain target engine");
    }
  } else if (configs.is_array() && dialogType == "kv-share") {
    if (configs.size() != 2) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "engine config for kv-share contain invalid number of engines");
    }
    bool engineRoleMask[2] = {false, false};
    for (auto& item : configs) {
      validateEngineConfig(item, dialogType);
      if (item["role"] == "primary") {
        engineRoleMask[0] = true;
      } else if (item["role"] == "secondary") {
        engineRoleMask[1] = true;
      }
    }
    if (!engineRoleMask[0]) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "engine config for kv-share does not contain primary");
    }
    if (!engineRoleMask[1]) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "engine config for kv-share does not contain secondary");
    }
  } else if (configs.is_array() && dialogType == "eaglet") {
    if (configs.size() != 2) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "engine config for kv-share contain invalid number of engines");
    }
    bool engineRoleMask[2] = {false, false};
    for (auto& item : configs) {
      validateEngineConfig(item, dialogType);
      if (item["role"] == "target") {
        engineRoleMask[0] = true;
      } else if (item["role"] == "draft") {
        engineRoleMask[1] = true;
      }
    }
    if (!engineRoleMask[0]) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "engine config for eaglet does not contain target engine");
    }
    if (!engineRoleMask[1]) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "engine config for eaglet does not contain draft engine");
    }
  } else {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "engine config is not an object or an array");
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
      quallaEngineConfig["role"] = Engine::changeRole(genieEngineConfig["role"].get<std::string>());
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
      if (genieEngineConfig["backend"]["QnnHtp"].contains("data-alignment-size")) {
        quallaEngineConfig["data-alignment-size"] =
            genieEngineConfig["backend"]["QnnHtp"]["data-alignment-size"];
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
    } else if (genieEngineConfig["backend"]["type"] == "libtorch") {
      quallaEngineConfig["type"] = "libtorch";
      quallaEngineConfig["training_config"]["device"] = genieEngineConfig["backend"]["device"];
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
      const nlohmann::json& genieCkptConfig = genieEngineConfig["model"].at("checkpoint");
      nlohmann::json& quallaTrainingConfig  = quallaEngineConfig["training_config"];
      if (genieCkptConfig.contains("model-basedir")) {
        quallaTrainingConfig["model_basedir"] = genieCkptConfig["model-basedir"];
      }
      if (genieCkptConfig.contains("output-dir")) {
        quallaTrainingConfig["output_dir"] = genieCkptConfig["output-dir"];
      }
      if (genieCkptConfig.contains("cache-dir")) {
        quallaTrainingConfig["cache_dir"] = genieCkptConfig["cache-dir"];
      }
      if (genieCkptConfig.contains("model-json-path")) {
        quallaTrainingConfig["model_json_path"] = genieCkptConfig["model-json-path"];
      }
      if (genieCkptConfig.contains("clipping-map-path")) {
        quallaTrainingConfig["clipping_map_path"] = genieCkptConfig["clipping-map-path"];
      }
      if (genieCkptConfig.contains("trainable-params-path")) {
        quallaTrainingConfig["trainable_params_path"] = genieCkptConfig["trainable-params-path"];
      }
      if (genieCkptConfig.contains("data-type")) {
        quallaTrainingConfig["dtype"] = genieCkptConfig["data-type"];
      }
      if (genieCkptConfig.contains("lora")) {
        translateLoraConfig(genieCkptConfig["lora"], quallaEngineConfig["loraConfig"]);
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
            if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["rope-type"] ==
                "linear") {
              if (genieEngineConfig["model"]["positional-encoding"]["rope-scaling"].contains(
                      "factor")) {
                quallaEngineConfig["positional-encoding"]["rope-scaling"]["factor"] =
                    genieEngineConfig["model"]["positional-encoding"]["rope-scaling"]["factor"];
              }
            }
          }
        }
      }
    }
    if (genieEngineConfig["model"].contains("draft-token-map")) {
      quallaEngineConfig["draft-token-map"] = genieEngineConfig["model"]["draft-token-map"];
    }
    if (genieEngineConfig["model"].contains("cross-attention")) {
      quallaEngineConfig["cross-attention"] = genieEngineConfig["model"]["cross-attention"];
    }
    if (genieEngineConfig["model"].contains("mask-type")) {
      quallaEngineConfig["mask-type"] = genieEngineConfig["model"]["mask-type"];
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
    if (genieEngineConfig.contains("training-config")) {
      const nlohmann::json& genieTrainingConfig = genieEngineConfig["training-config"];
      nlohmann::json& quallaTrainingConfig      = quallaEngineConfig["training_config"];
      if (genieTrainingConfig.contains("random-seed")) {
        quallaTrainingConfig["random_seed"] = genieTrainingConfig["random-seed"];
      }
      if (genieTrainingConfig.contains("torch-num-threads")) {
        quallaTrainingConfig["torch_num_threads"] = genieTrainingConfig["torch-num-threads"];
      }
      if (genieTrainingConfig.contains("torch-num-inter-op-threads")) {
        quallaTrainingConfig["torch_num_inter_op_threads"] = genieTrainingConfig["torch-num-inter-op-threads"];
      }
      if (genieTrainingConfig.contains("device")) {
        quallaTrainingConfig["device"] = genieTrainingConfig["device"];
      }
      if (genieTrainingConfig.contains("max-iterations")) {
        quallaTrainingConfig["max_iterations"] = genieTrainingConfig["max-iterations"];
      }
      if (genieTrainingConfig.contains("batch-size")) {
        quallaTrainingConfig["batch_size"] = genieTrainingConfig["batch-size"];
      }
      if (genieTrainingConfig.contains("softmax-topk")) {
        quallaTrainingConfig["softmax_topk"] = genieTrainingConfig["softmax-topk"];
      }
      if (genieTrainingConfig.contains("gclip-max-norm")) {
        quallaTrainingConfig["gclip_max_norm"] = genieTrainingConfig["gclip-max-norm"];
      }
      if (genieTrainingConfig.contains("enable-logit-masking")) {
        quallaTrainingConfig["enable_logit_masking"] = genieTrainingConfig["enable-logit-masking"];
      }
      if (genieTrainingConfig.contains("enable-clear-torch-op-cache")) {
        quallaTrainingConfig["enable_clear_torch_op_cache"] = genieTrainingConfig["enable-clear-torch-op-cache"];
      }
      if (genieTrainingConfig.contains("loss-type")) {
        quallaTrainingConfig["loss_type"] = genieTrainingConfig["loss-type"];
      }
    }
    if (genieEngineConfig.contains("optimizer-config")) {
      const nlohmann::json& genieOptimizerConfig = genieEngineConfig["optimizer-config"];
      nlohmann::json quallaOptimizerConfig;
      if (genieOptimizerConfig.contains("opt-name")) {
        quallaOptimizerConfig["opt_name"] = genieOptimizerConfig["opt-name"];
      }
      if (genieOptimizerConfig.contains("learning-rate")) {
        quallaOptimizerConfig["learning_rate"] = genieOptimizerConfig["learning-rate"];
      }
      if (genieOptimizerConfig.contains("weight-decay")) {
        quallaOptimizerConfig["weight_decay"] = genieOptimizerConfig["weight-decay"];
      }
      if (genieOptimizerConfig.contains("momentum")) {
        quallaOptimizerConfig["momentum"] = genieOptimizerConfig["momentum"];
      }
      if (genieOptimizerConfig.contains("beta-1")) {
        quallaOptimizerConfig["beta_1"] = genieOptimizerConfig["beta-1"];
      }
      if (genieOptimizerConfig.contains("beta-2")) {
        quallaOptimizerConfig["beta_2"] = genieOptimizerConfig["beta-2"];
      }
      if (genieOptimizerConfig.contains("epsilon")) {
        quallaOptimizerConfig["epsilon"] = genieOptimizerConfig["epsilon"];
      }
      quallaEngineConfig["optimizer_config"] = quallaOptimizerConfig;
    }
  }
}

static void translateEngineDebugConfig(const nlohmann::json& genieConfig,
                                       nlohmann::json& quallaEngineConfig) {
  if (genieConfig["dialog"].contains("debug")) {
    if (genieConfig["dialog"]["debug"].contains("path")) {
      quallaEngineConfig["debug-path"] = genieConfig["dialog"]["debug"]["path"];
    } else {
      quallaEngineConfig["debug-path"] = "genie_debug";
    }
    if (genieConfig["dialog"]["debug"].contains("dump-tensors")) {
      quallaEngineConfig["debug-tensors"] = genieConfig["dialog"]["debug"]["dump-tensors"];
    }
    if (genieConfig["dialog"]["debug"].contains("dump-specs")) {
      quallaEngineConfig["debug-specs"] = genieConfig["dialog"]["debug"]["dump-specs"];
    }
    if (genieConfig["dialog"]["debug"].contains("dump-outputs")) {
      quallaEngineConfig["debug-outputs"] = genieConfig["dialog"]["debug"]["dump-outputs"];
    }
  }
}

static void translateMultiEngineConfig(const nlohmann::json& genieConfig,
                                       nlohmann::json& quallaConfig,
                                       const size_t reservedTokens) {
  if (!genieConfig["dialog"].contains("engine")) return;
  if (genieConfig["dialog"]["engine"].is_array()) {
    quallaConfig["engine"] = nlohmann::json::array();
    for (auto& item : genieConfig["dialog"]["engine"]) {
      nlohmann::json quallaEngineConfig;
      translateEngineConfig(item, quallaEngineConfig, reservedTokens);
      translateEngineDebugConfig(genieConfig, quallaEngineConfig);
      quallaConfig["engine"].push_back(quallaEngineConfig);
    }
  } else {
    nlohmann::json quallaEngineConfig;
    translateEngineConfig(genieConfig["dialog"]["engine"], quallaEngineConfig, reservedTokens);
    translateEngineDebugConfig(genieConfig, quallaEngineConfig);
    quallaConfig["engine"] = quallaEngineConfig;
  }
}

static void validateDebugConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Dialog config is not an object");
  }
  // component is used in the "ENFORCE" macros
  std::string component = "debug";

  for (auto& item : config.items()) {
    if (item.key() == "path") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "dump-tensors") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "dump-specs") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "dump-outputs") {
      JSON_ENFORCE_BOOLEAN();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown multistream config key: " + item.key());
    }
  }
}

//=============================================================================
// Dialog::Config functions
//=============================================================================
qnn::util::HandleManager<Dialog::Config>& Dialog::Config::getManager() {
  static qnn::util::HandleManager<Dialog::Config> s_manager;
  return s_manager;
}

GenieDialogConfig_Handle_t Dialog::Config::add(std::shared_ptr<Dialog::Config> config) {
  return reinterpret_cast<GenieDialogConfig_Handle_t>(getManager().add(config));
}

std::shared_ptr<Dialog::Config> Dialog::Config::get(GenieDialogConfig_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Dialog::Config::remove(GenieDialogConfig_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

static void validateDialogSsdConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "ssd-q1 config is not an object");
  }

  std::set<std::string> mandatoryFields{"version",
                                        "ssd-version",
                                        "forecast-token-count",
                                        "branches",
                                        "forecast-prefix",
                                        "forecast-prefix-name"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing ssd-q1 field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "ssd-q1";

  int branchesSize       = 0;
  int forecastTokenCount = 0;

  int nStreams     = 1;
  float pThreshold = 0.0;

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid ssd-q1 config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "ssd-version") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "forecast-token-count") {
      JSON_ENFORCE_NUMERIC();
      forecastTokenCount = item.value();
    } else if (item.key() == "branches") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        if (!elem.is_number_integer()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE, "branches must be an array of integers");
        }
      }
      branchesSize = item.value().size();
    } else if (item.key() == "branch-mode") {
      JSON_ENFORCE_STRING();
      if (item.value() != "top-1" && item.value() != "all-expand") {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "branch-mode must be either top-1 or all-expand");
      }
    } else if (item.key() == "forecast-prefix") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "forecast-prefix-name") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "prefix-quant") {
      JSON_ENFORCE_NUMERIC();
      int prefixQuant = item.value();
      if (prefixQuant < 0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "prefix-quant must be a non-negative integer");
      }
    } else if (item.key() == "n-streams") {
      JSON_ENFORCE_NUMERIC();
      nStreams = item.value();
    } else if (item.key() == "p-threshold") {
      JSON_ENFORCE_NUMERIC();
      pThreshold = item.value();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown ssd-q1 config key: " + item.key());
    }
  }

  if ((pThreshold > 0.0f) && (nStreams <= 1)) {
    throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                    "p-threshold can only be used with multistream (n-streams > 1)");
  }

  if (branchesSize > forecastTokenCount) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                    "Size of branches array must be less than forecast-token-count");
  }
}

static void validateDialogEagletConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "ssd-q1 config is not an object");
  }

  std::set<std::string> mandatoryFields{"version",
                                        "eaglet-version",
                                        "draft-len",
                                        "n-branches",
                                        "max-tokens-target-can-evaluate",
                                        "draft-kv-cache"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing eaglet field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "eaglet";
  int draftLen          = 0;
  int nBranches         = 0;
  int maxTargetEvaluate = 0;

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid eaglet config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "eaglet-version") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "draft-len") {
      JSON_ENFORCE_NUMERIC();
      draftLen = item.value();
      if (draftLen <= 0) {
        throw Exception(
            GENIE_STATUS_ERROR_JSON_VALUE,
            "Invalid eaglet draft-len config: unsupported value: " + item.value().dump());
      }
    } else if (item.key() == "n-branches") {
      JSON_ENFORCE_NUMERIC();
      nBranches = item.value();
      if (nBranches <= 0) {
        throw Exception(
            GENIE_STATUS_ERROR_JSON_VALUE,
            "Invalid eaglet n-branch config: unsupported value: " + item.value().dump());
      }
    } else if (item.key() == "max-tokens-target-can-evaluate") {
      JSON_ENFORCE_NUMERIC();
      maxTargetEvaluate = item.value();
      if (maxTargetEvaluate <= 0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid eaglet max-tokens-target-can-evaluate config: unsupported value " +
                            item.value().dump());
      }
    } else if (item.key() == "draft-kv-cache") {
      JSON_ENFORCE_BOOLEAN();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown eaglet config key: " + item.key());
    }
  }
}

static void validateDialogLadeConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "lade config is not an object");
  }

  std::set<std::string> mandatoryFields{"version"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing lade field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "lade";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid lade config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "update-mode") {
      JSON_ENFORCE_STRING();
      std::string mode = item.value().get<std::string>();
      if ((mode != "FWD_MAX_HIT") && (mode != "FWD_LEVEL") && (mode != "ALWAYS_FWD_ONE")) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid lade config: unsupported update-mode: " + item.value().dump());
      }
    } else if (item.key() == "window") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "ngram") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "gcap") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown lade config key: " + item.key());
    }
  }
}

static void validateDialogSpdConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "spd config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "draft-len"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing spd field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "spd";
  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid spd config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "draft-len") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown spd config key: " + item.key());
    }
  }
}

static void validateDialogKVShareConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "kv-share config is not an object");
  }

  // component is used in the "ENFORCE" macros
  std::string component = "kv-share";
  for (auto& item : config.items()) {
    if (item.key() == "enable-in-memory-kv-share") {
      JSON_ENFORCE_BOOLEAN();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown kv-share config key: " + item.key());
    }
  }
}

static void validateDialogMultistreamConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "multistream config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "n-streams"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing multistream field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "multistream";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid multistream config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "n-streams") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "p-threshold") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown multistream config key: " + item.key());
    }
  }
}

// used for updating the dialog config for older configs for kv-share dialog
void Dialog::updateDialogConfigForKVShare(nlohmann::json& config) {
  if (!config["dialog"].contains("type") || !config["dialog"]["type"].is_string()) {
    return;
  }

  // Proceed only if 'kv-share' is missing
  if (config["dialog"]["type"] != "kv-share" || config["dialog"].contains("kv-share")) {
    return;
  }

  if (!config["dialog"].contains("engine") || !config["dialog"]["engine"].is_array()) {
    return;
  }

  for (auto& e : config["dialog"]["engine"]) {
    if (!e.is_object() || !e.contains("role") || !e["role"].is_string()) {
      return;
    }

    if (e["role"] != "secondary") {
      continue;
    }

    if (!e.contains("backend") || !e["backend"].is_object()) {
      return;
    }

    if (!e["backend"].contains("QnnGenAiTransformer") ||
        !e["backend"]["QnnGenAiTransformer"].is_object()) {
      return;
    }

    // update kv-share config
    /*
        ...
        "kv-share": {
          "enable-in-memory-kv-share": true/false
        },
        ...
    */
    nlohmann::json kvShareConfig;
    if (e["backend"]["QnnGenAiTransformer"].contains("enable-in-memory-kv-share")) {
      kvShareConfig["enable-in-memory-kv-share"] =
          e["backend"]["QnnGenAiTransformer"]["enable-in-memory-kv-share"];
      // remove the option from the engine config
      e["backend"]["QnnGenAiTransformer"].erase("enable-in-memory-kv-share");
    } else {
      kvShareConfig["enable-in-memory-kv-share"] = false;
    }
    config["dialog"]["kv-share"] = kvShareConfig;
    break;
  }
}

void Dialog::validateDialogConfig(const nlohmann::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Dialog config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "type", "context", "tokenizer", "engine"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing dialog field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "dialog";

  std::string dialogType = "basic";
  bool ssdq1             = false;
  nlohmann::json ssdq1Config;
  bool lade = false;
  nlohmann::json ladeConfig;
  bool spd = false;
  nlohmann::json spdConfig;
  bool kvshare = false;
  nlohmann::json kvshareConfig;
  bool multistream = false;
  nlohmann::json multistreamConfig;
  bool eaglet = false;
  nlohmann::json eagletConfig;

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid dialog config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
      dialogType = item.value();
      if (dialogType == "basic") {
        // Do nothing
      } else if (dialogType == "ssd-q1") {
        ssdq1 = true;
      } else if (dialogType == "lade") {
        lade = true;
      } else if (dialogType == "spd") {
        spd = true;
      } else if (dialogType == "multistream") {
        multistream = true;
      } else if (dialogType == "eaglet") {
        eaglet = true;
      } else if (dialogType == "kv-share") {
        kvshare = true;
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE, "Invalid dialog type: " + dialogType);
      }
    } else if (item.key() == "accumulator-size") {
      // Do nothing
    } else if (item.key() == "ssd-q1") {
      JSON_ENFORCE_OBJECT();
      ssdq1Config = item.value();
      // ssd-q1 validation is done below
    } else if (item.key() == "lade") {
      JSON_ENFORCE_OBJECT();
      ladeConfig = item.value();
      // ssd-q1 validation is done below
    } else if (item.key() == "spd") {
      JSON_ENFORCE_OBJECT();
      spdConfig = item.value();
      // spd validation is done below
    } else if (item.key() == "kv-share") {
      JSON_ENFORCE_OBJECT();
      kvshareConfig = item.value();
      // kv-share validation is done below
    } else if (item.key() == "multistream") {
      JSON_ENFORCE_OBJECT();
      multistreamConfig = item.value();
      // multistream validation is done below
    } else if (item.key() == "eaglet") {
      JSON_ENFORCE_OBJECT();
      eagletConfig = item.value();
      // eaglet validation is done below
    } else if (item.key() == "stop-sequence") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                          "stop-sequence must be an array of strings");
        }
      }
    } else if (item.key() == "max-num-tokens") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() < 0) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "number of tokens must be > 0. provided: " + item.value().dump());
      }
    } else if (item.key() == "context") {
      JSON_ENFORCE_OBJECT();
      Context::ContextConfig::validateContextConfig(item.value());
    } else if (item.key() == "tokenizer") {
      JSON_ENFORCE_OBJECT();
      validateTokenizerConfig(item.value());
    } else if (item.key() == "sampler") {
      JSON_ENFORCE_OBJECT();
      Sampler::SamplerConfig::validateSamplerConfig(item.value());
    } else if (item.key() == "engine") {
      JSON_ENFORCE_ARRAY_OR_OBJECT();
    } else if (item.key() == "embedding") {
      JSON_ENFORCE_OBJECT();
      validateEmbeddingConfig(item.value());
    } else if (item.key() == "debug") {
      JSON_ENFORCE_OBJECT();
      validateDebugConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown dialog config key: " + item.key());
    }
  }

  // Engine Verification requires dialogType for engine roles. Since "type" is encounterd
  // later than "engine" in loop. Therefore, moving engine validation out of the loop.
  validateMultiEngineConfig(config["engine"], dialogType);
  if (ssdq1) {
    if (!ssdq1Config.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing ssd-q1 dialog config");
    }
    validateDialogSsdConfig(ssdq1Config);
  } else {
    if (ssdq1Config.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "ssd-q1 dialog config for incorrect dialog type: " + dialogType);
    }
  }

  if (lade) {
    if (!ladeConfig.empty()) {
      if (!ladeConfig.is_object()) {
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing lade dialog config");
      }
      validateDialogLadeConfig(ladeConfig);
    }
  } else {
    if (ladeConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "lade dialog config for incorrect dialog type: " + dialogType);
    }
  }

  if (spd) {
    if (!spdConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing spd dialog config");
    }
    validateDialogSpdConfig(spdConfig);
  } else {
    if (spdConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "spd dialog config for incorrect dialog type: " + dialogType);
    }
  }

  if (kvshare) {
    if (kvshareConfig.is_object()) {
      validateDialogKVShareConfig(kvshareConfig);
    }
  } else {
    if (kvshareConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "kv-share dialog config for incorrect dialog type: " + dialogType);
    }
  }

  if (multistream) {
    if (!multistreamConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing multistream dialog config");
    }
    validateDialogMultistreamConfig(multistreamConfig);
  }
  if (eaglet) {
    if (!eagletConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "eaglet dialog config for incorrect dialog type: " + dialogType);
    }
    validateDialogEagletConfig(eagletConfig);
  }
}

void Dialog::translateDialogConfig(const nlohmann::json& genieConfig,
                                   nlohmann::json& quallaConfig) {
  size_t ssdPrefixLength{0};
  if (genieConfig["dialog"]["version"] == 1) {
    if (genieConfig["dialog"]["type"] == "lade") {
      quallaConfig["type"] = "lhd-dec";
    } else if (genieConfig["dialog"]["type"] == "spd") {
      quallaConfig["type"] = "spec-dec";
    } else if (genieConfig["dialog"]["type"] == "multistream") {
      quallaConfig["type"] = "multistream";
    } else if (genieConfig["dialog"]["type"] == "eaglet") {
      quallaConfig["type"] = "eaglet";
    } else if (genieConfig["dialog"]["type"] == "kv-share") {
      quallaConfig["type"] = "kv-share";
    } else {
      quallaConfig["type"] = genieConfig["dialog"]["type"];
    }

    if (genieConfig["dialog"]["type"] == "ssd-q1") {
      quallaConfig["ssd-version"] = genieConfig["dialog"]["ssd-q1"]["ssd-version"];
      quallaConfig["forecast-token-count"] =
          genieConfig["dialog"]["ssd-q1"]["forecast-token-count"];
      quallaConfig["branches"] = genieConfig["dialog"]["ssd-q1"]["branches"];
      if (genieConfig["dialog"]["ssd-q1"].contains("branch-mode")) {
        if (genieConfig["dialog"]["ssd-q1"]["branch-mode"] == "top-1") {
          auto branches = genieConfig["dialog"]["ssd-q1"]["branches"];
          std::vector<std::vector<size_t>> quallaBranches;
          for (size_t i = 0; i < branches.size(); i++) {
            if (i == 0) {
              quallaBranches.push_back({branches[i]});
            } else {
              quallaBranches.push_back({branches[i], 0});
            }
          }
          quallaConfig["branches"] = quallaBranches;
        }
      }
      ssdPrefixLength                 = genieConfig["dialog"]["ssd-q1"]["forecast-prefix"];
      quallaConfig["forecast-prefix"] = ssdPrefixLength;
      quallaConfig["forecast-prefix-name"] =
          genieConfig["dialog"]["ssd-q1"]["forecast-prefix-name"];
      if (genieConfig["dialog"]["ssd-q1"].contains("prefix-quant")) {
        quallaConfig["prefix-quant"]  = genieConfig["dialog"]["ssd-q1"]["prefix-quant"];
      }
      if (genieConfig["dialog"]["ssd-q1"].contains("n-streams")) {
        quallaConfig["n-streams"] = genieConfig["dialog"]["ssd-q1"]["n-streams"];
      }
      if (genieConfig["dialog"]["ssd-q1"].contains("p-threshold")) {
        quallaConfig["p-threshold"] = genieConfig["dialog"]["ssd-q1"]["p-threshold"];
      }
    } else if (genieConfig["dialog"]["type"] == "lade") {
      if (genieConfig["dialog"]["lade"].contains("update-mode"))
        quallaConfig["lhd-update-mode"] = genieConfig["dialog"]["lade"]["update-mode"];
      if (genieConfig["dialog"]["lade"].contains("window"))
        quallaConfig["window"] = genieConfig["dialog"]["lade"]["window"];
      if (genieConfig["dialog"]["lade"].contains("ngram"))
        quallaConfig["ngram"] = genieConfig["dialog"]["lade"]["ngram"];
      if (genieConfig["dialog"]["lade"].contains("gcap"))
        quallaConfig["gcap"] = genieConfig["dialog"]["lade"]["gcap"];
    } else if (genieConfig["dialog"]["type"] == "spd") {
      quallaConfig["draft-len"] = genieConfig["dialog"]["spd"]["draft-len"];
    } else if (genieConfig["dialog"]["type"] == "multistream") {
      quallaConfig["n-streams"] = genieConfig["dialog"]["multistream"]["n-streams"];
      if (genieConfig["dialog"]["multistream"].contains("p-threshold")) {
        quallaConfig["p-threshold"] = genieConfig["dialog"]["multistream"]["p-threshold"];
      }
    } else if (genieConfig["dialog"]["type"] == "eaglet") {
      quallaConfig["eaglet-version"] = genieConfig["dialog"]["eaglet"]["eaglet-version"];
      quallaConfig["draft-len"]      = genieConfig["dialog"]["eaglet"]["draft-len"];
      quallaConfig["n-branches"]     = genieConfig["dialog"]["eaglet"]["n-branches"];
      quallaConfig["max-tokens-target-can-evaluate"] =
          genieConfig["dialog"]["eaglet"]["max-tokens-target-can-evaluate"];
      quallaConfig["draft-kv-cache"] = genieConfig["dialog"]["eaglet"]["draft-kv-cache"];
    } else if (genieConfig["dialog"].contains("kv-share")) {
      if (genieConfig["dialog"]["kv-share"].contains("enable-in-memory-kv-share")) {
        quallaConfig["kv-share"]["enable-in-memory-kv-share"] =
            genieConfig["dialog"]["kv-share"]["enable-in-memory-kv-share"];
      }
    }
  }
  if (genieConfig["dialog"].contains("stop-sequence")) {
    quallaConfig["prompt"]["stop-sequence"] = genieConfig["dialog"]["stop-sequence"];
  }

  Context::ContextConfig::translateContextConfig(genieConfig["dialog"], quallaConfig);
  translateTokenizerConfig(genieConfig, quallaConfig);
  Sampler::SamplerConfig::translateSamplerConfig(genieConfig["dialog"], quallaConfig);
  translateMultiEngineConfig(genieConfig, quallaConfig, ssdPrefixLength);
  translateEmbeddingConfig(genieConfig, quallaConfig);

  if (genieConfig.contains("loraConfig")) {
    quallaConfig["loraConfig"] = nlohmann::json::array();
    for (auto& lc : genieConfig["loraConfig"]) {
      nlohmann::json temp;
      translateLoraConfig(lc, temp);
      quallaConfig["loraConfig"].push_back(temp);
    }
  }
}

void Dialog::getStandaloneEnginesConfig(nlohmann::json& genieConfig,
                                        nlohmann::json& genieStandaloneEnginesConfig) {
  genieStandaloneEnginesConfig["shared-engines"] = nlohmann::json::array();
  genieConfig["loraConfig"]                      = nlohmann::json::array();
  if (genieConfig["dialog"]["engine"].is_array()) {
    for (auto it = genieConfig["dialog"]["engine"].begin();
         it != genieConfig["dialog"]["engine"].end();) {
      auto& engine = *it;
      if (engine["backend"].contains("QnnHtp") &&
          engine["backend"]["QnnHtp"].contains("shared-engine") &&
          engine["backend"]["QnnHtp"]["shared-engine"]) {
        nlohmann::json engineConfig;
        engineConfig["standalone-engine"]["version"] = 1;
        if (genieConfig["dialog"].contains("embedding")) {
          engineConfig["standalone-engine"]["embedding"] = genieConfig["dialog"]["embedding"];
        }
        if (genieConfig["dialog"].contains("context")) {
          engineConfig["standalone-engine"]["context"] = genieConfig["dialog"]["context"];
        }
        engineConfig["standalone-engine"]["engine"] = engine;
        if (engineConfig["standalone-engine"]["engine"]["model"]["binary"].contains("lora")) {
          nlohmann::json lora =
              engineConfig["standalone-engine"]["engine"]["model"]["binary"]["lora"];
          lora["role"] =
              engineConfig["standalone-engine"]["engine"].contains("role")
                  ? Engine::changeRole(engineConfig["standalone-engine"]["engine"]["role"])
                  : Engine::changeRole("primary");
          genieConfig["loraConfig"].push_back(lora);
          engineConfig["standalone-engine"]["engine"]["model"]["binary"].erase("lora");
        }
        genieStandaloneEnginesConfig["shared-engines"].push_back(engineConfig);
        genieConfig["dialog"]["engine"].erase(it);
        it = genieConfig["dialog"]["engine"].begin();
      } else if (engine["backend"].contains("QnnGenAiTransformer") &&
                 engine["backend"]["QnnGenAiTransformer"].contains("shared-engine") &&
                 engine["backend"]["QnnGenAiTransformer"]["shared-engine"]) {
        nlohmann::json engineConfig;
        engineConfig["standalone-engine"]["version"] = 1;
        if (genieConfig["dialog"].contains("embedding")) {
          engineConfig["standalone-engine"]["embedding"] = genieConfig["dialog"]["embedding"];
        }
        if (genieConfig["dialog"].contains("context")) {
          engineConfig["standalone-engine"]["context"] = genieConfig["dialog"]["context"];
        }
        engineConfig["standalone-engine"]["engine"] = engine;

        if (engineConfig["standalone-engine"]["engine"]["model"]["library"].contains("lora")) {
          nlohmann::json lora =
              engineConfig["standalone-engine"]["engine"]["model"]["library"]["lora"];
          lora["role"] =
              engineConfig["standalone-engine"]["engine"].contains("role")
                  ? Engine::changeRole(engineConfig["standalone-engine"]["engine"]["role"])
                  : Engine::changeRole("primary");
          genieConfig["loraConfig"].push_back(lora);
          engineConfig["standalone-engine"]["engine"]["model"]["library"].erase("lora");
        }

        genieStandaloneEnginesConfig["shared-engines"].push_back(engineConfig);
        genieConfig["dialog"]["engine"].erase(it);
        it = genieConfig["dialog"]["engine"].begin();
      } else {
        it++;
      }
    }
  } else {
    if (genieConfig["dialog"]["engine"]["backend"].contains("QnnGenAiTransformer") &&
        genieConfig["dialog"]["engine"]["backend"]["QnnGenAiTransformer"].contains(
            "shared-engine") &&
        genieConfig["dialog"]["engine"]["backend"]["QnnGenAiTransformer"]["shared-engine"]) {
      nlohmann::json engineConfig;
      engineConfig["standalone-engine"]["version"] = 1;
      if (genieConfig["dialog"].contains("embedding")) {
        engineConfig["standalone-engine"]["embedding"] = genieConfig["dialog"]["embedding"];
      }
      if (genieConfig["dialog"].contains("context")) {
        engineConfig["standalone-engine"]["context"] = genieConfig["dialog"]["context"];
      }
      engineConfig["standalone-engine"]["engine"] = genieConfig["dialog"]["engine"];

      if (engineConfig["standalone-engine"]["engine"]["model"]["library"].contains("lora")) {
        nlohmann::json lora =
            engineConfig["standalone-engine"]["engine"]["model"]["library"]["lora"];
        lora["role"] = engineConfig["standalone-engine"]["engine"].contains("role")
                           ? Engine::changeRole(engineConfig["standalone-engine"]["engine"]["role"])
                           : Engine::changeRole("primary");
        genieConfig["loraConfig"].push_back(lora);
        engineConfig["standalone-engine"]["engine"]["model"]["library"].erase("lora");
      }
      genieStandaloneEnginesConfig["shared-engines"].push_back(engineConfig);
      genieConfig["dialog"].erase("engine");
    } else if (genieConfig["dialog"]["engine"]["backend"].contains("QnnHtp") &&
               genieConfig["dialog"]["engine"]["backend"]["QnnHtp"].contains("shared-engine") &&
               genieConfig["dialog"]["engine"]["backend"]["QnnHtp"]["shared-engine"]) {
      nlohmann::json engineConfig;
      engineConfig["standalone-engine"]["version"] = 1;
      if (genieConfig["dialog"].contains("embedding")) {
        engineConfig["standalone-engine"]["embedding"] = genieConfig["dialog"]["embedding"];
      }
      if (genieConfig["dialog"].contains("context")) {
        engineConfig["standalone-engine"]["context"] = genieConfig["dialog"]["context"];
      }
      engineConfig["standalone-engine"]["engine"] = genieConfig["dialog"]["engine"];
      if (engineConfig["standalone-engine"]["engine"]["model"]["binary"].contains("lora")) {
        nlohmann::json lora =
            engineConfig["standalone-engine"]["engine"]["model"]["binary"]["lora"];
        lora["role"] = engineConfig["standalone-engine"]["engine"].contains("role")
                           ? Engine::changeRole(engineConfig["standalone-engine"]["engine"]["role"])
                           : Engine::changeRole("primary");
        genieConfig["loraConfig"].push_back(lora);
        engineConfig["standalone-engine"]["engine"]["model"]["binary"].erase("lora");
      }
      genieStandaloneEnginesConfig["shared-engines"].push_back(engineConfig);
      genieConfig["dialog"].erase("engine");
    }
  }

  // Clean up empty loraConfig array if no configurations were added
  if (genieConfig["loraConfig"].empty()) {
    genieConfig.erase("loraConfig");
  }
}

uint32_t getMaxNumTokens(const nlohmann::json& genieConfig) {
  uint32_t tokenLimit{UINT32_MAX};
  if (genieConfig["dialog"]["version"] == 1) {
    if (genieConfig["dialog"].contains("max-num-tokens")) {
      tokenLimit = genieConfig["dialog"]["max-num-tokens"];
    }
  }
  return tokenLimit;
}

void Dialog::verifyAndUpdateConfig(nlohmann::json& config) {
  std::set<std::string> mandatoryFields{"dialog"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing dialog field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "dialog";

  for (auto& item : config.items()) {
    if (item.key() == "dialog") {
      JSON_ENFORCE_OBJECT();
      // update the config for kv-share to support older configs
      updateDialogConfigForKVShare(config);
      validateDialogConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown dialog config key: " + item.key());
    }
  }
}

void Dialog::Config::createConfigFromJson(const char* configStr) {
  rope_theta_set   = false;
  position_dim_set = false;
  nlohmann::json config;
  util::getJsonFromStr(configStr, config);
  verifyAndUpdateConfig(config);
  m_config = config;
}

void Dialog::Config::createConfigFromDlc(const char* useCaseName, const char* configStr) {
  rope_theta_set   = false;
  position_dim_set = false;
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
          m_config = genieConfigJson;

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

Dialog::Config::Config(const char* inputStr) { createConfigFromJson(inputStr); }

Dialog::Config::Config(std::shared_ptr<Dlc> dlc, const char* useCaseName, const char* configStr) {
  m_dlc = std::move(dlc);
  createConfigFromDlc(useCaseName, configStr);
}

nlohmann::json& Dialog::Config::getJson() {
  if (!m_config.empty()) {
    return m_config;
  }
  return m_config;
}

std::shared_ptr<Dlc>& Dialog::Config::getDlc() { return m_dlc; }

void Dialog::Config::bindLogger(std::shared_ptr<genie::log::Logger> logger) {
  if (!logger) return;
  logger->incrementUseCount();
  m_logger.insert(logger);
}

void Dialog::Config::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& Dialog::Config::getLogger() {
  return m_logger;
}

void Dialog::Config::bindProfiler(std::shared_ptr<Profiler> profiler) {
  if (!profiler) return;
  for (auto it : m_profiler) {
    if (it->m_traceLogger != profiler->m_traceLogger) {
      if (it->m_traceLogger && profiler->m_traceLogger) {
        throw Exception(GENIE_STATUS_ERROR_INVALID_ARGUMENT,
                        "Cannot bind multiple trace profilers to the same Dialog config.");
      } else {
        throw Exception(
            GENIE_STATUS_ERROR_INVALID_ARGUMENT,
            "Cannot bind profilers with different configurations to the same Dialog config.");
      }
    }
  }
  profiler->incrementUseCount();
  m_profiler.insert(profiler);
}

void Dialog::Config::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& Dialog::Config::getProfiler() { return m_profiler; }

//=============================================================================
// Dialog functions
//=============================================================================
std::atomic<std::uint32_t> Dialog::s_nameCounter{0u};

qnn::util::HandleManager<Dialog>& Dialog::getManager() {
  static qnn::util::HandleManager<Dialog> s_manager;
  return s_manager;
}

GenieDialog_Handle_t Dialog::add(std::shared_ptr<Dialog> dialog) {
  return reinterpret_cast<GenieDialog_Handle_t>(getManager().add(dialog));
}

std::shared_ptr<Dialog> Dialog::get(GenieDialog_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Dialog::remove(GenieDialog_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Dialog::initDialog(nlohmann::json config,
                        std::shared_ptr<qualla::Env> env,
                        std::shared_ptr<ProfileStat> profileStat,
                        std::shared_ptr<genie::log::Logger> logger,
                        std::shared_ptr<genie::Profiler> profiler) {
  if (logger) env->bindLogger(logger);
  if (profiler) env->setTraceLogger(profiler->m_traceLogger);
  nlohmann::json quallaConfig;
  nlohmann::json standaloneEnginesConfig;
  getStandaloneEnginesConfig(config, standaloneEnginesConfig);
  translateDialogConfig(config, quallaConfig);
  m_tokenLimit   = getMaxNumTokens(config);
  m_name         = "dialog" + std::to_string(s_nameCounter.fetch_add(1u));
  m_quallaDialog = qualla::Dialog::create(env, m_name, quallaConfig);
  m_sharedEngine = false;
  qualla::Timer start;

  if (!standaloneEnginesConfig["shared-engines"].empty()) {
    m_sharedEngineKeys = Registry::getKeysFromRegistry(standaloneEnginesConfig);
    auto enginesToBind =
        Registry::getEngineFromRegistry(m_sharedEngineKeys, profileStat, logger, env);
    m_quallaDialog->bindSharedEngines(enginesToBind);
    m_sharedEngine = true;
    m_quallaDialog->addSupplementInitTime(start.elapsed_usec());
  }

  if (!m_quallaDialog) {
    throw Exception(GENIE_STATUS_ERROR_MEM_ALLOC, "Could not create a dialog object");
  }
  if (m_quallaDialog->failed()) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL,
                    "Dialog create failed. Error: " + m_quallaDialog->error());
  }

  m_quallaDialog->validate();

  /*
   * spec-dec has a mandatory "primary" sampler and an optional "secondary" sampler
   * Check their availability and pass their references to Dialog Sampler to update with
   * applyConfig()
   */
  std::string samplerName = "sampler_" + m_name;
  std::shared_ptr<Sampler> sampler =
      std::make_shared<Sampler>(samplerName, m_quallaDialog->getSamplers());
  m_samplerHandle                                           = Sampler::add(sampler);
  std::reference_wrapper<qualla::Tokenizer> quallaTokenizer = m_quallaDialog->tokenizer();
  std::shared_ptr<Tokenizer> tokenizer = std::make_shared<Tokenizer>(quallaTokenizer);
  m_tokenizerHandle                    = Tokenizer::add(tokenizer);

  qualla::Dialog::KPIs kpis = m_quallaDialog->kpis();
  if (profileStat) profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_DIALOG_CREATE, kpis);
}
Dialog::Dialog(nlohmann::json& config,
               std::shared_ptr<ProfileStat> profileStat,
               std::shared_ptr<genie::log::Logger> logger,
               std::shared_ptr<genie::Profiler> profiler,
               std::shared_ptr<qualla::Env> env) {
  if (!env) {
    auto envNew = qualla::Env::create(std::make_shared<ResourceManager>(), nlohmann::json{});
    initDialog(config, envNew, profileStat, logger, profiler);
  } else {
    initDialog(config, env, profileStat, logger, profiler);
  }
}

Dialog::Dialog(std::shared_ptr<Config> config,
               std::shared_ptr<ProfileStat> profileStat,
               std::shared_ptr<genie::log::Logger> logger) {
  std::shared_ptr<genie::Profiler> profiler{nullptr};
  if (!config->getProfiler().empty()) {
    profiler = *(config->getProfiler().begin());
  }
  auto env =
      qualla::Env::create(config->getDlc() != nullptr ? config->getDlc()->getResourceManager()
                                                      : std::make_shared<ResourceManager>(),
                          nlohmann::json{});

  initDialog(config->getJson(), env, profileStat, logger, profiler);
}

GenieSampler_Handle_t Dialog::getSamplerHandle(std::shared_ptr<Dialog> dialog) {
  return dialog->m_samplerHandle;
}

GenieTokenizer_Handle_t Dialog::getTokenizerHandle(std::shared_ptr<Dialog> dialog) {
  return dialog->m_tokenizerHandle;
}

void Dialog::setStopSequence(const char* newStopSeqs) {
  nlohmann::json config;
  util::getJsonFromStr(newStopSeqs, config);

  // component is used in the "ENFORCE" macros
  const std::string component = "stop-sequence";
  for (auto& item : config.items()) {
    if (item.key() == "stop-sequence") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                          "stop-sequence must be an array of strings");
        }
      }
    }
  }
  m_quallaDialog->setStopSequence(config);
}

void Dialog::bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& logger) {
  for (auto it : logger) {
    it->incrementUseCount();
    m_logger.insert(it);
    m_quallaDialog->getEnv()->bindLogger(it);
  }
}

void Dialog::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& Dialog::getLogger() { return m_logger; }

void Dialog::bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler) {
  for (auto it : profiler) {
    if (it->m_traceLogger != m_quallaDialog->getTraceLogger()) {
      if (it->m_traceLogger && m_quallaDialog->getTraceLogger()) {
        throw Exception(GENIE_STATUS_ERROR_INVALID_ARGUMENT,
                        "Cannot bind multiple trace profilers to the same dialog.");
      } else {
        throw Exception(GENIE_STATUS_ERROR_INVALID_ARGUMENT,
                        "Cannot bind profilers with different configurations to the same Dialog.");
      }
    }
    it->incrementUseCount();
    m_profiler.insert(it);
    m_quallaDialog->setTraceLogger(it->m_traceLogger);
  }
}

void Dialog::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& Dialog::getProfiler() { return m_profiler; }

std::string Dialog::getName() { return m_name; }

static_assert(qualla::Sentence::Code::COMPLETE ==
              static_cast<qualla::Sentence::Code>(GENIE_DIALOG_SENTENCE_COMPLETE));
static_assert(qualla::Sentence::Code::BEGIN ==
              static_cast<qualla::Sentence::Code>(GENIE_DIALOG_SENTENCE_BEGIN));
static_assert(qualla::Sentence::Code::CONTINUE ==
              static_cast<qualla::Sentence::Code>(GENIE_DIALOG_SENTENCE_CONTINUE));
static_assert(qualla::Sentence::Code::END ==
              static_cast<qualla::Sentence::Code>(GENIE_DIALOG_SENTENCE_END));
static_assert(qualla::Sentence::Code::ABORT ==
              static_cast<qualla::Sentence::Code>(GENIE_DIALOG_SENTENCE_ABORT));
static_assert(qualla::Sentence::Code::REWIND ==
              static_cast<qualla::Sentence::Code>(GENIE_DIALOG_SENTENCE_REWIND));
static_assert(qualla::Sentence::Code::RESUME ==
              static_cast<qualla::Sentence::Code>(GENIE_DIALOG_SENTENCE_RESUME));

int32_t Dialog::signalAction(GenieDialog_Action_t action) {
  if (action == GENIE_DIALOG_ACTION_ABORT) {
    if (m_activeQuery != 0) {  // Only if there is an active query
      m_abort = true;
    }
    return GENIE_STATUS_SUCCESS;
  } else if (action == GENIE_DIALOG_ACTION_PAUSE) {
    m_pause = true;
    if (m_quallaDialog->supportsPauseResume()) {
      m_quallaDialog->pauseQuery();
    } else {
      return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
    }
    m_quallaDialog->pauseQuery();
    return GENIE_STATUS_SUCCESS;
  } else {
    return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

int32_t Dialog::query(const char* queryStr,
                      GenieDialog_SentenceCode_t sentenceCode,
                      GenieDialog_QueryCallback_t callback,
                      const void* userData,
                      std::shared_ptr<ProfileStat> profileStat) {
  if (m_sharedEngine) {
    if (!m_quallaDialog->markEnginesBusy()) {
      return GENIE_STATUS_ERROR_QUERY_FAILED;
    }
    m_quallaDialog->applyEnginesState();
  }
  m_activeQuery++;

  std::string query;
  if (queryStr == nullptr)
    query = '\0';
  else
    query = queryStr;
  uint32_t genTokenCount = 0u;
  bool status            = m_quallaDialog->query(
      query,
      static_cast<qualla::Sentence::Code>(sentenceCode),
      [&](const std::string& response, qualla::Sentence::Code code) {
        callback(response.c_str(), static_cast<GenieDialog_SentenceCode_t>(code), userData);
        bool keepGoing = ++genTokenCount < m_tokenLimit;
        keepGoing      = (!m_abort && keepGoing);
        if (!keepGoing && ((code == qualla::Sentence::Code::BEGIN) ||
                           (code == qualla::Sentence::Code::CONTINUE) ||
                           (code == qualla::Sentence::Code::RESUME))) {
          callback("", GENIE_DIALOG_SENTENCE_END, userData);
        }
        return keepGoing;
      });
  if (status) {
    qualla::Dialog::KPIs kpis = m_quallaDialog->kpis();
    if (profileStat) profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY, kpis);
  }
  m_activeQuery--;

  if (m_sharedEngine) {
    m_quallaDialog->markEnginesFree();
  }

  if (m_abort) {
    m_abort = false;  // Reset the abort signal.
    return status ? GENIE_STATUS_WARNING_ABORTED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  if (m_pause) {
    m_pause = false;
    return status ? GENIE_STATUS_WARNING_PAUSED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_QUERY_FAILED);
}

int32_t Dialog::query(const std::vector<const char*>& queryStr_vec,
                      GenieDialog_SentenceCode_t sentenceCode,
                      GenieDialog_QueryCallback_t callback,
                      const void** userData,
                      std::shared_ptr<ProfileStat> profileStat) {
  if (m_sharedEngine) {
    if (!m_quallaDialog->markEnginesBusy()) {
      return GENIE_STATUS_ERROR_QUERY_FAILED;
    }
    m_quallaDialog->applyEnginesState();
  }
  m_activeQuery++;

  std::vector<std::string> queries;
  queries.reserve(queryStr_vec.size());
  for (const char* s : queryStr_vec) {
    queries.emplace_back(s ? s : "");
  }
  uint32_t genTokenCount = 0u;
  bool status            = m_quallaDialog->query(
      queries,
      static_cast<qualla::Sentence::Code>(sentenceCode),
      [&](const std::vector<std::string>& responses,
          const std::vector<bool>& batchFinished,
          qualla::Sentence::Code code) {
        for (size_t i = 0; i < responses.size(); ++i) {
          if (!batchFinished[i]) {
            callback(const_cast<char*>(responses[i].c_str()),
                     static_cast<GenieDialog_SentenceCode_t>(code),
                     userData ? userData[i] : nullptr);
          }
        }

        bool keepGoing = ++genTokenCount < m_tokenLimit;
        keepGoing      = (!m_abort && keepGoing);
        if (!keepGoing && ((code == qualla::Sentence::Code::BEGIN) ||
                           (code == qualla::Sentence::Code::CONTINUE) ||
                           (code == qualla::Sentence::Code::RESUME))) {
          callback("", GENIE_DIALOG_SENTENCE_END, userData ? userData[0] : nullptr);
        }

        if (code == qualla::Sentence::Code::END) {
          callback("", GENIE_DIALOG_SENTENCE_END, userData ? userData[0] : nullptr);
        }
        return keepGoing;
      });
  if (status) {
    qualla::Dialog::KPIs kpis = m_quallaDialog->kpis();
    if (profileStat) profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY, kpis);
  }
  m_activeQuery--;

  if (m_sharedEngine) {
    m_quallaDialog->markEnginesFree();
  }

  if (m_abort) {
    m_abort = false;  // Reset the abort signal.
    return status ? GENIE_STATUS_WARNING_ABORTED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  if (m_pause) {
    m_pause = false;
    return status ? GENIE_STATUS_WARNING_PAUSED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_QUERY_FAILED);
}

int32_t Dialog::query(const char* queryStr,
                      GenieNode_TextOutput_SentenceCode_t sentenceCode,
                      GenieNode_TextOutput_Callback_t callback,
                      const void* userData,
                      std::shared_ptr<ProfileStat> profileStat) {
  if (m_sharedEngine) {
    if (!m_quallaDialog->markEnginesBusy()) {
      return GENIE_STATUS_ERROR_QUERY_FAILED;
    }
    m_quallaDialog->applyEnginesState();
  }
  m_activeQuery++;
  std::string query;
  if (queryStr == nullptr)
    query = '\0';
  else
    query = queryStr;
  uint32_t genTokenCount = 0u;
  bool status            = m_quallaDialog->query(
      query,
      static_cast<qualla::Sentence::Code>(sentenceCode),
      [&](const std::string& response, qualla::Sentence::Code code) {
        callback(
            response.c_str(), static_cast<GenieNode_TextOutput_SentenceCode_t>(code), userData);
        bool keepGoing = ++genTokenCount < m_tokenLimit;
        keepGoing      = (!m_abort && keepGoing);
        if (!keepGoing && ((code == qualla::Sentence::Code::BEGIN) ||
                           (code == qualla::Sentence::Code::CONTINUE) ||
                           (code == qualla::Sentence::Code::RESUME))) {
          callback("", GENIE_NODE_SENTENCE_END, userData);
        }
        return keepGoing;
      });
  if (status) {
    qualla::Dialog::KPIs kpis = m_quallaDialog->kpis();
    if (profileStat) profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY, kpis);
  }
  m_activeQuery--;

  if (m_sharedEngine) {
    m_quallaDialog->markEnginesFree();
  }

  if (m_abort) {
    m_abort = false;  // Reset the abort signal.
    return status ? GENIE_STATUS_WARNING_ABORTED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  if (m_pause) {
    m_pause = false;
    return status ? GENIE_STATUS_WARNING_PAUSED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_QUERY_FAILED);
}

int32_t Dialog::save(const std::string& name) {
  return m_quallaDialog->save(name) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_QUERY_FAILED);
}

int32_t Dialog::restore(const std::string& name) {
  return m_quallaDialog->restore(name) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_QUERY_FAILED);
}

int32_t Dialog::embeddingQuery(const void* embeddings,
                               const uint32_t embeddingsSize,
                               GenieDialog_SentenceCode_t sentenceCode,
                               GenieDialog_TokenToEmbeddingCallback_t t2eCallback,
                               GenieDialog_QueryCallback_t callback,
                               const void* userData,
                               std::shared_ptr<ProfileStat> profileStat) {
  if (m_sharedEngine) {
    if (!m_quallaDialog->markEnginesBusy()) {
      return GENIE_STATUS_ERROR_QUERY_FAILED;
    }
    m_quallaDialog->applyEnginesState();
  }

  m_activeQuery++;
  uint32_t genTokenCount = 0u;
  if (embeddingsSize % m_quallaDialog->getEmbeddingBufferSize() != 0) {
    throw std::runtime_error(
        "The embeddings buffer size must be an integer multiple of "
        "the embedding vector size in "
        "bytes.");
  }
  std::vector<uint8_t> embeddingVector;
  if (embeddings != nullptr) {
    const uint8_t* embeddingsSrc = static_cast<const uint8_t*>(embeddings);
    embeddingVector.assign(embeddingsSrc, embeddingsSrc + embeddingsSize);
  }
  qualla::Dialog::T2ECallback t2eQuallaCallback{nullptr};
  if (t2eCallback) {
    t2eQuallaCallback =
        [&](qualla::Dialog&, const int32_t token, void* embedding, const uint32_t embd_size) {
          t2eCallback(token, embedding, embd_size, userData);
        };
  }
  bool status = m_quallaDialog->query(
      embeddingVector,
      static_cast<qualla::Sentence::Code>(sentenceCode),
      t2eQuallaCallback,
      [&](const std::string& response, qualla::Sentence::Code code) {
        callback(response.c_str(), static_cast<GenieDialog_SentenceCode_t>(code), userData);
        bool keepGoing = ++genTokenCount < m_tokenLimit;
        keepGoing      = (!m_abort && keepGoing);
        if (!keepGoing && ((code == qualla::Sentence::Code::BEGIN) ||
                           (code == qualla::Sentence::Code::CONTINUE) ||
                           (code == qualla::Sentence::Code::RESUME))) {
          callback("", GENIE_DIALOG_SENTENCE_END, userData);
        }
        return keepGoing;
      });
  if (status) {
    qualla::Dialog::KPIs kpis = m_quallaDialog->kpis();
    if (profileStat) profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY, kpis);
  }
  m_activeQuery--;

  if (m_sharedEngine) {
    m_quallaDialog->markEnginesFree();
  }

  if (m_abort) {
    m_abort = false;  // Reset the abort signal.
    return status ? GENIE_STATUS_WARNING_ABORTED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  if (m_pause) {
    m_pause = false;
    return status ? GENIE_STATUS_WARNING_PAUSED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_QUERY_FAILED);
}

int32_t Dialog::embeddingQuery(const void* embeddings,
                               const uint32_t embeddingsSize,
                               GenieDialog_SentenceCode_t sentenceCode,
                               GenieDialog_TokenToEmbeddingCallback_t t2eCallback,
                               GenieDialog_TokenQueryCallback_t callback,
                               const void* userData,
                               std::shared_ptr<ProfileStat> profileStat) {
  if (m_sharedEngine) {
    if (!m_quallaDialog->markEnginesBusy()) {
      return GENIE_STATUS_ERROR_QUERY_FAILED;
    }
    m_quallaDialog->applyEnginesState();
  }

  m_activeQuery++;
  uint32_t genTokenCount = 0u;
  if (embeddingsSize % m_quallaDialog->getEmbeddingBufferSize() != 0) {
    throw std::runtime_error(
        "The embeddings buffer size must be an integer multiple of "
        "the embedding vector size in "
        "bytes.");
  }
  std::vector<uint8_t> embeddingVector;
  if (embeddings != nullptr) {
    const uint8_t* embeddingsSrc = static_cast<const uint8_t*>(embeddings);
    embeddingVector.assign(embeddingsSrc, embeddingsSrc + embeddingsSize);
  }
  qualla::Dialog::T2ECallback t2eQuallaCallback{nullptr};
  if (t2eCallback) {
    t2eQuallaCallback =
        [&](qualla::Dialog&, const int32_t token, void* embedding, const uint32_t embd_size) {
          t2eCallback(token, embedding, embd_size, userData);
        };
  }
  dialogCallback.setCallBackType(qualla::QUALLA_CALLBACK_TYPE_TOKEN);
  dialogCallback.getTokenCbFunc() = std::make_shared<
      std::function<bool(const int32_t*, const uint32_t, qualla::Sentence::Code)>>();
  *(dialogCallback.getTokenCbFunc()) = [&](const int32_t* responseTokens,
                                           const uint32_t sizeResponseTokens,
                                           qualla::Sentence::Code code) {
    callback(reinterpret_cast<const uint32_t*>(responseTokens),
             sizeResponseTokens,
             static_cast<GenieDialog_SentenceCode_t>(code),
             userData);
    bool keepGoing = ++genTokenCount < m_tokenLimit;
    keepGoing      = (!m_abort && keepGoing);
    if (!keepGoing &&
        ((code == qualla::Sentence::Code::BEGIN) || (code == qualla::Sentence::Code::CONTINUE) ||
         (code == qualla::Sentence::Code::RESUME))) {
      callback(nullptr, 0, GENIE_DIALOG_SENTENCE_END, userData);
    }
    return keepGoing;
  };
  bool status = m_quallaDialog->query(embeddingVector,
                                      static_cast<qualla::Sentence::Code>(sentenceCode),
                                      t2eQuallaCallback,
                                      dialogCallback);
  if (status) {
    qualla::Dialog::KPIs kpis = m_quallaDialog->kpis();
    if (profileStat) profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY, kpis);
  }
  m_activeQuery--;

  if (m_sharedEngine) {
    m_quallaDialog->markEnginesFree();
  }

  if (m_abort) {
    m_abort = false;  // Reset the abort signal.
    return status ? GENIE_STATUS_WARNING_ABORTED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  if (m_pause) {
    m_pause = false;
    return status ? GENIE_STATUS_WARNING_PAUSED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_QUERY_FAILED);
}

int32_t Dialog::embeddingQuery(const void* embeddings,
                               const uint32_t embeddingsSize,
                               GenieNode_TextOutput_SentenceCode_t sentenceCode,
                               GenieNode_TextOutput_Callback_t callback,
                               const void* userData,
                               std::shared_ptr<ProfileStat> profileStat) {
  if (m_sharedEngine) {
    if (!m_quallaDialog->markEnginesBusy()) {
      return GENIE_STATUS_ERROR_QUERY_FAILED;
    }
    m_quallaDialog->applyEnginesState();
  }

  m_activeQuery++;
  uint32_t genTokenCount = 0u;
  if (embeddingsSize % m_quallaDialog->getEmbeddingBufferSize() != 0) {
    throw std::runtime_error(
        "The embeddings buffer size must be an integer multiple of "
        "the embedding vector size in "
        "bytes.");
  }
  std::vector<uint8_t> embeddingVector;
  if (embeddings != nullptr) {
    const uint8_t* embeddingsSrc = static_cast<const uint8_t*>(embeddings);
    embeddingVector.assign(embeddingsSrc, embeddingsSrc + embeddingsSize);
  }
  bool status = m_quallaDialog->query(
      embeddingVector,
      static_cast<qualla::Sentence::Code>(sentenceCode),
      nullptr,
      [&](const std::string& response, qualla::Sentence::Code code) {
        callback(
            response.c_str(), static_cast<GenieNode_TextOutput_SentenceCode_t>(code), userData);
        bool keepGoing = ++genTokenCount < m_tokenLimit;
        keepGoing      = (!m_abort && keepGoing);
        if (!keepGoing && ((code == qualla::Sentence::Code::BEGIN) ||
                           (code == qualla::Sentence::Code::CONTINUE) ||
                           (code == qualla::Sentence::Code::RESUME))) {
          callback("", GENIE_NODE_SENTENCE_END, userData);
        }
        return keepGoing;
      });
  if (status) {
    qualla::Dialog::KPIs kpis = m_quallaDialog->kpis();
    if (profileStat) profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY, kpis);
  }
  m_activeQuery--;
  if (m_sharedEngine) {
    m_quallaDialog->markEnginesFree();
  }
  if (m_abort) {
    m_abort = false;  // Reset the abort signal.
    return status ? GENIE_STATUS_WARNING_ABORTED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  if (m_pause) {
    m_pause = false;
    return status ? GENIE_STATUS_WARNING_PAUSED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_QUERY_FAILED);
}

std::vector<float> Dialog::train(qualla::TrainingData& trainingData) {
  return m_quallaDialog->train(trainingData);
}

void Dialog::reset() { m_quallaDialog->reset(); }

int32_t Dialog::applyLora(std::string loraAdapterName,
                          std::string engineRole,
                          std::shared_ptr<ProfileStat> profileStat) {
  std::string role = Engine::changeRole(engineRole);
  bool status      = m_quallaDialog->applyLoraAdapter(loraAdapterName, role);
  if (status) {
    qualla::Dialog::KPIs kpis = m_quallaDialog->kpis();
    if (profileStat)
      profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_DIALOG_APPLY_LORA, kpis);
  }
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
}

int32_t Dialog::applyLoraStrength(std::string tensorName, std::string engineRole, float alpha) {
  std::string role = Engine::changeRole(engineRole);
  bool status      = m_quallaDialog->applyLoraStrength(tensorName, alpha, role);
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
}

int32_t Dialog::releaseLoraMemory(std::string loraAdapterName, std::string engineRole) {
  std::string role = Engine::changeRole(engineRole);
  bool status = m_quallaDialog->releaseLoraAdapter(loraAdapterName, role);
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
}

std::string Dialog::getAppliedLoraAdapter(const std::string& engineRole) const {
  std::string role = Engine::changeRole(engineRole);
  return m_quallaDialog->getAppliedLoraAdapter(role);
}

int32_t Dialog::tokenQuery(const uint32_t* tokens,
                           const uint32_t sizeInputTokens,
                           GenieDialog_SentenceCode_t sentenceCode,
                           GenieDialog_TokenQueryCallback_t callback,
                           const void* userData,
                           std::shared_ptr<ProfileStat> profileStat) {
  m_activeQuery++;
  std::vector<uint32_t> inputTokens;
  for (size_t i = 0; i < sizeInputTokens; i++) {
    inputTokens.push_back(tokens[i]);
  }
  uint32_t genTokenCount = 0u;
  dialogCallback.setCallBackType(qualla::QUALLA_CALLBACK_TYPE_TOKEN);
  dialogCallback.getTokenCbFunc() = std::make_shared<
      std::function<bool(const int32_t*, const uint32_t, qualla::Sentence::Code)>>();
  *(dialogCallback.getTokenCbFunc()) = [&](const int32_t* responseTokens,
                                           const uint32_t sizeResponseTokens,
                                           qualla::Sentence::Code code) {
    callback(reinterpret_cast<const uint32_t*>(responseTokens),
             sizeResponseTokens,
             static_cast<GenieDialog_SentenceCode_t>(code),
             userData);
    bool keepGoing = ++genTokenCount < m_tokenLimit;
    keepGoing      = (!m_abort && keepGoing);
    if (!keepGoing &&
        ((code == qualla::Sentence::Code::BEGIN) || (code == qualla::Sentence::Code::CONTINUE) ||
         (code == qualla::Sentence::Code::RESUME))) {
      callback(nullptr, 0, GENIE_DIALOG_SENTENCE_END, userData);
    }
    return keepGoing;
  };
  bool status = m_quallaDialog->query(
      inputTokens, static_cast<qualla::Sentence::Code>(sentenceCode), dialogCallback);
  if (status) {
    qualla::Dialog::KPIs kpis = m_quallaDialog->kpis();
    if (profileStat) profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY, kpis);
  }
  m_activeQuery--;
  if (m_abort) {
    m_abort = false;  // Reset the abort signal.
    return status ? GENIE_STATUS_WARNING_ABORTED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  if (m_pause) {
    m_pause = false;
    return status ? GENIE_STATUS_WARNING_PAUSED : GENIE_STATUS_ERROR_QUERY_FAILED;
  }
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_QUERY_FAILED);
}

int32_t Dialog::setPriority(const std::string engineRole, const GenieDialog_Priority_t priority) {
  std::string role = Engine::changeRole(engineRole);
  bool status      = m_quallaDialog->setExecutionPriority(role, static_cast<uint32_t>(priority));
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
}

int32_t Dialog::setOemkey(const std::string& oemKey) {
  bool status = m_quallaDialog->setOemKey(oemKey);
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
}

int32_t Dialog::bindEngine(const std::string& engineRole,
                           std::shared_ptr<Engine> engine,
                           std::shared_ptr<ProfileStat> profileStat) {
  std::string role = Engine::changeRole(engineRole);
  if (role != "secondary") {
    throw Exception(GENIE_STATUS_ERROR_INVALID_ARGUMENT,
                    "specified " + role +
                        " engine can't be bound to a dialog. Currently only draft binding and "
                        "switching is allowed.");
  }
  bool status = m_quallaDialog->bindEngine(role, engine->getEngine());
  if (status) {
    qualla::Dialog::KPIs kpis = m_quallaDialog->kpis();
    if (profileStat)
      profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_DIALOG_BINDENGINE, kpis);
  }
  return status ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
}

GenieEngine_Handle_t Dialog::getEngineHandle(const std::string& engineRole,
                                             std::shared_ptr<ProfileStat> profileStat) {
  std::string role = Engine::changeRole(engineRole);
  auto engine      = m_quallaDialog->getEngine(role);
  if (!engine) {
    throw Exception(GENIE_STATUS_ERROR_GET_HANDLE_FAILED,
                    "Specified " + role + " engine is not associated with dialog.");
  }
  std::string engineName = "engine_" + m_name;
  auto engineHandle      = genie::Engine::add(std::make_shared<Engine>(engine, engineName));
  if (engineHandle != nullptr) {
    qualla::Dialog::KPIs kpis = m_quallaDialog->kpis();
    if (profileStat)
      profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_DIALOG_GETENGINE, kpis);
  }
  return engineHandle;
}

int32_t Dialog::getDimensions(qualla::LayerType layerType,
                              std::vector<uint32_t>& dimensions) {
  if (!m_quallaDialog) {
    return GENIE_STATUS_ERROR_GENERAL;
  }

  try {
    m_quallaDialog->getDimensions(dimensions, layerType);
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return GENIE_STATUS_SUCCESS;
}

int32_t Dialog::getInputQuantParam(std::string& dataType,
                                   double& scale,
                                   int32_t& offset,
                                   size_t& byteWidth,
                                   bool isCrossAttention) {
  m_quallaDialog->inputTensorQuantParam(dataType, scale, offset, byteWidth, isCrossAttention);
  return GENIE_STATUS_SUCCESS;
}

void Dialog::setPerformancePolicy(const Genie_PerformancePolicy_t policy) {
  m_quallaDialog->setPerformancePolicy(static_cast<qualla::PerformanceProfile>(policy));
}

const Genie_PerformancePolicy_t& Dialog::getPerformancePolicy() {
  m_performancePolicy =
      static_cast<Genie_PerformancePolicy_t>(m_quallaDialog->getPerformancePolicy());
  return m_performancePolicy;
}

void Dialog::setMaxNumTokens(const uint32_t maxNumTokens) { m_tokenLimit = maxNumTokens; }

bool Dialog::setVisionParam(const std::vector<uint32_t>& visionParam) {
  return m_quallaDialog->setVisionParam(visionParam);
}

bool Dialog::setCrossAttentionHiddenStates(const void* crossAttentionEmbeddings,
                                           const uint32_t embeddingsSize) {
  std::vector<float> crossAttentionVector;
  if (crossAttentionEmbeddings != nullptr) {
    const float* embeddingsSrc = static_cast<const float*>(crossAttentionEmbeddings);
    crossAttentionVector.assign(embeddingsSrc, embeddingsSrc + embeddingsSize / sizeof(float));
  }
  return m_quallaDialog->setCrossAttentionHiddenStates(crossAttentionVector);
}

void Dialog::registerModelAdaptor(std::shared_ptr<qualla::ModelIOAdaptor> adaptor) {
  m_quallaDialog->registerModelAdaptor(adaptor);
}

Dialog::~Dialog() {
  if (m_sharedEngine) {
    Registry::deleteEnginesFromRegistry(m_sharedEngineKeys);
  }
  Tokenizer::remove(m_tokenizerHandle);
  Sampler::remove(m_samplerHandle);
}
