//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#include <set>

#include "Exception.hpp"
#include "Util.hpp"
#include "nlohmann/json.hpp"

namespace genie {
namespace util {

/**
 * @brief Check if a JSON field is considered non-empty
 * @param v The JSON value to check
 * @return true if the field is non-empty, false otherwise
 */
bool isJsonFieldNonEmpty(const nlohmann::json& v) {
  if (v.is_null()) return false;
  if (v.is_object() || v.is_array()) {
    return !v.empty();
  }
  if (v.is_string()) {
    return !v.get_ref<const std::string&>().empty();
  }
  // numbers, booleans, etc. are always considered �non-empty�
  return true;
}

/**
 * @brief Recursively overwrite configuration values
 * @param dlcConfig The destination configuration to be overwritten
 * @param jsonConfig The source configuration to overwrite with
 */
void recursiveConfigOverwrite(nlohmann::json& dlcConfig, const nlohmann::json& jsonConfig) {
  // Base case: if jsonConfig is not an object, overwrite dlcConfig with jsonConfig
  if (!dlcConfig.is_object() || !jsonConfig.is_object()) {
    // If either side isn't an object, treat jsonConfig as authoritative
    if (isJsonFieldNonEmpty(jsonConfig)) {
      dlcConfig = nlohmann::json::parse(jsonConfig.dump());
    }
    return;
  }

  for (auto it = jsonConfig.begin(); it != jsonConfig.end(); ++it) {
    const auto& key = it.key();
    const auto& val = it.value();

    auto dst_it = dlcConfig.find(key);
    if (dst_it == dlcConfig.end()) {
      // New key: just copy
      dlcConfig[key] = val;
    } else {
      // Existing key: recurse for objects, otherwise overwrite
      if (dst_it->is_object() && val.is_object()) {
        recursiveConfigOverwrite(*dst_it, val);
      } else {
        *dst_it = nlohmann::json::parse(val.dump());
      }
    }
  }
}

/**
 * @brief Overwrite configuration with JSON string
 * @param config The configuration to overwrite
 * @param jsonStr The JSON string to overwrite with
 */
void overwriteConfig(nlohmann::json& configToBeOverwritten, const char* jsonStr) {
  nlohmann::json config;
  if (jsonStr == nullptr) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Input Json to overwrite the DLC is empty");
  }
  getJsonFromStr(jsonStr, config);
  // TODO: Add verification for the non-model specific options
  if (!config.empty() && !configToBeOverwritten.empty()) {
    recursiveConfigOverwrite(configToBeOverwritten, config);
  }
}

/**
 * @brief Retrieve qualla JSON from config string
 * @param configStr The config string
 * @param jsonStr The output qualla JSON
 */
void getJsonFromStr(const char* configStr, nlohmann::json& config) {
  if (!configStr) {
    throw genie::Exception(GENIE_STATUS_ERROR_INVALID_ARGUMENT, "Config buffer cannot be null");
  }
  {
    std::set<nlohmann::json> keys;

    auto callback = [&keys](int depth, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
      if ((depth == 1) && (event == nlohmann::json::parse_event_t::key)) {
        if (keys.count(parsed) > 0) {
          throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                          "Multiple top level config key: " + parsed.dump());
        }
        keys.insert(parsed);
      }
      return true;
    };
    config = nlohmann::json::parse(configStr, callback);
  }

  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Top level config is not an object");
  }
}

/**
 * @brief Retrieve qualla JSON from a config record in DLC
 * @param dlc The DLC that holds the config record
 * @param recordName The name of the config record
 * @param parsedJson The output qualla JSON
 */
void getJsonFromDlc(std::shared_ptr<Dlc>& dlc,
                    const std::string& recordName,
                    nlohmann::json& parsedJson) {
  std::shared_ptr<const uint8_t[]> recordBuffer;
  uint64_t recordBufferSize{};
  dlc->getRecordBuffer(recordName, recordBuffer, &recordBufferSize);

  util::getJsonFromStr(reinterpret_cast<const char*>(recordBuffer.get()), parsedJson);
}

/**
 * @brief Sanitize qualla JSON from a json record in DLC
 * @param buffer     The raw input buffer
 * @param bufferSize The size of the input buffer
 * @param outBuffer  Output shared_ptr that will own the sanitized, null-terminated copy
 * @return           The number of sanitized bytes (excluding the null terminator)
 */
uint64_t sanitizeJsonBuffer(const uint8_t* buffer,
                            uint64_t bufferSize,
                            std::shared_ptr<uint8_t[]>& outBuffer) {
  std::string sanitized(reinterpret_cast<const char*>(buffer), bufferSize);

  // 1. Remove all carriage-return characters (\r) to normalise CRLF -> LF.
  sanitized.erase(std::remove(sanitized.begin(), sanitized.end(), '\r'), sanitized.end());

  // 2. Truncate anything after the last '}' or ']' to drop trailing garbage bytes.
  auto lastValid = sanitized.find_last_of("}]");
  if (lastValid != std::string::npos) {
    sanitized.erase(lastValid + 1);
  }

  uint64_t sanitizedSize = static_cast<uint64_t>(sanitized.size());
  outBuffer              = std::shared_ptr<uint8_t[]>(new uint8_t[sanitizedSize + 1]);
  std::memcpy(outBuffer.get(), sanitized.c_str(), sanitizedSize);
  outBuffer.get()[sanitizedSize] = '\0';

  return sanitizedSize;
}

}  // namespace util
}  // namespace genie