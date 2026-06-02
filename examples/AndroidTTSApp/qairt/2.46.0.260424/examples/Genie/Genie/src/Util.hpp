//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>

#include "Dlc.hpp"
#include "nlohmann/json.hpp"
#include "qualla/env.hpp"

namespace genie {
namespace util {

/**
 * @brief Check if a JSON field is considered non-empty
 * @param v The JSON value to check
 * @return true if the field is non-empty, false otherwise
 */
bool isJsonFieldNonEmpty(const nlohmann::json& v);

/**
 * @brief Recursively overwrite configuration values
 * @param dlcConfig The destination configuration to be overwritten
 * @param jsonConfig The source configuration to overwrite with
 */
void recursiveConfigOverwrite(nlohmann::json& dlcConfig, const nlohmann::json& jsonConfig);

/**
 * @brief Overwrite configuration with JSON string
 * @param configToBeOverwritten The configuration to overwrite
 * @param jsonStr The JSON string to overwrite with
 */
void overwriteConfig(nlohmann::json& configToBeOverwritten, const char* jsonStr);

/**
 * @brief Retrieve qualla JSON from config string
 * @param configStr The config string
 * @param jsonStr The output qualla JSON
 */
void getJsonFromStr(const char* configStr, nlohmann::json& config);

/**
 * @brief Retrieve qualla JSON from a config record in DLC
 * @param dlc The DLC that holds the config record
 * @param recordName The name of the config record
 * @param parsedJson The output qualla JSON
 */
void getJsonFromDlc(std::shared_ptr<Dlc>& dlc,
                    const std::string& recordName,
                    nlohmann::json& parsedJson);

/**
 * @brief Sanitize a raw JSON buffer: strips CR characters and trailing garbage,
 *        then writes the result into a null-terminated, shared-owned buffer.
 * @param buffer     The raw input buffer
 * @param bufferSize The size of the input buffer
 * @param outBuffer  Output shared_ptr that will own the sanitized, null-terminated copy
 * @return           The number of sanitized bytes (excluding the null terminator)
 */
uint64_t sanitizeJsonBuffer(const uint8_t* buffer,
                            uint64_t bufferSize,
                            std::shared_ptr<uint8_t[]>& outBuffer);

}  // namespace util
}  // namespace genie