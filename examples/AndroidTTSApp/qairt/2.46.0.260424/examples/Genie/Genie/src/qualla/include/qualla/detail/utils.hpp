//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <string>

#include "nlohmann/json.hpp"

// Loop unrolling macros
#if defined(__clang__)
#define PRAGMA_LOOP_VECTORIZE _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define PRAGMA_LOOP_VECTORIZE _Pragma("GCC ivdep")
#elif defined(_MSC_VER)
#define PRAGMA_LOOP_VECTORIZE __pragma(loop(hint_parallel(4)))
#else
#define PRAGMA_LOOP_VECTORIZE  // Compiler does not support loop unrolling
#endif

namespace qualla {
namespace utils {

/**
 * @brief Remove leading and trailing whitespace from a string
 * @param str The input string to trim
 * @return A new string with leading and trailing whitespace removed
 */
std::string trim(const std::string& str);

/**
 * @brief Remove leading and trailing whitespace from a json string value
 * @param jsonValue The input json value to trim
 * @return A new string with leading and trailing whitespace removed, or empty string if not a string
*/
std::string trim(const nlohmann::json& jsonValue);

/**
* @brief Remove leading whitespace from a string
* @param str The input string to trim
* @return A new string with leading whitespace removed
 */
std::string ltrim(const std::string& str);

/**
* @brief Remove trailing whitespace from a string
* @param str The input string to trim
* @return A new string with trailing whitespace removed
 */
std::string rtrim(const std::string& str);

}   //namespace utils
}   //namespace qualla
