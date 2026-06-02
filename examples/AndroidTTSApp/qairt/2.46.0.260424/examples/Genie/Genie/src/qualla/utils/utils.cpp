//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>
#include <cctype>

#include "qualla/detail/utils.hpp"

namespace qualla {
namespace utils {

std::string ltrim(const std::string& str) {
    auto start = std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    });
    return std::string(start, str.end());
}

std::string rtrim(const std::string& str) {
    auto end = std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base();
    return std::string(str.begin(), end);
}

std::string trim(const std::string& str) {
    return ltrim(rtrim(str));
}

std::string trim(const nlohmann::json& jsonValue) {
    if (jsonValue.is_string()) {
        return trim(jsonValue.get<std::string>());
    }
    return "";
}

} // namespace utils
} // namespace qualla
