//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <sstream>

#include "nlohmann/json.hpp"

namespace qualla {

struct Config {
  const nlohmann::json& json;
  const std::string pref;

  Config(const nlohmann::json& j, const std::string& p = "qualla:")
      : json(j), pref(p) {}

  // Optional value, returns the default if the key is not found.
  template <typename T>
  T optional(const std::string& k, T d) {
    return json.contains(k) ? json[k].get<T>() : d;
  }

  // Mandatory value, throws runtime_error if the key is not found.
  template <typename T>
  T mandatory(const std::string& k) {
    if (json.contains(k)) return json[k].get<T>();
    std::stringstream ss;
    ss << pref << " mandatory config key : (" << k << ") not found in : " << json << std::endl;
    throw std::runtime_error(ss.str());
  }

  // Optional value, returns the default if the key is not found.
  template <typename T>
  static inline T optional(const nlohmann::json& j, const std::string& k, T d) {
    return j.contains(k) ? j[k].get<T>() : d;
  }

  // Mandatory value, throws runtime_error if the key is not found.
  template <typename T>
  static inline T mandatory(const nlohmann::json& j, const std::string& k) {
    if (j.contains(k)) return j[k].get<T>();
    std::stringstream ss;
    ss << "qualla: mandatory config key : (" << k << ") not found in : " << j << std::endl;
    throw std::runtime_error(ss.str());
  }
};

}  // namespace qualla
