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

#include "Context.hpp"
#include "Exception.hpp"
#include "Macro.hpp"

using namespace genie;

//=============================================================================
// Context functions
//=============================================================================

std::atomic<std::uint32_t> Context::s_nameCounter{0u};

//=============================================================================
// Context::Config functions
//=============================================================================

void Context::ContextConfig::validateContextConfig(const nlohmann::json& config) {
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
    } else if (item.key() == "img-token") {
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

void Context::ContextConfig::translateContextConfig(const nlohmann::json& genieConfig,
                                                    nlohmann::json& quallaConfig) {
  if (genieConfig.contains("context")) {
    if (genieConfig["context"].contains("bos-token")) {
      quallaConfig["context"]["bos-token"] = genieConfig["context"]["bos-token"];
    }
    if (genieConfig["context"].contains("eos-token")) {
      quallaConfig["context"]["eos-token"] = genieConfig["context"]["eos-token"];
    }
    if (genieConfig["context"].contains("eot-token")) {
      quallaConfig["context"]["eot-token"] = genieConfig["context"]["eot-token"];
    }
    if (genieConfig["context"].contains("img-token")) {
      quallaConfig["context"]["img-token"] = genieConfig["context"]["img-token"];
    }
    if (genieConfig["context"].contains("size")) {
      quallaConfig["context"]["size"] = genieConfig["context"]["size"];
    }
    if (genieConfig["context"].contains("n-vocab")) {
      quallaConfig["context"]["n-vocab"] = genieConfig["context"]["n-vocab"];
    }
    if (genieConfig["context"].contains("draft-n-vocab")) {
      quallaConfig["context"]["draft-n-vocab"] = genieConfig["context"]["draft-n-vocab"];
    }
    if (genieConfig["context"].contains("pad-token")) {
      quallaConfig["context"]["pad-token"] = genieConfig["context"]["pad-token"];
    }
    if (genieConfig["context"].contains("n-embd")) {
      quallaConfig["context"]["n-embd"] = genieConfig["context"]["n-embd"];
    }
    if (genieConfig["context"].contains("embedding-length")) {
      quallaConfig["context"]["embedding-length"] = genieConfig["context"]["embedding-length"];
    }
  }
}

std::shared_ptr<qualla::Context> Context::getQuallaContext() { return m_quallaContext; }

Context::Context(const nlohmann::json& config, std::shared_ptr<qualla::Env> env) {
  m_env = env;
  nlohmann::json quallaContextConfig;
  ContextConfig::translateContextConfig(config, quallaContextConfig);
  m_name          = "context" + std::to_string(s_nameCounter.fetch_add(1u));
  m_quallaContext = qualla::Context::create(m_env, m_name, quallaContextConfig["context"]);
  if (!m_quallaContext) {
    throw Exception(GENIE_STATUS_ERROR_MEM_ALLOC, "Could not create a context object");
  }
}

Context::~Context() {}

nlohmann::json& Context::ContextConfig::getJson() { return m_config; }
