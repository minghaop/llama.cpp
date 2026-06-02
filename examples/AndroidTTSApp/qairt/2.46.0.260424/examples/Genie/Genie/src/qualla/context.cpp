//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include "ResourceManager.hpp"
#include "qualla/context.hpp"
#include "qualla/detail/config.hpp"

namespace qualla {

Context::Context(std::shared_ptr<Env> env, const std::string& name, const nlohmann::json& json)
    : _name(name), _env(env), _conf(json) {
  __DEBUG("ctx-new: {} config {}", _name, _conf.dump());

  qualla::Config conf(json, "context:");
  _size               = conf.optional<size_t>("size", 1024);
  _size               = conf.optional<size_t>("n-ctx", _size);
  _n_vocab            = conf.optional<size_t>("n-vocab", 32000);
  _draft_n_vocab      = conf.optional<size_t>("draft-n-vocab", _n_vocab);
  _n_embd             = conf.optional<size_t>("n-embd", 1024);
  _bos_tok            = conf.optional<int32_t>("bos-token", -1);
  _img_tok            = conf.optional<int32_t>("img-token", -1);
  _embedding_length   = conf.optional<int32_t>("embedding-length", -1);
  _embedding_datatype = conf.optional<std::string>("embedding-datatype", "QNN_DATATYPE_FLOAT_32");

  // For backward compatibility. When eot-token is removed, this logic can be simplified
  // Currently, EOT is marked as default truncating token if available
  int32_t eot_tok = conf.optional<int32_t>("eot-token", -1);
  if (eot_tok >= 0) _eos_tok_list.insert(eot_tok);

  const nlohmann::json eos_conf = conf.optional<nlohmann::json>("eos-token", _eos_tok);
  if (eos_conf.is_array() && eos_conf.size() > 0) {
    const std::vector<int32_t>& eos_tokens = eos_conf.get<std::vector<int32_t>>();
    _eos_tok                               = eos_tokens[0];
    for (const int32_t& eos_tok : eos_tokens) _eos_tok_list.insert(eos_tok);
  } else if (eos_conf.is_number_integer()) {
    int32_t eos_tok = eos_conf.get<int32_t>();
    _eos_tok        = (eot_tok >= 0) ? eot_tok : eos_tok;
    _eos_tok_list.insert(eos_tok);
  }

  _pad_tok = conf.optional<nlohmann::json>("pad-token", _eos_tok);
}

std::shared_ptr<Context> Context::create(std::shared_ptr<Env> env,
                                         const std::string& name,
                                         const nlohmann::json& conf) {
  return std::make_shared<Context>(env, name, conf);
}

std::shared_ptr<Context> Context::create(std::shared_ptr<Env> env,
                                         const std::string& name,
                                         std::istream& json_stream) {
  return create(env, name, nlohmann::json::parse(json_stream));
}

std::shared_ptr<Context> Context::create(std::shared_ptr<Env> env,
                                         const std::string& name,
                                         const std::string& json_str) {
  return create(env, name, nlohmann::json::parse(json_str));
}

}  // namespace qualla
