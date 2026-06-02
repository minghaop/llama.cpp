//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>
#include <unordered_set>

#include "nlohmann/json.hpp"
#include "qualla/detail/state.hpp"
#include "qualla/env.hpp"

namespace qualla {

class Context : public State {
 public:
  QUALLA_API Context(std::shared_ptr<Env> env,
                     const std::string& name,
                     const nlohmann::json& conf);

  std::shared_ptr<Env> env() { return _env; }

  const std::string& name() const { return _name; }
  size_t size() const { return _size; }
  size_t n_ctx() const { return _size; }
  size_t n_vocab() const { return _n_vocab; }
  size_t draft_n_vocab() const { return _draft_n_vocab; }
  bool is_trimmed_vocab() const { return _n_vocab != _draft_n_vocab; }
  size_t n_embd() const { return _n_embd; }

  int32_t bos_tok() const { return _bos_tok; }
  int32_t bos() const { return _bos_tok; }
  int32_t eos_tok() const { return _eos_tok; }
  int32_t eos() const { return _eos_tok; }
  bool is_eos(int32_t tok) const { return _eos_tok_list.find(tok) != _eos_tok_list.end(); }
  int32_t pad_tok() const { return _pad_tok; }
  int32_t pad() const { return _pad_tok; }
  int32_t img_tok() const { return _img_tok; }
  int32_t img() const { return _img_tok; }

  int32_t embeddingLength() const { return _embedding_length; }
  int32_t featureLength() const { return _feature_length; }
  const std::string& embeddingDatatype() const { return _embedding_datatype; }

  const nlohmann::json& conf() const { return _conf; }

  QUALLA_API static std::shared_ptr<Context> create(std::shared_ptr<Env> env,
                                                    const std::string& name,
                                                    const nlohmann::json& conf = {});
  QUALLA_API static std::shared_ptr<Context> create(std::shared_ptr<Env> env,
                                                    const std::string& name,
                                                    std::istream& json_stream);
  QUALLA_API static std::shared_ptr<Context> create(std::shared_ptr<Env> env,
                                                    const std::string& name,
                                                    const std::string& json_str);

 private:
  const std::string _name;    // Contex name
  std::shared_ptr<Env> _env;  // Reference to global env
  nlohmann::json _conf;       // Complete context config

  size_t _size{1024};             // Context size
  size_t _n_vocab{32000};         // Vocab size
  size_t _draft_n_vocab{32000};   // draft Vocab size
  size_t _n_embd{4096};           // Word embedding size
  int32_t _bos_tok{1};            // BOS token id
  int32_t _eos_tok{-1};           // EOS token id
  int32_t _pad_tok{-1};           // Pad token id
  int32_t _img_tok{-1};           // image token id
  int32_t _embedding_length{-1};  // Embedding vector length. Required for E2T
  int32_t _feature_length{0};     // feature vector length, required for Eaglet

  // List of EOS tokens to stop generation
  std::unordered_set<int32_t> _eos_tok_list;

  // E2T query input datatype. "float32" or "native".
  std::string _embedding_datatype{"QNN_DATATYPE_FLOAT_32"};
};

}  // namespace qualla
