//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_EMBEDDING_HPP
#define QUALLA_EMBEDDING_HPP

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "qualla/context.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/sentence.hpp"
#include "qualla/encoder.hpp"
#include "qualla/engine.hpp"
#include "qualla/env.hpp"
#include "qualla/tokenizer.hpp"

namespace qualla {

class Embedding : public Encoder {
 public:
  static constexpr const char* TYPE = "basicTextEncoder";

  Embedding(std::shared_ptr<Env> env, const nlohmann::json& conf);
  virtual ~Embedding();

  // Encode sentence
  virtual bool query(const std::string& str, std::vector<uint8_t>& output);

  virtual bool encode(const std::string& str,
                      std::vector<uint8_t>& output,
                      std::vector<int32_t>& tokenizedInput);

  // Get refs to various layers
  Context& context() { return *_ctx; }
  Tokenizer& tokenizer() { return *_tokenizer; }

  // Get latest KPIs.
  // Updates TPS, etc as needed.
  virtual KPIs& kpis();

  // Get output dimensions
  void output_dimensions(std::vector<std::uint32_t>& outputDimensions);

  void outputTensorQuantParam(std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              float& byteWidth);

 protected:
  std::shared_ptr<Context> _ctx;
  std::shared_ptr<Tokenizer> _tokenizer;

  bool _input_truncation;

  std::vector<std::string> _tags;

  std::vector<std::uint32_t> _output_dimensions{};

  uint32_t _n_queries{0};  // number of queries
  uint32_t _n_prompt{0};   // number of prompt tokens

  // Some encoders may have a sub-encoder stage,
  // e.g. basic encoder models with a separate LUT binary.
  std::unique_ptr<Encoder> m_subEncoder;

  template <typename T>
  bool process(std::vector<T>& tokens, std::vector<float>& output);

 private:
  void requantEmbedding(void* from, void* to, size_t length);
  void calculateRequantEncodings();

  std::string inputDataType{"QNN_DATATYPE_FLOAT_32"};
  double inputScale{1.0};
  int32_t inputOffset{0};
  size_t inputByteWidth{4};

  std::string lutDataType{"QNN_DATATYPE_FLOAT_32"};
  double lutScale{1.0};
  int32_t lutOffset{0};
  // lutByteWidth can be non integer, i.e., 0.5 for uint4 data type.
  float lutByteWidth{4.0};

  double requantScale{1.0};
  int32_t requantOffset{0};
};

}  // namespace qualla

#endif  // QUALLA_DIALOG_HPP
