//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_LUT_HPP
#define QUALLA_LUT_HPP

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

class LUT : public Encoder {
 public:
  static constexpr const char* TYPE = "lut";

  LUT(std::shared_ptr<Env> env, const nlohmann::json& conf);

  virtual ~LUT();

  virtual bool encode(const std::string& str,
                      std::vector<uint8_t>& output,
                      std::vector<int32_t>& tokenizedInput) override;

  virtual bool encode(const std::vector<int32_t>& tokens, std::vector<uint8_t>& output) override;

  size_t getEmbeddingLutSize() override;

  void* getEmbeddingLut() override;

  virtual int32_t getLastToken() override;

  void outputTensorQuantParam(std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              float& byteWidth) override;
  // Get output dimensions
  void output_dimensions(std::vector<std::uint32_t>& outputDimensions) override;

 protected:
  std::shared_ptr<Tokenizer> _tokenizer;
  std::shared_ptr<Context> _ctx;
  std::vector<std::string> _tags;
  std::string embedding_file_path = "";
  std::string inputDataType       = "QNN_DATATYPE_FLOAT_32";
  size_t bitWidth                 = 8;
  double lutScale{1.0};
  int32_t lutOffset{0};
  bool _input_truncation;
  std::vector<std::uint32_t> _output_dimensions{};
  size_t embeddingLutSize{0};
  void* embeddingLut;
  uint32_t _n_queries{0};  // number of queries
  uint32_t _n_prompt{0};   // number of prompt tokens
  std::shared_ptr<uint8_t> m_embeddingBuffer;
  int32_t lastToken{0};
};

}  // namespace qualla

#endif  // QUALLA_DIALOG_HPP
