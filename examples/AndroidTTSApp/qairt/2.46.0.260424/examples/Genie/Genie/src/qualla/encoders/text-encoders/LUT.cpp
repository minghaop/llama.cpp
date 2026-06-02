//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>

#include "LUT.hpp"
#include "ResourceManager.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/timer.hpp"

namespace fs = std::filesystem;

namespace qualla {

LUT::LUT(std::shared_ptr<Env> env, const nlohmann::json& json) : Encoder(env, "lut", json) {
  Timer start;
  __DEBUG("LUT-new: {} config {}", _type, json.dump());

  using qc = qualla::Config;

  // Parse prompt config
  embedding_file_path = qc::optional<std::string>(json, "lut-path", "");

  if (json["context"].contains("quant-param")) {
    lutScale  = json["context"]["quant-param"]["scale"];
    lutOffset = json["context"]["quant-param"]["offset"];
  }
  const nlohmann::json& pmt_conf = qc::optional<nlohmann::json>(json, "prompt", {});
  _tags = qc::optional<std::vector<std::string>>(pmt_conf, "tags", {"", ""});

  // Create the context first
  _ctx = Context::create(_env, _type, qc::optional<nlohmann::json>(json, "context", {}));

  // Load LUT
  inputDataType = _ctx->embeddingDatatype();
  if (inputDataType == "QNN_DATATYPE_FLOAT_32") {
    bitWidth = 32;
  } else if (inputDataType == "QNN_DATATYPE_SFIXED_POINT_16" ||
             inputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
    bitWidth = 16;
  } else if (inputDataType == "QNN_DATATYPE_SFIXED_POINT_8" ||
             inputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
    bitWidth = 8;
  } else if (inputDataType == "QNN_DATATYPE_UFIXED_POINT_4") {
    bitWidth = 4;
  }

  auto resourceManager = _env->getResourceManager();
  if (!resourceManager->getBuffer(embedding_file_path, m_embeddingBuffer)) {
    throw std::runtime_error("ResourceManager failed to load embedding from: " +
                             embedding_file_path);
  }
  embeddingLutSize = resourceManager->getBufferSize(embedding_file_path);
  embeddingLut     = m_embeddingBuffer.get();

  // Truncation of input to context
  _input_truncation = qc::optional<nlohmann::json>(json, "truncate-input", false);

  // Create Tokenizer
  fs::path tok_path = _env->path().models / qc::mandatory<std::string>(json, "tokenizer");
  _tokenizer        = Tokenizer::create(*_ctx, tok_path, _env->getResourceManager());

  _kpis.init.update(start.elapsed_usec());
}

LUT::~LUT(){};

bool LUT::encode(const std::vector<int32_t>& tokens, std::vector<uint8_t>& output) {
  __DEBUG("embedding-tokens: {}", tokens);

  size_t embeddingSize = _ctx->n_embd() * bitWidth / 8;  // embeddingSize in bytes
  output.resize(tokens.size() * embeddingSize);
  for (size_t i = 0; i < tokens.size(); i++) {
    uint32_t lutIndex = static_cast<uint32_t>(tokens[i]) * embeddingSize;
    if ((lutIndex + embeddingSize) <= embeddingLutSize) {
      uint8_t* embeddingSrc = static_cast<uint8_t*>(embeddingLut) + lutIndex;
      uint8_t* embeddingDst = output.data() + (i * embeddingSize);
      std::copy(embeddingSrc, embeddingSrc + embeddingSize, embeddingDst);
    } else {
      throw std::runtime_error("Error: T2E conversion overflow.");
      return false;
    }
  }
  return true;
}

bool LUT::encode(const std::string& str,
                 std::vector<uint8_t>& output,
                 std::vector<int32_t>& tokenizedInput) {
  _output_dimensions.clear();
  std::string p_str;           // prompt string
  std::vector<int32_t> p_vec;  // prompt tokens

  p_vec.reserve(_ctx->n_ctx());
  if (_ctx->bos_tok() >= 0) {
    p_vec.push_back(_ctx->bos_tok());
  }

  p_str = _tags[0] + str + _tags[1];

  __DEBUG("embedding-query: {}", str);
  __DEBUG("embedding-prompt: {}", p_str);

  _n_queries++;
  _tokenizer->encode(p_str, p_vec);

  __DEBUG("embedding-tokens: {}", p_vec);

  if (p_vec.size() > (_ctx->n_ctx())) {  // Condition to not allow input to exceed context.
    if (_input_truncation == false) {
      throw std::runtime_error("Input exceeds the context of the model.");
    } else {
      p_vec.resize(_ctx->n_ctx());
      std::vector<int32_t> lastToks;
      _tokenizer->encode(_tags[1], lastToks);
      for (size_t i = 0; i < lastToks.size(); i++) {
        p_vec[p_vec.size() - lastToks.size() + i] = lastToks[i];
      }
    }
  }
  lastToken            = p_vec.back();
  size_t embeddingSize = _ctx->n_embd() * bitWidth / 8;  // embeddingSize in bytes
  _output_dimensions.push_back(p_vec.size());
  _output_dimensions.push_back(_ctx->n_embd());
  output.resize(p_vec.size() * embeddingSize);
  for (size_t i = 0; i < p_vec.size(); i++) {
    uint32_t lutIndex = static_cast<uint32_t>(p_vec[i]) * embeddingSize;
    if ((lutIndex + embeddingSize) <= embeddingLutSize) {
      uint8_t* embeddingSrc = static_cast<uint8_t*>(embeddingLut) + lutIndex;
      uint8_t* embeddingDst = output.data() + (i * embeddingSize);
      std::copy(embeddingSrc, embeddingSrc + embeddingSize, embeddingDst);
    } else {
      throw std::runtime_error("Error: T2E conversion overflow.");
      return false;
    }
  }
  tokenizedInput = std::move(p_vec);
  return true;
}

// Get output dimensions
void LUT::output_dimensions(std::vector<std::uint32_t>& outputDimensions) {
  outputDimensions = _output_dimensions;
}

void LUT::outputTensorQuantParam(std::string& dataType,
                                 double& scale,
                                 int32_t& offset,
                                 float& byteWidth) {
  dataType  = inputDataType;
  scale     = lutScale;
  offset    = lutOffset;
  byteWidth = bitWidth / 8.0;
}

size_t LUT::getEmbeddingLutSize() { return embeddingLutSize; }

void* LUT::getEmbeddingLut() { return embeddingLut; }

int32_t LUT::getLastToken() { return lastToken; }

}  // namespace qualla
