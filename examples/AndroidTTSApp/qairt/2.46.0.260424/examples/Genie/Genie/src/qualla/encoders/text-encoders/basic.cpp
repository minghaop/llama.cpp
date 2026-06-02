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

#include "basic.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/timer.hpp"

namespace fs = std::filesystem;

namespace qualla {

Embedding::Embedding(std::shared_ptr<Env> env, const nlohmann::json& json)
    : Encoder(env, "basicTextEncoder", json) {
  Timer start;

  __DEBUG("embedding-new: {} config {}", _type, json.dump());

  using qc = qualla::Config;

  if (json["context"].contains("quant-param")) {
    lutScale  = json["context"]["quant-param"]["scale"];
    lutOffset = json["context"]["quant-param"]["offset"];
  }

  const nlohmann::json& pmt_conf = qc::optional<nlohmann::json>(json, "prompt", {});
  _tags = qc::optional<std::vector<std::string>>(pmt_conf, "tags", {"", ""});

  // Create the context first
  _ctx = Context::create(_env, _type, qc::optional<nlohmann::json>(json, "context", {}));

  // Load LUT
  lutDataType = _ctx->embeddingDatatype();
  if (lutDataType == "QNN_DATATYPE_FLOAT_32") {
    lutByteWidth = 4;
  } else if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_16" ||
             lutDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
    lutByteWidth = 2;
  } else if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_8" ||
             lutDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
    lutByteWidth = 1;
  } else if (lutDataType == "QNN_DATATYPE_UFIXED_POINT_4") {
    // Add u4 embedding lut in leq_v2 feature, bytewidth = 4/8 = 0.5
    lutByteWidth = 0.5;
  }

  // Create Tokenizer
  fs::path tok_path = _env->path().models / qc::mandatory<std::string>(json, "tokenizer");
  _tokenizer        = Tokenizer::create(*_ctx, tok_path, _env->getResourceManager());

  // Create Engine
  const nlohmann::json& eng_conf = qc::mandatory<nlohmann::json>(json, "engine");
  _engine                        = Engine::create(*_ctx, eng_conf);
  _engine->bound();

  // Truncation of input to context
  _input_truncation = qc::optional<nlohmann::json>(json, "truncate-input", false);

  // Encoder translation for LUT sub-encoder + encoder
  if (json.contains("lut-path")) {
    nlohmann::json lutJson = json;
    lutJson["type"]        = "lut";
    m_subEncoder           = Encoder::create(_env, "basicTextEncoder", lutJson);
  }

  calculateRequantEncodings();

  using FF = Engine::Feature::Flags;
  if (!_engine->supports(FF::OUTPUT_EMBEDDINGS)) {
    throw std::runtime_error("engine must output embeddings");
  }

  _engine->getPerfProfile(m_defaultPerfProfile);
  m_perfProfile = m_defaultPerfProfile;
  _kpis.init.update(start.elapsed_usec());
}

Embedding::~Embedding() {}

template <typename T>
bool Embedding::process(std::vector<T>& inputs, std::vector<float>& output) {
  Timer start;

  State::clear();

  size_t n = _engine->process(inputs, {}, output, false);
  if (!n) {
    State::error("engine prompt processing failed");
    return false;
  }

  if constexpr (std::is_same_v<T, int32_t>) {
    _n_prompt += inputs.size();  // Token inputs
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    _n_prompt += inputs.size() / engine().getEmbeddingBufferSize();  // Embedding inputs
  }

  // Clean the buffer before using
  _output_dimensions.clear();

  uint64_t output_size = 1;
  // push number of tokens present in the result.
  _output_dimensions.push_back(n);
  // push back the dimension of the each embedding
  _output_dimensions.push_back(_ctx->n_embd());

  output_size = n * _ctx->n_embd();

  output.resize(output_size);

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return true;
}

void Embedding::requantEmbedding(void* from, void* to, size_t length) {
  if (lutDataType == "QNN_DATATYPE_UFIXED_POINT_4") {
    auto* src = static_cast<uint8_t*>(from);
    if (inputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      auto* dst16 = static_cast<uint16_t*>(to);
      for (size_t j = 0; j < length / 2; ++j) {
        uint8_t high     = (src[j] >> 4) & 0x0F;
        uint8_t low      = src[j] & 0x0F;
        dst16[2 * j]     = static_cast<uint16_t>(requantScale * low + requantOffset);
        dst16[2 * j + 1] = static_cast<uint16_t>(requantScale * high + requantOffset);
      }
    } else if (inputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      auto* dst8 = static_cast<uint8_t*>(to);
      for (size_t j = 0; j < length / 2; ++j) {
        uint8_t high    = (src[j] >> 4) & 0x0F;
        uint8_t low     = src[j] & 0x0F;
        dst8[2 * j]     = static_cast<uint8_t>(requantScale * low + requantOffset);
        dst8[2 * j + 1] = static_cast<uint8_t>(requantScale * high + requantOffset);
      }
    } else {
      throw std::runtime_error("Unsupported requantization operation.");
    }
    return;
  }

  for (size_t i = 0; i < length; i++) {
    if (lutDataType == "QNN_DATATYPE_FLOAT_32" && inputDataType == "QNN_DATATYPE_FLOAT_32") {
      static_cast<float*>(to)[i] = static_cast<float*>(from)[i];
    } else if (lutDataType == "QNN_DATATYPE_FLOAT_32" && inputDataType == "QNN_DATATYPE_FLOAT_16") {
      // TODO
      // static_cast<uint16_t*>(to)[i] = fp16_ieee_from_fp32_value(static_cast<float*>(from)[i]);
    } else if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_8" &&
               inputDataType == "QNN_DATATYPE_SFIXED_POINT_8") {
      static_cast<int8_t*>(to)[i] =
          static_cast<int8_t>(requantScale * static_cast<int8_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_8" &&
               inputDataType == "QNN_DATATYPE_SFIXED_POINT_16") {
      static_cast<int16_t*>(to)[i] =
          static_cast<int16_t>(requantScale * static_cast<int8_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_UFIXED_POINT_8" &&
               inputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t*>(to)[i] =
          static_cast<uint8_t>(requantScale * static_cast<uint8_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_UFIXED_POINT_8" &&
               inputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t*>(to)[i] =
          static_cast<uint16_t>(requantScale * static_cast<uint8_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_16" &&
               inputDataType == "QNN_DATATYPE_SFIXED_POINT_8") {
      static_cast<int8_t*>(to)[i] =
          static_cast<int8_t>(requantScale * static_cast<int16_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_16" &&
               inputDataType == "QNN_DATATYPE_SFIXED_POINT_16") {
      static_cast<int16_t*>(to)[i] =
          static_cast<int16_t>(requantScale * static_cast<int16_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
               inputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t*>(to)[i] =
          static_cast<uint8_t>(requantScale * static_cast<uint16_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
               inputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t*>(to)[i] =
          static_cast<uint16_t>(requantScale * static_cast<uint16_t*>(from)[i] + requantOffset);
    } else {
      throw std::runtime_error("Unsupported requantization operation.");
    }
  }
}

void Embedding::calculateRequantEncodings() {
  _engine->getTensorParam(LayerType::INPUT, inputDataType, inputScale, inputOffset, inputByteWidth);
  requantScale  = lutScale / inputScale;
  requantOffset = requantScale * lutOffset - inputOffset;
}

bool Embedding::query(const std::string& str, std::vector<uint8_t>& output) {
  std::string p_str;           // prompt string
  std::vector<int32_t> p_vec;  // prompt tokens

  p_vec.reserve(_ctx->n_ctx());

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

  std::vector<float> floatOutput;
  bool status = true;

  if (m_subEncoder && m_subEncoder->type() == "lut") {
    size_t embedBufSize = _engine->getEmbeddingBufferSize();
    // LUT + E2E encoder invocation
    std::vector<uint8_t> encoderOutput;
    std::vector<uint8_t> decoderInput;
    status = m_subEncoder->encode(p_vec, encoderOutput);
    {
      std::vector<uint8_t> padEmbedding;
      // pad-token is a mandatory field for encoders.
      // Unlike dialogs, we don't need to fallback to EOS.
      m_subEncoder->encode(std::vector<int32_t>({_ctx->pad()}), padEmbedding);
      std::vector<uint8_t> padEmbeddingRequant(embedBufSize, 0.0);
      requantEmbedding(padEmbedding.data(), padEmbeddingRequant.data(), _ctx->n_embd());
      if (!_engine->cacheEosEmbedding(padEmbeddingRequant)) {
        __DEBUG("Failed to set the pad token embedding.");
        return false;
      }
    }
    decoderInput.resize((encoderOutput.size() * inputByteWidth) / lutByteWidth);
    requantEmbedding(encoderOutput.data(), decoderInput.data(), p_vec.size() * _ctx->n_embd());
    process(decoderInput, floatOutput);
  } else {
    status = process(p_vec, floatOutput);
  }

  if (status == false) return status;
  output.resize(floatOutput.size() * sizeof(float));
  std::memcpy(output.data(), floatOutput.data(), floatOutput.size() * sizeof(float));
  return status;
}

// Embedding KPIs helpers

void Embedding::output_dimensions(std::vector<std::uint32_t>& outputDimensions) {
  outputDimensions = _output_dimensions;
}

void Embedding::outputTensorQuantParam(std::string& dataType,
                                       double& scale,
                                       int32_t& offset,
                                       float& byteWidth) {
  // TODO: Dequant data only when needed. Else use native encoding for output tensor.
  //_engine->getTensorParam(LayerType::OUTPUT, dataType, scale, offset, bitWidth);
  dataType  = "QNN_DATATYPE_FLOAT_32";
  scale     = 1.0;
  offset    = 0;
  byteWidth = 4;
}

bool Embedding::encode(const std::string& str,
                       std::vector<uint8_t>& output,
                       std::vector<int32_t>& /*tokenizedInput*/) {
  return query(str, output);
}

// Get latest KPIs
Embedding::KPIs& Embedding::kpis() {
  // Update TPS
  if (_n_prompt) {
    float t            = _kpis.prompt.total_usec / _n_prompt;
    _kpis.tps.n_prompt = _n_prompt;
    _kpis.tps.prompt   = 1000000.0f / (t ? t : 1000000.0f);
  }

  // We could synthesize more KPIs from from other layers (engine, sampler, etc)
  return _kpis;
}

std::string Embedding::KPIs::dump(std::string_view sep) const {
  return fmt::format("init:[{}]{}prompt:[{}]{} tps-prompt:{:.2f}",
                     init.dump(),
                     sep,
                     prompt.dump(),
                     sep,
                     tps.prompt);
}

void Embedding::KPIs::reset() {
  init.reset();
  prompt.reset();
  tps.prompt = 0.0;
}

}  // namespace qualla
