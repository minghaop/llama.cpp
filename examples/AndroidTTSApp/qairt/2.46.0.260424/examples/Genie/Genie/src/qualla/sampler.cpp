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
#include <span>
#include <unordered_map>

#include "qualla/detail/config.hpp"
#include "qualla/detail/htp-io.hpp"
#include "qualla/detail/utils.hpp"
#include "qualla/sampler.hpp"

namespace fs = std::filesystem;
using qc     = qualla::Config;

namespace qualla {

SamplerCbFunctionMap& Sampler::getSamplerCbFunctionMap() {
  static SamplerCbFunctionMap s_samplerCbFunctionMap;
  return s_samplerCbFunctionMap;
}

Sampler::Sampler(std::shared_ptr<Context>& ctx, const std::string& type, const nlohmann::json& conf)
    : State(ctx->env()->getTraceLogger()),
      _type(type),
      _ctx(ctx),
      _env(ctx->env()),
      m_penalty(qc::optional<nlohmann::json>(conf, "token-penalty", {})) {
  __DEBUG("sampler-new: {} ctx {} config {}", type, ctx->name(), conf.dump());

  // Parse config
  _role   = qc::optional<std::string>(conf, "role", "primary");
  _seed   = qc::optional<int32_t>(conf, "seed", -1);
  _greedy = qc::optional<bool>(conf, "greedy", _greedy);

  _gumbel = qc::optional(conf, "use-gumbel", false);
  _gumbel = qc::optional(conf, "gumbel", _gumbel);

  if (_type == "basic") {
    _temp   = qc::optional<float>(conf, "temp", 0.1f);
    _top_k  = qc::optional<size_t>(conf, "top-k", 0);
    _top_p  = qc::optional<float>(conf, "top-p", 0.8f);
    _greedy = (_temp <= 0.f || _top_k == 1);
    _rng.seed(static_cast<uint32_t>(_seed != -1 ? _seed : std::time(nullptr)));
  } else if (_type == "custom") {
    _greedy                    = true;  // only support greedy sampling in custom sampler
    _customProcessCallbackName = qc::mandatory<std::string>(conf, "callback-name");
    auto& samplerCbFunctionMap = getSamplerCbFunctionMap();
    if (samplerCbFunctionMap.find(_customProcessCallbackName) == samplerCbFunctionMap.end()) {
      __ERROR("callback-name {} passed not registered ", _customProcessCallbackName);
    }
  } else {
    __ERROR("Invalid sampler type ", _type);
  }
}

Sampler::Sampler(std::shared_ptr<Context>& ctx)
    : State(ctx->env()->getTraceLogger()),
      _type("basic"),
      _ctx(ctx),
      _env(ctx->env()),
      m_penalty({}) {}

bool Sampler::restore(const std::string& name) {
  if (_type == "basic") {
    fs::path restore_path = std::filesystem::path(name) / fmt::format("sampler.{}.rng", _role);

    std::fstream f(restore_path, std::ios::in);
    if (!f.is_open()) {
      __ERROR("basic-sampler: failed to open {} for reading", restore_path.string());
      return false;
    }

    f >> _rng;
    f.close();

    return true;
  }
  __WARN("{}-sampler does not support restore", _type);
  return false;
}

bool Sampler::save(const std::string& name) {
  if (_type == "basic") {
    fs::path save_path = std::filesystem::path(name) / fmt::format("sampler.{}.rng", _role);

    std::fstream f(save_path, std::ios::out | std::ios::trunc);
    if (!f.is_open()) {
      __ERROR("basic-sampler: failed to open {} for writing", save_path.string());
      return false;
    }

    f << _rng;
    f.close();

    return true;
  }
  __WARN("{}-sampler does not support save", _type);
  return false;
}

void Sampler::reset() {
  if (_type == "basic") {
    // Just need to reinit rng
    _rng.seed(static_cast<uint32_t>(_seed));
    m_penalty.reset();
  } else {
    __WARN("{}-sampler does not support reset", _type);
  }
}

std::vector<int32_t> Sampler::processUnified(Tensor& logits,
                                             std::vector<float>* probs,
                                             int32_t numReturn,
                                             int32_t streamIdx,
                                             size_t topn_probs,
                                             bool output_all_probs) {
  if (_type == "basic") {
    // basic sampler bool fn
    switch (logits.getDataType()) {
      case TENSOR_DATATYPE_UFIXED_POINT_8: {
        return basic_process<uint8_t>(
            logits, probs, numReturn, streamIdx, topn_probs, output_all_probs);
      }
      case TENSOR_DATATYPE_UFIXED_POINT_16: {
        return basic_process<uint16_t>(
            logits, probs, numReturn, streamIdx, topn_probs, output_all_probs);
      }
      case TENSOR_DATATYPE_FLOAT_POINT_16: {
        return basic_process<uint16_t>(
            logits, probs, numReturn, streamIdx, topn_probs, output_all_probs);
      }
      case TENSOR_DATATYPE_FLOAT_32: {
        return basic_process<float>(
            logits, probs, numReturn, streamIdx, topn_probs, output_all_probs);
      }
      default: {
        __WARN("Unsupported datatype");
        return {};
      }
    }
  } else if (_type == "custom") {
    // Custom sampling only supports returning tokens, not probabilities
    if (probs != nullptr) {
      __WARN("Custom sampler does not support returning probabilities");
    }

    // custom sampling fn
    switch (logits.getDataType()) {
      case TENSOR_DATATYPE_UFIXED_POINT_8: {
        return custom_process<uint8_t>(logits, numReturn);
      }
      case TENSOR_DATATYPE_UFIXED_POINT_16: {
        return custom_process<uint16_t>(logits, numReturn);
      }
      case TENSOR_DATATYPE_FLOAT_POINT_16: {
        return custom_process<uint16_t>(logits, numReturn);
      }
      case TENSOR_DATATYPE_FLOAT_32: {
        return custom_process<float>(logits, numReturn);
      }
      default: {
        __WARN("Unsupported datatype");
        return {};
      }
    }
  }
  return {};
}

template <typename T>
std::vector<int32_t> Sampler::basic_process(Tensor& logits,
                                            std::vector<float>* probs_out,
                                            int32_t num_return,
                                            int32_t streamIdx,
                                            size_t topn_probs,
                                            bool output_all_probs) {
  const bool disable_probs = probs_out == nullptr;

  const float temp  = _temp;
  const float top_p = _top_p;

  const std::span<const T> logitsSpan =
      std::span(reinterpret_cast<T*>(logits.getData()), logits.getSize());

  // Logging the logits purely for debugging
  __DEBUG("input-logits: {} ... {}", logitsSpan.first(10), logitsSpan.last(10));

  std::vector<int32_t> ids;

  // Error case - if neither tokens nor probabilities are being requested
  if (num_return == 0 && disable_probs) return {};

  // Hot-path. Greedy sampling without requiring probabilities
  if (_greedy && disable_probs && num_return == 1) {
    genie::profiling::FunctionTracer greedyTracer(*static_cast<const Traceable*>(this), "argmax");
    return {argmax(logitsSpan)};
  }

  // Create indexed logits with the template type T and apply penalties
  IndexedQuantLogits<T> indexed_logits(getTraceLogger(), logits, _rng, m_penalty);
  indexed_logits.penalizeLogits(streamIdx);

  // Apply top-k if either top-k is set or top-n-probs is set
  if (topn_probs > 0) {
    indexed_logits.topK(topn_probs);
  } else if (_top_k > 0) {
    indexed_logits.topK(_top_k);
  }

  // Apply top-p if p < 1.0
  indexed_logits.topP(top_p, 1);

  int32_t id = -1;
  if (_gumbel) {
    // TODO: Remove? Gumbel sampling is not being used currently
    indexed_logits.logSoftmax(temp);
    if (num_return == 1) {
      id = indexed_logits.sampleUsingGumbelMax();
      ids.push_back(id);
    } else if (num_return > 1) {
      indexed_logits.topK(num_return);
      ids = indexed_logits.indices;
    }

    // Add gumbelNoise if probabilities are requested
    if (!disable_probs) {
      indexed_logits.addGumbelNoise();
    }
  } else {
    // Calculate softmax probabilities upon requested, or to sampleFromProbs
    if (!disable_probs || num_return == 1) {
      indexed_logits.softmax(temp);
    }

    // Tokens are sampled using probability for n=1, else a simple topK is used
    if (num_return == 1) {
      ids.push_back(indexed_logits.sampleFromProbs());
    } else if (num_return > 1) {
      indexed_logits.topK(num_return);
      ids = indexed_logits.indices;
    }
  }

  // Handle probability output
  if (!disable_probs && !output_all_probs) {
    QUALLA_ASSERT(indexed_logits.probs_valid);

    const size_t startSize = probs_out->size();
    probs_out->resize(startSize + indexed_logits.size(),
                      _gumbel ? -std::numeric_limits<float>::infinity() : 0);

    // Copy the probabilities out to the user buffer
    std::memcpy(probs_out->data() + startSize,
                &indexed_logits.probs[0],
                indexed_logits.size() * sizeof(float));
  } else if (!disable_probs && output_all_probs) {
    QUALLA_ASSERT(indexed_logits.probs_valid);
    const size_t n_vocab = _ctx->n_vocab();
    // Expand the output vector and fill it with the default values
    probs_out->resize(probs_out->size() + n_vocab,
                      _gumbel ? -std::numeric_limits<float>::infinity() : 0);
    auto p = std::span(probs_out->data(), probs_out->size()).last(n_vocab);
    for (size_t i = 0; i < indexed_logits.size(); i++) {
      p[static_cast<uint32_t>(indexed_logits.indices[i])] = indexed_logits.probs[i];
    }
  }

  return ids;
}

template <typename T>
std::vector<int32_t> Sampler::custom_process(Tensor& logits, int32_t numTokens) {
  std::vector<int32_t> retToken(static_cast<uint32_t>(numTokens));
  std::vector<float> logitVector(logits.getSize());
  std::span<const T> logitsSpan(reinterpret_cast<T*>(logits.getData()), logits.getSize());

  TensorQuantizationParams qp = logits.getQuantizationParams();
  const float scale           = static_cast<double>(qp.scale);
  const int32_t offset        = qp.offset;
  for (size_t i = 0; i < logits.getSize(); i++) {
    logitVector[i] = (static_cast<float>(logitsSpan[i]) + offset) * scale;
  }

  auto& samplerCbFunctionMap = getSamplerCbFunctionMap();
  if (std::get<0>(samplerCbFunctionMap[_customProcessCallbackName]) == nullptr) {
    auto userData = std::get<2>(samplerCbFunctionMap[_customProcessCallbackName]);
    std::get<1>(samplerCbFunctionMap[_customProcessCallbackName])(logits.getSize() * sizeof(float),
                                                                  logitVector.data(),
                                                                  static_cast<uint32_t>(numTokens),
                                                                  retToken.data(),
                                                                  userData);
  } else {
    std::get<0>(samplerCbFunctionMap[_customProcessCallbackName])(logits.getSize() * sizeof(float),
                                                                  logitVector.data(),
                                                                  static_cast<uint32_t>(numTokens),
                                                                  retToken.data());
  }
  return retToken;
}

void Sampler::applyConfig(const nlohmann::json& conf) {
  if (conf.contains("type")) {
    _type = conf["type"];
  }
  if (_type == "basic") {
    if (conf.contains("seed")) _seed = conf["seed"];
    if (conf.contains("temp")) _temp = conf["temp"];

    if (conf.contains("top-k")) _top_k = conf["top-k"];
    if (conf.contains("top-p")) _top_p = conf["top-p"];
  } else if (_type == "custom") {
    if (conf.contains("callback-name")) {
      _customProcessCallbackName = conf["callback-name"];
      auto& samplerCbFunctionMap = getSamplerCbFunctionMap();
      if (samplerCbFunctionMap.find(_customProcessCallbackName) == samplerCbFunctionMap.end()) {
        __ERROR("callback-name {} passed not registered ", _customProcessCallbackName);
      }
    }
  } else {
    __ERROR("Invalid sampler type ", _type);
  }
}

void Sampler::updateSampledTokenHistory(int32_t tokenIdx, int32_t streamIdx) {
  m_penalty.updateSampledTokenHistory(tokenIdx, streamIdx);
}

void Sampler::updateSampledTokenHistory(std::vector<int32_t>& tokenIdxs, int32_t streamIdx) {
  for (auto& idx : tokenIdxs) {
    updateSampledTokenHistory(idx, streamIdx);
  }
}

void Sampler::registerProcessCallBack(std::string name, qualla::SamplerCbFunction callback) {
  getSamplerCbFunctionMap()[name] = std::make_tuple(callback, nullptr, nullptr);
}

void Sampler::registerUserDataCallBack(std::string name,
                                       qualla::SamplerUserDataCbFunction callback,
                                       const void* userData) {
  getSamplerCbFunctionMap()[name] = std::make_tuple(nullptr, callback, userData);
}

std::vector<int32_t> Sampler::sampleData(const void* data,
                                         size_t dataSize,
                                         const char* dataConfigJson,
                                         int32_t numReturn) {
  nlohmann::json dataCfg = nlohmann::json::parse(dataConfigJson);
  qualla::DataConfig cfg(dataCfg);

  // Create a tensor view over provided logits buffer
  qualla::TensorQuantizationParams qp{1.0, 0};

  qualla::TensorDataType tensorType = qualla::TENSOR_DATATYPE_UNKNOWN;
  switch (cfg.m_dataType) {
    case qualla::DataType::UINT_8:
      tensorType = qualla::TENSOR_DATATYPE_UFIXED_POINT_8;
      break;
    case qualla::DataType::UINT_16:
      tensorType = qualla::TENSOR_DATATYPE_UFIXED_POINT_16;
      break;
    case qualla::DataType::INT32:
    case qualla::DataType::UINT_32:
      tensorType = qualla::TENSOR_DATATYPE_INT_32;
      break;
    case qualla::DataType::FLOAT32:
      tensorType = qualla::TENSOR_DATATYPE_FLOAT_32;
      break;
    default:
      __ERROR("Unsupported data-type for sampler");
      return {};
  }

  // Validate buffer size matches config
  const size_t elemBytes     = (tensorType == qualla::TENSOR_DATATYPE_UFIXED_POINT_8)
                                   ? 1
                                   : (tensorType == qualla::TENSOR_DATATYPE_UFIXED_POINT_16 ||
                                          tensorType == qualla::TENSOR_DATATYPE_FLOAT_POINT_16
                                          ? 2
                                          : 4);
  const size_t expectedBytes = cfg.m_dataLength * elemBytes;
  if (dataSize != expectedBytes) {
    __ERROR("Data size mismatch: expected {} bytes, got {} bytes", expectedBytes, dataSize);
    return {};
  }

  qualla::Tensor logits(const_cast<void*>(data),
                        tensorType,
                        qp,
                        cfg.m_dataLength,
                        cfg.m_shape,
                        /*user_owned*/ true);

  // Return one sampled token per logit vector (dims[0] = number of rows, dims[1] = vocab).
  // For dims with rank==1, default to single row.
  if (numReturn <= 0) {
    if (cfg.m_shape.size() >= 2) {
      numReturn = static_cast<int32_t>(cfg.m_shape[0]);
    } else {
      numReturn = 1;
    }
  }

  return processUnified(logits, nullptr, numReturn);
}

std::shared_ptr<Sampler> Sampler::create(std::shared_ptr<Context> ctx, const nlohmann::json& conf) {
  std::string type = qc::optional<std::string>(conf, "type", "basic");

  return std::shared_ptr<Sampler>(new Sampler(ctx, type, conf));
}

std::shared_ptr<Sampler> Sampler::create(std::shared_ptr<Context> ctx, std::istream& json_stream) {
  return create(ctx, nlohmann::json::parse(json_stream));
}

std::shared_ptr<Sampler> Sampler::create(std::shared_ptr<Context> ctx,
                                         const std::string& json_str) {
  return create(ctx, nlohmann::json::parse(json_str));
}

}  // namespace qualla
