//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include "Exception.hpp"
#include "qnn-cpu.hpp"
#include "qualla/detail/timer.hpp"

namespace fs = std::filesystem;

namespace qualla {

CpuEngine::CpuEngine(Context& ctx, const nlohmann::json& json)
    : Engine(ctx, "qnn-cpu", json) {
  GENIE_TRACE();
  qualla::Timer start;

  using FF  = Feature::Flags;
  _features = FF::OUTPUT_LOGITS | FF::SAVE_RESTORE | FF::OUTPUT_EMBEDDINGS;

  __DEBUG("qnn-cpu: init start");

  qualla::Config conf(json, _type + "-engine:");

  // Parse config
  QnnCpuModel::Params p;

  std::string model_input = conf.optional<std::string>("model-input", "tokens");
  if (model_input == "tokens")
    p.model_input = QnnCpuModel::ModelInput::TOKENS;
  else if (model_input == "embeddings")
    p.model_input = QnnCpuModel::ModelInput::INPUT_EMBEDDINGS;
  else
    throw std::runtime_error(
        "Only tokens and embeddings outputs are supported. Invalid output supplied : " +
        model_input);
  std::string model_output = conf.optional<std::string>("model-output", "logits");
  if (model_output == "logits")
    p.model_output = QnnCpuModel::ModelOutput::LOGITS;
  else if (model_output == "embeddings")
    p.model_output = QnnCpuModel::ModelOutput::EMBEDDINGS;
  else
    throw std::runtime_error(
        "Only logits and embeddings outputs are supported. Invalid output supplied : " +
        model_output);

  if (conf.json.contains("longcontext")) {
    throw std::runtime_error("Long Context is not supported on CPU.");
  }

  p.model_basedir        = _env->path().models / conf.optional<std::string>("model-basedir", "");
  p.model_bin_path       = conf.mandatory<std::string>("model-bin-path");
  p.model                = conf.mandatory<std::string>("model");
  p.backend_lib          = conf.mandatory<std::string>("backend-lib");
  p.n_threads            = conf.optional<uint32_t>("n-threads", 6);
  p.n_logits             = conf.optional<uint32_t>("n_logits", 1);
  p.n_layer              = conf.optional<uint32_t>("n_layer", 32);
  p.n_embd               = conf.optional<uint32_t>("n_embd", 4096);
  p.n_heads              = conf.optional<uint32_t>("n_heads", 32);
  p.n_kv_heads           = conf.optional<uint32_t>("n_kv_heads", 32);
  p.use_mmap             = conf.optional<bool>("use-mmap", false);
  p.kv_quant             = conf.optional<bool>("kv-quantization", false);
  p.shared_engine        = conf.optional<bool>("shared-engine", false);
  p.ctx_size             = _ctx.size();
  p.n_vocab_size         = _ctx.n_vocab();
  p.embedding_datatype   = _ctx.embeddingDatatype();
  p.lora_conf_type       = LoraConfigType::LORA_DISABLE;
  nlohmann::json lora_conf = conf.optional<nlohmann::json>("lora", {});
  p.model_params_provided =
      json.contains("n_layer") || json.contains("n_embd") || json.contains("n_heads");

  nlohmann::json lora_group_conf = conf.optional<nlohmann::json>("group", {});

  // All the LoRA params are captured and maintained by LoRA class
  if (conf.json.contains("loraConfig")) {
    try {
      auto loraConfig  = Config(conf.json["loraConfig"], "loraConfig");
      p.lora_config    = std::make_shared<LoraConfig>(loraConfig, _env);
      p.lora_conf_type = p.lora_config->getLoraConfigType();
    } catch (const std::runtime_error& e) {
      State::fatal(fmt::format("Error in parsing params - {}", e.what()));
      throw std::runtime_error(State::error());
    }
  }

  _model = std::make_unique<QnnCpuModel>(_env, p);

  // Load model
  if (true != _model->initializeModel()) {
    throw std::runtime_error("Failure to initialize model");
  }

  // Initialize IO Tensor buffers
  if (true != _model->initializeIOTensors()) {
    throw std::runtime_error("Error in setting up IO Tensors");
  }

  if (true != _model->validateModel()) {
    throw std::runtime_error("Error validating model. Please check your I/O");
  }

  __DEBUG("qnn-cpu: model has been validated!");

  if (true != _model->initializeTensorPointers()) {
    throw std::runtime_error("Error : Could not find I/O tensors in loaded graphs");
  }

  _kpis.load.update(start.elapsed_usec());
};

CpuEngine::~CpuEngine() { __DEBUG("qnn-cpu: destroyed"); }

bool CpuEngine::usesCrossAttention() { return _model->usesCrossAttention(); }

bool CpuEngine::isKVQuantized() { return _model->m_kv_quant; }

bool CpuEngine::getRopePermute() { return _model->getRopePermute(); }

bool CpuEngine::updateKV(size_t n_past) {
  qualla::Timer start;

  if (n_past > _ctx.size()) {
    __ERROR("qnn-cpu: context size exceeded : n_past {}", n_past);
    State::error("context size exceeded");
    throw genie::ContextLimitException("Context Size was exceeded.");
  }

  __DEBUG("qnn-cpu: update-kv start : n_past {}", n_past);

  _model->setKVCacheNPast(n_past);

  __DEBUG("qnn-cpu: update-kv complete : {} usec", start.elapsed_usec());

  _kpis.update_kv.update(start.elapsed_usec());

  return true;
}

bool CpuEngine::updateKV(size_t n_past, const std::vector<bool>& /*selected*/) {
  qualla::Timer start;

  if (n_past > _ctx.size()) {
    __ERROR("qnn-cpu: context size exceeded : n_past {}", n_past);
    State::error("context size exceeded");
    throw genie::ContextLimitException("Context Size was exceeded.");
  }

  __DEBUG("qnn-cpu: update-kv start : n_past {}", n_past);

  _model->setKVCacheNPast(n_past);

  __DEBUG("qnn-cpu: update-kv complete : {} usec", start.elapsed_usec());

  _kpis.update_kv.update(start.elapsed_usec());

  return true;
}

size_t CpuEngine::process(const std::vector<int32_t>& tokens,
                          std::vector<float>& logits,
                          bool logits_all = false) {
  qualla::Timer start;

  __DEBUG("qnn-cpu: inference start: n_tokens {}", tokens.size());

  _model->runInference(tokens, logits_all);

  __DEBUG("qnn-cpu: inference complete : {} usec", start.elapsed_usec());

  size_t n_tok;

  {
    qualla::Timer t;

    __DEBUG("qnn-cpu: get-logits start: all {}", logits_all);

    n_tok = _model->getDequantLogits(logits, logits_all);

    __DEBUG("qnn-cpu: get-logits complete : {} usec", t.elapsed_usec());
  }

  _kpis.process.update(start.elapsed_usec());

  return n_tok;
}

qualla::InputType CpuEngine::getInputType() {
  return static_cast<qualla::InputType>(_model->model_input);
}

void CpuEngine::getTensorParam(LayerType layerType,
                               std::string& dataType,
                               double& scale,
                               int32_t& offset,
                               size_t& bitWidth) {
  _model->getTensorParam(layerType, dataType, scale, offset, bitWidth);
}

size_t CpuEngine::process(const std::vector<int32_t>& tokens,
                          Tensor& logits,
                          bool logits_all = false) {
  qualla::Timer start;

  __DEBUG("qnn-cpu: inference start: n_tokens {}", tokens.size());

  _model->runInference(tokens, logits_all);

  __DEBUG("qnn-cpu: inference complete : {} usec", start.elapsed_usec());

  size_t n_tok;

  {
    qualla::Timer t;

    __DEBUG("qnn-cpu: get-logits start: all {}", logits_all);

    n_tok = _model->getLogits(logits, logits_all);

    __DEBUG("qnn-cpu: get-logits complete : {} usec", t.elapsed_usec());
  }

  _kpis.process.update(start.elapsed_usec());

  return n_tok;
}

size_t CpuEngine::process(const std::vector<int32_t>& tokens,
                          const std::vector<int32_t>& /*attention_map*/,
                          Tensor& logits,
                          bool logits_all = false) {
  return process(tokens, logits, logits_all);
}

size_t CpuEngine::process(const std::vector<int32_t>& tokens,
                          const std::vector<int32_t>& /*attention_map*/,
                          std::vector<float>& logits,
                          bool logits_all = false) {
  return process(tokens, logits, logits_all);
}

size_t CpuEngine::process(std::vector<uint8_t>& embeddings,
                          const std::vector<int32_t>& /*attention_map*/,
                          Tensor& logits,
                          bool logits_all) {
  qualla::Timer start;

  __DEBUG("qnn-cpu: inference start: n_tokens {}", embeddings.size());

  _model->runInference(embeddings, logits_all);

  __DEBUG("qnn-cpu: inference complete : {} usec", start.elapsed_usec());

  size_t n_tok;

  {
    qualla::Timer t;

    __DEBUG("qnn-cpu: get-logits start: all {}", logits_all);

    n_tok = _model->getLogits(logits, logits_all);

    __DEBUG("qnn-cpu: get-logits complete : {} usec", t.elapsed_usec());
  }

  _kpis.process.update(start.elapsed_usec());

  return n_tok;
}

size_t CpuEngine::getEmbeddingBufferSize() { return _model->getEmbeddingBufferSize(); }

size_t CpuEngine::restore(const std::string& name, bool /*chooseHigherVariant*/) {
  GENIE_TRACE();
  fs::path cache_path = std::filesystem::path(name) / fmt::format("kv-cache.{}.qnn-cpu", _role);
  return _model->loadKVCache(cache_path.string());
}

bool CpuEngine::setKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  bool ret = _model->setKVHead(spec, layer, head, data, scale);

  return ret;
}

bool CpuEngine::save(const std::string& name) {
  GENIE_TRACE();
  fs::path cache_path = std::filesystem::path(name) / fmt::format("kv-cache.{}.qnn-cpu", _role);
  return _model->saveKVCache(cache_path.string());
}

void CpuEngine::reset() {
  // It's enough to just drop the KV$
  updateKV(0);
  m_tokensCheckpoint.clear();
}

// For Lora
bool CpuEngine::applyLoraAdapter(std::string lora_adapter_name) {
  GENIE_TRACE();
  if (!_model) {
    __ERROR("qnn-cpu: applyLoraAdapter failed, model not initialized");
    return false;
  }
  return _model->applyLoraAdapter(lora_adapter_name);
}

bool CpuEngine::applyLoraStrength(std::string tensor_name, float tensor_val) {
  if (!_model) {
    __ERROR("qnn-cpu: applyLoraStrength failed, model not initialized");
    return false;
  }
  return _model->applyLoraStrength(tensor_name, tensor_val);
}

std::pair<uint32_t, int32_t> CpuEngine::rewindKVCacheToPrefixMatch(std::vector<int32_t>& tokens,
                                                                   uint32_t& past) {
  GENIE_TRACE();
  if (!_model) {
    __ERROR("qnn-cpu: rewindKVCacheToPrefixMatch failed, model not initialized");
    return {};
  }
  uint32_t idx         = 0;
  uint32_t last_n_past = 0;
  uint32_t rewindIndex = 0;
  int32_t nextToken    = 0;

  for (size_t i = 0; i < m_tokensCheckpoint.size() && idx < tokens.size(); i++) {
    if (static_cast<int32_t>(m_tokensCheckpoint[i].first) != tokens[idx]) {
      break;
    }

    last_n_past = m_tokensCheckpoint[i].second;
    rewindIndex = idx;
    if (i + 1 < m_tokensCheckpoint.size()) {
      nextToken = static_cast<int32_t>(m_tokensCheckpoint[i + 1].first);
    } else {
      nextToken = -1;
    }
    idx++;
  }

  updateKV(last_n_past + 1);
  past                      = last_n_past + 1;
  size_t lastCheckpointSize = m_tokensCheckpoint.size();
  m_tokensCheckpoint.resize(rewindIndex + 1);
  if (idx >= tokens.size() && idx <= lastCheckpointSize) {
    return {rewindIndex + 1, nextToken};
  } else {
    return {rewindIndex + 1, -1};
  }
}

bool CpuEngine::removeTokenCheckpoint(size_t removeAmt) {
  if (!_model) {
    __ERROR("qnn-cpu: removeTokenCheckpoint failed model not initialized");
    return false;
  }

  m_tokensCheckpoint.erase(m_tokensCheckpoint.end() - static_cast<long>(removeAmt),
                           m_tokensCheckpoint.end());
  return true;
}

bool CpuEngine::updateTokenCheckpoint(uint32_t token, uint32_t kvCacheIndx) {
  m_tokensCheckpoint.push_back(std::make_pair(token, kvCacheIndx));
  return true;
}

bool CpuEngine::applyEngineState(std::shared_ptr<EngineState>& engineState) {
  m_engineState = engineState;
  return _model->finalizeState(m_engineState);
}

std::shared_ptr<EngineState> CpuEngine::getEngineState() { return m_engineState; }

}  // namespace qualla
