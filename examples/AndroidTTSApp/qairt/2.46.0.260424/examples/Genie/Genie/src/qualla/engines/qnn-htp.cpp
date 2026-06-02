//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "Exception.hpp"
#include "Trace.hpp"
#include "qnn-htp.hpp"
#include "qualla/LoraConfig.hpp"
#include "qualla/ModelIOAdaptor.hpp"
#include "qualla/detail/htp-io.hpp"

namespace fs = std::filesystem;

namespace qualla {

NspEngine::NspEngine(Context& ctx, const nlohmann::json& json) : Engine(ctx, "qnn-htp", json) {
  GENIE_TRACE();
  qualla::Timer start;

  using FF  = Feature::Flags;
  _features = FF::OUTPUT_LOGITS | FF::SAVE_RESTORE | FF::DYNAMIC_LOAD | FF::OUTPUT_EMBEDDINGS;

  __DEBUG("qnn-htp: init start");

  qualla::Config conf(json, _type + "-engine:");

  // Parse config
  _params.model_basedir = conf.optional<std::string>("model-basedir", "");
  if (_params.model_basedir.is_relative()) {
    _params.model_basedir = _env->path().models / _params.model_basedir;
    _params.model_basedir = _params.model_basedir.make_preferred();
  }
  _params.model_list = conf.mandatory<std::vector<std::string>>("model-list");

  // Parse model architecture
  std::string model_architecture = conf.optional<std::string>("model-architecture-type", "decoder");
  // Parse model architecture
  _model_type = conf.optional<std::string>("model-type", "text");

  if (model_architecture == "decoder")
    _params.modelArchitectureType = ModelArchitectureType::DECODER;
  else if (model_architecture == "encoder")
    _params.modelArchitectureType = ModelArchitectureType::ENCODER;
  else
    throw std::runtime_error(
        "Only Encoder and Decoder architectures are supported. Invalid architecture supplied : " +
        model_architecture);

  _params.backend_lib          = conf.optional<std::string>("backend-lib", "");
  _params.backend_ext_conf     = conf.optional<std::string>("backend-ext-conf", "");
  _params.shared_Engine        = conf.optional<bool>("shared-engine", false);
  _params.draft_tok_map        = conf.optional<std::string>("draft-token-map", "");
  _params.ctx_size             = _ctx.size();
  _params.mmap_budget          = conf.optional<uint64_t>("mmap-budget", 0);
  _params.mask_type            = conf.optional<std::string>("mask-type", "");
  _params.cross_attention      = conf.optional<bool>("cross-attention", false);
  _params.use_mmap             = conf.optional<bool>("use-mmap", true);
  _params.data_alignment_size  = conf.optional<uint64_t>("data-alignment-size", 0);
  _params.use_async_Init       = conf.optional<bool>("use-async-Init", true);
  _params.spill_fill_bufsize   = conf.optional<size_t>("spill-fill-bufsize", 0);
  _params.kv_dim               = conf.optional<int64_t>("kv-dim", 128);
  _params.batch_size           = conf.optional<uint32_t>("batch-size", 1);
  _params.n_embd               = _ctx.n_embd();
  _params.pad_token            = _ctx.pad();
  _params.img_token            = _ctx.img();
  _params.variant_latency      = std::map<int, int>();
  _params.disable_kv_cache     = conf.optional<bool>("disable-kv-cache", false);
  _params.pooled_output        = conf.optional<bool>("pooled-output", true);
  _params.lmhead_weight_dir    = conf.optional<std::string>("lmhead-weight-dir", "");
  _params.graph_switching      = conf.optional<bool>("enable-graph-switching", false);
  _params.lazy_lora            = conf.optional<std::string>("graph-switching-lora-policy", "");
  _params.weight_shared_lora   = conf.optional<bool>("weight-shared-lora", false);
  _params.skip_lora_validation = conf.optional<bool>("skip-lora-validation", false);
  _params.exec_select_graphs = conf.optional<std::vector<std::string>>("execute-select-graphs", {});
  _params.load_select_graphs = conf.optional<bool>("load-select-graphs", false);

  // external buffer params
  _params.externalBufferType =
      getExternalBufferTypeFromString(conf.optional<std::string>("buffer-type", "persistent"));
  nlohmann::json latencies = conf.optional<nlohmann::json>("latency-map", {});
  for (auto& [variant, latency] : latencies.items())
    _params.variant_latency[std::stoi(variant)] = latency;
  _params.kv_update_method = conf.optional<std::string>("kv-update-method", "POINTER_SHIFT");
  _params.n_threads        = conf.optional<uint32_t>("n-threads", 4);
  _params.dlc_path         = conf.optional<std::string>("dlc-path", "");

  if (_params.disable_kv_cache) {
    _params.n_threads = 0;
  }
  _params.poll = conf.optional<bool>("poll", false);

  // Positional encodings parameters
  if (conf.json.contains("positional-encoding")) {
    try {
      conf.json["positional-encoding"].get_to(_params.positional_encoding_params);
    } catch (const std::runtime_error& e) {
      State::fatal(fmt::format("Error in positional-encoding - {}", e.what()));
      throw std::runtime_error(State::error());
    }
  } else {  // For Backward compatibility. May be removed in future releases
    // __WARN("Using depracated positional encoding config. Please switch to positional-encoding");
    auto& pos_type = _params.positional_encoding_params;
    if (_params.modelArchitectureType == ModelArchitectureType::DECODER) {
      pos_type.type                     = PositionalEncoding::ROPE;
      pos_type.rope_params.dims         = conf.optional<int64_t>("pos-id-dim", 64);
      pos_type.rope_params.dims         = conf.optional("pos-id-dims", pos_type.rope_params.dims);
      pos_type.rope_params.theta        = conf.optional<double>("rope-theta", 10000.0);
      pos_type.rope_params.rope_scaling = conf.optional("rope-scaling", RopeScalingParams());
    } else {
      pos_type.type = PositionalEncoding::ABSOLUTE;
      if (_model_type == "image") pos_type.type = PositionalEncoding::UNDEFINED;
      // Other parameters for ENCODER ONLY model doesn't matter.
    }
  }

  // All the LoRA params are captured and maintained by LoRA class
  _params.lora_conf_type = LoraConfigType::LORA_DISABLE;
  if (conf.json.contains("loraConfig")) {
    try {
      auto loraConfig        = Config(conf.json["loraConfig"], "loraConfig");
      _params.lora_config    = std::make_shared<LoraConfig>(loraConfig, _env);
      _params.lora_conf_type = _params.lora_config->getLoraConfigType();
    } catch (const std::runtime_error& e) {
      State::fatal(fmt::format("Error in parsing params - {}", e.what()));
      throw std::runtime_error(State::error());
    }
  }

  // Long context parameters
  try {
    _params.default_group = conf.optional<std::string>("default-group", "past_");

    if (conf.json.contains("cache-groups")) {
      conf.json["cache-groups"].get_to(_params.cache_group_params);
    } else {
      _params.cache_group_params[_params.default_group] = CacheGroupParams();
      if (conf.json.contains("longcontext")) {
        conf.json["longcontext"].get_to(
            _params.cache_group_params.at(_params.default_group).longcontext_params);
      }
    }
  } catch (const std::runtime_error& e) {
    State::fatal(fmt::format("Error in parsing params - {}", e.what()));
    throw std::runtime_error(State::error());
  }

  {
    nlohmann::json j = _params.cache_group_params;
    __DEBUG("Cache groups parameters = {}", j.dump());
  }

  // Vision parameter
  if (conf.json.contains("vision-param")) {
    try {
      conf.json["vision-param"].get_to(_params.vision_param);
    } catch (const std::runtime_error& e) {
      State::fatal(fmt::format("Error in parsing params - {}", e.what()));
      throw std::runtime_error(State::error());
    }
  }

  // Validation checks for CacheGroup configs
  // Check 1 - Default group should exist in the CacheGroup params
  if (!_params.cache_group_params.contains(_params.default_group)) {
    State::fatal(fmt::format(
        "Default cache group set to {} but no corresponding entry found in cache-groups config",
        _params.default_group));
    throw std::runtime_error(State::error());
  }

  // Check 2 - No prefix is a prefix of another prefix
  const auto& cache_groups = _params.cache_group_params;
  for (auto it1 = cache_groups.begin(); it1 != cache_groups.end(); it1++) {
    for (auto it2 = std::next(it1); it2 != cache_groups.end(); it2++) {
      const std::string &prefix1 = it1->first, prefix2 = it2->first;
      if (prefix1.starts_with(prefix2) || prefix2.starts_with(prefix1)) {
        State::fatal(fmt::format(
            "Configuration error: Cache groups {} and {} are not unique", prefix1, prefix2));
        throw std::runtime_error(State::error());
      }
    }
  }

  _params.embedding_length   = _ctx.embeddingLength();
  _params.embedding_datatype = _ctx.embeddingDatatype();
  // cpumask needs to be a string because JSON RFC doesn't allow for hex ints.
  std::string cpumask = conf.optional<std::string>("cpumask", "0");
  _params.cpumask     = std::stoull(cpumask, nullptr, 0);

  // Debug flags
  _params.debug_specs   = conf.optional<bool>("debug-specs", false);
  _params.debug_tensors = conf.optional<bool>("debug-tensors", false);
  _params.debug_outputs = conf.optional<bool>("debug-outputs", false);
  _params.debug_qnn     = conf.optional<bool>("debug-qnn", static_cast<bool>(_env->logger()));
  _params.debug_path    = conf.optional<std::string>("debug-path", "genie_debug");
  if (!_params.debug_path.empty()) {
    _params.debug_path = (fs::path(_params.debug_path) / (_model_type + "_" + _role)).string();
  }

  if (!conf.optional<bool>("dynamic-load", false)) {
    load();
  }
};

NspEngine::~NspEngine() {
  __DEBUG("qnn-htp: destroyed");
  unload();
}

bool NspEngine::load() {
  GENIE_TRACE();
  if (_model) return true;

  qualla::Timer start;

  __INFO("qnn-htp: loading model");

  if (_model_type == "image") {
    _model = std::make_unique<QnnNspImageModel>(_env, _params);
  } else {
    _model = std::make_unique<QnnNspModel>(_env, _params);
  }

  // Load model
  if (true != _model->initializeModel()) {
    throw std::runtime_error("Failure to initialize model. " + _model->error());
  }

  if (true != _model->validateModel()) {
    throw std::runtime_error("Error validating model. Please check your I/O. " + _model->error());
  }

  // Initialize IO Tensor buffers
  if (true != _model->initializeIOTensors()) {
    throw std::runtime_error("Error in setting up IO Tensors. " + _model->error());
  }

  __INFO("qnn-htp: model has been validated!");

  _model->initializeUnconnectedInputs();

  if (true != _model->initializeKVManager()) {
    throw std::runtime_error("Error initializing KVCache managers: " + _model->error());
  }

  if (true != _model->initializeTensorPointers()) {
    throw std::runtime_error("Error : Could not find I/O tensors in loaded graphs. " +
                             _model->error());
  }

  if (true != _model->calculate_rope_embeddings()) {
    throw std::runtime_error("Error : Could not load precomputed position ids");
  }

  // Initialize LoRA
  if (_model->lora_conf_type == LoraConfigType::LORA_INPUT_WEIGHT_ENABLE) {
    if (true != _model->flushLoraWeightsBuffers())
      throw std::runtime_error("Error : Failed to flush the lora buffers");
  }

  if (true != _model->load_lmhead_weight_as_input()) {
    throw std::runtime_error("Error : Could not load lmhead weight input");
  }

  _kpis.load.update(start.elapsed_usec());
  return true;
}

bool NspEngine::unload() {
  qualla::Timer start;

  __DEBUG("qnn-htp: unloading model");
  _model.reset(nullptr);

  _kpis.unload.update(start.elapsed_usec());

  return true;
}

bool NspEngine::updateKV(size_t n_past) { return updateKV(n_past, {}); }

bool NspEngine::updateKV(size_t n_past, const std::vector<bool>& selected) {
  if (!_model && !load()) return false;

  qualla::Timer start;

  if (n_past > _ctx.size()) {
    __ERROR("qnn-htp: context size exceeded : n_past {}", n_past);
    State::error("context size exceeded");
    throw genie::ContextLimitException("Context Size was exceeded.");
  }

  if (!_model->setKVCacheNPast(n_past, selected)) {
    __ERROR("qnn-htp: Error updating KV$");
    return false;
  }

  __DEBUG("qnn-htp: Dispatched KV$ Update (n_past={}) in {} usec", n_past, start.elapsed_usec());

  _kpis.update_kv.update(start.elapsed_usec());

  return true;
}

size_t NspEngine::process(const std::vector<int32_t>& tokens,
                          std::vector<float>& logits,
                          bool logits_all) {
  return process(tokens, {}, logits, logits_all);
}

size_t NspEngine::process(const std::vector<int32_t>& tokens, Tensor& logits, bool logits_all) {
  return process(tokens, std::vector<int32_t>{}, logits, logits_all);
}

size_t NspEngine::process(std::vector<int32_t>& tokens,
                          std::vector<size_t>& tokenNumPerBatch,
                          std::vector<Tensor>& logits,
                          bool logits_all) {
  return process(tokens, std::vector<int32_t>{}, tokenNumPerBatch, logits, logits_all);
}

size_t NspEngine::process(const std::vector<int32_t>& tokens,
                          const std::vector<int32_t>& attention_map,
                          std::vector<float>& logits,
                          bool logits_all) {
  std::vector<uint8_t> embeddings;
  return processAll(tokens,
                    embeddings,
                    nullptr,  // featureVector
                    {},       // selected
                    0,        // start_idx
                    false,    // post_update
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::process(const std::vector<int32_t>& tokens,
                          const std::vector<int32_t>& attention_map,
                          Tensor& logits,
                          bool logits_all) {
  std::vector<uint8_t> embeddings;
  return processAll(tokens,
                    embeddings,
                    nullptr,  // featureVector
                    {},       // selected
                    0,        // start_idx
                    false,    // post_update
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::process(std::vector<uint8_t>& embeddings,
                          const std::vector<int32_t>& attention_map,
                          Tensor& logits,
                          bool logits_all) {
  return processAll({},  // tokens
                    embeddings,
                    nullptr,  // featureVector
                    {},       // selected
                    0,        // start_idx
                    false,    // post_update
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::process(std::vector<uint8_t>& embeddings,
                          const uint16_t* featureVector,
                          const std::vector<int32_t>& selected,
                          uint32_t start_idx,
                          bool post_update,
                          const std::vector<int32_t>& attention_map,
                          Tensor& logits,
                          bool logits_all) {
  return processAll({},  // tokens
                    embeddings,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::process(std::vector<uint8_t>& embeddings,
                          const uint16_t* featureVector,
                          const std::vector<int32_t>& selected,
                          uint32_t start_idx,
                          bool post_update,
                          const std::vector<int32_t>& attention_map,
                          std::vector<float>& logits,
                          bool logits_all) {
  return processAll({},  // tokens
                    embeddings,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::process(std::vector<uint8_t>& embeddings,
                          const std::vector<int32_t>& attention_map,
                          std::vector<float>& logits,
                          bool logits_all) {
  return processAll({},  // tokens
                    embeddings,
                    nullptr,  // featureVector
                    {},       // selected
                    0,        // start_idx
                    false,    // post_update
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::process(std::vector<int32_t>& tokens,
                          const std::vector<int32_t>& attention_map,
                          std::vector<size_t>& tokenNumPerBatch,
                          std::vector<Tensor>& logits,
                          bool logits_all) {
  std::vector<uint8_t> embeddings;
  return processAll(tokens, embeddings, attention_map, tokenNumPerBatch, logits, logits_all);
}

size_t NspEngine::processAll(const std::vector<int32_t>& tokens,
                             std::vector<uint8_t>& embeddings,
                             const uint16_t* featureVector,
                             const std::vector<int32_t>& selected,
                             uint32_t start_idx,
                             bool post_update,
                             const std::vector<int32_t>& attention_map,
                             std::vector<float>& logits,
                             bool logits_all) {
  if (!_model && !load()) return 0;
  qualla::Timer start;

  __DEBUG("qnn-htp: inference start: n_tokens {} ", embeddings.size());

  size_t n_tok = _model->runInference(tokens,
                                      embeddings,
                                      featureVector,
                                      selected,
                                      start_idx,
                                      post_update,
                                      attention_map,
                                      logits,
                                      logits_all);
  if (_model->failed()) {
    State::error(_model->error());
  }
  __DEBUG("qnn-htp: inference complete : {} usec", start.elapsed_usec());

  _kpis.process.update(start.elapsed_usec());

  return n_tok;
}

size_t NspEngine::processAll(const std::vector<int32_t>& tokens,
                             std::vector<uint8_t>& embeddings,
                             const uint16_t* featureVector,
                             const std::vector<int32_t>& selected,
                             uint32_t start_idx,
                             bool post_update,
                             const std::vector<int32_t>& attention_map,
                             Tensor& logits,
                             bool logits_all) {
  if (!_model && !load()) return 0;
  qualla::Timer start;

  __DEBUG("qnn-htp: inference start: n_tokens {}", embeddings.size());

  size_t n_tok = _model->runInference(tokens,
                                      embeddings,
                                      featureVector,
                                      selected,
                                      start_idx,
                                      post_update,
                                      attention_map,
                                      logits,
                                      logits_all);
  if (_model->failed()) {
    State::error(_model->error());
  }
  __DEBUG("qnn-htp: inference complete : {} usec", start.elapsed_usec());

  _kpis.process.update(start.elapsed_usec());

  return n_tok;
}

size_t NspEngine::processAll(std::vector<int32_t>& tokens,
                             std::vector<uint8_t>& embeddings,
                             const std::vector<int32_t>& attention_map,
                             std::vector<size_t>& tokenNumPerBatch,
                             std::vector<Tensor>& logits,
                             bool logits_all) {
  if (!_model && !load()) return 0;
  qualla::Timer start;

  __DEBUG("qnn-htp: inference start: n_tokens {}", embeddings.size());

  size_t n_tok =
      _model->runInference(tokens, embeddings, attention_map, tokenNumPerBatch, logits, logits_all);
  if (_model->failed()) {
    State::error(_model->error());
  }
  __DEBUG("qnn-htp: multi-batch inference complete : {} usec", start.elapsed_usec());

  _kpis.process.update(start.elapsed_usec());

  return n_tok;
}

size_t NspEngine::process(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
                          std::vector<uint8_t>& outputs) {
  if (!_model && !load()) return 0;
  qualla::Timer start;

  size_t status = _model->runInference(inputs, outputs);
  if (status == 0) {
    State::error("qnn-htp : runInference failed!");
  }
  __DEBUG("qnn-htp: inference complete : {} usec", start.elapsed_usec());

  _kpis.process.update(start.elapsed_usec());

  return status;
}

bool NspEngine::cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding) {
  if (!_model && !load()) {
    return false;
  }
  return _model->cacheEosEmbedding(eosEmbedding);
};

void NspEngine::getInputQuantParam(double& scale, int& offset) {
  _model->getInputQuantParam(scale, offset);
}

size_t NspEngine::getEmbeddingBufferSize() { return _model->getEmbeddingBufferSize(); }

bool NspEngine::set(nlohmann::json data) {
  bool ret = false;

  if (data.contains("kv-prefix-skip")) {
    _model->_size_to_skip_kv_prefix = data["kv-prefix-skip"].get<size_t>();
    ret                             = true;
  }

  if (data.contains("kv-prefix-offset")) {
    _model->_offset_to_apply_kv_prefix = data["kv-prefix-offset"].get<size_t>();
    ret                                = true;
  }
  return ret;
}

nlohmann::json NspEngine::get() {
  return {{"kv-prefix-skip", _model->_size_to_skip_kv_prefix},
          {"kv-prefix-offset", _model->_offset_to_apply_kv_prefix}};
}

qualla::InputType NspEngine::getInputType() { return _model->m_inputType; }

void NspEngine::getTensorParam(
    LayerType layerType, std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth) {
  _model->getTensorParam(layerType, dataType, scale, offset, bitWidth);
}

void NspEngine::getTensorDimensions(LayerType layerType, std::vector<uint32_t>& dimensions) {
  _model->getTensorDimensions(layerType, dimensions);
}

void NspEngine::getInputTensorNames(std::unordered_set<std::string>& inputTensorNames) {
  _model->getInputTensorNames(inputTensorNames);
}

size_t NspEngine::restore(const std::string& name, bool chooseHigherVariant) {
  GENIE_TRACE();
  if (!_model && !load()) return 0;
  if (_savedTokenCheckpoints.find(name) != _savedTokenCheckpoints.end())
    _tokensCheckpoint = _savedTokenCheckpoints[name];

  std::string kvCachePath;

  if (_env->getResourceManager()->isDLCRecord(name)) {
    kvCachePath = name;
  } else {
    fs::path cachePath = std::filesystem::path(name) / fmt::format("kv-cache.{}.qnn-htp", _role);
    kvCachePath        = cachePath.string();
  }

  size_t ret = _model->loadKVCache(kvCachePath, chooseHigherVariant);
  if (_model->failed()) State::error(_model->error());

  fs::path tokenHistoryPath = std::filesystem::path(name) / "token-history";
  _model->loadTokenHistory(tokenHistoryPath.string());

  return ret;
}

bool NspEngine::save(const std::string& name) {
  GENIE_TRACE();
  if (!_model && !load()) return false;

  fs::path tokenHistoryPath = std::filesystem::path(name) / "token-history";
  if (!_model->saveTokenHistory(tokenHistoryPath.string())) return false;

  fs::path cachePath = std::filesystem::path(name) / fmt::format("kv-cache.{}.qnn-htp", _role);

  _savedTokenCheckpoints[name] = _tokensCheckpoint;

  bool ret = _model->saveKVCache(cachePath.string());
  if (_model->failed()) State::error(_model->error());
  return ret;
}

bool NspEngine::saveKvToBuffer(Buffer* kvBuff) {
  GENIE_TRACE();
  if (!_model && !load()) return false;

  bool ret = _model->saveKVCacheToBuffer(kvBuff);
  if (_model->failed()) State::error(_model->error());
  return ret;
}

bool NspEngine::getCacheSpec(CacheFileSpec& spec) {
  if (!_model && !load()) return false;

  bool ret = _model->getCacheSpec(spec);
  return ret;
}

bool NspEngine::getKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  if (!_model && !load()) return false;

  bool ret = _model->getKVHead(spec, layer, head, data, scale);
  return ret;
}

void NspEngine::reset() {
  if (!_model && !load()) return;

  // It's enough to just drop the KV$
  updateKV(0);
  _tokensCheckpoint.clear();

  if (_model) {
    _model->resetState();
  }
}

bool NspEngine::applyLoraAdapter(std::string lora_adapter_name) {
  GENIE_TRACE();
  if (!_model) {
    __ERROR("qnn-htp: applyLoraAdapter failed model not initialized");
    return false;
  }
  if (_model->lora_conf_type == LoraConfigType::LORA_INPUT_WEIGHT_ENABLE) {
    return _model->applyLoraWeights(lora_adapter_name);
  } else
    return _model->applyLoraAdapter(lora_adapter_name);
}

bool NspEngine::releaseLoraAdapter(std::string lora_adapter_name) {
  if (!_model) {
    __ERROR("qnn-htp: releaseLoraAdapter failed model not initialized");
    return false;
  }
  return _model->releaseLoraAdapter(lora_adapter_name);
}

std::string NspEngine::getAppliedLoraAdapter() const {
  if (!_model || !_model->lora_config) {
    return "";
  }
  return _model->lora_config->getAppliedAdapterName();
}

bool NspEngine::setPerfProfile(qualla::PerformanceProfile& perfProfile) {
  return _model->setPerfProfile(perfProfile);
}

bool NspEngine::getPerfProfile(qualla::PerformanceProfile& perfProfile) {
  return _model->getPerfProfile(perfProfile);
}

bool NspEngine::applyLoraStrength(std::string tensor_name, float tensor_val) {
  if (!_model) {
    __ERROR("qnn-htp: applyLoraStrength failed model not initialized");
    return false;
  }
  return _model->applyLoraStrength(tensor_name, tensor_val);
}

bool NspEngine::updateTokenCheckpoint(uint32_t token, uint32_t kvCacheIndx) {
  if (!_model) {
    __ERROR("qnn-htp: updateTokenCheckpoint failed model not initialized");
    return false;
  }
  _tokensCheckpoint.push_back(std::make_pair(token, kvCacheIndx));
  return false;
}

bool NspEngine::removeTokenCheckpoint(size_t removeAmt) {
  if (!_model) {
    __ERROR("qnn-htp: removeTokenCheckpoint failed model not initialized");
    return false;
  }

  _tokensCheckpoint.erase(_tokensCheckpoint.end() - static_cast<long>(removeAmt),
                          _tokensCheckpoint.end());
  return true;
}

std::pair<uint32_t, int32_t> NspEngine::rewindKVCacheToPrefixMatch(std::vector<int32_t>& tokens,
                                                                   uint32_t& past) {
  GENIE_TRACE();
  if (!_model) {
    __ERROR("qnn-htp: revertKVCacheToToken failed model not initialized");
    return {};
  }
  uint32_t idx         = 0;
  uint32_t last_n_past = 0;
  uint32_t rewindIndex = 0;
  int32_t nextToken    = 0;

  for (size_t i = 0; i < _tokensCheckpoint.size() && idx < tokens.size(); i++) {
    if (static_cast<int32_t>(_tokensCheckpoint[i].first) != tokens[idx]) {
      break;
    }

    last_n_past = _tokensCheckpoint[i].second;
    rewindIndex = idx;
    if (i + 1 < _tokensCheckpoint.size()) {
      nextToken = static_cast<int32_t>(_tokensCheckpoint[i + 1].first);
    } else {
      nextToken = -1;
    }
    idx++;
  }
  updateKV(last_n_past + 1);
  if (_model) {
    _model->setHigherVariant();
  } else {
    return {};
  }
  past                      = last_n_past + 1;
  size_t lastCheckpointSize = _tokensCheckpoint.size();
  _tokensCheckpoint.resize(rewindIndex + 1);
  if (idx >= tokens.size() && idx <= lastCheckpointSize) {
    return {rewindIndex + 1, nextToken};
  } else {
    return {rewindIndex + 1, -1};
  }
}

bool NspEngine::setOemkey(const std::string& oemKey) {
  return _model ? _model->setOemKey(oemKey) : false;
}
bool NspEngine::setExecutionPriority(const uint32_t executionPriority) {
  return _model ? _model->setExecutionPriority(executionPriority) : false;
}

size_t NspEngine::getBuffer(void*& buffer, std::string bufferName, bool isPrompt) {
  return _model->getIOBufferByName(bufferName, buffer, isPrompt);
}

void NspEngine::setSharedCounter(std::atomic<int32_t>& counter) {
  if (_model) {
    _model->setSharedCounter(counter);
  }
}

void NspEngine::resetSharedCounter() {
  if (_model) {
    _model->resetSharedCounter();
  }
}

std::string NspEngine::getTokenMapFilePath() { return _model ? _model->m_draft_tok_map : ""; }

void NspEngine::setRunProcess(uint8_t run_process) { _model->setRunProcess(run_process); }

void NspEngine::updatedEmbeddingLength(uint32_t embedLength) {
  if (_model) {
    _model->updatedEmbeddingLength(embedLength);
  }
}

bool NspEngine::isLongContextEnabled() const {
  return _model ? _model->isLongContextEnabled() : false;
}

void NspEngine::pauseQuery() {
  if (_model) {
    _model->pauseQuery();
  }
}

bool NspEngine::applyEngineState(std::shared_ptr<EngineState>& engineState) {
  m_engineState = engineState;
  _model->finalizeState(m_engineState);
  return true;
}

std::shared_ptr<EngineState> NspEngine::getEngineState() { return m_engineState; }

bool NspEngine::isIoLoadingLazy() { return _model ? _model->m_lazyInitialization : false; }

bool NspEngine::setVisionParam(const std::vector<uint32_t>& visionParam) {
  return _model ? _model->setVisionParam(visionParam) : false;
}

bool NspEngine::setCrossAttentionHiddenStates(const std::vector<float>& crossAttentionStates) {
  __DEBUG("NspEngine::setCrossAttentionHiddenStates");
  return _model ? _model->setCrossAttentionHiddenStates(crossAttentionStates) : false;
}

size_t NspEngine::getMaxSingleTurnInputLength() {
  return _model ? _model->getMaxSingleTurnInputLength() : 0;
}

bool NspEngine::registerModelAdaptor(std::shared_ptr<ModelIOAdaptor> adaptor) {
  GENIE_TRACE();
  if (!adaptor) {
    __ERROR("qnn-htp: Cannot register null ModelIOAdaptor");
    return false;
  }

  // Follow standard model loading pattern
  if (!_model && !load()) {
    __ERROR("qnn-htp: Failed to load model for ModelIOAdaptor registration");
    return false;
  }

  __DEBUG("qnn-htp: Registering ModelIOAdaptor: {}", adaptor->name());
  return _model->registerModelIOAdaptor(adaptor);
}

size_t NspEngine::setInputBufferData(LayerType inputLayer,
                                     void* data,
                                     size_t dataSize,
                                     nlohmann::json& dataConfig) {
  GENIE_TRACE();
  if (!_model) {
    __ERROR("qnn-htp: failed to set Input.");
    return false;
  }
  return _model->setInputBufferData(inputLayer, data, dataSize, dataConfig);
}

size_t NspEngine::processPresetInput(const nlohmann::json& config) {
  GENIE_TRACE();
  if (!_model) {
    __ERROR("qnn-htp: failed to execute preset input.");
    return false;
  }
  qualla::Timer start;
  size_t status = _model->runPrefillInputInference(config);
  if (status == 0) {
    State::error("qnn-htp : runInference failed!");
  }
  __DEBUG("qnn-htp: inference complete : {} usec", start.elapsed_usec());

  _kpis.process.update(start.elapsed_usec());

  return status;
}

bool NspEngine::getOutputBufferData(LayerType outputLayer,
                                    const nlohmann::json& outputConfig,
                                    OutputCallback callback) {
  GENIE_TRACE();
  if (!_model) {
    __ERROR("qnn-htp: failed to get output.");
    return false;
  }
  return _model->getOutputBufferData(outputLayer, outputConfig, callback);
}

}  // namespace qualla
