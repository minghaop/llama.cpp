//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#define _USE_MATH_DEFINES  // Used for M_PI

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <sstream>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif  // defined(__GNUC__) || defined(__clang__)
#include "fp16/fp16.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) || defined(__clang__)

#include "Quantization.hpp"
#include "Trace.hpp"
#include "TraceLogger.hpp"
#include "attention-mask.hpp"
#include "fmt/format.h"
#include "fmt/os.h"
#include "fmt/ranges.h"
#include "native-kv.hpp"
#include "nsp-model.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/timer.hpp"
#include "smart-mask.hpp"

namespace fs = std::filesystem;

namespace qualla {

QnnNspModel::QnnNspModel(std::shared_ptr<Env> env, const QnnNspBaseModel::Params& params)
    : QnnNspBaseModel(env, params) {
  GENIE_TRACE();
  spill_fill_buffer_size  = params.spill_fill_bufsize;
  m_kv_dim                = params.kv_dim;
  m_batch_size            = params.batch_size;
  m_use_mmap              = params.use_mmap;
  mmap_budget             = params.mmap_budget;
  m_dataAlignmentSize     = params.data_alignment_size;
  m_ctx_size              = params.ctx_size;
  m_pad_token             = params.pad_token;
  m_img_token             = params.img_token;
  lmhead_weight_dir       = params.lmhead_weight_dir;
  graph_switching         = params.graph_switching;
  lazy_lora               = params.lazy_lora;
  weight_shared_lora      = params.weight_shared_lora;
  skip_lora_validation    = params.skip_lora_validation;
  load_select_graphs      = params.load_select_graphs;
  embedding_length        = params.embedding_length;
  embedding_datatype      = params.embedding_datatype;
  m_disableKvCache        = params.disable_kv_cache;
  m_embd_size             = params.n_embd;
  m_modelArchitectureType = params.modelArchitectureType;
  m_positional_encoding   = params.positional_encoding_params;
  m_mask_type             = params.mask_type;
  m_cross_attention       = params.cross_attention;
  m_externalBufferType    = params.externalBufferType;
  m_dataFillPolicy        = DataFillPolicy::EXTERNAL;  // for now data is externally populated

  if (m_positional_encoding.type == PositionalEncoding::ROPE) {
    m_pos_dim = static_cast<uint32_t>(m_positional_encoding.rope_params.dims);
  }

  // Longcontext params
  m_default_group          = params.default_group;
  m_cache_group_params_map = params.cache_group_params;

  m_draft_tok_map = params.draft_tok_map;
  if (graph_switching && !m_use_mmap)
    __WARN("Graph switching with non-mmaped implementation can cause high sustained memory usage");

  variant_latency = params.variant_latency;

  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    m_pooled_output = params.pooled_output;
  }

  exec_select_graphs = params.exec_select_graphs;
  if (!exec_select_graphs.empty())
    __DEBUG("qnn-htp : Execute selected graphs = {}", exec_select_graphs);

  if (params.kv_update_method == "SHIFT_CONCAT" || (params.kv_update_method == "POINTER_SHIFT"))
    __WARN("kv-update-method is deprecated. Defaulting to SMART_MASK or NATIVE_KV");
  _kv_update_method =
      KVManagerMode::SMART_MASK;  // Updates to NATIVE_KV if HMX_WEIGHT_LAYOUT tensor is found

  // Set up filename list.
  for (auto& i : params.model_list) {
    auto resourceManager = env->getResourceManager();
    if (resourceManager->getDlcHandle() == nullptr) {
      fs::path model_path = fs::path(i);
      if (model_path.is_relative()) model_path = model_basedir / fs::path(i);
      if (!fs::is_regular_file(model_path)) {
        __ERROR("NSPModel: Can't access model file : {}", model_path.string());
        throw std::runtime_error("NSPModel: Can't access model file : " + model_path.string());
      }
      model_filelist.push_back(model_path.string());
    } else {
      model_filelist.push_back(i);
    }
  }

  m_qnnApi->setKVDim(static_cast<uint32_t>(m_kv_dim));
  m_qnnApi->setContextSize(m_ctx_size);
  m_qnnApi->setKVUpdateMethod(_kv_update_method);
  m_qnnApi->setDataAlignmentSize(m_dataAlignmentSize);

  for (const auto& [prefix, _] : m_cache_group_params_map) {
    m_cache_group_prefixes.insert(prefix);
  }
  m_qnnApi->setCacheGroupPrefixes(m_cache_group_prefixes);

  if (params.debug_specs || params.debug_tensors) {
    if (!fs::exists(params.debug_path) && !fs::create_directories(params.debug_path))
      throw std::runtime_error("Could not create debug directory : " + params.debug_path);
  }

  // Instantiation of threapool must be done at last, To avoid owner less state of it.
  if (params.n_threads > 0) {
    __DEBUG("nsp-model: starting threadpool : n_threads {} params. {:#x} poll {}",
            params.n_threads,
            params.cpumask,
            params.poll);
    m_threadpool = std::make_shared<ThreadPool>();
    m_threadpool->start(params.n_threads, params.cpumask, params.poll);
  }
}

QnnNspModel::~QnnNspModel() {
  qualla::Timer start;

  // The threadpool needs to be stopped before KVManager
  // destruction to avoid race conditions.
  if (m_kvmanager) {
    m_kvmanager->deRegisterAll();
  }

  // Free cached RoPE memory
  if (rope_sin != nullptr) free(rope_sin);
  if (rope_cos != nullptr) free(rope_cos);
  m_ropeBufferReady = false;

  if (eagle_extra_feature != nullptr) {
    free(eagle_extra_feature);
    eagle_extra_feature = nullptr;
  }
  _counter = nullptr;
  if (m_threadpool) m_threadpool->stop();
  __DEBUG("qnn-htp: model destruct complete: {} usec", start.elapsed_usec());

  if (_attention_scratch != nullptr) {
    free(_attention_scratch);
    _attention_scratch = nullptr;
  }
}

// Given a filename, initializeModel load and initializes QNN runtime libraries and the model
bool QnnNspModel::initializeModel(void) {
  GENIE_TRACE();
  qualla::Timer start;

  __DEBUG("qnn-htp: model init start");

  // Default backends
#ifdef _WIN32
  const std::string m_backend                = _backend_lib.empty() ? "QnnHtp.dll" : _backend_lib;
  const std::string m_systemLib              = "QnnSystem.dll";
  const std::string backendExtensionsLibPath = "QnnHtpNetRunExtensions.dll";
#else
  const std::string m_backend                = _backend_lib.empty() ? "libQnnHtp.so" : _backend_lib;
  const std::string m_systemLib              = "libQnnSystem.so";
  const std::string backendExtensionsLibPath = "libQnnHtpNetRunExtensions.so";
#endif

  if (_backend_ext_conf.empty()) {
    __INFO("No backend extension config provided");
  }
  fs::path m_backendExtensionsConfigPath = fs::path(_backend_ext_conf);

  __INFO("Backend library : {}", m_backend);
  __INFO("System library  : {}", m_systemLib);
  __INFO("Model dir   : {}", model_basedir.string());
  __INFO("Model files : {}", model_filelist);
  __INFO("Backend extensions lib path : {}", backendExtensionsLibPath);
  __INFO("Backend extensions config path : {}", m_backendExtensionsConfigPath.string());

  auto logger       = _env->logger();
  uint32_t logLevel = 1;  // error
  std::function<void(const char* fmt, uint32_t level, uint64_t timestamp, va_list args)>
      logCallback = nullptr;
  if (_debug_qnn && logger) {
    logLevel                          = static_cast<uint32_t>(logger->getMaxLevel());
    GenieLog_Callback_t localCallback = logger->getCallback();
    GenieLog_Handle_t localHandle     = logger->getHandle();
    logCallback                       = [localCallback, localHandle](
                      const char* fmt, uint32_t level, uint64_t timestamp, va_list args) {
      // Convert the parameters to match the GenieLog_Callback_t signature
      GenieLog_Level_t genieLevel = static_cast<GenieLog_Level_t>(level);
      localCallback(localHandle, fmt, genieLevel, timestamp, args);
    };
  }

  if (!m_qnnApi->populateGraphBinaryInfo(model_filelist, graph_switching, m_use_mmap)) {
    __ERROR("populateGraphBinaryInfo failed");
    return false;
  }

  if (_debug_specs) dumpTensorSpecs();

  // Compile the number of LLM graphs and auxiliary graphs
  const size_t num_graphs = static_cast<size_t>(m_qnnApi->getGraphsCount());
  const auto graphs_info  = m_qnnApi->getGraphsInfo();

  __INFO("qnn-api initialized with {} graph(s)", num_graphs);

  // Finalize the CacheGroup config, filling in missing values with detected tensors
  // We run one pass across all input tensors for all graphs
  for (size_t graph_idx = 0; graph_idx < num_graphs; graph_idx++) {
    qnn_wrapper_api::GraphInfo_t* const graph_info = graphs_info[graph_idx];
    for (size_t tensor_idx = 0; tensor_idx < graph_info->numInputTensors; tensor_idx++) {
      std::string tname = QNN_TENSOR_GET_NAME(graph_info->inputTensors[tensor_idx]);

      if (tname == "cross_attention_states") {
        m_layerNames[LayerType::CROSS_ATTN_STATES] = "cross_attention_states";
      } else if (tname == "cross_attention_mask") {
        m_layerNames[LayerType::CROSS_ATTN_MASK] = "cross_attention_mask";
      }

      bool foundKeyDiffConfig{false};
      for (auto& [prefix, param] : m_cache_group_params_map) {
        // For any empty tensor name in the cache group configuration, match this schema:
        //    - For the default-group, match either prefix.*m_layerNames or just m_layerNames
        //    - For all other groups, must match prefix.*m_layerNames
        if (param.longcontext_params.mode == LongContextParams::KEYDIFF) {
          if (foundKeyDiffConfig) {
            State::error("Unsupported configuration: Multiple keydiff objects.");
            return false;
          } else {
            m_anchorAlpha      = param.longcontext_params.anchor_alpha;
            foundKeyDiffConfig = true;
          }
        }
        if (param.attention_mask_tensor_name.empty()) {
          if ((prefix == m_default_group && tname == m_layerNames[LayerType::ATTN_MASK]) ||
              (tname.starts_with(prefix) &&
               tname.find(m_layerNames[LayerType::ATTN_MASK]) != std::string::npos)) {
            param.attention_mask_tensor_name = tname;
          }
        }
        if (param.cache_index_tensor_name.empty()) {
          if ((prefix == m_default_group && tname == m_layerNames[LayerType::CACHE_INDEX]) ||
              (tname.starts_with(prefix) &&
               tname.find(m_layerNames[LayerType::CACHE_INDEX]) != std::string::npos)) {
            param.cache_index_tensor_name = tname;
          }
        }
      }
    }
  }

  {
    nlohmann::json j = m_cache_group_params_map;
    __DEBUG("Detected CacheGroup parameters = {}", j.dump());
  }

  for (auto& [prefix, param] : m_cache_group_params_map) {
    if (param.attention_mask_tensor_name.empty()) {
      __WARN("Could not find attention mask tensor for CacheGroup {}", prefix);
      if (prefix == m_default_group) {
        State::error(fmt::format("Default Group {} has no associated attention mask", prefix));
        return false;
      }
    }
    if (param.cache_index_tensor_name.empty()) {
      __DEBUG("Could not find cache index tensor for CacheGroup {}", prefix);
    }
  }

  m_variant_list.reserve(num_graphs);
  std::map<std::pair<int32_t, int32_t>, std::set<std::string>> graph_names;
  for (size_t graph_idx = 0; graph_idx < num_graphs; graph_idx++) {
    qnn_wrapper_api::GraphInfo_t* const graph_info = graphs_info[graph_idx];
    const std::string graph_name                   = std::string(graph_info->graphName);

    __DEBUG("qnn-htp: Graph {}", graph_name);
    GraphVariant graph(graph_info,
                       m_layerNames,
                       _env,
                       m_cache_group_prefixes,
                       m_default_group,
                       static_cast<uint32_t>(m_batch_size));
    if (!variant_latency.empty() && !variant_latency.contains(graph.n_tokens)) {
      __WARN("qnn-htp: Disabling {} based on conf file", graph_name);
      continue;
    }
    if (exec_select_graphs.size() != 0 &&
        std::find(exec_select_graphs.begin(), exec_select_graphs.end(), graph_name) ==
            exec_select_graphs.end()) {
      __DEBUG("qnn-htp: Graph {} is not selected to execute based on conf file", graph_name);
      continue;
    }
    m_variant_list.emplace_back(graph);
    m_graph_map[graph_name] = &m_variant_list.back();

    std::pair<int32_t, int32_t> variant_spec = {graph.n_tokens, graph.ctx_size};
    nsp_graph_count[variant_spec]++;
    graph_names[variant_spec].insert(graph_name);
  }
  // Collect all available ctx_sizes so we can handle not being able to detect ctx_size in a variant
  std::unordered_set<int32_t> available_ctx_size;
  for (const auto& [variant_spec, count] : nsp_graph_count) {
    if (variant_spec.second != -1) available_ctx_size.insert(variant_spec.second);
  }
  std::vector<std::pair<int32_t, int32_t>> keysToDelete;
  // For all variants where we did not detect a ctx_size, add it to all ctx_sizes
  for (const auto& [variant_spec, count] : nsp_graph_count) {
    if (variant_spec.second != -1) continue;
    auto& prev_names = graph_names.at(variant_spec);
    for (const auto& new_ctx : available_ctx_size) {
      std::pair<int32_t, int32_t> new_spec = {variant_spec.first, new_ctx};
      nsp_graph_count[new_spec]++;
      auto& new_names = graph_names[new_spec];
      new_names.insert(prev_names.begin(), prev_names.end());
    }
    keysToDelete.push_back(variant_spec);
  }
  for (const auto& key : keysToDelete) {
    graph_names.erase(key);
    nsp_graph_count.erase(key);
  }

  if (exec_select_graphs.size() != 0 && graph_names.empty()) {
    __ERROR("No matching graphs based on conf file");
  }

  // Create NSPGraph for each splits
  int32_t n_splits = 0;
  for (auto& [_, count] : nsp_graph_count) {
    n_splits = std::max(n_splits, count);
  }
  m_nsp_graphs.reserve(static_cast<uint32_t>(n_splits));
  for (int idx = 0; idx < n_splits; idx++) {
    m_nsp_graphs.emplace_back(idx, _env, m_qnnApi.get(), m_ioTensor);
    m_nsp_graphs.back().setDebugMode(_debug_specs, _debug_tensors, _debug_path);
  }

  // Insert all GraphVariants into corresponding NSPGraph
  for (auto& [variant_spec, graphs] : graph_names) {
    const auto& [variant, ctx_size] = variant_spec;
    uint32_t idx = 0;  // Graph names are sorted by default (std::set<>), so iterate by split
    for (auto& graph_name : graphs) {
      __INFO("Inserting graph {} as idx {} for AR-{} CL-{}", graph_name, idx, variant, ctx_size);
      m_nsp_graphs[idx++].addGraph(m_graph_map.at(graph_name));
    }
  }

  // Detect whether NATIVE_KV needs to be activated
  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.input_specs) {
      // If QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT is detected, we switch to NATIVE_KV format
      if (tspec.tensor->v1.dataFormat == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT) {
        _kv_update_method    = KVManagerMode::NATIVE_KV;
        m_expectedDataFormat = tspec.tensor->v1.dataFormat;
        m_qnnApi->setKVUpdateMethod(_kv_update_method);
        break;
      }
    }
    if (_kv_update_method == KVManagerMode::NATIVE_KV) break;
  }

  // Populate the graph variant type map for use in QnnApi.cpp
  for (auto& variant : m_variant_list) {
    m_graph_variant_type_map[variant.graph_name] = variant.variantType;
  }

  __INFO("qnn-htp: Graphs loaded ((AR-n, CL-x): #splits): {}", nsp_graph_count);

  size_t max_ctx_size = 0;
  for (auto& [variant_spec, count] : nsp_graph_count) {
    max_ctx_size = std::max(max_ctx_size, static_cast<size_t>(variant_spec.second));
  }

  // If LongContext is disabled, make sure the config CL matches loaded CL
  if (max_ctx_size < m_ctx_size && !isLongContextEnabled()) {
    State::error(fmt::format(
        "Config specifies context->size={}, but loaded max-CL={}", m_ctx_size, max_ctx_size));
    return false;
  }

  if (!analyzeCacheGroupKV()) {
    return false;
  }
  m_qnnApi->setGraphVariantType(m_graph_variant_type_map);
  m_qnnApi->setCacheGroupCtxSize(m_cache_group_ctx_size);
  m_qnnApi->setCacheGroupUseScatter(m_cache_group_use_scatter);
  if (!m_qnnApi->initializeHtp(m_backend,
                               model_filelist,
                               BackendExtensionsConfigs(backendExtensionsLibPath,
                                                        m_backendExtensionsConfigPath.string()),
                               {},           // graphConfigs
                               true,         // loadFromCachedBinary
                               m_systemLib,  // systemLibraryPath
                               false,
                               static_cast<size_t>(spill_fill_buffer_size),
                               m_use_mmap,
                               m_use_async_Init,
                               mmap_budget,
                               _debug_qnn,
                               graph_switching,
                               exec_select_graphs,
                               load_select_graphs,
                               skip_lora_validation,
                               m_lazyInitialization,
                               logLevel,
                               logCallback)) {
    __ERROR("qnn-api initialization failed!");
    return false;
  }
  __DEBUG("qnn-htp: Model Init complete: {} usec", start.elapsed_usec());
  return true;
}

// Once the model has been loaded, initialize IO Tensors
// m_ioTensors is initialized by the context for now
bool QnnNspModel::initializeIOTensors() {
  GENIE_TRACE();
  // IO Tensor Mem Registration is already done within the
  // model_initailize by Qnn_API for Sync Init.
  if (m_lazyInitialization) return true;
  // set lmHeadWeightsEnabled and loraWeights Enabled
  _lmhead_weight_input = m_qnnApi->getLmHeadWeightInputEnabled();
  _lora_enabled        = m_qnnApi->getLoraWeightEnabled();
  for (auto it = nsp_graph_count.rbegin(); it != nsp_graph_count.rend(); ++it) {
    for (QnnNspGraph& graph : m_nsp_graphs) {
      // TensorAllocInfo is added to each NSP graph.
      // Needed by Pointer_SHIFT Registration During Execute.
      graph.tensor_alloc_info = m_qnnApi->getTensorAllocInfo();
      graph.g_buffer_mgr      = m_ioTensor;
      if (graph.tensor_alloc_info == NULL) {
        __ERROR("Error Tensor Allocation Failed.");
        return false;
      }
    }
  }

  return true;
}

/* Converts "don't care" dimensions into "*" */
static std::string translateDim(int32_t dim) { return (dim == -1) ? "*" : std::to_string(dim); }

static bool checkShape(const std::string& tensor_name,
                       const QnnUtils::Tensor* tensor,
                       int32_t height,
                       int32_t width,
                       int32_t channel,
                       int32_t bitwidth,
                       std::vector<std::tuple<std::string, std::string, std::string>>& errors) {
  if (tensor != nullptr) {
    const QnnUtils::Dims& tDims = tensor->dims;
    if ((height == -1 || static_cast<uint32_t>(height) == tDims.height) &&
        (width == -1 || static_cast<uint32_t>(width) == tDims.width) &&
        (channel == -1 || static_cast<uint32_t>(channel) == tDims.channel) &&
        (bitwidth == -1 || static_cast<uint32_t>(bitwidth) == tDims.bitwidth)) {
      return true;
    }

    std::stringstream err_msg;
    err_msg << "Expected [ " << translateDim(height) << ", " << translateDim(width) << ", "
            << translateDim(channel) << "] "
            << "bitwidth=" << translateDim(bitwidth) << ". Found [ " << tDims.height << ", "
            << tDims.width << ", " << tDims.channel << "] "
            << "bitwidth=" << tDims.bitwidth;

    errors.push_back({"ShapeError", tensor_name, err_msg.str()});
  }

  return false;
}

// Utility functions
auto toInput  = [](const std::string& s) { return QnnUtils::replaceSubstring(s, "_out", "_in"); };
auto toOutput = [](const std::string& s) { return QnnUtils::replaceSubstring(s, "_in", "_out"); };
auto toVal = [](const std::string& s) { return QnnUtils::replaceSubstring(s, "_key", "_value"); };

bool QnnNspModel::analyzeCacheGroupKV() {
  GENIE_TRACE();
  // Detect if the model uses Scatter (new_key -> past_key) or Concat (past_key + new_key)
  // We can iterate through all graph variants (arn/ctx_size) until we find one with KV$ input
  // If no KV$ input is found, it's AR-c only model and doesn't use Scatter or Concat
  for (const auto& [prefix, param] : m_cache_group_params_map) {
    m_cache_group_use_scatter[prefix] = false;

    // Initialize CacheGroup variant map to a default global->global mapping
    m_cache_group_variant_map[prefix] = {};
    for (auto& nsp_graph : m_nsp_graphs) {
      for (auto& [global_variant, variant] : nsp_graph.variants) {
        m_cache_group_variant_map[prefix][global_variant] = global_variant;
      }
    }

    if (param.attention_mask_tensor_name.empty()) continue;

    // Detect whether KV$ uses Scatter or Concat
    bool detected = false;
    for (const auto& [variant_spec, count] : nsp_graph_count) {
      const auto& [n_tokens, ctx_size] = variant_spec;

      int32_t kv_ctx{0};
      QnnUtils::Tensor* attention_mask = nullptr;
      for (auto& graph : m_nsp_graphs) {
        if (!graph.variants.contains({n_tokens, ctx_size})) continue;
        GraphVariant* variant = graph(n_tokens, ctx_size);

        if (kv_ctx == 0) {
          for (auto& [tname, tspec] : variant->input_specs) {
            if (tname.starts_with(prefix) && tname.find("key") != std::string::npos) {
              kv_ctx = static_cast<int32_t>(tspec.dims.channel);
              break;
            }
          }
        }

        if (attention_mask == nullptr) {
          attention_mask = variant->getInput(param.attention_mask_tensor_name);
        }

        if (kv_ctx != 0 && attention_mask != nullptr) {
          int32_t group_ctx = static_cast<int32_t>(attention_mask->dims.getMaxDim());
          if (kv_ctx == group_ctx) {
            m_cache_group_use_scatter[prefix] = true;
          } else if (kv_ctx == group_ctx - n_tokens) {
            m_cache_group_use_scatter[prefix] = false;
          } else {
            std::string err_msg = "Could not determine whether KV$ uses Scatter or Concat. ";
            err_msg += fmt::format("KV$ has input dimension {}.", kv_ctx);
            err_msg += fmt::format("Expected CL={} or CL - AR-n={}", ctx_size, ctx_size - n_tokens);
            QNN_ERROR("%s", err_msg.c_str());
            State::error(err_msg);
            return false;
          }
          m_cache_group_ctx_size[prefix] = size_t(group_ctx);
          detected                       = true;
          break;
        }
      }
      if (detected) break;
    }

    // Iterate across all [AR-n, CL] to determine variant mapping
    __DEBUG("Mapping for Cachegroup {}", prefix);
    bool found = false;
    for (auto& nsp_graph : m_nsp_graphs) {
      auto& graph_outputs = nsp_graph.variants.begin()->second->output_specs;
      for (auto& [tname, tensor] : graph_outputs) {
        if (!tname.starts_with(prefix) || tname.find("key") == std::string::npos) {
          continue;
        }

        // Found representative tensor for this cache group
        found = true;

        const std::string keyout_name = tname;
        const std::string keyin_name  = toInput(tname);

        for (auto& [global_variant, variant] : nsp_graph.variants) {
          const auto& [global_arn, global_ctx] = global_variant;

          QnnUtils::Tensor* key_out = variant->getOutput(keyout_name);
          QnnUtils::Tensor* key_in  = variant->getInput(keyin_name);

          // Key Cache has shape input[n_heads, n_embed, n_ctx] + output [n_heads, n_embed, arn]
          bool scatter = m_cache_group_use_scatter.at(prefix);
          int32_t arn  = static_cast<int32_t>(key_out->dims.channel);
          int32_t ctx =
              !key_in ? arn : (static_cast<int32_t>(key_in->dims.channel) + (scatter ? 0 : arn));
          if (variant->variantType == GraphType::DECODER_PREFILL) {
            ctx = m_cache_group_ctx_size[prefix];
          }
          __DEBUG("Found AR-{} CL-{} -> AR-{} CL-{}", global_arn, global_ctx, arn, ctx);
          m_cache_group_variant_map[prefix][global_variant] = {arn, ctx};
        }
        break;
      }

      if (found) break;
    }
  }
  return true;
}

// Run all validations for the model here so we can exit early
bool QnnNspModel::validateModel() {
  GENIE_TRACE();
  // Checks we will be running
  // 1a. input_ids or inputs_embeds exists in the first split
  // 1b. token_type_ids should exists in case of Bert
  // 2. logits exists in the last split
  // 3. Shapes for all named tensors are correct
  // 4. All tensors with identical names (incl kv_in/kv_out) have identical quantization params
  // Missing check : Shape of tensor between splits match up

  // Important : These variables need to be set correctly
  // m_vocab_size  - Calculated as max(logits.shape) since len()
  // m_kv_dim      - Calculated in this function before usage
  // m_ctx_size    - Provided by the user as n_ctx
  std::vector<std::tuple<std::string, std::string, std::string>> errors;

  QnnUtils::Tensor* tt;

  // default input type is token
  m_inputType = InputType::TOKENS;

  // detect cross attention
  for (auto& [variant_spec, variant] : m_nsp_graphs.back().variants) {
    if ((tt = variant->getOutput(m_layerNames[LayerType::CROSS_ATTN_STATES])) != nullptr) {
      if (!m_cross_attention || (m_modelArchitectureType != ModelArchitectureType::ENCODER)) {
        throw std::runtime_error("Unexpected output: " +
                                 m_layerNames[LayerType::CROSS_ATTN_STATES]);
      }
      break;
    }

    if ((tt = variant->getInput(m_layerNames[LayerType::CROSS_ATTN_STATES])) != nullptr) {
      if (!m_cross_attention || (m_modelArchitectureType != ModelArchitectureType::DECODER)) {
        throw std::runtime_error("Unexpected input: " + m_layerNames[LayerType::CROSS_ATTN_STATES]);
      }
      break;
    }
  }

  if ((m_modelArchitectureType == ModelArchitectureType::ENCODER) && m_cross_attention &&
      m_pooled_output) {
    throw std::runtime_error("pooled-output is unsupported for cross attention encoders.");
  }

  // Check 1 - input layer exists
  for (auto& [variant_spec, variant] : m_nsp_graphs.front().variants) {
    const auto& [n_tokens, ctx_size] = variant_spec;
    // Update model expectations for E2T if an inputs_embeds layer is present. marks the input Type
    if ((tt = variant->getInput("inputs_embeds")) != nullptr) {
      m_layerNames[LayerType::INPUT] = "inputs_embeds";
      m_inputType                    = InputType::EMBEDDINGS;
    } else if ((tt = variant->getInput("_model_embed_tokens_Gather_Gather_output_0")) != nullptr) {
      // workaround to support split LLM (LUT + Decoder)
      m_layerNames[LayerType::INPUT] = "_model_embed_tokens_Gather_Gather_output_0";
      m_inputType                    = InputType::EMBEDDINGS;
    } else if ((tt = variant->getInput("_model_model_embed_tokens_Gather_Gather_output_0")) !=
               nullptr) {
      // workaround to support split LLM (LUT + Decoder)
      m_layerNames[LayerType::INPUT] = "_model_model_embed_tokens_Gather_Gather_output_0";
      m_inputType                    = InputType::EMBEDDINGS;
    } else if ((tt = variant->getInput("_embed_tokens_Gather_Gather_output_0")) != nullptr) {
      // workaround to support split LLM (LUT + Decoder)
      m_layerNames[LayerType::INPUT] = "_embed_tokens_Gather_Gather_output_0";
      m_inputType                    = InputType::EMBEDDINGS;
    } else if ((tt = variant->getInput("_model_embedding_concat_Concat_Concat_output_0")) !=
               nullptr) {
      // workaround to support split LLM (LUT + Decoder)
      m_layerNames[LayerType::INPUT] = "_model_embedding_concat_Concat_Concat_output_0";
      m_inputType                    = InputType::EMBEDDINGS;
    }
    if ((tt = variant->getInput(m_layerNames[LayerType::INPUT])) == nullptr) {
      errors.push_back({variant->graph_name, m_layerNames[LayerType::INPUT], "Tensor not found"});
    } else {
      input_bitwidth = tt->dtype.bw();
      checkShape(m_layerNames[LayerType::INPUT],
                 tt,
                 -1,
                 -1,
                 -1,
                 static_cast<int32_t>(input_bitwidth),
                 errors);

      if (embedding_datatype == "QNN_DATATYPE_FLOAT_32") {
        m_embeddingBufferSize = m_embd_size * sizeof(float);
      } else {
        m_embeddingBufferSize = m_embd_size * input_bitwidth;
      }

      // For embedding inputs, the expected count is multiplied by the embedding size.
      size_t expectedElementCount = static_cast<size_t>(n_tokens) * m_batch_size *
                                    ((m_inputType == InputType::TOKENS) ? 1 : m_embd_size);
      if (m_layerNames[LayerType::INPUT] == "_model_embedding_concat_Concat_Concat_output_0") {
        expectedElementCount = expectedElementCount * 2;
      }
      if (tt->dims.getNumElements() != expectedElementCount)
        errors.push_back(
            {variant->graph_name, m_layerNames[LayerType::INPUT], "Wrong input shape"});
    }
  }

  // Check 1b - In case of BERT :-> token_type_ids
  if ((m_modelArchitectureType == ModelArchitectureType::ENCODER) && !m_cross_attention) {
    for (auto& [variant_spec, variant] : m_nsp_graphs.front().variants) {
      const auto& [n_tokens, ctx_size] = variant_spec;
      if ((tt = variant->getInput(m_layerNames[LayerType::TOKEN_TYPE_IDS])) == nullptr)
        errors.push_back(
            {variant->graph_name, m_layerNames[LayerType::TOKEN_TYPE_IDS], "Tensor not found"});
      else {
        checkShape(m_layerNames[LayerType::TOKEN_TYPE_IDS], tt, -1, -1, -1, 4, errors);
        if (tt->dims.getNumElements() != static_cast<uint32_t>(n_tokens))
          errors.push_back({variant->graph_name,
                            m_layerNames[LayerType::TOKEN_TYPE_IDS],
                            "Wrong token_type_ids shape"});
      }
    }
  }

  // Check 2 - In case of LLama :-> logits exists
  //           In case of BERT :-> pooled_output & sequence_outputs exists
  //           In case of MT5 cross encoder :-> cross_attention_states exists
  for (auto& [variant_spec, variant] : m_nsp_graphs.back().variants) {
    const auto& [n_tokens, ctx_size] = variant_spec;
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
      if (m_cross_attention) {
        if ((tt = variant->getOutput(m_layerNames[LayerType::CROSS_ATTN_STATES])) == nullptr)
          errors.push_back({variant->graph_name,
                            m_layerNames[LayerType::CROSS_ATTN_STATES],
                            "Tensor not found"});
        else {
          if (tt->dims.channel != m_embd_size)
            errors.push_back({variant->graph_name,
                              m_layerNames[LayerType::CROSS_ATTN_STATES],
                              "Wrong cross_attetnion_states shape"});
        }
        continue;
      }
      if ((tt = variant->getOutput(m_layerNames[LayerType::POOL_OUTPUT])) == nullptr)
        errors.push_back(
            {variant->graph_name, m_layerNames[LayerType::POOL_OUTPUT], "Tensor not found"});
      else {
        if (tt->dims.getNumElements() != m_embd_size)
          errors.push_back({variant->graph_name,
                            m_layerNames[LayerType::POOL_OUTPUT],
                            "Wrong pooled_outputs shape"});
      }
      if (!m_pooled_output) {
        if ((tt = variant->getOutput(m_layerNames[LayerType::SEQ_OUTPUT])) == nullptr)
          errors.push_back(
              {variant->graph_name, m_layerNames[LayerType::SEQ_OUTPUT], "Tensor not found"});
        else {
          if (tt->dims.getNumElements() != static_cast<uint32_t>(n_tokens) * m_embd_size)
            errors.push_back({variant->graph_name,
                              m_layerNames[LayerType::SEQ_OUTPUT],
                              "Wrong sequence_output shape"});
        }
      }
    } else {
      if (variant->variantType != GraphType::DECODER_PREFILL) {
        if ((tt = variant->getOutput(m_layerNames[LayerType::OUTPUT])) == nullptr)
          errors.push_back(
              {variant->graph_name, m_layerNames[LayerType::OUTPUT], "Tensor not found"});
        else {
          m_vocab_size = (m_vocab_size == 0) ? tt->dims.getMaxDim() : m_vocab_size;
          if ((tt->dims.getNumElements() / m_batch_size) != m_vocab_size &&
              (tt->dims.getNumElements() / m_batch_size) !=
                  m_vocab_size * static_cast<uint32_t>(n_tokens)) {
            errors.push_back(
                {variant->graph_name, m_layerNames[LayerType::OUTPUT], "Wrong logits shape"});
          }
        }
      }
    }
  }

  // Check 3 - Shapes for all names tensors are correct
  if (m_kv_dim == -1) {  // Deduce KV$ embed_dim if not already available
    for (auto& variant : m_variant_list) {
      for (auto& [tname, tspec] : variant.output_specs) {
        if (tname.starts_with("past_key")) {
          m_kv_dim = static_cast<int32_t>(tspec.dims.width);
          if (m_batch_size != static_cast<size_t>(tspec.dims.batch)) {
            errors.push_back({variant.graph_name,
                              tname,
                              "Input tensor batch size does not match configuration!"});
          }
        }
      }

      if (m_kv_dim != -1) {
        break;
      }
    }
  }

  for (auto& variant : m_variant_list) {
    const int32_t n_tokens = variant.n_tokens;
    const int32_t ctx_size = variant.ctx_size;

    // Verify attention mask tensors
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
      tt = variant.getInput(m_layerNames[LayerType::ATTN_MASK]);
      checkShape(m_layerNames[LayerType::ATTN_MASK], tt, 1, 1, ctx_size, -1, errors);
    } else {
      for (const auto& [prefix, param] : m_cache_group_params_map) {
        if (!param.attention_mask_tensor_name.empty()) {
          tt = variant.getInput(param.attention_mask_tensor_name);
          const auto& [group_arn, group_ctx] =
              m_cache_group_variant_map.at(prefix).at({n_tokens, ctx_size});
          checkShape(param.attention_mask_tensor_name, tt, 1, group_arn, group_ctx, -1, errors);
        }
      }
    }

    // Verify positional encoding tensors
    if (m_positional_encoding.type == PositionalEncoding::ROPE) {
      tt = variant.getInput(m_layerNames[LayerType::POS_SIN]);
      checkShape(m_layerNames[LayerType::POS_SIN],
                 tt,
                 1,
                 n_tokens,
                 static_cast<int32_t>(m_pos_dim),
                 -1,
                 errors);
      tt = variant.getInput(m_layerNames[LayerType::POS_COS]);
      checkShape(m_layerNames[LayerType::POS_COS],
                 tt,
                 1,
                 n_tokens,
                 static_cast<int32_t>(m_pos_dim),
                 -1,
                 errors);
    } else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE) {
      tt = variant.getInput(m_layerNames[LayerType::POS_IDS]);
      checkShape(m_layerNames[LayerType::POS_IDS], tt, 1, 1, n_tokens, -1, errors);
    } else if (m_positional_encoding.type == PositionalEncoding::ALIBI) {
      tt = variant.getInput(m_layerNames[LayerType::POS_IDS]);
      checkShape(m_layerNames[LayerType::POS_IDS], tt, 1, n_tokens, ctx_size, -1, errors);
    }

    // Verify KV$ tensors
    if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
      for (const auto& [prefix, param] : m_cache_group_params_map) {
        const auto& [group_arn, group_ctx] =
            m_cache_group_variant_map.at(prefix).at({n_tokens, ctx_size});
        const int32_t past_dim =
            m_cache_group_use_scatter.at(prefix) ? group_ctx : group_ctx - group_arn;

        for (auto& [tname, tspec] : variant.input_specs) {
          if (!tname.starts_with(prefix)) continue;
          if (tname.find("key") != std::string::npos)
            checkShape(tname, &tspec, -1, m_kv_dim, past_dim, -1, errors);
          else if (tname.find("value") != std::string::npos)
            checkShape(tname, &tspec, -1, past_dim, m_kv_dim, -1, errors);
        }

        for (auto& [tname, tspec] : variant.output_specs) {
          if (!tname.starts_with(prefix)) continue;
          if (tname.find("key") != std::string::npos)
            checkShape(tname, &tspec, -1, m_kv_dim, group_arn, -1, errors);
          else if (tname.find("value") != std::string::npos)
            checkShape(tname, &tspec, -1, group_arn, m_kv_dim, -1, errors);
        }
      }
    }
  }

  // skip check in case of BERT architecture since no KV cache tensors are existing
  if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
    // Check 4 - Quantization parameter match
    std::unordered_map<std::string, QnnUtils::QuantParam> quant_params;
    for (auto& variant : m_variant_list) {
      for (auto& tensor_specs : {variant.input_specs, variant.output_specs}) {
        for (auto& [tname, tspec] : tensor_specs) {
          std::string name = tname;
          if (tname.ends_with("_in")) {  // Convert [kv_prefix]*_in to [kv_prefix]*_out
            for (const auto& [prefix, param] : m_cache_group_params_map) {
              if (tname.starts_with(prefix)) {
                name = tname.substr(0, tname.rfind("_")).append("_out");
                break;
              }
            }
          }

          if (name.compare(m_layerNames[LayerType::OUTPUT]) == 0) continue;
          if (quant_params.contains(name)) {
            if (quant_params.at(name).scale != tspec.quantParam[0].scale ||
                quant_params.at(name).offset != tspec.quantParam[0].offset) {
              errors.push_back({variant.graph_name,
                                tname,
                                "Non-identical quantization parameters found for the same tensor"});
            }
          } else {
            quant_params[tname] = {tspec.quantParam[0].scale, tspec.quantParam[0].offset};
          }
        }
      }
    }
  }

  if (errors.size() > 0) {
    QNN_ERROR("Model Validation Errors found");
    for (auto& [graph_name, tensor_name, err_msg] : errors)  // Log the list of errors
      QNN_ERROR("%s : %s - %s", graph_name.c_str(), tensor_name.c_str(), err_msg.c_str());
    QNN_ERROR("Note: Dimensions denoted by '%s' are ignored (i.e. no comparison)",
              translateDim(-1).c_str());
    QNN_ERROR("Check model i/o specs (set dump-specs=true in config) for debugging");
    State::fatal("Error validating HTP models");
    return false;
  }

  return true;
}

bool QnnNspModel::initializeKVManager() {
  GENIE_TRACE();
  if (m_lazyInitialization) return true;

  static std::map<KVManagerMode, std::string> managerModeToString = {
      {KVManagerMode::POINTER_SHIFT, "POINTER_SHIFT"},
      {KVManagerMode::SHIFT_CONCAT, "SHIFT_CONCAT"},
      {KVManagerMode::SMART_MASK, "SMART_MASK"},
      {KVManagerMode::NATIVE_KV, "NATIVE_KV"}};
  __DEBUG("Initializing with KV$ update method = {}", managerModeToString[_kv_update_method]);

  m_kvmanager = std::make_shared<KVManager>(_env, m_qnnApi.get(), m_ioTensor, m_threadpool);
  // Register supported variants
  for (const auto& graph : m_nsp_graphs) {
    for (const auto& [_, variant] : graph.variants) {
      if (variant->ctx_size != -1) {
        m_kvmanager->registerSupportedVariant(
            variant->n_tokens, variant->ctx_size, variant->variantType);
      }
    }
  }

  // TODO: Select largest CL but smallest AR. We want both KV$ input and output tensors here

  // Pick largest variant/context size. This is not important for tensor mapping since
  // all buffers link to the same address anyway, but it will be important for scorer validation.
  const auto [n_tokens, ctx_size] = nsp_graph_count.rbegin()->first;
  // Initialize each cache group
  // A cache group is created for each "unique" set of KV$ Tensors
  // Unique here is defined by a difference in context size, long context modes, etc.
  // Each CacheGroup is associated with its own prefix (e.g. past_ , swa_)
  // The default CacheGroup is defined by the past_ prefix
  std::map<std::string, CacheGroup>& cache_groups = m_kvmanager->getCacheGroups();
  std::map<std::string,
           std::map<int, std::map<uint32_t, std::array<std::pair<QnnUtils::Tensor*, size_t>, 4>>>>
      group_kv_tensors;
  for (const auto& [prefix, param] : m_cache_group_params_map) {
    // Collect all KV$ tensors associated with this CacheGroup prefix
    auto& kv_map = group_kv_tensors[prefix];
    for (auto& graph : m_nsp_graphs) {
      if (!graph.variants.contains({n_tokens, ctx_size})) continue;
      GraphVariant* variant = graph(n_tokens, ctx_size);
      for (auto& [tname, tensor] : variant->output_specs) {
        if (!tname.starts_with(prefix) || tname.find("key") == std::string::npos) {
          continue;
        }

        const uint32_t index = QnnUtils::parseLayerIndex(tname);
        auto key_out_tensor  = variant->getOutput(tname);
        auto key_in_tensor   = variant->getInput(toInput(tname));
        auto val_out_tensor  = variant->getOutput(toVal(tname));
        // Get prefix for key input tensor and check if it exists in m_cache_group_ctx_size
        std::string key_in_prefix = QnnUtils::getPrefix(toInput(tname), m_cache_group_prefixes);
        size_t key_val_ctx_size   = 0;
        if (!key_in_prefix.empty() && m_cache_group_ctx_size.contains(key_in_prefix)) {
          key_val_ctx_size =
              m_cache_group_ctx_size[key_in_prefix] -
              (m_cache_group_use_scatter[key_in_prefix] ? 0 : key_out_tensor->dims.channel);
          if (!key_in_tensor &&
              variant->variantType != GraphType::DECODER_PREFILL) {  // Bert-kv models
            key_val_ctx_size = 0;
          }
        } else {
          // Use a fallback value (context size from the variant) and log a warning
          key_val_ctx_size = static_cast<size_t>(ctx_size);
          __WARN("Missing context size for key input prefix: {}. Using fallback value: {}",
                 key_in_prefix.empty() ? "empty" : key_in_prefix,
                 ctx_size);
        }
        size_t key_in_size = key_val_ctx_size * key_out_tensor->dims.batch *
                             key_out_tensor->dims.height * key_out_tensor->dims.width *
                             key_out_tensor->dims.bitwidth;

        size_t val_in_size = key_val_ctx_size * val_out_tensor->dims.batch *
                             val_out_tensor->dims.height * val_out_tensor->dims.channel *
                             val_out_tensor->dims.bitwidth;
        if (variant->variantType == GraphType::DECODER_PREFILL) {
          const auto [min_variant, min_ctx_size] = nsp_graph_count.begin()->first;
          GraphVariant* variantDecoder =
              graph(min_variant,
                    min_ctx_size);  // Use Decoder Graph variant for initializing KV IN tensors
          kv_map[graph.idx()][index] = std::array<std::pair<QnnUtils::Tensor*, size_t>, 4>{
              std::make_pair(variantDecoder->getInput(toInput(tname)), key_in_size),
              std::make_pair(&tensor, tensor_alloc_info[tname].second),
              std::make_pair(variantDecoder->getInput(toVal(toInput(tname))), val_in_size),
              std::make_pair(variant->getOutput(toVal(tname)),
                             tensor_alloc_info[toVal(tname)].second)};
        } else {
          kv_map[graph.idx()][index] = std::array<std::pair<QnnUtils::Tensor*, size_t>, 4>{
              std::make_pair(variant->getInput(toInput(tname)), key_in_size),
              std::make_pair(&tensor, tensor_alloc_info[tname].second),
              std::make_pair(variant->getInput(toVal(toInput(tname))), val_in_size),
              std::make_pair(variant->getOutput(toVal(tname)),
                             tensor_alloc_info[toVal(tname)].second)};
        }

        if (variant->variantType != GraphType::DECODER_PREFILL) {
          if (kv_map.at(graph.idx()).at(index)[3].first == nullptr) {
            uint16_t layer_idx = index >> 16, head_idx = index & 0xffff;
            State::error(fmt::format("Error in layer {} head {}. ", layer_idx, head_idx) +
                         fmt::format("Found Key {} but no Value {}", tname, toVal(tname)));
            return false;
          }
        }
      }
    }
    bool use_scatter = false;
    if (kv_map.empty()) {
      if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
        State::error(fmt::format("Invalid cache-group prefix detected: {}", prefix));
        return false;
      }
    } else {
      use_scatter = m_cache_group_use_scatter.at(prefix);
    }
    cache_groups.emplace(prefix, CacheGroup(_env, prefix, use_scatter, param.longcontext_params));
  }
  // Register KVTensors into each CacheGroup
  for (auto& [prefix, cache_group] : cache_groups) {
    cache_group.context_manager->cache_group = &cache_group;

    auto& kv_map = group_kv_tensors[prefix];
    cache_group.registerTensors(kv_map);

    cache_group.m_variant_map = m_cache_group_variant_map.at(prefix);

    if (cache_group.context_manager->params.mode != LongContextParams::KEYDIFF) continue;

    // Get buffers for anchor input, anchor output and score tensors
    // Get the allocation information for Anchor input buffer and Key Cache Buffer
    std::map<uint32_t, std::array<std::tuple<int, size_t>, 2>> scorer_allocs;
    std::map<uint32_t, std::array<QnnUtils::Tensor*, 2>> anchor_tensors;
    for (auto& graph : m_nsp_graphs) {
      GraphVariant* variant = graph(n_tokens, ctx_size);
      for (auto& [tname, tensor] : variant->input_specs) {
        if (!tname.starts_with(m_layerNames.at(LayerType::ANCHOR))) continue;
        const uint32_t index  = QnnUtils::parseLayerIndex(tname);
        anchor_tensors[index] = {&tensor, variant->getOutput(toOutput(tname))};
        scorer_allocs[index]  = {graph.tensor_alloc_info->at(tname),
                                 graph.tensor_alloc_info->at(QNN_TENSOR_GET_NAME(
                                    kv_map.at(graph.idx()).at(index)[1].first->tensor))};
      }
    }
    // Initialize all tensors for the scorer model (anchor/keys/scores)
    // Also add the score buffer pointer associated with each
    std::string scorer_path =
        (model_basedir / fs::path(cache_group.context_manager->params.scoring_network)).string();
    __DEBUG("Initializing KeyDiff Scorer {}", scorer_path);
    std::map<uint32_t, uint8_t*> score_memptr;
    if (!m_qnnApi->initializeScorer(scorer_path,
                                    scorer_allocs,
                                    score_memptr,
                                    static_cast<size_t>(ctx_size),
                                    m_expectedDataFormat)) {
      State::error("Failed to initialize scorer");
      return false;
    }
    __DEBUG("cache group = {:p} keydiff.group={:p}",
            fmt::ptr(&cache_group),
            fmt::ptr(cache_group.context_manager->cache_group));

    __DEBUG("anchor_tensors = [");
    for (auto& [index, anchor_io] : anchor_tensors)
      __DEBUG("\t{}: [{}, {}, {}] {}",
              index,
              fmt::ptr(anchor_io[0]),
              fmt::ptr(anchor_io[1]),
              fmt::ptr(score_memptr.at(index)),
              cache_group.m_tensor_index.contains(index));
    __DEBUG("]");
    auto keydiff = dynamic_cast<KeyDiff*>(cache_group.context_manager.get());
    keydiff->registerKeydiffBuffers(anchor_tensors, score_memptr);
    __DEBUG("Completed registerKeydiffBuffers");

    for (auto& [index, t] : cache_group.m_tensor_index) {
      __DEBUG("\t{}:[anchor in={:p} out={:p} score={:p}]",
              index,
              fmt::ptr(t->anchor_tensor_in),
              fmt::ptr(t->anchor_tensor_out),
              fmt::ptr(t->scores));
    }
  }

  if (_kv_update_method == KVManagerMode::NATIVE_KV) {
    std::map<std::pair<int32_t, int32_t>, bool> isKvOutputNativeFormat;
    for (auto& [prefix, cache_group] : cache_groups) {
      bool found_decoder_layer = false;
      for (auto& select_graph : m_nsp_graphs) {
        if (found_decoder_layer) break;
        for (const auto& [key, value] : nsp_graph_count) {
          int32_t var           = key.first;
          int32_t ctx           = key.second;
          GraphVariant* variant = select_graph(var, ctx);
          if (variant->variantType != GraphType::DEFAULT &&
              variant->variantType != GraphType::DECODER &&
              variant->variantType != GraphType::DECODER_PREFILL)
            break;
          found_decoder_layer = true;
          for (auto& [mtname, mtensor] : variant->output_specs) {
            if (mtname.starts_with(prefix) && QnnUtils::isKVTensor(mtname)) {
              isKvOutputNativeFormat[key] =
                  mtensor.tensor->v1.dataFormat == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT;
              if (!isKvOutputNativeFormat[key]) {
                __WARN("The graph {}'s KVCache has Native input and FlatBuffer output",
                       variant->graph_name);
              }
              break;
            }
          }
        }
      }
      cache_group.registerKvOutputNativeFormat(isKvOutputNativeFormat);
    }
  }
  m_kvmanager->initComplete(m_ctx_size, m_default_group);
  if (m_kvmanager->failed()) {
    State::fatal(m_kvmanager->error());
    return false;
  }
  m_kvmanager->dispatchUpdate(0);
  if (m_kvmanager->failed()) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  // Detect which variants have logits outputs
  std::set<std::pair<int32_t, int32_t>> logit_containing_variants;
  QnnNspGraph& last_split = m_nsp_graphs.back();
  for (auto& [variant_spec, graph_variant] : last_split.variants) {
    QnnUtils::Tensor* logit_tensor = graph_variant->getOutput(m_layerNames[LayerType::OUTPUT]);
    if (logit_tensor != nullptr) {
      logit_containing_variants.insert(variant_spec);
    }
  }
  m_kvmanager->registerLogitVariants(logit_containing_variants);

  return true;
}

inline bool QnnNspModel::updateTensorPointer(GraphVariant& variant,
                                             std::string& key,
                                             QnnUtils::Tensor*& t) {
  QnnUtils::Tensor* tensor_ptr = variant.getInput(key);
  if (tensor_ptr == nullptr) return true;
  if (t == nullptr) t = tensor_ptr;
  if (getBuffer(t) == getBuffer(tensor_ptr)) {
    if (getBufferSize(tensor_ptr) > getBufferSize(t)) {
      t = tensor_ptr;
    }
    return true;
  }
  __ERROR("{} has different addresses: {} vs {}",
          key,
          static_cast<void*>(t),
          static_cast<void*>(tensor_ptr));
  return false;
}

bool QnnNspModel::initializeTensorPointers() {
  GENIE_TRACE();
  // Ideally this needs to be done for all sets of AR-n available, e.g. for AR-1 and AR-1024
  if (m_lazyInitialization) return true;

  bool status = true;

  for (auto& variant : m_variant_list) {
    status &= updateTensorPointer(variant, m_layerNames[LayerType::INPUT], t_input_ids);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::ATTN_MASK], t_attn_mask);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::VALID_MASK], t_valid_mask);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::POS_SIN], t_position_ids_sin);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::POS_COS], t_position_ids_cos);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::POS_IDS], t_position_ids);
    status &=
        updateTensorPointer(variant, m_layerNames[LayerType::TOKEN_TYPE_IDS], t_token_type_ids);
    status &= updateTensorPointer(
        variant, m_layerNames[LayerType::CROSS_ATTN_STATES], t_cross_attn_states);
    status &=
        updateTensorPointer(variant, m_layerNames[LayerType::CROSS_ATTN_MASK], t_cross_attn_mask);
    status &= updateTensorPointer(
        variant, m_layerNames[LayerType::FULL_TEXT_ROW_MASK], t_full_text_row_mask);
    // Eaglet draft/target feature input tensors
    status &= updateTensorPointer(
        variant, m_layerNames[LayerType::EAGLET_DM_HIDDEN_STATES_IN], t_dm_draft_feature_in);
    status &= updateTensorPointer(
        variant, m_layerNames[LayerType::EAGLET_TM_HIDDEN_STATES_IN], t_dm_target_feature_in);
  }
  if (!status) __ERROR("qnn-htp: Error in setting up named tensor pointers.");

  // Eaglet DM Input Tensors
  if (t_dm_target_feature_in == nullptr && t_dm_draft_feature_in != nullptr) {
    t_dm_target_feature_in = t_dm_draft_feature_in;
    t_dm_draft_feature_in  = nullptr;
  }

  // Get the largest tensor for DM feature output
  for (auto& variant : m_variant_list) {
    QnnUtils::Tensor* tensor_ptr =
        variant.getOutput(m_layerNames[LayerType::EAGLET_DM_HIDDEN_STATES_OUT]);
    if (tensor_ptr != nullptr && (t_dm_feature_out == nullptr ||
                                  getBufferSize(tensor_ptr) > getBufferSize(t_dm_feature_out))) {
      t_dm_feature_out    = tensor_ptr;
      dm_draftFeatureSize = t_dm_feature_out->dims.channel * t_dm_feature_out->dtype.bw();
    }
  }

  // Find tensors for each group, iff it's been provided via the user config
  for (auto& [prefix, param] : m_cache_group_params_map) {
    QnnUtils::Tensor* group_attn_mask{nullptr};
    QnnUtils::Tensor* group_cache_index{nullptr};
    for (auto& variant : m_variant_list) {
      if (!param.attention_mask_tensor_name.empty()) {
        status &= updateTensorPointer(variant, param.attention_mask_tensor_name, group_attn_mask);
      }
      if (!param.cache_index_tensor_name.empty()) {
        status &= updateTensorPointer(variant, param.cache_index_tensor_name, group_cache_index);
      }
    }

    if (!param.attention_mask_tensor_name.empty()) {
      if (!group_attn_mask) {
        status = false;
        __ERROR(
            "Couldn't find attn mask {} for group {}", param.attention_mask_tensor_name, prefix);
      } else {
        m_group_attn_mask[prefix] = group_attn_mask;
      }
    }
    if (!param.cache_index_tensor_name.empty()) {
      if (!group_cache_index) {
        status = false;
        __ERROR("Couldn't find cache-index {} for group {}", param.cache_index_tensor_name, prefix);
      } else {
        m_group_cache_index[prefix] = group_cache_index;
      }
    }
  }

  status &= !(!t_input_ids || !t_attn_mask);
  if (!t_input_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::INPUT]);
  if (!t_attn_mask) __ERROR("Tensor not found: {}", m_layerNames[LayerType::ATTN_MASK]);

  if (m_modelArchitectureType == ModelArchitectureType::ENCODER &&
      !m_cross_attention) {  // This input only valid for
                             // Encoder only model like bert.
    status &= !(!t_token_type_ids);
    if (!t_token_type_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::TOKEN_TYPE_IDS]);
  }

  if (m_positional_encoding.type == PositionalEncoding::ROPE) {
    status &= !(!t_position_ids_sin || !t_position_ids_cos);
    if (!t_position_ids_sin) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_SIN]);
    if (!t_position_ids_cos) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_COS]);
  } else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE) {
    status &= !(!t_position_ids);
    if (!t_position_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_IDS]);
  } else if (m_positional_encoding.type == PositionalEncoding::ALIBI) {
    status &= !(!t_position_ids);
    if (!t_position_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_IDS]);
  } else if (m_positional_encoding.type == PositionalEncoding::NONE) {
    __WARN("Positional Encodings are disabled.");
  } else {
    __ERROR("Unknown Rope Type found for tensor: {}", m_layerNames[LayerType::POS_IDS]);
  }

  // Detect activation bitwidth
  if (status) {
    // Check Input-> Input_ID or Input_Embed
    d_input = t_input_ids->dtype;
    if (!isSupportedActivation(d_input)) {
      __ERROR("Input Tensor: {} as unsupported activation type {}",
              m_layerNames[LayerType::INPUT],
              d_input.str());
      status = false;
    }
    // Check Attention Mask
    d_attn_map = t_attn_mask->dtype;
    if (!isSupportedActivation(d_attn_map)) {
      __ERROR("attention_mask has unsupported type {}", d_attn_map.str());
      status = false;
    }

    uint32_t attn_bitwidth = d_attn_map.bw();
    bool attn_quantized    = (d_attn_map.type() != 2);
    if (attn_quantized) {
      // Support uint8, uint16 and uint32
      if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
        m_attention_positive_value.u32 = 1;  // This sets u8=u16=u32=1
      } else {
        m_attention_positive_value.u32 = 0xffffffff;  // This sets u8=0xff u16=0xffff
      }
      m_attention_negative_value.u32 = 0;  // This sets u8=u16=u32=0
    } else {
      // Support float16 or float32
      m_attention_positive_value.u32 = 0;  // Set u16=u32=0 for fp16 or fp32
      if (attn_bitwidth == 1) {            // float8 is not currently supported
        status = false;
      } else if (attn_bitwidth == 2) {
        m_attention_negative_value.u16 = fp16_ieee_from_fp32_value(-1000.0f);
      } else if (attn_bitwidth == 4) {
        float value = -1000.0f;
        std::memcpy(&m_attention_negative_value.u32, &value, sizeof(float));
      }
    }

    // For Encoder only model BERT, Check for Token_type_ids
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER && !m_cross_attention) {
      d_token_type = t_token_type_ids->dtype;
      if (!isSupportedActivation(d_token_type)) {
        __ERROR("token_type_ids has unsupported type {}", d_token_type.str());
        status = false;
      }
    }

    // For Position_IDs check data bitwidth
    if (m_positional_encoding.type == PositionalEncoding::ROPE) {
      d_pos = t_position_ids_sin->dtype;
    } else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE) {
      d_pos = t_position_ids->dtype;
    } else if (m_positional_encoding.type == PositionalEncoding::ALIBI) {
      d_pos = t_position_ids->dtype;
    }

    if (((m_positional_encoding.type == PositionalEncoding::ABSOLUTE ||
          m_positional_encoding.type == PositionalEncoding::ALIBI) &&
         d_pos != QNN_DATATYPE_INT_32) ||
        (m_positional_encoding.type == PositionalEncoding::ROPE && !isSupportedActivation(d_pos))) {
      __ERROR("position encoding tensor has unsupported type {}", d_pos.str());
      status = false;
    }

    if (t_valid_mask != nullptr && t_valid_mask->dtype != QNN_DATATYPE_UFIXED_POINT_16) {
      __ERROR("Valid mask tensor has unsupported type {}", t_valid_mask->dtype.str());
      status = false;
    }

    __DEBUG("qnn-htp datatypes: d_input {} d_attn_map {} d_pos {}",
            d_input.str(),
            d_attn_map.str(),
            d_pos.str());

    if (!status) __ERROR("Only 8-bit, 16-bit and 32-bit activations are supported");
  }

  // cache max Chunk length
  for (auto const& [variant_spec, count] : nsp_graph_count) {
    m_maxChunkedInputLength =
        std::max(m_maxChunkedInputLength, static_cast<size_t>(variant_spec.first));
  }
  return status;
}

void QnnNspModel::initializeUnconnectedInputs() {
  GENIE_TRACE();
  if (m_lazyInitialization) {
    return;
  }
  // Compile the set of tensors that should be considered connected,
  // either via expected tensor names or implicit connections.
  std::unordered_set<std::string> connectedTensors;
  for (auto& [layerType, name] : m_layerNames) {
    connectedTensors.insert(name);
  }
  // Include output tensor names from all splits. Any input
  // with the same name as an output refers to the same buffer
  // and is therefore implicitly connected.
  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.output_specs) {
      connectedTensors.insert(tname);
    }
  }
  // Include the lora alpha tensor if applicable.
  if (lora_config) {
    connectedTensors.insert(lora_config->getAlphaTensorName());
  }

  std::unordered_map<std::string, QnnUtils::Tensor*> unconnectedInputTensors;
  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.input_specs) {
      if (connectedTensors.contains(tname)) {
        continue;
      }
      // Check if the tensor belongs to a cache group.
      std::string kvPrefix = QnnUtils::getPrefix(tname, m_cache_group_prefixes);
      if (!kvPrefix.empty()) {
        continue;
      }
      // Check if it's an anchor buffer.
      if (tname.starts_with(m_layerNames[LayerType::ANCHOR])) {
        continue;
      }
      unconnectedInputTensors[tname] = (&tspec);
    }
  }

  for (auto& [tname, tspecPtr] : unconnectedInputTensors) {
    __DEBUG("Found unconnected input tensor \"{}\": Initializing to zero.", tname);
    clearBuffer(tspecPtr);
  }
}

template <typename DType>
void QnnNspModel::setupAttentionMask(const InferenceStep& step, AttentionMask& attention_mask) {
  GENIE_TRACE();
  DType pos_val, neg_val;
  if constexpr (std::is_same_v<DType, uint8_t>) {
    pos_val = m_attention_positive_value.u8;
    neg_val = m_attention_negative_value.u8;
  } else if constexpr (std::is_same_v<DType, uint16_t>) {
    pos_val = m_attention_positive_value.u16;
    neg_val = m_attention_negative_value.u16;
  } else {
    pos_val = m_attention_positive_value.u32;
    neg_val = m_attention_negative_value.u32;
  }

  const size_t variant    = static_cast<size_t>(step.variant);
  const size_t ctx_size   = static_cast<size_t>(step.ctx_size);
  const size_t n_past     = static_cast<size_t>(step.n_past);
  const size_t n_valid_kv = static_cast<size_t>(step.n_valid_kv);
  const size_t n_process  = static_cast<size_t>(step.n_process);
  const size_t past_idx   = static_cast<size_t>(step.past_idx);
  const size_t new_idx    = static_cast<size_t>(step.new_idx);

  DType* attn_buffer = reinterpret_cast<DType*>(getBuffer(t_attn_mask));

  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    size_t n_valid = n_valid_kv + n_process;
    size_t offset  = (variant == ctx_size) ? ctx_size - n_valid : 0;
    std::fill_n(attn_buffer, ctx_size, neg_val);
    std::fill_n(attn_buffer + offset, n_valid, pos_val);
    return;
  }

  // Clear entire attention buffer
  std::fill_n(attn_buffer, variant * ctx_size, neg_val);

  // Fill the attention mask row-by-row
  for (size_t i = 0; i < n_process; i++) {
    attention_mask.fillAttentionRow<DType>(std::span<DType>(&attn_buffer[i * ctx_size], ctx_size),
                                           i,
                                           n_past,
                                           n_valid_kv,
                                           past_idx,
                                           new_idx,
                                           pos_val);
  }

  // Handle attention masks for non-default cache groups
  for (auto& [prefix, param] : m_cache_group_params_map) {
    if (prefix == m_default_group || !m_group_attn_mask.contains(prefix)) {
      continue;
    }

    if (param.longcontext_params.mode != LongContextParams::SLIDING_WINDOW) {
      __ERROR("CacheGroup-specific attention mask only supported for SWA Cache groups");
      continue;
    }

    // Step 1: Get the tensor for this groups attention mask
    // Step 2: Parse the indexes based on the group's current state
    // Step 3: Construct the group attention mask based on the global attention mask

    // Step 1
    DType* group_attn_buffer = static_cast<DType*>(getBuffer(m_group_attn_mask[prefix]));

    // Step 2
    CacheGroup& group = m_kvmanager->getCacheGroups().at(prefix);
    const std::vector<std::pair<int32_t, size_t>>& gather_indexes =
        group.context_manager->translateAttentionMask(step);
    __DEBUG("SWA Gather index = {}", gather_indexes);

    // Calculate an exclusion zone, so we don't attend more tokens than our max_attention_span
    const int32_t max_attention_span = param.longcontext_params.window_size;
    const int32_t exclusion_start    = std::max(step.n_valid_kv - group.m_n_valid_kv,
                                             static_cast<int32_t>(_size_to_skip_kv_prefix));
    const int32_t exclusion_end      = step.n_valid_kv - max_attention_span + 1;

    // Step 3
    std::fill_n(group_attn_buffer, group.m_cur_variant * group.m_cur_ctx, neg_val);
    const auto& position_ids =
        attention_mask.getPositionIds(n_past - attention_mask.get_n_past(), n_process, variant);
    for (size_t i = 0; i < n_process; i++) {
      size_t row_offset               = 0;
      const int32_t row_exclusion_end = exclusion_end + position_ids[i] - position_ids[0];

      // Check if exclusion zone should be disabled
      const bool disable_exclusion =
          (row_exclusion_end <= exclusion_start || max_attention_span <= 0);

      // If exclusion zone is invalid or max_attention_span is 0, set it beyond the context size to
      // disable exclusion
      const size_t exclude_start_idx =
          static_cast<size_t>(disable_exclusion ? step.ctx_size : exclusion_start);
      const size_t exclude_end_idx =
          static_cast<size_t>(disable_exclusion ? step.ctx_size : row_exclusion_end);

      for (const auto& [offset, count] : gather_indexes) {
        if (offset < 0) {
          row_offset += count;
          continue;
        }

        const size_t local_ctx        = static_cast<size_t>(group.m_cur_ctx);
        const size_t global_start_idx = static_cast<size_t>(offset);
        const size_t global_end_idx   = global_start_idx + count;

        // Check for overlap with exclusion zone
        const bool has_overlap =
            !(exclude_end_idx <= global_start_idx || global_end_idx <= exclude_start_idx);

        if (!has_overlap) {
          // No overlap - copy entire range
          std::memcpy(&group_attn_buffer[i * local_ctx + row_offset],
                      &attn_buffer[i * ctx_size + global_start_idx],
                      count * sizeof(DType));
        } else {
          // Handle overlap by copying before and after exclusion zone

          // Copy segment before exclusion zone
          if (global_start_idx < exclude_start_idx) {
            const size_t before_count = exclude_start_idx - global_start_idx;
            std::memcpy(&group_attn_buffer[i * local_ctx + row_offset],
                        &attn_buffer[i * ctx_size + global_start_idx],
                        before_count * sizeof(DType));
          }

          // Copy segment after exclusion zone
          if (exclude_end_idx < global_end_idx) {
            const size_t after_count = global_end_idx - exclude_end_idx;
            const size_t dst_offset  = row_offset + (exclude_end_idx - global_start_idx);

            std::memcpy(&group_attn_buffer[i * local_ctx + dst_offset],
                        &attn_buffer[i * ctx_size + exclude_end_idx],
                        after_count * sizeof(DType));
          }
        }

        row_offset += count;
      }
    }
  }

  // Handle sliding-window attention in the main cache group
  for (auto& [prefix, param] : m_cache_group_params_map) {
    if (prefix != m_default_group ||
        param.longcontext_params.mode != LongContextParams::SLIDING_WINDOW) {
      continue;
    }

    CacheGroup& group = m_kvmanager->getCacheGroups().at(prefix);
    const std::vector<std::pair<int32_t, size_t>>& gather_indexes =
        group.context_manager->translateAttentionMask(step);

    // Gather indexes is an ordered list of attention mask columns
    // Check which columns need to be translated based on gather_indexes
    size_t start_gather_index_idx = 0;
    int32_t expected_cache_idx    = 0;
    for (const auto& [offset, count] : gather_indexes) {
      if (offset != expected_cache_idx) {
        break;
      }
      expected_cache_idx += static_cast<int32_t>(count);
      start_gather_index_idx++;
    }

    // If all columns are already in the correct position, simply continue
    if (expected_cache_idx + step.variant - step.n_process >= step.ctx_size) {
      break;
    }

    // We need to shuffle columns starting at expected_cache_idx
    // Let's create a scratch buffer to copy entries into
    if (_attention_scratch == nullptr) {
      size_t max_ctx_size = 0;  // Repeated calculation to clean up
      for (auto& [variant_spec, count] : nsp_graph_count) {
        max_ctx_size = std::max(max_ctx_size, static_cast<size_t>(variant_spec.second));
      }
      _attention_scratch = malloc(max_ctx_size * sizeof(DType));
      if (_attention_scratch == nullptr) {
        throw std::runtime_error("Failed to allocate attention scratch buffer");
      }
    }

    // Iterate across attention mask rows and shuffle it around
    const size_t row_start_idx = static_cast<size_t>(expected_cache_idx);
    const size_t shuffle_size  = (ctx_size - row_start_idx) * sizeof(DType);
    DType* attn_scratch        = reinterpret_cast<DType*>(_attention_scratch);
    for (size_t i = 0; i < n_process; i++) {
      DType* attn_row = &attn_buffer[i * ctx_size];
      std::memcpy(attn_scratch, &attn_row[row_start_idx], shuffle_size);

      size_t row_offset = row_start_idx;
      for (size_t gather_index_idx = start_gather_index_idx;
           gather_index_idx < gather_indexes.size();
           gather_index_idx++) {
        const auto& [offset, count] = gather_indexes[gather_index_idx];
        if (offset < 0) {
          std::fill_n(&attn_row[row_offset], count, neg_val);
        } else {
          const size_t scratch_idx = static_cast<size_t>(offset) - row_start_idx;
          std::memcpy(&attn_row[row_offset], &attn_scratch[scratch_idx], count * sizeof(DType));
        }
        row_offset += count;
      }
    }
  }
}

template <typename DType>
void QnnNspModel::setupAttentionMask(const InferenceStep& step,
                                     AttentionMask& attention_mask,
                                     std::vector<size_t>& tokenNumPerBatch) {
  GENIE_TRACE();
  DType pos_val, neg_val;
  if constexpr (std::is_same_v<DType, uint8_t>) {
    pos_val = m_attention_positive_value.u8;
    neg_val = m_attention_negative_value.u8;
  } else if constexpr (std::is_same_v<DType, uint16_t>) {
    pos_val = m_attention_positive_value.u16;
    neg_val = m_attention_negative_value.u16;
  } else {
    pos_val = m_attention_positive_value.u32;
    neg_val = m_attention_negative_value.u32;
  }

  const size_t variant    = static_cast<size_t>(step.variant);
  const size_t ctx_size   = static_cast<size_t>(step.ctx_size);
  const size_t n_valid_kv = static_cast<size_t>(step.n_valid_kv);
  const size_t n_process  = static_cast<size_t>(step.n_process);
  const size_t past_idx   = static_cast<size_t>(step.past_idx);
  const size_t new_idx    = static_cast<size_t>(step.new_idx);

  if (m_batch_prompt_token_lens.empty()) {
    set_n_batch_inputs(tokenNumPerBatch);
  }

  DType* attn_buffer = reinterpret_cast<DType*>(getBuffer(t_attn_mask));

  // Clear entire attention buffer
  std::fill_n(attn_buffer, variant * ctx_size * m_batch_size, neg_val);
  // Fill the attention mask row-by-row
  for (size_t batch_idx = 0; batch_idx < m_n_queryBatch; batch_idx++) {
    for (size_t i = 0; i < n_process; i++) {
      attention_mask.fillAttentionRow<DType>(
          std::span<DType>(&attn_buffer[i * ctx_size + batch_idx * variant * ctx_size], ctx_size),
          i,
          m_batch_prompt_token_lens.at(batch_idx),
          m_max_batch_prompt_token_len,
          n_valid_kv,
          past_idx,
          new_idx,
          pos_val);
    }
  }

  // Handle attention masks for non-default cache groups
  for (auto& [prefix, param] : m_cache_group_params_map) {
    if (prefix == m_default_group || !m_group_attn_mask.contains(prefix)) {
      continue;
    }

    if (param.longcontext_params.mode != LongContextParams::SLIDING_WINDOW) {
      __ERROR("CacheGroup-specific attention mask only supported for SWA Cache groups");
      continue;
    }

    __ERROR(
        "Long context and sliding window attention are not supported for multi-batch processing.");
    throw std::runtime_error("Sliding window attention is not supported in multi-batch queries");
  }
}

template <typename DType>
bool QnnNspModel::setupAlibiPositionEmbedding(const InferenceStep& step) {
  DType* alibi_buffer = reinterpret_cast<DType*>(getBuffer(t_position_ids));
  const DType pad_val = static_cast<DType>(step.ctx_size);

  // Clear alibi buffer
  std::fill_n(alibi_buffer, step.variant * step.ctx_size, pad_val);

  // Detect start of past tokens and new tokens based on ctx_size and n_tokens (variant)
  DType* alibi_past = &alibi_buffer[step.past_idx];
  DType* alibi_new  = &alibi_buffer[step.new_idx];

  // Fill alibi positions from [-n_past-i, -i) and [-i, 0]
  for (int i = 0; i < step.n_process; i++) {
    std::iota(std::reverse_iterator<DType*>(alibi_past + step.n_past),
              std::reverse_iterator<DType*>(alibi_past),
              i + 1);  // Fill past tokens
    std::iota(std::reverse_iterator<DType*>(alibi_new + i + 1),
              std::reverse_iterator<DType*>(alibi_new),
              0);  // Fill new tokens

    alibi_past += step.ctx_size;  // Update pointers to next row
    alibi_new += step.ctx_size;
  }

  return true;
}

bool QnnNspModel::setupInputEmbeddings(const InferenceStep& step,
                                       size_t start,
                                       size_t eagletStartIdxOffset,
                                       bool isEagletDMMergedInput,
                                       std::vector<uint8_t>& embeddings) {
  const size_t variant   = static_cast<size_t>(step.variant);
  const size_t n_process = static_cast<size_t>(step.n_process);

  if (m_eosEmbedding.empty()) {
    __ERROR("SetupInput: EOS Embedding data is NULL.");
    return false;
  }

  if (embedding_datatype == "QNN_DATATYPE_FLOAT_32" || embedding_datatype == "float32") {
    // Quantize the embedding input into the embedding buffer
    float* embeddingSrc = reinterpret_cast<float*>(embeddings.data());
    quantizeInput(&embeddingSrc[start * m_embd_size], 0, n_process * m_embd_size);

    // Fill up the remaining buffer with EOS embedding.
    // TODO: This can be optimized by caching the quantized EOS and running memcpy()
    embeddingSrc = reinterpret_cast<float*>(m_eosEmbedding.data());
    for (size_t i = n_process; i < variant; i++) {
      quantizeInput(embeddingSrc, i * m_embd_size, m_embd_size);
    }
  } else {
    // Copy the quantized embedding input into the embedding buffer
    uint8_t* embeddingSrc         = &embeddings[start * m_embeddingBufferSize];
    uint8_t* inputEmbeddingBuffer = reinterpret_cast<uint8_t*>(getBuffer(t_input_ids)) +
                                    eagletStartIdxOffset * m_embeddingBufferSize;

    const size_t bufferStridePerToken = m_embeddingBufferSize * (isEagletDMMergedInput ? 2 : 1);

    for (size_t i = 0; i < n_process; i++) {
      std::copy(&embeddingSrc[i * m_embeddingBufferSize],
                &embeddingSrc[(i + 1) * m_embeddingBufferSize],
                &inputEmbeddingBuffer[i * bufferStridePerToken]);
    }

    for (size_t i = n_process; i < variant; i++) {
      std::copy(&m_eosEmbedding[0],
                &m_eosEmbedding[m_embeddingBufferSize],
                &inputEmbeddingBuffer[i * bufferStridePerToken]);
    }
  }

  return true;
}

bool QnnNspModel::setupEagletFeatureVectors(const InferenceStep& step,
                                            size_t start,
                                            const size_t featureCount,
                                            const uint8_t* featureVector,
                                            const std::vector<int32_t>& selected,
                                            const size_t eagletStartIdxOffset,
                                            bool post_update) {
  const size_t variant   = static_cast<size_t>(step.variant);
  const size_t n_process = static_cast<size_t>(step.n_process);

  // Feature elements per token and bytes per feature vector
  uint8_t* featureBuffer{nullptr};  // Pointer to the input feature vector tensor buffer
  size_t featureElementsPerToken;   // Dimension of the feature vector (for 1 token)
  size_t featureBytesPerToken;      // Total size of the feature vector (for 1 token)
  size_t featureStridePerToken;     // Stride size for the feature vector (for 1 token)

  // Note: featureStridePerToken is purely for Concat (embedding+feature is the same input tensor)
  // If this is depracated, code flows below can be simplified (e.g. grouped memcpy)

  // Check if eagle_feature_in comes from TM or DM
  bool isFeatureFromTM = true;
  if (featureVector != nullptr && t_dm_feature_out != nullptr) {
    const uintptr_t featureVectorAddr = reinterpret_cast<uintptr_t>(featureVector);
    const uintptr_t dmOutputStartAddr = reinterpret_cast<uintptr_t>(getBuffer(t_dm_feature_out));
    const uintptr_t dmOutputEndAddr   = dmOutputStartAddr + getBufferSize(t_dm_feature_out);

    if (featureVectorAddr >= dmOutputStartAddr && featureVectorAddr <= dmOutputEndAddr) {
      isFeatureFromTM = false;  // Feature comes from the DM output buffer
    }
  }

  // Get feature buffer pointer - either from separate buffer or concatenated with embedding
  if (t_dm_target_feature_in == nullptr) {
    // No separate tensor exists for feature input, it is concatenated with the embedding
    featureBuffer = reinterpret_cast<uint8_t*>(getBuffer(t_input_ids)) + m_embeddingBufferSize;
    featureElementsPerToken = m_embedding_length;
    featureBytesPerToken    = featureElementsPerToken * t_input_ids->dtype.bw();
    featureStridePerToken   = featureBytesPerToken * 2;
  } else if (isFeatureFromTM || t_dm_draft_feature_in == nullptr) {
    // Setup separate feature tensor, with hidden state coming from TM
    featureBuffer           = reinterpret_cast<uint8_t*>(getBuffer(t_dm_target_feature_in));
    featureElementsPerToken = m_embedding_length;
    featureBytesPerToken    = featureElementsPerToken * t_dm_target_feature_in->dtype.bw();
    featureStridePerToken   = featureBytesPerToken;
  } else {
    // Separate tensor exists for feature input coming from DM, different size from TM
    featureBuffer           = reinterpret_cast<uint8_t*>(getBuffer(t_dm_draft_feature_in));
    featureElementsPerToken = dm_draftFeatureSize / t_dm_feature_out->dtype.bw();
    featureBytesPerToken    = dm_draftFeatureSize;
    featureStridePerToken   = featureBytesPerToken;
  }

  // Handle offset in the case of draftModel KV$ disabled
  featureBuffer += eagletStartIdxOffset * featureBytesPerToken;

  // Initialize extra feature buffer (if needed)
  if (eagle_extra_feature == nullptr) {
    eagle_extra_feature = reinterpret_cast<uint8_t*>(calloc(dm_draftFeatureSize, sizeof(uint8_t)));
  }

  if (selected.empty()) {
    // Empty selection index vector is identical to [-1, 0, 1, ..., n_process]
    std::memcpy(featureBuffer, eagle_extra_feature, featureBytesPerToken);

    for (size_t i = 0; i < n_process; i++) {
      std::memcpy(&featureBuffer[i * featureStridePerToken],
                  &featureVector[i * featureBytesPerToken],
                  featureBytesPerToken);
    }

    // Store the extra feature buffer to be used in next iteration
    std::memcpy(eagle_extra_feature,
                featureBuffer + (n_process - 1) * featureStridePerToken,
                featureBytesPerToken);
  } else {
    if (selected.size() != featureCount && selected.size() != featureCount + 1) {
      __ERROR("setupInputEmbeddings ERROR: wrong selected vector size");
      return false;
    }

    const size_t featureEndIdx = std::min(start + variant, featureCount);
    const uint8_t* featureData = nullptr;
    size_t bufferIdx = 0;  // Track the input buffer index where data is being written into

    // Feature vector needs to be copied based on the provided selected index
    // Each featureBuffer index sees the hidden state generated from it's predecessor token
    // selection idx of -1 means use the last feature vector
    for (size_t j = start; j < featureEndIdx; j++) {
      if (selected[j] >= 0) {
        featureData = &featureVector[static_cast<size_t>(selected[j]) * featureBytesPerToken];
      } else {
        featureData = eagle_extra_feature;
      }

      std::memcpy(
          &featureBuffer[bufferIdx * featureStridePerToken], featureData, featureBytesPerToken);
      bufferIdx++;
    }

    // Store the extra feature buffer to be used in next iteration, if selection if is -1
    // We should never need to do this when the feature comes from the TM
    if (!post_update && !isFeatureFromTM) {
      const size_t featureIdx =
          (featureEndIdx == featureCount) ? featureCount - start : variant - 1;
      std::memcpy(eagle_extra_feature,
                  &featureVector[featureIdx * featureBytesPerToken],
                  featureBytesPerToken);
    }
  }

  return true;
}

// Utility function to "clear" a buffer, i.e. set it to all 0 values.
// Currently, only floating point tensors, and quantized fixed point tensors are supported
bool QnnNspModel::clearBuffer(QnnUtils::Tensor* tensor) {
  if (tensor == nullptr) return true;

  uint8_t* buffer         = reinterpret_cast<uint8_t*>(getBuffer(tensor));
  const size_t bufferSize = getBufferSize(tensor);
  if (tensor->dtype.type() == 2) {  // Floating point tensor. Zero out with 0s
    std::memset(buffer, 0, bufferSize);
    return true;
  }

  // For quantized tensors, clearing requires setting each value to -offset
  int32_t offset = -tensor->quantParam[0].offset;
  if (tensor->dtype.bw() == 1) {
    std::memset(buffer, static_cast<uint8_t>(offset), bufferSize);
  } else if (tensor->dtype.bw() == 2) {
    std::fill_n(reinterpret_cast<uint16_t*>(buffer), bufferSize / 2, static_cast<uint16_t>(offset));
  } else if (tensor->dtype.bw() == 4) {
    std::fill_n(reinterpret_cast<uint32_t*>(buffer), bufferSize / 4, static_cast<uint32_t>(offset));
  } else {
    return false;
  }

  return true;
}

bool QnnNspModel::setupInputTokensEmbedding(const InferenceStep& step,
                                            const size_t start,
                                            const std::vector<int32_t>& tokens,
                                            std::vector<uint8_t>& embeddings,
                                            const uint8_t* featureVector,
                                            const std::vector<int32_t>& selected,
                                            const size_t eagletStartIdxOffset,
                                            const bool post_update) {
  GENIE_TRACE();
  const size_t variant   = static_cast<size_t>(step.variant);
  const size_t ctx_size  = static_cast<size_t>(step.ctx_size);
  const size_t n_process = static_cast<size_t>(step.n_process);

  if (!tokens.empty()) {
    // Setup input id tensor
    uint32_t* input_id_buffer = reinterpret_cast<uint32_t*>(getBuffer(t_input_ids));
    std::fill_n(input_id_buffer, variant, static_cast<uint32_t>(m_pad_token));
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
      const size_t pad_offset =
          (!m_cross_attention && (variant == ctx_size)) ? variant - n_process : 0;
      std::memcpy(&input_id_buffer[pad_offset], &tokens[start], n_process * sizeof(uint32_t));
    } else if (variant == ctx_size) {
      // Special handling for AR-c models. All past tokens must be re-processed
      const size_t n_history = token_history.size();
      const size_t n_current = tokens.size();
      if (n_history != n_process - n_current) {
        __ERROR("setupInput : Missing token history. Expected {}, Found {}.",
                n_process - n_current,
                n_history);
        return false;
      }
      std::memcpy(input_id_buffer, token_history.data(), n_history * sizeof(uint32_t));
      std::memcpy(
          &input_id_buffer[n_history], &tokens[start], (n_process - n_history) * sizeof(uint32_t));
    } else {
      // For normal cases (variant < ctx_size), tokens are processed normally
      std::memcpy(input_id_buffer, &tokens[start], n_process * sizeof(uint32_t));
    }
  } else if (!embeddings.empty()) {
    // For Eaglet DM, detect if the input tensor is merged/concat (embed+hidden_state)
    bool isEagletDMMergedInput = (featureVector != nullptr && t_dm_target_feature_in == nullptr);
    if (!setupInputEmbeddings(
            step, start, eagletStartIdxOffset, isEagletDMMergedInput, embeddings)) {
      return false;
    }
  }

  if (featureVector != nullptr) {
    clearBuffer(t_dm_target_feature_in);
    clearBuffer(t_dm_draft_feature_in);
    const size_t featureCount = embeddings.size() / m_embeddingBufferSize;
    if (!setupEagletFeatureVectors(step,
                                   start,
                                   featureCount,
                                   featureVector,
                                   selected,
                                   eagletStartIdxOffset,
                                   post_update)) {
      return false;
    }
  }

  return true;
}
bool QnnNspModel::setupInput(const InferenceStep& step,
                             const size_t start,
                             const std::vector<int32_t>& tokens,
                             std::vector<uint8_t>& embeddings,
                             const uint8_t* featureVector,
                             const std::vector<int32_t>& selected,
                             const size_t eagletStartIdxOffset,
                             const bool post_update,
                             AttentionMask& attention_mask,
                             bool prefilledExecution) {
  GENIE_TRACE();
  const size_t variant   = static_cast<size_t>(step.variant);
  const size_t n_past    = static_cast<size_t>(step.n_past);
  const size_t n_process = static_cast<size_t>(step.n_process);

  if (!prefilledExecution) {
    if (!setupInputTokensEmbedding(step,
                                   start,
                                   tokens,
                                   embeddings,
                                   featureVector,
                                   selected,
                                   eagletStartIdxOffset,
                                   post_update)) {
      return false;
    }
  }

  int32_t cache_index_boundary =
      step.ctx_size -
      static_cast<int32_t>(std::ceil(static_cast<double>(step.variant) / 32.0) * 32);

  if (_kv_update_method == KVManagerMode::NATIVE_KV && step.new_idx > cache_index_boundary) {
    State::error(fmt::format("Error: cache_index {} cannot be greater than {} in native mode.",
                             step.new_idx,
                             cache_index_boundary));
    return false;
  }
  // Set up the input scatter index as new_idx (i.e. the index where new KV$ is stored)
  for (auto& [prefix, group_index_tensor] : m_group_cache_index) {
    CacheGroup& group        = m_kvmanager->getCacheGroups().at(prefix);
    InferenceStep group_step = group.translateInferenceStep(step);

    uint32_t* group_index_buffer = reinterpret_cast<uint32_t*>(getBuffer(group_index_tensor));
    std::iota(group_index_buffer,
              group_index_buffer + group_index_tensor->dims.getNumElements(),
              group_step.new_idx);
  }

  if (t_valid_mask != nullptr) {  // Set up a valid mask. Assumes u16 datatype (from validateModel)
    // Quantize mask value to u16
    const auto [scale, offset] = t_valid_mask->quantParam[0];
    const double alpha         = m_useAnchorAlpha ? m_anchorAlpha : 1.0;
    const uint16_t maskValue =
        QnnUtils::quantize<double, uint16_t>(alpha / n_process, offset, scale);

    bool hasSpeculativeTokens = false;
    if (!tokens.empty()) {
      for (int32_t token : tokens) {
        if (token >= static_cast<int32_t>(m_vocab_size)) {
          hasSpeculativeTokens = true;
          break;
        }
      }
    }

    const size_t n_masked = hasSpeculativeTokens ? 1 : n_process;
    // Setup the buffer
    uint16_t* mask_buffer = reinterpret_cast<uint16_t*>(getBuffer(t_valid_mask));
    std::fill_n(mask_buffer, n_masked, maskValue);
    std::memset(&mask_buffer[n_masked], 0, (variant - n_masked) * sizeof(uint16_t));

    // Apply anchor alpha weight on subsequent inferences.
    m_useAnchorAlpha = true;
  }

  // Setup the attention mask correctly
  if (m_cross_attention && (m_modelArchitectureType == ModelArchitectureType::ENCODER)) {
    // in cross encoder case, use tokens.size()
    setupCrossAttentionMask<int32_t>(
        step, tokens, tokens.size() + embeddings.size() / m_embeddingBufferSize);
  } else if (d_attn_map.bw() == 1) {
    setupAttentionMask<uint8_t>(step, attention_mask);
  } else if (d_attn_map.bw() == 2) {
    setupAttentionMask<uint16_t>(step, attention_mask);
  } else if (d_attn_map.bw() == 4) {
    setupAttentionMask<uint32_t>(step, attention_mask);
  }

  auto crossSize = m_cross_attention_input_size / m_embd_size;
  __DEBUG("cross attention state count: {}", crossSize);
  if (m_cross_attention && crossSize != 0) {
    auto c_attn_map_bw = t_cross_attn_mask->dtype.bw();
    if (c_attn_map_bw == 1) {
      setupCrossAttentionMask<uint8_t>(step, tokens, crossSize);
    } else if (c_attn_map_bw == 2) {
      setupCrossAttentionMask<uint16_t>(step, tokens, crossSize);
    } else if (c_attn_map_bw == 4) {
      setupCrossAttentionMask<int32_t>(step, tokens, crossSize);
    }
  }

  // Setup token type IDs
  if (m_modelArchitectureType == ModelArchitectureType::ENCODER && !m_cross_attention) {
    // BERT Specific
    uint32_t* token_type_id_buffer = reinterpret_cast<uint32_t*>(getBuffer(t_token_type_ids));
    std::memset(token_type_id_buffer, 0, variant * sizeof(uint32_t));
  }

  if (m_positional_encoding.type == PositionalEncoding::ROPE) {
    // Simple RoPE position ID setup
    const auto& position_ids = attention_mask.getPositionIds(start, n_process, variant);
    // __DEBUG("Position IDs = {}", position_ids);

    // TODO: Compile position_ids to translate from [0,1,2,3,4,0,0,...,0] -> [(0,5), (0,1),...]
    // This is to batch the memory copy calls together, which is more optimal (theoretically)
    uint8_t* cos_buffer    = reinterpret_cast<uint8_t*>(getBuffer(t_position_ids_cos));
    uint8_t* sin_buffer    = reinterpret_cast<uint8_t*>(getBuffer(t_position_ids_sin));
    const size_t rope_size = m_pos_dim * d_pos.bw();
    for (uint32_t i = 0; i < variant; i++) {
      const size_t src_offset = static_cast<size_t>(position_ids[i]) * rope_size;
      const size_t dst_offset = i * rope_size;
      std::memcpy(
          &sin_buffer[dst_offset], reinterpret_cast<uint8_t*>(rope_sin) + src_offset, rope_size);
      std::memcpy(
          &cos_buffer[dst_offset], reinterpret_cast<uint8_t*>(rope_cos) + src_offset, rope_size);
    }
  } else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE) {
    uint32_t* position_id_buffer = reinterpret_cast<uint32_t*>(getBuffer(t_position_ids));
    std::memset(position_id_buffer, 0, variant * sizeof(uint32_t));

    // Fill up position_ids buffer
    size_t pad_offset =
        (m_modelArchitectureType == ModelArchitectureType::ENCODER) ? variant - n_process : 0;
    uint32_t* pos_id_start = position_id_buffer + pad_offset;
    uint32_t* pos_id_end   = pos_id_start + n_process;
    std::iota(pos_id_start, pos_id_end, n_past);
  } else if (m_positional_encoding.type == PositionalEncoding::ALIBI) {
    setupAlibiPositionEmbedding<int32_t>(step);
  }
  return true;
}

bool QnnNspModel::setupInput(const InferenceStep& step,
                             uint32_t start,
                             std::vector<int32_t>& tokens,
                             std::vector<size_t>& tokenNumPerBatch,
                             AttentionMask& attention_mask) {
  GENIE_TRACE();
  const size_t variant   = static_cast<size_t>(step.variant);
  const size_t ctx_size  = static_cast<size_t>(step.ctx_size);
  const size_t n_process = static_cast<size_t>(step.n_process);
  if (!tokens.empty()) {
    // Setup input id tensor
    uint32_t* input_id_buffer = reinterpret_cast<uint32_t*>(getBuffer(t_input_ids));
    std::fill_n(input_id_buffer, variant * m_batch_size, static_cast<uint32_t>(m_pad_token));
    if (m_modelArchitectureType != ModelArchitectureType::ENCODER && variant < ctx_size) {
      // For normal cases (variant < ctx_size), tokens are processed normally
      uint32_t batch_start_idx = 0;
      size_t n_max_batch_tokens =
          *std::max_element(tokenNumPerBatch.begin(), tokenNumPerBatch.end());
      for (size_t batch_idx = 0; batch_idx < m_n_queryBatch; batch_idx++) {
        std::memcpy(input_id_buffer + batch_idx * variant,
                    &tokens[batch_start_idx + start],
                    n_process * sizeof(uint32_t));
        batch_start_idx += n_max_batch_tokens;
      }
    } else {
      __ERROR(
          "Multi-batch processing currently does not support ENCODER model architecture or variant "
          "== ctx_size.");
      return false;
    }
  } else {
    __ERROR(
        "Multi-batch processing currently requires non-empty token input."
        "Please provide valid tokens or use single-batch mode.");
    return false;
  }
  // Set up the input scatter index as new_idx (i.e. the index where new KV$ is stored)
  for (auto& [prefix, group_index_tensor] : m_group_cache_index) {
    CacheGroup& group        = m_kvmanager->getCacheGroups().at(prefix);
    InferenceStep group_step = group.translateInferenceStep(step);

    uint32_t* group_index_buffer = reinterpret_cast<uint32_t*>(getBuffer(group_index_tensor));
    std::iota(group_index_buffer,
              group_index_buffer + group_index_tensor->dims.getNumElements(),
              group_step.new_idx);
  }

  if (t_valid_mask != nullptr) {  // Set up a valid mask. Assumes u16 datatype (from validateModel)
    // Quantize mask value to u16
    const auto [scale, offset] = t_valid_mask->quantParam[0];
    const double alpha         = m_useAnchorAlpha ? m_anchorAlpha : 1.0;
    const uint16_t maskValue =
        QnnUtils::quantize<double, uint16_t>(alpha / n_process, offset, scale);

    bool hasSpeculativeTokens = false;
    if (!tokens.empty()) {
      for (int32_t token : tokens) {
        if (token >= static_cast<int32_t>(m_vocab_size)) {
          hasSpeculativeTokens = true;
          break;
        }
      }
    }

    const size_t n_masked = hasSpeculativeTokens ? 1 : n_process;
    // Setup the buffer
    uint16_t* mask_buffer = reinterpret_cast<uint16_t*>(getBuffer(t_valid_mask));
    std::fill_n(mask_buffer, n_masked, maskValue);
    std::memset(&mask_buffer[n_masked], 0, (variant - n_masked) * sizeof(uint16_t));

    // Apply anchor alpha weight on subsequent inferences.
    m_useAnchorAlpha = true;
  }

  // Setup the attention mask correctly
  if (d_attn_map.bw() == 1) {
    setupAttentionMask<uint8_t>(step, attention_mask, tokenNumPerBatch);
  } else if (d_attn_map.bw() == 2) {
    setupAttentionMask<uint16_t>(step, attention_mask, tokenNumPerBatch);
  } else if (d_attn_map.bw() == 4) {
    setupAttentionMask<uint32_t>(step, attention_mask, tokenNumPerBatch);
  }

  if (m_positional_encoding.type == PositionalEncoding::ROPE) {
    // Simple RoPE position ID setup

    // TODO: Compile position_ids to translate from [0,1,2,3,4,0,0,...,0] -> [(0,5), (0,1),...]
    // This is to batch the memory copy calls together, which is more optimal (theoretically)
    uint8_t* cos_buffer    = reinterpret_cast<uint8_t*>(getBuffer(t_position_ids_cos));
    uint8_t* sin_buffer    = reinterpret_cast<uint8_t*>(getBuffer(t_position_ids_sin));
    const size_t rope_size = m_pos_dim * d_pos.bw();
    for (size_t batch_idx = 0; batch_idx < m_n_queryBatch; batch_idx++) {
      const auto& position_ids =
          attention_mask.getPositionIds(start,
                                        n_process,
                                        variant,
                                        m_batch_prompt_token_lens.at(batch_idx),
                                        m_max_batch_prompt_token_len);
      for (uint32_t i = 0; i < variant; i++) {
        const size_t src_offset = static_cast<size_t>(position_ids[i]) * rope_size;
        const size_t dst_offset = i * rope_size;
        std::memcpy(&sin_buffer[dst_offset + batch_idx * rope_size * variant],
                    reinterpret_cast<uint8_t*>(rope_sin) + src_offset,
                    rope_size);
        std::memcpy(&cos_buffer[dst_offset + batch_idx * rope_size * variant],
                    reinterpret_cast<uint8_t*>(rope_cos) + src_offset,
                    rope_size);
      }
    }
  } else {
    __ERROR(
        "Unsupported positional encoding type for multi-batch processing."
        "Please use ROPE encoding or switch to single-batch mode.");
    return false;
  }
  return true;
}

inline void QnnNspModel::syncDrafTargetPrefill(bool isDraft, bool isReset) {
  auto* c = _counter;
  if (!c) return;

#if __cplusplus >= 202002L
  if (!isReset) {
    c->wait(isDraft ? 0 : 1, std::memory_order_seq_cst);
  } else {
    c->store(isDraft ? 0 : 1, std::memory_order_seq_cst);
    c->notify_all();
  }
#else
  if (!isReset) {
    int target_value = isDraft ? 0 : 1;
    while (c->load(std::memory_order_acquire) != target_value) {
      std::this_thread::yield();
    }
  } else {
    c->store(isDraft ? 0 : 1, std::memory_order_release);
  }
#endif
}

size_t QnnNspModel::runInference(const std::vector<int32_t>& tokens,
                                 std::vector<uint8_t>& embedding,
                                 const uint16_t* featureVector,
                                 const std::vector<int32_t>& selected,
                                 uint32_t start_idx,
                                 bool post_update,
                                 const std::vector<int32_t>& attention_map,
                                 std::vector<float>& output,
                                 bool output_all) {
  GENIE_TRACE();
  qualla::Timer start;
  __TRACE("runInference logits_all={} tokens={} featureVector {}",
          output_all,
          tokens,
          reinterpret_cast<uintptr_t>(featureVector));

  bool draft = false;
  if (featureVector != nullptr) {
    draft = true;
  }

  if ((tokens.size() == 0) && (embedding.size() == 0)) return 0;

  size_t embedBufSize   = m_embeddingBufferSize;
  size_t embeddingCount = embedding.size() / embedBufSize;

  // Disable token_history (required for AR-c models) if embedding input is processed
  if (embeddingCount > 0) {
    token_history_enabled = false;
  }

  // Construct an attention mask processor to simplify handling of complex masks
  const size_t n_inputs = tokens.size() + embeddingCount;
  AttentionMask attention_mask(attention_map,
                               static_cast<size_t>(m_kvmanager->n_past()),
                               static_cast<size_t>(m_kvmanager->n_valid_kv()),
                               n_inputs,
                               _offset_to_apply_kv_prefix,
                               _size_to_skip_kv_prefix);

  // Create a strategy to run the inference
  if (!m_kvmanager->prepareInferenceStrategy(static_cast<int32_t>(n_inputs), output_all)) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  // user choice overwrites the default behaviour in case of Embedding models
  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    output_all = !m_pooled_output;
  }

  size_t output_size = output_all ? n_inputs : 1;
  output.resize(output_size * ((m_modelArchitectureType == ModelArchitectureType::ENCODER)
                                   ? m_embd_size
                                   : m_vocab_size));
  __TRACE("runInference output ={}", output.size());

  // Loop over each planned iteration in the inference strategy
  InferenceStep step;
  uint32_t n_processed = 0;  // Number of total tokens processed so far

  while (m_kvmanager->nextInferenceStep(step)) {
    if (m_pause && n_processed != 0 && n_processed != 1) {
      m_pause = false;
      return n_processed;
    }

    __DEBUG("Inference step {}: {}", m_inference_count, step.str());
    syncDrafTargetPrefill(draft, false);
    if (!setupInput(step,
                    n_processed,
                    tokens,
                    embedding,
                    reinterpret_cast<const uint8_t*>(featureVector),
                    selected,
                    static_cast<size_t>(start_idx),
                    post_update,
                    attention_mask)) {
      throw std::runtime_error("Error setting up input tensors");
    }

    // Call prepareInputs for all registered adaptors using base class helper
    bool shouldSkipInference = false;
    if (!prepareAdaptorInputs(step.variant,
                              step.ctx_size,
                              static_cast<uint32_t>(step.n_past),
                              n_processed,
                              static_cast<uint32_t>(step.n_process),
                              m_kvmanager->isFinalInferenceStep(),
                              shouldSkipInference)) {
      __ERROR("nsp-model: prepareAdaptorInputs failed");
      return false;
    }

    // Skip inference if any adaptor requested it
    if (!shouldSkipInference) {
      for (auto& nsp_graph : m_nsp_graphs) {
        const int graph_idx = nsp_graph.idx();

        if (nsp_graph.m_graphType == GraphType::LMHEAD && (!output_all) &&
            !m_kvmanager->isFinalInferenceStep()) {  // Must meet all the criteria to skip LMHEAD
          continue;
        }

        if (!m_kvmanager->block(Scope::per_graph(graph_idx))) {
          State::error(m_kvmanager->error());
          return false;
        }

        if (!nsp_graph.execute(step.variant,
                               step.ctx_size,
                               m_inference_count,
                               graph_switching,
                               lazy_lora,
                               weight_shared_lora)) {
          throw std::runtime_error(fmt::format("Failed to execute graph {}", graph_idx));
        }

        if (!m_kvmanager->unblock(Scope::per_graph(graph_idx))) {
          throw std::runtime_error(m_kvmanager->error());
        }
      }
    }

    // Call handleOutputs for all registered adaptors using base class helper
    if (!handleAdaptorOutputs(step.variant,
                              step.ctx_size,
                              static_cast<uint32_t>(step.n_past),
                              n_processed,
                              static_cast<uint32_t>(step.n_process),
                              m_kvmanager->isFinalInferenceStep())) {
      __ERROR("nsp-model: handleAdaptorOutputs failed");
      return false;
    }

    m_kvmanager->completeInferenceStep();

    if (m_modelArchitectureType != ModelArchitectureType::ENCODER && output_all)
      getDequantLogits(std::span(&output[n_processed * m_vocab_size],
                                 static_cast<uint32_t>(step.n_process) * m_vocab_size),
                       step,
                       step.n_process);

    // Debug dump outputs
    if (_debug_outputs) {
      if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
        debugOutputs(step, m_layerNames[LayerType::POOL_OUTPUT]);
        debugOutputs(step, m_layerNames[LayerType::SEQ_OUTPUT]);
      } else {
        debugOutputs(step, m_layerNames[LayerType::OUTPUT]);
      }
    }

    n_processed += static_cast<uint32_t>(step.n_process);
    m_inference_count++;
    syncDrafTargetPrefill(draft, true);
  }

  if (post_update) {
    updateFeatureBuffer(embeddingCount);
  }

  // If only the last output is required, then process this request here
  if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
    if (!output_all) {
      getDequantLogits(std::span{output.data(), output.size()}, step, 1);
    }
  } else {
    getEmbeddings(std::span{output.data(), output.size()}, step);
  }

  // Maintain a history of all processed tokens
  if (token_history_enabled)
    token_history.insert(token_history.end(), tokens.begin(), tokens.end());

  __DEBUG("qnn-htp: run-inference complete : {} usec ", start.elapsed_usec());
  return output_size;
}

size_t QnnNspModel::setInputBufferData(LayerType inputLayer,
                                       void* data,
                                       size_t dataSize,
                                       nlohmann::json& dataConfig) {
  size_t inputCount = 0;
  if (inputLayer == LayerType::INPUT) {
    if (!m_tokenEmbedIO) {
      // Only creates one
      m_tokenEmbedIO = std::make_unique<TokenEmbedIO>(m_pad_token,
                                                      m_inputType,
                                                      t_input_ids,
                                                      m_externalBufferType,
                                                      m_dataFillPolicy,
                                                      m_ioTensor,
                                                      _env);
    }
    inputCount = m_tokenEmbedIO->addData(data, dataSize, dataConfig);
  } else {
    throw std::runtime_error("Unsupported Input setting requested.");
  }
  return inputCount;
}

void QnnNspModel::allocateOutputBuffer(uint32_t nLogits) {
  size_t requiredSize = nLogits * m_vocab_size;
  if (m_outputLogit.size() != requiredSize) {
    m_outputLogit.resize(requiredSize, 0);
  }
}

size_t QnnNspModel::runPrefillInputInference(const nlohmann::json executionConfig) {
  GENIE_TRACE();
  qualla::Timer start;
  const size_t n_inputs = m_tokenEmbedIO->m_dataLength;
  if (n_inputs == 0) return 0;
  bool output_all = false;
  if (!executionConfig.empty() && executionConfig.contains("output-all")) {
    output_all = executionConfig["output-all"].get<bool>();
  }
  if (output_all)  // this could only be supported for float logits, as it will defeat the purpose
                   // of the raw logits
    allocateOutputBuffer(n_inputs);
  else if (m_outputLogit.size() == 0)
    allocateOutputBuffer(1);
  const std::vector<int32_t> attention_map;
  AttentionMask attention_mask(attention_map,
                               static_cast<size_t>(m_kvmanager->n_past()),
                               static_cast<size_t>(m_kvmanager->n_valid_kv()),
                               n_inputs,
                               _offset_to_apply_kv_prefix,
                               _size_to_skip_kv_prefix);

  // Create a strategy to run the inference
  if (!m_kvmanager->prepareInferenceStrategy(static_cast<int32_t>(n_inputs), output_all)) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  __TRACE("runInference output ={}", m_outputLogit.size());

  size_t output_size = output_all ? n_inputs : 1;
  // Loop over each planned iteration in the inference strategy
  InferenceStep step;
  uint32_t n_processed = 0;  // Number of total tokens processed so far

  // dummy variables
  const std::vector<int32_t> tokens;
  std::vector<uint8_t> embeddings;
  uint8_t* featureVector = nullptr;
  std::vector<int32_t> selected;
  uint32_t start_idx = 0;
  bool post_update   = false;

  while (m_kvmanager->nextInferenceStep(step)) {
    if (m_pause && n_processed != 0 && n_processed != 1) {
      m_pause = false;
      return n_processed;
    }

    __DEBUG("Inference step: {}", step.str());
    // Should populate Tensor with the required tokens and embeddings for the current run.
    m_tokenEmbedIO->populateTensor(step);
    // setups attention mask and position tensors
    if (!setupInput(step,
                    n_processed,
                    tokens,
                    embeddings,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    attention_mask,
                    true))
      return false;

    for (auto& nsp_graph : m_nsp_graphs) {
      const int graph_idx = nsp_graph.idx();

      if (nsp_graph.m_graphType == GraphType::LMHEAD && (!output_all) &&
          !m_kvmanager->isFinalInferenceStep()) {  // Must meet all the criteria to skip LMHEAD
        continue;
      }

      if (!m_kvmanager->block(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }

      if (!nsp_graph.execute(step.variant,
                             step.ctx_size,
                             m_inference_count,
                             graph_switching,
                             lazy_lora,
                             weight_shared_lora)) {
        fatal(fmt::format("Failed to execute graph {}", graph_idx));
        return false;
      }

      if (!m_kvmanager->unblock(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }
    }
    m_kvmanager->completeInferenceStep();

    if (m_modelArchitectureType != ModelArchitectureType::ENCODER && output_all)
      getDequantLogits(std::span(&m_outputLogit[n_processed * m_vocab_size],
                                 static_cast<uint32_t>(step.n_process) * m_vocab_size),
                       step,
                       step.n_process);

    // Debuging output dump
    if (_debug_outputs) {
      if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
        debugOutputs(step, m_layerNames[LayerType::POOL_OUTPUT]);
        debugOutputs(step, m_layerNames[LayerType::SEQ_OUTPUT]);
      } else {
        debugOutputs(step, m_layerNames[LayerType::OUTPUT]);
      }
    }

    n_processed += static_cast<uint32_t>(step.n_process);
    m_inference_count++;
  }
  // If only the last output is required, then process this request here
  if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
    if (!output_all) {
      getDequantLogits(std::span{m_outputLogit.data(), m_outputLogit.size()}, step, 1);
    }
  } else {
    getEmbeddings(std::span{m_outputLogit.data(), m_outputLogit.size()}, step);
  }

  __DEBUG("qnn-htp: run-inference complete : {} usec ", start.elapsed_usec());
  return output_size;
}

bool QnnNspModel::getOutputBufferData(LayerType outputLayer,
                                      const nlohmann::json& /*config*/,
                                      Engine::OutputCallback callback) {
  if (outputLayer != LayerType::OUTPUT) {
    __ERROR("Unsupported layer for getting the output");
    return false;
  }
  if (!callback) {
    __ERROR("getOutput requires a valid callback");
    return false;
  }

  // Build minimal metadata: dtype=float and 2D dims [rows, cols]
  nlohmann::json meta;
  meta["data-type"]  = "float32";
  meta["dimensions"] = nlohmann::json::array();

  meta["dimensions"].push_back(static_cast<uint32_t>(m_outputLogit.size() / m_vocab_size));
  meta["dimensions"].push_back(static_cast<uint32_t>(m_vocab_size));

  // Pass the data buffer and size to the callback (do not embed data in JSON)
  void* data_ptr = static_cast<void*>(m_outputLogit.data());
  const size_t dataSize =
      static_cast<size_t>(m_outputLogit.size()) * static_cast<size_t>(sizeof(float));

  if (!callback(data_ptr, dataSize, meta)) {
    __ERROR("getOutput: callback returned false");
    return false;
  }
  return true;
}
size_t QnnNspModel::runInference(const std::vector<int32_t>& tokens,
                                 std::vector<uint8_t>& embedding,
                                 const uint16_t* featureVector,
                                 const std::vector<int32_t>& selected,
                                 uint32_t start_idx,
                                 bool post_update,
                                 const std::vector<int32_t>& attention_map,
                                 Tensor& output,
                                 bool output_all) {
  GENIE_TRACE();
  qualla::Timer start;

  if ((tokens.size() == 0) && (embedding.size() == 0)) return 0;

  size_t embedBufSize   = m_embeddingBufferSize;
  size_t embeddingCount = embedding.size() / embedBufSize;
  // Disable token_history (required for AR-c models) if embedding input is processed
  if (embeddingCount > 0) {
    token_history_enabled = false;
  }

  bool draft = false;
  if (featureVector != nullptr) {
    draft = true;
  }

  // Construct an attention mask processor to simplify handling of complex masks
  const size_t n_inputs = tokens.size() + embeddingCount;
  AttentionMask attention_mask(attention_map,
                               static_cast<size_t>(m_kvmanager->n_past()),
                               static_cast<size_t>(m_kvmanager->n_valid_kv()),
                               n_inputs,
                               _offset_to_apply_kv_prefix,
                               _size_to_skip_kv_prefix);

  // Create a strategy to run the inference
  if (!m_kvmanager->prepareInferenceStrategy(static_cast<int32_t>(n_inputs), output_all)) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  size_t output_size = output_all ? n_inputs : 1;  // actual number of logits
  output.setSize(0);

  // Loop over each planned iteration in the inference strategy
  InferenceStep step;
  uint32_t n_processed = 0;  // Number of total tokens processed so far

  bool requireLogitsCopy = false;
  if (m_kvmanager->getStrategySize() > 1 && output_all && draft == false) {
    // If we need to return logits from multiple inferences to the caller,
    // then enable logit copying to accumulate logits in the output Tensor.
    requireLogitsCopy = true;
  }

  while (m_kvmanager->nextInferenceStep(step)) {
    if (m_pause && n_processed != 0 && n_processed != 1) {
      m_pause = false;
      return n_processed;
    }

    __DEBUG("Inference step {} : {}", m_inference_count, step.str());
    syncDrafTargetPrefill(draft, false);
    if (!setupInput(step,
                    n_processed,
                    tokens,
                    embedding,
                    reinterpret_cast<const uint8_t*>(featureVector),
                    selected,
                    static_cast<size_t>(start_idx),
                    post_update,
                    attention_mask)) {
      throw std::runtime_error("Error setting up input tensors");
    }

    // Call prepareInputs for all registered adaptors
    bool shouldSkipInference = false;
    if (!prepareAdaptorInputs(step.variant,
                              step.ctx_size,
                              static_cast<uint32_t>(step.n_past),
                              n_processed,
                              static_cast<uint32_t>(step.n_process),
                              m_kvmanager->isFinalInferenceStep(),
                              shouldSkipInference)) {
      __ERROR("nsp-model: prepareAdaptorInputs failed");
      return false;
    }

    // Skip inference if any adaptor requested it
    if (!shouldSkipInference) {
      for (auto& nsp_graph : m_nsp_graphs) {
        const int graph_idx = nsp_graph.idx();

        if (nsp_graph.m_graphType == GraphType::LMHEAD && (!output_all) &&
            !m_kvmanager->isFinalInferenceStep()) {  // Must meet all the criteria to skip LMHEAD
          continue;
        }

        if (!m_kvmanager->block(Scope::per_graph(graph_idx))) {
          State::error(m_kvmanager->error());
          return false;
        }

        if (!nsp_graph.execute(step.variant,
                               step.ctx_size,
                               m_inference_count,
                               graph_switching,
                               lazy_lora,
                               weight_shared_lora)) {
          fatal(fmt::format("Failed to execute graph {}", graph_idx));
          return false;
        }

        if (!m_kvmanager->unblock(Scope::per_graph(graph_idx))) {
          State::error(m_kvmanager->error());
          return false;
        }
      }
    }

    // Call handleOutputs for all registered adaptors
    if (!handleAdaptorOutputs(step.variant,
                              step.ctx_size,
                              static_cast<uint32_t>(step.n_past),
                              n_processed,
                              static_cast<uint32_t>(step.n_process),
                              m_kvmanager->isFinalInferenceStep())) {
      __ERROR("nsp-model: handleAdaptorOutputs failed");
      return false;
    }

    if (output_all) getLogits(output, step, step.n_process, requireLogitsCopy);

    // Debug dump outputs
    if (_debug_outputs) {
      debugOutputs(step, m_layerNames[LayerType::OUTPUT]);
      debugOutputs(step, m_layerNames[LayerType::EAGLET_DM_HIDDEN_STATES_IN]);
    }

    n_processed += static_cast<uint32_t>(step.n_process);
    m_inference_count++;
    syncDrafTargetPrefill(draft, true);
  }
  if (post_update) {
    updateFeatureBuffer(embeddingCount);
  }
  // If only the last output is required, then process this request here
  if (!output_all) {
    getLogits(output, step, 1);
  }

  // Maintain a history of all processed tokens
  if (token_history_enabled)
    token_history.insert(token_history.end(), tokens.begin(), tokens.end());

  __DEBUG("qnn-htp: run-inference complete : {} usec ", start.elapsed_usec());
  return output_size;
}

size_t QnnNspModel::runInference(std::vector<int32_t>& tokens,
                                 std::vector<uint8_t>& embedding,
                                 const std::vector<int32_t>& attention_map,
                                 std::vector<size_t>& tokenNumPerBatch,
                                 std::vector<Tensor>& output,
                                 bool output_all) {
  GENIE_TRACE();
  qualla::Timer start;

  if ((tokens.size() == 0) && (embedding.size() == 0)) return 0;

  m_n_queryBatch = tokenNumPerBatch.size();
  if (m_n_queryBatch > m_batch_size) {
    fatal(
        fmt::format("Query batch size ({}) exceeds model's configured batch size ({}). "
                    "Please reduce the number of input queries.",
                    m_n_queryBatch,
                    m_batch_size));
    return 0;
  }
  m_kvmanager->setBatchSize(m_batch_size);
  size_t n_max_batch_tokens = *std::max_element(tokenNumPerBatch.begin(), tokenNumPerBatch.end());

  size_t embedBufSize   = m_embeddingBufferSize;
  size_t embeddingCount = embedding.size() / embedBufSize;

  // Construct an attention mask processor to simplify handling of complex masks
  const size_t n_inputs = tokens.size() + embeddingCount;
  AttentionMask attention_mask(attention_map,
                               static_cast<size_t>(m_kvmanager->n_past()),
                               static_cast<size_t>(m_kvmanager->n_valid_kv()),
                               n_inputs,
                               _offset_to_apply_kv_prefix,
                               _size_to_skip_kv_prefix);

  if (attention_map.size() > n_inputs && isLongContextEnabled()) {
    State::fatal("LongContext has not been enabled for this dialog");
    return false;
  }

  // Create a strategy to run the inference
  if (!m_kvmanager->prepareInferenceStrategy(n_max_batch_tokens, output_all)) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  size_t output_size = output_all ? n_inputs : 1 * m_n_queryBatch;  // actual number of logits

  output.resize(m_n_queryBatch);

  for (size_t batch_idx = 0; batch_idx < m_n_queryBatch; batch_idx++) {
    (output.at(batch_idx)).setSize(0);
  }

  // Loop over each planned iteration in the inference strategy
  InferenceStep step;
  uint32_t n_processed = 0;  // Number of total tokens processed so far

  bool requireLogitsCopy = false;
  if (m_kvmanager->getStrategySize() > 1 && output_all) {
    // If we need to return logits from multiple inferences to the caller,
    // then enable logit copying to accumulate logits in the output Tensor.
    requireLogitsCopy = true;
  }

  size_t n_padding = 0;
  if (m_n_queryBatch > 1) {
    std::vector<int32_t> aligned_tokens;
    aligned_tokens.reserve(m_n_queryBatch * n_max_batch_tokens);
    size_t batch_start_idx = 0;

    for (size_t batch_idx = 0; batch_idx < m_n_queryBatch; batch_idx++) {
      size_t current_len = tokenNumPerBatch.at(batch_idx);
      n_padding = (current_len < n_max_batch_tokens) ? (n_max_batch_tokens - current_len) : 0;
      if (n_padding > 0) {
        aligned_tokens.insert(aligned_tokens.end(), n_padding, 0);
      }
      aligned_tokens.insert(
          aligned_tokens.end(),
          tokens.begin() + static_cast<std::ptrdiff_t>(batch_start_idx),
          tokens.begin() + static_cast<std::ptrdiff_t>(batch_start_idx + current_len));
      batch_start_idx += current_len;
    }
    tokens = std::move(aligned_tokens);
  }

  while (m_kvmanager->nextInferenceStep(step)) {
    if (m_pause && n_processed != 0 && n_processed != 1) {
      m_pause = false;
      return n_processed;
    }

    __DEBUG("Inference step: {}", step.str());
    if (!setupInput(step, n_processed, tokens, tokenNumPerBatch, attention_mask)) return false;

    for (auto& nsp_graph : m_nsp_graphs) {
      const int graph_idx = nsp_graph.idx();

      if (nsp_graph.m_graphType == GraphType::LMHEAD && (!output_all) &&
          !m_kvmanager->isFinalInferenceStep()) {  // Must meet all the criteria to skip LMHEAD
        continue;
      }

      if (!m_kvmanager->block(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }

      if (!nsp_graph.execute(step.variant,
                             step.ctx_size,
                             m_inference_count,
                             graph_switching,
                             lazy_lora,
                             weight_shared_lora)) {
        fatal(fmt::format("Failed to execute graph {}", graph_idx));
        return false;
      }

      if (!m_kvmanager->unblock(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }
    }

    if (output_all) getLogits(output, step, step.n_process, requireLogitsCopy);

    // Debug dump outputs
    if (_debug_outputs) {
      debugOutputs(step, m_layerNames[LayerType::OUTPUT]);
    }

    n_processed += static_cast<uint32_t>(step.n_process);
    m_inference_count++;
  }

  // If only the last output is required, then process this request here
  if (!output_all) {
    getLogits(output, step, 1);
  }

  __DEBUG("qnn-htp: run-inference complete : {} usec ", start.elapsed_usec());
  return output_size;
}

void QnnNspModel::updateFeatureBuffer(uint32_t embeddingCount) {
  if (eagle_extra_feature == nullptr) {
    eagle_extra_feature = reinterpret_cast<uint8_t*>(calloc(dm_draftFeatureSize, sizeof(uint8_t)));
  }

  const size_t feature_offset = embeddingCount - 1;
  const uint8_t* feature_data = reinterpret_cast<uint8_t*>(getBuffer(t_dm_feature_out)) +
                                feature_offset * dm_draftFeatureSize;
  std::memcpy(eagle_extra_feature, feature_data, dm_draftFeatureSize);
}

// Dumps out the specified tensor to _debug_path numbered according to m_inference_count
bool QnnNspModel::debugOutputs(const InferenceStep& step, const std::string& tensor_name) {
  GENIE_TRACE();
  if (!m_nsp_graphs.back().variants.contains({step.variant, step.ctx_size})) {
    __DEBUG("No outputs found for AR-{} CL-{}", step.variant, step.ctx_size);
    return true;
  }
  GraphVariant* graph_variant = m_nsp_graphs.back()(step.variant, step.ctx_size);
  QnnUtils::Tensor* tensor    = graph_variant->getOutput(tensor_name);
  if (tensor == nullptr) {
    __DEBUG("qnn-htp: Couldn't find tensor {} in graph {}", tensor_name, graph_variant->graph_name);
    return false;
  }

  // For ENCODER models, dump the complete buffer. For LLM models, dump the generated logits
  uint32_t output_bitwidth = tensor->dtype.bw();  // Detect 8-bit vs 16-bit logits
  size_t output_size       = (m_modelArchitectureType == ModelArchitectureType::ENCODER)
                                 ? static_cast<size_t>(step.ctx_size) * output_bitwidth * m_embd_size
                                 : static_cast<size_t>(step.n_process) * output_bitwidth * m_vocab_size;
  std::string fname = fmt::format("{}/{}/{:03d}", _debug_path, tensor_name, m_inference_count);
  if (!QnnUtils::writeRawData(getBuffer(tensor), output_size, fname)) {
    __DEBUG("qnn-htp: Failed to save {}. Error when writing to {}", tensor_name, fname);
    return false;
  }

  return true;
}

bool QnnNspModel::quantizeInput(float* in, size_t tensorOffset, size_t length) {
  if (t_input_ids == nullptr) {
    __ERROR("Input Tensor {} not found during execute", m_layerNames[LayerType::INPUT]);
    return false;
  }

  const auto scale  = t_input_ids->quantParam[0].scale;
  const auto offset = t_input_ids->quantParam[0].offset;
  switch (t_input_ids->dtype) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      QnnUtils::quantizeTensorPtr(in,
                                  reinterpret_cast<uint8_t*>(getBuffer(t_input_ids)) + tensorOffset,
                                  offset,
                                  scale,
                                  length);
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      QnnUtils::quantizeTensorPtr(
          in,
          reinterpret_cast<uint16_t*>(getBuffer(t_input_ids)) + tensorOffset,
          offset,
          scale,
          length);
      break;
    case QNN_DATATYPE_FLOAT_16:
      float32ToFloat16(
          reinterpret_cast<uint8_t*>(getBuffer(t_input_ids)) + tensorOffset, in, length);
      break;
    case QNN_DATATYPE_FLOAT_32:
      std::copy(in, in + length, reinterpret_cast<float*>(getBuffer(t_input_ids)) + tensorOffset);
      ;
      break;
    default:
      __ERROR("Unsupported alpha tensor dtype {}", t_input_ids->dtype.str());
      return false;
  }

  return true;
}

size_t QnnNspModel::getEmbeddingBufferSize() { return m_embeddingBufferSize; }

void QnnNspModel::getTensorParam(
    LayerType layerType, std::string& dataType, double& scale, int32_t& offset, size_t& bitwidth) {
  if (layerType == LayerType::INPUT) {
    dataType = t_input_ids->dtype.str();
    scale    = t_input_ids->quantParam[0].scale;
    offset   = t_input_ids->quantParam[0].offset;
    bitwidth = static_cast<size_t>(t_input_ids->dtype.bw());
  }
}

bool QnnNspModel::cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding) {
  m_eosEmbedding = eosEmbedding;
  return true;
}

bool QnnNspModel::setKVCacheNPast(size_t n_past, const std::vector<bool>& selected) {
  GENIE_TRACE();
  if (!m_kvmanager->dispatchUpdate(n_past, selected)) {
    __ERROR("qnn-htp: KV$ update failed. {}", m_kvmanager->error());
    State::error(m_kvmanager->error());
    return false;
  }

  if (n_past == 0) {
    resetAnchorAlpha();
    // Call reset on all registered adaptors when KV cache is reset
    callAdaptorsReset();
  }

  // Manage the token history based on the KV$ accepted by the user
  // If no selection mask is passed, we can simply resize the history to n_past
  // If a selection mask is passed, we must selectively filter out rejected KV$
  if (token_history_enabled) {  // Token history must be disabled on embedding input or
                                // longcontext
    if (selected.empty())
      token_history.resize(n_past);
    else {
      auto it = token_history.begin() + static_cast<long>(token_history.size() - selected.size());
      for (const bool& isSelected : selected) {
        it = (isSelected) ? it + 1 : token_history.erase(it);  // Erase if not selected, else no-op
      }
    }
  }
  return true;
}

size_t QnnNspModel::getDequantLogits(std::span<float> buffer, InferenceStep& step, int32_t count) {
  GENIE_TRACE();
  qualla::Timer start;

  QnnUtils::Tensor* const spec =
      m_nsp_graphs.back()(step.variant, step.ctx_size)->getOutput(m_layerNames[LayerType::OUTPUT]);
  if (spec == nullptr) {
    State::error("Failed to get output layer tensor spec");
    return 0;
  }

  auto [scale, offset] = spec->quantParam[0];  // Quantization parameters
  QnnUtils::DataType dtype(spec->tensor);      // Datatype of the generated output
  uint32_t bitwidth = spec->dtype.bw();        // Number of bytes per output element
  auto logit_buffer =
      reinterpret_cast<uint8_t*>(getBuffer(spec));  // Pointer to the actual output data

  if (spec->dims.getNumElements() == m_vocab_size && count > 1) {
    State::error("Requested all logits, but graph only produces one logit");
    return 0;
  }

  // Offset to the appropriate location in the output buffer. Note this assumes right-padded input
  logit_buffer += static_cast<uint32_t>(step.n_process - count) * bitwidth * m_vocab_size;

  const size_t size = m_vocab_size * static_cast<size_t>(count);
  __TRACE("qnn-htp: getDequantLogits Returning {}*{} from [{}]", count, m_vocab_size, step.str());
  switch (dtype) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      deQuantizeOutputs(reinterpret_cast<uint8_t*>(logit_buffer), buffer, scale, offset, size);
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      deQuantizeOutputs(reinterpret_cast<uint16_t*>(logit_buffer), buffer, scale, offset, size);
      break;
    case QNN_DATATYPE_FLOAT_16:
      castOutputs(reinterpret_cast<uint16_t*>(logit_buffer), buffer, size, bitwidth);
      break;
    case QNN_DATATYPE_FLOAT_32:
      castOutputs(reinterpret_cast<float*>(logit_buffer), buffer, size, bitwidth);
      break;
    default:
      State::error(fmt::format("Unsupported logits dtype {}", dtype.str()));
      return 0;
  }

  __DEBUG("qnn-htp: getDequantLogits complete. Returning {} outputs in {} usec",
          count,
          start.elapsed_usec());
  return size;
}

size_t QnnNspModel::getLogits(Tensor& logits,
                              InferenceStep& step,
                              int32_t count,
                              bool requireLogitsCopy) {
  qualla::Timer start;

  QnnUtils::Tensor* const spec =
      m_nsp_graphs.back()(step.variant, step.ctx_size)->getOutput(m_layerNames[LayerType::OUTPUT]);
  if (spec == nullptr) {
    State::error("Failed to get output layer tensor spec");
    return 0;
  }

  auto [scale, offset] = spec->quantParam[0];  // Quantization parameters
  QnnUtils::DataType dtype(spec->tensor);      // Datatype of the generated output
  uint32_t bitwidth = spec->dtype.bw();        // Number of bytes per output element
  auto logit_buffer =
      reinterpret_cast<uint8_t*>(getBuffer(spec));  // Pointer to the actual output data

  if (spec->dims.getNumElements() == m_vocab_size && count > 1) {
    State::error("Requested all logits, but graph only produces one logit");
    return 0;
  }

  // Offset to the appropriate location in the output buffer. Note this assumes right-padded input
  logit_buffer += static_cast<uint32_t>(step.n_process - count) * m_vocab_size * bitwidth;

  const size_t size = m_vocab_size * static_cast<size_t>(count);
  __TRACE("qnn-htp: getLogits Returning {}*{} from [{}]", count, m_vocab_size, step.str());

  switch (dtype) {
    case QNN_DATATYPE_UFIXED_POINT_8: {
      if (requireLogitsCopy) {
        logits.logits.reserve(logits.getSize() + size);
        uint8_t* logit_buffer_u8 = reinterpret_cast<uint8_t*>(logit_buffer);
        for (uint32_t i = 0; i < size; i++) {
          logits.logits[logits.getSize() + i] =
              static_cast<float>(scale) *
              (static_cast<float>(logit_buffer_u8[i]) + static_cast<float>(offset));
        }
        logits.setQuantizationParams(1, 0);
        logits.setData(static_cast<void*>(logits.logits.data()));
        logits.setSize(logits.getSize() + size);
        logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      } else {
        logits.setQuantizationParams(scale, offset);
        logits.setData(static_cast<void*>(logit_buffer));
        logits.setSize(size);
        logits.setDataType(TENSOR_DATATYPE_UFIXED_POINT_8);
      }
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      if (requireLogitsCopy) {
        logits.logits.reserve(logits.getSize() + size);
        uint16_t* logit_buffer_u16 = reinterpret_cast<uint16_t*>(logit_buffer);
        for (uint32_t i = 0; i < size; i++) {
          logits.logits[logits.getSize() + i] =
              static_cast<float>(scale) *
              (static_cast<float>(logit_buffer_u16[i]) + static_cast<float>(offset));
        }
        logits.setQuantizationParams(1, 0);
        logits.setData(reinterpret_cast<void*>(logits.logits.data()));
        logits.setSize(logits.getSize() + size);
        logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      } else {
        logits.setQuantizationParams(scale, offset);
        logits.setData(reinterpret_cast<void*>(logit_buffer));
        logits.setSize(size);
        logits.setDataType(TENSOR_DATATYPE_UFIXED_POINT_16);
      }
      break;
    }
    case QNN_DATATYPE_FLOAT_16: {
      // Downstream tasks (like sampling) can't handle float16 yet. Always convert to float32.
      logits.logits.reserve(logits.getSize() + size);
      uint16_t* logit_buffer_fp16 = reinterpret_cast<uint16_t*>(logit_buffer);
      for (uint32_t i = 0; i < size; i++) {
        logits.logits[logits.getSize() + i] = fp16_ieee_to_fp32_value(logit_buffer_fp16[i]);
      }
      logits.setQuantizationParams(1, 0);
      logits.setData(reinterpret_cast<void*>(logits.logits.data()));
      logits.setSize(logits.getSize() + size);
      logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      if (requireLogitsCopy) {
        logits.logits.reserve(logits.getSize() + size);
        float* logit_buffer_fp32 = reinterpret_cast<float*>(logit_buffer);
        for (uint32_t i = 0; i < size; i++) {
          logits.logits[logits.getSize() + i] = logit_buffer_fp32[i];
        }
        logits.setData(reinterpret_cast<void*>(logits.logits.data()));
        logits.setSize(logits.getSize() + size);
      } else {
        logits.setData(reinterpret_cast<void*>(logit_buffer));
        logits.setSize(size);
      }
      logits.setQuantizationParams(1, 0);
      logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      break;
    }
    default: {
      State::error(fmt::format("Unsupported logits dtype {}", dtype.str()));
      return 0;
    }
  }

  __DEBUG(
      "qnn-htp: getLogits complete. Returning {} outputs in {} usec", count, start.elapsed_usec());
  return size;
}

size_t QnnNspModel::getLogits(std::vector<Tensor>& logits_vec,
                              InferenceStep& step,
                              int32_t count,
                              bool requireLogitsCopy) {
  qualla::Timer start;

  QnnUtils::Tensor* const spec =
      m_nsp_graphs.back()(step.variant, step.ctx_size)->getOutput(m_layerNames[LayerType::OUTPUT]);
  if (spec == nullptr) {
    State::error("Failed to get output layer tensor spec");
    return 0;
  }

  auto [scale, offset] = spec->quantParam[0];  // Quantization parameters
  QnnUtils::DataType dtype(spec->tensor);      // Datatype of the generated output
  uint32_t bitwidth = spec->dtype.bw();        // Number of bytes per output element
  auto logit_buffer =
      reinterpret_cast<uint8_t*>(getBuffer(spec));  // Pointer to the actual output data

  if (spec->dims.getNumElements() == m_vocab_size && count > 1) {
    State::error("Requested all logits, but graph only produces one logit");
    return 0;
  }

  const size_t logits_elements_per_batch = getBufferSize(spec) / m_batch_size;
  const size_t size                      = m_vocab_size * static_cast<size_t>(count);
  __TRACE("qnn-htp: getLogits Returning {}*{} from [{}]",
          static_cast<size_t>(count) * m_n_queryBatch,
          m_vocab_size,
          step.str());

  for (size_t batch_idx = 0; batch_idx < m_n_queryBatch; batch_idx++) {
    Tensor logits;
    logit_buffer = reinterpret_cast<uint8_t*>(getBuffer(spec));
    // Offset to the appropriate location in the output buffer. Note this assumes right-padded input
    logit_buffer += static_cast<uint32_t>(step.n_process - count) * m_vocab_size * bitwidth +
                    batch_idx * logits_elements_per_batch;
    switch (dtype) {
      case QNN_DATATYPE_UFIXED_POINT_8: {
        if (requireLogitsCopy) {
          logits.logits.reserve(logits.getSize() + size);
          uint8_t* logit_buffer_u8 = reinterpret_cast<uint8_t*>(logit_buffer);
          for (uint32_t i = 0; i < size; i++) {
            logits.logits[logits.getSize() + i] =
                static_cast<float>(scale) *
                (static_cast<float>(logit_buffer_u8[i]) + static_cast<float>(offset));
          }
          logits.setQuantizationParams(1, 0);
          logits.setData(static_cast<void*>(logits.logits.data()));
          logits.setSize(logits.getSize() + size);
          logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
        } else {
          logits.setQuantizationParams(scale, offset);
          logits.setData(static_cast<void*>(logit_buffer));
          logits.setSize(size);
          logits.setDataType(TENSOR_DATATYPE_UFIXED_POINT_8);
        }
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_16: {
        if (requireLogitsCopy) {
          logits.logits.reserve(logits.getSize() + size);
          uint16_t* logit_buffer_u16 = reinterpret_cast<uint16_t*>(logit_buffer);
          for (uint32_t i = 0; i < size; i++) {
            logits.logits[logits.getSize() + i] =
                static_cast<float>(scale) *
                (static_cast<float>(logit_buffer_u16[i]) + static_cast<float>(offset));
          }
          logits.setQuantizationParams(1, 0);
          logits.setData(reinterpret_cast<void*>(logits.logits.data()));
          logits.setSize(logits.getSize() + size);
          logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
        } else {
          logits.setQuantizationParams(scale, offset);
          logits.setData(reinterpret_cast<void*>(logit_buffer));
          logits.setSize(size);
          logits.setDataType(TENSOR_DATATYPE_UFIXED_POINT_16);
        }
        break;
      }
      case QNN_DATATYPE_FLOAT_16: {
        // Downstream tasks (like sampling) can't handle float16 yet. Always convert to float32.
        logits.logits.reserve(logits.getSize() + size);
        uint16_t* logit_buffer_fp16 = reinterpret_cast<uint16_t*>(logit_buffer);
        for (uint32_t i = 0; i < size; i++) {
          logits.logits[logits.getSize() + i] = fp16_ieee_to_fp32_value(logit_buffer_fp16[i]);
        }
        logits.setQuantizationParams(1, 0);
        logits.setData(reinterpret_cast<void*>(logits.logits.data()));
        logits.setSize(logits.getSize() + size);
        logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
        break;
      }
      case QNN_DATATYPE_FLOAT_32: {
        if (requireLogitsCopy) {
          logits.logits.reserve(logits.getSize() + size);
          float* logit_buffer_fp32 = reinterpret_cast<float*>(logit_buffer);
          for (uint32_t i = 0; i < size; i++) {
            logits.logits[logits.getSize() + i] = logit_buffer_fp32[i];
          }
          logits.setData(reinterpret_cast<void*>(logits.logits.data()));
          logits.setSize(logits.getSize() + size);
        } else {
          logits.setData(reinterpret_cast<void*>(logit_buffer));
          logits.setSize(size);
        }
        logits.setQuantizationParams(1, 0);
        logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
        break;
      }
      default: {
        State::error(fmt::format("Unsupported logits dtype {}", dtype.str()));
        return 0;
      }
    }
    logits_vec.at(batch_idx) = logits;
  }
  __DEBUG("qnn-htp: getLogits complete. Returning {} outputs in {} usec",
          static_cast<size_t>(count) * m_n_queryBatch,
          start.elapsed_usec());
  return size * m_n_queryBatch;
}

// Helper function: Calculate the dimension of frequency correction
double yarn_find_correction_dim(double num_rotations,
                                size_t dim,
                                double base,
                                int max_position_embeddings) {
  return (dim * std::log(max_position_embeddings / (num_rotations * 2 * M_PI))) /
         (2 * std::log(base));
}

// Helper function: Calculate the range of frequency correction
std::pair<int, int> yarn_find_correction_range(
    double low_rot, double high_rot, size_t dim, double base, int max_position_embeddings) {
  double low_dim  = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings);
  double high_dim = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings);

  // Validate bounds before casting to prevent overflow
  const double int_min = static_cast<double>(std::numeric_limits<int>::min());
  const double int_max = static_cast<double>(std::numeric_limits<int>::max());

  if (low_dim < int_min || low_dim > int_max || high_dim < int_min || high_dim > int_max) {
    throw std::runtime_error(
        "YaRN correction range calculation resulted in values outside valid integer range");
  }

  int low  = static_cast<int>(std::floor(low_dim));
  int high = static_cast<int>(std::ceil(high_dim));

  // Ensure dim > 0 before using dim - 1
  int max_dim = (dim > 0) ? static_cast<int>(dim - 1) : 0;
  return {std::max(low, 0), std::min(high, max_dim)};
}

// Helper function: Generate linear interpolation mask
std::vector<double> yarn_linear_ramp_mask(int min, int max, size_t dim) {
  std::vector<double> ramp(dim);
  if (min >= max) {
    // When min >= max, return all zeros (no interpolation needed)
    std::fill(ramp.begin(), ramp.end(), 0.0);
    return ramp;
  }
  const double range_inv = 1.0 / static_cast<double>(max - min);
  for (size_t i = 0; i < dim; ++i) {
    const double linear_func = (static_cast<double>(i) - static_cast<double>(min)) * range_inv;
    ramp[i]                  = std::clamp(linear_func, 0.0, 1.0);
  }
  return ramp;
}

// Helper function: Calculate scaling factor
double yarn_get_mscale(double scale) {
  if (scale <= 1.0) return 1.0;
  return 0.1 * std::log(scale) + 1.0;
}

bool QnnNspModel::calculate_rope_embeddings(void) {
  GENIE_TRACE();
  if (m_positional_encoding.type != PositionalEncoding::ROPE) {
    return true;
  }
  if (m_lazyInitialization || m_ropeInitialized) return true;
  const size_t nmemb    = m_ctx_size * m_pos_dim;
  const uint32_t pos_bw = d_pos.bw();

  double theta                          = m_positional_encoding.rope_params.theta;
  const RopeScalingParams& rope_scaling = m_positional_encoding.rope_params.rope_scaling;

  if (rope_scaling.rope_type == RopeScalingParams::ROPE_HUNYUAN) {
    const double& base_factor   = rope_scaling.hunyuan_params.base_factor;
    const double& dim_factor    = rope_scaling.hunyuan_params.dim_factor;
    const double& offset_factor = rope_scaling.hunyuan_params.offset_factor;
    theta = theta * (std::pow(base_factor, dim_factor / (dim_factor - offset_factor)));
  }

  if (!m_ropeBufferReady) {
    rope_sin          = malloc(nmemb * pos_bw);
    rope_cos          = malloc(nmemb * pos_bw);
    m_ropeBufferReady = true;
  }

  auto [q_scale, q_offset] = t_position_ids_cos->quantParam[0];
  if (d_pos == QNN_DATATYPE_FLOAT_16 || d_pos == QNN_DATATYPE_FLOAT_32) {
    // If floating point, don't quantize!
    q_scale  = 1.0;
    q_offset = 0;
  }

  // Calculate inv_freq array
  std::vector<double> inv_freq(m_pos_dim);
  const double exponent = 1.0 / static_cast<double>(m_pos_dim);
  for (uint32_t j = 0; j < m_pos_dim; j++) {
    inv_freq[j] = 1.0 / pow(theta, j * exponent);
  }
  double attention_factor = 1.0;
  if (rope_scaling.rope_type == RopeScalingParams::ROPE_LLAMA3) {
    // Implemented from HuggingFace
    // https://github.com/huggingface/transformers/blob/47c29ccfaf56947d845971a439cbe75a764b63d7/src/transformers/modeling_rope_utils.py#L298
    const double& factor           = rope_scaling.llama3_params.factor;
    const double& low_freq_factor  = rope_scaling.llama3_params.low_freq_factor;
    const double& high_freq_factor = rope_scaling.llama3_params.high_freq_factor;
    const int& old_context_len     = rope_scaling.llama3_params.original_max_position_embeddings;

    const double low_freq_wavelen  = old_context_len / low_freq_factor;
    const double high_freq_wavelen = old_context_len / high_freq_factor;

    for (uint32_t j = 0; j < m_pos_dim; j++) {
      const double wavelen = 2 * M_PI / inv_freq[j];
      if (wavelen < high_freq_wavelen)  // wavelen < high_freq_wavelen: do nothing
        continue;
      else if (wavelen > low_freq_wavelen)  // wavelen > low_freq_wavelen: divide by factor
        inv_freq[j] = 1.0 / static_cast<double>(factor * pow(theta, j * exponent));
      else {  // otherwise: interpolate between the two, using a smooth factor
        assert(low_freq_wavelen != high_freq_wavelen);
        const double smooth = (static_cast<double>(old_context_len) / wavelen - low_freq_factor) /
                              (high_freq_factor - low_freq_factor);
        inv_freq[j] = ((1 - smooth) * inv_freq[j] / factor + smooth * inv_freq[j]);
      }
    }
  } else if (rope_scaling.rope_type == RopeScalingParams::ROPE_LONGROPE) {
    // Validate factor >= 1.0, len(long_factor) == rope-dim and len(short_factor) == rope-dim
    const double& factor       = rope_scaling.longrope_params.factor;
    const int& old_context_len = rope_scaling.longrope_params.original_max_position_embeddings;

    const auto& inv_factors = (m_ctx_size > static_cast<size_t>(old_context_len))
                                  ? rope_scaling.longrope_params.long_factor
                                  : rope_scaling.longrope_params.short_factor;

    if (inv_factors.size() != m_pos_dim)
      throw std::runtime_error(
          fmt::format("long-factor (len={}) and short-factor (len={}) must have length rope-dim={}",
                      rope_scaling.longrope_params.long_factor.size(),
                      rope_scaling.longrope_params.short_factor.size(),
                      m_pos_dim));

    for (uint32_t j = 0; j < m_pos_dim; j++) {
      inv_freq[j] = inv_freq[j] / inv_factors[j];
    }

    attention_factor =
        std::sqrt(1.0 + std::log(factor) / std::log(static_cast<double>(old_context_len)));
  } else if (rope_scaling.rope_type == RopeScalingParams::ROPE_YARN) {
    const double& factor               = rope_scaling.yarn_params.factor;
    const size_t& dim                  = rope_scaling.yarn_params.dim;
    const double& extrapolation_factor = rope_scaling.yarn_params.extrapolation_factor;
    const double& attn_factor          = rope_scaling.yarn_params.attn_factor;
    const double& beta_fast            = rope_scaling.yarn_params.beta_fast;
    const double& beta_slow            = rope_scaling.yarn_params.beta_slow;
    const int& old_context_len         = rope_scaling.yarn_params.original_max_position_embeddings;
    // Validate dim before using it
    if (dim == 0 || dim % 2 != 0) {
      throw std::runtime_error("YaRN dim must be a positive even number");
    }
    const size_t half_dim = dim / 2;
    // Calculate correction range
    auto [low, high] =
        yarn_find_correction_range(beta_fast, beta_slow, dim, theta, old_context_len);
    // Generate linear interpolation mask
    std::vector<double> ramp_mask = yarn_linear_ramp_mask(low, high, half_dim);
    // Calculate frequency array
    for (size_t j = 0; j < half_dim; ++j) {
      const double pos_freq =
          std::pow(theta, static_cast<double>(j) / static_cast<double>(half_dim));
      const double inv_freq_extrapolation = 1.0 / pos_freq;
      const double inv_freq_interpolation = 1.0 / (factor * pos_freq);
      const double mask_value             = (1.0 - ramp_mask[j]) * extrapolation_factor;
      inv_freq[j] =
          inv_freq_interpolation * (1.0 - mask_value) + inv_freq_extrapolation * mask_value;
    }
    // Calculate scaling factor
    attention_factor = yarn_get_mscale(factor) * attn_factor;
  } else if (rope_scaling.rope_type == RopeScalingParams::ROPE_LINEAR) {
    const double& factor = rope_scaling.linear_params.factor;
    for (uint32_t j = 0; j < m_pos_dim; j++) {
      inv_freq[j] = inv_freq[j] / factor;
    }
  }

  // Calculate freq array
  std::vector<std::vector<double>> freqs(m_ctx_size, std::vector<double>(m_pos_dim));
  if (rope_scaling.rope_type == RopeScalingParams::ROPE_QWEN2VL_MROPE && m_visionParam.size() > 0) {
    // Set 3d position_ids. Refence HuggingFace
    // https://github.com/huggingface/transformers/blob/ca402e2116f5917ce0a03659b779a02a555b285f/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L968
    const uint32_t time_step          = rope_scaling.mrope_params.time_step;
    const uint32_t spatial_merge_size = rope_scaling.mrope_params.spatial_merge_size;

    std::vector<std::vector<uint32_t>> position_ids(3, std::vector<uint32_t>(m_ctx_size));
    uint32_t p           = 0;
    uint32_t pos_id_base = 0;
    for (size_t i = 0; i < m_visionParam.size(); i += 4) {
      const uint32_t vision_pos = m_visionParam[i];
      const uint32_t temporal   = m_visionParam[i + 1];
      const uint32_t height     = m_visionParam[i + 2] / spatial_merge_size;
      const uint32_t width      = m_visionParam[i + 3] / spatial_merge_size;
      // set position_id of pre-image text
      for (uint32_t j = 0; p < vision_pos; ++p, ++j) {
        position_ids[0][p] = j + pos_id_base;
        position_ids[1][p] = j + pos_id_base;
        position_ids[2][p] = j + pos_id_base;
      }

      // set position_id of image
      pos_id_base = vision_pos;
      for (uint32_t t = 0; t < temporal; ++t) {
        for (uint32_t h = 0; h < height; ++h) {
          for (uint32_t w = 0; w < width; ++w) {
            if (p >= m_ctx_size) break;
            position_ids[0][p] = t * time_step + pos_id_base;
            position_ids[1][p] = h + pos_id_base;
            position_ids[2][p] = w + pos_id_base;
            ++p;
          }
        }
      }
      pos_id_base =
          std::max({position_ids[0][p - 1], position_ids[1][p - 1], position_ids[2][p - 1]}) + 1;
    }

    // set position_id for rest place
    for (uint32_t j = 0; p < m_ctx_size; ++p, ++j) {
      position_ids[0][p] = j + pos_id_base;
      position_ids[1][p] = j + pos_id_base;
      position_ids[2][p] = j + pos_id_base;
    }

    // Compress 3d data. Refence HuggingFace
    // https://github.com/huggingface/transformers/blob/ca402e2116f5917ce0a03659b779a02a555b285f/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L564
    // Currently mrope_section size must be 3, it has been verified. Default: {16, 24, 24}
    const std::vector<uint32_t>& mrope_section = rope_scaling.mrope_params.mrope_section;
    std::vector<uint32_t> split_pos = {mrope_section[0], mrope_section[0] + mrope_section[1]};
    // The sum of mrope_section should equal to m_pos_dim.
    if (m_pos_dim !=
        std::accumulate(mrope_section.begin(), mrope_section.end(), static_cast<uint32_t>(0))) {
      __ERROR("The sum mrope-section {} is not equal to pos-dim {}", mrope_section, m_pos_dim);
      return false;
    }
    for (size_t i = 0; i < m_ctx_size; ++i) {
      for (size_t j = 0; j < m_pos_dim; ++j) {
        if (j < split_pos[0])
          freqs[i][j] = position_ids[0][i] * inv_freq[j];
        else if (j < split_pos[1])
          freqs[i][j] = position_ids[1][i] * inv_freq[j];
        else
          freqs[i][j] = position_ids[2][i] * inv_freq[j];
      }
    }
  } else if (rope_scaling.rope_type == RopeScalingParams::ROPE_QWEN3VL_MROPE &&
             m_visionParam.size() > 0) {
    // Set 3d position_ids for Qwen3-VL. HuggingFace reference:
    // https://github.com/huggingface/transformers/blob/v5.0.0rc2/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L953
    const uint32_t time_step          = rope_scaling.mrope_params.time_step;
    const uint32_t spatial_merge_size = rope_scaling.mrope_params.spatial_merge_size;

    std::vector<std::vector<uint32_t>> position_ids(3, std::vector<uint32_t>(m_ctx_size));
    uint32_t p           = 0;
    uint32_t pos_id_base = 0;

    for (size_t i = 0; i < m_visionParam.size(); i += 4) {
      const uint32_t vision_pos = m_visionParam[i];
      const uint32_t temporal   = m_visionParam[i + 1];
      const uint32_t height     = m_visionParam[i + 2] / spatial_merge_size;
      const uint32_t width      = m_visionParam[i + 3] / spatial_merge_size;

      // Set position_id of pre-image text
      for (uint32_t j = 0; p < vision_pos; ++p, ++j) {
        position_ids[0][p] = j + pos_id_base;
        position_ids[1][p] = j + pos_id_base;
        position_ids[2][p] = j + pos_id_base;
      }

      // Set position_id of image
      pos_id_base = vision_pos;
      for (uint32_t t = 0; t < temporal; ++t) {
        for (uint32_t h = 0; h < height; ++h) {
          for (uint32_t w = 0; w < width; ++w) {
            if (p >= m_ctx_size) break;
            position_ids[0][p] = t * time_step + pos_id_base;
            position_ids[1][p] = h + pos_id_base;
            position_ids[2][p] = w + pos_id_base;
            ++p;
          }
        }
      }
      pos_id_base =
          std::max({position_ids[0][p - 1], position_ids[1][p - 1], position_ids[2][p - 1]}) + 1;
    }

    // Set position_id for rest place
    for (uint32_t j = 0; p < m_ctx_size; ++p, ++j) {
      position_ids[0][p] = j + pos_id_base;
      position_ids[1][p] = j + pos_id_base;
      position_ids[2][p] = j + pos_id_base;
    }

    // Interleave 3D Position Encoding. HuggingFace reference
    // https://github.com/huggingface/transformers/blob/v5.0.0rc2/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L646
    // Currently mrope_section size must be 3. Default: {24, 20, 20}
    const std::vector<uint32_t>& mrope_section = rope_scaling.mrope_params.mrope_section;

    // The sum of mrope_section should equal to m_pos_dim.
    if (m_pos_dim !=
        std::accumulate(mrope_section.begin(), mrope_section.end(), static_cast<uint32_t>(0))) {
      __ERROR("The sum mrope-section {} is not equal to pos-dim {}", mrope_section, m_pos_dim);
      return false;
    }

    // Interleave the position IDs across the three dimensions
    for (size_t i = 0; i < m_ctx_size; ++i) {
      for (size_t j = 0; j < m_pos_dim; ++j) {
        size_t use_dim_idx = 0;
        size_t dim_idx     = j % 3;  // Cycle through temporal, height, width
        size_t freq_idx    = j / 3;  // Which frequency within the dimension
        if (dim_idx > 0 && freq_idx < mrope_section[dim_idx]) {
          use_dim_idx = dim_idx;
        }
        freqs[i][j] = position_ids[use_dim_idx][i] * inv_freq[j];
      }
    }
  } else {
    for (size_t i = 0; i < m_ctx_size; i++) {
      for (size_t j = 0; j < m_pos_dim; j++) {
        freqs[i][j] = i * inv_freq[j];
      }
    }
  }

  for (size_t i = 0; i < m_ctx_size; i++) {
    for (size_t j = 0; j < m_pos_dim; j++) {
      const double sin_val = ((sin(freqs[i][j]) * attention_factor) / q_scale) - q_offset;
      const double cos_val = ((cos(freqs[i][j]) * attention_factor) / q_scale) - q_offset;

      // round() instead of floor() seems to produce an acuracy drop. To debug later
      switch (d_pos) {
        case QNN_DATATYPE_UFIXED_POINT_8:
          (reinterpret_cast<uint8_t*>(rope_sin))[i * m_pos_dim + j] = static_cast<uint8_t>(sin_val);
          (reinterpret_cast<uint8_t*>(rope_cos))[i * m_pos_dim + j] = static_cast<uint8_t>(cos_val);
          break;
        case QNN_DATATYPE_UFIXED_POINT_16:
          (reinterpret_cast<uint16_t*>(rope_sin))[i * m_pos_dim + j] =
              static_cast<uint16_t>(sin_val);
          (reinterpret_cast<uint16_t*>(rope_cos))[i * m_pos_dim + j] =
              static_cast<uint16_t>(cos_val);
          break;
        case QNN_DATATYPE_FLOAT_16:
          (reinterpret_cast<uint16_t*>(rope_sin))[i * m_pos_dim + j] =
              fp16_ieee_from_fp32_value(sin_val);
          (reinterpret_cast<uint16_t*>(rope_cos))[i * m_pos_dim + j] =
              fp16_ieee_from_fp32_value(cos_val);
          break;
        case QNN_DATATYPE_FLOAT_32:
          (reinterpret_cast<float*>(rope_sin))[i * m_pos_dim + j] = static_cast<float>(sin_val);
          (reinterpret_cast<float*>(rope_cos))[i * m_pos_dim + j] = static_cast<float>(cos_val);
          break;
        default:
          __ERROR("Unsupported position ids datatype {}", d_pos.str());
          return false;
      }
    }
  }

  if (_debug_tensors) {
    std::string dtype =
        fmt::format("{}{}", (d_pos == QNN_DATATYPE_FLOAT_16) ? "f" : "u", pos_bw * 8);
    std::string fname_sin = fmt::format("{}/position_ids_sin.{}.dat", _debug_path, dtype);
    std::string fname_cos = fmt::format("{}/position_ids_cos.{}.dat", _debug_path, dtype);
    QnnUtils::writeRawData(rope_sin, nmemb * pos_bw, fname_sin);
    QnnUtils::writeRawData(rope_cos, nmemb * pos_bw, fname_cos);
  }

  m_ropeInitialized = true;
  return true;
}

bool QnnNspModel::load_lmhead_weight_as_input(void) {
  if (!_lmhead_weight_input) return true;
  if (_lmhead_weight_input && lmhead_weight_dir.empty()) {
    __ERROR("NSPModel: LMhead weight file not found");
    return false;
  }
  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.input_specs) {
      if (tname.compare("weight") == 0) {
        // weight tensor file name should be in same format as tensor name present in graph
        std::string weight_file =
            (model_basedir / fs::path(lmhead_weight_dir) / fs::path(tname + ".raw")).string();

        QnnUtils::Dims dims = tspec.dims;
        size_t numElements  = dims.getNumElements();

        size_t size = sizeof(float);
        std::vector<float> weight_f32;  // Temporary variable to load fp32 values
        weight_f32.resize(numElements);

        FILE* fp = fopen(weight_file.c_str(), "rb");
        if (fp == NULL) {
          __ERROR("NSPModel: Error opening file: {}", weight_file);
          return false;
        }

        size_t count = fread(weight_f32.data(), size, numElements, fp);
        fclose(fp);

        if (count != numElements) {
          __ERROR("NSPModel: Could not load {} - expected file size {}",
                  weight_file,
                  numElements * size);
          return false;
        }

        int8_t* weight_buffer = reinterpret_cast<int8_t*>(getBuffer(tspec));
        // Quantize the values, per width quantization
        QnnUtils::perWidthQuantizeTensorPtr(weight_f32.data(),
                                            weight_buffer,
                                            tspec.quantParam,
                                            dims.height,
                                            dims.width,
                                            dims.channel);
      }
    }
  }
  return true;
}

void QnnNspModel::getInputQuantParam(double& scale, int& offset) {
  auto tmp = t_input_ids->quantParam[0];
  scale    = tmp.scale;
  offset   = tmp.offset;
}

size_t QnnNspModel::loadKVCache(const std::string& load_path, bool /*chooseHigherVariant*/) {
  m_kvmanager->block(Scope::global());
  size_t ret = m_kvmanager->loadKVCache(load_path);
  if (m_kvmanager->failed()) State::error(m_kvmanager->error());
  return ret;
}

bool QnnNspModel::saveKVCache(const std::string& save_path) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->dumpKVCache(save_path);
  if (m_kvmanager->failed()) State::error(m_kvmanager->error());
  return ret;
}

bool QnnNspModel::saveTokenHistory(const std::string& savePath) {
  if (!token_history_enabled || token_history.empty()) {
    return true;
  }

  fs::path tokenHistoryPath = fs::path(savePath);
  std::ofstream file(tokenHistoryPath, std::ios::out | std::ios::binary);
  if (!file) {
    __ERROR("Failed to open file for writing: {}", tokenHistoryPath.string());
    return false;
  }

  // Write raw int32_t data
  file.write(reinterpret_cast<const char*>(token_history.data()),
             static_cast<std::streamsize>(token_history.size() * sizeof(int32_t)));

  if (!file) {
    __ERROR("Failed to write token history to: {}", tokenHistoryPath.string());
    return false;
  }

  return true;
}

bool QnnNspModel::loadTokenHistory(const std::string& loadPath) {
  fs::path tokenHistoryPath(loadPath);

  if (!fs::exists(tokenHistoryPath)) {
    __WARN("Token history file not found: {}", tokenHistoryPath.string());
    return false;
  }

  std::ifstream file(tokenHistoryPath, std::ios::in | std::ios::binary | std::ios::ate);
  if (!file) {
    __ERROR("Failed to open file for reading: {}", tokenHistoryPath.string());
    return false;
  }

  auto endPos = file.tellg();
  if (endPos <= 0) {
    __ERROR("Token history file is empty or invalid: {}", tokenHistoryPath.string());
    return false;
  }

  std::size_t fileSize = static_cast<std::size_t>(endPos);
  if (fileSize % sizeof(int32_t) != 0) {
    __ERROR("Improper file size (not multiple of int32_t): {}", tokenHistoryPath.string());
    return false;
  }

  std::size_t size = fileSize / sizeof(int32_t);
  token_history.assign(size, 0);

  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(token_history.data()), static_cast<std::streamsize>(fileSize));

  if (!file) {
    __ERROR("Failed to read tokens from {}", tokenHistoryPath.string());
    token_history.clear();
    return false;
  }

  return true;
}

bool QnnNspModel::saveKVCacheToBuffer(Buffer* kvBuff) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->dumpKVCache(kvBuff);
  if (m_kvmanager->failed()) State::error(m_kvmanager->error());
  return ret;
}

bool QnnNspModel::getCacheSpec(CacheFileSpec& spec) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->getCacheSpec(spec);
  return ret;
}

bool QnnNspModel::getKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->getKVHead(spec, layer, head, data, scale);
  return ret;
}

void QnnNspModel::setHigherVariant() {
  auto& [new_variant, _] = nsp_graph_count.rbegin()->first;  // Guarantees largest variant, then ctx
  m_kvmanager->setActiveVariant(new_variant, -1);
}

size_t QnnNspModel::getEmbeddings(std::span<float> embds, InferenceStep& step) {
  qualla::Timer start;

  QnnUtils::Tensor* output_spec =
      m_nsp_graphs.back()(step.variant, step.ctx_size)
          ->getOutput(m_cross_attention ? m_layerNames[LayerType::CROSS_ATTN_STATES]
                                        : (m_pooled_output ? m_layerNames[LayerType::POOL_OUTPUT]
                                                           : m_layerNames[LayerType::SEQ_OUTPUT]));

  if (output_spec == nullptr) {
    __ERROR("encountered null buffer");
    throw std::runtime_error("Model is not supporting per token embedding");
  }
  const auto scale  = output_spec->quantParam[0].scale;
  const auto offset = output_spec->quantParam[0].offset;

  auto output_datatype   = QnnUtils::DataType(output_spec->tensor);
  uint32_t output_bw     = output_spec->dtype.bw();
  uint8_t* output_buffer = reinterpret_cast<uint8_t*>(getBuffer(output_spec));

  const int return_size = (m_pooled_output ? 1 : step.n_process);
  if (!m_cross_attention) {
    if (!m_pooled_output) {
      // If multiple tokens embedding are returned, offset to the correct location in the buffer
      if (step.variant == step.ctx_size) {
        // This was left-padded, tokens embedding are at [n_tokens - n_processed, n_tokens]
        output_buffer +=
            static_cast<uint32_t>(step.variant - return_size) * m_embd_size * output_bw;
      } else {
        // This was right-padded, tokens embedding are at indexes [0, n_processed]
        output_buffer += static_cast<uint32_t>(step.n_process - 1) * m_embd_size * output_bw;
      }
    }
  }

  const size_t output_len = static_cast<size_t>(return_size) * m_embd_size;
  __TRACE("qnn-htp: get-embds for {} tokens. scale = {}, offset = {}, Returning {}",
          step.n_process,
          scale,
          offset,
          output_len);

  switch (output_datatype) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      deQuantizeOutputs(
          reinterpret_cast<uint8_t*>(output_buffer), embds, scale, offset, output_len);
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      deQuantizeOutputs(
          reinterpret_cast<uint16_t*>(output_buffer), embds, scale, offset, output_len);
      break;
    case QNN_DATATYPE_FLOAT_16:
      castOutputs(reinterpret_cast<uint16_t*>(output_buffer), embds, output_len, output_bw);
      break;
    case QNN_DATATYPE_FLOAT_32:
      castOutputs(reinterpret_cast<float*>(output_buffer), embds, output_len, output_bw);
      break;
    default:
      __ERROR("Unsupported output datatype");
  }

  __DEBUG("qnn-htp: getEmbeddings complete : {} usec (return_size={})",
          start.elapsed_usec(),
          output_len);
  return output_len;
}

size_t QnnNspModel::getIOBufferByName(std::string tensor_name, void*& buffer, bool isPrompt) {
  int32_t token =
      isPrompt ? nsp_graph_count.rbegin()->first.first : nsp_graph_count.begin()->first.first;
  int32_t ctxt =
      isPrompt ? nsp_graph_count.rbegin()->first.second : nsp_graph_count.begin()->first.second;
  __DEBUG("getIOBufferByName isPrompt {} token {} ctxt {}", isPrompt, token, ctxt);

  for (QnnNspGraph& graph : m_nsp_graphs) {
    if (!graph.variants.contains({token, ctxt})) continue;
    GraphVariant* variant = graph(token, ctxt);
    if (variant->getOutput(tensor_name) != nullptr) {
      buffer             = getBuffer(variant->getOutput(tensor_name));
      size_t buffer_size = getBufferSize(variant->getOutput(tensor_name));
      __DEBUG("qnn-htp: getIOBufferByNam output tensor_name {} address {} buffer_size {}",
              tensor_name,
              reinterpret_cast<uintptr_t>(buffer),
              buffer_size);

      break;
    }
    if (variant->getInput(tensor_name) != nullptr) {
      buffer             = getBuffer(variant->getInput(tensor_name));
      size_t buffer_size = getBufferSize(variant->getInput(tensor_name));
      __DEBUG("qnn-htp: getIOBufferByNam input tensor_name {} address {} buffer_size {}",
              tensor_name,
              reinterpret_cast<uintptr_t>(buffer),
              buffer_size);
      break;
    }
  }

  return static_cast<size_t>(token);
}

bool QnnNspModel::finalizeState(std::shared_ptr<EngineState>& engineState) {
  IOEVENT event = engineState->isInitialize() ? engineState->getIOBuffer()->m_event
                                              : IOEVENT::ALLOCATE_REGISTER_EVENT;

  __DEBUG("qnn-htp: Event triggered {}", ioEventMap[event]);
  if (event == IOEVENT::NO_EVENT) {
    return true;
  }

  if (m_kvmanager) {
    m_kvmanager->deRegisterAll();
  }

  if (true != QnnNspBaseModel::finalizeState(engineState)) {
    return false;
  }

  m_lazyInitialization = false;

  if (!initializeIOTensors()) {
    __ERROR("Error in re-initializing the Tensors");
    return false;
  }

  if (event == IOEVENT::ALLOCATE_REGISTER_EVENT) {
    // reinitialize the KV manager or initialize
    if (true != initializeKVManager()) {
      __ERROR("Error in allocating the KV manager memory");
      return false;
    }
    if (true != initializeTensorPointers()) {
      __ERROR("Error in initializing Tensor pointers");
      return false;
    }
    if (true != calculate_rope_embeddings()) {
      __ERROR("Error in creating Rope Data");
      return false;
    }
    engineState->initialize(std::dynamic_pointer_cast<IOBuffer>(m_kvmanager));

  } else if (event == IOEVENT::REGISTER_EVENT) {
    m_kvmanager = std::dynamic_pointer_cast<KVManager>(engineState->getIOBuffer());
    // might need to update some static fields.
    if (true != initializeTensorPointers()) {
      __ERROR("Error in initializing Tensor pointers");
      return false;
    }
    if (true != calculate_rope_embeddings()) {
      __ERROR("Error in creating Rope Data");
      return false;
    }
  }
  // always change event to NOEVENT after all processing is done
  if (true != engineState->changeIOEvent(IOEVENT::NO_EVENT)) {
    __ERROR("Error: Failed to set IO Event for engine states");
    return false;
  }

  m_lazyInitialization = true;

  return true;
}

bool QnnNspModel::setVisionParam(const std::vector<uint32_t>& visionParam) {
  if (visionParam.empty() || (m_positional_encoding.rope_params.rope_scaling.rope_type !=
                                  RopeScalingParams::ROPE_QWEN2VL_MROPE &&
                              m_positional_encoding.rope_params.rope_scaling.rope_type !=
                                  RopeScalingParams::ROPE_QWEN3VL_MROPE)) {
    return true;
  }

  bool sameVisionParam = visionParam.size() == m_visionParam.size();
  for (size_t i = 0; i < visionParam.size(); ++i) {
    if (!sameVisionParam) {
      break;
    }
    sameVisionParam = visionParam[i] == m_visionParam[i];
  }

  bool result = true;
  if (!sameVisionParam) {
    m_visionParam     = visionParam;
    m_ropeInitialized = false;
    __INFO("after copy visionParam to m_visionParam, visionParam: {}, m_visionParam: {}",
           visionParam,
           m_visionParam);
    result = calculate_rope_embeddings();
    ;
  }
  return result;
}

bool QnnNspModel::setCrossAttentionHiddenStates(const std::vector<float>& crossAttentionStates) {
  uint16_t* tensor_ptr         = reinterpret_cast<uint16_t*>(getBuffer(t_cross_attn_states));
  auto [scale, offset]         = t_cross_attn_states->quantParam[0];
  const auto byteWidth         = static_cast<size_t>(t_cross_attn_states->dtype.bw());
  m_cross_attention_input_size = crossAttentionStates.size();
  __DEBUG("nsp-model: setting cross attention hidden states of size {}",
          m_cross_attention_input_size / m_embd_size);

  if ((byteWidth == 4) && (scale == 0.0) && (offset == 0)) {
    // Tensor is float -- no quantization required.
    std::memcpy(tensor_ptr, crossAttentionStates.data(), crossAttentionStates.size() * byteWidth);
  } else if ((byteWidth == 2) && (scale == 0.0) && (offset == 0)) {
    // Tensor is float16 -- convert from float32 to float16
    for (size_t i = 0; i < crossAttentionStates.size(); i++) {
      tensor_ptr[i] = fp16_ieee_from_fp32_value(crossAttentionStates[i]);
    }
  } else {
    QnnUtils::quantizeTensorPtr(
        crossAttentionStates.data(), tensor_ptr, offset, scale, crossAttentionStates.size());
  }
  return true;
}

template <typename DType>
void QnnNspModel::setupCrossAttentionMask(const InferenceStep& step,
                                          const std::vector<int32_t>& tokens,
                                          uint32_t cross_attention_len) {
  if (m_mask_type == "cross-attn-row-mask") {
    const size_t variant             = static_cast<size_t>(step.variant);
    DType* cross_attn_mask_buffer    = reinterpret_cast<DType*>(getBuffer(t_cross_attn_mask));
    DType* full_text_row_mask_buffer = reinterpret_cast<DType*>(getBuffer(t_full_text_row_mask));

    if (!t_cross_attn_mask || !t_full_text_row_mask) {
      __ERROR(
          "cross-attn-row-mask mask type requires cross_attn_mask and full_text_row_mask tensors");
      return;
    }

    int image_token_pos = -1;
    DType pos_val = std::numeric_limits<DType>::max(), neg_val = 0;

    // get the position of the image token in the seq
    for (int i = 0; i < static_cast<int>(tokens.size()); i++) {
      if (tokens[static_cast<size_t>(i)] == m_img_token) {
        image_token_pos = i;
        break;
      }
    }

    DType* cur_attn_ptr = cross_attn_mask_buffer;
    size_t n_patches =
        static_cast<size_t>(getBufferSize(t_cross_attn_mask) / variant / sizeof(DType));
    for (int i = 0; i < static_cast<int>(variant); i++) {
      if ((i) < image_token_pos || i >= static_cast<int>(tokens.size())) {
        full_text_row_mask_buffer[i] = neg_val;
      } else {
        full_text_row_mask_buffer[i] = pos_val;
      }
      // mllama: attend to all patches for cross-attention mask
      std::fill_n(cur_attn_ptr, n_patches, pos_val);
      cur_attn_ptr += n_patches;
    }
  } else {
    // default to mt5 style cross-attention mask
    DType* cross_attn_mask_buffer = reinterpret_cast<DType*>(
        getBuffer((m_modelArchitectureType == ModelArchitectureType::ENCODER) ? t_attn_mask
                                                                              : t_cross_attn_mask));
    DType pos_val = 1, neg_val = 0;
    DType* cur_attn_ptr = cross_attn_mask_buffer;
    __DEBUG("nsp-model: cross attention mask is setup to attend to {} cross tokens out of {}",
            cross_attention_len,
            step.ctx_size);
    std::fill_n(cur_attn_ptr, cross_attention_len, pos_val);
    std::fill_n(cur_attn_ptr + cross_attention_len, step.ctx_size, neg_val);
  }
}

void QnnNspModel::resetState() {
  if (m_qnnApi) {
    m_qnnApi->resetAdapterGraphHandleState();
  }
  if (!m_visionParam.empty()) {
    m_visionParam.clear();
    m_ropeInitialized    = false;
    bool prevLazy        = m_lazyInitialization;
    m_lazyInitialization = false;
    calculate_rope_embeddings();
    m_lazyInitialization = prevLazy;
  }
}
}  // namespace qualla
