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
#include <functional>
#include <iostream>
#include <numeric>
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

#include "fmt/format.h"
#include "fmt/os.h"
#include "fmt/ranges.h"
#include "nsp-image-model.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/timer.hpp"

namespace fs = std::filesystem;

namespace qualla {

QnnNspImageModel::QnnNspImageModel(std::shared_ptr<Env> env, const QnnNspBaseModel::Params& params)
    : QnnNspBaseModel(env, params) {
  // Initialize QnnAPI
  spill_fill_buffer_size  = params.spill_fill_bufsize;
  m_use_mmap              = params.use_mmap;
  mmap_budget             = params.mmap_budget;
  graph_switching         = params.graph_switching;
  load_select_graphs      = params.load_select_graphs;
  m_modelArchitectureType = params.modelArchitectureType;
  m_mask_type             = params.mask_type;
  // Positional encoding parameters
  m_positional_encoding   = params.positional_encoding_params;
  m_internal_pos_encoding = m_positional_encoding.type != PositionalEncoding::UNDEFINED;

  m_vision_param = params.vision_param;
  m_dlc_path     = params.dlc_path;

  if (graph_switching && !m_use_mmap)
    __WARN("Graph switching with non-mmaped implementation can cause high sustained memory usage");

  exec_select_graphs = params.exec_select_graphs;
  if (!exec_select_graphs.empty())
    __DEBUG("qnn-htp : Execute selected graphs = {}", exec_select_graphs);

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

  m_qnnApi->setIOTensor(m_ioTensor);
  m_qnnApi->setDataAlignmentSize(m_dataAlignmentSize);

  if (params.debug_specs || params.debug_tensors) {
    if (!fs::exists(params.debug_path) && !fs::create_directories(params.debug_path))
      throw std::runtime_error("Could not create debug directory : " + params.debug_path);
  }
}

QnnNspImageModel::~QnnNspImageModel() {
  qualla::Timer start;

  __DEBUG("qnn-htp: model destruct complete: {} usec", start.elapsed_usec());
}

// Given a filename, initializeModel load and initializes QNN runtime libraries and the model
bool QnnNspImageModel::initializeModel(void) {
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
  if (logger) {
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

  size_t n_splits = m_num_graphs = m_qnnApi->getGraphsCount();
  __INFO("qnn-api initialized with {} graph(s)", m_num_graphs);
  auto graphs_info = m_qnnApi->getGraphsInfo();
  m_variant_list.reserve(m_num_graphs);
  // Create NSPGraph for each splits
  m_nsp_graphs.reserve(n_splits);
  std::map<int32_t, std::vector<std::string>> graph_names;
  for (size_t graph_idx = 0; graph_idx < m_num_graphs; graph_idx++) {
    qnn_wrapper_api::GraphInfo_t* const graph_info = graphs_info[graph_idx];

    for (size_t i = 0; i < graph_info->numInputTensors; ++i) {
      std::string input_name = QNN_TENSOR_GET_NAME(graph_info->inputTensors[i]);
      // skip checking of default input "pixel_values"
      if (input_name == "position_ids_sin") {
        m_layerNames[LayerType::POS_SIN] = "position_ids_sin";
      } else if (input_name == "position_ids_cos") {
        m_layerNames[LayerType::POS_COS] = "position_ids_cos";
      } else if (input_name == "window_attention_mask") {
        m_layerNames[LayerType::WINDOW_ATTN_MASK] = "window_attention_mask";
      } else if (input_name == "full_attention_mask") {
        m_layerNames[LayerType::FULL_ATTN_MASK] = "full_attention_mask";
      } else if (input_name == "aspect_ratio_mask") {
        m_layerNames[LayerType::FULL_ATTN_MASK] = "aspect_ratio_mask";
      } else if (input_name == "pretile_embedding") {
        m_layerNames[LayerType::PRETILE_EMBED] = "pretile_embedding";
      } else if (input_name == "posttile_embedding") {
        m_layerNames[LayerType::POSTTILE_EMBED] = "posttile_embedding";
      } else if (input_name == "gated_pos_embedding") {
        m_layerNames[LayerType::GATED_POS_EMBED] = "gated_pos_embedding";
      }
    }
    for (size_t i = 0; i < graph_info->numOutputTensors; ++i) {
      std::string ouput_name = QNN_TENSOR_GET_NAME(graph_info->outputTensors[i]);
      if (ouput_name == "vision_embedding") {
        m_layerNames[LayerType::OUTPUT] = "vision_embedding";
      } else if (ouput_name == "cross_attention_states") {
        m_layerNames[LayerType::OUTPUT] = "cross_attention_states";
      }
    }
    if (!m_layerNames.count(LayerType::OUTPUT)) {
      std::string ouput_name = QNN_TENSOR_GET_NAME(graph_info->outputTensors[0]);
      if (ouput_name.starts_with("deepstack_visual_embed")) {
        m_layerNames[LayerType::OUTPUT] = ouput_name;
      }
    }

    GraphVariant graph(graph_info, m_layerNames, _env);
    graph.n_tokens = 0;
    __DEBUG("qnn-htp: Graph {}", graph.graph_name);

    if (exec_select_graphs.size() != 0 &&
        std::find(exec_select_graphs.begin(), exec_select_graphs.end(), graph.graph_name) ==
            exec_select_graphs.end()) {
      __DEBUG("qnn-htp: Graph {} is not selected to execute based on conf file", graph.graph_name);
      continue;
    }
    m_variant_list.emplace_back(graph);
    graph_names[n_splits].push_back(graph.graph_name);
    m_graph_map[std::string(graph_info->graphName)] = &m_variant_list.back();
    m_nsp_graphs.emplace_back(graph_idx, _env, m_qnnApi.get(), m_ioTensor);
    m_nsp_graphs.back().setDebugMode(_debug_specs, _debug_tensors, _debug_path);
  }

  if (exec_select_graphs.size() != 0 && graph_names.empty()) {
    __ERROR("No matching graphs based on conf file");
  }

  // Insert all GraphVariants into corresponding NSPGraph
  for (auto& [input_size, graphs] : graph_names) {
    std::sort(graphs.begin(), graphs.end());
    for (size_t idx = 0; idx < graphs.size(); idx++)
      m_nsp_graphs.at(idx).addGraph(m_graph_map.at(graphs[idx]));
  }

  if (_debug_specs) dumpTensorSpecs();
  if (!m_qnnApi->initializeHtp(m_backend,
                               model_filelist,
                               BackendExtensionsConfigs(backendExtensionsLibPath,
                                                        m_backendExtensionsConfigPath.string()),
                               {},           // graphConfigs
                               true,         // loadFromCachedBinary
                               m_systemLib,  // systemLibraryPath
                               false,
                               spill_fill_buffer_size,
                               m_use_mmap,
                               m_use_async_Init,
                               mmap_budget,
                               _debug_qnn,
                               graph_switching,
                               exec_select_graphs,
                               load_select_graphs,
                               false,
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
bool QnnNspImageModel::initializeIOTensors() {  // IO Tensor Mem Registration is already done within
                                                // the
                                                // model_initailize by Qnn_API for Sync Init.
  if (m_lazyInitialization) return true;

  // set loraWeights Enabled
  _lora_enabled = m_qnnApi->getLoraWeightEnabled();
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
  return true;
}

static bool checkShape(const std::string& tensor_name,
                       const QnnUtils::Tensor* tensor,
                       int32_t height,
                       int32_t width,
                       int32_t channel,
                       int32_t bitwidth,
                       std::vector<std::tuple<std::string, std::string, std::string>>& errors) {
  if (tensor == nullptr) return true;
  const QnnUtils::Dims& tDims = tensor->dims;

  if ((height == -1 || static_cast<uint32_t>(height) == tDims.height) &&
      (width == -1 || static_cast<uint32_t>(width) == tDims.width) &&
      (channel == -1 || static_cast<uint32_t>(channel) == tDims.channel) &&
      (bitwidth == -1 || static_cast<uint32_t>(bitwidth) == tDims.bitwidth)) {
    return true;
  }

  std::stringstream err_msg;
  err_msg << "Expected [ " << height << ", " << width << ", " << channel << "] "
          << "bitwidth=" << bitwidth << ". Found [ " << tDims.height << ", " << tDims.width << ", "
          << tDims.channel << "] "
          << "bitwidth=" << tDims.bitwidth;

  errors.push_back({"ShapeError", tensor_name, err_msg.str()});
  return false;
}

// Run all validations for the model here so we can exit early
bool QnnNspImageModel::validateModel() {
  std::vector<std::tuple<std::string, std::string, std::string>> errors;

  QnnUtils::Tensor* tensor;

  // default input type is token
  m_inputType = InputType::PIXELS;

  for (auto& [input_size, variant] : m_nsp_graphs.front().variants) {
    // checking output
    if (variant->getOutput(m_layerNames[LayerType::OUTPUT]) == nullptr) {
      errors.push_back({variant->graph_name, m_layerNames[LayerType::OUTPUT], "Tensor not found"});
    }
    // checking input
    for (auto& [layer_type, layer_name] : m_layerNames) {
      if (layer_type == LayerType::OUTPUT) {
        continue;
      }
      if ((tensor = variant->getInput(layer_name)) == nullptr) {
        errors.push_back({variant->graph_name, layer_name, "Tensor not found"});
      } else {
        checkShape(
            layer_name, tensor, -1, -1, -1, static_cast<int32_t>(tensor->dtype.bw()), errors);
      }
    }

    if (m_internal_pos_encoding) {
      if (m_positional_encoding.type == PositionalEncoding::ROPE) {
        uint32_t pos_dim = static_cast<uint32_t>(m_positional_encoding.rope_params.dims);
        auto pos_sin_t   = variant->getInput(m_layerNames[LayerType::POS_SIN]);
        auto pos_cos_t   = variant->getInput(m_layerNames[LayerType::POS_COS]);
        if (pos_dim != pos_sin_t->dims.channel) {
          errors.push_back({variant->graph_name,
                            m_layerNames[LayerType::POS_SIN],
                            fmt::format("parameter pos_dim {} does not match the shape [{}]",
                                        pos_dim,
                                        pos_sin_t->dims.getVector())});
        }
        if (pos_dim != pos_cos_t->dims.channel) {
          errors.push_back({variant->graph_name,
                            m_layerNames[LayerType::POS_COS],
                            fmt::format("parameter pos_dim {} does not match the shape [{}]",
                                        pos_dim,
                                        pos_cos_t->dims.getVector())});
        }
        if (m_positional_encoding.rope_params.rope_scaling.rope_type ==
                RopeScalingParams::ROPE_QWEN2VL &&
            m_vision_param.type == VisionParam::UNDEFINED) {
          errors.push_back({variant->graph_name,
                            m_layerNames[LayerType::INPUT],
                            fmt::format("Missing parameter option vision-param")});
        }
      }
    }

    if (m_vision_param.type != VisionParam::UNDEFINED) {
      uint32_t height = m_vision_param.height;
      uint32_t width  = m_vision_param.width;
      auto pixel_dim  = variant->getInput(m_layerNames[LayerType::INPUT])->dims.getVector();
      auto it         = std::find(pixel_dim.begin(), pixel_dim.end(), height * width);
      if (it == pixel_dim.end()) {
        errors.push_back(
            {variant->graph_name,
             m_layerNames[LayerType::INPUT],
             fmt::format("parameter height {} and width {} does not match the shape [{}]",
                         height,
                         width,
                         pixel_dim)});
      }
    }
  }

  if (errors.size() > 0) {
    QNN_ERROR("Model Validation Errors found");
    for (auto& [graph_name, tensor_name, err_msg] : errors)  // Log the list of errors
      QNN_ERROR("%s : %s - %s", graph_name.c_str(), tensor_name.c_str(), err_msg.c_str());
    QNN_ERROR("Note: Dimensions denoted by '-1' are ignored (i.e. no comparison)");
    QNN_ERROR("Check model i/o specs (set dump-specs=true in config) for debugging");
    State::fatal("Error validating HTP models");
    return false;
  }

  return true;
}

inline bool QnnNspImageModel::updateTensorPointer(GraphVariant& variant,
                                                  std::string& key,
                                                  QnnUtils::Tensor*& t) {
  QnnUtils::Tensor* tensor_ptr = variant.getInput(key);
  if (tensor_ptr == nullptr) {
    tensor_ptr = variant.getOutput(key);
    if (tensor_ptr == nullptr) return true;
  }
  if (t == nullptr) t = tensor_ptr;
  getBuffer(tensor_ptr);
  if (getBuffer(t) == getBuffer(tensor_ptr)) return true;
  __ERROR("{} has different addresses: {} vs {}",
          key,
          static_cast<void*>(t),
          static_cast<void*>(tensor_ptr));
  return false;
}

bool QnnNspImageModel::initializeTensorPointers() {
  // Ideally this needs to be done for all sets of AR-n available, e.g. for AR-1 and AR-1024
  if (m_lazyInitialization) return true;

  bool status = true;
  for (auto& variant : m_variant_list) {
    for (auto& [layer_type, layer_name] : m_layerNames) {
      if (layer_type == LayerType::OUTPUT) {
        status &= updateTensorPointer(variant, layer_name, t_output_tensor);
      } else if (layer_type == LayerType::POS_SIN && m_internal_pos_encoding) {
        status &= updateTensorPointer(variant, layer_name, t_position_ids_sin);
      } else if (layer_type == LayerType::POS_COS && m_internal_pos_encoding) {
        status &= updateTensorPointer(variant, layer_name, t_position_ids_cos);
      } else if (layer_type == LayerType::FULL_ATTN_MASK && m_internal_pos_encoding) {
        status &= updateTensorPointer(variant, layer_name, t_full_attention_mask);
      } else if (layer_type == LayerType::FULL_ATTN_MASK && m_mask_type == "aspect-ratio") {
        status &= updateTensorPointer(variant, layer_name, t_full_attention_mask);
      } else if (layer_type == LayerType::WINDOW_ATTN_MASK && m_internal_pos_encoding) {
        status &= updateTensorPointer(variant, layer_name, t_window_attention_mask);
      } else {
        status &= updateTensorPointer(variant, layer_name, t_input_tensors[layer_name]);
      }
    }
  }

  if (!status) {
    __ERROR("qnn-htp: Error in setting up named tensor pointers.");
    return false;
  }

  // Detect activation bitwidth
  if (!t_output_tensor) {
    __ERROR("Tensor not found: {}", m_layerNames[LayerType::OUTPUT]);
    return false;
  }
  d_output = t_output_tensor->dtype;
  if (!isSupportedActivation(d_output)) {
    __ERROR("Output Tensor: {} as unsupported activation type {}",
            m_layerNames[LayerType::OUTPUT],
            d_output.str());
    return false;
  }
  __DEBUG("qnn-htp datatypes: d_output {} ", d_output.str());
  for (auto& [layer_name, input_tensor] : t_input_tensors) {
    if (!input_tensor) {
      __ERROR("Tensor not found: {}", layer_name);
      return false;
    }
    QnnUtils::DataType& d_input = d_inputs[layer_name];
    d_input                     = input_tensor->dtype;
    if (!isSupportedActivation(d_input)) {
      __ERROR("Input Tensor: {} as unsupported activation type {}", layer_name, d_input.str());
      return false;
    }
    __DEBUG("qnn-htp datatypes: d_input {} ", d_input.str());
  }

  // For Position_IDs check data bitWidth
  if (m_internal_pos_encoding) {
    if (m_positional_encoding.type == PositionalEncoding::ROPE) d_pos = t_position_ids_sin->dtype;
    if (m_positional_encoding.type == PositionalEncoding::ROPE && !isSupportedActivation(d_pos)) {
      __ERROR("position encoding tensor has unsupported type {}", d_pos.str());
      return false;
    }
  }

  prepareAttentionMask();
  return true;
}

template <typename DType>
bool QnnNspImageModel::setupInput(const std::vector<uint8_t>& inputs, const std::string& name) {
  // Setup input tensor
  {
    QnnUtils::Tensor* input_tensor = t_input_tensors[name];
    size_t numElements =
        std::accumulate(input_tensor->tensor->v1.dimensions,
                        input_tensor->tensor->v1.dimensions + input_tensor->tensor->v1.rank,
                        static_cast<size_t>(1),
                        std::multiplies<size_t>());

    size_t bufferSize = d_inputs[name].bw() * numElements;
    if (embedding_datatype == "QNN_DATATYPE_FLOAT_32") {
      float* embeddingSrc = reinterpret_cast<float*>(const_cast<uint8_t*>(inputs.data()));
      quantizeInput(embeddingSrc, name, 0, numElements);
    } else {  // native datatype
      // Copy the data input vector
      std::copy(inputs.data(),
                inputs.data() + bufferSize,
                reinterpret_cast<uint8_t*>(getBuffer(input_tensor)));
    }
  }

  return true;
}

bool QnnNspImageModel::setupInputFP16(const std::vector<uint8_t>& /*inputs*/,
                                      const std::string& /*name*/) {
  // Placeholder for FP16 inputs
  return true;
}

bool QnnNspImageModel::setupInputTensors(
    const std::unordered_map<std::string, std::vector<uint8_t>>& inputs) {
  qualla::Timer start;
  // clang-format off
  for (auto& [name, data] : inputs) {
    switch (d_inputs[name]) {
      case QNN_DATATYPE_UFIXED_POINT_8:
        setupInput<uint8_t>(data, name);
        break;
      case QNN_DATATYPE_UFIXED_POINT_16:
        setupInput<uint16_t>(data, name);
        break;
      case QNN_DATATYPE_INT_32:
        setupInput<int32_t>(data, name);
        break;
      case QNN_DATATYPE_FLOAT_16:
        setupInputFP16(data, name);
        break;
      default:
        __ERROR("Unsupported input tensor {} dtype {}", name, d_inputs[name].str());
        return false;
    }
  }
  // clang-format on

  __TRACE("qnn-htp: setup-input-tensors complete : {} usec", start.elapsed_usec());
  return true;
}

size_t QnnNspImageModel::runInference(
    const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
    std::vector<uint8_t>& outputs) {
  qualla::Timer start;

  // Select variant based on variant_latency, or default to current variant

  // Technical note: int32_t can hold upto 596 hours
  // Even int16_t should be sufficient here - it holds upto 32.8 seconds
  int32_t total_wait = 0;
  int32_t total_exec = 0;

  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    if (!setupInputTensors(inputs)) return 0;

    // Call prepareInputs for all registered adaptors using base class helper
    bool shouldSkipInference = false;
    if (!prepareAdaptorInputs(0,     // variant
                              -1,    // ctx_size
                              0,     // n_past
                              0,     // n_processed
                              1,     // n_inputs
                              true,  // isFinalInferenceStep
                              shouldSkipInference)) {
      __ERROR("nsp-image-model: prepareAdaptorInputs failed");
      return 0;
    }

    // Skip inference if any adaptor requested it
    if (!shouldSkipInference) {
      for (auto& nsp_graph : m_nsp_graphs) {
        if (!nsp_graph.execute(
                0, -1, m_inference_count, graph_switching, lazy_lora, weight_shared_lora)) {
          return false;
        }
      }
    }

    // Call handleOutputs for all registered adaptors using base class helper
    if (!handleAdaptorOutputs(0,        // variant
                              -1,       // ctx_size
                              0,        // n_past
                              0,        // n_processed
                              1,        // n_inputs
                              true)) {  // isFinalInferenceStep
      __ERROR("nsp-image-model: handleAdaptorOutputs failed");
      return 0;
    }

    m_inference_count++;
  }
  uint32_t rank      = t_output_tensor->tensor->v1.rank;
  size_t numElements = 1;
  for (size_t i = 0; i < rank; i++) {
    numElements *= t_output_tensor->tensor->v1.dimensions[i];
  }

  // outputs.resize(numElements * t_image_features->dtype.bw());
  uint32_t bw            = t_output_tensor->dtype.bw();
  uint8_t* output_buffer = reinterpret_cast<uint8_t*>(getBuffer(t_output_tensor));
  outputs.insert(outputs.begin(), output_buffer, output_buffer + numElements * bw);

  __DEBUG("qnn-htp: run-inference complete : {} usec : wait {} exec {}",
          start.elapsed_usec(),
          total_wait,
          total_exec);

  return 1;
}

bool QnnNspImageModel::quantizeInput(float* in,
                                     const std::string& tensorName,
                                     size_t tensorOffset,
                                     size_t length) {
  if (t_input_tensors.count(tensorName) == 0 || t_input_tensors[tensorName] == nullptr) {
    __ERROR("Input Tensor {} not found during execute", tensorName);
    return false;
  }

  auto tensor       = t_input_tensors[tensorName];
  const auto scale  = tensor->quantParam[0].scale;
  const auto offset = tensor->quantParam[0].offset;
  // clang-format off
    switch (tensor->dtype) {
      case QNN_DATATYPE_UFIXED_POINT_8:
        QnnUtils::quantizeTensorPtr(in, reinterpret_cast<uint8_t*>(getBuffer(tensor)) + tensorOffset, offset, scale, length);
        break;
      case QNN_DATATYPE_UFIXED_POINT_16:
        QnnUtils::quantizeTensorPtr(in, reinterpret_cast<uint16_t*>(getBuffer(tensor)) + tensorOffset, offset, scale, length);
        break;
      default:
        __ERROR("Unsupported alpha tensor dtype {}", tensor->dtype.str());
        return false;
    }
    return true;
}

void QnnNspImageModel::getTensorDimensions(LayerType layerType,
                                   std::vector<uint32_t>& dimensions)
{
  if (layerType == LayerType::OUTPUT) {
    dimensions.push_back(t_output_tensor->dims.height);
    dimensions.push_back(t_output_tensor->dims.width);
    dimensions.push_back(t_output_tensor->dims.channel);
  }
}

void QnnNspImageModel::getTensorParam(LayerType layerType,
                              std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& bitwidth)
{
  if (layerType == LayerType::OUTPUT) {
    dataType = t_output_tensor->dtype.str();
    scale    = t_output_tensor->quantParam[0].scale;
    offset   = t_output_tensor->quantParam[0].offset;
    bitwidth = t_output_tensor->dtype.bw();
  }
}

size_t QnnNspImageModel::getEmbeddingBufferSize() {
    return 0;
}

void QnnNspImageModel::getInputTensorNames(std::unordered_set<std::string>& inputTensorNames) {
  inputTensorNames.clear();
  for (const auto& it : t_input_tensors) {
    inputTensorNames.emplace(it.first);
  }
}

void QnnNspImageModel::setHigherVariant(){
    return;
}

bool QnnNspImageModel::finalizeState(std::shared_ptr<EngineState>& engineState) {

  IOEVENT event = engineState->isInitialize() ? engineState->getIOBuffer()->m_event
                                            : IOEVENT::ALLOCATE_REGISTER_EVENT;
  __DEBUG("qnn-htp: Event triggered {}",ioEventMap[event]);
  if (event == IOEVENT::NO_EVENT) {
    return true;
  }

  if (true != QnnNspBaseModel::finalizeState(engineState)) {
    return false;
  }

  m_lazyInitialization = false;

  if (!initializeIOTensors()){
    __ERROR("Error in re-initializing the Tensors");
    return false;
  }

  if (event == IOEVENT::ALLOCATE_REGISTER_EVENT) {
    // reinitialize the Tensor Pointers to updated once
    if (true != initializeTensorPointers()) {
      __ERROR("Error in initializing Tensor pointers");
      return false;
    }
    engineState->initialize(std::dynamic_pointer_cast<IOBuffer>(m_ioTensor));

  } else if (event == IOEVENT::REGISTER_EVENT) {
    m_ioTensor = std::dynamic_pointer_cast<IOTensor>(engineState->getIOBuffer());
    // might need to update some static fields.
    if (true != initializeTensorPointers()) {
      __ERROR("Error in initializing Tensor pointers");
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

bool QnnNspImageModel::calculate_rope_embeddings(void) {
  if (!m_internal_pos_encoding) return true;
  if (m_positional_encoding.type != PositionalEncoding::ROPE) return true;

  // Calculate inv_freq array
  // int freq_dim = config.hidden_size / config.num_attention_heads / 2;
  // uint32_t freq_dim = t_position_ids_sin->dims.channel;
  uint32_t freq_dim = static_cast<uint32_t>(m_positional_encoding.rope_params.dims);
  std::vector<float> inv_freq(freq_dim / 2);
  const float theta = m_positional_encoding.rope_params.theta;
  for (uint32_t i = 0; i < freq_dim / 2; ++i) {
    inv_freq[i] = 1.0f / std::pow(theta, (2 * i) / static_cast<float>(freq_dim));
  }

  // seq_len = height * width;
  uint32_t seq_len = t_position_ids_sin->dims.width;
  std::vector<double> pos_id_sin(seq_len * freq_dim);
  std::vector<double> pos_id_cos(seq_len * freq_dim);

  if (m_positional_encoding.rope_params.rope_scaling.rope_type == RopeScalingParams::ROPE_QWEN2VL) {
    // Reference HuggingFace
    // https://github.com/huggingface/transformers/blob/ca402e2116f5917ce0a03659b779a02a555b285f/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L360

    // qwen2vl_params take reference from HuggingFace
    // https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json
    // spatial_merge_size default: 2
    const uint32_t spatial_merge_size =
                   m_positional_encoding.rope_params.rope_scaling.qwen2vl_params.spatial_merge_size;
    uint32_t height = m_vision_param.height;
    uint32_t width = m_vision_param.width;

    uint32_t spatial_h = height / spatial_merge_size;
    uint32_t spatial_w = width / spatial_merge_size;

    std::vector<uint32_t> hpos_ids(seq_len);
    std::vector<uint32_t> wpos_ids(seq_len);
    for (uint32_t s_h = 0; s_h < spatial_h; ++s_h) {
      for (uint32_t s_w = 0; s_w < spatial_w; ++s_w) {
        for (uint32_t i1 = 0; i1 < spatial_merge_size; ++i1) {
          for (uint32_t i2 = 0; i2 < spatial_merge_size; ++i2) {
            size_t i = s_h * spatial_w * spatial_merge_size * spatial_merge_size
                    + s_w * spatial_merge_size * spatial_merge_size
                    + i1 * spatial_merge_size
                    + i2;
            hpos_ids[i] = i1 + s_h * spatial_merge_size;
            wpos_ids[i] = i2 + s_w * spatial_merge_size;
          }
        }
      }
    }

    // stack hpos_ids, wpos_ids with dim -1
    std::vector<std::vector<uint32_t>> pos_ids(seq_len, std::vector<uint32_t>(2));
    for (size_t i = 0; i < pos_ids.size(); ++i) {
      pos_ids[i][0] = hpos_ids[i];
      pos_ids[i][1] = wpos_ids[i];
    }

    std::vector<float> seq(std::max(height, width));
    for (size_t i = 0; i < seq.size(); ++i) {
      seq[i] = static_cast<float>(i);
    }

    // outer seq, inv_freq
    std::vector<std::vector<float>> freqs(seq.size(), std::vector<float>(inv_freq.size()));
    for (size_t i1 = 0; i1 < seq.size(); ++i1) {
      for (size_t i2 = 0; i2 < inv_freq.size(); ++i2) {
        freqs[i1][i2] = seq[i1] * inv_freq[i2];
      }
    }

    std::vector<std::vector<float>> rotary_pos(seq_len, std::vector<float>(freq_dim));
    for (size_t i = 0; i < pos_ids.size(); ++i) {
      auto pos_id = pos_ids[i][0];
      std::copy(freqs[pos_id].begin(), freqs[pos_id].end(), rotary_pos[i].begin());
      pos_id = pos_ids[i][1];
      std::copy(freqs[pos_id].begin(), freqs[pos_id].end(),
                rotary_pos[i].begin() + static_cast<uint32_t>(freqs[pos_id].size()));
    }

    if (m_layerNames.find(LayerType::FULL_ATTN_MASK) != m_layerNames.end() &&
        m_layerNames.find(LayerType::WINDOW_ATTN_MASK) != m_layerNames.end()) {
      // if detect FULL_ATTN_MASK and WINDOW_ATTN_MASK, we can confirm it's vit of qwen2.5-vl
      // currently image encoder only process single frame, temporal = 1
      const uint32_t temporal = 1;

      // set full_attention_mask and window_attention_mask
      std::vector<uint32_t> window_index;
      std::vector<uint32_t> window_seq_lens;
      cal_window_index({temporal, height, width}, window_index, window_seq_lens);
      cal_attention_mask(temporal, seq_len, window_seq_lens);

      //reset posidion_id_sin and posidion_id_cos with window_index
      uint32_t spatial_merge_unit = spatial_merge_size * spatial_merge_size;
      std::vector<std::vector<float>> rotary_pos_reset(seq_len, std::vector<float>(freq_dim));
      for (size_t i = 0; i < window_index.size(); ++i) {
        uint32_t window_id = window_index[i];
        uint32_t src_start = window_id * spatial_merge_unit;
        uint32_t dst_start = i * spatial_merge_unit;
        for (uint32_t j = 0; j < spatial_merge_unit; ++j) {
          std::copy(rotary_pos[src_start + j].begin(), rotary_pos[src_start + j].end(),
                    rotary_pos_reset[dst_start + j].begin());
        }
      }

      for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < freq_dim; ++j) {
          pos_id_sin[j + i * freq_dim] = std::sin(static_cast<double>(rotary_pos_reset[i][j]));
          pos_id_cos[j + i * freq_dim] = std::cos(static_cast<double>(rotary_pos_reset[i][j]));
        }
      }
    }
    else {
      for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < freq_dim; ++j) {
          pos_id_sin[j + i * freq_dim] = std::sin(static_cast<double>(rotary_pos[i][j]));
          pos_id_cos[j + i * freq_dim] = std::cos(static_cast<double>(rotary_pos[i][j]));
        }
      }
    }
  }

  auto [q_scale, q_offset] = t_position_ids_cos->quantParam[0];
  if (d_pos == QNN_DATATYPE_FLOAT_16 || d_pos == QNN_DATATYPE_FLOAT_32) {
    // If floating point, don't quantize!
    q_scale  = 1.0;
    q_offset = 0;
  }
  void* rope_sin = getBuffer(t_position_ids_sin);
  void* rope_cos = getBuffer(t_position_ids_cos);

  for (uint32_t i = 0; i < seq_len; ++i) {
    for (uint32_t j = 0; j < freq_dim; ++j) {
      const double sin_val = pos_id_sin[i * freq_dim + j] / q_scale - q_offset;
      const double cos_val = pos_id_cos[i * freq_dim + j] / q_scale - q_offset;

      switch (d_pos) {
        case QNN_DATATYPE_UFIXED_POINT_8:
          (reinterpret_cast<uint8_t*>(rope_sin))[i * freq_dim + j] = static_cast<uint8_t>(sin_val);
          (reinterpret_cast<uint8_t*>(rope_cos))[i * freq_dim + j] = static_cast<uint8_t>(cos_val);
          break;
        case QNN_DATATYPE_UFIXED_POINT_16:
          (reinterpret_cast<uint16_t*>(rope_sin))[i * freq_dim + j] =
              static_cast<uint16_t>(sin_val);
          (reinterpret_cast<uint16_t*>(rope_cos))[i * freq_dim + j] =
              static_cast<uint16_t>(cos_val);
          break;
        case QNN_DATATYPE_FLOAT_16:
          (reinterpret_cast<uint16_t*>(rope_sin))[i * freq_dim + j] =
              fp16_ieee_from_fp32_value(sin_val);
          (reinterpret_cast<uint16_t*>(rope_cos))[i * freq_dim + j] =
              fp16_ieee_from_fp32_value(cos_val);
          break;
        case QNN_DATATYPE_FLOAT_32:
          (reinterpret_cast<float*>(rope_sin))[i * freq_dim + j] = static_cast<float>(sin_val);
          (reinterpret_cast<float*>(rope_cos))[i * freq_dim + j] = static_cast<float>(cos_val);
          break;
        default:
          __ERROR("Unsupported position ids datatype {}", d_pos.str());
          return false;
      }
    }
  }
  return true;
}

void QnnNspImageModel::cal_attention_mask(const uint32_t temporal,
                                          const uint32_t seq_len,
                                          const std::vector<uint32_t>& window_seq_lens) {
#define ATTN_MASK_PAD -1000
  std::vector<uint32_t> full_seq_lens(temporal + 1, 0);
  for (size_t t = 1; t < full_seq_lens.size(); ++t) {
    full_seq_lens[t] =  full_seq_lens[t - 1] + seq_len;
  }

  std::vector<std::vector<float>> full_attention_mask(seq_len,
                                                      std::vector<float>(seq_len, ATTN_MASK_PAD));
  for (size_t i = 1; i < full_seq_lens.size(); ++i) {
    for (size_t j = full_seq_lens[i - 1]; j < full_seq_lens[i]; ++j) {
      auto first = &full_attention_mask[j][full_seq_lens[i - 1]];
      auto last = &full_attention_mask[j][full_seq_lens[i]];
      std::fill(first, last, 0);
    }
  }

  std::vector<std::vector<float>> window_attention_mask(seq_len,
                                                        std::vector<float>(seq_len, ATTN_MASK_PAD));
  for (size_t i = 1; i < window_seq_lens.size(); ++i) {
    for (size_t j = window_seq_lens[i - 1]; j < window_seq_lens[i]; ++j) {
      auto first = &window_attention_mask[j][window_seq_lens[i - 1]];
      auto last = &window_attention_mask[j][window_seq_lens[i]];
      std::fill(first, last, 0);
    }
  }

  auto [q_scale_full, q_offset_full] = t_full_attention_mask->quantParam[0];
  auto [q_scale_window, q_offset_window] = t_window_attention_mask->quantParam[0];
  if (t_full_attention_mask->dtype == QNN_DATATYPE_FLOAT_16 ||
      t_full_attention_mask->dtype == QNN_DATATYPE_FLOAT_32) {
    // If floating point, don't quantize!
    q_scale_full  = 1.0;
    q_offset_full = 0;
  }
  if (t_window_attention_mask->dtype == QNN_DATATYPE_FLOAT_16 ||
      t_window_attention_mask->dtype == QNN_DATATYPE_FLOAT_32) {
    // If floating point, don't quantize!
    q_scale_window  = 1.0;
    q_offset_window = 0;
  }
  void* fullAttn = getBuffer(t_full_attention_mask);
  void* windowAttn = getBuffer(t_window_attention_mask);

  for (uint32_t i = 0; i < seq_len; ++i) {
    for (uint32_t j = 0; j < seq_len; ++j) {
      const size_t attnIdx = i * seq_len + j;

      const double fullAttnVal   =
          static_cast<double>(full_attention_mask[i][j])/q_scale_full - q_offset_full;
      const double windowAttnVal =
          static_cast<double>(window_attention_mask[i][j])/q_scale_window - q_offset_window;
      switch (d_pos) {
        case QNN_DATATYPE_UFIXED_POINT_8:
          (reinterpret_cast<uint8_t*>(fullAttn))[attnIdx] = static_cast<uint8_t>(fullAttnVal);
          (reinterpret_cast<uint8_t*>(windowAttn))[attnIdx] = static_cast<uint8_t>(windowAttnVal);
          break;
        case QNN_DATATYPE_UFIXED_POINT_16:
          (reinterpret_cast<uint16_t*>(fullAttn))[attnIdx] = static_cast<uint16_t>(fullAttnVal);
          (reinterpret_cast<uint16_t*>(windowAttn))[attnIdx] = static_cast<uint16_t>(windowAttnVal);
          break;
        case QNN_DATATYPE_FLOAT_16:
          (reinterpret_cast<uint16_t*>(fullAttn))[attnIdx] = fp16_ieee_from_fp32_value(fullAttnVal);
          (reinterpret_cast<uint16_t*>(windowAttn))[attnIdx] = fp16_ieee_from_fp32_value(windowAttnVal);
          break;
        case QNN_DATATYPE_FLOAT_32:
          (reinterpret_cast<float*>(fullAttn))[attnIdx] = static_cast<float>(fullAttnVal);
          (reinterpret_cast<float*>(windowAttn))[attnIdx] = static_cast<float>(windowAttnVal);
          break;
        default:
          __ERROR("Unsupported datatype: {}: {}, {}: {}",
              m_layerNames[LayerType::FULL_ATTN_MASK], t_full_attention_mask->dtype.str(),
              m_layerNames[LayerType::WINDOW_ATTN_MASK], t_window_attention_mask->dtype.str());
          return;
      }
    }
  }
}

void QnnNspImageModel::cal_window_index(const std::vector<uint32_t>& grid_thw,
                                        std::vector<uint32_t>& window_index,
                                        std::vector<uint32_t>& window_seq_lens) {
  // Reference HuggingFace
  // https://github.com/huggingface/transformers/blob/ca402e2116f5917ce0a03659b779a02a555b285f/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L389
#define WINDOW_INDEX_PAD -100
  uint32_t temporal = grid_thw[0];
  uint32_t height = grid_thw[1];
  uint32_t width = grid_thw[2];

  // window_size default: 112
  uint32_t window_size = m_positional_encoding.rope_params.rope_scaling.qwen2vl_params.window_size;
  // patch_size default: 14
  uint32_t patch_size = m_positional_encoding.rope_params.rope_scaling.qwen2vl_params.patch_size;
  // spatial_merge_size default: 2
  uint32_t spatial_merge_size =
                  m_positional_encoding.rope_params.rope_scaling.qwen2vl_params.spatial_merge_size;

  uint32_t spatial_merge_unit = spatial_merge_size * spatial_merge_size;
  uint32_t vit_merger_window_size = window_size / spatial_merge_size / patch_size;
  uint32_t llm_grid_h = height / spatial_merge_size;
  uint32_t llm_grid_w = width / spatial_merge_size;
  uint32_t pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
  uint32_t pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;
  uint32_t grid_h_padded = llm_grid_h + pad_h;
  uint32_t grid_w_padded = llm_grid_w + pad_w;
  uint32_t num_windows_h = grid_h_padded / vit_merger_window_size;
  uint32_t num_windows_w = grid_w_padded / vit_merger_window_size;

  std::vector<std::vector<std::vector<int32_t>>> indices(temporal,
                                              std::vector<std::vector<int32_t>>(llm_grid_h,
                                              std::vector<int32_t>(llm_grid_w, WINDOW_INDEX_PAD)));
  int32_t index = 0;
  for (size_t t = 0; t < temporal; ++t) {
    for (size_t h = 0; h < llm_grid_h; ++h) {
      for (size_t w = 0; w < llm_grid_w; ++w) {
        indices[t][h][w] = index++;
      }
    }
  }

  std::vector<std::vector<std::vector<int32_t>>> index_padded(temporal,
                              std::vector<std::vector<int32_t>>(num_windows_h * num_windows_w,
                              std::vector<int32_t>(vit_merger_window_size * vit_merger_window_size,
                              WINDOW_INDEX_PAD)));
  for (size_t t = 0; t < temporal; ++t) {
    for (size_t h = 0; h < llm_grid_h; ++h) {
      for (size_t w = 0; w < llm_grid_w; ++w) {
        size_t hw = (h / vit_merger_window_size) * num_windows_h
                  + w / vit_merger_window_size;
        size_t i = (h % vit_merger_window_size) * vit_merger_window_size
                 + w % vit_merger_window_size;
        index_padded[t][hw][i] = indices[t][h][w];
      }
    }
  }

  std::vector<uint32_t> seqlens(temporal * num_windows_h * num_windows_w, 0);
  for (size_t t = 0; t < index_padded.size(); ++t) {
    for (size_t hw = 0; hw < index_padded[t].size(); ++hw) {
      for (size_t i = 0; i < index_padded[t][hw].size(); ++i) {
        size_t seq_idx = t * (num_windows_h * num_windows_w) + hw;
        seqlens[seq_idx] += static_cast<unsigned int>(index_padded[t][hw][i] != WINDOW_INDEX_PAD ? 1 : 0);
      }
    }
  }

  std::vector<uint32_t> cu_window_seqlens(seqlens.size() + 1, 0);
  for (size_t i = 1; i < cu_window_seqlens.size(); ++i) {
    cu_window_seqlens[i] = cu_window_seqlens[i - 1] + seqlens[i - 1] * spatial_merge_unit;
  }

  // set window_seq_lens
  window_seq_lens = std::vector<uint32_t>(1, cu_window_seqlens[0]);
  uint32_t cur_val = cu_window_seqlens[0];
  for (size_t i = 1; i < cu_window_seqlens.size(); ++i) {
    if (cu_window_seqlens[i] != cur_val) {
      cur_val = cu_window_seqlens[i];
      window_seq_lens.push_back(cur_val);
    }
  }

  window_index = std::vector<uint32_t>(temporal * llm_grid_h * llm_grid_w, 0);
  size_t p = 0;
  for (size_t t = 0; t < index_padded.size(); ++t) {
    for (size_t hw = 0; hw < index_padded[t].size(); ++hw) {
      for (size_t i = 0; i < index_padded[t][hw].size(); ++i) {
        if (index_padded[t][hw][i] != WINDOW_INDEX_PAD) {
          window_index[p++] = static_cast<uint32_t>(index_padded[t][hw][i]);
        }
      }
    }
  }
}

void QnnNspImageModel::prepareAttentionMask() {
  if (m_mask_type == "") {
    return; // no mask type specified, skip preparation
  }

  if (m_mask_type == "aspect-ratio") {
    uint16_t* aspect_ratio_mask_buffer  = reinterpret_cast<uint16_t*>(getBuffer(t_full_attention_mask));

    size_t num_patches = static_cast<size_t>(t_output_tensor->dims.width);
    size_t pad_patches = static_cast<size_t>(t_full_attention_mask->dims.width) - num_patches;
    size_t size_x = (getBufferSize(t_full_attention_mask)/sizeof(uint16_t))/(num_patches + pad_patches);
    size_t size_y = (getBufferSize(t_full_attention_mask)/sizeof(uint16_t))/size_x;
    uint16_t max_val = std::numeric_limits<uint16_t>::max(), min_val = 0;

    for(size_t i = 0; i < (size_y - pad_patches) * size_x; ++i) aspect_ratio_mask_buffer[i] = max_val;

    for(size_t y = size_y - pad_patches; y < size_y; ++y) {
      for(size_t x = 0; x < size_x - pad_patches; ++x) aspect_ratio_mask_buffer[(y*size_x) + x] = max_val;
      for(size_t x = size_x - pad_patches; x < size_x; ++x) aspect_ratio_mask_buffer[(y*size_x) + x] = min_val;
    }
  } else {
    __WARN("Unsupported mask type {}. Skipping attention mask preparation.", m_mask_type);
  }
}

void QnnNspImageModel::resetState(){
  if(m_qnnApi){
    m_qnnApi->resetAdapterGraphHandleState();
  }
}
} // namespace qualla
