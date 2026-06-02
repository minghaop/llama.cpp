//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "Traceable.hpp"
#include "qnn-utils.hpp"
#include "qualla/detail/tensor.hpp"
#include "qualla/env.hpp"

namespace qualla {

inline std::string getGraphTypeStr(GraphType type) {
  switch (type) {
    case GraphType::NONE:
      return "NONE";
    case GraphType::DEFAULT:
      return "DEFAULT";
    case GraphType::LUT:
      return "LUT";
    case GraphType::DECODER:
      return "DECODER";
    case GraphType::DECODER_PREFILL:
      return "DECODER_PREFILL";
    case GraphType::LMHEAD:
      return "LMHEAD";
    case GraphType::IMAGE_ENCODER:
      return "IMAGE_ENCODER";
    default:
      return "ERROR: GraphType not found";
  }
}

struct GraphVariant : public genie::profiling::Traceable {
  int32_t n_tokens;
  int32_t ctx_size{-1};
  std::string graph_name;

  // Graph Type
  GraphType variantType{GraphType::NONE};
  // QNN API specific variables
  qnn_wrapper_api::GraphInfo_t* graph_info;
  // Qnn_ContextHandle_t context_handle;

  QnnUtils::TensorMap input_specs;
  QnnUtils::TensorMap output_specs;

  std::map<LayerType, std::string>& m_layerNames;

  std::shared_ptr<Env> _env;
  uint32_t m_batchSize{1};

  GraphVariant() = delete;
  GraphVariant(qnn_wrapper_api::GraphInfo_t* g_info,
               std::map<LayerType, std::string>& layerNames,
               std::shared_ptr<Env> env,
               const std::unordered_set<std::string>& cacheGroupPrefixes = {},
               const std::string& defaultGroup                           = "past_",
               uint32_t batchSize                                        = 1);

  QnnUtils::Tensor* getTensor(const std::string& tensor_name) {
    QnnUtils::Tensor* ret = getInput(tensor_name);
    return (ret != nullptr) ? ret : getOutput(tensor_name);
  }
  QnnUtils::Tensor* getInput(const std::string& tensor_name) {
    return input_specs.contains(tensor_name) ? &input_specs.at(tensor_name) : nullptr;
  }
  QnnUtils::Tensor* getOutput(const std::string& tensor_name) {
    return output_specs.contains(tensor_name) ? &output_specs.at(tensor_name) : nullptr;
  }

  bool refreshTensorQuantParams();

  virtual const char* getTraceNamespace() const override { return "GraphVariant"; }

 private:
  int32_t determineGraphInputSize(const std::string& defaultGroup);
  int32_t determineGraphContextSize(const std::string& defaultGroup);
  GraphType determineGraphType(const std::unordered_set<std::string>& cacheGroupPrefixes);
};

/**
 * The idea behind QnnNspGraph is to represent "common" graphs
 * For instance, both BERT-mode and KV$-mode are the same graph with different input sizes
 * QnnNspGraph will contain and manage both BERT-split-n and KV$mode-split-n
 * I/O tensors are mostly shared between these graphs, and can be managed collectively
 */
class QnnNspGraph : public genie::profiling::Traceable {
 private:
  int _idx;
  std::shared_ptr<Env> _env;

  // Useful pointers for graph execution (managed by NSPModel)
  QnnApi* g_qnn_api;

  int32_t run_wait_time, run_exec_time;  // Add more stats into a struct

  // Debug mode settings
  bool _debug_specs{false};
  bool _debug_tensors{false};
  std::string _debug_path;

 public:
  int32_t _counter{-1};
  std::shared_ptr<IOTensor> g_buffer_mgr;

  // TODO: Remove this reference
  std::unordered_map<std::string, std::pair<uint64_t, size_t>>* tensor_alloc_info;

  // Keys represent input_id size (1<=input_size<=ctx_size)
  // Values are graph description for that input_id size
  std::map<std::pair<int32_t, int32_t>, GraphVariant*> variants;

  // Graph Type
  GraphType m_graphType{GraphType::NONE};

  QnnNspGraph(int idx,
              std::shared_ptr<Env> env,
              QnnApi* qnnApi,
              std::shared_ptr<IOTensor> ioTensor);

  ~QnnNspGraph();

  int idx() { return _idx; }
  bool addGraph(GraphVariant* graph_spec);
  void printAvailableConfigs();

  // Overload the () operator to access a [variant, ctx_size (or -1 for global match)]
  GraphVariant* operator()(const int32_t variant, const int32_t ctx_size) {
    return variants.contains({variant, ctx_size}) ? variants.at({variant, ctx_size})
                                                  : variants.at({variant, -1});
  }

  bool execute(const int32_t n_tokens,
               const int32_t ctx_size,
               const int n_inference,
               bool graphSwitch,
               std::string& lazyLora,
               bool weightSharedLora);

  void setDebugMode(bool debug_specs, bool debug_tensors, std::string debug_path) {
    _debug_path    = debug_path;
    _debug_specs   = debug_specs;
    _debug_tensors = debug_tensors;
  }
  void dumpTensors(GraphVariant* const variant, bool mode, int n_inference) const;

  virtual const char* getTraceNamespace() const override { return "GraphVariant"; }
};

}  // namespace qualla
