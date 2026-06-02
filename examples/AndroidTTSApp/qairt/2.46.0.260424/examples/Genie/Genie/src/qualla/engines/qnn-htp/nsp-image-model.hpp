//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <atomic>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "kvmanager.hpp"
#include "nsp-base-model.hpp"
#include "nsp-graph.hpp"
#include "qnn-utils.hpp"
#include "qualla/detail/threadpool.hpp"
#include "qualla/env.hpp"

namespace qualla {

class QnnNspImageModel : public QnnNspBaseModel {
 protected:
  // Maps tensor name to allocation block index and block offset
  std::unordered_map<std::string, std::pair<uint64_t, size_t>> tensor_alloc_info;

  std::string embedding_datatype{"QNN_DATATYPE_FLOAT_32"};

  // Maps layers to their tensor names.
  std::map<LayerType, std::string> m_layerNames{{LayerType::INPUT, "pixel_values"},
                                                {LayerType::OUTPUT, "image_features"}};
  inline bool updateTensorPointer(GraphVariant& variant, std::string& key, QnnUtils::Tensor*& t);

  void cal_window_index(const std::vector<uint32_t>& grid_thw,
                        std::vector<uint32_t>& window_index,
                        std::vector<uint32_t>& window_seq_lens);
  void cal_attention_mask(const uint32_t temporal,
                          const uint32_t seq_len,
                          const std::vector<uint32_t>& window_seq_lens);

 public:
  std::vector<std::string> model_filelist;
  std::vector<std::string> exec_select_graphs;
  bool load_select_graphs;

  // Model parameters
  ModelArchitectureType m_modelArchitectureType;
  std::string m_mask_type;

  std::unordered_map<std::string, QnnUtils::DataType> d_inputs;
  QnnUtils::DataType d_output{QNN_DATATYPE_INT_32};

  // Model specific variables
  uint32_t m_num_graphs;

  // Store some pointers for easier access
  std::unordered_map<std::string, QnnUtils::Tensor*> t_input_tensors;
  QnnUtils::Tensor* t_output_tensor{nullptr};

  bool m_internal_pos_encoding{false};
  QnnUtils::Tensor* t_full_attention_mask{nullptr};
  QnnUtils::Tensor* t_window_attention_mask{nullptr};

  VisionParam m_vision_param;

  QnnNspImageModel(std::shared_ptr<Env> env, const QnnNspBaseModel::Params& params);

  ~QnnNspImageModel();

  bool setupInputTensors(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs);

  bool setupInputFP16(const std::vector<uint8_t>& inputs, const std::string& name);

  template <typename DType>
  bool setupInput(const std::vector<uint8_t>& inputs, const std::string& name);

  bool quantizeInput(float* in, const std::string& tensorName, size_t tensorOffset, size_t length);

  bool initializeModel(void) override;
  bool validateModel(void) override;
  bool initializeIOTensors(void) override;
  bool initializeTensorPointers() override;
  bool calculate_rope_embeddings(void) override;

  size_t getEmbeddingBufferSize() override;

  bool finalizeState(std::shared_ptr<EngineState>& engineState) override;

  void resetState() override;

  void getTensorDimensions(LayerType layerType, std::vector<uint32_t>& dimensions) override;

  void getTensorParam(LayerType layerType,
                      std::string& dataType,
                      double& scale,
                      int32_t& offset,
                      size_t& bitWidth) override;

  void getInputTensorNames(std::unordered_set<std::string>& inputTensorNames) override;

  size_t runInference(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
                      std::vector<uint8_t>& outputs) override;

  void setHigherVariant() override;
  void prepareAttentionMask();
  virtual const char* getTraceNamespace() const override { return "QnnNspImageModel"; }
};

}  // namespace qualla
