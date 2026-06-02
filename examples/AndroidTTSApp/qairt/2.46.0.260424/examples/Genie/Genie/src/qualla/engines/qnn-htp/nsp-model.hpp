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

#include "QnnApi.hpp"
#include "attention-mask.hpp"
#include "buffer/IOTensor.hpp"
#include "kvmanager.hpp"
#include "nsp-base-model.hpp"
#include "nsp-graph.hpp"
#include "qnn-utils.hpp"
#include "qualla/detail/tensor.hpp"
#include "qualla/detail/threadpool.hpp"
#include "qualla/env.hpp"
#include "tokenEmbedIO.hpp"

namespace qualla {

class QnnNspModel : public QnnNspBaseModel {
 protected:
  // Populated by allocateTensors()
  // Maps tensor name to allocation block index and block offset
  std::unordered_map<std::string, std::pair<uint64_t, size_t>> tensor_alloc_info;

  int32_t input_width     = 1;
  int32_t input_channel   = 1;
  uint32_t input_bitwidth = 4;

  int32_t embedding_length = -1;
  std::string embedding_datatype{"float32"};
  // Maps layers to their tensor names.
  std::map<LayerType, std::string> m_layerNames{
      {LayerType::INPUT, "input_ids"},
      {LayerType::OUTPUT, "logits"},
      {LayerType::TOKEN_TYPE_IDS, "token_type_ids"},
      {LayerType::POOL_OUTPUT, "pooled_output"},
      {LayerType::SEQ_OUTPUT, "sequence_output"},
      {LayerType::ATTN_MASK, "attention_mask"},
      {LayerType::POS_SIN, "position_ids_sin"},
      {LayerType::POS_COS, "position_ids_cos"},
      {LayerType::POS_IDS, "position_ids"},
      {LayerType::ANCHOR, "anchor_buffer"},
      {LayerType::CACHE_INDEX, "cache_index"},
      {LayerType::INPUT_EMBED, "inputs_embeds"},
      {LayerType::VALID_MASK, "valid_token_mask"},
      {LayerType::CROSS_ATTN_STATES, "encoder_hidden_states"},
      {LayerType::CROSS_ATTN_MASK, "encoder_attention_mask"},
      {LayerType::FULL_TEXT_ROW_MASK, "full_text_row_masked_out_mask"},
      {LayerType::EAGLET_DM_HIDDEN_STATES_IN, "hidden_states"},
      {LayerType::EAGLET_DM_HIDDEN_STATES_OUT, "last_hidden_states"},
      {LayerType::EAGLET_TM_HIDDEN_STATES_IN, "target_hidden_states"},
  };

  std::vector<uint8_t> m_eosEmbedding;

  std::vector<float> m_outputLogit;

  Qnn_TensorDataFormat_t m_expectedDataFormat{QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER};

  bool m_pause{false};

  std::vector<uint32_t> m_visionParam;

  // KeyDiff anchor update weight
  bool m_useAnchorAlpha{false};
  double m_anchorAlpha{0.02};

  // On the first inference, we don't apply alpha to anchor updates.
  void resetAnchorAlpha() { m_useAnchorAlpha = false; }

  // External Input params
  ExternalBufferType m_externalBufferType{BUFFER_UNDEFINED};
  DataFillPolicy m_dataFillPolicy{POLICY_UNDEFINED};

 public:
  std::vector<std::string> model_filelist;
  std::string lmhead_weight_dir;
  std::string m_mask_type;
  bool m_cross_attention{false};
  size_t m_cross_attention_input_size{0};
  bool token_history_enabled{true};
  size_t m_maxChunkedInputLength{0};
  std::vector<int32_t> token_history;
  std::map<int32_t, int32_t> variant_latency;
  std::vector<std::string> exec_select_graphs;
  bool load_select_graphs;

  // Model parameters
  ModelArchitectureType m_modelArchitectureType;
  size_t m_ctx_size{0};
  size_t m_vocab_size{0};
  size_t m_embd_size{0};
  int32_t m_kv_dim{-1};
  int32_t m_pad_token{-1};
  int32_t m_img_token{-1};

  size_t m_embeddingBufferSize{0};

  // threadpool
  std::shared_ptr<ThreadPool> m_threadpool;

  QnnUtils::DataType d_input{QNN_DATATYPE_INT_32}, d_attn_map{QNN_DATATYPE_UFIXED_POINT_16},
      d_token_type{QNN_DATATYPE_INT_32};

  // Information regarding model execution settings and last inference
  struct RunInfo {
    int32_t n_tokens;
    size_t n_processed;

    std::vector<int32_t> tokens;
  } run_info{-1, 0, {}};

  // Model specific variables
  uint32_t m_num_graphs;
  bool _lmhead_weight_input{false};

  bool _threaded{false};
  uint64_t _cpumask{0};
  bool m_ropeInitialized{false};
  bool m_ropeBufferReady{false};

  KVManagerMode _kv_update_method{KVManagerMode::SMART_MASK};
  bool m_kv_use_scatter{false};
  std::shared_ptr<KVManager> m_kvmanager;

  // Store some pointers for easier access
  QnnUtils::Tensor* t_input_ids{nullptr};
  QnnUtils::Tensor* t_cache_index{nullptr};
  QnnUtils::Tensor* t_attn_mask{nullptr};
  QnnUtils::Tensor* t_token_type_ids{nullptr};
  QnnUtils::Tensor* t_valid_mask{nullptr};
  QnnUtils::Tensor* t_cross_attn_states{nullptr};
  QnnUtils::Tensor* t_cross_attn_mask{nullptr};

  // used for mllama
  QnnUtils::Tensor* t_full_text_row_mask{nullptr};

  // Tensor pointers for Eagle features
  QnnUtils::Tensor* t_dm_draft_feature_in{nullptr};   // DM input with DM hidden state
  QnnUtils::Tensor* t_dm_target_feature_in{nullptr};  // DM input with TM hidden state
  QnnUtils::Tensor* t_dm_feature_out{nullptr};        // DM output hidden_state

  // Variables for attention mask
  union {
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
  } m_attention_positive_value, m_attention_negative_value;

  // PositionalEncodingType::ABSOLUTE OR PositionalEncodingType::ALIBI
  QnnUtils::Tensor* t_position_ids{nullptr};
  // PositionalEncodingType::ROPE variables
  uint32_t m_pos_dim{0};    // Dimension of positional embedding tensor (incl partial_factor)
  void* rope_sin{nullptr};  // Pre-calculated RoPE sin table of size [ctx_size, m_pos_dim]
  void* rope_cos{nullptr};  // Pre-calculated RoPE cos table of size [ctx_size, m_pos_dim]

  // Variables for CacheGroups
  std::string m_default_group{"past_"};
  std::unordered_set<std::string> m_cache_group_prefixes;
  CacheGroupParamsMap m_cache_group_params_map;
  std::map<std::string, bool> m_cache_group_use_scatter;
  std::map<std::string, size_t> m_cache_group_ctx_size;
  std::unordered_map<std::string, QnnUtils::Tensor*> m_group_attn_mask;
  std::unordered_map<std::string, QnnUtils::Tensor*> m_group_cache_index;
  std::map<std::string, std::map<VariantSpec, VariantSpec>> m_cache_group_variant_map;

  // Variables for batch queries
  size_t m_batch_size{1};  // Model batch size
  std::vector<size_t> m_batch_prompt_token_lens{};
  size_t m_max_batch_prompt_token_len;
  size_t m_n_queryBatch;  // Actual number of queries from user

  QnnNspModel(std::shared_ptr<Env> env, const Params& params);

  ~QnnNspModel();

  bool initializeModel(void) override;

  bool validateModel(void) override;

  bool initializeIOTensors(void) override;

  bool initializeTensorPointers() override;

  void initializeUnconnectedInputs() override;

  bool initializeKVManager() override;

  bool calculate_rope_embeddings(void) override;

  bool load_lmhead_weight_as_input(void) override;

  bool analyzeCacheGroupKV();

  void getInputQuantParam(double& scale, int& offset) override;

  size_t getIOBufferByName(std::string tensor_name, void*& buffer, bool isPrompt) override;

  bool clearBuffer(QnnUtils::Tensor* tensor);

  bool setupInput(const InferenceStep& step,
                  size_t start,
                  const std::vector<int32_t>& tokens,
                  std::vector<uint8_t>& embeddings,
                  const uint8_t* featureVector,
                  const std::vector<int32_t>& selected,
                  const size_t eagletStartIdxOffset,
                  const bool post_update,
                  AttentionMask& attention_mask,
                  bool prefilledExecution = false);

  bool setupInput(const InferenceStep& step,
                  uint32_t start,
                  std::vector<int32_t>& tokens,
                  std::vector<size_t>& tokenNumPerBatch,
                  AttentionMask& attention_mask);

  bool setupInputEmbeddings(const InferenceStep& step,
                            size_t start,
                            size_t eagletStartIdxOffset,
                            bool isEagletDMMergedInput,
                            std::vector<uint8_t>& embeddings);

  bool setupEagletFeatureVectors(const InferenceStep& step,
                                 const size_t start,
                                 const size_t featureCount,
                                 const uint8_t* featureVector,
                                 const std::vector<int32_t>& selected,
                                 const size_t eagletStartIdxOffset,
                                 bool post_update);

  bool setupInputTokensEmbedding(const InferenceStep& step,
                                 const size_t start,
                                 const std::vector<int32_t>& tokens,
                                 std::vector<uint8_t>& embeddings,
                                 const uint8_t* featureVector,
                                 const std::vector<int32_t>& selected,
                                 const size_t eagletStartIdxOffset,
                                 const bool post_update);

  template <typename DType>
  void setupCrossAttentionMask(const InferenceStep& step,
                               const std::vector<int32_t>& tokens,
                               uint32_t cross_attention_len);

  bool setCrossAttentionHiddenStates(const std::vector<float>& crossAttentionStates) override;

  template <typename DType>
  void setupAttentionMask(const InferenceStep& step, AttentionMask& attention_mask);

  template <typename DType>
  void setupAttentionMask(const InferenceStep& step,
                          AttentionMask& attention_mask,
                          std::vector<size_t>& tokenNumPerBatch);

  template <typename DType>
  bool setupAlibiPositionEmbedding(const InferenceStep& step);

  bool quantizeInput(float* in, size_t tensorOffset, size_t length);

  size_t getEmbeddingBufferSize() override;

  size_t runInference(const std::vector<int32_t>& tokens,
                      std::vector<uint8_t>& embeddings,
                      const uint16_t* featureVector,
                      const std::vector<int32_t>& selected,
                      uint32_t start_idx,
                      bool post_update,
                      const std::vector<int32_t>& attention_map,
                      std::vector<float>& output,
                      bool output_all = false) override;

  size_t runInference(const std::vector<int32_t>& tokens,
                      std::vector<uint8_t>& embeddings,
                      const uint16_t* featureVector,
                      const std::vector<int32_t>& selected,
                      uint32_t start_idx,
                      bool post_update,
                      const std::vector<int32_t>& attention_map,
                      Tensor& output,
                      bool output_all = false) override;

  size_t runPrefillInputInference(const nlohmann::json executionConfig = {}) override;
  size_t runInference(std::vector<int32_t>& tokens,
                      std::vector<uint8_t>& embeddings,
                      const std::vector<int32_t>& attention_map,
                      std::vector<size_t>& tokenNumPerBatch,
                      std::vector<Tensor>& output,
                      bool output_all = false) override;

  bool cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding) override;

  bool setKVCacheNPast(size_t n_past, const std::vector<bool>& selected) override;

  size_t getEmbeddings(std::span<float> embds, InferenceStep& step);

  size_t getEmbeddings(Tensor& embds, InferenceStep& step);

  size_t setInputBufferData(LayerType inputLayer,
                            void* data,
                            size_t dataSize,
                            nlohmann::json& dataConfig) override;

  bool getOutputBufferData(LayerType outputLayer,
                           const nlohmann::json& outputConfig,
                           Engine::OutputCallback callback) override;

  void allocateOutputBuffer(uint32_t nLogits);
  void getTensorParam(LayerType layerType,
                      std::string& dataType,
                      double& scale,
                      int32_t& offset,
                      size_t& bitWidth) override;

  size_t getDequantLogits(std::span<float> buffer, InferenceStep& step, int32_t count);

  size_t getLogits(Tensor& logits,
                   InferenceStep& step,
                   int32_t count,
                   bool requireLogitsCopy = false);

  size_t getLogits(std::vector<Tensor>& logits_vec,
                   InferenceStep& step,
                   int32_t count,
                   bool requireLogitsCopy = false);

  bool debugOutputs(const InferenceStep& step, const std::string& tensor_name);

  void setHigherVariant() override;

  size_t loadKVCache(const std::string& load_path, bool chooseHigherVariant = false) override;

  bool saveKVCache(const std::string& save_path) override;

  bool saveTokenHistory(const std::string& savePath) override;

  bool loadTokenHistory(const std::string& loadPath) override;

  bool saveKVCacheToBuffer(Buffer* kv_buff) override;

  bool getCacheSpec(CacheFileSpec& spec) override;

  bool getKVHead(
      CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) override;

  template <typename T>
  size_t getQuantLogits(std::span<T> logits, bool logits_all);

  bool setupInputTensors(const std::vector<uint16_t*> eagle_embed_in,
                         const uint16_t* eagle_feature_in,
                         const std::vector<int32_t> selected,
                         uint32_t start_idx,
                         bool post_update,
                         int32_t n_past,
                         std::span<const int32_t> attention_map,
                         size_t n_skip_prefix,
                         size_t n_apply_prefix_offsef);

  // Utility function to check if the default/global cache group supports LongContext
  bool isLongContextEnabled() const override {
    // If no cache groups are availabe, LCL is trivially disabled.
    if (m_cache_group_params_map.size() == 0) {
      return false;
    }
    // Check if any cache group is not using LCL.
    for (auto& [prefix, cache_group_params] : m_cache_group_params_map) {
      if (cache_group_params.longcontext_params.mode == LongContextParams::DISABLED) return false;
    }
    return true;
  }

  void pauseQuery() override { m_pause = true; }

  bool setVisionParam(const std::vector<uint32_t>& visionParam) override;

  bool _skip_logits_tensor_check{false};

  // Eagle extra feature buffer
  uint8_t* eagle_extra_feature{nullptr};

  size_t dm_draftFeatureSize{0};

  // Temporary variable to support global SWA attention
  void* _attention_scratch{nullptr};
  // for vocab trim
  bool m_vocab_trim{false};
  int32_t m_vocab_trim_size{-1};

  bool finalizeState(std::shared_ptr<EngineState>& engineState) override;
  void set_n_batch_inputs(const std::vector<size_t>& batch_prompt_token_lens) {
    m_batch_prompt_token_lens = batch_prompt_token_lens;
    m_max_batch_prompt_token_len =
        *std::max_element(m_batch_prompt_token_lens.begin(), m_batch_prompt_token_lens.end());
  }

  void resetState() override;

  std::unique_ptr<TokenEmbedIO> m_tokenEmbedIO;

 protected:
  void updateFeatureBuffer(uint32_t count);

  inline void syncDrafTargetPrefill(bool isDraft, bool isReset);

  // Internal functions to separate different runInference logic
  inline bool updateTensorPointer(GraphVariant& variant, std::string& key, QnnUtils::Tensor*& t);

  virtual const char* getTraceNamespace() const override { return "QnnNspModel"; }

  size_t getMaxSingleTurnInputLength() override { return m_maxChunkedInputLength; };

 private:
  void zeroFillTensor(QnnUtils::Tensor& tensor);
};

}  // namespace qualla
