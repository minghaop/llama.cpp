//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <fmt/format.h>

#include "nsp-image-model.hpp"
#include "nsp-model.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/engine.hpp"

namespace qualla {

class NspEngine : public Engine {
 protected:
  QnnNspBaseModel::Params _params;

  std::unique_ptr<QnnNspBaseModel> _model;
  std::vector<std::pair<uint32_t, uint32_t>> _tokensCheckpoint;
  std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint32_t>>>
      _savedTokenCheckpoints;
  std::string _model_type;
  std::shared_ptr<EngineState> m_engineState;

  // Internal helper function to process all input types.
  size_t processAll(const std::vector<int32_t>& tokens,
                    std::vector<uint8_t>& embeddings,
                    const uint16_t* featureVector,
                    const std::vector<int32_t>& selected,
                    uint32_t start_idx,
                    bool post_update,
                    const std::vector<int32_t>& attention_map,
                    std::vector<float>& logits,
                    bool logits_all);

  size_t processAll(const std::vector<int32_t>& tokens,
                    std::vector<uint8_t>& embeddings,
                    const uint16_t* featureVector,
                    const std::vector<int32_t>& selected,
                    uint32_t start_idx,
                    bool post_update,
                    const std::vector<int32_t>& attention_map,
                    Tensor& logits,
                    bool logits_all);

  size_t processAll(std::vector<int32_t>& tokens,
                    std::vector<uint8_t>& embeddings,
                    const std::vector<int32_t>& attention_map,
                    std::vector<size_t>& tokenNumPerBatch,
                    std::vector<Tensor>& logits,
                    bool logits_all);

 public:
  static constexpr const char* TYPE = "qnn-htp";

  NspEngine(Context& ctx, const nlohmann::json& json);
  virtual ~NspEngine();

  virtual size_t process(const std::vector<int32_t>& tokens,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(std::vector<int32_t>& tokens,
                         std::vector<size_t>& tokenNumPerBatch,
                         std::vector<Tensor>& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         const std::vector<int32_t>& attention_map,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         const std::vector<int32_t>& attention_map,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(std::vector<int32_t>& tokens,
                         const std::vector<int32_t>& attention_map,
                         std::vector<size_t>& tokenNumPerBatch,
                         std::vector<Tensor>& logits,
                         bool logits_all) override;

  virtual size_t process(std::vector<uint8_t>& embeddings,
                         const std::vector<int32_t>& attention_map,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(std::vector<uint8_t>& embeddings,
                         const std::vector<int32_t>& attention_map,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(std::vector<uint8_t>& embedding,
                         const uint16_t* featureVector,
                         const std::vector<int32_t>& selected,
                         uint32_t start_idx,
                         bool post_update,
                         const std::vector<int32_t>& attention_map,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(std::vector<uint8_t>& embedding,
                         const uint16_t* featureVector,
                         const std::vector<int32_t>& selected,
                         uint32_t start_idx,
                         bool post_update,
                         const std::vector<int32_t>& attention_map,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
                         std::vector<uint8_t>& outputs) override;

  virtual size_t setInputBufferData(LayerType inputLayer,
                                    void* data,
                                    size_t dataSize,
                                    nlohmann::json& dataConfig) override;

  virtual size_t processPresetInput(const nlohmann::json& config) override;
  virtual bool getOutputBufferData(LayerType outputLayer,
                                   const nlohmann::json& outputConfig,
                                   OutputCallback callback) override;

  /** Stores a precomputed EOS embedding vector. */
  virtual bool cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding) override;

  void getInputQuantParam(double& scale, int& offset);

  virtual qualla::InputType getInputType() override;

  virtual void getTensorParam(LayerType layerType,
                              std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& bitWidth) override;

  virtual void getTensorDimensions(LayerType layerType, std::vector<uint32_t>& dimensions) override;

  virtual void getInputTensorNames(std::unordered_set<std::string>& inputTensorNames) override;

  virtual size_t getEmbeddingBufferSize() override;

  virtual bool updateKV(size_t n_past) override;

  virtual bool updateKV(size_t n_past, const std::vector<bool>& selected) override;

  virtual bool save(const std::string& name) override;

  virtual bool saveKvToBuffer(Buffer* kv_buff) override;

  virtual bool getCacheSpec(CacheFileSpec& spec) override;

  virtual bool getKVHead(
      CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) override;

  virtual size_t restore(const std::string& name, bool chooseHigherVariant) override;

  virtual void reset() override;

  virtual bool set(nlohmann::json data) override;

  virtual nlohmann::json get() override;

  virtual bool load() override;

  virtual bool unload() override;

  virtual bool applyLoraAdapter(std::string lora_adapter_name) override;

  virtual bool releaseLoraAdapter(std::string lora_adapter_name) override;

  virtual std::string getAppliedLoraAdapter() const override;

  virtual bool applyLoraStrength(std::string tensor_name, float tensor_val) override;

  virtual bool setPerfProfile(qualla::PerformanceProfile& perfProfile) override;

  virtual bool getPerfProfile(qualla::PerformanceProfile& perfProfile) override;

  virtual bool updateTokenCheckpoint(uint32_t token, uint32_t kvCacheIndx) override;

  virtual bool removeTokenCheckpoint(size_t removeAmt) override;

  virtual std::pair<uint32_t, int32_t> rewindKVCacheToPrefixMatch(std::vector<int32_t>& tokens,
                                                                  uint32_t& past) override;
  virtual bool setOemkey(const std::string& oemKey) override;

  virtual bool setExecutionPriority(const uint32_t executionPriority) override;

  virtual size_t getBuffer(void*& buffer, std::string bufferName, bool isPrompt) override;

  virtual void setSharedCounter(std::atomic<int32_t>& counter) override;

  virtual void resetSharedCounter() override;

  virtual std::string getTokenMapFilePath() override;

  virtual void setRunProcess(uint8_t run_process) override;

  virtual bool applyEngineState(std::shared_ptr<EngineState>& engineState) override;

  virtual std::shared_ptr<EngineState> getEngineState() override;

  virtual bool isIoLoadingLazy() override;

  QUALLA_API virtual void updatedEmbeddingLength(uint32_t embedLength) override;

  QUALLA_API virtual bool isLongContextEnabled() const override;

  QUALLA_API virtual void pauseQuery() override;

  QUALLA_API virtual bool setVisionParam(const std::vector<uint32_t>& visionParam) override;

  QUALLA_API virtual bool setCrossAttentionHiddenStates(
      const std::vector<float>& crossAttentionStates) override;

  virtual const char* getTraceNamespace() const override { return "QnnHtp"; }

  virtual size_t getMaxSingleTurnInputLength() override;

  virtual bool registerModelAdaptor(std::shared_ptr<ModelIOAdaptor> adaptor) override;
};

}  // namespace qualla
