//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "cpu-model.hpp"
#include "qualla/engine.hpp"

namespace qualla {

class CpuEngine : public Engine {
 private:
  // Model parameters
  std::unique_ptr<QnnCpuModel> _model;
  std::vector<std::pair<uint32_t, uint32_t>> m_tokensCheckpoint;

  // Engine Sharing
  std::shared_ptr<EngineState> m_engineState;

 public:
  static constexpr const char* TYPE = "qnn-cpu";

  CpuEngine(Context& ctx, const nlohmann::json& json);
  ~CpuEngine();

  virtual bool usesCrossAttention() override;

  virtual bool isKVQuantized() override;

  virtual bool getRopePermute() override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         const std::vector<int32_t>& attention_map,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         const std::vector<int32_t>& attention_map,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(std::vector<uint8_t>& embeddings,
                         const std::vector<int32_t>& attention_map,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t getEmbeddingBufferSize() override;

  virtual bool updateKV(size_t n_past) override;

  virtual bool updateKV(size_t n_past, const std::vector<bool>& selected) override;

  virtual bool save(const std::string& name) override;

  virtual size_t restore(const std::string& name, bool chooseHigherVariant) override;

  virtual void reset() override;

  virtual bool applyLoraAdapter(std::string lora_adapter_name) override;

  virtual bool applyLoraStrength(std::string tensor_name, float tensor_val) override;

  virtual bool removeTokenCheckpoint(size_t removeAmt) override;

  virtual bool updateTokenCheckpoint(uint32_t token, uint32_t kvCacheIndx) override;

  virtual std::pair<uint32_t, int32_t> rewindKVCacheToPrefixMatch(std::vector<int32_t>& tokens,
                                                                  uint32_t& past) override;

  virtual qualla::InputType getInputType() override;

  virtual void getTensorParam(LayerType layerType,
                              std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& bitWidth) override;

  virtual bool setKVHead(
      CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) override;

  virtual const char* getTraceNamespace() const override { return "QnnCpu"; }

  virtual bool applyEngineState(std::shared_ptr<EngineState>& engineState) override;

  virtual std::shared_ptr<EngineState> getEngineState() override;
};

}  // namespace qualla
