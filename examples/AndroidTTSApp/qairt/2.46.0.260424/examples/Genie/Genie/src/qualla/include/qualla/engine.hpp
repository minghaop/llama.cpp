//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "IOBuffer.hpp"
#include "engineState.hpp"
#include "qualla/context.hpp"
#include "qualla/detail/buffer.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/kpi.hpp"
#include "qualla/detail/tensor.hpp"

namespace qualla {

// Type alias for qualla training data
using TrainingData = std::unordered_map<std::string, std::vector<std::vector<qualla::Tensor>>>;

class ModelIOAdaptor;

class Engine : public State {
 public:
  QUALLA_API Engine(Context& ctx, const std::string& type, const nlohmann::json& conf = {});
  QUALLA_API virtual ~Engine();

  using OutputCallback = std::function<bool(void*, size_t, nlohmann::json&)>;
  // Engine features
  struct Feature {
    enum Flags {
      OUTPUT_LOGITS     = (1UL << 0),  // Output of this engine is Logits
      OUTPUT_EMBEDDINGS = (1UL << 1),  // Output of this engine is Embeddings
      SAVE_RESTORE      = (1UL << 2),  // Save and restore support
      DYNAMIC_LOAD      = (1UL << 3)   // Dynamic loading / unloading support
    };
  };

  // Get engine feature mask
  uint32_t features() const { return _features; }
  bool supports(uint32_t flag) const { return _features & flag; }

  // Get engine type
  const std::string& type() const { return _type; }

  // Get engine role
  const std::string& role() const { return _role; }

  // Check if the engine uses cross attention
  virtual bool usesCrossAttention();

  // Check if the KV quantization is enabled
  virtual bool isKVQuantized();

  // Check if RoPE permutation is required
  virtual bool getRopePermute();

  // Process input tokens and generate output.
  // The output is Logits for LLM and Embeddings for Sentence Transformers.
  // Returns the number of tokens in the output.
  // TODO: remove implementation here once supported by all engines
  QUALLA_API virtual size_t process(const std::vector<int32_t>& tokens,
                                    std::vector<float>& output,
                                    bool output_all = false) = 0;

  QUALLA_API virtual size_t process(const std::vector<int32_t>& tokens,
                                    Tensor& output,
                                    bool output_all = false) = 0;

  QUALLA_API virtual size_t process(std::vector<int32_t>& tokens,
                                    std::vector<size_t>& tokenNumPerBatch,
                                    std::vector<Tensor>& output,
                                    bool output_all = false) {
    (void)tokens;
    (void)tokenNumPerBatch;
    (void)output;
    (void)output_all;
    return 0;
  }

  // Process input tokens and generate output.
  // The output is Logits for LLM and Embeddings for Sentence Transformers.
  // If provided, use attention mask and postion ids based on attention_map
  // Returns the number of tokens in the output.
  // TODO: remove implementation here once supported by all engines
  QUALLA_API virtual size_t process(const std::vector<int32_t>& tokens,
                                    const std::vector<int32_t>& attention_map,
                                    std::vector<float>& output,
                                    bool output_all = false);

  QUALLA_API virtual size_t process(std::vector<uint8_t>& embeddings,
                                    const std::vector<int32_t>& attention_map,
                                    std::vector<float>& output,
                                    bool output_all = false);

  QUALLA_API virtual size_t process(const std::vector<int32_t>& tokens,
                                    const std::vector<int32_t>& attention_map,
                                    Tensor& output,
                                    bool output_all = false);

  QUALLA_API virtual size_t process(std::vector<int32_t>& tokens,
                                    const std::vector<int32_t>& attention_map,
                                    std::vector<size_t>& tokenNumPerBatch,
                                    std::vector<Tensor>& output,
                                    bool output_all = false);

  // for embedding as input
  QUALLA_API virtual size_t process(std::vector<uint8_t>& embeddings,
                                    const std::vector<int32_t>& attention_map,
                                    Tensor& output,
                                    bool output_all = false);

  // Process input tokens without returning the output.
  // Returns the number of tokens in the output.
  QUALLA_API virtual size_t process(const std::vector<int32_t>& tokens);

  QUALLA_API virtual size_t process(
      const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
      std::vector<uint8_t>& outputs);

  // Synchronize the state of the context & engine.
  // n_past is the number of tokens in the KV Cache.
  QUALLA_API virtual bool updateKV(size_t n_past);
  // selected: selected tokens in the current run
  QUALLA_API virtual bool updateKV(size_t n_past, const std::vector<bool>& selected);

  QUALLA_API virtual bool save(const std::string& name);
  QUALLA_API virtual size_t restore(const std::string& name, bool chooseHigherVariant = false);
  QUALLA_API virtual bool saveKvToBuffer(qualla::Buffer* kv_buff);

  QUALLA_API virtual void reset();
  QUALLA_API virtual bool getCacheSpec(CacheFileSpec& spec);
  QUALLA_API virtual bool getKVHead(
      CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale);
  QUALLA_API virtual bool setKVHead(
      CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale);

  QUALLA_API virtual bool cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding);

  // Calculates the expected size of an embedding vector in bytes.
  QUALLA_API virtual size_t getEmbeddingBufferSize();

  QUALLA_API virtual qualla::InputType getInputType();

  QUALLA_API virtual void getTensorParam(
      LayerType layerType, std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth);

  QUALLA_API virtual void getTensorDimensions(LayerType layerType,
                                              std::vector<uint32_t>& dimensions);

  QUALLA_API virtual void getInputTensorNames(std::unordered_set<std::string>& inputTensorNames);

  // Load/unload all model/state data
  QUALLA_API virtual bool load();
  QUALLA_API virtual bool unload();

  // Set internal/run-time engine params/data
  QUALLA_API virtual bool set(nlohmann::json data);

  // Get internal/run-time engine params/data
  QUALLA_API virtual nlohmann::json get();

  QUALLA_API virtual size_t getBuffer(void*& buffer, std::string bufferName, bool isPrompt);

  QUALLA_API virtual size_t process(std::vector<uint8_t>& embedding_vectors,
                                    const uint16_t* featureVector,
                                    const std::vector<int32_t>& selected,
                                    uint32_t start_idx,
                                    bool post_update,
                                    const std::vector<int32_t>& attention_map,
                                    std::vector<float>& logits,
                                    bool logits_all);

  QUALLA_API virtual size_t process(std::vector<uint8_t>& embedding_vectors,
                                    const uint16_t* featureVector,
                                    const std::vector<int32_t>& selected,
                                    uint32_t start_idx,
                                    bool post_update,
                                    const std::vector<int32_t>& attention_map,
                                    Tensor& logits,
                                    bool logits_all);
  QUALLA_API virtual size_t processPresetInput(const nlohmann::json& config);

  QUALLA_API virtual void setSharedCounter(std::atomic<int32_t>& counter);

  QUALLA_API virtual void resetSharedCounter();

  QUALLA_API virtual void setRunProcess(uint8_t run_process = 0);

  QUALLA_API virtual void updatedEmbeddingLength(uint32_t embedLength);

  QUALLA_API virtual bool isLongContextEnabled() const;

  QUALLA_API virtual void pauseQuery();

  QUALLA_API virtual bool applyEngineState(std::shared_ptr<EngineState>& engineState);

  QUALLA_API virtual std::shared_ptr<EngineState> getEngineState();

  virtual const char* getTraceNamespace() const override { return "Engine"; }

  // Engine KPIs
  struct KPIs {
    Kpi load;
    Kpi process;
    Kpi update_kv;
    Kpi unload;

    KPIs() { reset(); }

    QUALLA_API void reset();                                        // reset to initial state
    QUALLA_API std::string dump(std::string_view sep = " ") const;  // dump KPIs as formated string
  };

  // Get Engine KPIs
  KPIs& kpis() { return _kpis; }

  // Get Engine context
  Context& context() { return _ctx; }

  // List available engines
  QUALLA_API static std::vector<std::string> list();

  // Create Engine instance
  QUALLA_API static std::shared_ptr<Engine> create(Context& ctx, std::istream& json_stream);
  QUALLA_API static std::shared_ptr<Engine> create(Context& ctx, const std::string& json_str);
  QUALLA_API static std::shared_ptr<Engine> create(Context& ctx, const nlohmann::json& conf = {});

  // Engine registration
  QUALLA_API virtual bool applyLoraAdapter(std::string lora_adapter_name);
  QUALLA_API virtual bool applyLoraStrength(std::string tensor_name, float tensor_val);
  QUALLA_API virtual bool releaseLoraAdapter(std::string lora_adapter_name);
  QUALLA_API virtual std::string getAppliedLoraAdapter() const;
  QUALLA_API virtual bool updateTokenCheckpoint(uint32_t token, uint32_t kvCacheIndx);
  QUALLA_API virtual bool setPerfProfile(qualla::PerformanceProfile& perfProfile);
  QUALLA_API virtual bool getPerfProfile(qualla::PerformanceProfile& perfProfile);
  QUALLA_API virtual bool removeTokenCheckpoint(size_t removeAmt);
  QUALLA_API virtual std::pair<uint32_t, int32_t> rewindKVCacheToPrefixMatch(
      std::vector<int32_t>& tokens, uint32_t& past);
  QUALLA_API virtual bool setOemkey(const std::string& oemKey);
  QUALLA_API virtual bool setExecutionPriority(const uint32_t executionPriority);

  QUALLA_API virtual std::string getTokenMapFilePath();
  QUALLA_API virtual bool isIoLoadingLazy();
  // Engine bound
  QUALLA_API bool isBound() { return _bound; }
  QUALLA_API void bound() { _bound = true; }
  QUALLA_API void unBound() { _bound = false; }

  QUALLA_API virtual bool setVisionParam(const std::vector<uint32_t>& visionParam);

  QUALLA_API virtual bool setCrossAttentionHiddenStates(
      const std::vector<float>& crossAttentionStates);

  QUALLA_API virtual std::vector<float> run_on_device_training(TrainingData& training_data);

  QUALLA_API virtual bool saveLoraAdapter(std::string lora_adapter_name);

  QUALLA_API virtual bool save_snapshot(const std::filesystem::path& path);

  QUALLA_API virtual bool load_snapshot(const std::filesystem::path& path);

  QUALLA_API virtual bool save_training_parameters(const std::filesystem::path& path);

  QUALLA_API virtual size_t getMaxSingleTurnInputLength();
  QUALLA_API virtual bool getOutputBufferData(LayerType outputLayer,
                                              const nlohmann::json& outputConfig,
                                              OutputCallback callback);
  QUALLA_API virtual size_t setInputBufferData(LayerType inputLayer,
                                               void* data,
                                               size_t dataSize,
                                               nlohmann::json& dataConfig);

  QUALLA_API virtual bool registerModelAdaptor(std::shared_ptr<ModelIOAdaptor> adaptor);

 protected:
  std::string _type;          // engine type
  std::string _role;          // engine role
  Context& _ctx;              // reference to the context
  std::shared_ptr<Env> _env;  // reference to the env
  KPIs _kpis;                 // our KPIs
  uint32_t _features{0};      // engine feature mask
  std::atomic_bool _bound{false};
};

}  // namespace qualla
