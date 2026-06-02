//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "kvmanager.hpp"
#include "nsp-graph.hpp"
#include "qnn-utils.hpp"
#include "qualla/LoraConfig.hpp"
#include "qualla/ModelIOAdaptor.hpp"
#include "qualla/detail/tensor.hpp"
#include "qualla/detail/utils.hpp"
#include "qualla/engine.hpp"
#include "qualla/engineState.hpp"
#include "qualla/env.hpp"
#include "tokenEmbedIO.hpp"

namespace qualla {

enum class ModelArchitectureType : uint8_t { DECODER = 0, ENCODER = 1 };

// Concrete implementation of ModelIOSpec for NspModel
class NspModelIOSpec : public ModelIOSpec {
 private:
  std::set<ModelVariant> m_variants;
  std::map<ModelVariant, std::unique_ptr<TensorMap>> m_input_specs;
  std::map<ModelVariant, std::unique_ptr<TensorMap>> m_output_specs;

  std::vector<std::unique_ptr<Tensor>> m_tensors;

 public:
  NspModelIOSpec() = default;
  void addModelVariant(ModelVariant variant,
                       const QnnUtils::TensorMap& qnn_input_spec,
                       const QnnUtils::TensorMap& qnn_output_spec,
                       IOTensor& ioTensorManager);

  const std::set<ModelVariant>& getVariants() const override;

  const TensorMap* getInputSpec(const ModelVariant& variant) const override;

  const TensorMap* getOutputSpec(const ModelVariant& variant) const override;

  static std::unique_ptr<Tensor> createTensorFromQnnTensor(const QnnUtils::Tensor& qnn_tensor,
                                                           IOTensor& ioTensorManager);
};

class QnnNspBaseModel : public State {
 public:
  const std::filesystem::path model_basedir;

  struct Params {
    ModelArchitectureType modelArchitectureType;  // Model architecture
    std::filesystem::path model_basedir;          // model basedir
    std::vector<std::string> model_list;          // model filenames
    std::map<int32_t, int32_t> variant_latency;   // latency for different variants
    std::vector<std::string> exec_select_graphs;  // Execute selected graphs
    bool load_select_graphs;  // Load only graphs mentioned in exec_select_graphs from the context
                              // bin, by default all graphs are loaded

    bool use_mmap;
    uint64_t data_alignment_size;
    bool use_async_Init;
    bool shared_Engine;
    uint64_t mmap_budget;
    size_t spill_fill_bufsize;
    size_t ctx_size;
    int32_t kv_dim;
    uint32_t batch_size{1};
    int32_t pad_token;
    int32_t img_token;
    size_t n_embd;
    uint32_t n_threads{0};
    uint64_t cpumask{0};
    bool poll{false};
    std::string backend_lib;
    std::string backend_ext_conf;
    std::string debug_path;
    std::string draft_tok_map;
    bool debug_specs;
    bool debug_tensors;
    bool debug_outputs;
    bool debug_qnn;
    std::string kv_update_method;
    std::string lmhead_weight_dir;
    std::string mask_type;
    bool cross_attention;
    bool graph_switching;
    std::string input_layer_name;
    int32_t embedding_length;
    std::string embedding_datatype;
    bool pooled_output;
    bool disable_kv_cache;
    // LoRA params
    std::string lazy_lora;
    LoraConfigType lora_conf_type{LoraConfigType::LORA_DISABLE};
    bool weight_shared_lora;
    bool skip_lora_validation;
    std::shared_ptr<LoraConfig> lora_config;

    // Parameters for positional encodings
    PositionalEncoding positional_encoding_params;

    // Parameters for cache groups
    std::string default_group;
    CacheGroupParamsMap cache_group_params;

    // vision parameters for image encoder
    VisionParam vision_param;

    // External Buffer Params
    ExternalBufferType externalBufferType;
    std::string dlc_path;
  };

  enum RUN_PROCESS : uint8_t {
    OVERALL_PROCESS = 0,
    PART_RUN        = 1,
    NO_RUN_LMHEAD   = 2
  } _run_process = OVERALL_PROCESS;

  QnnNspBaseModel(std::shared_ptr<Env> env, const QnnNspBaseModel::Params& params);
  virtual ~QnnNspBaseModel();

  virtual bool initializeModel(void)      = 0;
  virtual bool validateModel(void)        = 0;
  virtual bool initializeIOTensors(void)  = 0;
  virtual bool initializeTensorPointers() = 0;

  /**
   * Some models may contain unused inputs, either for debugging purposes
   * or for models which have optional encoders (e.g. Deepstack.)
   *
   * Initialize any such inputs to their quantized zero point.
   */
  virtual void initializeUnconnectedInputs(){};

  virtual bool initializeKVManager() { return true; };
  virtual bool calculate_rope_embeddings(void) { return true; };
  virtual bool load_lmhead_weight_as_input(void) { return true; };

  virtual size_t loadKVCache(const std::string& /*load_path*/, bool chooseHigherVariant = false) {
    static_cast<void>(chooseHigherVariant);
    return 0;
  };
  virtual void setHigherVariant() { return; };
  virtual bool saveTokenHistory(const std::string& /*savePath*/) { return true; };
  virtual bool loadTokenHistory(const std::string& /*loadPath*/) { return true; };
  virtual bool saveKVCache(const std::string& /*save_path*/) { return true; };
  virtual bool saveKVCacheToBuffer(Buffer* /*kv_buff*/) { return true; };
  virtual bool getCacheSpec(CacheFileSpec& /*spec*/) { return true; };
  virtual bool getKVHead(CacheFileSpec /*spec*/,
                         uint32_t /*layer*/,
                         uint32_t /*head*/,
                         void* /*data*/,
                         double* /*scale*/) {
    return true;
  };

  virtual size_t getEmbeddingBufferSize() { return 0; };

  virtual size_t runInference(const std::vector<int32_t>& /*tokens*/,
                              std::vector<uint8_t>& /*embeddings*/,
                              const uint16_t* /*featureVector*/,
                              const std::vector<int32_t>& /*selected*/,
                              uint32_t /*start_idx*/,
                              bool /*post_update*/,
                              const std::vector<int32_t>& /*attention_map*/,
                              std::vector<float>& /*output*/,
                              bool output_all = false) {
    static_cast<void>(output_all);
    return 0;
  }

  virtual size_t runInference(const std::vector<int32_t>& /*tokens*/,
                              std::vector<uint8_t>& /*embeddings*/,
                              const uint16_t* /*featureVector*/,
                              const std::vector<int32_t>& /*selected*/,
                              uint32_t /*start_idx*/,
                              bool /*post_update*/,
                              const std::vector<int32_t>& /*attention_map*/,
                              Tensor& /*output*/,
                              bool output_all = false) {
    static_cast<void>(output_all);
    return 0;
  }

  virtual size_t runInference(std::vector<int32_t>& /*tokens*/,
                              std::vector<uint8_t>& /*embeddings*/,
                              const std::vector<int32_t>& /*attention_map*/,
                              std::vector<size_t>& /*tokenNumPerBatch*/,
                              std::vector<Tensor>& /*output*/,
                              bool output_all = false) {
    static_cast<void>(output_all);
    return 0;
  }

  virtual size_t runInference(
      const std::unordered_map<std::string, std::vector<uint8_t>>& /*inputs*/,
      std::vector<uint8_t>& /*outputs*/) {
    return 0;
  }

  virtual size_t runPrefillInputInference(const nlohmann::json executionConfig = {}) {
    static_cast<void>(executionConfig);
    return false;
  }

  virtual bool getOutputBufferData(LayerType /*outputLayer*/,
                                   const nlohmann::json& /*outputConfig*/,
                                   Engine::OutputCallback /*callback*/) {
    return false;
  }
  virtual size_t setInputBufferData(LayerType /*inputLayer*/,
                                    void* /*data*/,
                                    size_t /*dataSize*/,
                                    nlohmann::json& /*dataConfig*/) {
    return false;
  }

  virtual void getInputQuantParam(double& /*scale*/, int& /*offset*/){};

  virtual bool cacheEosEmbedding(std::vector<uint8_t>& /*eosEmbedding*/) { return true; };

  virtual bool setKVCacheNPast(size_t /*n_past*/, const std::vector<bool>& /*selected*/) {
    return true;
  };

  virtual void getTensorParam(LayerType /*layerType*/,
                              std::string& /*dataType*/,
                              double& /*scale*/,
                              int32_t& /*offset*/,
                              size_t& /*bitWidth*/){};

  virtual void getTensorDimensions(LayerType /*layerType*/,
                                   std::vector<uint32_t>& /*dimensions*/){};

  virtual void getInputTensorNames(std::unordered_set<std::string>& /*inputTensorNames*/) {}

  virtual void pauseQuery(){};

  virtual bool setVisionParam(const std::vector<uint32_t>& /*visionParam*/) { return false; }

  virtual bool setCrossAttentionHiddenStates(const std::vector<float>& /*crossAttentionStates*/) {
    return false;
  }

  // ModelIOAdaptor registration and management
  virtual bool registerModelIOAdaptor(std::shared_ptr<ModelIOAdaptor> adaptor);

  void setSharedCounter(std::atomic<int32_t>& counter) { _counter = &counter; };
  void resetSharedCounter() { _counter = nullptr; };

  virtual bool finalizeState(std::shared_ptr<EngineState>& engineState);

  virtual void resetState();
  // Variables for positional encodings
  PositionalEncoding m_positional_encoding;
  QnnUtils::DataType d_pos{QNN_DATATYPE_UFIXED_POINT_16};

  QnnUtils::Tensor* t_position_ids_sin{nullptr};
  QnnUtils::Tensor* t_position_ids_cos{nullptr};

  // Self-Specualtive Decoding
  // This prefix is not for input tokens, but just for speical tokens
  // Only the special tokens from the offset should attend the kv prefix
  size_t _size_to_skip_kv_prefix{0};
  size_t _offset_to_apply_kv_prefix{0};

  std::atomic<int32_t>* _counter{nullptr};

  InputType m_inputType{InputType::UNKNOWN};

  // LoRA params and configs
  LoraConfigType lora_conf_type{LoraConfigType::LORA_DISABLE};
  std::shared_ptr<LoraConfig> lora_config;
  bool _lora_enabled{false};

  // QNN specific variables
  const bool m_sharedBuffer{true};
  bool m_lazyInitialization{false};
  std::unique_ptr<QnnApi> m_qnnApi;
  std::shared_ptr<IOTensor> m_ioTensor;
  size_t spill_fill_buffer_size;
  bool m_use_mmap{false};
  uint32_t m_dataAlignmentSize{0};
  bool m_use_async_Init{true};
  uint64_t mmap_budget;
  bool graph_switching{false};
  bool weight_shared_lora{false};
  std::string lazy_lora{""};
  bool skip_lora_validation{false};
  size_t n_embd;

  bool m_pooled_output{true};
  bool m_disableKvCache{false};

  std::string _backend_lib;
  std::string _backend_ext_conf;
  std::string m_draft_tok_map;

  // Debug mode settings
  bool _debug_specs{false};
  bool _debug_tensors{false};
  bool _debug_outputs{false};
  bool _debug_qnn{false};
  std::string _debug_path;

  // Keep track of inference count
  int m_inference_count = 0;

  // QnnNspGraph contains all GraphVariants for a specific split (with index=split_idx)
  std::vector<QnnNspGraph> m_nsp_graphs;
  std::unordered_map<std::string, GraphType> m_graph_variant_type_map;
  // GraphVariant represents one input size within one split (e.g. KV$_split_1)
  std::vector<GraphVariant> m_variant_list;

  // For ease of usage: Map from graph name to the corresponding GraphVariant
  std::unordered_map<std::string, GraphVariant*> m_graph_map;
  // This map records how many graphs have been loaded for a particular input size and context size
  std::map<std::pair<int32_t, int32_t>, int32_t> nsp_graph_count;  // [variant, ctx_size] -> count

  uint32_t m_embedding_length{0};
  std::string m_dlc_path;

  // Base functionality
  bool setExecutionPriority(const uint32_t executionPriority);
  bool setOemKey(const std::string& oemKey);
  bool flushLoraWeightsBuffers(void);
  bool applyLoraStrength(const std::string& alpha_name, const float alpha_val);
  bool applyLoraWeights(const std::string& lora_weights_name);
  bool applyBinarySections(std::vector<std::string>& binsection_list);
  bool applyLoraAdapter(const std::string& lora_adapter_name);
  bool releaseLoraAdapter(const std::string& lora_adapter_name);
  bool setPerfProfile(qualla::PerformanceProfile& perfProfile);
  bool getPerfProfile(qualla::PerformanceProfile& perfProfile);
  void setRunProcess(uint8_t run_process) { _run_process = static_cast<RUN_PROCESS>(run_process); }
  void updatedEmbeddingLength(uint32_t embedLength) { m_embedding_length = embedLength; };

  virtual bool isLongContextEnabled() const { return false; }

  bool debugOutputs(const InferenceStep& step, const std::string& tensor_name);
  void dumpTensorSpecs();
  virtual size_t getIOBufferByName(std::string /*tensor_name*/,
                                   void*& /*buffer*/,
                                   bool /*isPrompt*/) {
    return 0;
  }

  virtual const char* getTraceNamespace() const override { return "QnnNspBaseModel"; }

  virtual size_t getMaxSingleTurnInputLength() { return 0; };

 protected:
  std::shared_ptr<Env> _env;

  NspModelIOSpec& IOSpec();
  // Helper methods for calling adaptors
  void callAdaptorsReset();

  bool prepareAdaptorInputs(int32_t variant,
                            int32_t ctx_size,
                            uint32_t n_past,
                            uint32_t n_processed,
                            uint32_t n_inputs,
                            bool isLastInference,
                            bool& shouldSkipInference);

  bool handleAdaptorOutputs(int32_t variant,
                            int32_t ctx_size,
                            uint32_t n_past,
                            uint32_t n_processed,
                            uint32_t n_inputs,
                            bool isLastInference);

  bool isSupportedActivation(Qnn_DataType_t type) const;

  bool float32ToFloat16(uint8_t* out, float* in, size_t numElements);

  // Retrieve tensor
  inline void* getBuffer(QnnUtils::Tensor& spec) {
    return m_ioTensor ? m_ioTensor->getBuffer(spec.tensor) : nullptr;
  }
  inline void* getBuffer(QnnUtils::Tensor* spec) { return spec ? getBuffer(*spec) : nullptr; }

  // Calculate tensor size
  inline size_t getBufferSize(QnnUtils::Tensor& spec) { return spec.dims.getSize(); }
  inline size_t getBufferSize(QnnUtils::Tensor* spec) { return spec ? getBufferSize(*spec) : 0; }

  bool finalizeLora(std::shared_ptr<EngineState>& engineState);

  template <typename U, typename T>
  inline void deQuantizeOutputs(
      U* inputs, std::span<T>& outputs, double scale, int32_t offset, size_t numElements) {
    PRAGMA_LOOP_VECTORIZE
    for (size_t i = 0; i < numElements; ++i) {
      outputs[i] = static_cast<T>((static_cast<int32_t>(inputs[i]) + offset) * scale);
    }
  }

  template <typename U, typename T>
  inline void castOutputs(U* inputs, std::span<T>& outputs, size_t numElements, uint32_t bitWidth) {
    if (bitWidth == 2) {
      PRAGMA_LOOP_VECTORIZE
      for (size_t i = 0; i < numElements; ++i) {
        outputs[i] = static_cast<T>(fp16_ieee_to_fp32_value(inputs[i]));
      }
    } else if (bitWidth == 4) {
      PRAGMA_LOOP_VECTORIZE
      for (size_t i = 0; i < numElements; i++) {
        outputs[i] = static_cast<T>(inputs[i]);
      }
    }
  }

 private:
  std::unique_ptr<NspModelIOSpec> m_ModelIOSpec;
  std::vector<std::shared_ptr<ModelIOAdaptor>> m_adaptors;
};

}  // namespace qualla
