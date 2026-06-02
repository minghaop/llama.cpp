//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <deque>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "Trace.hpp"
#include "Traceable.hpp"
#include "nlohmann/json.hpp"
#include "qualla/context.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/sampler-utils.hpp"
#include "qualla/detail/tensor.hpp"

namespace qualla {

typedef std::function<void(const uint32_t, const void*, const uint32_t, int32_t*)>
    SamplerCbFunction;

typedef std::function<void(const uint32_t, const void*, const uint32_t, int32_t*, const void*)>
    SamplerUserDataCbFunction;

typedef std::unordered_map<std::string,
                           std::tuple<SamplerCbFunction, SamplerUserDataCbFunction, const void*>>
    SamplerCbFunctionMap;

class Sampler : public State {
 public:
  QUALLA_API Sampler(std::shared_ptr<Context>& ctx);
  QUALLA_API Sampler(std::shared_ptr<Context>& ctx,
                     const std::string& type,
                     const nlohmann::json& conf);

  const char* getTraceNamespace() const override { return "Sampler"; }

  // Sample a single token from logits
  QUALLA_API inline int32_t process(Tensor& logits, int32_t streamIdx = 0) {
    GENIE_TRACE();
    auto result = processUnified(logits, nullptr, 1, streamIdx);
    return result.empty() ? -1 : result[0];
  }

  // Sample multi_batch query tokens from logits vector
  QUALLA_API inline std::vector<int32_t> process(std::vector<Tensor>& logits, int32_t streamIdx) {
    std::vector<int32_t> result;
    for(int32_t batch_idx = 0; batch_idx < streamIdx; batch_idx++){
      auto result_per_batch = processUnified(logits.at(batch_idx), nullptr, 1, batch_idx);
      result.push_back(result_per_batch[0]);
    }
    return result.empty() ? std::vector<int32_t>{-1} : result;
  }

  // Sample a single token and output probabilities
  // Probs are appended to the existing vector
  QUALLA_API inline int32_t process(Tensor& logits,
                                    std::vector<float>& probs,
                                    bool out_tok      = true,
                                    int32_t streamIdx = 0) {
    GENIE_TRACE();
    auto result = processUnified(logits, &probs, out_tok ? 1 : 0, streamIdx, 0, true);
    return out_tok && !result.empty() ? result[0] : -1;
  }

  QUALLA_API inline std::vector<int32_t> process(Tensor& logits,
                                                 std::vector<float>& probs,
                                                 int32_t numReturn,
                                                 size_t topn_probs = 0,
                                                 int32_t streamIdx = 0) {
    GENIE_TRACE();
    return processUnified(logits, &probs, numReturn, streamIdx, topn_probs);
  }

  /**
   * Unified process method that handles all sampling scenarios
   * @param logits - logits tensor
   * @param probs - if provided, return probabilities (appended to vector)
   * @param numReturn - number of tokens to return (0 = no tokens, only probs; 1 = single token; >1
   * = multiple tokens)
   * @param topn_probs - top-n probabilities to output (0 = all)
   * @param streamIdx - stream index
   * @return vector of sampled token IDs
   */
  QUALLA_API std::vector<int32_t> processUnified(Tensor& logits,
                                                 std::vector<float>* probs = nullptr,
                                                 int32_t numReturn         = 1,
                                                 int32_t streamIdx         = 0,
                                                 size_t topn_probs         = 0,
                                                 bool output_all_probs     = false);

  QUALLA_API bool save(const std::string& name);
  QUALLA_API bool restore(const std::string& name);
  QUALLA_API void reset();
  QUALLA_API void applyConfig(const nlohmann::json& conf);

  // Get sampler type
  const std::string& type() const { return _type; }

  // Get sampler role
  const std::string& role() const { return _role; }

  // Get sampler params
  bool greedy() const { return _greedy; }
  bool gumbel() const { return _gumbel; }
  int32_t seed() const { return _seed; }

  // Get reference to the random number generator
  std::mt19937& rng() { return _rng; }

  // Get sampler params
  float temp() const { return _temp; }
  size_t top_k() const { return _top_k; }
  float top_p() const { return _top_p; }
  int32_t penalizeLastN() { return m_penalty.m_penaltyLastN; }
  float freqPenalty() { return m_penalty.m_penaltyFreq; }
  float repetitionPenalty() { return m_penalty.m_penaltyRepeat; }
  float presencePenalty() { return m_penalty.m_penaltyPresent; }
  Penalty& getPenalty() { return m_penalty; }

  // Set sampler params
  void temp(float t) { _temp = t; }
  void top_k(size_t k) { _top_k = k; }
  void top_p(float p) { _top_p = p; }
  void penalizeLastN(int32_t lastN) { m_penalty.m_penaltyLastN = lastN; }
  void freqPenalty(float penalty) { m_penalty.m_penaltyFreq = penalty; }
  void repetitionPenalty(float penalty) { m_penalty.m_penaltyRepeat = penalty; }
  void presencePenalty(float penalty) { m_penalty.m_penaltyPresent = penalty; }
  void updatePenalty(Penalty& penalty) { m_penalty = penalty; }
  void rng(std::mt19937& rng) { _rng = rng; }

  // Create Sampler instance
  QUALLA_API static std::shared_ptr<Sampler> create(std::shared_ptr<Context> ctx,
                                                    std::istream& json_stream);
  QUALLA_API static std::shared_ptr<Sampler> create(std::shared_ptr<Context> ctx,
                                                    const std::string& json_str);
  QUALLA_API static std::shared_ptr<Sampler> create(std::shared_ptr<Context> ctx,
                                                    const nlohmann::json& conf = {});

  QUALLA_API static void registerProcessCallBack(std::string name,
                                                 qualla::SamplerCbFunction callback);
  QUALLA_API static void registerUserDataCallBack(std::string name,
                                                  qualla::SamplerUserDataCbFunction callback,
                                                  const void* userData);

  QUALLA_API void updateSampledTokenHistory(int32_t tokenIdx, int32_t streamIdx = 0);
  QUALLA_API void updateSampledTokenHistory(std::vector<int32_t>& tokenIdx, int32_t streamIdx = 0);

  // Sample data from raw buffer
  QUALLA_API std::vector<int32_t> sampleData(const void* data,
                                             size_t dataSize,
                                             const char* dataConfigJson,
                                             int32_t numReturn = 1);

 protected:
  static SamplerCbFunctionMap& getSamplerCbFunctionMap();

  std::string _type;              // sampler type
  std::string _role;              // sampler role (primary, secondary, ...)
  std::shared_ptr<Context> _ctx;  // reference to the context
  std::shared_ptr<Env> _env;      // reference to the environment
  std::mt19937 _rng;
  int32_t _seed{-1};
  bool _greedy{false};
  bool _gumbel{false};
  float _temp{0.0f};
  size_t _top_k{0};
  float _top_p{1.0f};
  Penalty m_penalty;
  std::string _customProcessCallbackName;

  /**
   * Unified basic_process function that handles all sampling scenarios
   * @param logits - logits tensor
   * @param probs_out - if not null, return probs (appended to vector)
   * @param num_return - number of tokens to return (0 = no tokens, only probs; 1 = single token; >1
   * = multiple tokens)
   * @param streamIdx - stream index
   * @param topn_probs - top-n probabilities to output (0 = all)
   * @return vector of sampled token IDs
   */
  template <typename T>
  std::vector<int32_t> basic_process(Tensor& logits,
                                     std::vector<float>* probs_out,
                                     int32_t num_return    = 1,
                                     int32_t streamIdx     = 0,
                                     size_t topn_probs     = 0,
                                     bool output_all_probs = false);

  template <typename T>
  std::vector<int32_t> custom_process(Tensor& logits, int numTokens);
};

}  // namespace qualla
