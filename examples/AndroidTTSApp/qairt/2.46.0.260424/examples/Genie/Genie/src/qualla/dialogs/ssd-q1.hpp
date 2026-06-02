//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <span>

#include "qualla/dialog.hpp"

namespace qualla {

class SelfSpecDecDialog : public Dialog {
 public:
  static constexpr const char* TYPE = "ssd-q1";

  SelfSpecDecDialog(std::shared_ptr<Env> env, const std::string& name, const nlohmann::json& conf);

  virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;

  virtual bool process(std::vector<uint8_t>& embedding_vectors,
                       Dialog::T2ECallback t2eCallback,
                       Dialog::Callback callback) override;

  virtual void reset() override;

  virtual bool process(std::vector<int32_t>& /*tokens*/, DialogCallback /*callback*/) override {
    return false;
  }

  virtual bool save(const std::string& name) override;

  virtual bool restore(const std::string& name) override;

  virtual bool supportsPauseResume() override { return true; };

  virtual const char* getTraceNamespace() const override { return "Dialog::SSD"; };

 protected:
  virtual bool supportsLongContext() const override {
    return (_n_streams <= 1);  // Multistream not supported.
  };

 private:
  enum { VERSION = 1 };

  Sampler& _t_sampler;
  uint32_t _vocab;
  std::string _kv_prefix_name{"forecast-prefix"};

  // AR8
  size_t _draft{1};
  std::vector<std::vector<size_t>> _branches{{3}};

  size_t _forecast_prefix{16};
  size_t _forecast_token_offset{32000};
  size_t _prefix_quant{0}; // Number of prefix quant tokens

  // Multistream parameters
  uint32_t _n_streams{1};
  float _p_threshold{0.0f};

  size_t m_num_draft_nodes;                       // Number of total draft nodes
  std::vector<size_t> m_samples_per_draft_level;  // Max number of tokens to sample at each level
  std::vector<size_t> m_nodes_per_draft_level;    // Number of total nodes at each level
  std::vector<int32_t> m_attention_map;  // Cache the attention mask used for SSD generation

  bool processFollowOnGeneration(std::vector<int32_t>& tokens,
                                 Tensor& logits,
                                 Dialog::Callback callback);
  // Multistream
  bool processFollowOnGeneration(std::vector<std::vector<int32_t>>& streams,
                                 Tensor& logits,
                                 Dialog::Callback callback);

  /**
   * Helper function for combining masks for SSD mulstistream.
   *
   * @param  masks           The attention mask to be tiled
   * @param  streamIndices   Indices of streams. The tiling count is equal to the size of this
   * vector.
   * @param  pastMap         A vector of stream indices for masking all past tokens after the
   * prompt.
   * @param  prefixOffset    Offset where KV prefix masking begins in each tile.
   * @param  finalMask       A mask that combines all of the independent masks such that
   *                         they can be executed in the same inference.
   */
  void tileAttentionMask(const std::vector<int32_t>& mask,
                         const std::vector<size_t>& streamIndices,
                         const std::vector<size_t>& pastMap,
                         const size_t prefixOffset,
                         std::vector<int32_t>& finalMask);

  std::vector<int32_t> gen_attention_map();

  auto gen_forecast_tokens(int repeat) const;

  // Sampling and verification
  std::vector<int32_t> build_sample_tree(int32_t last_token,
                                         Tensor& logits,
                                         size_t start_offset,
                                         int32_t streamIdx = 0);

  std::pair<std::vector<int32_t>, std::vector<int32_t>> verify_draft_tree(
      std::span<int32_t> draft_tree, Tensor& logits);

  std::vector<int32_t> sample_to_draft(Tensor& logits,
                                       size_t index,
                                       size_t count,
                                       int32_t streamIdx);

  int32_t sample_to_verify(Tensor& logits, size_t index);

  void convertTokensToEmbeddings(std::vector<int32_t>& tokens,
                                 std::vector<uint8_t>& embeddings,
                                 size_t embeddingBufferSize,
                                 Dialog::T2ECallback t2eCallback);

  void completeInit() override;
};

}  // namespace qualla
