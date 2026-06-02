//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace qualla {

/**
 * @brief Optimized attention mask processor
 */
class AttentionMask {
 public:
  /**
   * @brief Represents a contiguous span of attention values
   */
  struct AttentionSpan {
    size_t start;   ///< Start position in the attention sequence
    size_t length;  ///< Length of the contiguous span

    AttentionSpan(size_t s, size_t l) : start(s), length(l) {}
  };

  /**
   * @brief Represents the different modes attention mask was provided
   */
  enum AttentionMode { CAUSAL, RELATIONAL, CUSTOM };

  /**
   * @brief Constructor
   * @param attention_map 1D attention map from input
   * @param n_past Number of past tokens
   * @param n_kv Number of current KV$ (n_past - n_evicted)
   * @param n_inputs Number of input tokens
   * @param offset_to_apply_kv_prefix Offset for SSD KV prefix handling
   * @param size_to_skip_kv_prefix Size to skip for SSD KV prefix handling
   */
  AttentionMask(const std::vector<int32_t>& attention_map,
                size_t n_past,
                size_t n_kv,
                size_t n_inputs,
                size_t offset_to_apply_kv_prefix = 0,
                size_t size_to_skip_kv_prefix    = 0);

  /**
   * @brief Fill one row in the attention mask for a specific query token
   * @param attention_row Buffer to fill with attention values
   * @param query_token_idx Index of the query token
   * @param n_past Number of past tokens
   * @param n_valid_kv Number of valid KV$ tokens (n_past - n_evicted)
   * @param past_idx Index where past KV$ starts (relative to the full ctx_size)
   * @param new_idx Index where new KV$ is inserted by the model (relative to the full ctx_size)
   * @param pos_val
   */
  template <typename DType>
  bool fillAttentionRow(std::span<DType> attention_row,
                        const size_t query_token_idx,
                        const size_t n_past,
                        const size_t n_valid_kv,
                        const size_t past_idx,
                        const size_t new_idx,
                        const DType pos_val) const;

  /**
   * @brief Fill one row in the attention mask for a specific query token
   * @param attention_row Buffer to fill with attention values
   * @param query_token_idx Index of the query token
   * @param batch_prompt_token_len Number of prompt tokens in each batch
   * @param max_batch_prompt_token_len The maximum prompt token length within multi-batch
   * @param n_valid_kv Number of valid KV$ tokens (n_past - n_evicted)
   * @param past_idx Index where past KV$ starts (relative to the full ctx_size)
   * @param new_idx Index where new KV$ is inserted by the model (relative to the full ctx_size)
   * @param pos_val
   */
  template <typename DType>
  bool fillAttentionRow(std::span<DType> attention_row,
                        const size_t query_token_idx,
                        const size_t batch_prompt_token_len,
                        const size_t max_batch_prompt_token_len,
                        const size_t n_valid_kv,
                        const size_t past_idx,
                        const size_t new_idx,
                        const DType pos_val) const;

  /**
   * @brief Parse the count of KV$ each token attends to, translating to it's position id
   * @param query_start_idx Index of the starting query token
   * @param query_num_tokens Number of query tokens
   * @param total_num_positions Total number of positions to generate. This is used to
   *        generate position ids for the entire context, not just the query tokens. The remaining
   *        positions are padded with 0s.
   * @return Vector of position ids for each position in the context
   */
  std::vector<int32_t> getPositionIds(size_t query_start_idx,
                                      size_t query_num_tokens,
                                      size_t total_num_positions) const;

    /**
   * @brief Parse the count of KV$ each token attends to, translating to it's position id
   * @param query_start_idx Index of the starting query token
   * @param query_num_tokens Number of query tokens
   * @param total_num_positions Total number of positions to generate. This is used to
   *        generate position ids for the entire context, not just the query tokens. The remaining
   *        positions are padded with 0s.
   * @param batch_prompt_token_len Prompt token length of each batch
   * @param max_batch_prompt_token_len The maximum prompt token length within multi-batch
   * @return Vector of position ids for each position in the context
   */
  std::vector<int32_t> getPositionIds(size_t query_start_idx,
                                      size_t query_num_tokens,
                                      size_t total_num_positions,
                                      size_t batch_prompt_token_len,
                                      size_t max_batch_prompt_token_len) const;

  AttentionMode get_mode() const { return m_attention_mode; }
  size_t get_n_past() const { return m_n_past; }
  size_t get_n_kv() const { return m_n_kv; }
  size_t get_n_inputs() const { return m_n_inputs; }

 private:
  const std::vector<int32_t>& m_attention_map;
  const size_t m_n_past;
  const size_t m_n_kv;
  const size_t m_n_inputs;

  AttentionMode m_attention_mode;

  // Parameters for SSD prefix handling
  // SSD loads forecast prefix KV$ (size=m_size_to_skip), which is only attended by forecast tokens
  // Other tokens (size=m_offset_to_apply) must skip attending to this prefix KV$
  const size_t m_offset_to_apply_kv_prefix;
  const size_t m_size_to_skip_kv_prefix;

  // Pre-calculate position ids for 1D attention masks
  std::vector<int32_t> m_cached_attention_counts = {};
  /**
   * @brief Get attention spans for a specific query token
   * @param query_token_idx Index of the query token (relative to start of inputs)
   * @param n_valid_kv Number of valid KV$ tokens (n_past - n_evicted)
   * @param past_idx Index where past KV$ starts (relative to the full ctx_size)
   * @param new_idx Index where new KV$ is inserted by the model (relative to the full ctx_size)
   * @return Vector of contiguous attention spans for bulk operations
   */
  std::vector<AttentionSpan> getAttentionSpans(size_t query_start_idx,
                                               size_t query_token_idx,
                                               size_t n_valid_kv,
                                               size_t past_idx,
                                               size_t new_idx) const;

   /**
   * @brief Get attention spans for a specific query token
   * @param query_token_idx Index of the query token (relative to start of inputs)
   * @param batch_idx Index of the batch
   * @param n_valid_kv Number of valid KV$ tokens (n_past - n_evicted)
   * @param past_idx Index where past KV$ starts (relative to the full ctx_size)
   * @param new_idx Index where new KV$ is inserted by the model (relative to the full ctx_size)
   * @return Vector of contiguous attention spans for bulk operations
   */
  std::vector<AttentionSpan> getAttentionSpans(size_t query_token_idx,
                                               size_t n_valid_kv,
                                               size_t past_idx,
                                               size_t new_idx,
                                               size_t batch_prompt_token_len,
                                               size_t max_batch_prompt_token_len) const;

  /**
   * @brief Apply SSD prefix skipping to spans
   * @param spans Input spans to modify
   * @param query_token_idx Query token index
   */
  void applySSDPrefixSkipping(std::vector<AttentionSpan>& spans, size_t query_token_idx) const;
};

}  // namespace qualla
