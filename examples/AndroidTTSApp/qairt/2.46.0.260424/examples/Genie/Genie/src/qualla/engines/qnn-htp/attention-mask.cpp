//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>

#include "attention-mask.hpp"
#include "qualla/detail/utils.hpp"

namespace qualla {

AttentionMask::AttentionMask(const std::vector<int32_t>& attention_map,
                             size_t n_past,
                             size_t n_kv,
                             size_t n_inputs,
                             size_t offset_to_apply_kv_prefix,
                             size_t size_to_skip_kv_prefix)
    : m_attention_map(attention_map),
      m_n_past(n_past),
      m_n_kv(n_kv),
      m_n_inputs(n_inputs),
      m_offset_to_apply_kv_prefix(offset_to_apply_kv_prefix),
      m_size_to_skip_kv_prefix(size_to_skip_kv_prefix) {
  // Parse type of attention being passed in
  if (m_attention_map.empty()) {
    m_attention_mode = AttentionMode::CAUSAL;
  } else if (m_attention_map.size() == m_n_inputs) {
    m_attention_mode = AttentionMode::RELATIONAL;

    // Validate that a token's attention mask cannot be based on a succeeding/future token
    for (size_t i = 0; i < m_n_inputs; i++) {
      if (m_attention_map[i] >= static_cast<int32_t>(i)) {
        throw std::runtime_error("Invalid attention mask provided by the Dialog");
      }
    }

  } else if (m_attention_map.size() == m_n_inputs * (m_n_past + m_n_inputs)) {
    m_attention_mode = AttentionMode::CUSTOM;
  } else {
    throw std::runtime_error("Invalid attention mask provided");
  }

  // Pre-calculate position ids for 1D attention masks
  if (m_attention_mode == RELATIONAL) {
    m_cached_attention_counts.resize(m_n_inputs, 0);
    for (size_t i = 0; i < m_n_inputs; i++) {
      if (m_attention_map[i] < 0) {
        m_cached_attention_counts[i] =
            m_n_past - static_cast<size_t>(-m_attention_map[i]) - m_size_to_skip_kv_prefix + 1;
      } else {
        const size_t parent_idx      = static_cast<size_t>(m_attention_map[i]);
        m_cached_attention_counts[i] = m_cached_attention_counts[parent_idx] + 1;
      }
    }
  }
}

std::vector<AttentionMask::AttentionSpan> AttentionMask::getAttentionSpans(size_t query_start_idx,
                                                                           size_t query_token_idx,
                                                                           size_t n_valid_kv,
                                                                           size_t past_idx,
                                                                           size_t new_idx) const {
  std::vector<AttentionSpan> spans;

  if (m_attention_mode == CAUSAL) {
    // Full causal attention - single contiguous span
    if (past_idx + n_valid_kv == new_idx) {
      spans.emplace_back(past_idx, n_valid_kv + query_token_idx + 1);
    } else {
      spans.emplace_back(past_idx, n_valid_kv);
      spans.emplace_back(new_idx, query_token_idx + 1);
    }
  } else if (m_attention_mode == RELATIONAL) {
    // For 1D attention masks, each index points to the parent row it attends to

    // Traverse up the attention tree to attend to parent tokens until the base case
    // Since we guarantee m_attention_map[i] < i (constructor check), this will always exit
    std::vector<AttentionSpan> reverse_spans;
    int32_t cur_token_idx = static_cast<int32_t>(query_start_idx + query_token_idx);
    while (cur_token_idx > 0) {
      reverse_spans.emplace_back(new_idx + static_cast<size_t>(cur_token_idx) - query_start_idx, 1);
      cur_token_idx = m_attention_map[static_cast<size_t>(cur_token_idx)];
    }

    // The base case is a negative parent token index, generally n=-1
    // -n means attend to all past tokens except the most recent n-1 tokens
    reverse_spans.emplace_back(new_idx, 1);
    reverse_spans.emplace_back(past_idx, m_n_kv - static_cast<size_t>(-cur_token_idx));

    // Merge adjacent spans together and reverse the list
    spans.emplace_back(reverse_spans.back());
    for (auto it = reverse_spans.rbegin() + 1; it != reverse_spans.rend(); ++it) {
      if (spans.back().start + spans.back().length == it->start) {
        spans.back().length += it->length;
      } else {
        spans.emplace_back(*it);
      }
    }
  }

  // Skip KV$ prefix if necessary
  applySSDPrefixSkipping(spans, query_start_idx + query_token_idx);

  return spans;
}

std::vector<AttentionMask::AttentionSpan> AttentionMask::getAttentionSpans(size_t query_token_idx,
                                                                           size_t n_valid_kv,
                                                                           size_t past_idx,
                                                                           size_t new_idx,
                                                                           size_t batch_prompt_token_len,
                                                                           size_t max_batch_prompt_token_len) const {
  std::vector<AttentionSpan> spans;

  if (m_attention_mode == CAUSAL) {
    size_t n_padding = max_batch_prompt_token_len - batch_prompt_token_len;
    if (past_idx + n_valid_kv == new_idx) {
      if (n_valid_kv >= max_batch_prompt_token_len) {
        spans.emplace_back(past_idx + n_padding, n_valid_kv - n_padding + query_token_idx + 1);
      } else {
        if (n_valid_kv <= n_padding) {
          size_t n_padding_left = n_padding - n_valid_kv;
          if (n_padding_left >= query_token_idx + 1) {
            spans.emplace_back(past_idx, 0);
          } else {
            spans.emplace_back(new_idx + n_padding_left, query_token_idx + 1 - n_padding_left);
          }
        } else {
          spans.emplace_back(past_idx + n_padding, n_valid_kv - n_padding + query_token_idx + 1);
        }
      }
    } else {
      if (n_valid_kv >= max_batch_prompt_token_len) {
        spans.emplace_back(past_idx + n_padding, n_valid_kv - n_padding);
        spans.emplace_back(new_idx, query_token_idx + 1);
      } else {
        if (n_valid_kv <= n_padding) {
          size_t n_padding_left = n_padding - n_valid_kv;
          if (n_padding_left >= query_token_idx + 1) {
            spans.emplace_back(past_idx, 0);
            spans.emplace_back(new_idx, 0);
          } else {
            spans.emplace_back(past_idx, 0);
            spans.emplace_back(new_idx + n_padding_left, query_token_idx + 1 - n_padding_left);
          }
        } else {
          spans.emplace_back(past_idx + n_padding, n_valid_kv - n_padding);
          spans.emplace_back(new_idx, query_token_idx + 1);
        }
      }
    }
  } else if (m_attention_mode == RELATIONAL) {
    throw std::runtime_error("RELATIONAL attention mode currently is not supported in multi-batch query processing");
  }

  return spans;
}

template <typename DType>
bool AttentionMask::fillAttentionRow(std::span<DType> attention_buffer,
                                     const size_t query_token_idx,
                                     const size_t n_past,
                                     const size_t n_valid_kv,
                                     const size_t past_idx,
                                     const size_t new_idx,
                                     const DType pos_val) const {
  if (m_attention_mode == AttentionMode::CUSTOM) {
    // For fully-specified/2D attn masks, we can directly iterate on the provided mask
    const size_t row_size  = m_n_past + m_n_inputs;  // Size of each row in the m_attention_map
    const size_t query_idx = (n_past - m_n_past) + query_token_idx;  // Which row we need to parse
    auto attention_row = std::span<const int32_t>(&m_attention_map[query_idx * row_size], row_size);

    // Fill the past attention buffer based on the attn_mask range [n_past-n_valid_kv, n_past]
    PRAGMA_LOOP_VECTORIZE
    for (size_t j = 0; j < n_valid_kv; j++) {
      if (attention_row[n_past - n_valid_kv + j]) {
        attention_buffer[past_idx + j] = pos_val;
      }
    }

    // Fill the new attention buffer based on the attn_mask range [n_past, n_past + query_token_idx]
    PRAGMA_LOOP_VECTORIZE
    for (size_t j = 0; j <= query_token_idx; j++) {
      if (attention_row[n_past + j]) {
        attention_buffer[new_idx + j] = pos_val;
      }
    }
  } else {
    // For 1D or empty attention masks, we can simplify construction into AttentionSpans
    for (const auto& span :
         getAttentionSpans(n_past - m_n_past, query_token_idx, n_valid_kv, past_idx, new_idx)) {
      std::fill_n(&attention_buffer[span.start], span.length, pos_val);
    }
  }

  return true;
}

template <typename DType>
bool AttentionMask::fillAttentionRow(std::span<DType> attention_buffer,
                                     const size_t query_token_idx,
                                     const size_t batch_prompt_token_len,
                                     const size_t max_batch_prompt_token_len,
                                     const size_t n_valid_kv,
                                     const size_t past_idx,
                                     const size_t new_idx,
                                     const DType pos_val) const {
  if (m_attention_mode != AttentionMode::CUSTOM) {
    // For 1D or empty attention masks, we can simplify construction into AttentionSpans
    for (const auto& span :
         getAttentionSpans(query_token_idx, n_valid_kv, past_idx, new_idx, batch_prompt_token_len, max_batch_prompt_token_len)) {
      std::fill_n(&attention_buffer[span.start], span.length, pos_val);
    }
  } else {
    throw std::runtime_error("CUSTOM attention mode currently is not supported in multi-batch query processing");
  }

  return true;
}

std::vector<int32_t> AttentionMask::getPositionIds(size_t query_start_idx,
                                                   size_t query_num_tokens,
                                                   size_t total_num_positions) const {
  std::vector<int32_t> position_ids(total_num_positions, 0);
  if (m_attention_mode == CAUSAL) {
    std::iota(&position_ids[0],
              &position_ids[query_num_tokens],
              m_n_past + query_start_idx - m_size_to_skip_kv_prefix);
  } else if (m_attention_mode == RELATIONAL) {
    PRAGMA_LOOP_VECTORIZE
    for (size_t i = 0; i < query_num_tokens; i++) {
      position_ids[i] = m_cached_attention_counts[query_start_idx + i];
    }
  } else {
    // For 2D attention masks, accumulate each row to calculate the position index
    for (size_t i = 0; i < query_num_tokens; i++) {
      const size_t row_size   = m_n_past + m_n_inputs;
      const int32_t* attn_row = &m_attention_map[(query_start_idx + i) * row_size];
      position_ids[i]         = std::accumulate(           // PositionID = #tokens attended
          &attn_row[m_size_to_skip_kv_prefix],     // Skip attention for forecast KV$
          &attn_row[m_n_past + query_num_tokens],  //
          -attn_row[m_n_past + query_start_idx + i]  // Skip counting itself, since ids start at 0
      );
    }
  }

  return position_ids;
}

std::vector<int32_t> AttentionMask::getPositionIds(size_t query_start_idx,
                                                   size_t query_num_tokens,
                                                   size_t total_num_positions,
                                                   size_t batch_prompt_token_len,
                                                   size_t max_batch_prompt_token_len) const {
  std::vector<int32_t> position_ids(total_num_positions, 0);
  if (m_attention_mode == CAUSAL) {
    size_t n_padding = max_batch_prompt_token_len - batch_prompt_token_len;
    if (m_n_past > 0) {
      std::iota(&position_ids[0],
                &position_ids[query_num_tokens],
                m_n_past - n_padding + query_start_idx - m_size_to_skip_kv_prefix);
    } else {
      if (query_start_idx <= n_padding) {
        size_t n_padding_left = n_padding - query_start_idx;
        if (n_padding_left < total_num_positions) {
          std::iota(&position_ids[n_padding_left],
                    &position_ids[query_num_tokens],
                    m_n_past - m_size_to_skip_kv_prefix);
        }
      } else {
        std::iota(&position_ids[0],
                  &position_ids[query_num_tokens],
                  m_n_past + query_start_idx - n_padding - m_size_to_skip_kv_prefix);
      }

    }
  }

  return position_ids;
}

void AttentionMask::applySSDPrefixSkipping(std::vector<AttentionSpan>& spans,
                                           size_t query_idx) const {
  // SSD prefix skipping is only applied to the first m_offset tokens for m_size_to_skip KV$
  if (m_size_to_skip_kv_prefix <= 0 || query_idx >= m_offset_to_apply_kv_prefix) {
    return;  // No prefix skipping needed
  }

  for (auto it = spans.begin(); it != spans.end();) {
    // Spans are sorted by start index. Once this exceeds m_size_to_skip_kv_prefix, we can stop
    if (it->start >= m_size_to_skip_kv_prefix) {
      return;
    }

    if (static_cast<size_t>(it->start) + it->length <= m_size_to_skip_kv_prefix) {
      // Entire span is in the skip region. Erase it.
      it = spans.erase(it);
    } else {
      // Partial overlap - adjust span to skip the prefix
      size_t skip_amount = m_size_to_skip_kv_prefix - it->start;
      it->start += skip_amount;
      it->length -= skip_amount;
      ++it;
    }
  }
}

// Explicit template instantiations
template bool qualla::AttentionMask::fillAttentionRow<uint8_t>(std::span<uint8_t>,
                                                               const size_t,
                                                               const size_t,
                                                               const size_t,
                                                               const size_t,
                                                               const size_t,
                                                               const uint8_t) const;
template bool qualla::AttentionMask::fillAttentionRow<uint16_t>(std::span<uint16_t>,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const uint16_t) const;
template bool qualla::AttentionMask::fillAttentionRow<uint32_t>(std::span<uint32_t>,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const uint32_t) const;
template bool qualla::AttentionMask::fillAttentionRow<uint8_t>(std::span<uint8_t>,
                                                               const size_t,
                                                               const size_t,
                                                               const size_t,
                                                               const size_t,
                                                               const size_t,
                                                               const size_t,
                                                               const uint8_t) const;
template bool qualla::AttentionMask::fillAttentionRow<uint16_t>(std::span<uint16_t>,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const uint16_t) const;
template bool qualla::AttentionMask::fillAttentionRow<uint32_t>(std::span<uint32_t>,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const size_t,
                                                                const uint32_t) const;

}  // namespace qualla
