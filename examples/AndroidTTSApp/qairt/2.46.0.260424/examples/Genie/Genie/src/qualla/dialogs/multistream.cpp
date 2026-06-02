//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "Trace.hpp"
#include "multistream.hpp"
#include "qualla/detail/timer.hpp"

namespace fs = std::filesystem;
using qc     = qualla::Config;

namespace qualla {

MultiStreamDialog::MultiStreamDialog(std::shared_ptr<Env> env,
                                     const std::string& name,
                                     const nlohmann::json& conf)
    : Dialog(env, name, conf) {
  _vocab       = _ctx->n_vocab();
  _n_streams   = qc::optional<uint32_t>(conf, "n-streams", 1);
  _p_threshold = qc::optional<float>(conf, "p-threshold", 0.0);
}

bool MultiStreamDialog::processFollowOnGeneration(std::vector<std::vector<int32_t>>& streams,
                                                  Tensor& logits,
                                                  Dialog::Callback callback) {
  GENIE_TRACE();
  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

  std::vector<std::vector<int32_t>> attention_mask(_n_streams);
  std::vector<size_t> streamIndices;

  if (streams.size() == 0) {
    callback("\n", Sentence::END);
    return true;
  }

  for (size_t i = 0; i < streams.size(); i++) {
    // Initialize all attention_masks to attend to all previous tokens
    attention_mask[i].resize(_n_past, 1);
    streamIndices.push_back(i);
  }

  State::busy(true);

  while (true) {
    if (State::canceled()) break;

    // If this exceeds context length, truncate all streams and return
    if (_n_past + streamIndices.size() > _ctx->size()) {
      for (auto stream : streamIndices)
        callback(_tokenizer->decode(streams[stream]) + "\n", Sentence::CONTINUE);
      break;
    }

    // Accumulate input tokens from all streams
    std::vector<int32_t> multi_tokens(streamIndices.size());

    for (size_t i = 0; i < streamIndices.size(); i++) {
      multi_tokens[i] = streams[streamIndices[i]].back();

      // Also add current iteration to the attention_mask
      for (auto _mask_row : streamIndices)
        // Set to true iff on diagonal, i.e. attend to itself
        attention_mask[streamIndices[i]].push_back((streamIndices[i] == _mask_row) ? 1 : 0);
    }

    // Concatenate attention_mask for all active streams
    std::vector<int32_t> multi_attn_mask;
    multi_attn_mask.reserve((_n_past + streamIndices.size()) * streamIndices.size());
    for (auto i : streamIndices)
      multi_attn_mask.insert(
          multi_attn_mask.end(), attention_mask[i].begin(), attention_mask[i].end());

    // __DEBUG("Multi attention mask = {}", multi_attn_mask);

    if (m_inputType == InputType::TOKENS) {
      // Process input tokens for all streams in one batch
      if (!engine.process(multi_tokens, multi_attn_mask, logits, true))
        return Dialog::abort("engine gen processing failed", callback);
    } else if (m_inputType == InputType::EMBEDDINGS) {
      // Accumulate input embeddings from all streams
      auto embedBufSize = engine.getEmbeddingBufferSize();
      std::vector<uint8_t> multi_embeddings;

      for (auto token : multi_tokens) {
        // Convert tokens to embedding for the processing in the engine.
        std::vector<uint8_t> curTokenEmbedding(embedBufSize, 0);
        m_t2eCallback(*this, token, curTokenEmbedding.data(), embedBufSize);
        multi_embeddings.insert(
            multi_embeddings.end(), curTokenEmbedding.begin(), curTokenEmbedding.end());
      }

      // Process input tokens for all streams in one batch
      if (!engine.process(multi_embeddings, multi_attn_mask, logits, true))
        return Dialog::abort("engine gen processing failed", callback);
    }

    // Process all logits independently
    for (size_t i = 0; i < streamIndices.size(); i++) {
      Tensor indexedLogits = logits.getIndexedTensor(i, _vocab);
      _last_tok            = sampler.process(indexedLogits);
      sampler.updateSampledTokenHistory(_last_tok, i);
      streams[streamIndices[i]].push_back(_last_tok);
    }

    _n_past += streamIndices.size();
    _n_generated += streamIndices.size();
    _n_decode += streamIndices.size();

    if (!engine.updateKV(_n_past)) return Dialog::abort("KV update failed", callback);

    for (auto it = streamIndices.begin(); it != streamIndices.end();) {
      uint32_t stream = *it;
      if (_ctx->is_eos(streams[stream].back())) {
        callback(_tokenizer->decode(streams[stream]) + "\n", Sentence::CONTINUE);
        it = streamIndices.erase(it);
      } else {
        ++it;
      }
    }

    if (streamIndices.size() == 0) break;
  }
  callback("\n", Sentence::END);

  State::busy(false);

  return true;
}

bool MultiStreamDialog::process(std::vector<int32_t>& tokens, Dialog::Callback callback) {
  GENIE_TRACE();
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  if (m_inputType != InputType::TOKENS) {
    __ERROR("Input type for model is not tokens.");
    return false;
  }

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor logits;

  State::clear();

  auto& engine = *_engine["primary"];

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

  if (_n_past + tokens.size() > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
    throw genie::ContextLimitException("Context Size was exceeded.");
  }

  if (!engine.process(tokens, logits, false))
    return Dialog::abort("engine prompt processing failed", callback);

  _n_prompt += tokens.size();
  _n_past += tokens.size();

  _prompt_len = _n_past;

  if (!engine.updateKV(_n_past)) return Dialog::abort("KV update failed", callback);

  std::vector<std::vector<int32_t>> streams;
  getTopK(logits, streams, _n_streams, _p_threshold, callback);

  _n_generated += streams.size();
  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  start.reset();

  bool status = processFollowOnGeneration(streams, logits, callback);

  _kpis.generate.update(start.elapsed_usec());

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return status;
}

bool MultiStreamDialog::process(std::vector<uint8_t>& embedding_vectors,
                                T2ECallback t2eCallback,
                                Dialog::Callback callback) {
  GENIE_TRACE();
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  if (m_inputType != InputType::EMBEDDINGS) {
    __ERROR("Input type for model is not embeddings.");
    return false;
  }

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor logits;

  State::clear();

  auto& engine = *_engine["primary"];

  // Store the t2e callback for reference during follow-on generation.
  m_t2eCallback = t2eCallback;

  size_t embedBufSize = engine.getEmbeddingBufferSize();

  {
    std::vector<uint8_t> eosEmbedding(embedBufSize, 0.0);
    if (m_t2eCallback) {
      m_t2eCallback(*this, _ctx->eos(), eosEmbedding.data(), embedBufSize);
    }
    // For non-autogenerative usecases (where t2eCallback is not supplied),
    // the EOS vector is all zero. This is fine for models with proper
    // attention masking support, but may degrade accuracy otherwise.
    if (!engine.cacheEosEmbedding(eosEmbedding)) {
      __DEBUG("Failed to set the eos token embedding.");
      return false;
    }
  }

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

  size_t curTokenCount = embedding_vectors.size() / embedBufSize;
  if (_n_past + curTokenCount > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, curTokenCount, _ctx->size());
    throw genie::ContextLimitException("Context Size was exceeded.");
  }

  if (!engine.process(embedding_vectors, {}, logits))
    return Dialog::abort("engine prompt processing failed", callback);

  _n_prompt += curTokenCount;
  _n_past += curTokenCount;

  _prompt_len = _n_past;

  if (!engine.updateKV(_n_past)) return Dialog::abort("KV update failed", callback);

  std::vector<std::vector<int32_t>> streams;
  getTopK(logits, streams, _n_streams, _p_threshold, callback);

  _n_generated += streams.size();
  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  start.reset();

  bool status = processFollowOnGeneration(streams, logits, callback);

  _kpis.generate.update(start.elapsed_usec());

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return status;
}

}  // namespace qualla
