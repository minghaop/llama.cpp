//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "Trace.hpp"
#include "qualla/detail/timer.hpp"
#include "ssd-q1.hpp"

namespace fs = std::filesystem;
using qc     = qualla::Config;

namespace qualla {

SelfSpecDecDialog::SelfSpecDecDialog(std::shared_ptr<Env> env,
                                     const std::string& name,
                                     const nlohmann::json& conf)
    : Dialog(env, name, conf), _t_sampler(*_sampler["primary"]) {
  auto ssd_version = qc::optional<int>(conf, "ssd-version", 0);
  if (ssd_version > SelfSpecDecDialog::VERSION) __WARN("newer ssd-version in config!");

  _vocab = _ctx->n_vocab();

  if (!conf.contains("branches")) {
    __WARN("No branching specification provided in the config. Using default branching of [3]");
  } else {
    _branches.clear();
    for (const auto& item : conf["branches"]) {
      if (item.is_number_integer())
        _branches.push_back({item.get<size_t>()});
      else if (item.is_array())
        _branches.push_back(item.get<std::vector<size_t>>());
    }
  }

  _draft          = _branches.size();
  m_attention_map = gen_attention_map();

  _forecast_prefix       = qc::optional(conf, "forecast-prefix", _forecast_prefix);
  _forecast_token_offset = _vocab;
  _prefix_quant          = qc::optional(conf, "prefix-quant", _prefix_quant);

  _kv_prefix_name = qc::optional(conf, "forecast-prefix-name", _kv_prefix_name);

  _n_streams   = qc::optional<uint32_t>(conf, "n-streams", 1);
  _p_threshold = qc::optional<float>(conf, "p-threshold", 0.0);

  completeInit();
}

void SelfSpecDecDialog::completeInit() {
  if (m_initFinished) return;
  Dialog::completeInit();
  if (!_engine.empty()) {
    if (!_engine.contains("primary")) {
      State::fatal("\"primary\" engine not present in config!");
      return;
    }

    // Load KV prefix
    Timer timer;
    size_t n_restored_prefix = _engine["primary"]->restore(_kv_prefix_name, true);
    if (n_restored_prefix != _forecast_prefix + _prefix_quant) {
      throw std::runtime_error(fmt::format("SSD : Loaded {} KV$ from {} but expected {} KV$",
                                           n_restored_prefix,
                                           _kv_prefix_name,
                                           _forecast_prefix + _prefix_quant));
    }
    _n_past = _forecast_prefix + _prefix_quant;
    _kpis.restore.update(timer.elapsed_usec());
    m_initFinished = true;
  }
}

auto SelfSpecDecDialog::gen_forecast_tokens(int repeat) const {
  GENIE_TRACE();
  std::vector<int32_t> forecast_tokens(_draft, 0);
  std::iota(forecast_tokens.begin(), forecast_tokens.end(), _forecast_token_offset);

  std::vector<int32_t> ret;
  for (auto i = 0; i < repeat; ++i)
    ret.insert(ret.end(), forecast_tokens.begin(), forecast_tokens.end());
  return ret;
}

// Generate attention map based on _branches
std::vector<int32_t> SelfSpecDecDialog::gen_attention_map() {
  GENIE_TRACE();
  std::vector<int32_t> attention_tree = {-1};  // Initialize with root node

  // For each layer, generate branches from each preceding node
  size_t start_idx = 0;
  for (size_t d = 0; d < _draft; d++) {
    const size_t end_idx = attention_tree.size();
    const auto& branches = _branches[d];

    // Calculate the max #tokens to sample at each draft level
    m_samples_per_draft_level.push_back(*std::max_element(branches.begin(), branches.end()) + 1);

    // Iterate over all nodes in the previous branch
    for (size_t node_idx = start_idx, j = 0; node_idx < end_idx; node_idx++, j++) {
      const size_t branch_count = branches.size() > j ? branches[j] : branches.back();
      for (size_t c = 0; c < branch_count; c++) attention_tree.push_back(node_idx);
    }

    // Calculate the number of nodes at each draft level
    m_nodes_per_draft_level.push_back(attention_tree.size() - end_idx);
    start_idx = end_idx;
  }

  m_num_draft_nodes = attention_tree.size();

  // Attach _draft forecast tokens to each node
  const size_t end_idx = attention_tree.size();
  for (size_t node_idx = 0; node_idx < end_idx; node_idx++) {
    attention_tree.push_back(node_idx);
    for (size_t d = 1; d < _draft; d++) attention_tree.push_back(attention_tree.size() - 1);
  }

  return attention_tree;
}

std::vector<int32_t> SelfSpecDecDialog::build_sample_tree(int32_t last_token,
                                                          Tensor& logits,
                                                          size_t start_offset,
                                                          int32_t streamIdx) {
  GENIE_TRACE();
  // Initialize the tree with one fully-validated token
  std::vector<int32_t> tree = {last_token};

  size_t draft_level    = 0;     // Draft level currently being processed
  size_t draft_node_idx = 0;     // Node counter within the current draft level
  std::vector<int32_t> samples;  // Placeholder variable to store sampled tokens for each level
  size_t sample_idx = 0;         // Counter for number of samples consumed by current parent node

  // Iterate across the attention mask, until all draft levels have been processed
  // Note the stopping condition is not the same as the iteration variable. This is intentional
  for (size_t cur_idx = 1; draft_level < _draft; cur_idx++) {
    const int32_t parent_idx = m_attention_map[cur_idx];

    // Reset the counter for samples when processing a new parent node
    if (parent_idx != m_attention_map[cur_idx - 1]) {
      sample_idx = 0;
    }

    // Re-populate the samples on switching to a new draft level
    if (draft_node_idx == 0) {
      samples = sample_to_draft(
          logits, start_offset + draft_level, m_samples_per_draft_level[draft_level], streamIdx);
    }

    // Repeating tokens (back-to-back) are unlikely to occur, so disable this from the draft tree
    if (samples[sample_idx] == tree[static_cast<uint32_t>(parent_idx)]) {
      sample_idx++;
    }
    tree.push_back(samples[sample_idx++]);

    // Once all nodes at the current draft level have been processed, move to the next draft level
    if (++draft_node_idx >= m_nodes_per_draft_level[draft_level]) {
      draft_level++;
      draft_node_idx = 0;
    }
  }

  return tree;
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> SelfSpecDecDialog::verify_draft_tree(
    std::span<int32_t> draft_tree, Tensor& logits) {
  GENIE_TRACE();
  // Sample the root node to kick off the draft tree verification process
  std::vector<int32_t> accepted_ids    = {0};
  std::vector<int32_t> accepted_tokens = {sample_to_verify(logits, 0)};

  // Early exit if root node produces an EOS
  if (_ctx->is_eos(accepted_tokens.back())) {
    return {accepted_tokens, accepted_ids};
  }

  // Iterate across all nodes in the draft tree, and track accepted nodes/tokens
  // A node is "accepted" if it is a child of an accepted node, and it matches a verified token
  for (size_t cur_idx = 1; cur_idx < m_num_draft_nodes; cur_idx++) {
    const int32_t parent_idx = m_attention_map[cur_idx];
    if (parent_idx == accepted_ids.back() && draft_tree[cur_idx] == accepted_tokens.back()) {
      // This node has been accepted. Sample the associated logit for the next iteration
      accepted_tokens.push_back(sample_to_verify(logits, cur_idx));
      accepted_ids.push_back(static_cast<int32_t>(cur_idx));

      // Exit if sampled token is an EOS
      if (_ctx->is_eos(accepted_tokens.back())) break;
    }
  }

  return {accepted_tokens, accepted_ids};
}

int32_t SelfSpecDecDialog::sample_to_verify(Tensor& logits, size_t index) {
  Tensor indexedTensor = logits.getIndexedTensor(index, _vocab);
  return _t_sampler.process(indexedTensor);
}

std::vector<int32_t> SelfSpecDecDialog::sample_to_draft(Tensor& logits,
                                                        size_t index,
                                                        size_t count,
                                                        int32_t streamIdx) {
  GENIE_TRACE();
  Tensor indexedTensor = logits.getIndexedTensor(index, _vocab);
  switch (logits.getDataType()) {
    case TENSOR_DATATYPE_UFIXED_POINT_8: {
      applyPenalty<uint8_t>(indexedTensor, _t_sampler.getPenalty(), streamIdx);
      return topK<uint8_t>(indexedTensor, count);
    }
    case TENSOR_DATATYPE_UFIXED_POINT_16: {
      applyPenalty<uint16_t>(indexedTensor, _t_sampler.getPenalty(), streamIdx);
      return topK<uint16_t>(indexedTensor, count);
    }
    case TENSOR_DATATYPE_FLOAT_POINT_16: {
      applyPenalty<uint16_t>(indexedTensor, _t_sampler.getPenalty(), streamIdx);
      return topK<uint16_t>(indexedTensor, count);
    }
    case TENSOR_DATATYPE_FLOAT_32: {
      applyPenalty<float>(indexedTensor, _t_sampler.getPenalty(), streamIdx);
      return topK<float>(indexedTensor, count);
    }
    default: {
      __ERROR("Incorrect logits datatype.");
      break;
    }
  }
  return {};
}

void SelfSpecDecDialog::tileAttentionMask(const std::vector<int32_t>& mask,
                                          const std::vector<size_t>& streamIndices,
                                          const std::vector<size_t>& pastMap,
                                          const size_t prefixOffset,
                                          std::vector<int32_t>& tiledMask) {
  GENIE_TRACE();
  const size_t pastMapLen = pastMap.size();
  const int posVal = 1, negVal = 0;

  const size_t maskSize  = mask.size();
  const size_t numTokens = maskSize * streamIndices.size();

  const size_t rowLength = _n_past + numTokens;
  tiledMask.resize(numTokens * rowLength);

  for (size_t maskIdx = 0; maskIdx < streamIndices.size(); maskIdx++) {
    // Number of rows to skip to reach the current tile.
    const size_t tileOffset  = maskIdx * maskSize;
    int32_t* const tileStart = &tiledMask[tileOffset * rowLength + tileOffset + _n_past];
    for (size_t i = 0; i < maskSize; i++) {
      // Pointer to the start of row i of the current mask
      int32_t* rowPtr = &tiledMask[(tileOffset + i) * rowLength];
      // Skip kv-prefix attention for rows without speculative tokens.
      const int prefixFillVal = (i < prefixOffset) ? negVal : posVal;
      std::fill_n(rowPtr, _forecast_prefix, prefixFillVal);
      rowPtr += _forecast_prefix;
      // Always attend to prompt.
      std::fill_n(rowPtr, _n_prompt, posVal);
      rowPtr += _n_prompt;

      // Fill in the past valid tokens for this stream.
      for (const size_t& pastIdx : pastMap) {
        *rowPtr = (pastIdx == streamIndices[maskIdx]) ? posVal : negVal;
        rowPtr++;
      }

      // Clear the rest of the row. It will mostly consist of 0's.
      std::fill_n(rowPtr, rowLength - _n_prompt - _forecast_prefix - pastMapLen, negVal);
      // Move to the correct tile.
      rowPtr += tileOffset;
      // Translate the mask.
      int32_t tokenId = mask[i];
      if (tokenId > -1) {
        std::copy_n(tileStart + (static_cast<uint32_t>(tokenId) * rowLength), tokenId + 1, rowPtr);
      }
      // Always attend to self.
      rowPtr[i] = posVal;
    }
  }
}

// Takes a vector of tokens and produces a vector of embeddings via the provided T2E callback.
void SelfSpecDecDialog::convertTokensToEmbeddings(std::vector<int32_t>& tokens,
                                                  std::vector<uint8_t>& embeddings,
                                                  size_t embeddingBufferSize,
                                                  Dialog::T2ECallback t2eCallback) {
  for (auto& token : tokens) {
    std::vector<uint8_t> embedding(embeddingBufferSize, 0);
    t2eCallback(*this, token, embedding.data(), embeddingBufferSize);
    embeddings.insert(embeddings.end(), embedding.begin(), embedding.end());
  }
}

bool SelfSpecDecDialog::processFollowOnGeneration(std::vector<int32_t>& tokens,
                                                  Tensor& logits,
                                                  Dialog::Callback callback) {
  GENIE_TRACE();
  // Handles the printing of the subsequent generated tokens
  bool keep_generating = true;

  // A buffer for tokens to be decoded (one at a time, per the Middleware's request)
  std::vector<int32_t> decode_buf(1, 0);
  auto decode_token = [&](int32_t t) {
    if (!keep_generating) return;
    // Decode new token.
    // Return true to continue generation, and false otherwise
    decode_buf[0] = _last_tok = t;
    ++_n_generated;
    ++_n_decode;
    if (_ctx->is_eos(t)) {
      keep_generating = false;
      callback("", Sentence::END);
    } else {
      keep_generating = callback(_tokenizer->decode(decode_buf), Sentence::CONTINUE);
    }
    return;
  };
  // set decode_buf from prompt processing
  decode_buf[0] = _last_tok;

  auto& engine = *_engine["primary"];

  auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool> selected) {
    if (!engine.updateKV(past, selected)) return Dialog::abort("KV update failed", callback);
    return true;
  };

  // prepare the next inference
  if (m_processState != TOKEN_GEN) {
    const int32_t token = sample_to_verify(logits, 0);
    tokens              = build_sample_tree(token, logits, 1);
    decode_token(tokens[0]);
  }

  // Prepare constant options for next inferences
  const auto forecast_tokens = gen_forecast_tokens(m_num_draft_nodes);
  const auto attention_map   = m_attention_map;

  engine.set({{"kv-prefix-offset", m_num_draft_nodes}});

  std::vector<int32_t> accepted_counts(_draft + 1, 0);
  std::vector<bool> selected(attention_map.size(), false);

  while (!State::canceled() && keep_generating) {
    // Append forecast tokens
    tokens.insert(tokens.end(), forecast_tokens.begin(), forecast_tokens.end());
    if (_n_past + tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
      throw genie::ContextLimitException("Context Size was exceeded.");
    }

    size_t n_tok_t = 0;

    // Bifurcate based on embedding as input or token as input
    if (m_inputType == InputType::TOKENS) {
      n_tok_t = engine.process(tokens, attention_map, logits, true /* all logits */);
    } else if (m_inputType == InputType::EMBEDDINGS) {
      std::vector<uint8_t> embedding;
      convertTokensToEmbeddings(tokens, embedding, engine.getEmbeddingBufferSize(), m_t2eCallback);
      n_tok_t = engine.process(embedding, attention_map, logits, true /* all logits */);
    } else {
      return Dialog::abort("No valid Input Type is used", callback);
    }
    if (n_tok_t != tokens.size() && !m_pause) {
      return Dialog::abort("engine processing failed", callback);
    }

    if (n_tok_t != tokens.size() && m_pause) {
      // save and process again in next resume
      m_pause = false;
      tokens.erase(tokens.end() - static_cast<long>(forecast_tokens.size()), tokens.end());
      m_unprocessedTokens = tokens;
      m_processState      = TOKEN_GEN;
      return true;
    }

    // Accept tokens
    auto [accepted_tokens, accepted_ids] =
        verify_draft_tree(std::span{tokens.data(), tokens.size()}, logits);

    // Commit accepted tokens to kv-caches
    std::fill(selected.begin(), selected.end(), false);
    for (int32_t id : accepted_ids) {
      selected[static_cast<uint32_t>(id)] = true;
    }
    accepted_counts[accepted_tokens.size() - 1] += 1;

    for (uint32_t idx = 0; idx < accepted_tokens.size(); idx++) {
      engine.updateTokenCheckpoint(static_cast<uint32_t>(accepted_tokens[idx]), _n_past + idx);
      _t_sampler.updateSampledTokenHistory(accepted_tokens[idx]);
    }
    _n_past += accepted_tokens.size();
    update_kv(_n_past, selected);

    // Decode tokens
    std::for_each(accepted_tokens.begin(), accepted_tokens.end(), decode_token);

    // Prepare new tokens
    size_t next_draft_offset =
        m_num_draft_nodes + static_cast<uint32_t>(accepted_ids.back()) * _draft;
    tokens = build_sample_tree(accepted_tokens.back(), logits, next_draft_offset);

    if (m_pause && keep_generating) {
      // save tokens for next execution
      m_pause             = false;
      m_unprocessedTokens = tokens;
      m_processState      = TOKEN_GEN;
      return true;
    }
  }

  State::busy(false);

  auto total_iteration = std::accumulate(accepted_counts.begin(), accepted_counts.end(), 0);
  auto accept_rate =
      float(_n_generated - 1) / total_iteration;  // -1: exclude first generated token
  _kpis.tps.tokenAcceptance = accept_rate;
  __KPIS(
      "SSD{{draft:{}, branch:{}, greedy:{}}}: accepted counts: {}, accept rate = {} "
      "tokens/iteration",
      _draft,
      _branches,
      _t_sampler.greedy(),
      accepted_counts,
      accept_rate);
  return true;
}

// Multistream AR generation
bool SelfSpecDecDialog::processFollowOnGeneration(std::vector<std::vector<int32_t>>& streams,
                                                  Tensor& logits,
                                                  Dialog::Callback callback) {
  GENIE_TRACE();
  auto& engine = *_engine["primary"];

  auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool> selected) {
    if (!engine.updateKV(past, selected)) return Dialog::abort("KV update failed", callback);
    return true;
  };

  std::vector<size_t> streamIndices(streams.size());
  std::vector<size_t> past_map(streams.size());

  std::iota(streamIndices.begin(), streamIndices.end(), 0);
  // Since the first inference is done separately, it is
  // expected that each stream already has 1 valid AR token.
  std::iota(past_map.begin(), past_map.end(), 0);
  // Add generated token count from first inference.
  _n_generated += streams.size();
  _n_decode += streams.size();

  if (streams.size() == 0) {
    callback("\n", Sentence::END);
    return true;
  }

  // Prepare constant options for next inferences
  const auto forecast_tokens = gen_forecast_tokens(m_num_draft_nodes);
  const auto attention_map   = m_attention_map;

  std::vector<std::vector<int32_t>> draftStreams(streams.size());

  std::vector<int32_t> accepted_counts(_draft + 1, 0);
  std::vector<int32_t> multi_attn_mask;

  for (size_t i = 0; i < streams.size(); i++) {
    // prepare the next inference
    draftStreams[i] = build_sample_tree(sample_to_verify(logits, i * (1 + _draft)), logits, 1, i);
    streams[i].push_back(draftStreams[i][0]);
  }

  engine.set({{"kv-prefix-offset", m_num_draft_nodes}});

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
    std::vector<int32_t> multi_tokens;
    for (auto streamIdx : streamIndices) {
      multi_tokens.insert(
          multi_tokens.end(), draftStreams[streamIdx].begin(), draftStreams[streamIdx].end());
      multi_tokens.insert(multi_tokens.end(), forecast_tokens.begin(), forecast_tokens.end());
    }

    if (_n_past + multi_tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, multi_tokens.size(), _ctx->size());
      throw genie::ContextLimitException("Context Size was exceeded.");
    }

    tileAttentionMask(attention_map, streamIndices, past_map, m_num_draft_nodes, multi_attn_mask);

    size_t n_tok_t = 0;

    if (m_inputType == InputType::TOKENS) {
      // Process input tokens for all streams in one batch
      n_tok_t = engine.process(multi_tokens, multi_attn_mask, logits, true);
    } else if (m_inputType == InputType::EMBEDDINGS) {
      // Accumulate input embeddings from all streams
      auto embedBufSize = engine.getEmbeddingBufferSize();
      std::vector<uint8_t> multi_embeddings;

      convertTokensToEmbeddings(multi_tokens, multi_embeddings, embedBufSize, m_t2eCallback);

      // Process input tokens for all streams in one batch
      n_tok_t = engine.process(multi_embeddings, multi_attn_mask, logits, true);
    }
    if (n_tok_t != multi_tokens.size()) return Dialog::abort("engine processing failed", callback);

    std::vector<bool> all_selected;

    // Process all logits independently
    std::span<int32_t> token_span = std::span{multi_tokens.data(), multi_tokens.size()};
    for (size_t i = 0; i < streamIndices.size(); i++) {
      const size_t streamIdx       = streamIndices[i];
      std::vector<int32_t>& stream = streams[streamIdx];

      const size_t tileStride = draftStreams[streamIdx].size() + forecast_tokens.size();

      Tensor tiled_logits = logits.getIndexedTensor(i * tileStride, _vocab);

      // Accept tokens
      auto [accepted_tokens, accepted_ids] =
          verify_draft_tree(token_span.subspan(i * tileStride, tileStride), tiled_logits);

      // Commit accepted tokens to kv-caches
      std::vector<bool> selected(tileStride, false);
      for (int32_t id : accepted_ids) {
        selected[static_cast<uint32_t>(id)] = true;
        past_map.push_back(streamIdx);
      }
      all_selected.insert(all_selected.end(), selected.begin(), selected.end());
      accepted_counts[accepted_tokens.size() - 1] += 1;
      _n_past += accepted_tokens.size();

      // Decode tokens
      stream.insert(stream.end(), accepted_tokens.begin(), accepted_tokens.end());
      _n_generated += accepted_tokens.size();
      _n_decode += accepted_tokens.size();

      // Prepare new tokens
      size_t next_draft_offset =
          m_num_draft_nodes + static_cast<uint32_t>(accepted_ids.back()) * _draft;
      draftStreams[streamIdx] =
          build_sample_tree(accepted_tokens.back(), tiled_logits, next_draft_offset, streamIdx);
      _t_sampler.updateSampledTokenHistory(accepted_tokens, static_cast<int32_t>(streamIdx));
    }

    update_kv(_n_past, all_selected);
    for (auto it = streamIndices.begin(); it != streamIndices.end();) {
      if (_ctx->is_eos(streams[*it].back())) {
        callback(_tokenizer->decode(streams[*it]) + "\n", Sentence::CONTINUE);
        it = streamIndices.erase(it);
      } else {
        ++it;
      }
    }

    if (streamIndices.size() == 0) break;
  }
  callback("\n", Sentence::END);

  State::busy(false);

  auto total_iteration = std::accumulate(accepted_counts.begin(), accepted_counts.end(), 0);
  auto accept_rate =
      float(_n_generated - 1) / total_iteration;  // -1: exclude first generated token
  _kpis.tps.tokenAcceptance = accept_rate;
  __KPIS(
      "SSD{{draft:{}, branch:{}, greedy:{}}}: accepted counts: {}, accept rate = {} "
      "tokens/iteration",
      _draft,
      _branches,
      _t_sampler.greedy(),
      accepted_counts,
      accept_rate);

  return true;
}

// Handle prompt processing and generation will be done processFollowOnGeneration
// Pass t2e callback using setter and remove as an argument. call setter from the base query
// function of dialog

bool SelfSpecDecDialog::process(std::vector<uint8_t>& embedding,
                                T2ECallback t2eCallback,
                                Dialog::Callback callback) {
  GENIE_TRACE();
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  if (m_inputType != InputType::EMBEDDINGS) {
    __ERROR("Input type for model is not embeddings.");
    return false;
  }

  Timer start;
  State::clear();

  Tensor logits;
  auto& engine = *_engine["primary"];

  auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool> selected) {
    if (!engine.updateKV(past, selected)) return Dialog::abort("KV update failed", callback);
    return true;
  };

  // Store the t2e callback for reference during follow-on generation.
  m_t2eCallback = t2eCallback;

  auto embedBufSize = engine.getEmbeddingBufferSize();

  {
    std::vector<uint8_t> eosEmbedding(embedBufSize, 0.0);
    if (m_t2eCallback) {
      m_t2eCallback(*this, _ctx->eos(), eosEmbedding.data(), embedBufSize);
    }
    if (!engine.cacheEosEmbedding(eosEmbedding)) {
      __DEBUG("Failed to set the eos token embedding.");
      return false;
    }
  }

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

  __KPIS("{}", kpis().dump(" "));
  start.reset();

  engine.set({{"kv-prefix-skip", _forecast_prefix}});

  std::vector<int32_t> tokens(1, 0);

  bool keepProcessing = false;
  if (m_processState == NO_RESUME || m_processState == PROMPT_PROCESSING) {
    keepProcessing = true;
    // Process prompt
    // get number of tokens in the input
    size_t curTokensCount = embedding.size() / embedBufSize;

    if (curTokensCount * embedBufSize != embedding.size()) {
      size_t expectedLength =
          (curTokensCount + (embedding.size() % embedBufSize != 0)) * embedBufSize;
      __DEBUG("Input is wrong expected {} and found {}.", expectedLength, embedding.size());
      return Dialog::abort("Input is not an multiple for the embedding Length", callback);
    }

    _n_prompt += curTokensCount;

    engine.set({{"kv-prefix-offset", curTokensCount}});  // Do not attend prefix

    if (_n_past + curTokensCount > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, curTokensCount, _ctx->size());
      throw genie::ContextLimitException("Context Size was exceeded.");
    }

    size_t numProcessed = engine.process(embedding, {}, logits, false);

    if (!numProcessed) {
      return Dialog::abort("engine prompt processing failed",
                           callback);  // Change this message also to some generic message.
    }

    if (m_pause && numProcessed != 1) {
      m_pause = false;
      _n_past += numProcessed;
      update_kv(_n_past, {});
      for (size_t idx = numProcessed * embedBufSize; idx < embedding.size(); idx++) {
        m_unprocessedEmbedding.push_back(embedding[idx]);
      }
      _n_prompt -= m_unprocessedEmbedding.size() / embedBufSize;
      m_processState = PROMPT_PROCESSING;
      return true;
    }
    _n_past += curTokensCount;
    update_kv(_n_past, {});
  }
  bool status = true;
  if (_n_streams <= 1) {
    if (keepProcessing) {
      tokens[0] = sample_to_verify(logits, 0);
      _t_sampler.updateSampledTokenHistory(tokens[0]);
      m_unprocessedTokens = tokens;

      // Decode the first token.
      _last_tok = tokens[0];
      if (_ctx->is_eos(_last_tok)) {
        callback("", Sentence::END);
        return true;
      }

      if (!callback(_tokenizer->decode(tokens), Sentence::BEGIN)) return true;
      _n_generated++;

      if (!m_t2eCallback) {
        callback("", Sentence::END);
        return true;
      }
    }

    if (m_pause) {
      m_pause        = false;
      m_processState = FIRST_TOKEN_GEN;
      return true;
    }

    // Mark TTFT
    _kpis.prompt.update(start.elapsed_usec());
    start.reset();
    State::busy(true);

    if (keepProcessing || m_processState == FIRST_TOKEN_GEN) {
      keepProcessing = true;
      if (m_processState == FIRST_TOKEN_GEN) {
        tokens = m_unprocessedTokens;
      }
      // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
      // process separately because logits are required for these tokens
      for (size_t i = 0; i < _draft; ++i) {
        tokens.push_back(_forecast_token_offset + i);
      }

      engine.set({{"kv-prefix-offset", 1}});  // Prevent the last token from attending

      if (_n_past + tokens.size() > _ctx->size()) {
        __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
        throw genie::ContextLimitException("Context Size was exceeded.");
      }

      // Convert tokens to embeddings
      // reset embedding vector to make space for the next runs
      embedding.clear();
      convertTokensToEmbeddings(tokens, embedding, embedBufSize, m_t2eCallback);

      if (!engine.process(embedding, {}, logits, true))
        return Dialog::abort("initial inference for SSD pipeline failed", callback);

      if (m_pause) {
        m_pause        = false;
        m_processState = FIRST_TOKEN_GEN;
        return true;
      }

      _n_past += 1;
      update_kv(_n_past, {});
    }
    if (keepProcessing || m_processState == TOKEN_GEN) {
      if (m_processState == TOKEN_GEN) {
        tokens = m_unprocessedTokens;
      }
      // Use existing as much as possible
      status = processFollowOnGeneration(tokens, logits, callback);
    }
  } else {
    std::vector<std::vector<int32_t>> streams;
    getTopK(logits, streams, _n_streams, _p_threshold, callback);
    _n_generated += streams.size();

    if (!m_t2eCallback) {
      for (auto& stream : streams) {
        if (!callback(_tokenizer->decode(stream) + "\n", Sentence::BEGIN)) return true;
      }
      callback("", Sentence::END);
      return true;
    }

    // Mark TTFT
    _kpis.prompt.update(start.elapsed_usec());
    start.reset();
    State::busy(true);

    if (streams.size() == 0) {
      callback("\n", Sentence::END);
      return true;
    }

    // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
    // process separately because logits are required for these tokens
    std::vector<int32_t> attention_map(1 + _draft);
    std::iota(attention_map.begin(), attention_map.end(), -1);

    std::vector<size_t> stream_indices(streams.size());
    std::iota(stream_indices.begin(), stream_indices.end(), 0);

    std::vector<int32_t> multi_attn_mask;
    std::vector<size_t> past_map;
    const size_t kvPrefixOffset = 1;

    tileAttentionMask(attention_map, stream_indices, past_map, kvPrefixOffset, multi_attn_mask);

    // Accumulate input tokens from all streams
    std::vector<int32_t> multi_tokens;

    multi_tokens.reserve(streams.size() * (1 + _draft));
    for (size_t i = 0; i < streams.size(); i++) {
      multi_tokens.insert(multi_tokens.end(), streams[i].begin(), streams[i].end());
      for (size_t j = 0; j < _draft; j++) {
        multi_tokens.push_back(_forecast_token_offset + j);
      }
    }

    // Convert tokens to embeddings
    // reset embedding vector to make space for the next runs
    embedding.clear();
    convertTokensToEmbeddings(multi_tokens, embedding, embedBufSize, m_t2eCallback);

    if (_n_past + multi_tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, multi_tokens.size(), _ctx->size());
      throw genie::ContextLimitException("Context Size was exceeded.");
    }

    if (!engine.process(embedding, multi_attn_mask, logits, true))
      return Dialog::abort("initial inference for SSD pipeline failed", callback);

    std::vector<bool> selected(multi_tokens.size(), false);
    for (size_t i = 0; i < multi_tokens.size(); i += (_draft + 1)) {
      selected[i] = true;
    }

    _n_past += streams.size();
    update_kv(_n_past, selected);

    status = processFollowOnGeneration(streams, logits, callback);
  }

  _kpis.generate.update(start.elapsed_usec());
  __KPIS("{}", kpis().dump(" "));
  start.reset();

  return status;
}

bool SelfSpecDecDialog::process(std::vector<int32_t>& tokens, Dialog::Callback callback) {
  GENIE_TRACE();
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  if (m_inputType != InputType::TOKENS) {
    __ERROR("Input type for model is not tokens.");
    return false;
  }

  State::clear();

  Tensor logits;
  auto& engine = *_engine["primary"];

  auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool> selected) {
    if (!engine.updateKV(past, selected)) return Dialog::abort("KV update failed", callback);
    return true;
  };

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

  __KPIS("{}", kpis().dump(" "));
  start.reset();

  engine.set({{"kv-prefix-skip", _forecast_prefix}});
  bool keepProcessing = false;
  if (m_processState == NO_RESUME || m_processState == PROMPT_PROCESSING) {
    // Process prompt
    keepProcessing = true;
    _n_prompt += tokens.size();
    // TODO: set to kv-prefix-offset _n_prompt?
    engine.set({{"kv-prefix-offset", tokens.size()}});  // Do not attend prefix

    if (_n_past + tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
      throw genie::ContextLimitException("Context Size was exceeded.");
    }

    size_t numProcessedTokens = engine.process(tokens, logits, false);
    if (!numProcessedTokens) return Dialog::abort("engine prompt processing failed", callback);

    if (m_pause && numProcessedTokens != 1) {
      m_pause = false;
      for (uint32_t idx = 0; idx < numProcessedTokens; idx++) {
        engine.updateTokenCheckpoint(static_cast<uint32_t>(tokens[idx]), _n_past + idx);
      }
      _n_past += numProcessedTokens;
      update_kv(_n_past, {});
      for (size_t idx = numProcessedTokens; idx < tokens.size(); idx++) {
        m_unprocessedTokens.push_back(tokens[idx]);
      }
      _n_prompt -= m_unprocessedTokens.size();
      m_processState = PROMPT_PROCESSING;
      return true;
    }

    for (uint32_t idx = 0; idx < tokens.size(); idx++) {
      engine.updateTokenCheckpoint(static_cast<uint32_t>(tokens[idx]), _n_past + idx);
    }
    _n_past += tokens.size();
    update_kv(_n_past, {});
  }

  bool status = true;
  if (_n_streams <= 1) {
    if (keepProcessing) {
      // print the Token for the boundary KV rewind as it is next generated token
      auto sCode = Sentence::BEGIN;
      if (m_rewindAtBoundary) {
        _n_prompt -= 1;
        if (!callback(_tokenizer->decode(tokens), sCode)) return true;
        _n_generated++;
        sCode = Sentence::CONTINUE;
      }

      tokens[0] = sample_to_verify(logits, 0);
      tokens.resize(1);
      m_unprocessedTokens = tokens;

      _last_tok = tokens[0];
      if (_ctx->is_eos(_last_tok)) {
        callback("", Sentence::END);
        return true;
      }

      if (!callback(_tokenizer->decode(tokens), sCode)) return true;
      _n_generated++;
    }

    if (m_pause) {
      m_pause        = false;
      m_processState = FIRST_TOKEN_GEN;
      return true;
    }

    // Mark TTFT
    _kpis.prompt.update(start.elapsed_usec());
    start.reset();
    State::busy(true);

    if (keepProcessing || m_processState == FIRST_TOKEN_GEN) {
      keepProcessing = true;

      // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
      // process separately because logits are required for these tokens
      for (size_t i = 0; i < _draft; ++i) {
        tokens.push_back(_forecast_token_offset + i);
      }
      engine.set({{"kv-prefix-offset", 1}});  // Prevent the last token from attending

      if (_n_past + tokens.size() > _ctx->size()) {
        __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
        throw genie::ContextLimitException("Context Size was exceeded.");
      }

      if (!engine.process(tokens, logits, true))
        return Dialog::abort("initial inference for SSD pipeline failed", callback);

      if (m_pause) {
        m_pause        = false;
        m_processState = FIRST_TOKEN_GEN;
        return true;
      }

      engine.updateTokenCheckpoint(static_cast<uint32_t>(tokens[0]), _n_past);
      _n_past += 1;
      update_kv(_n_past, {});
    }
    if (keepProcessing || m_processState == TOKEN_GEN) {
      status = processFollowOnGeneration(tokens, logits, callback);
    }
  } else {
    std::vector<std::vector<int32_t>> streams;
    getTopK(logits, streams, _n_streams, _p_threshold, callback);
    _n_generated += streams.size();

    // Mark TTFT
    _kpis.prompt.update(start.elapsed_usec());
    start.reset();
    State::busy(true);

    if (streams.size() == 0) {
      callback("\n", Sentence::END);
      return true;
    }

    // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
    // process separately because logits are required for these tokens
    std::vector<int32_t> attention_map(1 + _draft);
    std::iota(attention_map.begin(), attention_map.end(), -1);

    std::vector<size_t> stream_indices(streams.size());
    std::iota(stream_indices.begin(), stream_indices.end(), 0);

    std::vector<int32_t> multi_attn_mask;
    std::vector<size_t> past_map;
    const size_t kvPrefixOffset = 1;

    tileAttentionMask(attention_map, stream_indices, past_map, kvPrefixOffset, multi_attn_mask);

    // Accumulate input tokens from all streams
    std::vector<int32_t> multi_tokens;

    multi_tokens.reserve(streams.size() * (1 + _draft));
    for (size_t i = 0; i < streams.size(); i++) {
      multi_tokens.insert(multi_tokens.end(), streams[i].begin(), streams[i].end());
      for (size_t j = 0; j < _draft; ++j) {
        multi_tokens.push_back(_forecast_token_offset + j);
      }
    }

    if (_n_past + multi_tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, multi_tokens.size(), _ctx->size());
      throw genie::ContextLimitException("Context Size was exceeded.");
    }

    if (!engine.process(multi_tokens, multi_attn_mask, logits, true)) {
      return Dialog::abort("initial inference for SSD pipeline failed", callback);
    }

    std::vector<bool> selected(multi_tokens.size(), false);
    for (size_t i = 0; i < multi_tokens.size(); i += (_draft + 1)) {
      selected[i] = true;
    }

    _n_past += streams.size();
    update_kv(_n_past, selected);

    status = processFollowOnGeneration(streams, logits, callback);
  }

  _kpis.generate.update(start.elapsed_usec());
  __KPIS("{}", kpis().dump(" "));
  start.reset();

  return status;
}

void SelfSpecDecDialog::reset() {
  Dialog::reset();
  _n_past                  = _forecast_prefix + _prefix_quant;
  size_t n_restored_prefix = _engine["primary"]->restore(_kv_prefix_name, true);
  if (n_restored_prefix != _forecast_prefix + _prefix_quant) {
    throw std::runtime_error(fmt::format("SSD : Loaded {} KV$ from {} but expected {} KV$",
                                         n_restored_prefix,
                                         _kv_prefix_name,
                                         _forecast_prefix + _prefix_quant));
  }
}

bool SelfSpecDecDialog::save(const std::string& name) {
  if (_n_streams > 1) {
    throw std::runtime_error("Save is unsupported for multistream dialogs.");
  }
  return Dialog::save(name);
}

bool SelfSpecDecDialog::restore(const std::string& name) {
  if (_n_streams > 1) {
    throw std::runtime_error("Restore is unsupported for multistream dialogs.");
  }
  return Dialog::restore(name);
}

}  // namespace qualla
