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
#include "basic.hpp"
#include "qualla/detail/timer.hpp"

namespace fs = std::filesystem;

namespace qualla {

BasicDialog::BasicDialog(std::shared_ptr<Env> env,
                         const std::string& name,
                         const nlohmann::json& conf)
    : Dialog(env, name, conf) {
  completeInit();
}

void BasicDialog::completeInit() {
  if (m_initFinished) return;
  Dialog::completeInit();
  if (!_engine.empty()) {
    if (!_engine.contains("primary")) {
      State::fatal("\"primary\" engine not present in config!");
      return;
    }
    m_initFinished = true;
  }
}

bool BasicDialog::processFollowOnGeneration(std::vector<int32_t>& tokens,
                                            Tensor& logits,
                                            Dialog::Callback callback) {
  qualla::DialogCallback dialogCallbackWrapper;
  dialogCallbackWrapper.setCallBackType(qualla::QUALLA_CALLBACK_TYPE_TEXT);
  dialogCallbackWrapper.getQueryCbFunc() =
      std::make_shared<std::function<bool(const std::string&, qualla::Sentence::Code)>>();
  *(dialogCallbackWrapper.getQueryCbFunc()) = callback;
  return processFollowOnGeneration(tokens, logits, dialogCallbackWrapper);
}

bool BasicDialog::process(std::vector<int32_t>& tokens, Dialog::Callback callback) {
  qualla::DialogCallback dialogCallbackWrapper;
  dialogCallbackWrapper.setCallBackType(qualla::QUALLA_CALLBACK_TYPE_TEXT);
  dialogCallbackWrapper.getQueryCbFunc() =
      std::make_shared<std::function<bool(const std::string&, qualla::Sentence::Code)>>();
  *(dialogCallbackWrapper.getQueryCbFunc()) = callback;
  return process(tokens, dialogCallbackWrapper);
}

bool BasicDialog::process(std::vector<int32_t>& tokens, std::vector<size_t>& tokenNumPerBatch, Dialog::BatchCallback callback) {
  qualla::DialogCallback dialogCallbackWrapper;
  dialogCallbackWrapper.setCallBackType(qualla::QUALLA_CALLBACK_TYPE_BATCH_TEXT);
  dialogCallbackWrapper.getBatchQueryCbFunc() =
      std::make_shared<std::function<bool(const std::vector<std::string>&, const std::vector<bool>&, qualla::Sentence::Code)>>();
  *(dialogCallbackWrapper.getBatchQueryCbFunc()) = callback;
  return process(tokens, tokenNumPerBatch, dialogCallbackWrapper);
}

bool BasicDialog::processFollowOnGeneration(std::vector<int32_t>& tokens,
                                            Tensor& logits,
                                            qualla::DialogCallback callback) {
  GENIE_TRACE();
  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

  while (true) {
    if (State::canceled()) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      break;
    }
    // This condition is valid for both tokens and embedding
    if (_n_past + 1 > _ctx->size()) {
      __WARN("Context limit exceeded ({} + 1 > {})", _n_past, _ctx->size());
      throw genie::ContextLimitException("Context Size was exceeded.");
    }
    if (m_inputType == InputType::TOKENS) {
      if (engine.process(tokens, logits, false) != 1 || engine.failed())
        return Dialog::abort("Engine processing failed. " + engine.error(), callback);
    } else if (m_inputType == InputType::EMBEDDINGS) {
      // Convert tokens to embedding for the processing in the engine.
      auto embedBufSize = engine.getEmbeddingBufferSize();
      std::vector<uint8_t> embedding;
      for (auto& token : tokens) {
        std::vector<uint8_t> curTokenEmbedding(embedBufSize, 0);
        m_t2eCallback(*this, token, curTokenEmbedding.data(), embedBufSize);
        embedding.insert(embedding.end(), curTokenEmbedding.begin(), curTokenEmbedding.end());
      }
      if (engine.process(embedding, {}, logits, false) != 1 || engine.failed())
        return Dialog::abort("Engine processing failed. " + engine.error(), callback);
    } else {
      return Dialog::abort("No valid Input Type is used", callback);
    }
    tokens[0] = _last_tok = sampler.process(logits);
    sampler.updateSampledTokenHistory(tokens[0]);

    _n_past++;
    _n_generated++;
    _n_decode++;
    engine.updateTokenCheckpoint(static_cast<uint32_t>(_last_tok), _n_past);
    if (!engine.updateKV(_n_past)) return Dialog::abort("KV update failed", callback);

    if (_ctx->is_eos(_last_tok)) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      break;
    }

    if (!callback.callBack(tokens.data(), tokens.size(), Sentence::CONTINUE, tokenizer())) break;

    if (m_pause) {
      // save tokens for next execution
      m_pause             = false;
      m_unprocessedTokens = tokens;
      m_processState      = TOKEN_GEN;
      return true;
    }
  }

  return true;
}

bool BasicDialog::processFollowOnGeneration(std::vector<int32_t>& tokens,
                                            std::vector<Tensor>& logits,
                                            qualla::DialogCallback callback) {
  GENIE_TRACE();
  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

  std::vector<bool> allBatchesFinished(tokens.size(), false);

  while (true) {
    bool allFinished = true;

    if (State::canceled()) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      break;
    }

    std::vector<size_t> tokenNumPerBatch(tokens.size(), 1);
    // This condition is valid for both tokens and embedding
    if (_n_past + 1 > _ctx->size()) {
      __WARN("Context limit exceeded ({} + 1 > {})", _n_past, _ctx->size());
      throw genie::ContextLimitException("Context Size was exceeded.");
    }
    if (m_inputType == InputType::TOKENS) {
      if (engine.process(tokens, tokenNumPerBatch, logits, false) != tokens.size() || engine.failed()){
        return Dialog::abort("Engine processing failed. " + engine.error(), callback);
      }
    } else if (m_inputType == InputType::EMBEDDINGS) {
      return Dialog::abort("Multi-batch processing currently does not support EMBEDDINGS input type. Use TOKENS input type or switch to single-batch mode.", callback);
    } else {
      return Dialog::abort("No valid Input Type is used", callback);
    }

    tokens = sampler.process(logits, tokenNumPerBatch.size());
    sampler.updateSampledTokenHistory(tokens,tokens.size());
    _n_past++;

    if (!engine.updateKV(_n_past)) return Dialog::abort("KV update failed", callback);

    for (size_t i = 0; i < tokens.size(); i++) {
      if (_ctx->is_eos(tokens[i]) && (!allBatchesFinished[i])) {
        allBatchesFinished[i] = true;
        _n_generated++;
        _n_decode++;
        __DEBUG("Batch {} finished (EOS).", i);
      }
      if (!allBatchesFinished[i]) {
        allFinished = false;
        _n_generated++;
        _n_decode++;
      }
    }

    callback.setBatchFinished(allBatchesFinished);

    const Sentence::Code code = allFinished ? Sentence::END : Sentence::CONTINUE;

    if (!callback.callBack(tokens.data(), tokens.size(), code, tokenizer())) {
      break;
    }

    if (allFinished) {
      return true;
    }

  }

  return true;
}

bool BasicDialog::process(std::vector<int32_t>& tokens, qualla::DialogCallback callback) {
  GENIE_TRACE();
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  if (m_inputType != InputType::TOKENS) {
    __ERROR("Input type for model is not tokens.");
    return false;
  }

  _gpio_marker->set();

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor logits;

  State::clear();

  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

  bool keepProcessing = false;
  if (m_processState == NO_RESUME || m_processState == PROMPT_PROCESSING) {
    keepProcessing = true;

    if (_n_past + tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
      throw genie::ContextLimitException("Context Size was exceeded.");
    }

    const size_t n_engine_returned = engine.process(tokens, logits, false);
    if ((n_engine_returned != 1 && !m_pause) || engine.failed()) {
      __ERROR("Engine processing failed. Engine returned {} logits. Failed={} Error={}",
              n_engine_returned,
              engine.failed(),
              engine.error());
      return Dialog::abort("Engine prompt processing failed. " + engine.error(), callback);
    }

    if (m_pause && n_engine_returned != 1) {
      m_pause = false;
      for (uint32_t idx = 0; idx < n_engine_returned; idx++) {
        engine.updateTokenCheckpoint(static_cast<uint32_t>(tokens[idx]), _n_past + idx);
      }
      _n_past += n_engine_returned;
      engine.updateKV(_n_past);
      for (size_t idx = n_engine_returned; idx < tokens.size(); idx++) {
        m_unprocessedTokens.push_back(tokens[idx]);
      }
      _n_prompt += n_engine_returned;
      m_processState = PROMPT_PROCESSING;
      return true;
    }

    for (uint32_t idx = 0; idx < tokens.size(); idx++) {
      engine.updateTokenCheckpoint(static_cast<uint32_t>(tokens[idx]), _n_past + idx);
    }
    _n_prompt += tokens.size();
    _n_past += tokens.size();

    if (!engine.updateKV(_n_past) || engine.failed())
      return Dialog::abort("KV cache update failed. " + engine.error(), callback);
  }
  auto sCode = Sentence::BEGIN;
  if (keepProcessing) {
    // print the Token for the boundary KV rewind as it is next generated token
    if (m_rewindAtBoundary) {
      _n_prompt -= 1;
      if (!callback.callBack(tokens.data(), tokens.size(), sCode, tokenizer())) return true;
      _n_generated++;
      sCode = Sentence::CONTINUE;
    }

    tokens[0] = _last_tok = sampler.process(logits);
    sampler.updateSampledTokenHistory(tokens[0]);
    tokens.resize(1);
    m_unprocessedTokens = tokens;

    engine.updateTokenCheckpoint(static_cast<uint32_t>(_last_tok), _n_past);
    _n_generated++;
  }

  if (m_pause) {
    m_pause        = false;
    m_processState = TOKEN_GEN;
    return true;
  }

  _gpio_marker->set();

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  start.reset();

  if (keepProcessing || m_processState == TOKEN_GEN) {
    if (_ctx->is_eos(_last_tok)) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      return true;
    }

    if (!callback.callBack(tokens.data(), tokens.size(), sCode, tokenizer())) return true;

    State::busy(true);
    processFollowOnGeneration(tokens, logits, callback);
  }
  State::busy(false);

  _gpio_marker->set();
  _gpio_marker->reset();

  _kpis.generate.update(start.elapsed_usec());

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return !State::failed();
}

bool BasicDialog::process(std::vector<uint8_t>& embedding_vectors,
                          T2ECallback t2eCallback,
                          Dialog::Callback callback) {
  qualla::DialogCallback dialogCallbackWrapper;
  dialogCallbackWrapper.setCallBackType(qualla::QUALLA_CALLBACK_TYPE_TEXT);
  dialogCallbackWrapper.getQueryCbFunc() =
      std::make_shared<std::function<bool(const std::string&, qualla::Sentence::Code)>>();
  *(dialogCallbackWrapper.getQueryCbFunc()) = callback;
  return process(embedding_vectors, t2eCallback, dialogCallbackWrapper);
}

bool BasicDialog::process(std::vector<uint8_t>& embedding_vectors,
                          T2ECallback t2eCallback,
                          qualla::DialogCallback callback) {
  GENIE_TRACE();
  Timer start;
  if (m_inputType != InputType::EMBEDDINGS) {
    __ERROR("Input type for model is not embeddings.");
    return false;
  }

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor logits;

  State::clear();

  _gpio_marker->set();

  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

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
  __KPIS("{}", kpis().dump(" "));
  start.reset();  // Don't include preprocessing time
  std::vector<int32_t> tokens(1, 0);
  bool keepProcessing = false;
  if (m_processState == NO_RESUME || m_processState == PROMPT_PROCESSING) {
    keepProcessing = true;
    if (_n_past + curTokenCount > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, curTokenCount, _ctx->size());
      throw genie::ContextLimitException("Context Size was exceeded.");
    }

    size_t numProcessed = engine.process(embedding_vectors, {}, logits);
    if (!numProcessed) return Dialog::abort("engine prompt processing failed", callback);

    if (m_pause && numProcessed != 1) {
      m_pause = false;
      _n_past += numProcessed;
      engine.updateKV(_n_past);
      for (size_t idx = numProcessed * embedBufSize; idx < embedding_vectors.size(); idx++) {
        m_unprocessedEmbedding.push_back(embedding_vectors[idx]);
      }
      _n_prompt += numProcessed;
      m_processState = PROMPT_PROCESSING;
      return true;
    }

    if (engine.usesCrossAttention()) {
      _n_prompt += numProcessed;
      _n_past += numProcessed;
    } else {
      _n_prompt += curTokenCount;
      _n_past += curTokenCount;
    }

    if (!engine.updateKV(_n_past)) return Dialog::abort("KV update failed", callback);
  }

  if (keepProcessing) {
    tokens[0] = _last_tok = sampler.process(logits);
    sampler.updateSampledTokenHistory(tokens[0]);
    m_unprocessedTokens = tokens;

    _n_generated++;
  }

  if (m_pause) {
    m_pause        = false;
    m_processState = TOKEN_GEN;
    return true;
  }

  _gpio_marker->set();

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  start.reset();

  if (keepProcessing || m_processState == TOKEN_GEN) {
    if (m_processState == TOKEN_GEN) {
      tokens = m_unprocessedTokens;
    }

    if (_ctx->is_eos(_last_tok)) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      return true;
    }

    if (!callback.callBack(tokens.data(), tokens.size(), Sentence::BEGIN, tokenizer())) {
      return true;
    }

    if (!m_t2eCallback) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      return true;
    }

    State::busy(true);
    processFollowOnGeneration(tokens, logits, callback);
  }

  State::busy(false);

  _gpio_marker->set();
  _gpio_marker->reset();

  _kpis.generate.update(start.elapsed_usec());
  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return !State::failed();
}

bool BasicDialog::process(std::vector<int32_t>& tokens, std::vector<size_t>& tokenNumPerBatch, qualla::DialogCallback callback) {
  GENIE_TRACE();
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  if (m_inputType != InputType::TOKENS) {
    __ERROR("Input type for model is not tokens.");
    return false;
  }

  _gpio_marker->set();

  // Vector for storing logits.
  // Allocated & filled by the engine.
  std::vector<Tensor> logits;

  State::clear();

  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();
  if (m_n_queries == 0) {
    m_n_queries = tokenNumPerBatch.size();
  } else {
    if (m_n_queries != tokenNumPerBatch.size()) {
      __ERROR("The number of queries mismatch for follow-up query: expected {} queries, but received {}.", m_n_queries, tokenNumPerBatch.size());
      return false;
    }
  }
  size_t maxTokenNum = tokenNumPerBatch.empty() ? 0 : *std::max_element(tokenNumPerBatch.begin(), tokenNumPerBatch.end());
  bool keepProcessing = false;
  if (m_processState == NO_RESUME || m_processState == PROMPT_PROCESSING) {
    keepProcessing = true;

    if (_n_past + maxTokenNum > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, maxTokenNum, _ctx->size());
      throw genie::ContextLimitException("Context Size was exceeded.");
    }
    uint32_t noPaddingTokenNum= tokens.size();

    const size_t n_engine_returned = engine.process(tokens, tokenNumPerBatch, logits, false);
    if ((n_engine_returned != tokenNumPerBatch.size() && !m_pause) || engine.failed()) {
      __ERROR("Engine processing failed. Engine returned {} logits. Failed={} Error={}",
              n_engine_returned,
              engine.failed(),
              engine.error());
      return Dialog::abort("Engine prompt processing failed. " + engine.error(), callback);
    }

    if (m_pause && n_engine_returned != tokenNumPerBatch.size()) {
      __ERROR("Currently Pause/resume is not supported for multi-batch processing. "
              "Engine returned {} batches but expected {}. ",
              n_engine_returned, tokenNumPerBatch.size());
      return false;
    }

    _n_prompt += noPaddingTokenNum;
    _n_past += maxTokenNum;

    if (!engine.updateKV(_n_past) || engine.failed())
      return Dialog::abort("KV cache update failed. " + engine.error(), callback);
  }
  auto sCode = Sentence::BEGIN;

  if (keepProcessing) {
    // print the Token for the boundary KV rewind as it is next generated token
    if (m_rewindAtBoundary) {
      __ERROR("Boundary KV rewind (m_rewindAtBoundary) is not supported for multi-batch processing. "
              "This feature is only available for single-batch inference. ");
      return false;
    }

    tokens = sampler.process(logits, tokenNumPerBatch.size());

    sampler.updateSampledTokenHistory(tokens,tokens.size());
    m_unprocessedTokens = tokens;
    _n_generated += tokens.size();
  }

  _gpio_marker->set();

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  start.reset();
  std::vector<bool> allBatchesFinished(tokens.size(), false);
  bool allFinished = true;
  if (keepProcessing || m_processState == TOKEN_GEN) {
    for (size_t i = 0; i < tokens.size(); i++) {
      if ((_ctx->is_eos(tokens[i]))) {
        allBatchesFinished[i] = true;
        __DEBUG("Batch {} finished (EOS).", i);
      }
      if (!allBatchesFinished[i]) {
        allFinished = false;
      }
    }

    callback.setBatchFinished(allBatchesFinished);

    sCode = allFinished ? Sentence::END : Sentence::CONTINUE;
    if (!callback.callBack(tokens.data(), tokens.size(), sCode, tokenizer())) return true;

    if (allFinished) {
      return true;
    }
    State::busy(true);
    processFollowOnGeneration(tokens, logits, callback);
  }
  State::busy(false);

  _gpio_marker->set();
  _gpio_marker->reset();

  _kpis.generate.update(start.elapsed_usec());

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return !State::failed();
}


}  // namespace qualla
