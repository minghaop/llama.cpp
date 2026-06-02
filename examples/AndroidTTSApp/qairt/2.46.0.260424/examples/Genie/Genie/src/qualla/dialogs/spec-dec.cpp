//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <thread>

#include "Trace.hpp"
#include "qualla/detail/timer.hpp"
#include "spec-dec.hpp"

namespace fs = std::filesystem;
using qc     = qualla::Config;

namespace qualla {

SpecDecDialog::SpecDecDialog(std::shared_ptr<Env> env,
                             const std::string& name,
                             const nlohmann::json& conf)
    : Dialog(env, name, conf),
      _d_sampler(_sampler.contains("secondary") ? *_sampler["secondary"] : *_sampler["primary"]),
      _t_sampler(*_sampler["primary"]) {
  _draft_len = qc::optional<size_t>(conf, "draft-len", 3);
  _parallel  = qc::optional<bool>(conf, "parallel", false);

  // Check all underlying components for correct types an config
  // If something is not right we set our error state that can be checked later

  if (!_sampler.contains("primary")) {
    State::fatal("\"target\" sampler not present in config!");
    return;
  }

  if (!_engine.contains("primary")) {
    State::fatal("\"target\" engine not present in config!");
    return;
  }
  if (!_engine.contains("secondary")) {
    State::fatal("\"draft\" engine not present in config!");
    return;
  }

  _accepted_counts.resize(_draft_len + 1, 0);
}

int32_t SpecDecDialog::sampleFromModifiedDist(std::span<float> src0_dst, std::span<float> src1) {
  GENIE_TRACE();
  const size_t size = src0_dst.size();

  if (_t_sampler.gumbel()) {
    // Avoid going in the denormal zone.
    const float tiny = 1.1754943508222875e-38;

    for (size_t i = 0U; i < size; i++) {
      float p_src0 = std::exp(src0_dst[i]);
      float p_src1 = std::exp(src1[i]);
      src0_dst[i]  = std::log(std::max(tiny, p_src0 - p_src1));
    }

    // NOTE: The output logps_target is unnormalized since we use Gumbel trick.
    //       If we use standard multinomial sampling, normalization should be added.

  } else {
    float sum = 0.0;  // Unlikely to overflow (?)
    PRAGMA_LOOP_VECTORIZE
    for (size_t i = 0U; i < size; i++) {
      float num = std::max(0.f, src0_dst[i] - src1[i]);
      sum += num;
      src0_dst[i] = num;
    }
    // Normalize
    PRAGMA_LOOP_VECTORIZE
    for (size_t i = 0U; i < size; i++) {
      src0_dst[i] /= sum;
    }
  }

  if (_t_sampler.greedy()) return argmax(src0_dst);

  if (_t_sampler.gumbel()) return sampleUsingGumbelMax(src0_dst, _t_sampler.rng());

  // Skipping softmax since the probs are already normalized
  return sampleFromProbs(src0_dst, _t_sampler.rng());
}

size_t SpecDecDialog::rejectionSampling(std::span<int32_t> tokens,
                                        Tensor& target_logits,
                                        std::span<float> draft_probs,
                                        Acceptor accept) {
  GENIE_TRACE();
  const size_t n_vocab = _ctx->n_vocab();
  const size_t n_tok   = tokens.size();

  assert(tokens.size() == draft_probs.size() / n_vocab);
  assert(target_logits.getSize() == draft_probs.size() + n_vocab);

  // Rejection sampling:
  // For each token in the n_tok tokens sampled from the draft model:
  // 1. Determine the probability of that token being accepted by the target model
  // 2. Accept the token with probability = prob_target[tok] / prob_draft[tok] (clamped to [0, 1])
  // 3. If the token is rejected, resample a new token from the following distribution:
  //      [max(prob_target[x] - prob_draft[x], 0.f) for all x in vocab]
  int32_t t_tok;
  size_t n_accepted = 0;

  std::vector<float> target_probs;

  for (size_t i = 0; i < n_tok; i++) {
    int32_t d_tok = tokens[i];

    Tensor index_t_logits = target_logits.getIndexedTensor(i, n_vocab);

    if (_t_sampler.greedy()) {  // Invoked for Custom and basic with greedy sampling
      t_tok = _t_sampler.process(index_t_logits);
      if (t_tok != d_tok) {
        // Reject
        break;
      }
    } else {
      target_probs.clear();
      t_tok = _t_sampler.process(index_t_logits, target_probs, false);  // only probs, no token
      // Acceptance threshold
      double threshold;
      float prob_draft  = draft_probs[i * n_vocab + static_cast<uint32_t>(d_tok)];
      float prob_target = target_probs[static_cast<uint32_t>(d_tok)];

      if (_t_sampler.gumbel()) {
        threshold = std::exp(double(prob_target) - double(prob_draft));
      } else {
        threshold = double(prob_target) / double(prob_draft);
      }

      double r = sampleFromUniform(_t_sampler.rng());
      if (r > threshold) {
        // Reject
        break;
      }
    }
    // Accepted!
    ++n_accepted;
    if (!accept(d_tok)) return n_accepted;
    _t_sampler.updateSampledTokenHistory(d_tok);
  }

  // Sample an extra token either from the target distribution or the modified distribution
  if (n_accepted == n_tok) {
    Tensor t = target_logits.getIndexedTensor(n_tok, n_vocab, true);
    t_tok    = _t_sampler.process(t);
  } else if (!_t_sampler.greedy()) {
    // Resample from modified distribution.
    t_tok = sampleFromModifiedDist(std::span{target_probs.data(), target_probs.size()},
                                   draft_probs.subspan(n_accepted * n_vocab, n_vocab));
  }  // for greedy, t_tok should be already valid from the loop above

  ++n_accepted;
  accept(t_tok);
  _t_sampler.updateSampledTokenHistory(t_tok);

  return n_accepted;
}

bool SpecDecDialog::processFollowOnGeneration(std::vector<int32_t>& /*tokens*/,
                                              Tensor& t_logits,
                                              Tensor& d_logits,
                                              Dialog::Callback callback) {
  GENIE_TRACE();
  const size_t n_vocab = _ctx->n_vocab();

  bool keep_generating = true;

  // A buffer for tokens to be decoded (one at a time, per the Middleware's request)
  std::vector<int32_t> decode_buf(1, 0);

  // Decode new token.
  // Return true to continue generation, and false otherwise
  auto decode_token = [&](int32_t t) {
    decode_buf[0] = _last_tok = t;

    if (_ctx->is_eos(t)) {
      keep_generating = false;
      callback("", Sentence::END);
    } else {
      keep_generating = callback(_tokenizer->decode(decode_buf), Sentence::CONTINUE);
    }

    return keep_generating;
  };

  auto& t_engine = *_engine["primary"];
  auto& d_engine = *_engine["secondary"];

  // Buffers for all the tokens that need to be considered for each iteration
  std::vector<int32_t> toks_to_target(_draft_len + 1);
  std::vector<int32_t> toks_to_draft(2);

  // Buffer for all the probability distributions from the draft sampler
  std::vector<float> d_probs(n_vocab * _draft_len);

  toks_to_target.assign(1, _last_tok);
  toks_to_draft.assign(1, _last_tok);

  // Draft n_past, either in sync with n_past or one token behind (accepted-all)
  size_t d_n_past = _n_past;

  Timer start;

  while (!State::canceled() && keep_generating) {
    // Step 1: Use draft model to decode draft_len (aka gamma) tokens, and accumulate probabilities
    d_probs.clear();

    for (size_t i = 0; i < _draft_len; i++) {
      if (d_n_past + toks_to_draft.size() > _ctx->size()) {
        __WARN("Context limit exceeded ({} + {} > {})", d_n_past, toks_to_target.size(), _ctx->size());
        _kpis.generate.update(start.elapsed_usec());

        // Log latest KPIs in a single line
        __KPIS("{}", kpis().dump(" "));
        throw genie::ContextLimitException("Context Size was exceeded.");
      }

      if (!d_engine.process(toks_to_draft, d_logits))
        return Dialog::abort("draft engine gen processing failed", callback);

      d_n_past += toks_to_draft.size();

      if (!d_engine.updateKV(d_n_past)) return Dialog::abort("draft KV update failed", callback);

      _d_sampler.updatePenalty(_t_sampler.getPenalty());
      int32_t token = _d_sampler.process(d_logits, d_probs);
      toks_to_draft.assign(1, token);
      toks_to_target.push_back(token);

      if (_ctx->is_eos(token)) break;
    }

    // Step 2: run the target model on the draft tokens
    if (_n_past + toks_to_target.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, toks_to_target.size(), _ctx->size());
      _kpis.generate.update(start.elapsed_usec());

      // Log latest KPIs in a single line
      __KPIS("{}", kpis().dump(" "));
      throw genie::ContextLimitException("Context Size was exceeded.");
    }

    std::vector<int32_t> attention_map(toks_to_target.size());
    std::iota(attention_map.begin(), attention_map.end(), -1);
    size_t n_tok_t =
        t_engine.process(toks_to_target, attention_map, t_logits, true /* all logits */);
    if (n_tok_t != toks_to_target.size())
      return Dialog::abort("target engine gen processing failed", callback);

    // Step 3: accept or reject draft tokens
    size_t n_accepted =
        rejectionSampling(std::span{toks_to_target.data(), toks_to_target.size()}.subspan(1),
                          t_logits,
                          std::span{d_probs.data(), d_probs.size()},
                          decode_token);

    _n_generated += n_accepted;
    _n_decode += n_accepted;
    _n_past += n_accepted;

    // Update stats
    _accepted_counts[n_accepted - 1]++;

    // Accepted all?
    if (n_accepted == _draft_len + 1) {
      // Grab the last 2 tokens
      toks_to_draft.assign({toks_to_target[_draft_len], _last_tok});
      d_n_past = _n_past - 1;
    } else {
      // Grab only the last token
      toks_to_draft.assign(1, _last_tok);
      d_n_past = _n_past;
    }

    toks_to_target.assign(1, _last_tok);

    __DEBUG("spec-dec: draft_len {} n_generated {} n_accepted {} n_past {}",
            _draft_len,
            _n_generated,
            n_accepted,
            _n_past);

    std::vector<bool> selected(attention_map.size(), false);
    selected[0]       = true;  // first token is selected always
    uint32_t last_sel = 0;
    for (size_t i = n_accepted - 1; i != 0; i = static_cast<uint32_t>(attention_map[i])) {
      selected[i] = true;
      last_sel    = i > last_sel ? i : last_sel;
    }

    // Step 4: commit accepted tokens to kv-caches
    if (!t_engine.updateKV(_n_past, selected))
      return Dialog::abort("target KV update failed", callback);
    if (!d_engine.updateKV(d_n_past)) return Dialog::abort("draft KV update failed", callback);
  }

  if (d_n_past != _n_past) {
    // The draft engine needs to process one last token to catch up
    toks_to_draft.resize(1);
    if (!d_engine.process(toks_to_draft))
      return Dialog::abort("draft engine gen processing failed", callback);
    if (!d_engine.updateKV(_n_past)) return Dialog::abort("draft KV update failed", callback);
  }

  return true;
}
bool SpecDecDialog::process(std::vector<int32_t>& tokens, Dialog::Callback callback) {
  GENIE_TRACE();
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor t_logits;
  Tensor d_logits;

  bool keep_generating = true;

  // A buffer for tokens to be decoded (one at a time, per the Middleware's request)
  std::vector<int32_t> decode_buf(1, 0);

  // Decode new token.
  // Return true to continue generation, and false otherwise
  auto decode_token = [&](int32_t t) {
    decode_buf[0] = _last_tok = t;

    if (_ctx->is_eos(t)) {
      keep_generating = false;
      callback("", Sentence::END);
    } else {
      keep_generating = callback(_tokenizer->decode(decode_buf), Sentence::CONTINUE);
    }

    return keep_generating;
  };

  State::clear();

  auto& t_engine = *_engine["primary"];
  auto& d_engine = *_engine["secondary"];

  if (_n_past + tokens.size() > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
    throw genie::ContextLimitException("Context Size was exceeded.");
  }

  // Step 0: Process the prompt both on the target and draft models.
  bool d_pmpt, t_pmpt;
  if (_parallel) {
    std::thread dt([&]() { d_pmpt = d_engine.process(tokens, d_logits, false); });
    std::thread tt([&]() { t_pmpt = t_engine.process(tokens, t_logits, false); });
    dt.join();
    tt.join();
  } else {
    d_pmpt = d_engine.process(tokens, d_logits, false);
    t_pmpt = t_engine.process(tokens, t_logits, false);
  }

  if (!d_pmpt) {
    return Dialog::abort("draft engine prompt processing failed", callback);
  }
  if (!t_pmpt) {
    return Dialog::abort("target engine prompt processing failed", callback);
  }

  for (uint32_t idx = 0; idx < tokens.size(); idx++) {
    t_engine.updateTokenCheckpoint(static_cast<uint32_t>(tokens[idx]), _n_past + idx);
    d_engine.updateTokenCheckpoint(static_cast<uint32_t>(tokens[idx]), _n_past + idx);
  }
  // KV state Update
  _n_prompt += tokens.size();
  _n_past += tokens.size();

  if (!t_engine.updateKV(_n_past)) {
    return Dialog::abort("target KV update failed", callback);
  }
  if (!d_engine.updateKV(_n_past)) {
    return Dialog::abort("draft KV update failed", callback);
  }

  // Sample one token from the target.
  _last_tok = _t_sampler.process(t_logits);
  _t_sampler.updateSampledTokenHistory(_last_tok);

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  if (!decode_token(_last_tok)) return true;

  // Done with the prompt, start generating
  start.reset();
  State::busy(true);

  processFollowOnGeneration(tokens, t_logits, d_logits, callback);

  State::busy(false);

  _kpis.generate.update(start.elapsed_usec());

  auto total_iteration = std::accumulate(_accepted_counts.begin(), _accepted_counts.end(), 0);
  auto accept_rate =
      float(_n_generated - 1) / total_iteration;  // -1: exclude first generated token
  _kpis.tps.tokenAcceptance = accept_rate;

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));
  __KPIS("spec-dec: accepted counts: {}", _accepted_counts);

  return true;
}

void SpecDecDialog::reset() {
  Dialog::reset();
  _accepted_counts.clear();
}

}  // namespace qualla
