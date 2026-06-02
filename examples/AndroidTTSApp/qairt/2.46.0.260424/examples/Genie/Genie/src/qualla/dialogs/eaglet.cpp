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
#include "eaglet.hpp"
#include "qualla/detail/timer.hpp"
#include "resource-manager/include/ResourceManager.hpp"

namespace fs = std::filesystem;
using qc     = qualla::Config;

namespace qualla {

void EagletDialog::initializeEagletDialogConfig(const nlohmann::json& conf) {
  _config.draftLength   = qc::optional<size_t>(conf, "draft-len", 10);
  _config.numBranches   = qc::optional<size_t>(conf, "n-branches", 5);
  _config.probsPerDraft = qc::optional<size_t>(conf, "topn_probs", _config.numBranches * 8);
  _config.maxSeqAllowed =
      (_config.draftLength - 2) * _config.numBranches * (_config.numBranches - 1) +
      _config.numBranches * _config.numBranches;

  _config.maxTargetTokens = qc::optional<size_t>(conf, "max-tokens-target-can-evaluate", 32);
  // drafting kv cache
  _config.draftingKvCache = qc::optional<bool>(conf, "draft-kv-cache", false);
  _config.special_eos     = qc::optional<std::string>(conf, "special-eos-token", "");
  _config.vocabTrim       = _ctx->is_trimmed_vocab();

  _config.contextSize             = _ctx->n_ctx();
  _config.embeddingLength         = _ctx->n_embd();
  _draftStateManager.m_numDrafted = 0;
  _draftStateManager.m_drafts.resize(_config.maxSeqAllowed);
  _config.eos = _ctx->eos();
}

void EagletDialog::loadDraftTokenMap() {
  // clean any residue in draftTokenMap
  _config.draftTokenMap.clear();
  const std::string tokenMapKey = _engine["secondary"]->getTokenMapFilePath();
  if (tokenMapKey.empty()) {
    throw std::runtime_error("EagletDialog: Can't access token map file : " + tokenMapKey);
  }

  auto resourceManager = _env->getResourceManager();
  std::shared_ptr<uint8_t> rawBuf;
  if (!resourceManager->getBuffer(tokenMapKey, rawBuf) || !rawBuf) {
    throw std::runtime_error("EagletDialog: ResourceManager failed to load token map from: " +
                             tokenMapKey);
  }
  const size_t bufLen = resourceManager->getBufferSize(tokenMapKey);
  const nlohmann::ordered_json tokenMapConf =
      nlohmann::ordered_json::parse(reinterpret_cast<const char*>(rawBuf.get()),
                                    reinterpret_cast<const char*>(rawBuf.get()) + bufLen);

  for (auto& tok_element : tokenMapConf.items()) {
    _config.draftTokenMap.push_back(tok_element.value());
  }
  _config.trimmedVocabSize = _config.draftTokenMap.size();
}

bool EagletDialog::sampleFromTargetModel(uint32_t currDraftLevelIdx,
                                         uint32_t longestMatchedSequenceIdx,
                                         uint32_t& tokenIdx,
                                         int32_t& token,
                                         bool& hasEos) {
  // Get current draft sequence for the longest matched branch
  const auto& currentDraft = _draftStateManager.m_drafts[longestMatchedSequenceIdx];
  // Retrieve target token index at the current draft level
  uint32_t targetTokenIdx = currentDraft.targetBatchIndices[currDraftLevelIdx];

  // Avoid going beyond max tokens
  if (targetTokenIdx >= _config.maxTargetTokens) {
    --targetTokenIdx;
    return false;  // Signal to break while(true) loop
  }

  Tensor previousTargetLogits =
      _draftStateManager.targetTokens.m_logits.getIndexedTensor(targetTokenIdx, _ctx->n_vocab());
  int32_t sampledToken = sampleTargetToken(previousTargetLogits);
  if (_ctx->is_eos(sampledToken)) {
    hasEos = true;
  }

  token    = sampledToken;
  tokenIdx = targetTokenIdx;

  return true;
}

bool EagletDialog::checkDraftMatch(uint32_t currDraftLevelIdx,
                                   int32_t targetTokenId,
                                   uint32_t& longestMatchedSequenceIdx) {
  bool match = false;
  for (size_t s = 0; s < _config.maxSeqAllowed; ++s) {
    if (!_draftStateManager.m_drafts[s].isActive) {
      continue;
    }

    if (currDraftLevelIdx < _draftStateManager.m_drafts[s].tokens.size()) {
      if (targetTokenId == _draftStateManager.m_drafts[s].tokens[currDraftLevelIdx]) {
        longestMatchedSequenceIdx = s;
        match                     = true;
      } else
        _draftStateManager.m_drafts[s].isActive = false;
    }
  }

  return match;
}

bool EagletDialog::updateKvCache(Engine* draftEngine,
                                 Engine* targetEngine,
                                 uint32_t targetTokenIdx,
                                 uint32_t numMatchedTokens) {
  auto attentionMapSize = _draftStateManager.targetTokens.m_attentionMap.size();

  std::vector<bool> selectedTarget(attentionMapSize, false);
  selectedTarget[0] = true;  // Select first token

  while (targetTokenIdx < _draftStateManager.targetTokens.m_attentionMap.size()) {
    selectedTarget[targetTokenIdx] = true;
    targetTokenIdx =
        static_cast<uint32_t>(_draftStateManager.targetTokens.m_attentionMap[targetTokenIdx]);
  }

  _draftStateManager.m_numPastTarget += numMatchedTokens;

  if (!targetEngine->updateKV(_draftStateManager.m_numPastTarget, selectedTarget)) {
    __ERROR("EagletDialog::updateKvCache target failed");
    return false;
  }

  if (_config.draftingKvCache) {
    if (draftEngine->updateKV(_draftStateManager.m_numPastDraft) == false) {
      __ERROR("EagletDialog::updateKvCache draft failed");
      return false;
    }
  }

  return true;
}

void EagletDialog::copyEmbeddingBuffer(void* targetEmbeddingBuffer,
                                       std::vector<uint8_t>& embedIn,
                                       std::vector<int32_t>& acceptedTokenIds,
                                       std::vector<int32_t>& selectedIndices) {
  size_t copySize   = embedBuffSize;  //_config.embeddingLength * sizeof(uint16_t);
  uint8_t* src      = reinterpret_cast<uint8_t*>(targetEmbeddingBuffer);
  uint32_t vecDelta = 0;

  for (size_t i = 0; i < acceptedTokenIds.size() - 1; ++i) {
    uint8_t* srcDelta = src + copySize * static_cast<size_t>(selectedIndices[i + 1]);
    embedIn.insert(embedIn.begin() + vecDelta, srcDelta, srcDelta + copySize);
    vecDelta += copySize;
  }
}

int32_t EagletDialog::processFeatureVectors(Engine* targetEngine,
                                            Engine* draftEngine,
                                            uint32_t longestMatchedSequenceIdx,
                                            uint32_t /*numMatchedTokens*/,
                                            std::vector<int32_t>& acceptedTokenIds,
                                            std::vector<int32_t>& selectedIndices) {
  GENIE_TRACE();
  void* targetFeatureBuffer   = nullptr;
  void* targetEmbeddingBuffer = nullptr;

  std::vector<uint8_t> eagleEmbedIn;

  if (longestMatchedSequenceIdx >= _config.maxSeqAllowed) {
    __DEBUG("EagletDialog::longestMatchedSequenceIdx {} greater than max allow {} ",
            longestMatchedSequenceIdx,
            _config.maxSeqAllowed);
    return 0;
  }
  targetEngine->getBuffer(targetFeatureBuffer, targetFeatureBuffName, false);
  if (targetFeatureBuffer == nullptr) {
    __ERROR("EagletDialog::Required tensor '{}' not found in target model.", targetFeatureBuffName);
    return false;
  }
  targetEngine->getBuffer(targetEmbeddingBuffer, targetEmbedBuffName, false);
  if (targetEmbeddingBuffer == nullptr) {
    // Try alternative buffer name
    targetEngine->getBuffer(targetEmbeddingBuffer, targetEmbedBuffNameAlt, false);
    if (targetEmbeddingBuffer == nullptr) {
      __ERROR("EagletDialog::Required tensor '{}' or '{}' not found in target model.",
              targetEmbedBuffName,
              targetEmbedBuffNameAlt);
      return false;
    }
  }
  copyEmbeddingBuffer(targetEmbeddingBuffer, eagleEmbedIn, acceptedTokenIds, selectedIndices);

  std::vector<uint8_t> embedding;
  uint32_t copySize = embedBuffSize;
  int32_t token     = acceptedTokenIds.back();
  std::vector<uint8_t> curTokenEmbedding(copySize, 0);
  if (tokEmbedMap.find(token) == tokEmbedMap.end()) {
    m_t2eCallback(*this, token, curTokenEmbedding.data(), copySize);
    tokEmbedMap[token] = curTokenEmbedding;
  }

  eagleEmbedIn.insert(eagleEmbedIn.end(), tokEmbedMap[token].begin(), tokEmbedMap[token].end());
  draftEngine->process(eagleEmbedIn,
                       reinterpret_cast<const uint16_t*>(targetFeatureBuffer),
                       selectedIndices,
                       0,
                       true,
                       {},
                       _draftStateManager.draftTokens.m_logits,
                       false);

  return 0;
}

void EagletDialog::resetAfterAcceptingFromTree(int32_t id) {
  // reset all active drafts
  for (size_t s = 0; s < _config.maxSeqAllowed; ++s) {
    auto& currentDraft    = _draftStateManager.m_drafts[s];
    currentDraft.isActive = false;
    currentDraft.tokens.clear();
    currentDraft.targetBatchIndices.clear();
    currentDraft.batchDraftOverallIndices.clear();
    currentDraft.batchDraftNextIndices.clear();
    currentDraft.cumulativeProbabilities.clear();
    currentDraft.endIdx = 0;
  }
  _draftStateManager.m_numDrafted           = 0;
  _draftStateManager.m_numNextDraftedTokens = 0;

  // Add the new token that was sampled from target to the draft
  for (size_t s = 0; s < _config.maxSeqAllowed; ++s) {
    auto& currentDraft      = _draftStateManager.m_drafts[s];
    currentDraft.isActive   = false;
    currentDraft.isDrafting = false;
  }
  auto& firstDraft = _draftStateManager.m_drafts[0];
  firstDraft.tokens.push_back(id);
  firstDraft.targetBatchIndices.push_back(0);
  firstDraft.isActive                 = true;
  firstDraft.isDrafting               = true;
  firstDraft.endIdx                   = 0;
  firstDraft.draftNextIdx             = 0;
  firstDraft.draftCurrentIdx          = 0;
  firstDraft.batchDraftOverallIndices = {-1};
  firstDraft.batchDraftNextIndices    = {-1};
  firstDraft.cumulativeProbabilities  = {1.0};

  _draftStateManager.m_numCurrSeq = 1;

  // clear out tokens and add the new one as well as its attention map
  _draftStateManager.targetTokens.clear();
  _draftStateManager.targetTokens.add(firstDraft.tokens[0], -1, 1.0);

  // clear draft tokens
  _draftStateManager.draftTokens.m_tokens.clear();

  _draftStateManager.nextDraftTokens.m_tokens.clear();
  _draftStateManager.nextDraftTokens.m_attentionMap.clear();
}

void EagletDialog::resetDraftSkipFlags() {
  // Reset skip flag for all drafts
  for (uint32_t seq = 0; seq < _config.maxSeqAllowed; seq++) {
    _draftStateManager.m_drafts[seq].skip = false;
  }
}

std::pair<std::vector<int32_t>, std::vector<float>> EagletDialog::sampleTokenCandidates(
    uint32_t seq) {
  GENIE_TRACE();
  auto& currDraft    = _draftStateManager.m_drafts[seq];
  size_t vocabSize   = _config.vocabTrim ? _config.trimmedVocabSize : _ctx->n_vocab();
  uint32_t sampleIdx = _config.draftingKvCache ? currDraft.draftCurrentIdx : currDraft.draftNextIdx;

  Tensor logitsLeaf =
      _draftStateManager.draftTokens.m_logits.getIndexedTensor(sampleIdx, vocabSize);

  std::vector<float> probs;
  Timer start;
  _d_sampler.updatePenalty(_t_sampler.getPenalty());
  std::vector<int32_t> tokenCandidates =
      _d_sampler.process(logitsLeaf, probs, _config.numBranches, _config.probsPerDraft, 0);

  // Handle token ID conversion if vocab trimming is enabled
  if (_config.vocabTrim) {
    for (size_t idx = 0; idx < tokenCandidates.size(); ++idx) {
      uint32_t tokenMapKey = static_cast<uint32_t>(tokenCandidates[idx]);
      if (tokenMapKey < _config.draftTokenMap.size()) {
        tokenCandidates[idx] = _config.draftTokenMap[tokenMapKey];
      }
    }
  }
  draftSampleTime += start.elapsed_usec();
  draftSampleCount++;

  return {tokenCandidates, probs};
}

std::vector<uint32_t> EagletDialog::splitSequenceIntoBranches(uint32_t seq, size_t numCandidates) {
  std::vector<uint32_t> sequenceArray(1, seq);
  size_t effectiveBranches = std::min(static_cast<size_t>(_config.numBranches), numCandidates);
  for (uint32_t branch = 1; branch < effectiveBranches; ++branch) {
    if (_draftStateManager.m_numCurrSeq >= _config.maxSeqAllowed) {
      break;
    }

    _draftStateManager.m_drafts[_draftStateManager.m_numCurrSeq] = _draftStateManager.m_drafts[seq];
    _draftStateManager.m_drafts[_draftStateManager.m_numCurrSeq].isActive   = true;
    _draftStateManager.m_drafts[_draftStateManager.m_numCurrSeq].isDrafting = true;
    _draftStateManager.m_drafts[_draftStateManager.m_numCurrSeq].skip       = true;
    sequenceArray.push_back(_draftStateManager.m_numCurrSeq);
    _draftStateManager.m_numCurrSeq++;
  }

  return sequenceArray;
}

void EagletDialog::updateDraftAndTargetTokens(uint32_t /*seq*/,
                                              const std::vector<int32_t>& tokenCandidates,
                                              const std::vector<float>& tokenProbabilities,
                                              std::vector<float>& currentLevelProbabilities,
                                              const std::vector<uint32_t>& sequenceArray,
                                              int32_t idxTgtParent,
                                              int32_t idxDftParent) {
  for (size_t branchIdx = 0; branchIdx < sequenceArray.size(); ++branchIdx) {
    auto sampledToken     = tokenCandidates[branchIdx];
    uint32_t currSequence = sequenceArray[branchIdx];
    _draftStateManager.m_drafts[currSequence].tokens.push_back(sampledToken);
    float cumulativeProb = tokenProbabilities[branchIdx];
    auto& currSeqDraft   = _draftStateManager.m_drafts[currSequence];
    if (!currSeqDraft.cumulativeProbabilities.empty()) {
      cumulativeProb *= currSeqDraft.cumulativeProbabilities.back();
    }

    currSeqDraft.cumulativeProbabilities.push_back(cumulativeProb);
    currSeqDraft.targetBatchIndices.push_back(_draftStateManager.targetTokens.m_tokens.size());
    _draftStateManager.targetTokens.add(sampledToken, idxTgtParent, cumulativeProb);

    currSeqDraft.endIdx = _draftStateManager.draftTokens.m_tokens.size();
    currSeqDraft.batchDraftOverallIndices.push_back(
        static_cast<int32_t>(_draftStateManager.draftTokens.m_numTokens));
    _draftStateManager.draftTokens.add(sampledToken, idxDftParent, cumulativeProb);
    currentLevelProbabilities.push_back(cumulativeProb);
  }
}

float EagletDialog::calculateTopKThreshold(std::vector<float>& currentLevelProbabilities,
                                           uint32_t topK) {
  std::sort(
      currentLevelProbabilities.begin(), currentLevelProbabilities.end(), std::greater<float>());
  float topk_probs_thd = currentLevelProbabilities.size() < topK
                             ? currentLevelProbabilities.back()
                             : currentLevelProbabilities[topK - 1];

  return topk_probs_thd;
}

void EagletDialog::markEligibleSequences(float topKThreshold,
                                         std::vector<int32_t>& currentDraftTokens) {
  for (uint32_t seq = 0; seq < _config.maxSeqAllowed; seq++) {
    auto& currDraft = _draftStateManager.m_drafts[seq];
    if (currDraft.isDrafting) {
      float currentProb = currDraft.cumulativeProbabilities.back();
      if (currentProb >= topKThreshold) {
        if (currentDraftTokens.size() >= _config.numBranches) {
          currDraft.isDrafting = false;
          break;
        }

        int32_t nextParentIdx     = currDraft.batchDraftNextIndices.back();
        currDraft.draftNextIdx    = _draftStateManager.nextDraftTokens.m_tokens.size();
        currDraft.draftCurrentIdx = currentDraftTokens.size();
        currDraft.batchDraftNextIndices.push_back(
            static_cast<int32_t>(_draftStateManager.m_numNextDraftedTokens++));
        currentDraftTokens.push_back(currDraft.tokens.back());
        _draftStateManager.nextDraftTokens.m_attentionMap.push_back(nextParentIdx);
        _draftStateManager.nextDraftTokens.m_tokens.push_back(currDraft.tokens.back());
      } else {
        currDraft.isDrafting = false;
      }

      if (_ctx->is_eos(currDraft.tokens.back())) currDraft.isDrafting = false;
    }
  }
}

void EagletDialog::updateAttentionMap(uint32_t startDraftIdx,
                                      uint32_t numTokensCurrentLevel,
                                      uint32_t numDraftsPast,
                                      std::vector<int32_t>& newAttentionMap,
                                      std::vector<int32_t>& /*selectedIndicesPerLevel*/) {
  uint32_t attentionStride     = numDraftsPast + numTokensCurrentLevel;
  uint32_t maxAttentionMapsize = attentionStride * numTokensCurrentLevel;
  for (uint32_t i = 0; i < numTokensCurrentLevel; i++) {
    std::fill_n(
        newAttentionMap.begin() + (i * attentionStride), _draftStateManager.m_numPastDraft, 1);
    const uint32_t selfIndex = startDraftIdx + i;
    int32_t draftParentIdx   = static_cast<int32_t>(selfIndex);
    while (draftParentIdx >= 0) {
      uint32_t draftParentIdxU32 = static_cast<uint32_t>(draftParentIdx);
      uint32_t attentionIdx      = static_cast<uint32_t>(
          _draftStateManager.nextDraftTokens.m_attentionMap[draftParentIdxU32] +
          static_cast<int32_t>(_draftStateManager.m_numPastDraft + (i * attentionStride)));

      if (attentionIdx < maxAttentionMapsize) {
        newAttentionMap[attentionIdx] = 1;
      } else {
        __DEBUG(" updateAttentionMap Not allowing attention {} crossed size {}",
                attentionIdx,
                maxAttentionMapsize);
      }

      draftParentIdx =
          (draftParentIdxU32 < _draftStateManager.nextDraftTokens.m_attentionMap.size())
              ? _draftStateManager.nextDraftTokens.m_attentionMap[draftParentIdxU32]
              : -1;
    }
    uint32_t attentionMapIdx =
        selfIndex + _draftStateManager.m_numPastDraft + (i * attentionStride);
    if (attentionMapIdx < maxAttentionMapsize) {
      newAttentionMap[attentionMapIdx] = 1;
    } else {
      __DEBUG(" drop new attention mask index {}", attentionMapIdx);
    }
  }
}

void EagletDialog::processDraftTokens(Engine* draftEngine,
                                      Engine* targetEngine,
                                      void* draftFeatureBuffer,
                                      std::vector<int32_t>& currentDraftTokens,
                                      std::vector<int32_t>& selectedIndicesPerLevel,
                                      uint32_t& startIdxOffset,
                                      uint32_t& numPastDraft,
                                      std::vector<int32_t>& newAttentionMap) {
  GENIE_TRACE();
  size_t embedbuffsize                 = targetEngine->getEmbeddingBufferSize();
  std::vector<int32_t> selectedIndices = _config.draftingKvCache
                                             ? selectedIndicesPerLevel
                                             : _draftStateManager.nextDraftTokens.m_attentionMap;

  std::vector<uint8_t> eagleEmbedIn;
  if (_config.draftingKvCache) {
    for (auto& token : currentDraftTokens) {
      if (tokEmbedMap.find(token) == tokEmbedMap.end()) {
        std::vector<uint8_t> curTokenEmbedding(embedbuffsize, 0);
        m_t2eCallback(*this, token, curTokenEmbedding.data(), embedbuffsize);
        tokEmbedMap[token] = curTokenEmbedding;
      }
      eagleEmbedIn.insert(eagleEmbedIn.end(), tokEmbedMap[token].begin(), tokEmbedMap[token].end());
    }
  } else {
    for (auto& token : _draftStateManager.nextDraftTokens.m_tokens) {
      if (tokEmbedMap.find(token) == tokEmbedMap.end()) {
        std::vector<uint8_t> curTokenEmbedding(embedbuffsize, 0);
        m_t2eCallback(*this, token, curTokenEmbedding.data(), embedbuffsize);
        tokEmbedMap[token] = curTokenEmbedding;
      }
      eagleEmbedIn.insert(eagleEmbedIn.end(), tokEmbedMap[token].begin(), tokEmbedMap[token].end());
    }
  }

  if (_config.draftingKvCache) {
    draftEngine->process(eagleEmbedIn,
                         reinterpret_cast<const uint16_t*>(draftFeatureBuffer),
                         selectedIndices,
                         0,
                         true,
                         newAttentionMap,
                         _draftStateManager.draftTokens.m_logits,
                         true);
  } else {
    draftEngine->process(eagleEmbedIn,
                         reinterpret_cast<const uint16_t*>(draftFeatureBuffer),
                         selectedIndices,
                         startIdxOffset,
                         true,
                         _draftStateManager.nextDraftTokens.m_attentionMap,
                         _draftStateManager.draftTokens.m_logits,
                         true);
  }
  startIdxOffset += currentDraftTokens.size();
  numPastDraft += currentDraftTokens.size();
  if (_config.draftingKvCache) {
    // drafting kv update
    if (!draftEngine->updateKV(numPastDraft)) {
      State::error("error in draft model updateKV");
      return;
    }
  }
}

std::vector<std::pair<int32_t, float>> EagletDialog::prepareTokenProbs() {
  std::vector<std::pair<int32_t, float>> tokenProbs;
  for (size_t i = 0; i < _draftStateManager.targetTokens.m_tokens.size(); ++i) {
    tokenProbs.emplace_back(_draftStateManager.targetTokens.m_tokens[i],
                            _draftStateManager.targetTokens.m_probs[i]);
  }

  // Sort tokens by their global probabilities in descending order
  std::sort(
      tokenProbs.begin(), tokenProbs.end(), [](auto& a, auto& b) { return a.second > b.second; });

  return tokenProbs;
}

std::vector<size_t> EagletDialog::pruneTargetTokens(size_t maxLength, float probThr) {
  std::vector<size_t> indicesToPrune;
  size_t thresholdCount = 0;
  auto targetTokensSize = _draftStateManager.targetTokens.m_tokens.size();
  for (size_t i = 0; i < targetTokensSize; ++i) {
    if (_draftStateManager.targetTokens.m_probs[i] < probThr) {
      indicesToPrune.push_back(i);
    } else if (_draftStateManager.targetTokens.m_probs[i] == probThr) {
      long numProbable = std::count_if(_draftStateManager.targetTokens.m_probs.begin(),
                                       _draftStateManager.targetTokens.m_probs.end(),
                                       [probThr](float p) { return p > probThr; });
      if (++thresholdCount > maxLength - static_cast<uint32_t>(numProbable)) {
        indicesToPrune.push_back(i);
      }
    }
  }
  return indicesToPrune;
}

void EagletDialog::updateSequenceDraft(float probThr, std::vector<size_t>& indicesToPrune) {
  for (auto& draft : _draftStateManager.m_drafts) {
    if (!draft.isActive) {
      continue;
    }

    for (size_t i = 0; i < draft.tokens.size(); ++i) {
      if (draft.cumulativeProbabilities[i] < probThr) {
        long itrOffset = static_cast<long>(i);
        draft.tokens.erase(draft.tokens.begin() + itrOffset, draft.tokens.end());
        draft.batchDraftOverallIndices.erase(draft.batchDraftOverallIndices.begin() + itrOffset,
                                             draft.batchDraftOverallIndices.end());
        draft.targetBatchIndices.erase(draft.targetBatchIndices.begin() + itrOffset,
                                       draft.targetBatchIndices.end());
        draft.cumulativeProbabilities.erase(draft.cumulativeProbabilities.begin() + itrOffset,
                                            draft.cumulativeProbabilities.end());
        break;
      }
    }
  }

  for (auto& draft : _draftStateManager.m_drafts) {
    if (!draft.isActive) {
      continue;
    }

    for (size_t i = 0; i < draft.tokens.size(); ++i) {
      size_t pruneCount = 0;
      for (size_t idx : indicesToPrune) {
        if (draft.targetBatchIndices[i] > idx) {
          ++pruneCount;
        }
      }
      draft.targetBatchIndices[i] -= pruneCount;
    }
  }
}

template <typename T>
void EagletDialog::removeElements(std::vector<T>& vec, const std::vector<size_t>& indicesToPrune) {
  std::vector<bool> toRemove(vec.size(), false);
  for (size_t idx : indicesToPrune) {
    if (idx < vec.size()) {
      toRemove[idx] = true;
    }
  }

  size_t index = 0;
  vec.erase(
      std::remove_if(
          vec.begin(), vec.end(), [&toRemove, &index](const T&) { return toRemove[index++]; }),
      vec.end());
}

EagletDialog::EagletDialog(std::shared_ptr<Env> env,
                           const std::string& name,
                           const nlohmann::json& conf)
    : Dialog(env, name, conf), _d_sampler(*_sampler["primary"]), _t_sampler(*_sampler["primary"]) {
  initializeEagletDialogConfig(conf);
  completeInit();
}

void EagletDialog::completeInit() {
  Dialog::completeInit();
  if (_engine.size() == 2 && !m_initFinished) {
    if (!_engine.contains("primary")) {
      throw std::runtime_error("\"target\" engine not present in config!");
    }
    if (!_engine.contains("secondary")) {
      throw std::runtime_error("\"draft\" engine not present in config!");
    }
    void* testBuffer = nullptr;
    _engine["primary"]->getBuffer(testBuffer, targetFeatureBuffName, true);
    if (testBuffer == nullptr) {
      throw std::runtime_error(
          fmt::format("EagleDialog::EagleDialog tensor '{}' not found in target model.",
                      targetFeatureBuffName));
    }
    testBuffer = nullptr;
    _engine["secondary"]->getBuffer(testBuffer, draftFeatureBuffName, true);
    if (testBuffer == nullptr) {
      throw std::runtime_error(fmt::format(
          "EagleDialog::EagleDialog tensor '{}' not found in draft model.", draftFeatureBuffName));
    }
    testBuffer = nullptr;
    _engine["primary"]->getBuffer(testBuffer, targetEmbedBuffName, true);
    if (testBuffer == nullptr) {
      // Try alternative buffer name
      _engine["primary"]->getBuffer(testBuffer, targetEmbedBuffNameAlt, true);
      if (testBuffer != nullptr) {
        // Use alternative name if found
        targetEmbedBuffName = targetEmbedBuffNameAlt;
      } else {
        State::fatal(
            fmt::format("EagleDialog::EagleDialog tensor '{}' or '{}' not found in target model.",
                        targetEmbedBuffName,
                        targetEmbedBuffNameAlt));
        return;
      }
    }
    if (_config.vocabTrim) {
      loadDraftTokenMap();
    }
    m_initFinished = true;
  }
}

std::vector<int32_t> EagletDialog::acceptFromTree(Engine* draftEngine,
                                                  Engine* targetEngine,
                                                  int32_t& acceptedTokensFromDraft) {
  std::vector<int32_t> acceptedTokenIds;

  int32_t targetTokenId              = 0;
  uint32_t currDraftLevelIdx         = 0;  // idx for level in the draft tree (0 = target token)
  uint32_t longestMatchedSequenceIdx = 0;
  uint32_t targetTokenIdx            = 0;
  uint32_t numMatchedTokens          = 0;
  bool hasEos                        = false;

  // Main loop for sampling & accepting tokens
  while (true) {
    auto iscont = sampleFromTargetModel(
        currDraftLevelIdx, longestMatchedSequenceIdx, targetTokenIdx, targetTokenId, hasEos);
    if (!iscont) break;
    bool matchFound = checkDraftMatch(currDraftLevelIdx, targetTokenId, longestMatchedSequenceIdx);
    if (matchFound) {
      ++numMatchedTokens;
      ++currDraftLevelIdx;
      acceptedTokenIds.push_back(targetTokenId);
      ++acceptedTokensFromDraft;
    } else
      break;
  }

  /*
   * The loop always ends with match == false.
   * Either because of rejection or because we ran out of draft tokens to check
   * In both cases, we have one more target token to be added
   */
  acceptedTokenIds.push_back(targetTokenId);

  if (!updateKvCache(draftEngine, targetEngine, targetTokenIdx, numMatchedTokens)) {
    __ERROR("EagletDialog::acceptFromTree error in update KV cache");
    //__ERROR("error in updateKV");
    State::error("error in updateKV");

    return {-1};
  }

  _draftStateManager.draftTokens.clear();
  _draftStateManager.draftTokens.m_tokens = acceptedTokenIds;

  const auto& currentDraft = _draftStateManager.m_drafts[longestMatchedSequenceIdx];
  std::vector<int32_t> selectedIndices(
      currentDraft.targetBatchIndices.begin(),
      currentDraft.targetBatchIndices.begin() + numMatchedTokens + 1);
  // for the first token, the last feature vector in target engine buffer is needed
  if (currentDraft.targetBatchIndices.size() == 1) {
    selectedIndices[0] = static_cast<int32_t>(_n_prompt % promptVariant - 1);
  }

  processFeatureVectors(targetEngine,
                        draftEngine,
                        longestMatchedSequenceIdx,
                        numMatchedTokens,
                        acceptedTokenIds,
                        selectedIndices);

  // update the number of drafted tokens
  // for prompt stage, kv cache has been updated internally
  _draftStateManager.m_numPastDraft += acceptedTokenIds.size();
  if (!draftEngine->updateKV(_draftStateManager.m_numPastDraft)) {
    //__ERROR("updateKV failed for draft model");
    __ERROR("EagletDialog::processFeatureVectors updateKV failed");
    State::error("updateKV failed for draft model");
    return {-1};
  }

  resetAfterAcceptingFromTree(targetTokenId);

  return acceptedTokenIds;
}

void EagletDialog::createDraftTokenTree(Engine* draftEngine, Engine* targetEngine) {
  GENIE_TRACE();
  uint32_t startIdxOffset = 0;
  uint32_t numPastDraft   = _draftStateManager.m_numPastDraft;
  std::vector<int32_t> pastDraftPerLevel;

  // Iterate over the levels of the token tree
  for (uint32_t level = 0; level < _config.draftLength; level++) {
    resetDraftSkipFlags();
    std::vector<float> currentLevelProbabilities;
    std::vector<int32_t> currentDraftTokens;
    // iterate over current sequences
    for (uint32_t seq = 0; seq < _config.maxSeqAllowed; seq++) {
      auto& currDraft = _draftStateManager.m_drafts[seq];
      if (!currDraft.isDrafting || currDraft.skip) continue;
      auto [tokenCandidates, probs]       = sampleTokenCandidates(seq);
      std::vector<uint32_t> sequenceArray = splitSequenceIntoBranches(seq, tokenCandidates.size());
      int32_t idxTgtParent = static_cast<int32_t>(currDraft.targetBatchIndices.back());
      int32_t idxDftParent = currDraft.batchDraftOverallIndices.back();
      updateDraftAndTargetTokens(seq,
                                 tokenCandidates,
                                 probs,
                                 currentLevelProbabilities,
                                 sequenceArray,
                                 idxTgtParent,
                                 idxDftParent);
    }
    if (currentLevelProbabilities.empty() || level == _config.draftLength - 1) break;
    auto topKThreshold = calculateTopKThreshold(currentLevelProbabilities, _config.numBranches);
    markEligibleSequences(topKThreshold, currentDraftTokens);
    std::vector<int32_t> selectedIndicesPerLevel;
    size_t currDraftTokensSize = currentDraftTokens.size();
    pastDraftPerLevel.push_back(static_cast<int32_t>(currDraftTokensSize + startIdxOffset));

    std::vector<int32_t> newAttentionMap((numPastDraft + currDraftTokensSize) * currDraftTokensSize,
                                         0);
    if (_config.draftingKvCache) {
      auto startIt = _draftStateManager.nextDraftTokens.m_attentionMap.begin() +
                     static_cast<long>(startIdxOffset);
      auto endIt = startIt + static_cast<long>(currDraftTokensSize);
      if (level > 1) {
        for (size_t idx = 0; idx < currDraftTokensSize; idx++) {
          if (idx + startIdxOffset < _draftStateManager.nextDraftTokens.m_attentionMap.size()) {
            selectedIndicesPerLevel.push_back(
                _draftStateManager.nextDraftTokens.m_attentionMap[idx + startIdxOffset] -
                pastDraftPerLevel[level - 2]);
          }
        }
      } else {
        selectedIndicesPerLevel.assign(startIt, endIt);
      }
      updateAttentionMap(startIdxOffset,
                         currDraftTokensSize,
                         numPastDraft,
                         newAttentionMap,
                         selectedIndicesPerLevel);
    }

    void* draftFeatureBuffer = nullptr;
    draftEngine->getBuffer(draftFeatureBuffer, draftFeatureBuffName, false);
    if (draftFeatureBuffer == nullptr) {
      __ERROR("EagletDialog::Required tensor '{}' not found in draft model.", draftFeatureBuffName);
      return;
    }
    processDraftTokens(draftEngine,
                       targetEngine,
                       draftFeatureBuffer,
                       currentDraftTokens,
                       selectedIndicesPerLevel,
                       startIdxOffset,
                       numPastDraft,
                       newAttentionMap);
  }
}

void EagletDialog::pruneDraftTokenTree(size_t maxLength) {
  GENIE_TRACE();
  // Return early if target can evaluate i.e. within maxLength
  size_t targetTokenSize = _draftStateManager.targetTokens.m_tokens.size();
  if (targetTokenSize <= maxLength) {
    return;
  }

  auto tokenProbs = prepareTokenProbs();
  float probThr   = (maxLength - 1 < tokenProbs.size()) ? tokenProbs[maxLength - 1].second : 0.0f;
  std::vector<size_t> indicesToPrune = pruneTargetTokens(maxLength, probThr);

  removeElements(_draftStateManager.targetTokens.m_tokens, indicesToPrune);
  removeElements(_draftStateManager.targetTokens.m_attentionMap, indicesToPrune);
  removeElements(_draftStateManager.targetTokens.m_probs, indicesToPrune);

  // Update target attention map
  for (size_t i = 0; i < _draftStateManager.targetTokens.m_attentionMap.size(); i++) {
    if (_draftStateManager.targetTokens.m_attentionMap[i] < 0) continue;
    int32_t pruneCount = 0;
    for (size_t idx : indicesToPrune) {
      if (_draftStateManager.targetTokens.m_attentionMap[i] > static_cast<int32_t>(idx)) {
        ++pruneCount;
      }
    }
    _draftStateManager.targetTokens.m_attentionMap[i] -= pruneCount;
  }

  updateSequenceDraft(probThr, indicesToPrune);
}

void EagletDialog::evaluateDraftTokenTree(Engine* targetEngine) {
  GENIE_TRACE();
  // std::vector<uint16_t*> targetEmbeddedBuffer;
  std::vector<uint8_t> targetEmbedBuff;

  size_t embedbuffsize = targetEngine->getEmbeddingBufferSize();

  for (auto& token : _draftStateManager.targetTokens.m_tokens) {
    if (tokEmbedMap.find(token) == tokEmbedMap.end()) {
      std::vector<uint8_t> curTokenEmbedding(embedbuffsize, 0);
      m_t2eCallback(*this, token, curTokenEmbedding.data(), embedbuffsize);
      tokEmbedMap[token] = curTokenEmbedding;
    }
    targetEmbedBuff.insert(
        targetEmbedBuff.end(), tokEmbedMap[token].begin(), tokEmbedMap[token].end());
  }

  targetEngine->process(targetEmbedBuff,
                        _draftStateManager.targetTokens.m_attentionMap,
                        _draftStateManager.targetTokens.m_logits,
                        true);
  _draftStateManager.m_numPastTarget++;

  // the first token is always proposed by the target model before the speculation loop so we erase
  // it here
  for (uint32_t seq = 0; seq < _config.maxSeqAllowed; ++seq) {
    auto& currDraft = _draftStateManager.m_drafts[seq];
    if (!currDraft.isActive) continue;
    currDraft.tokens.erase(currDraft.tokens.begin());
  }
}

void EagletDialog::clearDraftTree() {
  resetDraftSkipFlags();
  __DEBUG("resetDraftSkipFlags ");
  for (size_t s = 0; s < _config.maxSeqAllowed; ++s) {
    auto& currentDraft    = _draftStateManager.m_drafts[s];
    currentDraft.isActive = false;
    currentDraft.tokens.clear();
    currentDraft.targetBatchIndices.clear();
    currentDraft.batchDraftOverallIndices.clear();
    currentDraft.batchDraftNextIndices.clear();
    currentDraft.cumulativeProbabilities.clear();
  }
  __DEBUG("Draft manger all seq clear ");
  _draftStateManager.draftTokens.m_numTokens = 0;
  _draftStateManager.m_numNextDraftedTokens  = 0;

  // Add the new token that was sampled from target to the draft
  for (size_t s = 0; s < _config.maxSeqAllowed; ++s) {
    auto& currentDraft      = _draftStateManager.m_drafts[s];
    currentDraft.isActive   = false;
    currentDraft.isDrafting = false;
  }

  _draftStateManager.targetTokens.clear();
  _draftStateManager.draftTokens.clear();
  _draftStateManager.nextDraftTokens.clear();
}

bool EagletDialog::process(std::vector<int32_t>& /*tokens*/, DialogCallback /*callback*/) {
  __ERROR("Eaglet does Not Supported tokens as input for now.");
  return false;
}

bool EagletDialog::process(std::vector<int32_t>& /*tokens*/, Dialog::Callback /*callback*/) {
  __ERROR("Eaglet does Not Supported tokens as input for now.");
  return false;
}

size_t EagletDialog::getEmbeddingBufferSize() {
  return _engine["primary"]->getEmbeddingBufferSize();
}

bool EagletDialog::removeStopSeqFromKV() {
  // remove stop seq tokens from KV$
  if (!_engine["primary"]->updateKV(_n_past)) return false;
  if (!_engine["secondary"]->updateKV(_n_past - _n_queries)) return false;
  // as d_engine.process input has one less embedding per query during prompt processing
  return true;
}

bool EagletDialog::process(std::vector<uint8_t>& embedding_vectors,
                           T2ECallback t2eCallback,
                           Dialog::Callback callback) {
  GENIE_TRACE();
  /****code copy */
  // Check for prev failures and bail out early
  if (State::failed()) return false;
  __INFO("EagletDialog::process started ");
  Timer start;
  State::clear();

  std::atomic<int32_t> process_token_counter(0);
  auto& t_engine = *_engine["primary"];
  auto& d_engine = *_engine["secondary"];

  m_t2eCallback       = t2eCallback;
  size_t embedBufSize = t_engine.getEmbeddingBufferSize();
  embedBuffSize       = embedBufSize;
  {
    std::vector<uint8_t> eosEmbedding(embedBufSize, 0.0);
    if (m_t2eCallback) {
      m_t2eCallback(*this, _ctx->eos(), eosEmbedding.data(), embedBufSize);
    }
    // For non-autogenerative usecases (where t2eCallback is not supplied),
    // the EOS vector is all zero. This is fine for models with proper
    // attention masking support, but may degrade accuracy otherwise.
    if (!t_engine.cacheEosEmbedding(eosEmbedding)) {
      __ERROR("Failed to set the eos token embedding for target engine.");
      return false;
    }

    if (!d_engine.cacheEosEmbedding(eosEmbedding)) {
      __ERROR("Failed to set the eos token embedding for draft engine.");
      return false;
    }
  }
  // comback here if this can be remove
  d_engine.updatedEmbeddingLength(_config.embeddingLength);
  t_engine.updatedEmbeddingLength(_config.embeddingLength);

  t_engine.setSharedCounter(process_token_counter);
  d_engine.setSharedCounter(process_token_counter);

  size_t n_input = embedding_vectors.size() / embedBufSize;

  // Process the prompt with the Target Model
  if (_n_past + n_input > _ctx->size()) {
    //__WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
    callback("", Sentence::END);
    return true;
  }

  // update last n_past value from the dialog.
  _draftStateManager.m_numPastTarget = _n_past;
  // d_engine.process input has one less embedding per query during prompt processing
  _draftStateManager.m_numPastDraft = _n_past - _n_queries + 1;

  void* targetFeatureBuffer = nullptr;
  std::vector<uint8_t> eagleEmbedIn(embedding_vectors.begin() + embedBuffSize,
                                    embedding_vectors.end());

  std::vector<int32_t> selectedIndices;
  uint32_t variant = promptVariant =
      t_engine.getBuffer(targetFeatureBuffer, targetFeatureBuffName, true);
  if (targetFeatureBuffer == nullptr) {
    __ERROR("EagletDialog::Required tensor '{}' not found in target model.", targetFeatureBuffName);
    return false;
  }
  const uint32_t num_iters = 1 + ((n_input - 1) / variant);
  for (size_t i = 0; i < num_iters; i++) {
    uint32_t n_input_cur = variant;
    if (i == num_iters - 1) {
      n_input_cur = n_input % variant == 0 ? n_input_cur : n_input % variant;
    }

    for (size_t j = 0; j < n_input_cur; j++) {
      selectedIndices.push_back(j);
    }
  }
  t_engine.setRunProcess(1);
  std::thread tt([&]() {
    if (!t_engine.process(embedding_vectors, {}, _draftStateManager.targetTokens.m_logits, false))
      return Dialog::abort("target engine prompt processing failed", callback);
    return true;
  });

  // come back here to remove the run_mode
  if (!d_engine.process(eagleEmbedIn,
                        reinterpret_cast<const uint16_t*>(targetFeatureBuffer),
                        selectedIndices,
                        0,
                        false,
                        {},
                        _draftStateManager.draftTokens.m_logits,
                        false))
    return Dialog::abort("draft engine prompt processing failed", callback);

  tt.join();

  _draftStateManager.m_numPastTarget += n_input;

  if (!t_engine.updateKV(_draftStateManager.m_numPastTarget))
    return Dialog::abort("target KV update failed", callback);

  // add only the last prompt token to tokens_tgt since we care about its logits
  int32_t latTok = _encoder->getLastToken();
  _draftStateManager.targetTokens.add(latTok, -1, 1.0);

  // targetBatchIndices only has one token @ index 0 in the overall target inference
  _draftStateManager.m_drafts[0].targetBatchIndices.resize(1);
  _draftStateManager.m_drafts[0].targetBatchIndices[0] = 0;

  _draftStateManager.m_numPastDraft += n_input - 1;
  // draft engine kv update
  if (!d_engine.updateKV(_draftStateManager.m_numPastDraft))
    return Dialog::abort("draft KV update failed",
                         callback);  // this updates _n_past internally

  _n_prompt += n_input;
  _n_past += n_input;

  t_engine.resetSharedCounter();
  d_engine.resetSharedCounter();
  t_engine.setRunProcess(0);
  d_engine.setRunProcess(0);

  // Generation loop : accept -> draft -> evaluate

  std::string token_str_complete     = "";
  int num_iterations                 = 0;
  bool keep_generating               = true;
  int32_t accepted_tokens_from_draft = 0;
  std::vector<size_t> accept_len;
  _kpis.prompt.update(start.elapsed_usec());
  start.reset();
  callback("", Sentence::BEGIN);
  while (!State::canceled() && keep_generating) {
    num_iterations += 1;
    accepted_tokens_from_draft = 0;
    std::vector<int32_t> accepted_ids =
        acceptFromTree(&d_engine, &t_engine, accepted_tokens_from_draft);

    if (accepted_ids.size() == 1 && accepted_ids[0] == -1) {
      return Dialog::abort("error in accept_from_tree", callback);
    }
    // iterate over accepted_ids, decode them, and send to the callback
    size_t accept_l = 0;
    for (const int32_t& id : accepted_ids) {
      _last_tok = id;
      _n_generated += 1;
      _n_decode += 1;
      accept_l += 1;
      if (_ctx->is_eos(id)) {
        keep_generating = false;
        callback("", Sentence::END);
        break;
      } else {
        std::string token_str = _tokenizer->decode({id});
        token_str_complete += token_str;
        keep_generating = callback(token_str, Sentence::CONTINUE);
        if (!keep_generating) break;  // break before incrementing _n_past
      }
      _n_past++;
    }
    accept_len.push_back(accept_l);

    if (!keep_generating) {
      break;
    }
    createDraftTokenTree(&d_engine, &t_engine);
    pruneDraftTokenTree(_config.maxTargetTokens);
    if (_n_past + _draftStateManager.targetTokens.m_tokens.size() > _ctx->size()) {
      callback("", Sentence::END);
      break;
    }
    evaluateDraftTokenTree(&t_engine);

    if (_n_generated >= _config.contextSize) {
      callback("", Sentence::END);
      break;
    }
  }
  _kpis.generate.update(start.elapsed_usec());
  clearDraftTree();
  _kpis.tps.tokenAcceptance = static_cast<float>(_n_generated) / (num_iterations - 1);
  __DEBUG("accept_len-{} Acceptance {}/{} {}",
          accept_len,
          _n_generated,
          (num_iterations - 1),
          _kpis.tps.tokenAcceptance);
  return true;
}

void EagletDialog::reset() {
  _n_past      = 0;
  _n_prompt    = 0;
  _n_generated = 0;
  _n_decode    = 0;
  _n_queries   = 0;
  _last_tok    = -1;

  _kpis.reset();

  // State::clear();
  Dialog::reset();
}

bool EagletDialog::bindEngine(const std::string& engineRole, std::shared_ptr<Engine> engine) {
  auto status = Dialog::bindEngine(engineRole, engine);
  Timer start;
  if (!status) return false;
  loadDraftTokenMap();
  _kpis.bindEngine.update(start.elapsed_usec());
  return status;
}

}  // namespace qualla
