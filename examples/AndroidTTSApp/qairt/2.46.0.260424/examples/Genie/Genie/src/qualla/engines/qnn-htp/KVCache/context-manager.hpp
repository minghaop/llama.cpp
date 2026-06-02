//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <set>

#include "nsp-params.hpp"
#include "qualla/env.hpp"

namespace qualla {

// Forward declarations from kvmanager.hpp
struct InferenceStep;
struct KVTensor;
struct CacheGroup;
struct UpdateStrategy;

struct ContextManager {
  std::shared_ptr<Env> m_env;
  CacheGroup* cache_group;
  LongContextParams params;

  ContextManager(std::shared_ptr<Env> env, LongContextParams _params)
      : m_env(env), params(_params) {}

  virtual ~ContextManager() = default;

  virtual void resetState(){};

  // Placeholder function for ContextManager subclasses to update their internal states
  // Currently, this is only used to update the KeyDiff anchors
  virtual bool afterExecution(int32_t, const InferenceStep&) { return true; }

  // processUpdate populates the KV$ copy strategy required to accept n_update KV$
  // Modifies: cache_group->m_n_valid_kv
  virtual UpdateStrategy processUpdate(const InferenceStep& step,
                                       const std::vector<int32_t>& src_idxes);

  // processMove populates the KV$ move strategy required to switch to the new variants
  // Modifies: cache_group->m_cur_variant, cache_group->m_cur_ctx, cache_group->m_n_valid_kv
  virtual UpdateStrategy processMove(int32_t variant, int32_t ctx_size);

  // processClear populates the KV$ clear strategy required to remove the requested KV$
  // Modifies: cache_group->m_n_valid_kv
  virtual UpdateStrategy processReduce(int32_t cur_n_past, int32_t new_n_past);

  // Translate the global attention mask into a group attention mask
  virtual std::vector<std::pair<int32_t, size_t>> translateAttentionMask(const InferenceStep&) {
    return {};
  }

  virtual void inferenceComplete() {}
};

struct SlidingWindow : public ContextManager {
  bool activated{false};
  std::deque<int32_t> recent_idxes;  // Indexes stored in order of generation
  std::set<int32_t> free_idxes;

  SlidingWindow(std::shared_ptr<Env> _env, LongContextParams _params)
      : ContextManager(_env, _params) {}

  void resetState() override;

  // processUpdate populates the KV$ copy strategy required to accept n_update KV$
  // Modifies: cache_group->m_n_valid_kv
  UpdateStrategy processUpdate(const InferenceStep& step,
                               const std::vector<int32_t>& src_idxes) override;

  // processMove populates the KV$ move strategy required to switch to the new variants
  // Modifies: cache_group->m_cur_variant, cache_group->m_cur_ctx, cache_group->m_n_valid_kv
  UpdateStrategy processMove(int32_t variant, int32_t ctx_size) override;

  // processReduce populates the KV$ reduce strategy required to reduce the requested KV$
  // Modifies: cache_group->m_n_valid_kv
  UpdateStrategy processReduce(const int32_t cur_n_past, const int32_t new_n_past) override;

  std::vector<std::pair<int32_t, size_t>> translateAttentionMask(
      const InferenceStep& step) override;
};

struct KeyDiff : public ContextManager {
  // Fields for the KeyDiff algorithm
  QnnApi* m_qnnApi;
  size_t anchor_n_bytes;
  size_t anchor_n_elements;
  int32_t m_eviction_queue_size{0};

  KeyDiff(std::shared_ptr<Env> _env, LongContextParams _params) : ContextManager(_env, _params) {}

  void registerKeydiffBuffers(const std::map<uint32_t, std::array<QnnUtils::Tensor*, 2>>& anchors,
                              const std::map<uint32_t, uint8_t*>& scores);

  void completeInit(QnnApi* qnnApi) { m_qnnApi = qnnApi; }

  void resetState() override;

  // For KeyDiff, runScorer invokes a scoring model to populate cache.score for each cache
  // updateEvictionIndexes must be run after each runScorer() call to utilize this data
  // Since runScorer operates on the HTP, it must be run on the main thread
  bool runScorer();

  // For HTP KeyDiff implementation, copy anchor outputs into anchor input buffers
  // For future optimization, this copy can be avoided in two ways
  // 1. Using a ping-pong Qnn_Tensor_t for anchor_in/anchor_out
  // 2. Using a READ_WRITE Qnn_Tensor_t that automatically reads/writes into the same buffer
  void clearAnchor(int32_t graph_idx);
  void updateAnchor(int32_t graph_idx);

  void updateEvictionIndexes(KVTensor& cache,
                             int32_t n_valid_kv,
                             int32_t n_evict,
                             int32_t head_idx);

  bool afterExecution(int32_t graph_idx, const InferenceStep& step) override;

  // processUpdate populates the KV$ copy strategy required to accept n_update KV$
  // Modifies: cache_group->m_n_valid_kv
  UpdateStrategy processUpdate(const InferenceStep& step,
                               const std::vector<int32_t>& src_idxes) override;

  // processMove populates the KV$ move strategy required to switch to the new variants
  // Modifies: cache_group->m_cur_variant, cache_group->m_cur_ctx, cache_group->m_n_valid_kv
  UpdateStrategy processMove(int32_t variant, int32_t ctx_size) override;
};

}  // namespace qualla