//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "kvmanager.hpp"

namespace qualla {

static inline std::vector<UpdateStep> compileIdxes(const std::vector<int32_t>& src_idxes,
                                                   const std::vector<int32_t>& dst_idxes) {
  // Compile src/dst_idxes into a vector of [src_idx, dst_idx, count]. This batches memory calls
  // This can be further optimized by detecting common contiguous copies (during token eviction)
  std::vector<UpdateStep> batch_idxes = {{src_idxes[0], dst_idxes[0], 1}};
  for (size_t i = 1; i < src_idxes.size(); i++) {
    // If the src/dst indexes are not consecutive, start a new batch with current src/dst indexes
    // Else, increment the size of the current batch
    if (src_idxes[i] != src_idxes[i - 1] + 1 || dst_idxes[i] != dst_idxes[i - 1] + 1)
      batch_idxes.push_back({src_idxes[i], dst_idxes[i], 1});
    else
      batch_idxes.back().count++;
  }

  return batch_idxes;
}

// ***********************************
// ContextManager default class (no longcontext)
// ***********************************

UpdateStrategy ContextManager::processUpdate(const InferenceStep& step,
                                             const std::vector<int32_t>& src_idxes) {
  const auto& [group_variant, group_ctx] =
      cache_group->getGroupVariant(step.variant, step.ctx_size);

  const int32_t n_valid_kv   = cache_group->m_n_valid_kv;
  const int32_t cache_budget = cache_group->manager->getCacheBudget(group_variant, group_ctx);
  const int32_t n_empty      = cache_budget - n_valid_kv;
  const int32_t n_update     = static_cast<int32_t>(src_idxes.size());

  // No longcontext is enabled, and the KV$ requested does not fit in this CacheGroup
  if (n_update > n_empty) {
    return UpdateStrategy(UpdateStrategy::ERROR);
  }

  // Update the group state m_n_valid_kv
  cache_group->m_n_valid_kv += n_update;

  // Construct the move required to perform this update
  UpdateStrategy updates = UpdateStrategy(UpdateStrategy::CACHED);
  std::vector<int32_t> dst_idxes(static_cast<uint32_t>(n_update));
  std::iota(dst_idxes.begin(), dst_idxes.end(), n_valid_kv);
  updates.steps = compileIdxes(src_idxes, dst_idxes);
  return updates;
}

UpdateStrategy ContextManager::processMove(int32_t variant, int32_t ctx_size) {
  const auto& [group_variant, group_ctx] = cache_group->getGroupVariant(variant, ctx_size);

  const int32_t n_valid_kv   = cache_group->m_n_valid_kv;
  const int32_t cache_budget =  cache_group->manager->getCacheBudget(group_variant, group_ctx);

  // No longcontext is enabled, and the KV$ requested does not fit in this CacheGroup
  if (n_valid_kv > cache_budget) return UpdateStrategy(UpdateStrategy::ERROR);

  // Update the group states m_cur_variant, m_cur_ctx
  cache_group->m_cur_variant = group_variant;
  cache_group->m_cur_ctx     = group_ctx;
  return UpdateStrategy();
}

UpdateStrategy ContextManager::processReduce(int32_t cur_n_past, int32_t new_n_past) {
  // If longcontext has been triggered already (n_past != n_valid_kv), reductions are disabled
  if (cur_n_past != cache_group->m_n_valid_kv) return UpdateStrategy(UpdateStrategy::ERROR);

  UpdateStrategy clears = UpdateStrategy(UpdateStrategy::CACHED);
  clears.steps.push_back({new_n_past, 0, static_cast<size_t>(cur_n_past - new_n_past)});

  cache_group->m_n_valid_kv = new_n_past;
  return clears;
}

// ***********************************
// Sliding Window long context ContextManager
// ***********************************

void SlidingWindow::resetState() {
  activated = false;
  while (!recent_idxes.empty()) recent_idxes.clear();
}

UpdateStrategy SlidingWindow::processReduce(const int32_t cur_n_past, const int32_t new_n_past) {
  UpdateStrategy clears = UpdateStrategy(UpdateStrategy::CACHED);
  if (!activated) {
    // If longcontext has been triggered already (n_past != n_valid_kv), reductions are disabled
    if (cur_n_past != cache_group->m_n_valid_kv) return UpdateStrategy(UpdateStrategy::ERROR);

    clears.steps.push_back({new_n_past, 0, static_cast<size_t>(cur_n_past - new_n_past)});
    cache_group->m_n_valid_kv = new_n_past;
  } else {
    auto clear_count = cur_n_past - new_n_past;

    // update recent_idxes
    std::set<int32_t> clear_idxes;
    for (auto i = 0; i < clear_count; ++i) {
      auto idx = recent_idxes.back();
      recent_idxes.pop_back();
      free_idxes.insert(idx);  // add to free list
      clear_idxes.insert(idx);
    }
    while(!clear_idxes.empty()) {
      int32_t count = 1;
      int32_t clear_idx = *clear_idxes.begin();
      clear_idxes.erase(clear_idxes.begin());
      while (clear_idxes.contains(clear_idx+count)){
        clear_idxes.erase(clear_idx+count);
        count++;
      }
      clears.steps.push_back({clear_idx, 0, static_cast<size_t>(count)});
    }
    cache_group->m_n_valid_kv -= clear_count;
  }
  return clears;
}

UpdateStrategy SlidingWindow::processUpdate(const InferenceStep& step,
                                            const std::vector<int32_t>& src_idxes) {
  const auto& [group_variant, group_ctx] =
      cache_group->getGroupVariant(step.variant, step.ctx_size);

  const int32_t n_valid_kv   = cache_group->m_n_valid_kv;
  const int32_t cache_budget = cache_group->manager->getCacheBudget(group_variant, group_ctx);

   // Calcaulte the number of available cache slots, and number of caches that need to be evicted
  const size_t n_update = src_idxes.size(); // #cache being accepted. # n_empty=slots available
  const size_t n_empty  = std::min(n_update, static_cast<size_t>(cache_budget - n_valid_kv));
  const size_t n_evict  = n_update - n_empty; // #cache to be evicted
  // Update the group state m_n_valid_kv
  cache_group->m_n_valid_kv = std::min(cache_budget, n_valid_kv + static_cast<int32_t>(n_update));

  // Initialize recency queue when we first reach KV capacity during execute flow
  const int32_t n_sink = params.sink_tokens;
  if (!activated && n_evict > 0) {
    if (!free_idxes.empty()) {
      fmt::println("Free index list should be empty at activation");
    }
    activated = true;
    for (int32_t i = n_sink; i < n_valid_kv; ++i) {
      recent_idxes.push_back(i);
    }
  }

  std::vector<int32_t> dst_idxes(n_update);
  std::set<int32_t> dst_idxes_set;
  int32_t idx = n_valid_kv;
  for (size_t i = 0; i < n_empty; i++) {
    if (!free_idxes.empty()) {
      auto free_idx = *free_idxes.begin();
      free_idxes.erase(free_idx);
      dst_idxes_set.insert(free_idx);
      dst_idxes[i] = free_idx;
    } else {
      while (dst_idxes_set.contains(idx)) {
        idx++;
      }
      dst_idxes[i] = idx;
      idx++;
    }
  }
  // Fill remaining updates via recency queue-based eviction
  for (size_t i = n_empty; i < n_update; ++i) {  // i.e. n_evict = n_update - n_empty
    int32_t curr_idx = recent_idxes.front();
    recent_idxes.pop_front();
    dst_idxes[i] = curr_idx;
  }

  if (activated) {
    for (auto& dst_idx: dst_idxes) {
      recent_idxes.push_back(dst_idx);
    }
  }

  UpdateStrategy updates = UpdateStrategy(UpdateStrategy::CACHED);
  updates.steps          = compileIdxes(src_idxes, dst_idxes);

  return updates;
}

UpdateStrategy SlidingWindow::processMove(int32_t variant, int32_t ctx_size) {
  const auto& [group_variant, group_ctx] = cache_group->getGroupVariant(variant, ctx_size);

  const int32_t cur_n_valid = cache_group->m_n_valid_kv;

  // Calcaulte the number of available cache slots, and number of caches that need to be evicted
  const int32_t cache_budget = cache_group->manager->getCacheBudget(group_variant, group_ctx);
  const int32_t n_valid      = std::min(cache_budget, cur_n_valid);
  const int32_t n_evict      = cur_n_valid - n_valid;

  // Update the group states m_cur_variant, m_cur_ctx, m_n_valid_kv
  cache_group->m_cur_variant = group_variant;
  cache_group->m_cur_ctx     = group_ctx;
  cache_group->m_n_valid_kv  = n_valid;

  if (n_evict <= 0) return UpdateStrategy();

  if (!activated) {
    activated = true;
    for (int i = params.sink_tokens; i < cur_n_valid; i++) recent_idxes.push_back(i);
  }

  auto moves = UpdateStrategy(UpdateStrategy::CACHED);

  // Create eviction set using recent indexes
  std::set<int32_t> evict_set;
  for (int i = 0; i < n_evict; i++) {
    int32_t curr = recent_idxes.front();
    evict_set.insert(curr);
    recent_idxes.pop_front();
  }

  // Use the eviction indexes to prune necessary KV$
  // Iterate through target indices and set up src and dst mappings
  std::vector<int32_t> src_idxes, dst_idxes;
  auto evict_iter = evict_set.begin();
  for (int idx = n_valid; idx < cur_n_valid; idx++) {
    if (evict_set.contains(idx)) continue;  // This index was slated for eviction, so no-op
    src_idxes.push_back(idx);
    dst_idxes.push_back(*evict_iter);
    evict_iter = evict_set.erase(evict_iter);
  }

  // Update invalidated recency queue src indexes to dst indexes
  std::unordered_map<int32_t, int32_t> idx_map;
  for (size_t i = 0; i < src_idxes.size(); ++i) {
    idx_map[src_idxes[i]] = dst_idxes[i];
  }
  size_t queue_size = recent_idxes.size();
  for (size_t i = 0; i < queue_size; ++i) {
    int32_t curr = recent_idxes.front();
    recent_idxes.pop_front();
    if (idx_map.count(curr)) curr = idx_map[curr];
    recent_idxes.push_back(curr);
  }
  moves.steps = compileIdxes(src_idxes, dst_idxes);
  return moves;
}

std::vector<std::pair<int32_t, size_t>> SlidingWindow::translateAttentionMask(
    const InferenceStep& step) {
  const auto& [group_variant, group_ctx] =
      cache_group->getGroupVariant(step.variant, step.ctx_size);
  const int32_t cache_budget = cache_group->manager->getCacheBudget(group_variant, group_ctx);

  if (!activated) {
    if (step.new_idx <= cache_budget) return {{0, group_ctx}};
    return {{0, cache_budget}, {step.new_idx, group_variant}};
  }

  std::vector<std::pair<int32_t, size_t>> gather_indexes;

  // If sink tokens are enabled, gather the sink attention first
  const int32_t n_sink = params.sink_tokens;
  if (n_sink > 0) gather_indexes.push_back({0, n_sink});

  // All other attention is offset by the dimensional difference
  // If both have identical dimensions, gather_indexes is purely based on recency
  const int32_t offset = step.n_valid_kv - cache_group->m_n_valid_kv;

  // Gather the data (group_index, n_contiguous, global_index) into a single vector
  std::vector<std::tuple<int32_t, int32_t, int32_t>> index_map;
  int32_t start_global_index = n_sink + offset;
  for (size_t i = 0; i < recent_idxes.size();) {
    // Count consecutive cache indices
    size_t count = 1;
    while (i + count < recent_idxes.size() &&
           recent_idxes[i + count] == recent_idxes[i] + static_cast<int32_t>(count)) {
      count++;
    }

    index_map.emplace_back(recent_idxes[i],
                           static_cast<int32_t>(count),
                           start_global_index + static_cast<int32_t>(i));
    i += count;
  }

  // Add an index map entry for the new KV$ to be generated at new_idx
  InferenceStep group_step = cache_group->translateInferenceStep(step);
  index_map.emplace_back(group_step.new_idx, step.n_process, step.new_idx);

  // Sorting here orders the map by the cache group indexes
  std::sort(index_map.begin(), index_map.end());

  // Build gather_indexes. This detects gaps in the cache group KV$ indexes and adds -1 accordingly
  int32_t expected_cache_idx = n_sink;
  for (const auto& [group_index, count, global_index] : index_map) {
    if (group_index > expected_cache_idx) {  // Insert a gap if needed
      gather_indexes.emplace_back(-1, group_index - expected_cache_idx);
    }

    gather_indexes.emplace_back(global_index, count);
    expected_cache_idx = group_index + count;
  }

  return gather_indexes;
}

// ***********************************
// KeyDiff long context ContextManager
// ***********************************

UpdateStrategy KeyDiff::processUpdate(const InferenceStep& step,
                                      const std::vector<int32_t>& src_idxes) {
  const auto& [group_variant, group_ctx] =
      cache_group->getGroupVariant(step.variant, step.ctx_size);

  const int32_t n_valid_kv   = cache_group->m_n_valid_kv;
  const int32_t cache_budget = cache_group->manager->getCacheBudget(group_variant, group_ctx);

  // Calcaulte the number of available cache slots, and number of caches that need to be evicted
  const int32_t n_update = static_cast<int32_t>(src_idxes.size());         // #cache being accepted
  const int32_t n_empty  = std::min(n_update, cache_budget - n_valid_kv);  // #slots available
  const int32_t n_evict  = n_update - n_empty;                             // #cache to be evicted

  // Update the group state m_n_valid_kv
  cache_group->m_n_valid_kv = std::min(cache_budget, n_valid_kv + n_update);

  UpdateStrategy updates = UpdateStrategy(UpdateStrategy::DYNAMIC);

  // Check if the eviction queue needs to be updated
  bool update_queue = false;
  if (n_evict > m_eviction_queue_size) {
    update_queue            = true;
    updates.update_preparer = [this]() { return runScorer(); };
    m_eviction_queue_size   = std::max(n_evict, params.update_frequency);
  }

  // Construct lambda function to generate source/destination indexes for each head
  updates.step_generator = [this, n_valid_kv, n_update, n_empty, n_evict, src_idxes, update_queue](
                               KVTensor& cache, int32_t head_idx) {
    if (update_queue) updateEvictionIndexes(cache, n_valid_kv, n_evict, head_idx);

    std::vector<int32_t> dst_idxes(static_cast<size_t>(n_update));
    std::iota(&dst_idxes[0], &dst_idxes[static_cast<size_t>(n_empty)], n_valid_kv);

    // Evict n_evict (i.e. n_update - n_empty) tokens, and overwrite
    for (int32_t i = n_empty; i < n_update; i++) {
      dst_idxes[static_cast<size_t>(i)] = cache.evict_idxes[static_cast<size_t>(head_idx)].front();
      cache.evict_idxes[static_cast<size_t>(head_idx)].pop();
    }

    return compileIdxes(src_idxes, dst_idxes);
  };

  // n_evict indexes have been consumed from the eviction queue, so update queue_size
  m_eviction_queue_size -= n_evict;

  return updates;
}

UpdateStrategy KeyDiff::processMove(int32_t variant, int32_t ctx_size) {
  const auto& [group_variant, group_ctx] = cache_group->getGroupVariant(variant, ctx_size);

  const int32_t cur_n_valid = cache_group->m_n_valid_kv;

  // Calcaulte the number of available cache slots, and number of caches that need to be evicted
  const int32_t cache_budget = cache_group->manager->getCacheBudget(group_variant, group_ctx);
  const int32_t n_valid      = std::min(cache_budget, cur_n_valid);
  const int32_t n_evict      = cur_n_valid - n_valid;

  // Update the group states m_cur_variant, m_cur_ctx, m_n_valid_kv
  cache_group->m_cur_variant = group_variant;
  cache_group->m_cur_ctx     = group_ctx;
  cache_group->m_n_valid_kv  = n_valid;

  if (n_evict <= 0) return UpdateStrategy();

  auto moves = UpdateStrategy(UpdateStrategy::DYNAMIC);

  // Check if the eviction queue needs to be updated
  bool update_queue = false;
  if (n_evict > m_eviction_queue_size) {
    update_queue          = true;
    moves.update_preparer = [this]() { return runScorer(); };
    m_eviction_queue_size = 0;  // Queue is invalidated after each move. See comment below.
  }

  // Construct lambda function to generate source/destination indexes for each head
  moves.step_generator = [this, cur_n_valid, n_valid, n_evict, update_queue](
                             KVTensor& cache, int32_t head) -> std::vector<UpdateStep> {
    if (update_queue) updateEvictionIndexes(cache, n_valid, n_evict, head);

    auto& evict_queue = cache.evict_idxes.at(static_cast<size_t>(head));

    // Collect eviction indexes. Only "valid" indexes (i.e. fits in new KV$) are considered
    std::set<int32_t> evict_set;
    for (int i = 0; i < n_evict; i++) {
      evict_set.insert(evict_queue.front());
      evict_queue.pop();
    }

    // Invalidate/empty the queue since indexes will change after eviction/reshape
    // Theoretically, we only need to invalidate the pruned idxes, so this can be optimized
    while (!evict_queue.empty()) evict_queue.pop();

    std::vector<int32_t> src_idxes, dst_idxes;
    auto evict_iter = evict_set.begin();
    for (int idx = n_valid; idx < cur_n_valid; idx++) {
      if (evict_set.contains(idx)) continue;  // This index was slated for eviction, so no-op
      src_idxes.push_back(idx);
      dst_idxes.push_back(*evict_iter);
      evict_iter = evict_set.erase(evict_iter);
    }

    return compileIdxes(src_idxes, dst_idxes);
  };

  return moves;
}

void KeyDiff::registerKeydiffBuffers(
    const std::map<uint32_t, std::array<QnnUtils::Tensor*, 2>>& anchors,
    const std::map<uint32_t, uint8_t*>& scores) {
  for (auto& [index, cache] : cache_group->m_tensor_index) {
    if (!anchors.contains(index) || !scores.contains(index))
      throw std::runtime_error(fmt::format("Couldn't find anchor tensor for KV$[{}]", index));

    const auto& [anchor_in, anchor_out] = anchors.at(index);

    cache->anchor_tensor_in  = anchor_in;
    cache->anchor_tensor_out = anchor_out;
    cache->anchor_offset     = static_cast<uint16_t>(-anchor_in->quantParam[0].offset);
    anchor_n_bytes           = anchor_in->dims.bitwidth;
    anchor_n_elements        = anchor_in->dims.getNumElements();

    cache->scores = scores.at(index);

    cache->evict_idxes.resize(cache->n_heads);
  }
}

void KeyDiff::resetState() {
  m_eviction_queue_size = 0;
  for (auto& [graph_index, graph_tensors] : cache_group->m_tensors) {
    for (auto& tensor : graph_tensors) {
      for (auto& head_evict_idxes : tensor.evict_idxes) {
        while (!head_evict_idxes.empty()) head_evict_idxes.pop();
      }
    }
    clearAnchor(graph_index);
  }
}

bool KeyDiff::runScorer() {
  // __DEBUG("Updating KeyDiff scores - executing scorer");
  if (m_qnnApi == nullptr) throw std::runtime_error("Qnn API not registered for scoring network");
  if (!m_qnnApi->executeScorer()) throw std::runtime_error("Error executing scorer network");
  return true;
}

void KeyDiff::clearAnchor(int32_t graph_idx) {
  for (auto& cache : cache_group->m_tensors.at(graph_idx)) {
    if (cache.anchor_in == nullptr || cache.anchor_out == nullptr) continue;
    std::fill_n(
        reinterpret_cast<uint16_t*>(cache.anchor_in), anchor_n_elements, cache.anchor_offset);
  }
}

void KeyDiff::updateAnchor(int32_t graph_idx) {
  for (auto& cache : cache_group->m_tensors.at(graph_idx)) {
    if (cache.anchor_in == nullptr || cache.anchor_out == nullptr) continue;
    std::memcpy(cache.anchor_in, cache.anchor_out, anchor_n_elements * anchor_n_bytes);
  }
}

void KeyDiff::updateEvictionIndexes(KVTensor& cache,
                                    int32_t n_valid_kv,
                                    int32_t n_evict,
                                    int32_t head_idx) {
  // Update cache.evict_idxes based on cache.scores [group.n_heads, cur_ctx]
  const int32_t n_sink     = params.sink_tokens;
  const int32_t n_queue    = std::max(n_evict, params.update_frequency);
  const int32_t n_eligible = n_valid_kv - n_sink;

  std::vector<size_t> indices(static_cast<size_t>(n_eligible));
  std::iota(indices.begin(), indices.end(), n_sink);

  uint16_t* const scores =
      reinterpret_cast<uint16_t*>(cache.scores) + head_idx * cache_group->m_cur_ctx;
  std::partial_sort(indices.begin(),
                    indices.begin() + n_queue,
                    indices.end(),
                    [scores](const size_t a, const size_t b) { return scores[a] > scores[b]; });

  auto& head_evict_queue = cache.evict_idxes[static_cast<size_t>(head_idx)];
  while (!head_evict_queue.empty()) head_evict_queue.pop();
  for (int i = 0; i < n_queue; i++) head_evict_queue.push(indices[static_cast<size_t>(i)]);
}

bool KeyDiff::afterExecution(int32_t graph_idx, const InferenceStep& /*step*/) {
  updateAnchor(graph_idx);
  return true;
}

}  // namespace qualla
