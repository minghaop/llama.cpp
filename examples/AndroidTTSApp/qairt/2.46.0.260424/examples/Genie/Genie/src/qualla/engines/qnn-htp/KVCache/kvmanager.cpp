//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fstream>  // For save/restore to file
#include <sstream>  // For in-memory stream (DLC record path)

#include "ResourceManager.hpp"
#include "Trace.hpp"
#include "TraceLogger.hpp"
#include "empty-manager.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "native-kv.hpp"
#include "qualla/detail/cache-file.hpp"
#include "smart-mask.hpp"

namespace qualla {

CacheGroup::CacheGroup(std::shared_ptr<Env> env,
                       std::string prefix,
                       bool scatter,
                       LongContextParams longcontext_params)
    : _env(env), m_prefix(prefix), m_use_scatter(scatter) {
  if (longcontext_params.mode == LongContextParams::KEYDIFF)
    context_manager = std::make_unique<KeyDiff>(_env, longcontext_params);
  else if (longcontext_params.mode == LongContextParams::SLIDING_WINDOW)
    context_manager = std::make_unique<SlidingWindow>(_env, longcontext_params);
  else
    context_manager = std::make_unique<ContextManager>(_env, longcontext_params);
}

void CacheGroup::registerTensors(
    const std::map<int, std::map<uint32_t, std::array<std::pair<QnnUtils::Tensor*, size_t>, 4>>>&
        tensors) {
  if (tensors.empty() || tensors.begin()->second.empty()) return;

  const auto& [key_in_pair, key_out_pair, val_in_pair, val_out_pair] =
      tensors.begin()->second.begin()->second;
  // Unpack the pairs
  const auto& [key_in, key_in_size]   = key_in_pair;
  const auto& [key_out, key_out_size] = key_out_pair;

  // Keys have [n_heads, kv_dim, ctx_size]. Values have [n_heads, ctx_size, kv_dim]
  n_bytes     = key_out->dims.bitwidth;
  n_embed_dim = static_cast<int32_t>(key_out->dims.width);
  m_quantized = (key_out->dtype.type() != 2);  // Based on QnnTypes.h, float types have type 0x02xx

  // Register tensors for this group
  for (const auto& [graph_index, kv_tensors] : tensors) {
    for (const auto& [index, kv] : kv_tensors) {
      m_tensors[graph_index].emplace_back(nullptr,
                                          index,
                                          kv[0].first == nullptr ? kv[1].first : kv[0].first,
                                          kv[2].first == nullptr ? kv[3].first : kv[2].first);
    }
  }

  for (auto& [graph_index, graph_tensors] : m_tensors) {
    for (auto& tensor : graph_tensors) {
      m_tensor_index[tensor.idx] = &tensor;
    }
  }

  // Establish the clear value
  if (m_quantized) {
    if (n_bytes == 1) {
      m_clear_value.u8 = static_cast<uint8_t>(1) << 7;
    } else if (n_bytes == 2) {
      m_clear_value.u16 = static_cast<uint16_t>(1) << 15;
    } else if (n_bytes == 4) {
      m_clear_value.u32 = static_cast<uint32_t>(1) << 31;
    }
  } else {
    m_clear_value.u32 = 0x0u;  // Float values are always cleared to 0s
  }

  bool isKvOutputHMXFormat =
      QNN_TENSOR_GET_DATA_FORMAT(key_out->tensor) == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT;
  if ((key_in == nullptr && isKvOutputHMXFormat) ||
      (key_in != nullptr &&
       QNN_TENSOR_GET_DATA_FORMAT(key_in->tensor) == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT)) {
    manager = std::make_unique<NativeKV>(_env, m_use_scatter);
  } else {
    manager = std::make_unique<SmartMask>(_env, m_use_scatter);
  }

  // Calculate n_elements based on key_in_size and key_out->dims.getNumElements()
  size_t key_elements = (key_in_size / size_t(n_bytes));
  size_t out_elements = m_use_scatter ? 0 : key_out->dims.getNumElements() * key_out->dims.batch;
  n_elements          = key_elements + out_elements;
}

void CacheGroup::registerKvOutputNativeFormat(std::map<VariantSpec, bool>& isKvOutputNativeFormat) {
  // Translate the global AR/CL to local AR/CL when registering NativeKV status
  for (const auto& [global_variant, isNative] : isKvOutputNativeFormat) {
    const VariantSpec group_variant = getGroupVariant(global_variant.first, global_variant.second);
    m_isKvOutputNativeFormat[group_variant] = isNative;
  }
}

bool CacheGroup::completeInit(QnnApi* qnnApi) {
  std::shared_ptr<IOTensor> ioTensor = qnnApi->getIOTensor();
  if (!manager) {
    // If manager is null, then no KV tensors were detected in the model.
    manager = std::make_unique<EmptyManager>(_env, m_use_scatter);
  }
  for (auto& [graph_index, graph_tensors] : m_tensors) {
    for (auto& tensor : graph_tensors) {
      tensor.key_buf = static_cast<uint8_t*>(ioTensor->getBuffer(tensor.key->tensor));
      tensor.val_buf = static_cast<uint8_t*>(ioTensor->getBuffer(tensor.value->tensor));
      if (tensor.anchor_tensor_in)
        tensor.anchor_in =
            static_cast<uint8_t*>(ioTensor->getBuffer(tensor.anchor_tensor_in->tensor));
      if (tensor.anchor_tensor_out)
        tensor.anchor_out =
            static_cast<uint8_t*>(ioTensor->getBuffer(tensor.anchor_tensor_out->tensor));

      if (tensor.anchor_tensor_in) {
        tensor.anchor_in =
            static_cast<uint8_t*>(ioTensor->getBuffer(tensor.anchor_tensor_in->tensor));
      }
      if (tensor.anchor_tensor_out) {
        tensor.anchor_out =
            static_cast<uint8_t*>(ioTensor->getBuffer(tensor.anchor_tensor_out->tensor));
      }
    }
  }
  manager->completeInit(*this);
  if (manager->failed()) {
    return false;
  }
  if (context_manager->params.mode == LongContextParams::KEYDIFF) {
    dynamic_cast<KeyDiff*>(context_manager.get())->m_qnnApi = qnnApi;
  }
  return true;
}

InferenceStep CacheGroup::translateInferenceStep(InferenceStep step) {
  std::tie(step.variant, step.ctx_size) = getGroupVariant(step.variant, step.ctx_size);
  step.n_valid_kv                       = m_n_valid_kv;
  step.new_idx                          = manager->getIndexForNewKV(step);
  return step;
}

KVTensor::KVTensor(std::shared_ptr<genie::profiling::TraceLogger> traceLogger,
                   uint32_t index,
                   QnnUtils::Tensor* k,
                   QnnUtils::Tensor* v)
    : Traceable(traceLogger), idx(index) {
  key   = k;
  value = v;

  n_heads     = k->dims.height;
  key_quant   = k->quantParam[0];
  value_quant = v->quantParam[0];
}

KVManager::KVManager(std::shared_ptr<Env> env,
                     QnnApi* qnnApi,
                     std::shared_ptr<IOTensor> m_ioTensor,
                     std::shared_ptr<ThreadPool> threadpool)
    : State(env->getTraceLogger()),
      IOTensor(m_ioTensor),
      _env(env),
      m_threadpool(threadpool),
      m_qnn_api(qnnApi) {}

KVManager::~KVManager() {}

void KVManager::registerSupportedVariant(int32_t variant, int32_t ctx_size, GraphType type) {
  if (ctx_size != -1) m_supported_variants[ctx_size].insert({variant, type});
}

std::string InferenceStep::str() const {
  return fmt::format("AR-{} CL-{} n_past={} n_kv={} n_process={} @ past_idx={} new_idx={}",
                     variant,
                     ctx_size,
                     n_past,
                     n_valid_kv,
                     n_process,
                     past_idx,
                     new_idx);
}

void KVManager::initComplete(int32_t max_ctx_size, std::string default_prefix) {
  m_max_ctx_size = max_ctx_size;

  if (m_cache_groups.size() == 0) {
    // Models without KV cache fall into this category. Create an empty CacheGroup.
    m_cache_groups.emplace(std::make_pair(std::string(""), CacheGroup(_env, "", false, {})));
    default_group = &m_cache_groups.at("");
  } else if (m_cache_groups.contains(default_prefix)) {
    default_group = &m_cache_groups.at(default_prefix);
  } else {
    State::error(fmt::format("No KV tensors found for default CacheGroup \"{}\"", default_prefix));
    return;
  }

  // Inspect if native execution mode is active.
  for (auto& [prefix, group] : m_cache_groups) {
    if (!group.m_isKvOutputNativeFormat.empty()) {
      m_isKvInputNativeFormat = true;
      break;
    }
  }

  // Register all graph indexes the KVManager is tracking
  // Also create a global set of KV$ across all cache groups, ordered by graph and tensor index
  // This populates m_graphs and m_cache
  std::map<int, std::map<uint32_t, std::pair<CacheGroup*, KVTensor*>>> tensor_ordering;
  for (auto& [prefix, group] : m_cache_groups) {
    for (auto& [graph_idx, graph_tensors] : group.m_tensors) {
      for (auto& tensor : graph_tensors) tensor_ordering[graph_idx][tensor.idx] = {&group, &tensor};
    }
  }

  for (auto& [graph_idx, graph_tensor_ordering] : tensor_ordering) {
    for (auto& [tensor_index, tensor_structs] : graph_tensor_ordering) {
      m_cache[graph_idx].push_back(tensor_structs);
    }
  }

  for (auto& [graph_idx, _] : m_cache) m_graphs.push_back(graph_idx);

  // Initialize the KV$ states to not busy
  {
    size_t numQueues = 1;
    if (m_threadpool) numQueues = m_threadpool->size();
    for (auto graph_idx : m_graphs) {
      auto& state = m_graph_state[graph_idx];
      state.jobSlices.reserve(numQueues);
      for (size_t i = 0; i < numQueues; i++) {
        state.jobSlices.emplace_back(std::make_unique<JobSlice>());
      }
      state.sync = 0;
    }
  }

  {
    if (m_supported_variants.empty()) {
      State::error(
          "Genie is not able to determine the context length for some of the graphs. Please name "
          "the graph properly.");
      return;
    }
    // Set the smallest context size and largest variant as a default start state
    int32_t first_ctx     = m_supported_variants.begin()->first;
    int32_t first_variant = m_supported_variants.begin()->second.rbegin()->arN;

    __DEBUG("Initializing to AR-{} CL-{}", first_variant, first_ctx);
    for (auto& [prefix, group] : m_cache_groups) group.updateVariant(first_variant, first_ctx);
  }

  for (auto& [prefix, group] : m_cache_groups) {
    if (!group.completeInit(m_qnn_api)) {
      State::fatal("Failed to initialize CacheGroups");
      return;
    }
  }

  if (m_traceLogger) {
    if (m_threadpool) {
      std::vector<std::thread::id> threadIds = m_threadpool->getThreadIds();
      for (const auto id : threadIds) {
        std::shared_ptr<genie::profiling::TraceLogger> logger =
            m_traceLogger->createSubLogger().lock();
        m_threadTraceLoggerMap.emplace(id, logger);
      }
    }
  }

  // The remaining code is purely for debug loggine. If logging is disabled, exit early
  if (!_env->logger() || GENIE_LOG_LEVEL_VERBOSE > _env->logger()->getMaxLevel()) return;

  __DEBUG("KVManager initialization complete wtih {} splits ", m_graphs.size());
  std::string variant_str = "";
  for (auto& [ctx_size, variants] : m_supported_variants)
    for (auto& variant : variants)
      variant_str += fmt::format("AR-{} CL-{}, ", variant.arN, ctx_size);
  __DEBUG("Supported configurations= [{}]", variant_str.substr(0, variant_str.size() - 2));

  for (auto& [prefix, group] : m_cache_groups) {
    __DEBUG("Group {}: nElem={} @ nBytes={} n_embed={} quantized={} scatter={}",
            prefix,
            group.n_elements,
            group.n_bytes,
            group.n_embed_dim,
            group.m_quantized,
            group.m_use_scatter);
    auto& clearVal = group.m_clear_value;
    __DEBUG("clear=({}u8, {}u16, {}u32)", clearVal.u8, clearVal.u16, clearVal.u32);

    std::string variant_map_str = "";
    for (auto& [global_variant, group_variant] : group.m_variant_map)
      variant_map_str += fmt::format("{} -> {}, ", global_variant, group_variant);
    __DEBUG(
        "Group {} variants = [{}]", prefix, variant_map_str.substr(0, variant_map_str.size() - 2));

    __DEBUG("#Splits = {}", group.m_tensors.size());
    for (auto& [graph_index, graph_tensors] : group.m_tensors) {
      __DEBUG("Graph[{}] #Tensors = {}", graph_index, graph_tensors.size());
      for (auto& tensor : graph_tensors) {
        __DEBUG(
            "\tlayer={} head={} n_heads={} key={:p} val={:p} anchor=({:p}:{:p}->{:p}:{:p}, {}) "
            "scores={:p}",
            tensor.idx >> 16,
            tensor.idx & 0xffff,
            tensor.n_heads,
            fmt::ptr(tensor.key_buf),
            fmt::ptr(tensor.val_buf),
            fmt::ptr(tensor.anchor_tensor_in),
            fmt::ptr(tensor.anchor_in),
            fmt::ptr(tensor.anchor_tensor_out),
            fmt::ptr(tensor.anchor_out),
            tensor.anchor_offset,
            fmt::ptr(tensor.scores));
      }
    }
  }
}

bool KVManager::prepareInferenceStrategy(int32_t n_inputs, bool output_all) {
  GENIE_TRACE();
  // The goal of this is to minimize latency
  // This includes heuristics for minimizing number of iterations and also using smallest ctx_size
  // Enforce maximum context size
  if (m_n_past + n_inputs > m_max_ctx_size) {
    State::error("Requested input exceeds the maximum context size.");
    throw genie::ContextLimitException("Context Size was exceeded.");
  }

  // Assumptions:
  // Lower ctx_size runs faster.
  // Different variants at the same ctx_size are close in time
  // Minimizing latency means picking smallest ctx_size and reducing number of iterations
  // Switching cost can be upto 100ms so avoid switches as much as possible
  // TODO: Once token_history_enabled=false (on embedding input or longcontext), disable AR-c
  InferenceStrategy strategy;

  int32_t n_past     = m_n_past;
  int32_t n_valid_kv = static_cast<int32_t>(default_group->m_n_valid_kv);

  // This is a simple lambda function that returns the smallest choice larger or equal to n
  // If no such choice exists, the largest choice is returned
  auto pick = [output_all](int32_t n, std::set<GraphVariantInfo>& choices) -> int32_t {
    int32_t upperBound = INT_MAX;
    int32_t lowerBound = 0;
    for (auto choice : choices) {
      if (output_all && choice.type == GraphType::DECODER_PREFILL) {
        continue;
      }
      if (choice.arN >= n && choice.arN < upperBound) {
        upperBound = choice.arN;
      } else if (choice.arN < n && choice.arN > lowerBound) {
        lowerBound = choice.arN;
      }
    }
    // didnt find anything more than n
    if (upperBound == INT_MAX) {
      return lowerBound;
    }
    return upperBound;
  };

  auto iter_ctx   = m_supported_variants.lower_bound(n_valid_kv);  // Pick the smallest CL
  int32_t variant = pick(n_inputs, iter_ctx->second);              // Pick the smallest variant
  // If we exceed CL (on both AR-c and non AR-c graphs), switch to a larger CL (if available)
  while (((iter_ctx->first != variant && (n_valid_kv + variant > iter_ctx->first)) ||
          ((iter_ctx->first == variant && (n_past + n_inputs > iter_ctx->first)))) &&
         (iter_ctx->first != m_supported_variants.rbegin()->first)) {
    iter_ctx++;  // If inference exceeds CL and larger CL is available, switch to a larger CL
    variant = pick(n_inputs, iter_ctx->second);  // Re-pick the variant for the larger CL
  }

  int32_t ctx_size = iter_ctx->first;
  int32_t n_remain = n_inputs;

  if (ctx_size == variant) {  // For AR-ctx graphs (i.e. bertcache), past tokens are reprocessed
    n_remain += n_past;
    n_past = n_valid_kv = 0;

    if (n_remain > ctx_size) {
      State::error("Input is too large for maximum context length available");
      throw genie::ContextLimitException("Context Size was exceeded.");
    }
  }
  while (n_remain > 0) {
    // If the iteration would exceed , and a larger CL is available, then switch to larger CL
    // Calculate how many inputs we can process in this iteration
    int32_t n_process     = std::min(n_remain, variant);
    int32_t cacheBoundary = ctx_size - variant;
    if (m_isKvInputNativeFormat) {
      cacheBoundary =
          ctx_size - static_cast<int32_t>(std::ceil(static_cast<double>(variant) / 32.0) * 32);
    }
    if (variant != ctx_size && n_valid_kv + variant > cacheBoundary) {
      auto it = m_supported_variants.lower_bound(ctx_size + 1);
      if (it != m_supported_variants.end()) {  // If a larger CL is available, switch to it
        ctx_size  = it->first;
        variant   = pick(n_remain, it->second);
        n_process = std::min(n_remain, variant);
      }
    }

    const int32_t past_dim = ctx_size - variant;
    strategy.emplace_back(variant, ctx_size, n_past, n_valid_kv, n_process, 0, past_dim);
    strategy.back().new_idx = default_group->manager->getIndexForNewKV(strategy.back());

    // Update the status for next iteration
    n_past += n_process;
    n_valid_kv += n_process;
    n_remain -= n_process;
    // At this point, if we are still exceeding CL, then longcontext must be enabled
    if (n_remain > 0 && (variant != ctx_size && n_valid_kv > past_dim)) {
      if (default_group->context_manager->params.mode == LongContextParams::DISABLED) {
        State::error("Input is too large and cannot be processed");
        throw genie::ContextLimitException("Context Size was exceeded.");
      } else {
        n_valid_kv = past_dim;
      }
    }
  }

  // Post-process. The last step must contain a logit producing variant
  {
    InferenceStep& last_step = strategy.back();
    if (!m_logit_variants.contains({last_step.variant, last_step.ctx_size})) {
      __DEBUG("Post-processing so that last step contains a logit containing variant");
      for (auto& [new_variant, new_ctx] : m_logit_variants) {
        int32_t n_process, new_n_past, new_n_valid_kv;
        if (last_step.n_process <= new_variant) {
          n_process = last_step.n_process;
          strategy.pop_back();
          last_step      = strategy.back();
          new_n_past     = last_step.n_past + last_step.n_process;
          new_n_valid_kv = last_step.n_valid_kv + last_step.n_process;
        } else {
          n_process = last_step.n_process;
          last_step.n_process -= new_variant;
          new_n_past     = last_step.n_past + n_process - new_variant;
          new_n_valid_kv = last_step.n_valid_kv + n_process - new_variant;
          n_process      = new_variant;
        }
        strategy.push_back({new_variant,
                            new_ctx,
                            new_n_past,
                            new_n_valid_kv,
                            n_process,
                            0,
                            new_ctx - new_variant});
        strategy.back().new_idx = default_group->manager->getIndexForNewKV(strategy.back());
      }
    }
  }

  if (strategy.empty()) {
    __TRACE("Failed to prepare the Inference strategy");
    return false;
  }

  m_strategy          = strategy;
  m_strategy_cur_step = 0;
  m_strategy_active   = true;
  __TRACE("Inference strategy prepared.");
  int step_idx = 0;
  for (InferenceStep& step : m_strategy) __TRACE("Step {}: {}", step_idx++, step.str());

  // Check global states and make sure they align with the first step in the strategy
  InferenceStep& step = m_strategy.front();
  if (cur_variant() != step.variant || cur_ctx() != step.ctx_size) {
    setActiveVariant(step.variant, step.ctx_size);
  }

  return true;
}

bool KVManager::nextInferenceStep(InferenceStep& step) {
  // This is equivalent to an EOF. The current strategy is now complete
  if (m_strategy_cur_step >= m_strategy.size()) {
    m_strategy_active = false;
    m_strategy.clear();
    m_strategy_cur_step = 0;
    return false;
  }

  // Get the next state and update global states accordingly
  step         = m_strategy.at(m_strategy_cur_step++);
  m_islastStep = (m_strategy_cur_step >= m_strategy.size());

  return true;
}

bool KVManager::completeInferenceStep() { return true; }

bool KVManager::block(Scope scope) {
  if (scope.is_per_graph() && !m_cache.contains(scope.graph_idx)) return true;

  if (scope.is_global()) {
    for (auto graph_idx : m_graphs) block(Scope::per_graph(graph_idx));
    return true;
  }
  GENIE_TRACE();

  __DEBUG("Blocking for graph {}", scope.graph_idx);
  GraphState& state = m_graph_state.at(scope.graph_idx);
  while (state.sync != 0) {
  }

  return true;
}

bool KVManager::unblock(Scope scope) {
  GENIE_TRACE();
  // All blocks during inference MUST go through m_strategy
  // If no strategy is active, the block must be for something else, e.g. saving/dumping the cache
  if (scope.is_global() || !m_strategy_active) return true;

  // If the graph has no registered tensors
  if (!m_cache.contains(scope.graph_idx)) return true;

  // Check if the next KV$ update needs to be processed
  // This is disabled for the final step, unless only 1 input was processed
  InferenceStep& step       = m_strategy.at(m_strategy_cur_step - 1);
  const bool is_final_step  = m_strategy_cur_step >= m_strategy.size();
  const bool process_update = !is_final_step || (step.n_process == 1);

  const bool is_first_graph = scope.graph_idx == m_graphs.front();
  const bool is_last_graph  = scope.graph_idx == m_graphs.back();

  if (process_update) {
    // At this point, we should never be under scope.is_global()
    // For the first graph, figure out which updates that are necessary and update global states
    // For the last graph, cleanup cached variables
    if (is_first_graph) {
      const int32_t new_variant = is_final_step
                                      ? m_supported_variants.at(step.ctx_size).begin()->arN
                                      : m_strategy.at(m_strategy_cur_step).variant;
      const int32_t new_ctx     = is_final_step ? -1 : m_strategy.at(m_strategy_cur_step).ctx_size;

      if (!processUpdate(step, step.n_past + step.n_process, new_variant, new_ctx)) {
        return false;
      }
      m_counter++;
    }
  }

  // afterExecution function mainly updates the anchor tensor, regardless of process_update
  for (auto& [prefix, group] : m_cache_groups)
    group.context_manager->afterExecution(scope.graph_idx, step);

  if (process_update) {
    // Use cached lambda to process the update
    prepareJob(scope, {"splitUpdate", m_cached_update});

    if (is_last_graph) {
      m_cached_update = nullptr;
    }
  } else {
    m_last_inference = step;
  }

  return true;
}

bool KVManager::setActiveVariant(int32_t variant, int32_t ctx_size) {
  const int32_t cur_variant = default_group->m_cur_variant;
  const int32_t cur_ctx     = default_group->m_cur_ctx;

  variant  = (variant == -1) ? cur_variant : variant;
  ctx_size = (ctx_size == -1) ? cur_ctx : ctx_size;

  // AR-c graphs do not take any KV$ input, so this simplifies to a no-op
  if (variant == ctx_size) return true;

  std::map<std::string, UpdateStrategy> group_moves;
  for (auto& [prefix, group] : m_cache_groups) {
    group_moves[prefix] = group.context_manager->processMove(variant, ctx_size);

    // Check if there were any errors, likely in cases where KV$ exceeds budget w/o LongContext
    const auto& moves = group_moves.at(prefix);
    if (moves.mode == UpdateStrategy::ERROR) {
      State::error("KV$ exceeded budget, but longcontext is not enabled for CacheGroup " + prefix);
      return false;
    }

    // Check if the reshape requires a blocking preparation, for e.g. running the KeyDiff scorer
    // Note that a global block must also be enforced to ensure all KV$ updates are synced
    if (moves.update_preparer != nullptr) {
      if (!block(Scope::global())) return false;
      if (!moves.update_preparer()) return false;
    }
  }

  const auto reshape_job = [&, cur_variant, cur_ctx, variant, ctx_size, group_moves](
                               CacheGroup& group, KVTensor& cache) {
    const auto& [group_variant, group_ctx] = group.getGroupVariant(cur_variant, cur_ctx);
    const auto& [new_variant, new_ctx]     = group.getGroupVariant(variant, ctx_size);

    if (group_variant == new_variant && group_ctx == new_ctx) return;

    // Evict if necessary
    auto& moves = group_moves.at(group.m_prefix);
    if (moves.mode != UpdateStrategy::NONE)
      group.manager->moveKV(group, cache, group_variant, group_ctx, moves);

    // Reshape the cache
    group.manager->reshapeCache(group, cache, group_variant, group_ctx, new_variant, new_ctx);
  };

  __DEBUG("reshapeCache(AR-{} CL-{} -> AR-{} CL-{})", cur_variant, cur_ctx, variant, ctx_size);
  prepareJob(Scope::global(), {"reshapeCache", reshape_job});
  return true;
}

void KVManager::prepareJob(Scope scope, Job job) {
  GENIE_TRACE();
  if (scope.is_per_graph() && !m_cache.contains(scope.graph_idx)) return;

  // For global jobs, split them into per graph
  if (scope.is_global()) {
    for (auto graph_idx : m_graphs) prepareJob(Scope::per_graph(graph_idx), job);
    return;
  }

  const int32_t graph_idx = scope.graph_idx;

  // Some splits may not contain KV$
  if (!m_cache.contains(graph_idx)) return;

  GraphState* state = &m_graph_state.at(graph_idx);
  auto kv_tensors   = &m_cache.at(graph_idx);

  if (m_threadpool) {
    const size_t n_tensors         = kv_tensors->size();
    const size_t n_slices          = state->jobSlices.size();
    const size_t tensors_per_slice = n_tensors / n_slices;
    const size_t remainder         = n_tensors - (tensors_per_slice * n_slices);
    state->sync += static_cast<int32_t>(n_slices);
    size_t startIdx = 0;
    size_t endIdx   = 0;
    for (size_t tidx = 0; tidx < n_slices; tidx++) {
      startIdx = endIdx;
      endIdx   = startIdx + tensors_per_slice;
      if (tidx < remainder) {
        endIdx++;
      }
      std::lock_guard<std::mutex> jobLock(state->jobSlices[tidx]->queuedMutex);
      const auto update_job = [this, kv_tensors, startIdx, endIdx, n_tensors, job]() {
        const std::thread::id threadId = std::this_thread::get_id();
        const auto& traceLogger        = m_threadTraceLoggerMap.contains(threadId)
                                             ? m_threadTraceLoggerMap.at(threadId)
                                             : nullptr;
        auto iter                      = kv_tensors->begin() + static_cast<long>(startIdx);
        auto end = kv_tensors->begin() + static_cast<long>(std::min(endIdx, n_tensors));
        for (; iter < end; iter++) {
          iter->second->setTraceLogger(traceLogger);
          iter->second->traceNamespace = iter->first->manager->getTraceNamespace();
          job.update_function(*iter->first, *iter->second);
        }
      };
      state->jobSlices[tidx]->queued.push(update_job);
    }
    // Add update requests to the threadpool.
    queueJob(scope.graph_idx);
  } else {
    // If this is a single-threaded environment, run the entire job immediately.
    for (auto iter = kv_tensors->begin(); iter < kv_tensors->end(); iter++) {
      iter->second->setTraceLogger(getTraceLogger());
      iter->second->traceNamespace = iter->first->manager->getTraceNamespace();
      job.update_function(*iter->first, *iter->second);
    }
  }
}

void KVManager::queueJob(int32_t graph_idx) {
  GraphState* state     = &m_graph_state.at(graph_idx);
  const auto requestJob = [state]() {
    // Process any available job queues
    for (auto& jobSlice : state->jobSlices) {
      std::unique_lock<std::mutex> runningLock(jobSlice->runningMutex, std::try_to_lock);
      if (runningLock) {
        do {
          // Run all jobs.
          while (!jobSlice->running.empty()) {
            auto job = jobSlice->running.front();
            jobSlice->running.pop();
            job();
            state->sync--;
          }
          // Quickly flush queued jobs from the main thread to the running jobs on this thread.
          // This is a fast operation which frees up the main thread to queue more jobs ASAP.
          std::unique_lock<std::mutex> queuedLock(jobSlice->queuedMutex);
          while (!jobSlice->queued.empty()) {
            auto job = jobSlice->queued.front();
            jobSlice->queued.pop();
            jobSlice->running.push(job);
          }
        } while (!jobSlice->running.empty());
      }
    }
  };

  size_t n_threads = m_threadpool->size();
  std::vector<std::function<void()>> kvUpdateRequests;
  for (size_t tidx = 0; tidx < n_threads; tidx++) {
    kvUpdateRequests.push_back(requestJob);
  }
  m_threadpool->enqueue(kvUpdateRequests);
}

// processUpdate consumes the last known inference (m_last_inference) to generate update jobs
// It is only called in two places - dispatchUpdate (global update) and on first_split unblock()
// CAUTION: m_last_inference is destroyed by this function, since KV$ can only be consumed once
bool KVManager::processUpdate(
    InferenceStep& step, int32_t n_past, int32_t new_variant, int32_t new_ctx, Mask& mask) {
  GENIE_TRACE();
  uint32_t n_update = static_cast<uint32_t>(n_past - m_n_past);

  __DEBUG("KV$ Update {}/{} @ AR-{} CL-{}", n_update, step.n_process, step.variant, step.ctx_size);

  if (n_update > static_cast<uint32_t>(step.n_process)) {
    State::error("KV update count exceeds the total processed inputs from last inference");
    return false;
  }

  if ((!mask.empty()) && (mask.size() != static_cast<uint32_t>(step.n_process))) {
    State::error(fmt::format(
        "Invalid selection mask size. Found {} but expected 0 or {}", mask.size(), step.n_process));
    return false;
  }

  std::vector<int32_t> src_idxes(n_update);  // Select which KV$ needs to be updated
  if (mask.empty()) {  // If the mask is empty, the sequential range [0, n_update] is copied
    std::iota(src_idxes.begin(), src_idxes.end(), 0);
  } else {  // If a mask is supplied, KV$ is selectively copied
    for (uint32_t i = 0, j = 0; i < static_cast<uint32_t>(step.n_process); i++) {
      if (mask[i]) {
        src_idxes[j++] = static_cast<int32_t>(i);
      }
    }
  }

  if (new_variant == -1) new_variant = step.variant;
  if (new_ctx == -1) new_ctx = step.ctx_size;

  std::map<std::string, UpdateStrategy> group_updates;
  for (auto& [prefix, group] : m_cache_groups) {
    if (group.m_cur_variant == group.m_cur_ctx) {
      // Special handling for AR-c models. Eviction may be necessary to make room once reshaped
      group.m_n_valid_kv    = step.n_process;
      group_updates[prefix] = group.context_manager->processMove(new_variant, new_ctx);
    } else {
      group_updates[prefix] = group.context_manager->processUpdate(step, src_idxes);
    }

    // Check if there were any errors, likely in cases where KV$ exceeds budget w/o LongContext
    const auto& updates = group_updates.at(prefix);
    if (updates.mode == UpdateStrategy::ERROR) {
      State::error("KV$ exceeded budget, but longcontext is not enabled for CacheGroup " + prefix);
      throw genie::ContextLimitException("Context Size was exceeded.");
    }

    // Check if the update requires a blocking preparation, for e.g. running the KeyDiff scorer
    // Note that a global block must also be enforced to ensure all KV$ updates are synced
    if (updates.update_preparer != nullptr) {
      if (!block(Scope::global())) return false;
      if (!updates.update_preparer()) return false;
    }

    group.updateVariant(new_variant, new_ctx);
  }

  // Log the updates for debugging
  if (_env->logger() && GENIE_LOG_LEVEL_VERBOSE <= _env->logger()->getMaxLevel()) {
    __DEBUG("Processing updates for InferenceStep {}", step.str());
    for (auto& [prefix, updates] : group_updates) {
      __DEBUG("CacheGroup prefix={}", prefix);
      if (updates.mode == UpdateStrategy::CACHED) {
        for (auto& [src, dst, count] : updates.steps) {
          __DEBUG("\tsource={} destination={} count={}", src, dst, count);
        }
      } else if (updates.mode == UpdateStrategy::DYNAMIC) {
        __DEBUG("\tUpdateStrategy created with lambdas");
      }

      const auto& [n_old, cl_old] =
          m_cache_groups.at(prefix).getGroupVariant(step.variant, step.ctx_size);
      const auto& [n_new, cl_new] = m_cache_groups.at(prefix).getGroupVariant(new_variant, new_ctx);
      if (n_old != n_new || cl_old != cl_new)
        __DEBUG("\tReshape[{}] AR-{} CL-{} -> AR-{} CL-{}", prefix, n_old, cl_old, n_new, cl_new);
    }
  }

  m_cached_update = [&,
                     cur_variant   = step.variant,
                     cur_ctx       = step.ctx_size,
                     new_variant   = new_variant,
                     new_ctx       = new_ctx,
                     batch_size    = getBatchSize(),
                     group_updates = std::move(group_updates)](CacheGroup& group, KVTensor& cache) {
    const auto& [group_cur_variant, group_cur_ctx] = group.getGroupVariant(cur_variant, cur_ctx);
    const auto& [group_new_variant, group_new_ctx] = group.getGroupVariant(new_variant, new_ctx);

    auto& updates = group_updates.at(group.m_prefix);
    group.manager->setBatchSize(batch_size);
    if (group_cur_variant != group_cur_ctx) {
      group.manager->updateKV(group, cache, group_cur_variant, group_cur_ctx, updates);
    }

    if (group_cur_variant != group_new_variant || group_cur_ctx != group_new_ctx) {
      group.manager->reshapeCache(
          group, cache, group_cur_variant, group_cur_ctx, group_new_variant, group_new_ctx);
    }
  };

  m_last_inference = {new_variant, new_ctx, n_past, default_group->m_n_valid_kv, 0, 0, 0};
  m_n_past         = n_past;
  return true;
}

bool KVManager::dispatchUpdate(int32_t n_past, Mask& mask) {
  // Assume this is a Scope::GLOBAL call since it is only called externally
  m_counter++;

  __TRACE("n_past: {}, m_n_past: {}", n_past, m_n_past);
  if (m_cache.empty()) return true;

  // Clear the cache
  if (n_past == 0) {
    __DEBUG("clearCache()");
    prepareJob(Scope::global(), {"clear", [&](CacheGroup& group, KVTensor& cache) {
                                   group.manager->clear(group, cache);
                                 }});

    m_n_past = 0;
    // Revert to the default start state of smallest CL, largest variant
    const int32_t min_ctx     = m_supported_variants.begin()->first;
    const int32_t max_variant = m_supported_variants.begin()->second.rbegin()->arN;
    m_last_inference          = {max_variant, min_ctx, 0, 0, 0, 0, 0};

    // Reset token eviction state and queues
    for (auto& [prefix, group] : m_cache_groups) {
      group.resetState();
      group.updateVariant(max_variant, min_ctx);
    }
    return true;
  }

  if (n_past == m_n_past) {
    return true;
  }

  // Requested n_past is smaller, so invoke reduction of KV$
  if (n_past < m_n_past) {
    InferenceStep& step = m_last_inference;
    if (!mask.empty()) {
      State::error("Selective KV$ removal not supported");
      return false;
    }

    std::map<std::string, UpdateStrategy> group_clears;
    for (auto& [prefix, group] : m_cache_groups) {
      group_clears[prefix] = group.context_manager->processReduce(m_n_past, n_past);

      // Check if there were any errors, likely in cases where KV$ exceeds budget w/o LongContext
      if (group_clears.at(prefix).mode == UpdateStrategy::ERROR) {
        State::error("KV$ removal is disabled after longcontext triggers for CacheGroup " + prefix);
        return false;
      }
    }

    const auto remove_job = [&, variant = step.variant, ctx_size = step.ctx_size, group_clears](
                                CacheGroup& group, KVTensor& cache) {
      const auto& [group_variant, group_ctx] = group.getGroupVariant(variant, ctx_size);
      auto& clears                           = group_clears.at(group.m_prefix);
      group.manager->reduceKV(group, cache, group_variant, group_ctx, clears);
    };

    __DEBUG("reduce(AR-{} CL-{}, n_past={} -> {})", step.variant, step.ctx_size, m_n_past, n_past);
    prepareJob(Scope::global(), {"remove", remove_job});

    m_n_past = n_past;
    return true;
  }

  // Requested n_past is larger. This involves accepting KV$ into the cache

  // dispatchUpdate is explicitly called by Dialog after prompt processing OR during generation
  // Either way, most likely the next inference occurs during the generation phase
  // In this case, the smallest variant is needed. Hence, potentially we can proactively switch
  const int32_t min_variant = m_supported_variants.at(m_last_inference.ctx_size).begin()->arN;
  if (!processUpdate(m_last_inference, n_past, min_variant, -1, mask)) {
    return false;
  }

  prepareJob(Scope::global(), {"accept", m_cached_update});
  m_cached_update = nullptr;

  return true;
}

size_t KVManager::loadKVCache(const std::string& filename) {
  GENIE_TRACE();
  __DEBUG("KVManager::loadKVCache {}", filename);

  // DLC record paths (record://) are resolved via ResourceManager, not the filesystem,
  // mirroring the same pattern used in LoraAdapter::LoraAdapter().
  std::shared_ptr<uint8_t> dlcBuffer;
  std::unique_ptr<std::istream> streamOwner;
  std::istream* handle = nullptr;

  auto resourceManager = _env->getResourceManager();

  if (resourceManager->isDLCRecord(filename)) {
    const size_t bufferSize = resourceManager->getBufferSize(filename);
    if (!resourceManager->getBuffer(filename, dlcBuffer)) {
      State::error(fmt::format("Failed to read KV cache DLC record: {}", filename));
      return 0;
    }
    // Wrap the DLC buffer in an in-memory stream so loadCache() can read from it uniformly
    streamOwner = std::make_unique<std::istringstream>(
        std::string(reinterpret_cast<const char*>(dlcBuffer.get()), bufferSize),
        std::ios::in | std::ios::binary);
    handle = streamOwner.get();
  } else {
    auto fileStream = std::make_unique<std::ifstream>(filename, std::ios::in | std::ios::binary);
    if (fileStream->fail()) {
      State::error(fmt::format("Error opening file {}", filename));
      return 0;
    }
    handle      = fileStream.get();
    streamOwner = std::move(fileStream);
  }

  CacheFileSpec spec;
  handle->read(reinterpret_cast<char*>(&spec), sizeof(spec));
  if (spec.magic != 0xC0DE) {
    State::error(fmt::format("Incorrect magic number. 0xC0DE. Found {:#x}", spec.magic));
    return 0;
  }

  __DEBUG(
      "KVManager::loadKVCache {{ num_tensors {}, magic {:x}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      spec.num_tensors,
      spec.magic,
      int(spec.dtype),
      spec.n_heads,
      spec.embed_dim,
      spec.update_size);

  for (auto& [graph_index, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      group->manager->loadCache(*group,
                                *cache,
                                handle,
                                true,
                                spec.update_size,
                                spec.n_heads,
                                group->m_cur_variant,
                                group->m_cur_ctx);
    }
  }

  for (auto& [graph_index, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      group->manager->loadCache(*group,
                                *cache,
                                handle,
                                false,
                                spec.update_size,
                                spec.n_heads,
                                group->m_cur_variant,
                                group->m_cur_ctx);
    }
  }

  m_counter++;
  m_n_past = spec.update_size;

  for (auto& [_, group] : m_cache_groups) group.m_n_valid_kv = spec.update_size;

  return spec.update_size;
}

bool KVManager::dumpKVCache(const std::string& filename) {
  GENIE_TRACE();
  __DEBUG("KVManager::dumpKVCache {}", filename);
  std::ofstream handle(filename, std::ios::out | std::ios::binary);
  if (handle.fail()) {
    State::error(fmt::format("Error opening file {}", filename));
    return false;
  }

  uint32_t max_n_heads = 0;
  uint32_t n_tensors   = 0;
  for (auto& [graph_index, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      if (cache->n_heads > max_n_heads) {
        max_n_heads = cache->n_heads;
      }
      n_tensors++;
    }
  }

  CacheFileSpec spec(2 * n_tensors,
                     0xc0de,
                     CacheFileSpec::UINT8_T,
                     0u,
                     max_n_heads,
                     default_group->n_embed_dim,
                     static_cast<uint16_t>(default_group->m_n_valid_kv));

  handle.write(reinterpret_cast<char*>(&spec), static_cast<std::streamsize>(sizeof(spec)));

  __DEBUG(
      "KVManager::dumpKVCache {{ num_tensors {}, magic {:x}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      spec.num_tensors,
      spec.magic,
      int(spec.dtype),
      spec.n_heads,
      spec.embed_dim,
      spec.update_size);

  for (auto& [graph_index, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      group->manager->dumpCache(*group,
                                *cache,
                                &handle,
                                true,
                                spec.update_size,
                                max_n_heads,
                                group->m_cur_variant,
                                group->m_cur_ctx);
    }
  }

  for (auto& [graph_index, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      group->manager->dumpCache(*group,
                                *cache,
                                &handle,
                                false,
                                spec.update_size,
                                max_n_heads,
                                group->m_cur_variant,
                                group->m_cur_ctx);
    }
  }

  std::vector<double> scales;
  for (auto& [graph_index, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      scales.push_back(cache->key_quant.scale);
    }
  }
  for (auto& [graph_index, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      scales.push_back(cache->value_quant.scale);
    }
  }
  handle.write(reinterpret_cast<char*>(scales.data()),
               static_cast<std::streamsize>(scales.size() * sizeof(double)));

  handle.close();

  return true;
}

bool KVManager::dumpKVCache(Buffer* kvBuff) {
  GENIE_TRACE();
  int32_t n_embed_dim  = 0;
  uint32_t max_n_heads = 0;
  uint32_t n_tensors   = 0;
  for (auto& [graph_idx, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      if (cache->n_heads > max_n_heads) {
        max_n_heads = cache->n_heads;
      }
      n_embed_dim = group->n_embed_dim;
      n_tensors++;
    }
  }

  CacheFileSpec spec(2 * n_tensors,
                     0xc0de,
                     CacheFileSpec::UINT8_T,
                     0u,
                     max_n_heads,
                     n_embed_dim,
                     static_cast<uint16_t>(default_group->m_n_valid_kv));

  kvBuff->appendBuffer(reinterpret_cast<uint8_t*>(&spec), sizeof(spec));

  __DEBUG(
      "KVManager::dumpKVCache {{ num_tensors {}, magic {:x}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      spec.num_tensors,
      spec.magic,
      int(spec.dtype),
      spec.n_heads,
      spec.embed_dim,
      spec.update_size);

  for (auto& [graph_index, graph_tensor_structs] : m_cache)
    for (auto& [group, cache] : graph_tensor_structs)
      group->manager->dumpCache(*group,
                                *cache,
                                kvBuff,
                                true,
                                spec.update_size,
                                max_n_heads,
                                group->m_cur_variant,
                                group->m_cur_ctx);

  for (auto& [graph_index, graph_tensor_structs] : m_cache)
    for (auto& [group, cache] : graph_tensor_structs)
      group->manager->dumpCache(*group,
                                *cache,
                                kvBuff,
                                false,
                                spec.update_size,
                                max_n_heads,
                                group->m_cur_variant,
                                group->m_cur_ctx);

  std::vector<double> scales;

  for (auto& [graph_index, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      scales.push_back(cache->key_quant.scale);
    }
  }
  for (auto& [graph_index, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      scales.push_back(cache->value_quant.scale);
    }
  }

  kvBuff->appendBuffer(reinterpret_cast<uint8_t*>(scales.data()), scales.size() * sizeof(double));

  return true;
}

bool KVManager::getCacheSpec(CacheFileSpec& spec) {
  int32_t n_embed_dim  = 0;
  uint32_t max_n_heads = 0;
  uint32_t n_tensors   = 0;
  for (auto& [graph_idx, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      if (cache->n_heads > max_n_heads) {
        max_n_heads = cache->n_heads;
      }
      n_embed_dim = group->n_embed_dim;
      n_tensors++;
    }
  }

  spec.num_tensors = 2 * n_tensors;
  spec.magic       = 0xc0de;
  spec.dtype       = CacheFileSpec::UINT8_T;
  spec.pad8_t      = 0x0;
  spec.n_heads     = max_n_heads;
  spec.embed_dim   = static_cast<uint16_t>(n_embed_dim);
  spec.update_size = static_cast<uint16_t>(default_group->m_n_valid_kv);

  __DEBUG(
      "KVManager::getCacheSpec {{ num_tensors {}, magic {:x}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      spec.num_tensors,
      spec.magic,
      int(spec.dtype),
      spec.n_heads,
      spec.embed_dim,
      spec.update_size);

  return true;
}

bool KVManager::getKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  uint32_t curr_layer = 0;
  for (auto& [graph_idx, graph_tensor_structs] : m_cache) {
    for (auto& [group, cache] : graph_tensor_structs) {
      if (curr_layer == layer) {
        group->manager->dumpHead(
            *group, *cache, head, spec.update_size, group->m_cur_variant, group->m_cur_ctx, data);
        scale[0] = cache->key_quant.scale;
        scale[1] = cache->value_quant.scale;
        return true;
      }
      curr_layer++;
    }
  }

  return false;
}

}  // namespace qualla
