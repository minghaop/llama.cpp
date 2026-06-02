//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <set>

#include "Exception.hpp"
#include "QnnApi.hpp"
#include "context-manager.hpp"
#include "nsp-params.hpp"
#include "qualla/IOBuffer.hpp"
#include "qualla/detail/buffer.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/threadpool.hpp"
#include "qualla/env.hpp"

/**
 * A special macro intended for use in CacheManagers. CacheManager operations
 * are generally executed on background threads, so they must log trace events
 * in the KV tensor's traceLogger to avoid race conditions.
 */
#define GENIE_KV_TRACE(...)                                                                 \
  genie::profiling::FunctionTracer functionTracer__(*static_cast<const Traceable*>(&cache), \
                                                    __func__);

namespace qualla {

using VariantSpec = std::pair<int32_t, int32_t>;

// Inference Step is a simple struct defining all variables necessary to execute graph iteration
struct InferenceStep {
  int32_t variant{0};
  int32_t ctx_size{0};
  int32_t n_past{0};
  int32_t n_valid_kv{0};
  int32_t n_process{0};
  int32_t past_idx{0};
  int32_t new_idx{0};

  InferenceStep() = default;
  InferenceStep(int32_t _variant,
                int32_t _ctx_size,
                int32_t _n_past,
                int32_t _n_valid_kv,
                int32_t _n_process,
                int32_t _past_idx,
                int32_t _new_idx)
      : variant(_variant),
        ctx_size(_ctx_size),
        n_past(_n_past),
        n_valid_kv(_n_valid_kv),
        n_process(_n_process),
        past_idx(_past_idx),
        new_idx(_new_idx) {}

  std::string str() const;
};

// InferenceStrategy is an alias defined as a list of InfereceStep
using InferenceStrategy = std::vector<InferenceStep>;

// Alias selection mask for readability
using Mask = const std::vector<bool>;

// KV$ Move operations can be defined as a set of UpdateSteps
// 'count' KV$ entries are copied from a src index to a dst index
// For reductions, only 'src_idx' and 'count' are considered
struct UpdateStep {
  int32_t src_idx{0};
  int32_t dst_idx{0};
  size_t count{0};

  UpdateStep() {}

  UpdateStep(int32_t _src_idx, int32_t _dst_idx, size_t _count)
      : src_idx(_src_idx), dst_idx(_dst_idx), count(_count) {}
};

struct KVTensor;
struct CacheGroup;

struct UpdateStrategy {
  enum Mode { NONE, CACHED, DYNAMIC, ERROR } mode{NONE};
  std::vector<UpdateStep> steps;
  std::function<std::vector<UpdateStep>(KVTensor&, int32_t)> step_generator{nullptr};

  // This function is called under block() before the update is queued e.g. to run KeyDiff scorer
  std::function<bool(void)> update_preparer{nullptr};

  UpdateStrategy() {}
  UpdateStrategy(Mode _mode) : mode(_mode) {}

  const std::vector<UpdateStep> get(KVTensor& cache, int32_t head_idx) const {
    if (mode == CACHED) return steps;
    return step_generator(cache, head_idx);
  }
};

struct CacheManager : public State {
  CacheManager(std::shared_ptr<Env> env, bool useScatter)
      : State(nullptr), _env(env), m_useScatter(useScatter) {}
  virtual ~CacheManager() = default;

  // Virtual function to allow subclasses to setup internal variables after init completes
  virtual void completeInit(CacheGroup&) {}

  // Clear the cache completely
  virtual void clear(CacheGroup& group, KVTensor& cache) = 0;

  // Get the index for the starting past KV$ and the new KV$
  virtual int32_t getIndexForPastKV(InferenceStep& /*step*/) { return 0; }

  virtual int32_t getIndexForNewKV(InferenceStep& step) { return step.ctx_size - step.variant; }

  // Get the cache budget
  virtual int32_t getCacheBudget(const int32_t variant, const int32_t ctx_size) {
    return (variant != ctx_size) ? ctx_size - variant : ctx_size;
  }

  virtual void setBatchSize(size_t batch_size) { m_n_batch = batch_size; }

  // Update KV$ - copy entries from output buffer into the cache buffer
  virtual void updateKV(CacheGroup& group,
                        KVTensor& cache,
                        int32_t variant,
                        int32_t ctx_size,
                        const UpdateStrategy& updates) = 0;

  // Reduce KV$ - remove entries from the cache buffer
  virtual void reduceKV(CacheGroup& group,
                        KVTensor& cache,
                        int32_t variant,
                        int32_t ctx_size,
                        const UpdateStrategy& clear_idxes) = 0;

  // Move KV$ - move entries within the cache buffer
  virtual void moveKV(CacheGroup& group,
                      KVTensor& cache,
                      int32_t variant,
                      int32_t ctx_size,
                      const UpdateStrategy& move_idxes) = 0;

  // Reshape KV$ - convert AR-{cur_variant} CL-{cur_ctx} cache into AR-{new_variant} CL-{new_ctx}
  virtual void reshapeCache(CacheGroup& group,
                            KVTensor& cache,
                            int32_t cur_variant,
                            int32_t cur_ctx,
                            int32_t new_variant,
                            int32_t new_ctx) = 0;

  // Load KV$ - read KV$ from a flat file buffer into the cache buffer
  virtual void loadCache(CacheGroup& group,
                         KVTensor& cache,
                         std::istream* fs,
                         bool is_key,
                         int32_t n_valid,
                         uint32_t n_heads,
                         int32_t variant,
                         int32_t ctx_size) = 0;

  // Dump KV$ - write KV$ from the cache buffer into a flat file buffer
  virtual void dumpCache(CacheGroup& group,
                         KVTensor& cache,
                         std::ofstream* fs,
                         bool is_key,
                         int32_t n_valid,
                         uint32_t n_heads,
                         int32_t variant,
                         int32_t ctx_size) = 0;

  // Dump KV$ - write KV$ from the cache buffer into an in-memory cache
  virtual void dumpCache(CacheGroup& group,
                         KVTensor& cache,
                         Buffer* kvBuff,
                         bool is_key,
                         int32_t n_valid,
                         uint32_t n_heads,
                         int32_t variant,
                         int32_t ctx_size) = 0;

  virtual void dumpHead(CacheGroup& group,
                        KVTensor& cache,
                        uint32_t head,
                        int32_t n_valid,
                        int32_t variant,
                        int32_t ctx_size,
                        void* data) = 0;

 protected:
  std::shared_ptr<Env> _env;
  bool m_useScatter{true};
  size_t m_n_batch{1};  // actual number of queries
};

struct KVTensor : public genie::profiling::Traceable {
  uint32_t idx;
  uint8_t* key_buf{nullptr};  // Pointer to the Key Cache
  uint8_t* val_buf{nullptr};  // Pointer to the Value Cache
  uint32_t n_heads;

  QnnUtils::Tensor* key{nullptr};
  QnnUtils::Tensor* value{nullptr};

  // Quantization prameters for keys and values
  QuantParam key_quant;
  QuantParam value_quant;

  // Fields for the KeyDiff algorithm
  QnnUtils::Tensor* anchor_tensor_in{nullptr};
  QnnUtils::Tensor* anchor_tensor_out{nullptr};

  uint16_t anchor_offset{0};
  uint8_t* anchor_in{nullptr};
  uint8_t* anchor_out{nullptr};
  uint8_t* scores{nullptr};

  // Indices to evict for each head
  std::vector<std::queue<int32_t>> evict_idxes;

  const char* traceNamespace{nullptr};

  KVTensor(std::shared_ptr<genie::profiling::TraceLogger> traceLogger,
           uint32_t idx,
           QnnUtils::Tensor* k,
           QnnUtils::Tensor* v);

  virtual const char* getTraceNamespace() const override { return traceNamespace; }
};

struct CacheGroup {
  std::shared_ptr<Env> _env;
  std::string m_prefix;

  // Cache Manager
  int8_t n_bytes{1};       // Size of each element (in bytes)
  size_t n_elements{0};    // Size of the KV$ Buffer
  int32_t n_embed_dim{0};  // Embedding size of KV$
  bool m_quantized{true};
  bool m_use_scatter{true};

  union {
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
  } m_clear_value;  // Value used for clearing the cache

  std::unique_ptr<CacheManager> manager;
  std::unique_ptr<ContextManager> context_manager;

  int32_t m_n_valid_kv{0};    // Total number of "physical" KV$ tensors (i.e. actual KV$ in memory)
  int32_t m_cur_variant{-1};  // Current variant
  int32_t m_cur_ctx{-1};      // Current context length

  std::map<VariantSpec, VariantSpec> m_variant_map;

  std::map<int, std::vector<KVTensor>> m_tensors;  // Maps graph_index to its caches
  std::map<uint32_t, KVTensor*> m_tensor_index;
  std::map<VariantSpec, bool> m_isKvOutputNativeFormat;

  CacheGroup(std::shared_ptr<Env> env,
             std::string prefix,
             bool scatter,
             LongContextParams longcontext_params);

  // Translate a global variant (AR/CL) to its corresponding group variant (AR/CL)
  VariantSpec getGroupVariant(int32_t variant, int32_t ctx_size) const {
    VariantSpec global_variant = {variant, ctx_size};
    return (m_variant_map.contains(global_variant)) ? m_variant_map.at(global_variant)
                                                    : global_variant;
  }

  bool resetState() {
    context_manager->resetState();
    m_n_valid_kv = 0;
    return true;
  }

  bool updateVariant(int32_t variant, int32_t ctx_size) {
    std::tie(m_cur_variant, m_cur_ctx) = getGroupVariant(variant, ctx_size);
    return true;
  }

  void registerTensors(
      const std::map<int, std::map<uint32_t, std::array<std::pair<QnnUtils::Tensor*, size_t>, 4>>>&
          tensors);

  void registerKvOutputNativeFormat(std::map<VariantSpec, bool>& isKvOutputNativeFormat);

  bool completeInit(QnnApi* qnnApi);

  // Translates a global inference step for the current group
  // WARNING: Only call this at the time of inference, as it directly uses m_n_valid_kv
  InferenceStep translateInferenceStep(InferenceStep step);
};

// Each KVTensor is independent to all other KVTensors. A Job function operates on one KVTensor
struct Job {
  std::string name;
  std::function<void(CacheGroup&, KVTensor&)> update_function;
  // Maybe add an estimated "cost" here
};

// Jobs in a JobSlice queue must be run sequentially, but are
// always independent of the Jobs in another JobSlice's queue.
//
// The main thread requests KV$ updates by adding jobs to
// JobSlice::queued. Worker threads will attempt to lock
// queuedMutex, then move the queued jobs to the running queue,
// then unlock queuedMutex. This allows the main thread to continue
// queueing work which will be picked up by a subsequent iteration
// of the worker thread. Meanwhile, the worker thread can flush
// the running queue.
struct JobSlice {
  std::mutex queuedMutex;
  std::mutex runningMutex;
  std::queue<std::function<void()>> queued;
  std::queue<std::function<void()>> running;
};

// The KVManager uses Scope to easily specify whether a KV$
// operation should apply to all graphs or a specific graph.
struct Scope {
  enum ScopeType : uint8_t {
    GLOBAL,    // [Default] Apply the operation to ALL KV$ tensors
    PER_GRAPH  // Apply the operation to one graph (by index)
  };

  ScopeType scope{GLOBAL};
  int32_t graph_idx{-1};

  static Scope global() { return Scope(Scope::GLOBAL, -1); }
  static Scope per_graph(int32_t graph_idx) { return {Scope::PER_GRAPH, graph_idx}; }

  bool is_global() { return scope == Scope::GLOBAL; }
  bool is_per_graph() { return scope == Scope::PER_GRAPH; }

 private:
  // Private constructor. Scope must be constructed using the static methods
  Scope(ScopeType _scope, int32_t _graph_idx) : scope(_scope), graph_idx(_graph_idx) {}
};

class KVManager : public State, public IOTensor {
 protected:
  std::shared_ptr<Env> _env;  // Reference to global environment for logging

  std::shared_ptr<ThreadPool> m_threadpool;  // Threadpool for async background processing
  std::map<std::thread::id, std::shared_ptr<genie::profiling::TraceLogger>>
      m_threadTraceLoggerMap;  // Per-thread trace loggers
  QnnApi* m_qnn_api{nullptr};
  // KVManager parameters. These are fixed and assumed not to change
  // bool strict_mode; // Manually clear out the buffer

  int32_t m_max_ctx_size{0};  // Maximum context length

  struct GraphVariantInfo {
    int32_t arN;
    GraphType type;

    // Overload operator< for comparison
    bool operator<(const GraphVariantInfo& other) const {
      if (arN != other.arN) {
        return arN < other.arN;
      }
      return type < other.type;
    }
  };

  std::vector<int32_t> m_graphs;  // List of graph indexes
  std::map<int32_t, std::vector<std::pair<CacheGroup*, KVTensor*>>> m_cache;
  std::map<std::string, CacheGroup> m_cache_groups;  // Maps graph index to CacheGroups
  CacheGroup* default_group;
  std::map<int32_t, std::set<GraphVariantInfo>> m_supported_variants;
  bool m_isKvInputNativeFormat;

  std::set<VariantSpec> m_logit_variants;
  // Requirements
  // Jobs must be Scopeable
  // Jobs must support COPY, CLEAR, and RESHAPE
  // Jobs must be splittable (i.e. one Job needs to be done across multiple threads)
  // [Optional] Jobs must be trackable (i.e. a record of all jobs done for logging)

  struct GraphState {
    std::atomic_int sync;

    // Maintain a separate job queue for independent "slices" of KV$ updates.
    // There will be a total of n_threads slices.
    std::vector<std::unique_ptr<JobSlice>> jobSlices;
  };
  std::map<int32_t, GraphState> m_graph_state;

  // Splits Job into slices that can be run in parallel,
  // then asks the background threads to execute the slices.
  void prepareJob(Scope scope, Job job);
  // Requests background threads to check for available jobs on the given graph.
  void queueJob(int32_t graph_idx);
  std::function<void(CacheGroup&, KVTensor&)> m_cached_update{nullptr};

  int32_t m_counter{0};      // Ticket counter for KV$ updates
  int32_t m_n_past{0};       // Total number of "virtual" KV$ tensors
  bool m_islastStep{false};  // true only for the last inference step.
  size_t m_n_batch{1};       // actual number of queries
  // Ideas: Keep track of whether last inference has already been processed
  InferenceStrategy m_strategy;
  uint32_t m_strategy_cur_step{0};
  bool m_strategy_active{false};
  InferenceStep m_last_inference;  // Keep track of the last known inference

  // processUpdate consumes the last known inference (m_last_inference) to generate update jobs
  // CAUTION: m_last_inference is destroyed by this function, since KV$ can only be consumed once
  bool processUpdate(
      InferenceStep& step, int32_t n_past, int32_t new_variant, int32_t new_ctx, Mask& mask = {});

 public:
  KVManager(std::shared_ptr<Env> env,
            QnnApi* qnnApi,
            std::shared_ptr<IOTensor> ioTensor,
            std::shared_ptr<ThreadPool> threadpool);
  virtual ~KVManager();

  void randomFn() override { return; }

  virtual const char* getTraceNamespace() const override { return "KVManager"; }

  // Add tensor pointers to keep track of in the graph
  // - These tensors are ordered. By graph_idx and tensor_idx. This is important for save/restore
  // - The shape of the tensors are deterministic based on each subclass
  // - The pointer points to the start of the buffer.
  // - Actual HTP tensor offsets may change (e.g. in POINTER_SHIFT) but buffer starts are constant
  // - Bitwidth is considered to be constant across tensors

  void registerSupportedVariant(int32_t variant, int32_t ctx_size, GraphType type);
  void registerQnnApi(QnnApi* qnn_api) { m_qnn_api = qnn_api; }
  void registerLogitVariants(std::set<VariantSpec>& variants) { m_logit_variants = variants; }
  std::set<VariantSpec>& getLogitVariants() { return m_logit_variants; }
  std::map<std::string, CacheGroup>& getCacheGroups() { return m_cache_groups; }
  void initComplete(int32_t max_ctx_size, std::string default_prefix);

  // Get the current states
  int32_t n_past() { return m_n_past; }
  int32_t n_valid_kv() { return default_group->m_n_valid_kv; }
  int32_t cur_variant() { return default_group->m_cur_variant; }
  int32_t cur_ctx() { return default_group->m_cur_ctx; }
  size_t getBatchSize() { return m_n_batch; }
  void setBatchSize(size_t batch_size) { m_n_batch = batch_size; }

  // checks if it is last infernce step or not
  bool isFinalInferenceStep() { return m_islastStep; }

  size_t getStrategySize() { return m_strategy.size(); }

  // Prepares an inference strategy (a set of InferenceSteps)
  // for the given number of inputs.
  bool prepareInferenceStrategy(int32_t n_inputs, bool output_all);
  bool nextInferenceStep(InferenceStep& step);
  bool completeInferenceStep();

  // Blocks the main thread until the given scope is ready,
  // i.e. there are no more background KV$ update jobs to run.
  bool block(Scope scope);
  // Stages update jobs for the next inference.
  bool unblock(Scope scope);

  // Simple getter function. Should we also have a setVariant() call?
  bool setActiveVariant(int32_t variant, int32_t ctx_size);

  // Functions for managing the cache directly called by the Engine/Dialog
  bool dispatchUpdate(int32_t n_past, Mask& mask = {});
  size_t loadKVCache(const std::string& filename);
  bool dumpKVCache(const std::string& filename);
  bool dumpKVCache(Buffer* kv_buff);
  bool getCacheSpec(CacheFileSpec& spec);
  bool getKVHead(CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale);
};
}  // namespace qualla