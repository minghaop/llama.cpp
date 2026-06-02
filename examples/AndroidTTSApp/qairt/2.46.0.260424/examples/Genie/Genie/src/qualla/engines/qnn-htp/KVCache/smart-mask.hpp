//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "kvmanager.hpp"

namespace qualla {

class SmartMask : public CacheManager {
 private:
  // Sets count elements in ptr to m_clear_value depending on m_bitwidth
  std::function<void(void*, const size_t)> erase;

 public:
  SmartMask(std::shared_ptr<Env> env, bool useScatter) : CacheManager(env, useScatter) {}
  ~SmartMask() {}

  void completeInit(CacheGroup& group) override;

  int32_t getIndexForNewKV(InferenceStep& step) override;

  void clear(CacheGroup& group, KVTensor& cache) override;

  void updateKV(CacheGroup& group,
                KVTensor& cache,
                int32_t variant,
                int32_t ctx_size,
                const UpdateStrategy& updates) override;

  void reduceKV(CacheGroup& group,
                KVTensor& cache,
                int32_t variant,
                int32_t ctx_size,
                const UpdateStrategy& clears) override;

  void moveKV(CacheGroup& group,
              KVTensor& cache,
              int32_t variant,
              int32_t ctx_size,
              const UpdateStrategy& moves) override;

  void reshapeCache(CacheGroup& group,
                    KVTensor& cache,
                    int32_t cur_variant,
                    int32_t cur_ctx,
                    int32_t new_variant,
                    int32_t new_ctx) override;

  // Functions for save/restore
  void loadCache(CacheGroup& group,
                 KVTensor& cache,
                 std::istream* fs,
                 bool is_key,
                 int32_t n_valid,
                 uint32_t n_heads,
                 int32_t variant,
                 int32_t ctx_size) override;

  void dumpCache(CacheGroup& group,
                 KVTensor& cache,
                 std::ofstream* fs,
                 bool is_key,
                 int32_t n_valid,
                 uint32_t n_heads,
                 int32_t variant,
                 int32_t ctx_size) override;

  void dumpCache(CacheGroup& group,
                 KVTensor& cache,
                 Buffer* kvBuff,
                 bool is_key,
                 int32_t n_valid,
                 uint32_t n_heads,
                 int32_t variant,
                 int32_t ctx_size) override;

  void dumpHead(CacheGroup& group,
                KVTensor& cache,
                uint32_t head,
                int32_t n_valid,
                int32_t variant,
                int32_t ctx_size,
                void* data) override;

  virtual const char* getTraceNamespace() const override { return "SmartMask"; }
};

}  // namespace qualla
