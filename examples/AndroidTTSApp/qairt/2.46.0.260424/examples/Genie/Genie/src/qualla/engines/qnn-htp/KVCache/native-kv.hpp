//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "kvmanager.hpp"

namespace qualla {

class NativeKV : public CacheManager {
 private:
  // These are determined by the QNN compiler, and is the size of the tile
  static const uint32_t K_TILE        = 256;
  static const uint32_t V_TILE        = 64;
  static const uint32_t KV_BLOCK_SIZE = 1024;

 protected:
  int32_t getIndexForNewKV(InferenceStep& step) override;

  int32_t getCacheBudget(const int32_t variant, const int32_t ctx_size) override;

  void completeInit(CacheGroup& group) override;

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
                 const bool is_key,
                 int32_t n_valid,
                 uint32_t n_heads,
                 int32_t variant,
                 int32_t ctx_size) override;

  void dumpCache(CacheGroup& group,
                 KVTensor& cache,
                 Buffer* kvBuff,
                 const bool is_key,
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

 public:
  NativeKV(std::shared_ptr<Env> env, bool useScatter) : CacheManager(env, useScatter) {}
  ~NativeKV() {}

  virtual const char* getTraceNamespace() const override { return "NativeKV"; }
};

}  // namespace qualla