//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "empty-manager.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"

namespace qualla {

void EmptyManager::completeInit(CacheGroup& /*group*/) { __DEBUG("Initializing Empty KV Manager"); }

int32_t EmptyManager::getIndexForNewKV(InferenceStep& /*step*/) { return 0; }

void EmptyManager::clear(CacheGroup& /*group*/, KVTensor& /*cache*/) {}

void EmptyManager::reduceKV(CacheGroup& /*group*/,
                            KVTensor& /*cache*/,
                            int32_t /*variant*/,
                            int32_t /*ctx_size*/,
                            const UpdateStrategy& /*clears*/) {}

void EmptyManager::updateKV(CacheGroup& /*group*/,
                            KVTensor& /*cache*/,
                            int32_t /*variant*/,
                            int32_t /*ctx_size*/,
                            const UpdateStrategy& /*updates*/) {}

void EmptyManager::moveKV(CacheGroup& /*group*/,
                          KVTensor& /*cache*/,
                          int32_t /*variant*/,
                          int32_t /*ctx_size*/,
                          const UpdateStrategy& /*moves*/) {}

void EmptyManager::reshapeCache(CacheGroup& /*group*/,
                                KVTensor& /*cache*/,
                                int32_t /*cur_variant*/,
                                int32_t /*cur_ctx*/,
                                int32_t /*new_variant*/,
                                int32_t /*new_ctx*/) {}

void EmptyManager::loadCache(CacheGroup& /*group*/,
                             KVTensor& /*cache*/,
                             std::istream* /*fs*/,
                             bool /*is_key*/,
                             int32_t /*n_valid*/,
                             uint32_t /*n_heads*/,
                             int32_t /*variant*/,
                             int32_t /*ctx_size*/) {}

void EmptyManager::dumpCache(CacheGroup& /*group*/,
                             KVTensor& /*cache*/,
                             std::ofstream* /*fs*/,
                             bool /*is_key*/,
                             int32_t /*n_valid*/,
                             uint32_t /*n_heads*/,
                             int32_t /*variant*/,
                             int32_t /*ctx_size*/) {}

void EmptyManager::dumpCache(CacheGroup& /*group*/,
                             KVTensor& /*cache*/,
                             Buffer* /*kvBuff*/,
                             bool /*is_key*/,
                             int32_t /*n_valid*/,
                             uint32_t /*n_heads*/,
                             int32_t /*variant*/,
                             int32_t /*ctx_size*/) {}

void EmptyManager::dumpHead(CacheGroup& /*group*/,
                            KVTensor& /*cache*/,
                            uint32_t /*head*/,
                            int32_t /*n_valid*/,
                            int32_t /*variant*/,
                            int32_t /*ctx_size*/,
                            void* /*data*/) {}

}  // namespace qualla
