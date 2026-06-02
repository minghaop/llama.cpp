//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>  // for std::max_element
#include <fstream>

#include "Trace.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "smart-mask.hpp"

namespace qualla {

void SmartMask::completeInit(CacheGroup& group) {
  // Construct a simple erase function that uses either memset() or fill_n
  auto& clear_val = group.m_clear_value;
  if (clear_val.u32 == 0)  // If the value is 0, use memset directly
    erase = [](void* s, const size_t n) { std::memset(s, 0, n); };
  else if (group.n_bytes == 1)  // For 8-bit values, use memset
    erase = [c = clear_val.u8](void* s, const size_t n) { std::memset(s, c, n); };
  else if (group.n_bytes == 2)  // For 16/32-bit values, use fill_n
    erase = [c = clear_val.u16](void* s, const size_t n) {
      std::fill_n(reinterpret_cast<uint16_t*>(s), n, c);
    };
  else if (group.n_bytes == 4)
    erase = [c = clear_val.u32](void* s, const size_t n) {
      std::fill_n(reinterpret_cast<uint32_t*>(s), n, c);
    };
}

int32_t SmartMask::getIndexForNewKV(InferenceStep& step) {
  if (m_useScatter)
    return step.n_valid_kv;
  else
    return step.ctx_size - step.variant;
}

void SmartMask::clear(CacheGroup& group, KVTensor& cache) {
  // Assumed check: KVTensor has max bitwidth of 32-bits per element (float32 or uint32)
  assert(group.n_bytes <= 4);
  erase(cache.key_buf, group.n_elements);
  erase(cache.val_buf, group.n_elements);
}

void SmartMask::reduceKV(CacheGroup& group,
                         KVTensor& cache,
                         int32_t variant,
                         int32_t ctx_size,
                         const UpdateStrategy& clears) {
  GENIE_KV_TRACE();
  // For the current implementation, clears are guaranteed to be CACHED, so head_idx is ignored
  const auto& clear_idxes = clears.get(cache, 0);

  int32_t past_dim = group.m_use_scatter ? ctx_size : ctx_size - variant;

  {
    // Key Cache has the axes [n_heads, n_embed, ctx_size]
    // So the operation is repeated for each "row", i.e. n_heads * n_embed iterations
    int32_t n_iter =
        static_cast<int32_t>(cache.n_heads) * group.n_embed_dim;  // Number of iterations
    int32_t esize     = group.n_bytes;  // Size to clear for each iteration
    int32_t iter_size = past_dim * esize;

    uint8_t* cache_ptr = cache.key_buf;  // input_buffer
    for (int32_t i = 0; i < n_iter; i++) {
      for (const auto& [idx, _, count] : clear_idxes) {
        erase(cache_ptr + idx * esize, count);
      }
      cache_ptr += iter_size;
    }
  }

  {
    // Value cache has the axes [n_heads, ctx_size, n_embed]
    // So the operation is repeated for each head, but each operation must copy n_embed elements
    int32_t n_iter    = static_cast<int32_t>(cache.n_heads);  // Number of iterations
    int32_t esize     = group.n_embed_dim * group.n_bytes;    // Size to clear for each iteration
    int32_t iter_size = past_dim * esize;

    uint8_t* cache_ptr = cache.val_buf;  // input_buffer
    for (int32_t i = 0; i < n_iter; i++) {
      for (const auto& [idx, _, count] : clear_idxes) {
        erase(cache_ptr + idx * esize, count * static_cast<uint32_t>(group.n_embed_dim));
      }
      cache_ptr += iter_size;
    }
  }
}

void SmartMask::updateKV(CacheGroup& group,
                         KVTensor& cache,
                         int32_t variant,
                         int32_t ctx_size,
                         const UpdateStrategy& updates) {
  GENIE_KV_TRACE();
  // Each buffer [ctx_size] is allocated as input[ctx-variant] + output[variant]
  int32_t past_dim = group.m_use_scatter ? ctx_size : ctx_size - variant;
  size_t past_size =
      cache.n_heads * static_cast<size_t>(group.n_embed_dim * past_dim * group.n_bytes);
  for (size_t batch_idx = 0; batch_idx < m_n_batch; batch_idx++) {
    for (int32_t head = 0; head < static_cast<int32_t>(cache.n_heads); head++) {
      const auto head_copies = updates.get(cache, head);

      {
        // Key Cache has the axes [batch_size, n_heads, n_embed, ctx_size]
        // So the operation is repeated for each "row", i.e. n_heads * n_embed iterations
        int32_t esize      = group.n_bytes;  // Size of copy for each iteration
        int32_t iter_size  = past_dim * esize;
        int32_t out_size   = variant * esize;
        uint8_t* write_ptr = cache.key_buf + batch_idx * past_size +
                             head * group.n_embed_dim * iter_size;  // input_buffer
        uint8_t* read_ptr =
            cache.key_buf + m_n_batch * past_size +
            static_cast<int32_t>(batch_idx * cache.n_heads) * group.n_embed_dim * out_size +
            head * group.n_embed_dim * out_size;  // output_buffer
        for (int32_t din = 0; din < group.n_embed_dim; din++) {
          for (const auto& [src_idx, dst_idx, count] : head_copies) {
            std::memcpy(reinterpret_cast<void*>(write_ptr + dst_idx * esize),
                        const_cast<const void*>(static_cast<void*>(read_ptr + src_idx * esize)),
                        static_cast<size_t>(count * static_cast<uint32_t>(esize)));
          }
          write_ptr += iter_size;
          read_ptr += out_size;
        }
      }

      {
        // Value cache has the axes [batch_size, n_heads, ctx_size, n_embed]
        // So the operation is repeated for each head, but each operation must copy n_embed elements
        int32_t esize     = group.n_embed_dim * group.n_bytes;  // Size of copy for each iteration
        int32_t iter_size = past_dim * esize;
        int32_t out_size  = variant * esize;
        uint8_t* write_ptr =
            cache.val_buf + batch_idx * past_size + head * iter_size;  // input_buffer
        uint8_t* read_ptr = cache.val_buf + m_n_batch * past_size +
                            static_cast<int32_t>(batch_idx * cache.n_heads) * out_size +
                            head * out_size;  // output_buffer

        for (const auto& [src_idx, dst_idx, count] : head_copies) {
          std::memcpy(reinterpret_cast<void*>(write_ptr + dst_idx * esize),
                      const_cast<const void*>(static_cast<void*>(read_ptr + src_idx * esize)),
                      static_cast<size_t>(count * static_cast<uint32_t>(esize)));
        }
        write_ptr += iter_size;
        read_ptr += out_size;
      }
    }
  }
}

void SmartMask::moveKV(CacheGroup& group,
                       KVTensor& cache,
                       int32_t variant,
                       int32_t ctx_size,
                       const UpdateStrategy& moves) {
  GENIE_KV_TRACE();
  // Each buffer [ctx_size] is allocated as input[ctx-variant] + output[variant]
  int32_t past_dim = group.m_use_scatter ? ctx_size : ctx_size - variant;

  for (int32_t head = 0; head < static_cast<int32_t>(cache.n_heads); head++) {
    const auto head_moves = moves.get(cache, head);

    {
      // Key Cache has the axes [n_heads, n_embed, ctx_size]
      // So the operation is repeated for each "row", i.e. n_heads * n_embed iterations
      int32_t esize     = group.n_bytes;  // Size of copy for each iteration
      int32_t iter_size = past_dim * esize;

      uint8_t* cache_ptr = cache.key_buf + head * group.n_embed_dim * iter_size;
      for (int32_t din = 0; din < group.n_embed_dim; din++) {
        for (const auto& [src_idx, dst_idx, count] : head_moves)
          std::memcpy(reinterpret_cast<void*>(cache_ptr + dst_idx * esize),
                      const_cast<const void*>(static_cast<void*>(cache_ptr + src_idx * esize)),
                      count * static_cast<size_t>(esize));
        cache_ptr += iter_size;
      }
    }

    {
      // Value cache has the axes [n_heads, ctx_size, n_embed]
      // So the operation is repeated for each head, but each operation must copy n_embed elements
      int32_t esize     = group.n_embed_dim * group.n_bytes;  // Size of copy for each iteration
      int32_t iter_size = past_dim * esize;

      uint8_t* cache_ptr = cache.key_buf + head * iter_size;
      for (const auto& [src_idx, dst_idx, count] : head_moves) {
        std::memcpy(reinterpret_cast<void*>(cache_ptr + dst_idx * esize),
                    const_cast<const void*>(static_cast<void*>(cache_ptr + src_idx * esize)),
                    count * static_cast<size_t>(esize));
      }
    }
  }
}

void SmartMask::reshapeCache(CacheGroup& group,
                             KVTensor& cache,
                             int32_t cur_variant,
                             int32_t cur_ctx,
                             int32_t new_variant,
                             int32_t new_ctx) {
  GENIE_KV_TRACE();
  // If using scatter, all AR-n variants have the same shape, so this is a no-op
  if (group.m_use_scatter && cur_ctx == new_ctx) {
    return;
  }

  // Both key/value are reshaped from a size of [cur_ctx - cur_variant] to [new_ctx - new_variant]
  size_t in_cache_dim = static_cast<size_t>(
      (cur_variant == cur_ctx || group.m_use_scatter) ? cur_ctx : cur_ctx - cur_variant);
  size_t out_cache_dim = static_cast<size_t>(group.m_use_scatter ? new_ctx : new_ctx - new_variant);

  {
    // For Key, reshape is done along axis -1
    // [n_heads, group.n_embed_dim, in_cache_dim] -> [n_heads, group.n_embed_dim, out_cache_dim]

    uint32_t n_iter   = cache.n_heads * static_cast<uint32_t>(group.n_embed_dim);
    size_t read_size  = in_cache_dim * static_cast<size_t>(group.n_bytes);
    size_t write_size = out_cache_dim * static_cast<size_t>(group.n_bytes);

    uint8_t* read_ptr  = cache.key_buf;
    uint8_t* write_ptr = cache.key_buf;
    if (in_cache_dim > out_cache_dim) {
      for (uint32_t i = 0; i < n_iter; i++) {
        std::memcpy(write_ptr, read_ptr, write_size);
        read_ptr += read_size;
        write_ptr += write_size;
      }
    } else {
      read_ptr += (n_iter - 1) * read_size;
      write_ptr += (n_iter - 1) * write_size;
      uint8_t* pad_ptr = write_ptr + read_size;
      size_t pad_count = out_cache_dim - in_cache_dim;
      for (uint32_t i = 0; i < n_iter; i++) {
        if (write_ptr >= read_ptr + read_size || write_ptr + write_size <= read_ptr) {
          std::memcpy(write_ptr, read_ptr, read_size);
        } else {
          std::memmove(write_ptr, read_ptr, read_size);
        }
        erase(pad_ptr, pad_count);
        read_ptr -= read_size;
        write_ptr -= write_size;
        pad_ptr -= write_size;
      }
    }
  }

  {
    // For Value, reshape is done along axis -2
    // [n_heads, in_cache_dim, group.n_embed_dim] -> [n_heads, out_cache_dim, group.n_embed_dim]

    uint32_t n_iter   = cache.n_heads;
    size_t read_size  = in_cache_dim * static_cast<size_t>(group.n_embed_dim * group.n_bytes);
    size_t write_size = out_cache_dim * static_cast<size_t>(group.n_embed_dim * group.n_bytes);

    uint8_t* read_ptr  = cache.val_buf;
    uint8_t* write_ptr = cache.val_buf;

    if (in_cache_dim > out_cache_dim) {
      for (uint32_t i = 0; i < n_iter; i++) {
        std::memcpy(write_ptr, read_ptr, write_size);
        read_ptr += read_size;
        write_ptr += write_size;
      }
    } else {
      read_ptr += (n_iter - 1) * read_size;
      write_ptr += (n_iter - 1) * write_size;
      uint8_t* pad_ptr = write_ptr + read_size;
      size_t pad_count = (out_cache_dim - in_cache_dim) * static_cast<size_t>(group.n_embed_dim);
      for (uint32_t i = 0; i < n_iter; i++) {
        if (write_ptr >= read_ptr + read_size || write_ptr + write_size <= read_ptr) {
          std::memcpy(write_ptr, read_ptr, read_size);
        } else {
          std::memmove(write_ptr, read_ptr, read_size);
        }
        erase(pad_ptr, pad_count);
        read_ptr -= read_size;
        write_ptr -= write_size;
        pad_ptr -= write_size;
      }
    }
  }
}

void SmartMask::loadCache(CacheGroup& group,
                          KVTensor& cache,
                          std::istream* fs,
                          bool is_key,
                          int32_t n_valid,
                          uint32_t n_heads,
                          int32_t variant,
                          int32_t ctx_size) {
  GENIE_KV_TRACE();
  int32_t past_dim = group.m_use_scatter ? ctx_size : ctx_size - variant;

  if (is_key) {
    uint32_t n_iter                 = cache.n_heads * static_cast<uint32_t>(group.n_embed_dim);
    size_t iter_size                = static_cast<size_t>(past_dim * group.n_bytes);
    const std::streamsize copy_size = static_cast<std::streamsize>(n_valid * group.n_bytes);

    uint8_t* buffer = cache.key_buf;
    for (uint32_t i = 0; i < n_iter; i++) {
      fs->read(reinterpret_cast<char*>(buffer), copy_size);
      buffer += iter_size;
    }
  } else {
    uint32_t n_iter  = cache.n_heads;
    size_t iter_size = static_cast<size_t>(past_dim * group.n_embed_dim * group.n_bytes);
    const std::streamsize copy_size =
        static_cast<std::streamsize>(n_valid * group.n_embed_dim * group.n_bytes);

    uint8_t* buffer = cache.val_buf;
    for (uint32_t i = 0; i < n_iter; i++) {
      fs->read(reinterpret_cast<char*>(buffer), copy_size);
      buffer += iter_size;
    }
  }

  fs->seekg((n_heads - cache.n_heads) *
                static_cast<uint32_t>(group.n_embed_dim * n_valid * group.n_bytes),
            std::ios::cur);
}

void SmartMask::dumpCache(CacheGroup& group,
                          KVTensor& cache,
                          std::ofstream* fs,
                          bool is_key,
                          int32_t n_valid,
                          uint32_t n_heads,
                          int32_t variant,
                          int32_t ctx_size) {
  GENIE_KV_TRACE();
  int32_t past_dim = group.m_use_scatter ? ctx_size : ctx_size - variant;

  if (is_key) {
    uint32_t n_iter                 = cache.n_heads * static_cast<uint32_t>(group.n_embed_dim);
    size_t iter_size                = static_cast<size_t>(past_dim * group.n_bytes);
    const std::streamsize copy_size = static_cast<std::streamsize>(n_valid * group.n_bytes);

    uint8_t* buffer = cache.key_buf;
    for (uint32_t i = 0; i < n_iter; i++) {
      fs->write(reinterpret_cast<char*>(buffer), copy_size);
      buffer += iter_size;
    }

  } else {
    uint32_t n_iter  = cache.n_heads;
    size_t iter_size = static_cast<size_t>(past_dim * group.n_embed_dim * group.n_bytes);
    const std::streamsize copy_size =
        static_cast<std::streamsize>(n_valid * group.n_embed_dim * group.n_bytes);

    uint8_t* buffer = cache.val_buf;
    for (uint32_t i = 0; i < n_iter; i++) {
      fs->write(reinterpret_cast<char*>(buffer), copy_size);
      buffer += iter_size;
    }
  }

  fs->seekp((n_heads - cache.n_heads) *
                static_cast<uint32_t>(group.n_embed_dim * n_valid * group.n_bytes),
            std::ios::cur);
}

void SmartMask::dumpCache(CacheGroup& group,
                          KVTensor& cache,
                          Buffer* kv_buff,
                          bool is_key,
                          int32_t n_valid,
                          uint32_t n_heads,
                          int32_t variant,
                          int32_t ctx_size) {
  int32_t past_dim = group.m_use_scatter ? ctx_size : ctx_size - variant;

  if (is_key) {
    uint32_t n_iter  = cache.n_heads * static_cast<uint32_t>(group.n_embed_dim);
    size_t iter_size = static_cast<size_t>(past_dim * group.n_bytes);
    size_t copy_size = static_cast<size_t>(n_valid * group.n_bytes);

    uint8_t* buffer = cache.key_buf;
    for (uint32_t i = 0; i < n_iter; i++) {
      kv_buff->appendBuffer(buffer, copy_size);
      buffer += iter_size;
    }

  } else {
    uint32_t n_iter  = cache.n_heads;
    size_t iter_size = static_cast<size_t>(past_dim * group.n_embed_dim * group.n_bytes);
    size_t copy_size = static_cast<size_t>(n_valid * group.n_embed_dim * group.n_bytes);

    uint8_t* buffer = cache.val_buf;
    for (uint32_t i = 0; i < n_iter; i++) {
      kv_buff->appendBuffer(buffer, copy_size);
      buffer += iter_size;
    }
  }
  kv_buff->setPosFromCurr(static_cast<int32_t>(n_heads - cache.n_heads) * group.n_embed_dim *
                          n_valid * group.n_bytes);
}

void SmartMask::dumpHead(CacheGroup& group,
                         KVTensor& cache,
                         uint32_t head,
                         int32_t n_valid,
                         int32_t variant,
                         int32_t ctx_size,
                         void* data) {
  if (head > cache.n_heads) {
    memset(data, 128, static_cast<size_t>(2 * group.n_embed_dim * n_valid * group.n_bytes));
    return;
  }

  int32_t past_dim = group.m_use_scatter ? ctx_size : ctx_size - variant;

  size_t key_head_iter_size = static_cast<size_t>(group.n_embed_dim * past_dim * group.n_bytes);
  uint8_t* read_buf         = cache.key_buf + head * key_head_iter_size;
  uint8_t* write_buf        = reinterpret_cast<uint8_t*>(data);

  for (int i = 0; i < group.n_embed_dim; i++) {
    for (int j = 0; j < n_valid; j++) {
      write_buf[j * group.n_embed_dim + i] = read_buf[i * past_dim + j];
    }
  }

  write_buf += (group.n_embed_dim * n_valid * group.n_bytes);
  size_t value_head_iter_size = static_cast<size_t>(past_dim * group.n_embed_dim * group.n_bytes);
  read_buf                    = cache.val_buf + head * value_head_iter_size;

  memcpy(write_buf, read_buf, static_cast<size_t>(n_valid * group.n_embed_dim * group.n_bytes));
}

}  // namespace qualla
