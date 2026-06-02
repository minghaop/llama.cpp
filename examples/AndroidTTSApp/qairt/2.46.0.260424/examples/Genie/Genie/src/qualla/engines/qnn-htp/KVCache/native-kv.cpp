//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fstream>

#include "Trace.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "native-kv.hpp"

namespace qualla {

void NativeKV::completeInit(CacheGroup& group) {
  // Internally, HMX does not apply an offset for NativeKV tensor
  // This means we do not set empty values to 128, but rather 0
  group.m_clear_value.u32 = 0x0;

  if (group.n_bytes != 1 || group.m_quantized != true)
    State::error("Native KV only supports uint8.");
}

int32_t NativeKV::getIndexForNewKV(InferenceStep& step) {
  return ((step.n_valid_kv + 31) / 32) * 32;  // Round up to 32
}

int32_t NativeKV::getCacheBudget(const int32_t variant, const int32_t ctx_size) {
  if (variant == ctx_size) return ctx_size;
  // Round variant up to nearest 32, ensuring we don't round up 0
  int32_t rounded_variant = variant > 0 ? ((variant + 31) / 32) * 32 : 0;
  return ctx_size - rounded_variant;
}

void NativeKV::clear(CacheGroup& group, KVTensor& cache) {
  GENIE_KV_TRACE();
  size_t cache_size = group.n_elements * static_cast<size_t>(group.n_bytes);
  std::memset(cache.key_buf, 0, cache_size);
  std::memset(cache.val_buf, 0, cache_size);
}

// Translates a flat index to an offset for the HMX weight format buffer
// Convert [din, dout] -> [dout/N_TILE, din/32, (dout%N_TILE)/32, [(din%32)/4, dout%32, din%4]]
//
// For Key$ (head, din=embed, dout=ctx_size) and K_TILE=256
// (head, tile=dout/K_TILE,  din: din/32, dout: K_TILE/32, din:8, dout:32, din:4)
//
// For Value$ (head, din=ctx_size, dout=embed) and V_TILE=64
// (head, tile=dout/V_TILE,  din: din/32, dout: V_TILE/32, din:8, dout:32, din:4)
static inline int32_t fromFlatOffset(
    int32_t DIN, int32_t DOUT, int32_t N_TILE, int32_t din, int32_t dout) {
  assert(DIN % 32 == 0);
  assert(DOUT % 32 == 0);

  // Each tensor then gets tiled into chunks of min(dout, N_TILE)
  const int32_t tile_size   = std::min(DOUT, N_TILE);  // head * tile * [N_EMBED, N_TILE or DOUT]
  const int32_t tile_stride = DIN * tile_size;         // head * tile * [N_EMBED, tile_size]

  // Split the dout into [dout // NTILE, (dout % NTILE) // 32 , (dout % tile_size) % 32]
  const int32_t tile_idx = dout / tile_size;
  const int32_t dout_0   = (dout % tile_size) >> 5;  // (dout % tile_size) / 32
  const int32_t dout_1   = dout & 0x1f;              // From (dout % tile_size) % 32 = dout % 32;

  // Split the din into [din // 32, (din % 32) // 4, (din % 32) % 4]
  const int32_t din_0 = din >> 5;           // From din / 32;
  const int32_t din_1 = (din & 0x1f) >> 2;  // From (din % 32) / 4;
  const int32_t din_2 = din & 0x3;          // From (din % 32) % 4 = din % 4

  // Strides for the chunk of (8:DIN, 32:tile_size, 4:N_EMBED). This is always constant
  static const int32_t bitshift[3] = {10, 7, 2};

  // Stride for each tile * chunk. This equals (tile_size/32) * (8*32*4). Note tile_size%32==0
  const int32_t din_0_stride = tile_size << 5;  // tile_shift * 32;

  // Construct the final flat offset as [head, tile_idx, din_0, dout_0, (din_1, dout_1, din_2)]
  return tile_idx * tile_stride + din_0 * din_0_stride +
         (dout_0 << bitshift[0] | (din_1 << bitshift[1]) | (dout_1 << bitshift[2]) | din_2);
}

// Reduce KV$ - remove entries from the cache buffer
void NativeKV::reduceKV(CacheGroup& group,
                        KVTensor& cache,
                        int32_t /*variant*/,
                        int32_t ctx_size,
                        const UpdateStrategy& clears) {
  GENIE_KV_TRACE();
  // For the current implementation, clears are guaranteed to be CACHED, so head_idx is ignored
  const auto& clear_idxes = clears.get(cache, 0);

  uint32_t head_stride = static_cast<uint32_t>(group.n_embed_dim * ctx_size * group.n_bytes);
  {
    uint8_t* cache_ptr = cache.key_buf;
    for (uint32_t head = 0; head < cache.n_heads; head++) {
      uint8_t* head_ptr = cache_ptr + head * head_stride;
      for (int32_t din = 0; din < group.n_embed_dim; din++) {
        for (const auto& [idx, _, count] : clear_idxes) {
          for (size_t i = 0; i < count; i++) {
            head_ptr[fromFlatOffset(group.n_embed_dim,
                                    ctx_size,
                                    K_TILE,
                                    din,
                                    static_cast<int32_t>(idx + static_cast<int32_t>(i)))] = 0;
          }
        }
      }
    }
  }

  {
    uint8_t* cache_ptr = cache.val_buf;
    for (uint32_t head = 0; head < cache.n_heads; head++) {
      uint8_t* head_ptr = cache_ptr + head * head_stride;
      for (const auto& [idx, _, count] : clear_idxes) {
        for (size_t i = 0; i < count; i++) {
          for (int32_t dout = 0; dout < group.n_embed_dim; dout++) {
            head_ptr[fromFlatOffset(ctx_size,
                                    group.n_embed_dim,
                                    V_TILE,
                                    static_cast<int32_t>(idx + static_cast<int32_t>(i)),
                                    dout)] = 0;
          }
        }
      }
    }
  }
}

// Optimization- directly copies the entire KV_BLOCK by avoiding per-element operations
static inline void key_buffer_aligned_update(int32_t src_idx,
                                             int32_t dst_idx,
                                             uint8_t* head_src_ptr,
                                             uint8_t* head_dst_ptr,
                                             int32_t variant,
                                             int32_t ctx_size,
                                             int32_t embed_dim,
                                             int32_t bitwidth,
                                             int32_t KV_BLOCK_SIZE,
                                             int32_t K_TILE,
                                             int32_t count_first_part,
                                             int32_t count_second_part,
                                             int32_t count_third_part) {
  for (int32_t din_block = 0; din_block < (embed_dim / 32); din_block++) {
    int32_t idx_offset = 0;
    if (count_first_part) {
      uint8_t* src_offset_ptr =
          head_src_ptr + fromFlatOffset(embed_dim, variant, K_TILE, din_block * 32, src_idx);
      uint8_t* dst_offset_ptr =
          head_dst_ptr + fromFlatOffset(embed_dim, ctx_size, K_TILE, din_block * 32, dst_idx);
      std::memcpy(dst_offset_ptr,
                  src_offset_ptr,
                  static_cast<size_t>(KV_BLOCK_SIZE * (count_first_part / 32) * bitwidth));
      idx_offset += count_first_part;
    }

    for (int32_t fill_count = 0; fill_count < count_second_part; fill_count += K_TILE) {
      uint8_t* src_offset_ptr =
          head_src_ptr +
          fromFlatOffset(embed_dim, variant, K_TILE, din_block * 32, src_idx + idx_offset);
      uint8_t* dst_offset_ptr =
          head_dst_ptr +
          fromFlatOffset(embed_dim, ctx_size, K_TILE, din_block * 32, dst_idx + idx_offset);
      std::memcpy(dst_offset_ptr,
                  src_offset_ptr,
                  static_cast<size_t>(KV_BLOCK_SIZE * (K_TILE / 32) * bitwidth));
      idx_offset += fill_count;
    }

    if (count_third_part) {
      uint8_t* src_offset_ptr =
          head_src_ptr +
          fromFlatOffset(embed_dim, variant, K_TILE, din_block * 32, src_idx + idx_offset);
      uint8_t* dst_offset_ptr =
          head_dst_ptr +
          fromFlatOffset(embed_dim, ctx_size, K_TILE, din_block * 32, dst_idx + idx_offset);
      std::memcpy(dst_offset_ptr,
                  src_offset_ptr,
                  static_cast<size_t>(KV_BLOCK_SIZE * (count_third_part / 32) * bitwidth));
    }
  }
}

// Optimization- directly copies the entire KV_BLOCK by avoiding per-element operations
static inline void value_buffer_aligned_update(int32_t src_idx,
                                               int32_t dst_idx,
                                               size_t count,
                                               uint8_t* head_src_ptr,
                                               uint8_t* head_dst_ptr,
                                               int32_t variant,
                                               int32_t ctx_size,
                                               int32_t embed_dim,
                                               int32_t bitwidth,
                                               int32_t KV_BLOCK_SIZE,
                                               int32_t V_TILE) {
  for (int32_t dout_block = 0; dout_block < (embed_dim / 64); dout_block++) {
    uint8_t* src_offset_ptr =
        head_src_ptr + fromFlatOffset(variant, embed_dim, V_TILE, src_idx, dout_block * 64);
    uint8_t* dst_offset_ptr =
        head_dst_ptr + fromFlatOffset(ctx_size, embed_dim, V_TILE, dst_idx, dout_block * 64);
    std::memcpy(dst_offset_ptr,
                src_offset_ptr,
                static_cast<size_t>(KV_BLOCK_SIZE * V_TILE) / 32 * (count / 32) *
                    static_cast<size_t>(bitwidth));
  }
}

// Update KV$ - copy entries from output buffer into the cache buffer
void NativeKV::updateKV(CacheGroup& group,
                        KVTensor& cache,
                        int32_t variant,
                        int32_t ctx_size,
                        const UpdateStrategy& updates) {
  GENIE_KV_TRACE();
  // Each buffer [ctx_size] is allocated as input[ctx-variant] + output[variant]
  int32_t head_stride_in  = group.n_embed_dim * ctx_size * group.n_bytes;
  int32_t head_stride_out = group.n_embed_dim * variant * group.n_bytes;
  size_t cache_size       = cache.n_heads * static_cast<uint32_t>(head_stride_in);
  bool isKvOutputNativeFormat =
      group.m_isKvOutputNativeFormat[std::pair<int32_t, int32_t>(variant, ctx_size)];

  // If the updates are aligned along block sizes, moves can be optimized
  // This code-path if triggered, calls the optimized move functions and exits
  if (updates.mode == UpdateStrategy::CACHED && updates.steps.size() == 1 &&
      isKvOutputNativeFormat) {
    const auto& [src_idx, dst_idx, count] = updates.steps[0];

    if (((dst_idx & 0x1F) == 0) && ((count & 0x1F) == 0)) {
      // Updates are aligned and moves can be optimized

      {
        // Update Key Buffer
        uint8_t* dst_ptr = cache.key_buf;               // input_buffer
        uint8_t* src_ptr = cache.key_buf + cache_size;  // output_buffer

        int32_t count_int32       = static_cast<int32_t>(count);
        int32_t k_tile_int32      = static_cast<int32_t>(K_TILE);
        int32_t count_first_part  = std::min(k_tile_int32 - dst_idx % k_tile_int32, count_int32);
        int32_t count_second_part = (count_int32 - count_first_part) / k_tile_int32 * k_tile_int32;
        int32_t count_third_part  = count_int32 - count_first_part - count_second_part;

        for (int32_t head = 0; head < static_cast<int32_t>(cache.n_heads); head++) {
          uint8_t* head_src_ptr = src_ptr + head * head_stride_out;
          uint8_t* head_dst_ptr = dst_ptr + head * head_stride_in;
          key_buffer_aligned_update(src_idx,
                                    dst_idx,
                                    head_src_ptr,
                                    head_dst_ptr,
                                    variant,
                                    ctx_size,
                                    group.n_embed_dim,
                                    group.n_bytes,
                                    KV_BLOCK_SIZE,
                                    K_TILE,
                                    count_first_part,
                                    count_second_part,
                                    count_third_part);
        }
      }

      {
        // Update Value Buffer
        uint8_t* dst_ptr = cache.val_buf;               // input_buffer
        uint8_t* src_ptr = cache.val_buf + cache_size;  // output_buffer
        for (int32_t head = 0; head < static_cast<int32_t>(cache.n_heads); head++) {
          uint8_t* head_src_ptr = src_ptr + head * head_stride_out;
          uint8_t* head_dst_ptr = dst_ptr + head * head_stride_in;
          value_buffer_aligned_update(src_idx,
                                      dst_idx,
                                      count,
                                      head_src_ptr,
                                      head_dst_ptr,
                                      variant,
                                      ctx_size,
                                      group.n_embed_dim,
                                      group.n_bytes,
                                      KV_BLOCK_SIZE,
                                      V_TILE);
        }
      }
      return;  // Aligned updates are complete
    }
  }

  // If the optimization above is not triggered, use individual memory moves
  for (int32_t head = 0; head < static_cast<int32_t>(cache.n_heads); head++) {
    const auto& head_copies = updates.get(cache, head);
    {
      // Update Key Buffer
      uint8_t* dst_ptr = cache.key_buf;               // input_buffer
      uint8_t* src_ptr = cache.key_buf + cache_size;  // output_buffer

      uint8_t* head_src_ptr = src_ptr + head * head_stride_out;
      uint8_t* head_dst_ptr = dst_ptr + head * head_stride_in;

      if (isKvOutputNativeFormat) {
        for (int32_t din_out = 0; din_out < group.n_embed_dim / 32; din_out++) {
          for (const auto& [src_idx, dst_idx, count] : head_copies) {
            for (int32_t i = 0; i < static_cast<int32_t>(count); i++) {
              int32_t din_target = din_out * 32;
              int32_t src_offset =
                  fromFlatOffset(group.n_embed_dim, variant, K_TILE, din_target, src_idx + i);
              int32_t dst_offset =
                  fromFlatOffset(group.n_embed_dim, ctx_size, K_TILE, din_target, dst_idx + i);
              for (int32_t din_indx = 0; din_indx < 8; din_indx++) {
                int32_t block_stride = din_indx * 128;
                std::memcpy(head_dst_ptr + dst_offset + block_stride,
                            head_src_ptr + src_offset + block_stride,
                            4 * static_cast<uint32_t>(group.n_bytes));
              }
            }
          }
        }
      } else {
        for (int32_t din = 0; din < group.n_embed_dim; din++) {
          for (const auto& [src_idx, dst_idx, count] : head_copies) {
            for (int32_t i = 0; i < static_cast<int32_t>(count); i++) {
              int32_t src_offset = din * variant + src_idx + i;
              int32_t dst_offset =
                  fromFlatOffset(group.n_embed_dim, ctx_size, K_TILE, din, dst_idx + i);
              head_dst_ptr[dst_offset] = head_src_ptr[src_offset] - 128;
            }
          }
        }
      }
    }

    {
      // Update Value Buffer
      uint8_t* dst_ptr = cache.val_buf;               // input_buffer
      uint8_t* src_ptr = cache.val_buf + cache_size;  // output_buffer

      uint8_t* head_src_ptr = src_ptr + head * head_stride_out;
      uint8_t* head_dst_ptr = dst_ptr + head * head_stride_in;

      if (isKvOutputNativeFormat) {
        for (const auto& [src_idx, dst_idx, count] : head_copies) {
          for (int32_t i = 0; i < static_cast<int32_t>(count); i++) {
            for (int32_t dout = 0; dout < group.n_embed_dim; dout++) {
              int32_t src_offset =
                  fromFlatOffset(variant, group.n_embed_dim, V_TILE, src_idx + i, dout);
              int32_t dst_offset =
                  fromFlatOffset(ctx_size, group.n_embed_dim, V_TILE, dst_idx + i, dout);
              head_dst_ptr[dst_offset] = head_src_ptr[src_offset];
            }
          }
        }
      } else {
        for (const auto& [src_idx, dst_idx, count] : head_copies) {
          for (int32_t i = 0; i < static_cast<int32_t>(count); i++) {
            for (int32_t dout = 0; dout < group.n_embed_dim; dout++) {
              int32_t src_offset = (src_idx + i) * group.n_embed_dim + dout;
              int32_t dst_offset =
                  fromFlatOffset(ctx_size, group.n_embed_dim, V_TILE, dst_idx + i, dout);
              head_dst_ptr[dst_offset] = head_src_ptr[src_offset] - 128;
            }
          }
        }
      }
    }
  }
}

// Move KV$ - move entries within the cache buffer
void NativeKV::moveKV(CacheGroup& group,
                      KVTensor& cache,
                      int32_t /*variant*/,
                      int32_t ctx_size,
                      const UpdateStrategy& moves) {
  GENIE_KV_TRACE();
  // Each buffer [ctx_size] is allocated as input[ctx-variant] + output[variant]
  const int32_t head_stride_in = group.n_embed_dim * ctx_size * group.n_bytes;

  for (int32_t head = 0; head < static_cast<int32_t>(cache.n_heads); head++) {
    const auto& head_moves = moves.get(cache, head);

    {
      // Update Key Buffer
      uint8_t* cache_ptr = cache.key_buf;  // input_buffer
      uint8_t* head_ptr  = cache_ptr + head * head_stride_in;
      for (int32_t din = 0; din < group.n_embed_dim; din++) {
        for (const auto& [src_idx, dst_idx, count] : head_moves) {
          for (int32_t i = 0; i < static_cast<int32_t>(count); i++) {
            int32_t src_offset =
                fromFlatOffset(group.n_embed_dim, ctx_size, K_TILE, din, src_idx + i);
            int32_t dst_offset =
                fromFlatOffset(group.n_embed_dim, ctx_size, K_TILE, din, dst_idx + i);
            head_ptr[dst_offset] = head_ptr[src_offset];
          }
        }
      }
    }

    {
      // Update Value Buffer
      uint8_t* cache_ptr = cache.key_buf;  // input_buffer
      uint8_t* head_ptr  = cache_ptr + head * head_stride_in;
      for (const auto& [src_idx, dst_idx, count] : head_moves) {
        for (int32_t i = 0; i < static_cast<int32_t>(count); i++) {
          for (int32_t dout = 0; dout < group.n_embed_dim; dout++) {
            int32_t src_offset =
                fromFlatOffset(ctx_size, group.n_embed_dim, V_TILE, src_idx + i, dout);
            int32_t dst_offset =
                fromFlatOffset(ctx_size, group.n_embed_dim, V_TILE, dst_idx + i, dout);
            head_ptr[dst_offset] = head_ptr[src_offset];
          }
        }
      }
    }
  }
}

void NativeKV::reshapeCache(CacheGroup& group,
                            KVTensor& cache,
                            int32_t /*cur_variant*/,
                            int32_t cur_ctx,
                            int32_t /*new_variant*/,
                            int32_t new_ctx) {
  GENIE_KV_TRACE();
  // All AR-n variants have the same shape, so this is a no-op for NativeKV
  if (new_ctx == cur_ctx) return;

  {
    // For Key cache, DIN=group.n_embed_dim and DOUT=ctx_size
    // cur_ctx -> (head, cur_ctx/K_TILE,  din: embed/32, dout: K_TILE/32, din:8, dout:32, din:4)
    // new_ctx -> (head, new_ctx/K_TILE,  din: embed/32, dout: K_TILE/32, din:8, dout:32, din:4)
    //
    // This translates to copying (embed/32)*(K_TILE/32)*(8*32*4) elements over head iterations
    uint32_t n_iter = cache.n_heads;  // Iterate for each head
    uint32_t stride =
        static_cast<uint32_t>(group.n_embed_dim) * K_TILE;  // Stride for each KV$ index

    // Size of a read for each iteration
    size_t read_size = static_cast<uint32_t>(cur_ctx) / K_TILE * stride;
    // Size of a write for each iteartion
    size_t write_size = static_cast<uint32_t>(new_ctx) / K_TILE * stride;

    if (cur_ctx > new_ctx) {
      // Context size decreases in size. Guarantees no memory overlap, and no padding required
      uint8_t* read_ptr  = cache.key_buf;
      uint8_t* write_ptr = cache.key_buf;
      for (uint32_t i = 0; i < n_iter; i++) {
        std::memcpy(write_ptr, read_ptr, write_size);
        read_ptr += read_size;
        write_ptr += write_size;
      }
    } else {
      // Context size decreases in size. Read/write must start from the last index backwards
      // This is to avoid overwriting memory before these are copied correctly
      uint8_t* read_ptr  = cache.key_buf + (n_iter - 1) * read_size;
      uint8_t* write_ptr = cache.key_buf + (n_iter - 1) * write_size;

      // The remaining elements will have to be padded, i.e. set to 0
      uint8_t* pad_ptr = write_ptr + read_size;   // Start padding after cur_ctx
      size_t pad_size  = write_size - read_size;  // Pad upto new_ctx

      for (uint32_t i = 0; i < n_iter; i++) {
        if (write_ptr >= read_ptr + read_size || write_ptr + write_size <= read_ptr) {
          std::memcpy(write_ptr, read_ptr, read_size);
        } else {
          std::memmove(write_ptr, read_ptr, read_size);
        }
        std::memset(pad_ptr, 0, pad_size);
        read_ptr -= read_size;
        write_ptr -= write_size;
        pad_ptr -= write_size;
      }
    }
  }

  {
    // For Value cache, DIN=ctx_size, DOUT=group.n_embed_dim
    // cur_ctx -> (head, embed/V_TILE,  din: cur_ctx/32, dout: V_TILE/32, din:8, dout:32, din:4)
    // new_ctx -> (head, embed/V_TILE,  din: new_ctx/32, dout: V_TILE/32, din:8, dout:32, din:4)
    //
    // This translates to copying (V_TILE/32)*8*32*4 elements over head*(embed/V_TILE) iterations
    uint32_t n_iter =
        cache.n_heads * (static_cast<uint32_t>(group.n_embed_dim) / V_TILE);  // #heads * #tiles
    uint32_t stride =
        V_TILE * 32 * static_cast<uint32_t>(group.n_bytes);  // Stride for each KV$ index

    // Size of a read for each iteration
    size_t read_size = static_cast<size_t>(static_cast<uint32_t>(cur_ctx) / 32 * stride);
    // Size of a write for each iteration
    size_t write_size = static_cast<size_t>(static_cast<uint32_t>(new_ctx) / 32 * stride);

    if (cur_ctx > new_ctx) {
      // Context size decreases in size. Guarantees no memory overlap, and no padding required
      uint8_t* read_ptr  = cache.val_buf;
      uint8_t* write_ptr = cache.val_buf;
      for (size_t i = 0; i < n_iter; i++) {
        std::memcpy(write_ptr, read_ptr, write_size);
        read_ptr += read_size;
        write_ptr += write_size;
      }
    } else {
      // Context size decreases in size. Read/write must start from the last index backwards
      // This is to avoid overwriting memory before these are copied correctly
      uint8_t* read_ptr  = cache.val_buf + (n_iter - 1) * read_size;
      uint8_t* write_ptr = cache.val_buf + (n_iter - 1) * write_size;

      // The remaining elements will have to be padded, i.e. set to 0
      uint8_t* pad_ptr = write_ptr + read_size;   // Start padding after cur_ctx
      size_t pad_size  = write_size - read_size;  // Pad upto new_ctx

      for (size_t i = 0; i < n_iter; i++) {
        if (write_ptr >= read_ptr + read_size || write_ptr + write_size <= read_ptr) {
          std::memcpy(write_ptr, read_ptr, read_size);
        } else {
          std::memmove(write_ptr, read_ptr, read_size);
        }
        std::memset(pad_ptr, 0, pad_size);
        read_ptr -= read_size;
        write_ptr -= write_size;
        pad_ptr -= write_size;
      }
    }
  }
}

void NativeKV::loadCache(CacheGroup& group,
                         KVTensor& cache,
                         std::istream* fs,
                         bool is_key,
                         int32_t n_valid,
                         uint32_t n_heads,
                         int32_t /*variant*/,
                         int32_t ctx_size) {
  GENIE_KV_TRACE();
  if (group.n_bytes != 1 || group.m_quantized != true) {
    State::error("Native KV only supports 8-bit KV$");
  }

  uint32_t head_stride = static_cast<uint32_t>(group.n_embed_dim * ctx_size * group.n_bytes);

  // Create a scratch buffer to help minimize IO calls, and allow for post-processing (uint8->int8)
  std::vector<char> scratch(static_cast<size_t>(group.n_embed_dim * n_valid * group.n_bytes));

  for (uint32_t head = 0; head < cache.n_heads; head++) {
    fs->read(scratch.data(),
             static_cast<std::streamsize>(scratch.size()));  // Batch fs->read() call for this head
    for (auto& ch : scratch) {
      ch -= 128;  // Convert uint8 -> int8
    }

    char* scratch_ptr = scratch.data();
    if (is_key) {
      char* head_ptr = reinterpret_cast<char*>(cache.key_buf + head * head_stride);
      for (int32_t din = 0; din < group.n_embed_dim; din++) {
        for (int i = 0; i < n_valid; i++) {
          head_ptr[fromFlatOffset(group.n_embed_dim, ctx_size, K_TILE, din, i)] = *scratch_ptr++;
        }
      }
    } else {
      char* head_ptr = reinterpret_cast<char*>(cache.val_buf + head * head_stride);
      for (int32_t i = 0; i < n_valid; i++) {
        for (int32_t dout = 0; dout < group.n_embed_dim; dout++) {
          head_ptr[fromFlatOffset(ctx_size, group.n_embed_dim, V_TILE, i, dout)] = *scratch_ptr++;
        }
      }
    }
  }

  fs->seekg((n_heads - cache.n_heads) *
                static_cast<uint32_t>(group.n_embed_dim * n_valid * group.n_bytes),
            std::ios::cur);
}

void NativeKV::dumpHead(CacheGroup& group,
                        KVTensor& cache,
                        uint32_t head,
                        int32_t n_valid,
                        int32_t /*variant*/,
                        int32_t ctx_size,
                        void* data) {
  if (group.n_bytes != 1 || group.m_quantized != true) {
    State::error("Native KV only supports 8-bit KV$");
  }
  uint32_t head_stride = static_cast<uint32_t>(group.n_embed_dim * ctx_size * group.n_bytes);

  if (head > static_cast<uint32_t>(cache.n_heads)) {
    memset(data, 128, static_cast<uint32_t>(2 * group.n_embed_dim * n_valid * group.n_bytes));
    return;
  }

  char* scratch_ptr = reinterpret_cast<char*>(data);
  char* head_ptr    = reinterpret_cast<char*>(cache.key_buf + head * head_stride);
  for (int i = 0; i < n_valid; i++) {
    for (int32_t din = 0; din < group.n_embed_dim; din++) {
      *scratch_ptr++ = head_ptr[fromFlatOffset(group.n_embed_dim, ctx_size, K_TILE, din, i)];
    }
  }

  head_ptr = reinterpret_cast<char*>(cache.val_buf) + head * head_stride;
  for (int i = 0; i < n_valid; i++) {
    for (int32_t dout = 0; dout < group.n_embed_dim; dout++) {
      *scratch_ptr++ = head_ptr[fromFlatOffset(ctx_size, group.n_embed_dim, V_TILE, i, dout)];
    }
  }

  for (int i = 0; i < (2 * group.n_embed_dim * n_valid * group.n_bytes); i++) {
    reinterpret_cast<char*>(data)[i] += 128;
  }
}

void NativeKV::dumpCache(CacheGroup& group,
                         KVTensor& cache,
                         std::ofstream* fs,
                         bool is_key,
                         int32_t n_valid,
                         uint32_t n_heads,
                         int32_t /*variant*/,
                         int32_t ctx_size) {
  GENIE_KV_TRACE();
  if (group.n_bytes != 1 || group.m_quantized != true) {
    State::error("Native KV only supports 8-bit KV$");
  }

  uint32_t head_stride = static_cast<uint32_t>(group.n_embed_dim * ctx_size * group.n_bytes);

  // Create a scratch buffer to help minimize IO calls, and allow for post-processing (uint8->int8)
  std::vector<char> scratch(static_cast<size_t>(group.n_embed_dim * n_valid * group.n_bytes));

  for (uint32_t head = 0; head < cache.n_heads; head++) {
    char* scratch_ptr = scratch.data();
    if (is_key) {
      char* head_ptr = reinterpret_cast<char*>(cache.key_buf) + head * head_stride;
      for (int32_t din = 0; din < group.n_embed_dim; din++) {
        for (int i = 0; i < n_valid; i++) {
          *scratch_ptr++ = head_ptr[fromFlatOffset(group.n_embed_dim, ctx_size, K_TILE, din, i)];
        }
      }
    } else {
      char* head_ptr = reinterpret_cast<char*>(cache.val_buf) + head * head_stride;
      for (int i = 0; i < n_valid; i++) {
        for (int32_t dout = 0; dout < group.n_embed_dim; dout++) {
          *scratch_ptr++ = head_ptr[fromFlatOffset(ctx_size, group.n_embed_dim, V_TILE, i, dout)];
        }
      }
    }

    for (auto& ch : scratch) {
      ch += 128;
    }
    fs->write(scratch.data(), static_cast<std::streamsize>(scratch.size()));
  }

  fs->seekp((n_heads - cache.n_heads) *
                static_cast<uint32_t>(group.n_embed_dim * n_valid * group.n_bytes),
            std::ios::cur);
}

void NativeKV::dumpCache(CacheGroup& group,
                         KVTensor& cache,
                         Buffer* kv_buff,
                         bool is_key,
                         int32_t n_valid,
                         uint32_t n_heads,
                         int32_t /*variant*/,
                         int32_t ctx_size) {
  if (group.n_bytes != 1 || group.m_quantized != true) {
    State::error("Native KV only supports 8-bit KV$");
  }

  uint32_t head_stride = static_cast<uint32_t>(group.n_embed_dim * ctx_size * group.n_bytes);

  // Create a scratch buffer to help minimize IO calls, and allow for post-processing (uint8->int8)
  std::vector<char> scratch(static_cast<size_t>(group.n_embed_dim * n_valid * group.n_bytes));

  for (uint32_t head = 0; head < cache.n_heads; head++) {
    char* scratch_ptr = scratch.data();
    if (is_key) {
      char* head_ptr = reinterpret_cast<char*>(cache.key_buf) + head * head_stride;
      for (int32_t din = 0; din < group.n_embed_dim; din++) {
        for (int i = 0; i < n_valid; i++) {
          *scratch_ptr++ = head_ptr[fromFlatOffset(group.n_embed_dim, ctx_size, K_TILE, din, i)];
        }
      }
    } else {
      char* head_ptr = reinterpret_cast<char*>(cache.val_buf) + head * head_stride;
      for (int32_t i = 0; i < n_valid; i++) {
        for (int32_t dout = 0; dout < group.n_embed_dim; dout++) {
          *scratch_ptr++ = head_ptr[fromFlatOffset(ctx_size, group.n_embed_dim, V_TILE, i, dout)];
        }
      }
    }

    for (auto& ch : scratch) {
      ch += 128;
    }
    kv_buff->appendBuffer(reinterpret_cast<uint8_t*>(scratch.data()),
                          static_cast<uint32_t>(scratch.size()));
  }

  kv_buff->setPosFromCurr(static_cast<int32_t>(n_heads - cache.n_heads) *
                          (group.n_embed_dim * n_valid * group.n_bytes));
}
}  // namespace qualla
