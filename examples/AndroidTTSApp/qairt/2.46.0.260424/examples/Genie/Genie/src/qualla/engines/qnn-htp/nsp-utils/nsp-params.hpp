//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "nlohmann/json.hpp"

namespace qualla {

struct RopeScalingParams {
  enum RopeType { DEFAULT, ROPE_LLAMA3, ROPE_LONGROPE, ROPE_QWEN2VL_MROPE, ROPE_QWEN2VL, ROPE_QWEN3VL_MROPE, ROPE_YARN, ROPE_LINEAR, ROPE_HUNYUAN} rope_type = DEFAULT;

  // This should be a union, but running into compilation issues with non-trivial dtr/copy-ctr
  struct {
    double factor;
    double low_freq_factor;
    double high_freq_factor;
    int original_max_position_embeddings;
  } llama3_params{};

  struct {
    double factor;
    std::vector<double> long_factor;
    std::vector<double> short_factor;
    int original_max_position_embeddings;
  } longrope_params{};

  struct {
    uint32_t spatial_merge_size;
    uint32_t patch_size;
    uint32_t window_size;
  } qwen2vl_params{};

struct {
    uint32_t time_step;
    uint32_t spatial_merge_size;
    std::vector<uint32_t> mrope_section;
  } mrope_params{};


  struct {
    double              factor;
    size_t              dim;
    double              extrapolation_factor;
    double              attn_factor;
    double              beta_fast;
    double              beta_slow;
    int                 original_max_position_embeddings;
  } yarn_params{};

  struct {
    double factor = 1.0;
  } linear_params{};

  struct {
    double base_factor    = 1000.0;
    double dim_factor     = 128.0;
    double offset_factor  = 2.0;
  } hunyuan_params{};

  RopeScalingParams() {}
};

struct PositionalEncoding {
  enum EncodingType : uint8_t { ROPE = 0x0, ABSOLUTE = 0x1, ALIBI = 0x2, NONE = 0x3, UNDEFINED = 0xff } type;
  struct {
    int32_t dims;
    double theta;
    RopeScalingParams rope_scaling;
  } rope_params{};

  PositionalEncoding() { type = ROPE; }
};

struct LongContextParams {
  enum Mode : uint8_t { DISABLED = 0, SLIDING_WINDOW = 1, KEYDIFF = 2 } mode = DISABLED;

  double anchor_alpha{0.02};
  int32_t sink_tokens{0};
  int32_t update_frequency{128};
  int32_t window_size{0};
  std::string scoring_network;
  LongContextParams() {}
};

struct CacheGroupParams {
  std::string prefix{"past_"};

  std::string attention_mask_tensor_name{""};
  std::string cache_index_tensor_name{""};
  LongContextParams longcontext_params;
  CacheGroupParams() {}
};

using CacheGroupParamsMap = std::map<std::string, CacheGroupParams>;

struct VisionParam {
  enum VisionType : uint8_t { UNDEFINED = 0, DEFINED = 1} type = UNDEFINED;
  uint32_t height{0};
  uint32_t width{0};
};

// Helper functions for converting to/from jsom
void from_json(const nlohmann::json& j, PositionalEncoding& p);
void to_json(nlohmann::json& j, const PositionalEncoding& p);
void from_json(const nlohmann::json& j, RopeScalingParams& p);
void to_json(nlohmann::json& j, const RopeScalingParams& p);
void from_json(const nlohmann::json& j, LongContextParams& p);
void to_json(nlohmann::json& j, const LongContextParams& p);
void from_json(const nlohmann::json& j, CacheGroupParams& p);
void to_json(nlohmann::json& j, const CacheGroupParams& p);
void from_json(const nlohmann::json& j, CacheGroupParamsMap& p);
void to_json(nlohmann::json& j, const CacheGroupParamsMap& p);
void from_json(const nlohmann::json& j, VisionParam& p);
void to_json(nlohmann::json& j, const VisionParam& p);

}  // namespace qualla
