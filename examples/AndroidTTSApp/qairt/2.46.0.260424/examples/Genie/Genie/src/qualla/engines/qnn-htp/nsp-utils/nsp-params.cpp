//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "fmt/format.h"
#include "nsp-params.hpp"
#include "qualla/detail/config.hpp"

namespace qualla {

// Helper functions to convert enum to string
NLOHMANN_JSON_SERIALIZE_ENUM(RopeScalingParams::RopeType,
                             {{RopeScalingParams::DEFAULT, "default"},
                              {RopeScalingParams::ROPE_LLAMA3, "llama3"},
                              {RopeScalingParams::ROPE_LONGROPE, "longrope"},
                              {RopeScalingParams::ROPE_QWEN2VL_MROPE, "qwen2vl-mrope"},
                              {RopeScalingParams::ROPE_QWEN2VL, "qwen2vl"},
                              {RopeScalingParams::ROPE_QWEN3VL_MROPE, "qwen3vl-mrope"},
                              {RopeScalingParams::ROPE_YARN, "yarnrope"},
                              {RopeScalingParams::ROPE_LINEAR, "linear"},
                              {RopeScalingParams::ROPE_HUNYUAN, "hunyuan"}})

NLOHMANN_JSON_SERIALIZE_ENUM(PositionalEncoding::EncodingType,
                             {{PositionalEncoding::UNDEFINED, "undefined"},
                              {PositionalEncoding::ROPE, "rope"},
                              {PositionalEncoding::ABSOLUTE, "absolute"},
                              {PositionalEncoding::ALIBI, "alibi"},
                              {PositionalEncoding::NONE, "none"}})

NLOHMANN_JSON_SERIALIZE_ENUM(LongContextParams::Mode,
                             {{LongContextParams::DISABLED, "disabled"},
                              {LongContextParams::SLIDING_WINDOW, "sliding-window"},
                              {LongContextParams::KEYDIFF, "keydiff"}})

// Utility functions to convert structs from/to json for parsing/dumping
void from_json(const nlohmann::json& j, RopeScalingParams& p) {
  p.rope_type = Config::optional(j, "rope-type", RopeScalingParams::DEFAULT);
  if (p.rope_type == RopeScalingParams::ROPE_LLAMA3) {
    try {
      j.at("factor").get_to(p.llama3_params.factor);
      j.at("low-freq-factor").get_to(p.llama3_params.low_freq_factor);
      j.at("high-freq-factor").get_to(p.llama3_params.high_freq_factor);
      j.at("original-max-position-embeddings")
          .get_to(p.llama3_params.original_max_position_embeddings);
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error(
          fmt::format("Parsing error for llama3 rope scaling - {}\n"
                      "llama3 requires keys ['original-max-position-embeddings', 'factor', "
                      "'low-freq-factor', 'high-freq-factor'].\n"
                      "Found config - {}",
                      e.what(),
                      j.dump()));
    }
  } else if (p.rope_type == RopeScalingParams::ROPE_LONGROPE) {
    try {
      j.at("original-max-position-embeddings")
          .get_to(p.longrope_params.original_max_position_embeddings);
      j.at("long-factor").get_to(p.longrope_params.long_factor);
      j.at("short-factor").get_to(p.longrope_params.short_factor);
      if (j.contains("factor"))
        j.at("factor").get_to(p.longrope_params.factor);
      else
        p.longrope_params.factor = j.at("max-position-embeddings").get<double>() /
                                   p.longrope_params.original_max_position_embeddings;
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error(
          fmt::format("Parsing error for longrope scaling - {}\n"
                      "LongRope requires keys ['original-max-position-embeddings', 'factor' or "
                      "'max-position-embeddings', 'long-factor', 'short-factor'].\n"
                      "Found config - {}",
                      e.what(),
                      j.dump()));
    }
  } else if (p.rope_type == RopeScalingParams::ROPE_QWEN2VL_MROPE || p.rope_type == RopeScalingParams::ROPE_QWEN3VL_MROPE) {
    try {
      if (j.contains("time-step"))
        j.at("time-step").get_to(p.mrope_params.time_step);
      else
        p.mrope_params.time_step = 50;
      if (j.contains("spatial-merge-size"))
        j.at("spatial-merge-size").get_to(p.mrope_params.spatial_merge_size);
      else
        p.mrope_params.spatial_merge_size = 2;
      if (j.contains("mrope-section")) {
        j.at("mrope-section").get_to(p.mrope_params.mrope_section);
        if (p.mrope_params.mrope_section.size() != 3) {
          throw std::runtime_error(
          fmt::format("The length of mrope-section must be 3.\n",
                      "Found mrope-section - {}",
                      j.dump()));
        }
      } else {
        p.mrope_params.mrope_section = {16, 24, 24};
      }
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error(
          fmt::format("Parsing error for qwen2vl-mrope rope scaling - {}\n",
                      "Found config - {}",
                      e.what(),
                      j.dump()));
    }
  } else if (p.rope_type == RopeScalingParams::ROPE_QWEN2VL) {
    try {
      if (j.contains("spatial-merge-size"))
        j.at("spatial-merge-size").get_to(p.qwen2vl_params.spatial_merge_size);
      else
        p.qwen2vl_params.spatial_merge_size = 2;
      if (j.contains("patch-size"))
        j.at("patch-size").get_to(p.qwen2vl_params.patch_size);
      else
        p.qwen2vl_params.patch_size = 14;
      if (j.contains("window-size"))
        j.at("window-size").get_to(p.qwen2vl_params.window_size);
      else
        p.qwen2vl_params.window_size = 112;
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error(
          fmt::format("Parsing error for qwen2vl rope scaling - {}\n"
                      "qwen2vl requires keys ['height', 'width'].\n"
                      "Found config - {}",
                      e.what(),
                      j.dump()));
    }
  } else if (p.rope_type == RopeScalingParams::ROPE_YARN) {
    try {
      j.at("factor").get_to(p.yarn_params.factor);
      j.at("dim").get_to(p.yarn_params.dim);
      if (p.yarn_params.dim <= 0 || p.yarn_params.dim % 2 != 0) {
        throw std::runtime_error(fmt::format(
            "YaRN dim must be a positive even number, got: {}", p.yarn_params.dim));
      }
      if (p.yarn_params.factor <= 1.0) {
          throw std::runtime_error(fmt::format(
              "YaRN factor must be greater than 1.0, got: {}", p.yarn_params.factor));
      }
      j.at("extrapolation-factor").get_to(p.yarn_params.extrapolation_factor);
      j.at("attn-factor").get_to(p.yarn_params.attn_factor);
      if (p.yarn_params.extrapolation_factor <= 0 || p.yarn_params.attn_factor <= 0) {
        throw std::runtime_error("YaRN extrapolation-factor and attn-factor must be positive");
      }
      j.at("beta-fast").get_to(p.yarn_params.beta_fast);
      j.at("beta-slow").get_to(p.yarn_params.beta_slow);
      if (p.yarn_params.beta_fast < p.yarn_params.beta_slow || p.yarn_params.beta_slow <= 0) {
        throw std::runtime_error("YaRN requires beta-fast >= beta-slow > 0");
      }
      j.at("original-max-position-embeddings").get_to(p.yarn_params.original_max_position_embeddings);
      if (p.yarn_params.original_max_position_embeddings <= 0) {
        throw std::runtime_error("YaRN original-max-position-embeddings must be positive");
      }
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error(fmt::format("Parsing error for yarn rope scaling - {}\n"
      "yarn requires keys ['original-max-position-embeddings', 'factor', 'dim', 'extrapolation-factor', 'attn-factor', 'beta-fast', 'beta-slow'].\n"
      "Found config - {}", e.what(), j.dump()));
    }
  } else if (p.rope_type == RopeScalingParams::ROPE_LINEAR) {
    try {
      if (j.contains("factor")) {
        j.at("factor").get_to(p.linear_params.factor);
      } else {
        p.linear_params.factor = 1.0;
      }
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error(
          fmt::format("Parsing error for linear scaling - {}\n"
                      "linear requires keys ['factor'].\n"
                      "Found config - {}",
                      e.what(),
                      j.dump()));
    }
  } else if (p.rope_type == RopeScalingParams::ROPE_HUNYUAN) {
    try {
      if (j.contains("base-factor")) {
        j.at("base-factor").get_to(p.hunyuan_params.base_factor);
      } else {
        p.hunyuan_params.base_factor = 1000.0;
      }
      if (j.contains("dim-factor")) {
        j.at("dim-factor").get_to(p.hunyuan_params.dim_factor);
      } else {
        p.hunyuan_params.dim_factor = 128.0;
      }
      if (j.contains("offset-factor")) {
        j.at("offset-factor").get_to(p.hunyuan_params.offset_factor);
      } else {
        p.hunyuan_params.offset_factor = 2.0;
      }
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error(
          fmt::format("Parsing error for hunyuan scaling - {}\n"
                      "hunyuan supports optional keys ['base-factor', 'dim-factor', 'offset-factor'].\n"
                      "Found config - {}",
                      e.what(),
                      j.dump()));
    }
  }
}

void to_json(nlohmann::json& j, const RopeScalingParams& p) {
  j["rope-type"] = p.rope_type;
  if (p.rope_type == RopeScalingParams::ROPE_LLAMA3) {
    j["factor"]                           = p.llama3_params.factor;
    j["low-freq-factor"]                  = p.llama3_params.low_freq_factor;
    j["high-freq-factor"]                 = p.llama3_params.high_freq_factor;
    j["original-max-position-embeddings"] = p.llama3_params.original_max_position_embeddings;
  } else if (p.rope_type == RopeScalingParams::ROPE_LONGROPE) {
    j["factor"]                           = p.longrope_params.factor;
    j["long-factor"]                      = p.longrope_params.long_factor;
    j["short-factor"]                     = p.longrope_params.short_factor;
    j["original-max-position-embeddings"] = p.longrope_params.original_max_position_embeddings;
  } else if (p.rope_type == RopeScalingParams::ROPE_QWEN2VL_MROPE || p.rope_type == RopeScalingParams::ROPE_QWEN3VL_MROPE) {
    j["time-step"]                        = p.mrope_params.time_step;
    j["spatial-merge-size"]               = p.mrope_params.spatial_merge_size;
    j["mrope-section"]                    = p.mrope_params.mrope_section;
  } else if (p.rope_type == RopeScalingParams::ROPE_QWEN2VL) {
    j["spatial-merge-size"] = p.qwen2vl_params.spatial_merge_size;
    j["patch-size"]         = p.qwen2vl_params.patch_size;
    j["window-size"]        = p.qwen2vl_params.window_size;
  } else if (p.rope_type == RopeScalingParams::ROPE_YARN) {
    j["factor"]                           = p.yarn_params.factor;
    j["dim"]                              = p.yarn_params.dim;
    j["extrapolation-factor"]             = p.yarn_params.extrapolation_factor;
    j["attn-factor"]                      = p.yarn_params.attn_factor;
    j["beta-fast"]                        = p.yarn_params.beta_fast;
    j["beta-slow"]                        = p.yarn_params.beta_slow;
    j["original-max-position-embeddings"] = p.yarn_params.original_max_position_embeddings;
  } else if (p.rope_type == RopeScalingParams::ROPE_LINEAR) {
    j["factor"]             = p.linear_params.factor;
  } else if (p.rope_type == RopeScalingParams::ROPE_HUNYUAN) {
    j["base-factor"]        = p.hunyuan_params.base_factor;
    j["dim-factor"]         = p.hunyuan_params.dim_factor;
    j["offset-factor"]      = p.hunyuan_params.offset_factor;
  }
}

void from_json(const nlohmann::json& j, PositionalEncoding& p) {
  p.type = Config::optional(j, "type", PositionalEncoding::ROPE);
  if (p.type == PositionalEncoding::ROPE) {
    p.rope_params.dims         = Config::mandatory<int32_t>(j, "rope-dim");
    p.rope_params.theta        = Config::optional<int32_t>(j, "rope-theta", 10000);
    p.rope_params.rope_scaling = Config::optional<RopeScalingParams>(j, "rope-scaling", {});
  }
}

void to_json(nlohmann::json& j, const PositionalEncoding& p) {
  j["type"] = p.type;
  if (p.type == PositionalEncoding::ROPE) {
    j["rope-dim"]     = p.rope_params.dims;
    j["rope-theta"]   = p.rope_params.theta;
    j["rope-scaling"] = p.rope_params.rope_scaling;
  }
}

void from_json(const nlohmann::json& j, LongContextParams& p) {
  p.mode = Config::optional(j, "type", LongContextParams::DISABLED);

  switch(p.mode) {
    case LongContextParams::SLIDING_WINDOW:
      p.sink_tokens = Config::optional<int32_t>(j, "reserved-tokens", 0);
      p.window_size = Config::optional<int32_t>(j, "window-size", 0);
      break;
    case LongContextParams::KEYDIFF:
      p.sink_tokens      = Config::optional<int32_t>(j, "reserved-tokens", 0);
      p.anchor_alpha     = Config::optional<double>(j, "anchor-alpha", 0.02);
      p.update_frequency = Config::optional<int32_t>(j, "update-frequency", 128);
      p.scoring_network  = Config::mandatory<std::string>(j, "scoring-network");
      break;
    default:
      break;
  }
}

void to_json(nlohmann::json& j, const LongContextParams& p) {
  j["type"] = p.mode;
  if (p.mode == LongContextParams::SLIDING_WINDOW) {
    j["reserved-tokens"] = p.sink_tokens;
    j["window-size"] = p.window_size;
  }
  if (p.mode == LongContextParams::KEYDIFF) {
    j["anchor-alpha"]     = p.anchor_alpha;
    j["reserved-tokens"]  = p.sink_tokens;
    j["update-frequency"] = p.update_frequency;
    j["scoring-network"]  = p.scoring_network;
  }
}

void from_json(const nlohmann::json& j, CacheGroupParams& p) {
  p.prefix                     = Config::mandatory<std::string>(j, "prefix");
  p.attention_mask_tensor_name = Config::optional<std::string>(j, "attention-mask-tensor-name", "");
  p.cache_index_tensor_name    = Config::optional<std::string>(j, "cache-index-tensor-name", "");

  if (j.contains("longcontext")) {
    j.at("longcontext").get_to(p.longcontext_params);
  }
}

void to_json(nlohmann::json& j, const CacheGroupParams& p) {
  j["prefix"]                     = p.prefix;
  j["attention-mask-tensor-name"] = p.attention_mask_tensor_name;
  j["cache-index-tensor-name"]    = p.cache_index_tensor_name;
  j["longcontext"]                = p.longcontext_params;
}

void from_json(const nlohmann::json& j, CacheGroupParamsMap& p) {
  for (auto& cacheParamConfig : j) {
    const auto prefix = Config::mandatory<std::string>(cacheParamConfig, "prefix");
    cacheParamConfig.get_to(p[prefix]);
  }
}

void to_json(nlohmann::json& j, const CacheGroupParamsMap& p) {
  j = nlohmann::json::array();
  for (const auto& [_, params] : p) {
    nlohmann::json config = params;
    j.push_back(config);
  }
}

void from_json(const nlohmann::json& j, VisionParam& p) {
  p.type = VisionParam::DEFINED;
  p.height = Config::mandatory<uint32_t>(j, "height");
  p.width  = Config::mandatory<uint32_t>(j, "width");
}

void to_json(nlohmann::json& j, const VisionParam& p) {
  j["height"] = p.height;
  j["width"]  = p.width;
}

}  // namespace qualla
