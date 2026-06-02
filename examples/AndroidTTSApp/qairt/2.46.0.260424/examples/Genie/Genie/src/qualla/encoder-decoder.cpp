//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <filesystem>
#include <fstream>

#include "qualla/detail/timer.hpp"

// EncoderDecoder
#include "encoder-decoder/unet.hpp"
#include "qualla/encoder-decoder.hpp"

namespace fs = std::filesystem;

namespace qualla {

EncoderDecoder::EncoderDecoder(std::shared_ptr<Env> env,
                               const std::string& type,
                               const nlohmann::json& json)
    : _type(type), _env(env) {
  Timer start;

  __DEBUG("embedding-new: {} config {}", type, json.dump());
}

void EncoderDecoder::input_names(std::unordered_set<std::string>& /*inputTensorNames*/) {
  __ERROR("{}-EncoderDecoder does not support input_names method", _type);
}

void EncoderDecoder::get_dimensions(std::vector<std::uint32_t>& /*dimensions*/,
                                    LayerType /*layerType*/) {
  __ERROR("{}-EncoderDecoder does not support get_dimensions method", _type);
}

void EncoderDecoder::get_tensorQuantParam(std::string& /*dataType*/,
                                          double& /*scale*/,
                                          int32_t& /*offset*/,
                                          size_t& /*byteWidth*/,
                                          LayerType /*layerType*/) {
  __ERROR("{}-EncoderDecoder does not support get_tensorQuantParam method", _type);
}

EncoderDecoder::KPIs& EncoderDecoder::kpis() {
  __ERROR("{}-EncoderDecoder does not support kpis method", _type);
  return _kpis;
}

EncoderDecoder::~EncoderDecoder() {}

std::vector<float> EncoderDecoder::train(TrainingData& /*training_data*/) {
  __ERROR("{}-EncoderDecoder does not support encode method", _type);
  return {};
}

bool EncoderDecoder::saveLoraAdapter(std::string lora_adapter_name, std::string engine_role) {
  __ERROR(
      "{}-EncoderDecoder does not support saveLoraAdapter method. Attempted to save adapter '{}' "
      "for role '{}'",
      _type,
      lora_adapter_name,
      engine_role);
  return false;
}

// Create API
std::unique_ptr<EncoderDecoder> EncoderDecoder::create(std::shared_ptr<Env> env,
                                                       const std::string& /*name*/,
                                                       const nlohmann::json& conf) {
  const std::string type = qualla::Config::optional<std::string>(conf, "type", UNET::TYPE);

  if (type == UNET::TYPE) {
    return std::make_unique<UNET>(env, conf);
  }

  throw std::runtime_error(type + ": encoder not found");
}

std::unique_ptr<EncoderDecoder> EncoderDecoder::create(std::shared_ptr<Env> env,
                                                       const std::string& name,
                                                       std::istream& json_stream) {
  return create(env, name, nlohmann::json::parse(json_stream));
}

std::unique_ptr<EncoderDecoder> EncoderDecoder::create(std::shared_ptr<Env> env,
                                                       const std::string& name,
                                                       const fs::path& json_path) {
  if (!fs::exists(json_path)) {
    throw std::runtime_error(json_path.string() + ": file does not exist");
  }
  std::ifstream ifs(json_path);
  return create(env, name, ifs);
}

}  // namespace qualla
