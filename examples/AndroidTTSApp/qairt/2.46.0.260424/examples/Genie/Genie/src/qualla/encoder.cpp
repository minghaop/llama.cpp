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

// Encoders
#include "encoders/image-encoders/imageEncoder.hpp"
#include "encoders/text-encoders/LUT.hpp"
#include "encoders/text-encoders/basic.hpp"
#include "qualla/encoder.hpp"

namespace fs = std::filesystem;

namespace qualla {

Encoder::Encoder(std::shared_ptr<Env> env, const std::string& type, const nlohmann::json& json)
    : _type(type), _env(env) {
  Timer start;

  __DEBUG("embedding-new: {} config {}", type, json.dump());
}

void Encoder::input_names(std::unordered_set<std::string>& /*inputTensorNames*/) {
  __ERROR("{}-Encoder does not support input_names method", _type);
}

void Encoder::output_dimensions(std::vector<std::uint32_t>& /*outputDimensions*/) {
  __ERROR("{}-Encoder does not support output_dimensions method", _type);
}

void Encoder::outputTensorQuantParam(std::string& /*dataType*/,
                                     double& /*scale*/,
                                     int32_t& /*offset*/,
                                     float& /*byteWidth*/) {
  __ERROR("{}-Encoder does not support outputTensorQuantParam method", _type);
}

void Encoder::get_dimensions(std::vector<std::uint32_t>& /*dimensions*/, LayerType /*layerType*/) {
  __ERROR("{}-Encoder does not support get_dimensions method", _type);
}

void Encoder::get_tensorQuantParam(std::string& /*dataType*/,
                                   double& /*scale*/,
                                   int32_t& /*offset*/,
                                   size_t& /*byteWidth*/,
                                   LayerType /*layerType*/) {
  __ERROR("{}-Encoder does not support get_tensorQuantParam method", _type);
}

Encoder::KPIs& Encoder::kpis() {
  __ERROR("{}-Encoder does not support kpis method", _type);
  return _kpis;
}

Encoder::~Encoder() {}

// Encode tokens
bool Encoder::encode(const std::vector<int32_t>& /*tokens*/, std::vector<uint8_t>& /*output*/) {
  __ERROR("{}-Encoder does not support encode method", _type);
  return false;
}

// Encode sentence
bool Encoder::encode(const std::string& /*str*/,
                     std::vector<uint8_t>& /*output*/,
                     std::vector<int>& /*tokenizedInput*/) {
  __ERROR("{}-Encoder does not support encode method", _type);
  return false;
}

size_t Encoder::getEmbeddingLutSize() {
  __ERROR("{}-Encoder does not support getEmbeddingLutSize method", _type);
  return 0;
}

void* Encoder::getEmbeddingLut() {
  __ERROR("{}-Encoder does not support getEmbeddingLut method", _type);
  return nullptr;
}

// Encode image
bool Encoder::encode(const std::unordered_map<std::string, std::vector<uint8_t>>& /*inputs*/,
                     std::vector<uint8_t>& /*image_features*/) {
  __ERROR("{}-Encoder does not support encode method", _type);
  return false;
}

int32_t Encoder::getLastToken() {
  __ERROR("{}-Encoder does not support encode method", _type);
  return 0;
}

// Create API
std::unique_ptr<Encoder> Encoder::create(std::shared_ptr<Env> env,
                                         const std::string& /*name*/,
                                         const nlohmann::json& conf) {
  const std::string type = qualla::Config::optional<std::string>(conf, "type", Embedding::TYPE);

  if (type == Embedding::TYPE) {
    return std::make_unique<Embedding>(env, conf);
  }
  if (type == LUT::TYPE) {
    return std::make_unique<LUT>(env, conf);
  }
  if (type == ImageEncoder::TYPE) {
    return std::make_unique<ImageEncoder>(env, conf);
  }

  throw std::runtime_error(type + ": encoder not found");
}

std::unique_ptr<Encoder> Encoder::create(std::shared_ptr<Env> env,
                                         const std::string& name,
                                         std::istream& json_stream) {
  return create(env, name, nlohmann::json::parse(json_stream));
}

std::unique_ptr<Encoder> Encoder::create(std::shared_ptr<Env> env,
                                         const std::string& name,
                                         const fs::path& json_path) {
  if (!fs::exists(json_path)) {
    throw std::runtime_error(json_path.string() + ": file does not exist");
  }
  std::ifstream ifs(json_path);
  return create(env, name, ifs);
}

bool Encoder::applyLoraAdapter(std::string lora_adapter_name, std::string engine_role) {
  if (!_engine) {
    __ERROR(
        "Encoder::applyLoraAdapter: specified {} engine type is invalid for apply LoRA adapters.",
        engine_role);
    return false;
  }
  _kpis.lora.last_usec  = 0;
  _kpis.lora.total_usec = 0;
  Timer start;
  auto& engine = *_engine;
  if (!engine.applyLoraAdapter(lora_adapter_name)) {
    __WARN("encoder-applyLoraAdapter: failed for {}", lora_adapter_name);
    return false;
  }
  _kpis.lora.update(start.elapsed_usec());
  return true;
}

bool Encoder::applyLoraStrength(std::string tensor_name,
                                float tensor_val,
                                std::string engine_role) {
  if (!_engine) {
    __ERROR("Encoder::applyLoraAdapter: specified {} engine type is invalid for set LoRA strength.",
            engine_role);
    return false;
  }
  auto& engine = *_engine;
  if (!engine.applyLoraStrength(tensor_name, tensor_val)) {
    __WARN("encoder-applyLoraStrength: failed for {}", tensor_name);
    return false;
  }
  return true;
}

void Encoder::setPerformancePolicy(qualla::PerformanceProfile policy) {
  m_perfProfile = policy;
  auto& engine  = *_engine;
  engine.setPerfProfile(policy);
}

qualla::PerformanceProfile& Encoder::getPerformancePolicy() { return m_perfProfile; }

void Encoder::registerModelAdaptor(std::shared_ptr<qualla::ModelIOAdaptor> adaptor) {
  __INFO("Dialog::registerModelAdaptor");
  _engine->registerModelAdaptor(adaptor);
}

}  // namespace qualla
