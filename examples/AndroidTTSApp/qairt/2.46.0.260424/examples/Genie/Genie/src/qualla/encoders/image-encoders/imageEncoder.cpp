//==============================================================================
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
#include <functional>
#include <string>
#include <unordered_map>

#include "imageEncoder.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/timer.hpp"

namespace fs = std::filesystem;

namespace qualla {

ImageEncoder::ImageEncoder(std::shared_ptr<Env> env, const nlohmann::json& json)
    : Encoder(env, "ImageEncoder", json) {
  Timer start;

  using qc = qualla::Config;

  // Create dummy context required by enging
  std::shared_ptr<Context> ctx =
      Context::create(_env, _type, qc::optional<nlohmann::json>(json, "context", {}));

  // Create Engine
  const nlohmann::json& eng_conf = qc::mandatory<nlohmann::json>(json, "engine");
  _engine = Engine::create(*ctx, eng_conf);
  _engine->getTensorDimensions(LayerType::OUTPUT, _output_dimensions);

  if (!_engine->supports(Engine::Feature::Flags::OUTPUT_EMBEDDINGS)) {
    throw std::runtime_error("engine must output embeddings");
  }

  _engine->getPerfProfile(m_defaultPerfProfile);
  m_perfProfile = m_defaultPerfProfile;
  _kpis.init.update(start.elapsed_usec());
}

ImageEncoder::~ImageEncoder() {}

bool ImageEncoder::process(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
                           std::vector<uint8_t>& outputs) {
  Timer start;

  State::clear();

  size_t n = _engine->process(inputs, outputs);
  if (!n) {
    State::error("engine image encoder failed");
    return false;
  }

  return true;
}

// Embedding KPIs helpers

void ImageEncoder::input_names(std::unordered_set<std::string>& inputNames) {
  _engine->getInputTensorNames(inputNames);
}

void ImageEncoder::output_dimensions(std::vector<std::uint32_t>& outputDimensions) {
  outputDimensions = _output_dimensions;
}

void ImageEncoder::outputTensorQuantParam(std::string& dataType,
                                          double& scale,
                                          int32_t& offset,
                                          float& byteWidth) {
  size_t imageByteWidth;
  _engine->getTensorParam(LayerType::OUTPUT, dataType, scale, offset, imageByteWidth);
  byteWidth = static_cast<float>(imageByteWidth);
}

void ImageEncoder::get_dimensions(std::vector<std::uint32_t>& dimensions, LayerType layerType) {
  if (!_engine) {
    throw std::runtime_error("Engine not initialized");
  }

  _engine->getTensorDimensions(layerType, dimensions);
}

void ImageEncoder::get_tensorQuantParam(
    std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth, LayerType layerType) {
  if (!_engine) {
    throw std::runtime_error("Engine not initialized");
  }

  _engine->getTensorParam(layerType, dataType, scale, offset, bitWidth);
}

bool ImageEncoder::encode(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
                          std::vector<uint8_t>& image_features) {
  return process(inputs, image_features);
}

}  // namespace qualla
