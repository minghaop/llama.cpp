//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <fmt/format.h>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "qualla/context.hpp"
#include "qualla/engine.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/tensor.hpp"
#include "qualla/detail/timer.hpp"

namespace genie {

class GenieTraining {
 public:
  virtual ~GenieTraining(){};

  virtual bool loadConfig(qualla::Context& ctx, const nlohmann::json& json) = 0;

  virtual std::vector<float> runOnDeviceTraining(qualla::TrainingData& trainingData) = 0;

  virtual size_t process(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
                         std::vector<uint8_t>& outputs) = 0;

  virtual bool saveSnapshot(const std::filesystem::path& path) = 0;

  virtual bool saveLoraAdapter(const std::string& lora_adapter_name,
                               std::vector<std::string> adapter_bins,
                               std::string metadata_dlc) = 0;

  virtual bool loadSnapshot(const std::filesystem::path& path) = 0;

  virtual bool saveTrainingParameters(const std::string& lora_adapter_name) = 0;

  virtual void getTensorParam(qualla::LayerType layerType,
                              std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& bitWidth) = 0;

  virtual void getTensorDimensions(qualla::LayerType layerType,
                                   std::vector<std::uint32_t>& dimensions) = 0;

  virtual void getInputTensorNames(std::unordered_set<std::string>& inputTensorNames) = 0;
};

typedef GenieTraining* (*CreateGenieTrainingInterfaceFnType_t)();
typedef void (*DestroyGenieTrainingInterfaceFnType_t)(GenieTraining*);

}  // namespace genie
