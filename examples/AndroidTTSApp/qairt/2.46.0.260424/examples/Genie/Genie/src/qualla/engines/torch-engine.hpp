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
#include <qualla/detail/config.hpp>
#include <qualla/detail/timer.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "GenieTraining.hpp"
#include "qualla/LoraConfig.hpp"
#include "qualla/engine.hpp"

// Forward declare the GenieTraining interface
namespace genie {
class GenieTraining;
typedef void (*DestroyGenieTrainingInterfaceFnType_t)(GenieTraining*);
}  // namespace genie

namespace qualla {

class LibTorchEngine : public Engine {
 public:

  static constexpr const char* TYPE = "libtorch";

  LibTorchEngine(Context& ctx, const nlohmann::json& json);

  ~LibTorchEngine();

  std::vector<float> run_on_device_training(TrainingData& trainingData) override;

  virtual bool saveLoraAdapter(std::string lora_adapter_name) override;

  size_t process(const std::vector<int32_t>& tokens,
                 std::vector<float>& output,
                 bool output_all = false) override;

  size_t process(const std::vector<int32_t>& tokens,
                 Tensor& output,
                 bool output_all = false) override;

  size_t process(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
                 std::vector<uint8_t>& outputs) override;

  bool save_snapshot(const std::filesystem::path& path) override;

  bool load_snapshot(const std::filesystem::path& path) override;

  bool save_training_parameters(const std::filesystem::path& path) override;

  virtual void getTensorParam(LayerType layerType,
                              std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& bitWidth) override;

  virtual void getTensorDimensions(LayerType layerType,
                                   std::vector<std::uint32_t>& dimensions) override;

  virtual void getInputTensorNames(std::unordered_set<std::string>& inputTensorNames) override;

 private:
  genie::GenieTraining* m_trainingInterface;
  genie::DestroyGenieTrainingInterfaceFnType_t m_destroyTrainingInterfaceFn;
  // LoRA params and configs
  LoraConfigType lora_conf_type{LoraConfigType::LORA_DISABLE};
  std::shared_ptr<LoraConfig> lora_config;
};

}  // namespace qualla
