//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <atomic>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "qualla/detail/CPUGenerator.hpp"
#include "qualla/detail/exports.h"
#include "qualla/diffusion-scheduler.hpp"
#include "qualla/encoder-decoder.hpp"
#include "qualla/engine.hpp"
#include "qualla/env.hpp"

namespace qualla {

class UNET : public EncoderDecoder {
 public:
  static constexpr const char* TYPE = "unet";

  UNET(std::shared_ptr<Env> env, const nlohmann::json& conf);
  virtual ~UNET();

  virtual std::vector<float> train(TrainingData& training_data) override;

  virtual bool saveLoraAdapter(std::string lora_adapter_name, std::string engine_role) override;

  virtual void get_dimensions(std::vector<std::uint32_t>& inputDimensions,
                              LayerType layerType) override;

  virtual void get_tensorQuantParam(std::string& dataType,
                                    double& scale,
                                    int32_t& offset,
                                    size_t& bitWidth,
                                    LayerType layerType) override;

  void setupTrainingData(TrainingData& training_data);

 protected:
  std::vector<std::uint32_t> _output_dimensions{};
  nlohmann::json _config;

  // Noise scheduler and random number generation
  std::shared_ptr<DiffusionScheduler> _noise_scheduler;
  // std::mt19937 _generator;
  std::shared_ptr<CPUGenerator> _generator;
  uint32_t _random_seed = 0;
  int _num_train_timesteps;
};

}  // namespace qualla
