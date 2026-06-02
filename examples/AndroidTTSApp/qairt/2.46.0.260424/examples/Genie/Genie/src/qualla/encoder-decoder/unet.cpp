//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>

#include "qualla/detail/config.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/diffusion-scheduler.hpp"
#include "unet.hpp"

namespace fs = std::filesystem;

namespace qualla {

UNET::UNET(std::shared_ptr<Env> env, const nlohmann::json& json) : EncoderDecoder(env, "unet", json) {
  Timer start;

  using qc = qualla::Config;

  // Store json configuration for later use
  _config = json;

  // Create dummy context required by enging
  std::shared_ptr<Context> ctx =
      Context::create(_env, _type, qc::optional<nlohmann::json>(json, "context", {}));

  // Create Engine
  const nlohmann::json& eng_conf = qc::mandatory<nlohmann::json>(json, "engine");
  _engine                      = Engine::create(*ctx, eng_conf);
  // pull model-input height and width parameters from QNN ctx-cache

  using FF = Engine::Feature::Flags;
  if (!_engine->supports(FF::OUTPUT_EMBEDDINGS))
    throw std::runtime_error("engine must output embeddings");

  // Initialize random number generator with seed
  if (eng_conf.contains("training_config")) {
    _random_seed = eng_conf["training_config"]["random_seed"];
  }

  _generator = std::make_shared<CPUGenerator>(_random_seed);

  // Initialize diffusion scheduler if scheduler config is provided
  _noise_scheduler     = nullptr;
  _num_train_timesteps = 1000;  // Default value

  if (_config.contains("scheduler") && _config["scheduler"].contains("path")) {
    std::string scheduler_config_path = _config["scheduler"]["path"].get<std::string>();
    _noise_scheduler                  = DiffusionScheduler::create(_env, scheduler_config_path);

    if (_noise_scheduler) {
      _num_train_timesteps = _noise_scheduler->num_train_timesteps;
    }
  }

  _kpis.init.update(start.elapsed_usec());
}

UNET::~UNET() {}

void UNET::setupTrainingData(TrainingData& training_data) {
  // Get the model input data (latents) - this should be provided in training_data
  // For now, we'll use encoder_hidden_states as model_input if available
  auto& input_batches = training_data["image_embeddings"];
  if (input_batches.empty() || input_batches[0].empty()) {
    throw std::runtime_error("Training data is empty");
  }

  const size_t max_iterations = input_batches.size();
  const size_t batch_size     = input_batches[0].size();

  // Check if user provided noise_input and timestep data
  const bool has_user_noise    = training_data.find("noise_input") != training_data.end();
  const bool has_user_timestep = training_data.find("timestep") != training_data.end();

  // Prepare noisy_model_input and target batches
  std::vector<std::vector<Tensor>> noisy_model_input_batches(max_iterations);
  std::vector<std::vector<Tensor>> target_batches(max_iterations);
  std::vector<std::vector<Tensor>> timestep_batches(max_iterations);

  for (size_t i = 0; i < max_iterations; ++i) {
    for (size_t j = 0; j < batch_size; ++j) {
      Tensor& model_input = input_batches[i][j];

      // Get model input data and size
      size_t data_size = model_input.getSize();
      if (data_size == 0) {
        throw std::runtime_error(
            fmt::format("Invalid model input data size at iteration {}, batch {}", i, j));
      }

      // Get quantization params from model_input
      TensorQuantizationParams quant_params = model_input.getQuantizationParams();

      // Handle noise: use provided or generate
      Tensor noise_tensor;
      if (has_user_noise) {
        // Use user-provided noise
        auto& noise_input_batches = training_data["noise_input"];
        if (i >= noise_input_batches.size() || j >= noise_input_batches[i].size()) {
          throw std::runtime_error(
              fmt::format("User-provided noise missing for iteration {}, batch {}", i, j));
        }
        noise_tensor = noise_input_batches[i][j];

        // Validate noise tensor size matches model input
        if (noise_tensor.getSize() != data_size) {
          throw std::runtime_error(
              fmt::format("User-provided noise size {} doesn't match model input size {} at "
                          "iteration {}, batch {}",
                          noise_tensor.getSize(),
                          data_size,
                          i,
                          j));
        }
      } else {
        // Generate random noise
        std::vector<float> noise_vec = _generator->randn<float>(data_size, 0.0, 1.0);

        std::vector<uint32_t> noise_tensor_dimensions;
        _engine->getTensorDimensions(LayerType::INPUT, noise_tensor_dimensions);
        noise_tensor = Tensor(noise_vec.data(),
                              model_input.getDataType(),
                              quant_params,
                              data_size,
                              noise_tensor_dimensions,
                              false);  // user_owned = false, Tensor manages memory
      }

      // Handle timestep: use provided or generate
      int timestep;
      Tensor timestep_tensor;
      if (has_user_timestep) {
        // Use user-provided timestep
        auto& timestep_input_batches = training_data["timestep"];
        if (i >= timestep_input_batches.size() || j >= timestep_input_batches[i].size()) {
          throw std::runtime_error(
              fmt::format("User-provided timestep missing for iteration {}, batch {}", i, j));
        }
        timestep_tensor = timestep_input_batches[i][j];

        // Extract timestep value from tensor
        if (timestep_tensor.getSize() < 1) {
          throw std::runtime_error(fmt::format(
              "User-provided timestep tensor is empty at iteration {}, batch {}", i, j));
        }

        // Get timestep value based on data type
        if (timestep_tensor.getDataType() == TENSOR_DATATYPE_INT_64) {
          timestep = *static_cast<const int64_t*>(timestep_tensor.getData());
        } else {
          throw std::runtime_error(fmt::format(
            "Unsupported timestep data type at iteration {}, batch {}. Expected INT_64", i, j));
        }
      } else {
        // Generate random timestep
        timestep = _generator->uniform_int(0, _num_train_timesteps - 1);
        // Create timestep data
        int64_t timestep_value = static_cast<int64_t>(timestep);

        std::vector<uint32_t> timestep_dimensions = {1};
        try {
          _engine->getTensorDimensions(LayerType::TIMESTEP, timestep_dimensions);
          if (timestep_dimensions.empty()) {
            timestep_dimensions = {1};
          }
        } catch (...) {
          timestep_dimensions = {1};
        }

        // Create quantization parameters for timestep
        timestep_tensor = Tensor(&timestep_value,
                                 TENSOR_DATATYPE_INT_64,
                                 {1.0, 0},
                                 1,
                                 timestep_dimensions,
                                 false);  // user_owned = false, Tensor manages memory
      }

      // Create noisy model input using Tensor-based add_noise
      Tensor noisy_tensor;
      if (_noise_scheduler) {
        std::vector<int> timesteps = {timestep};
        noisy_tensor = _noise_scheduler->add_noise(model_input, noise_tensor, timesteps);

        if (noisy_tensor.getSize() != data_size) {
          throw std::runtime_error(fmt::format(
              "Scheduler returned wrong size: expected {}, got {} at iteration {}, batch {}",
              data_size,
              noisy_tensor.getSize(),
              i,
              j));
        }
      }

      // Create tensor for target (noise) - reuse noise_tensor
      Tensor target_tensor = noise_tensor;
      // Add to batches
      noisy_model_input_batches[i].push_back(noisy_tensor);
      target_batches[i].push_back(target_tensor);
      timestep_batches[i].push_back(timestep_tensor);
    }
  }

  // Add the prepared tensors to training_data
  training_data["noisy_model_input"] = noisy_model_input_batches;
  training_data["noise_input"]       = target_batches;
  training_data["timestep"]          = timestep_batches;
}

std::vector<float> UNET::train(TrainingData& training_data) {
  setupTrainingData(training_data);
  // Call the engine's training function with the complete training data
  return _engine->run_on_device_training(training_data);
}

bool UNET::saveLoraAdapter(std::string lora_adapter_name, std::string engine_role) {
  if (!_engine) {
    __ERROR("UNET::saveLoraAdapter: specified {} engine type is invalid for saving LoRA adapters.",
            engine_role);
    return false;
  }
  if (!_engine->saveLoraAdapter(lora_adapter_name)) {
    __WARN("UNET::saveLoraAdapter: failed for {}", lora_adapter_name);
    return false;
  }
  return true;
}

void UNET::get_dimensions(std::vector<std::uint32_t>& dimensions, LayerType layerType) {
  if (!_engine) {
    throw std::runtime_error("Engine not initialized");
  }

  _engine->getTensorDimensions(layerType, dimensions);
}

void UNET::get_tensorQuantParam(
    std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth, LayerType layerType) {
  if (!_engine) {
    throw std::runtime_error("Engine not initialized");
  }

  _engine->getTensorParam(layerType, dataType, scale, offset, bitWidth);
}

}  // namespace qualla
