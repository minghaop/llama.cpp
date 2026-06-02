//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Exception.hpp"
#include "PAL/DynamicLoading.hpp"
#include "Trace.hpp"
#include "qualla/detail/Log.hpp"
#include "qualla/detail/timer.hpp"
#include "torch-engine.hpp"

namespace qualla {

LibTorchEngine::LibTorchEngine(Context& ctx, const nlohmann::json& json)
    : Engine(ctx, "libtorch", json),
      m_trainingInterface(nullptr),
      m_destroyTrainingInterfaceFn(nullptr) {
  GENIE_TRACE();
  qualla::Timer start;

  using FF  = Feature::Flags;
  _features = FF::OUTPUT_LOGITS | FF::SAVE_RESTORE | FF::DYNAMIC_LOAD | FF::OUTPUT_EMBEDDINGS;

  __DEBUG("libtorch: init start");

  // All the LoRA params are captured and maintained by LoRA class
  if (json.contains("loraConfig")) {
    try {
      auto loraConfig = Config(json["loraConfig"], "loraConfig");
      lora_config     = std::make_shared<LoraConfig>(loraConfig, _env);
      lora_conf_type  = lora_config->getLoraConfigType();
    } catch (const std::runtime_error& e) {
      State::fatal(fmt::format("Error in parsing params - {}", e.what()));
      throw std::runtime_error(State::error());
    }
  }

  // Load the GenieTraining library
  void* libHandle = pal::dynamicloading::dlOpen(
      "libGenieTraining.so", pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
  if (nullptr == libHandle) {
    const char* msg = pal::dynamicloading::dlError();
    __ERROR("libtorch: Unable to load libGenieTraining.so. dlerror(): [{}]",
            msg ? msg : "Unknown error");
    throw std::runtime_error("Unable to open libGenieTraining library.");
  }

  // Load the create function
  auto createTrainingInterfaceFn = reinterpret_cast<genie::CreateGenieTrainingInterfaceFnType_t>(
      pal::dynamicloading::dlSym(libHandle, "createGenieTrainingInterface"));
  if (nullptr == createTrainingInterfaceFn) {
    __ERROR("libtorch: Unable to resolve createGenieTrainingInterface symbol");
    throw std::runtime_error("Unable to resolve createGenieTrainingInterface.");
  }

  // Load the destroy function
  m_destroyTrainingInterfaceFn = reinterpret_cast<genie::DestroyGenieTrainingInterfaceFnType_t>(
      pal::dynamicloading::dlSym(libHandle, "destroyGenieTrainingInterface"));
  if (nullptr == m_destroyTrainingInterfaceFn) {
    __ERROR("libtorch: Unable to resolve destroyGenieTrainingInterface symbol");
    throw std::runtime_error("Unable to resolve destroyGenieTrainingInterface.");
  }

  // Create the training interface
  m_trainingInterface = createTrainingInterfaceFn();
  if (nullptr == m_trainingInterface) {
    __ERROR("libtorch: Unable to create GenieTraining interface");
    throw std::runtime_error("Unable to create GenieTraining interface.");
  }
  m_trainingInterface->loadConfig(ctx, json);

  __DEBUG("libtorch: init complete : {} usec", start.elapsed_usec());
}

LibTorchEngine::~LibTorchEngine() {
  __DEBUG("libtorch: destroyed");
  if (m_trainingInterface && m_destroyTrainingInterfaceFn) {
    m_destroyTrainingInterfaceFn(m_trainingInterface);
  }
}

std::vector<float> LibTorchEngine::run_on_device_training(TrainingData& trainingData) {
  GENIE_TRACE();
  qualla::Timer start;

  __DEBUG("libtorch: run_on_device_training start");

  if (!m_trainingInterface) {
    __ERROR("libtorch: Training interface not initialized");
    return {};
  }

  auto result = m_trainingInterface->runOnDeviceTraining(trainingData);

  __DEBUG("libtorch: run_on_device_training complete : {} usec", start.elapsed_usec());

  return result;
}

bool LibTorchEngine::saveLoraAdapter(std::string lora_adapter_name) {
  GENIE_TRACE();
  qualla::Timer start;

  __DEBUG("libtorch: saveLoraAdapter start: {}", lora_adapter_name);

  if (!m_trainingInterface) {
    __ERROR("libtorch: Training interface not initialized");
    return false;
  }
  std::shared_ptr<LoraAdapter> curAdapter = nullptr;
  std::vector<std::string> adapter_bins;
  std::string metadata_dlc;
  if (lora_config) {
    curAdapter = lora_config->getAdapter(lora_adapter_name);
    if (!curAdapter) {
      __ERROR("libtorch: Could not find lora adapters config to save");
      return false;
    }
    adapter_bins = curAdapter->m_binList;
    metadata_dlc = curAdapter->m_metadataDlc;
  }
  bool result = m_trainingInterface->saveLoraAdapter(lora_adapter_name, adapter_bins, metadata_dlc);

  __DEBUG("libtorch: saveLoraAdapter complete : {} usec", start.elapsed_usec());

  return result;
}

size_t LibTorchEngine::process(const std::vector<int32_t>& /*tokens*/,
                               std::vector<float>& /*output*/,
                               bool /*output_all*/) {
  __ERROR("libtorch: process not implemented for training engine");
  return 0;
}

size_t LibTorchEngine::process(const std::vector<int32_t>& /*tokens*/,
                               Tensor& /*output*/,
                               bool /*output_all*/) {
  __ERROR("libtorch: process not implemented for training engine");
  return 0;
}

size_t LibTorchEngine::process(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
                               std::vector<uint8_t>& outputs) {
  GENIE_TRACE();
  qualla::Timer start;

  __DEBUG("libtorch: inference start");

  if (!m_trainingInterface) {
    __ERROR("libtorch: Training interface not initialized");
    return 0;
  }

  auto result = m_trainingInterface->process(inputs, outputs);

  __DEBUG("libtorch: inference complete : {} usec", start.elapsed_usec());

  return result;
}

bool LibTorchEngine::save_snapshot(const std::filesystem::path& path) {
  GENIE_TRACE();

  __DEBUG("libtorch: save_snapshot start: {}", path.string());

  if (!m_trainingInterface) {
    __ERROR("libtorch: Training interface not initialized");
    return false;
  }

  bool result = m_trainingInterface->saveSnapshot(path);

  __DEBUG("libtorch: save_snapshot complete");

  return result;
}

bool LibTorchEngine::load_snapshot(const std::filesystem::path& path) {
  GENIE_TRACE();

  __DEBUG("libtorch: load_snapshot start: {}", path.string());

  if (!m_trainingInterface) {
    __ERROR("libtorch: Training interface not initialized");
    return false;
  }

  bool result = m_trainingInterface->loadSnapshot(path);

  __DEBUG("libtorch: load_snapshot complete");

  return result;
}

bool LibTorchEngine::save_training_parameters(const std::filesystem::path& path) {
  GENIE_TRACE();

  __DEBUG("libtorch: save_training_parameters start: {}", path.string());

  if (!m_trainingInterface) {
    __ERROR("libtorch: Training interface not initialized");
    return false;
  }

  bool result = m_trainingInterface->saveTrainingParameters(path.string());

  __DEBUG("libtorch: save_training_parameters complete");

  return result;
}

void LibTorchEngine::getInputTensorNames(std::unordered_set<std::string>& inputTensorNames) {
  if (!m_trainingInterface) {
    __ERROR("libtorch: Training interface not initialized");
    return;
  }
  m_trainingInterface->getInputTensorNames(inputTensorNames);
}

void LibTorchEngine::getTensorParam(
    LayerType layerType, std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth) {
  if (!m_trainingInterface) {
    __ERROR("libtorch: Training interface not initialized");
    return;
  }
  m_trainingInterface->getTensorParam(layerType, dataType, scale, offset, bitWidth);
}

void LibTorchEngine::getTensorDimensions(LayerType layerType, std::vector<uint32_t>& dimensions) {
  if (!m_trainingInterface) {
    __ERROR("libtorch: Training interface not initialized");
    return;
  }
  m_trainingInterface->getTensorDimensions(layerType, dimensions);
}

}  // namespace qualla
