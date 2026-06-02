//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "Exception.hpp"
#include "LmExecutor.hpp"
#include "Pipeline.hpp"

using namespace genie;
namespace fs = std::filesystem;

pipeline::LmExecutor::LmExecutor(nlohmann::json config,
                                 std::shared_ptr<ProfileStat> profileStat,
                                 std::shared_ptr<genie::log::Logger> logger,
                                 std::shared_ptr<qualla::Env> env)
    : Node(config) {
  for (auto item : m_config.items()) {
    Engine::validateStandaloneEngineConfig(item.value());
    // Create standaloneEngine Config
    nlohmann::json standloneEngineConfig;
    standloneEngineConfig["standalone-engine"] = item.value();
    m_executor = std::make_shared<genie::Engine>(standloneEngineConfig, profileStat, logger, env);
  }
  if (m_config["lm-executor"].contains("execution-strategy")) {
    m_executionStrategy = pipeline::Node::getExecutionStrategyFromString(
        m_config["lm-executor"]["execution-strategy"]);
  }
}

int32_t pipeline::LmExecutor::setInputBufferData(GenieNode_IOName_t nodeIOName,
                                                 const void* data,
                                                 size_t dataSize,
                                                 nlohmann::json dataConfig,
                                                 std::shared_ptr<ProfileStat>) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_LM_EXECUTOR_TOKEN_INPUT &&
      nodeIOName != GenieNode_IOName_t::GENIE_NODE_LM_EXECUTOR_EMBEDDING_INPUT) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL,
                    "setInputBufferData can only be set for GENIE_NODE_LM_EXECUTOR_TOKEN_INPUT and "
                    "GENIE_NODE_LM_EXECUTOR_EMBEDDING_INPUT");
  }
  if (!m_executor) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Executor not initialized");
  }

  qualla::LayerType inputLayer = getLayerTypeFromNodeIO(nodeIOName);
  // Api should not be involve in quantization/Requantization/deQuantization let engine own it.
  try {
    size_t inputsToProcess = m_executor->setInputBufferData(inputLayer, data, dataSize, dataConfig);
    if (m_inputsToProcess == 0) {
      m_inputsToProcess = inputsToProcess;
    } else if (m_inputsToProcess != inputsToProcess) {
      throw Exception(GENIE_STATUS_ERROR_GENERAL,
                      "All inputs must have the same length. Expected " +
                          std::to_string(m_inputsToProcess) + " but got " +
                          std::to_string(inputsToProcess));
    }
  } catch (...) {
    // Reset state on error to prevent inconsistent state
    m_inputsToProcess = 0;
    throw;
  }

  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::LmExecutor::getOutputBufferData(GenieNode_IOName_t nodeIOName,
                                                  nlohmann::json& outputConfig,
                                                  GenieNode_IOCallback_t callback,
                                                  const void* userData,
                                                  std::shared_ptr<ProfileStat>) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_LM_EXECUTOR_LOGIT_OUTPUT) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL,
                    "getOutputBufferData can only be used for GENIE_NODE_LM_EXECUTOR_LOGIT_OUTPUT");
  }
  if (!m_executor) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Executor not initialized");
  }
  qualla::LayerType outputLayer = getLayerTypeFromNodeIO(nodeIOName);
  // Api should not be involve in quantization/Requantization/deQuantization let engine own it.
  qualla::Engine::OutputCallback outputCallback =
      [&](void* data, size_t dataSize, nlohmann::json& config) -> bool {
    callback(data, dataSize, config.dump().c_str(), userData);
    return true;
  };
  if (!m_executor->getOutputBufferData(outputLayer, outputConfig, outputCallback)) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Failed to get the output for the IO.");
  }
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::LmExecutor::execute(void* /*userData*/,
                                      std::shared_ptr<ProfileStat>,
                                      nlohmann::json executionConfig) {
  if (!m_executor) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Executor not initialized");
  }
  if (m_executionStrategy != ExecutionStrategy::GENIE_EXECUTION_STRATEGY_EXPLICIT &&
      !isConnected()) {
    throw Exception(
        GENIE_STATUS_ERROR_GENERAL,
        "standalone node execution is not configured and node is not connected to a pipeline");
  }

  size_t inputsToProcess = m_inputsToProcess;
  m_inputsToProcess      = 0;  // reset the processing count

  if (!m_executor->process(executionConfig)) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "execution failed for current Input");
  }

  m_nPast += inputsToProcess;  // add processed and trigger updateKV
  if (!m_executor->updateKV(m_nPast)) {
    m_nPast -= inputsToProcess;
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "KV update failed for the current Input");
  }

  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::LmExecutor::save(const std::string& name) {
  fs::path save_path = name;
  if (!m_nPast) {
    return false;
  }

  if (!fs::exists(save_path) && !fs::create_directories(save_path)) {
    return false;
  }
  // save the state like dialog
  nlohmann::json j{{"nPast", m_nPast}};
  {
    fs::path p = save_path / "LmExecutor.json";
    std::ofstream f(p);
    f << j;
  }
  return m_executor->save(name);
}

int32_t pipeline::LmExecutor::restore(const std::string& name) {
  // restore the state like dialog
  // Restore using session name unless override is provided
  fs::path restore_path = name;

  // Try to restore the Dialog state (optional)
  // If this fails we reset everything and try to restore the engine.
  nlohmann::json j{};
  {
    fs::path p = restore_path / "LmExecutor.json";
    if (fs::exists(p)) {
      std::ifstream f(p);
      j = nlohmann::json::parse(f);
    }
  }
  m_nPast      = qualla::Config::optional<uint32_t>(j, "nPast", 0);
  size_t nPast = m_executor->restore(name);
  if (!nPast) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  if (m_nPast != nPast) {
  }
  m_nPast = std::min(m_nPast, nPast);
  if (!m_executor->updateKV(m_nPast)) {  // sync both the engine and executor on nPast.
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

void pipeline::LmExecutor::reset() {
  // reset the executor and state
  m_nPast           = 0;
  m_inputsToProcess = 0;
  m_executor->reset();
}
