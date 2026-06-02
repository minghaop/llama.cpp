//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <memory>
#include <vector>

#include "Exception.hpp"
#include "Pipeline.hpp"
#include "Unet.hpp"

using namespace genie;

pipeline::Unet::Unet(nlohmann::json config,
                     std::shared_ptr<ProfileStat> profileStat,
                     std::shared_ptr<genie::log::Logger> logger,
                     std::shared_ptr<qualla::Env> env)
    : Node(config) {
  for (auto item : m_config.items()) {
    nlohmann::json encoderDecoderConfig;
    encoderDecoderConfig["encoder-decoder"]         = item.value();
    encoderDecoderConfig["encoder-decoder"]["type"] = "unet";
    genie::EncoderDecoder::validateEncoderDecoderConfig(encoderDecoderConfig["encoder-decoder"]);
    m_encoderDecoder =
        std::make_shared<genie::EncoderDecoder>(encoderDecoderConfig, profileStat, logger, env);
    if (encoderDecoderConfig["encoder-decoder"].contains("engine")) {
      if (encoderDecoderConfig["encoder-decoder"]["engine"].contains("type")) {
        m_type = encoderDecoderConfig["encoder-decoder"]["engine"]["type"];
      }
      if (encoderDecoderConfig["encoder-decoder"]["engine"].contains("training-config")) {
        m_batchSize =
            encoderDecoderConfig["encoder-decoder"]["engine"]["training-config"]["batch-size"];
        m_maxIterations =
            encoderDecoderConfig["encoder-decoder"]["engine"]["training-config"]["max-iterations"];
      }
    }
  }
}

int32_t pipeline::Unet::execute(void* /*userData*/,
                                std::shared_ptr<ProfileStat> /*profileStat*/,
                                nlohmann::json /*executionConfig*/) {
  // TODO: Implement execute logic for encoder-decoder
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::Unet::setEmbeddingInputData(GenieNode_IOName_t nodeIOName,
                                              const void* embedding,
                                              size_t embeddingSize,
                                              std::shared_ptr<ProfileStat>) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_DIFFUSER_TEXT_EMBEDDING_INPUT &&
      nodeIOName != GenieNode_IOName_t::GENIE_NODE_DIFFUSER_IMAGE_EMBEDDING_OUTPUT &&
      nodeIOName != GenieNode_IOName_t::GENIE_NODE_DIFFUSER_NOISE_INPUT &&
      nodeIOName != GenieNode_IOName_t::GENIE_NODE_DIFFUSER_TIMESTEP_INPUT) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL,
                    "setEmbeddingInputData can only be set for "
                    "GENIE_NODE_DIFFUSER_TEXT_EMBEDDING_INPUT");
  }

  const std::unordered_map<GenieNode_IOName_t, std::pair<std::string, qualla::LayerType>> IOToInputMap {
    {GENIE_NODE_DIFFUSER_TEXT_EMBEDDING_INPUT, {"encoder_hidden_states", qualla::LayerType::INPUT_EMBED}},
    {GENIE_NODE_DIFFUSER_IMAGE_EMBEDDING_OUTPUT, {"image_embeddings", qualla::LayerType::INPUT}},
    {GENIE_NODE_DIFFUSER_NOISE_INPUT, {"noise_input", qualla::LayerType::INPUT}},
    {GENIE_NODE_DIFFUSER_TIMESTEP_INPUT, {"timestep", qualla::LayerType::TIMESTEP}}
  };
  if (m_pipeline && m_pipeline->getDataLoader()) {
    if (IOToInputMap.find(nodeIOName) != IOToInputMap.end()) {
      std::vector<uint32_t> dimensions;
      m_encoderDecoder->getDimensions(IOToInputMap.at(nodeIOName).second, dimensions);
      std::string inputDataType;
      double inputScale;
      int32_t inputOffset;
      size_t inputByteWidth;
      m_encoderDecoder->getQuantParam(IOToInputMap.at(nodeIOName).second,
                                      inputDataType,
                                      inputScale,
                                      inputOffset,
                                      inputByteWidth);

      m_pipeline->getDataLoader()->addTensorData(
          IOToInputMap.at(nodeIOName).first,  // tensor name for text embedding input
          reinterpret_cast<uint8_t*>(const_cast<void*>(embedding)),
          embeddingSize,
          inputDataType,
          inputScale,
          inputOffset,
          dimensions);
    }
  }
  return GENIE_STATUS_SUCCESS;
}

void pipeline::Unet::validateTrainingDataConsistency() {
  qualla::TrainingData& trainingData = m_pipeline->getDataLoader()->getTrainingData();

  // Validate required keys
  static const std::unordered_set<std::string> requiredInputs = {
      "image_embeddings", "encoder_hidden_states" };
  for (const auto& key : requiredInputs) {
    if (trainingData.find(key) == trainingData.end()) {
      throw Exception(GENIE_STATUS_ERROR_GENERAL,
                      fmt::format("Missing required training data: {}", key));
    }
  }

  // Validate training data dimensions
  auto validateTrainingDataShape = [&](const std::string& key){
    if (trainingData[key].size() != m_maxIterations) {
      throw Exception(GENIE_STATUS_ERROR_GENERAL,
                      fmt::format("Training data \'{}\' has an inconsistent number of iterations: "
                                  "expected: {} actual: {}",
                                  key, m_maxIterations, trainingData[key].size()));
    }
    for(size_t i = 0; i < m_maxIterations; i++) {
      if (trainingData[key][i].size() != m_batchSize) {
        throw Exception(GENIE_STATUS_ERROR_GENERAL,
                        fmt::format("Training data \'{}\' @ iteration={} has an inconsistent batch size: "
                                    "expected: {} actual: {}",
                                    key, i, m_batchSize, trainingData[key][i].size()));
      }
    }
  };

  for (const std::string& inputName : requiredInputs) {
    // User may set input tensor(s) just once. In such cases, duplicate input tensor(s) to shape
    const size_t originalNumItr = trainingData[inputName].size();
    if (originalNumItr > 0) {
      // Cyclically duplicate existing tensor batches to shape (m_batchSize)
      for (size_t i = 0; i < originalNumItr; i++) {
        const size_t originalBatchSize = trainingData[inputName][i].size();
        if (originalBatchSize > 0) {
          for (size_t batch = originalBatchSize; batch < m_batchSize; batch++) {
            trainingData[inputName][i].push_back(trainingData[inputName][i][batch % originalBatchSize]);
          }
        }
      }

      // Cyclically duplicate existing tensor iterations to shape (m_maxIterations)
      for (size_t i = originalNumItr; i < m_maxIterations; ++i) {
        trainingData[inputName].push_back(trainingData[inputName][i % originalNumItr]);
      }
    }

    validateTrainingDataShape(inputName);
  }

  if (trainingData.find("noise_input") != trainingData.end()) {
    validateTrainingDataShape("noise_input");
  }
  if (trainingData.find("timestep") != trainingData.end()) {
    validateTrainingDataShape("timestep");
  }
}

int32_t pipeline::Unet::train() {
  if (!m_pipeline || !m_pipeline->getDataLoader()) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "No DataLoader available for training");
  }

  validateTrainingDataConsistency();
  auto& trainingData = m_pipeline->getDataLoader()->getTrainingData();
  m_encoderDecoder->train(trainingData);
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::Unet::saveLora(std::string loraAdapterName,
                                 std::string engine,
                                 std::shared_ptr<ProfileStat> profileStat) {
  auto status = m_encoderDecoder->saveLora(loraAdapterName, engine, profileStat);
  return status;
}
