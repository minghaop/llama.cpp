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
#include "TextEncoder.hpp"

using namespace genie;

pipeline::TextEncoder::TextEncoder(nlohmann::json config,
                                   std::shared_ptr<ProfileStat> /*profileStat*/,
                                   std::shared_ptr<genie::log::Logger> logger,
                                   std::shared_ptr<qualla::Env> env)
    : Node(config) {
  for (auto item : m_config.items()) {
    Embedding::validateEmbeddingConfig(item.value(), false);
    nlohmann::json embeddingConfig;
    embeddingConfig["embedding"] = item.value();
    m_encoder = std::make_shared<genie::Embedding>(embeddingConfig, nullptr, logger, env);
    if (embeddingConfig["embedding"].contains("engine")) {
      if (embeddingConfig["embedding"]["engine"].contains("type")) {
        m_type = embeddingConfig["embedding"]["engine"]["type"];
      }
    }
  }
  m_embeddingOutputCallback = nullptr;
}

int32_t pipeline::TextEncoder::setEmbeddingOutputCallback(
    GenieNode_IOName_t nodeIOName, GenieNode_EmbeddingOutputCallback_t callback) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT) {
    throw Exception(
        GENIE_STATUS_ERROR_GENERAL,
        "setEmbeddingOutputCallback can only be set for GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT");
  }
  m_embeddingOutputCallback = callback;
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextEncoder::setTextInputData(GenieNode_IOName_t nodeIOName,
                                                const char* txt,
                                                std::shared_ptr<ProfileStat>) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_TEXT_ENCODER_TEXT_INPUT) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL,
                    "setTextInputData can only be set for GENIE_NODE_TEXT_ENCODER_TEXT_INPUT");
  }
  std::vector<int32_t> tokenizedResult;
  m_encoder->encode(txt, m_data, nullptr);
  std::vector<uint32_t> dimensions;
  m_encoder->getOutputDimensions(dimensions);
  // If connected to pipeline with DataLoader, store data
  if (m_pipeline && m_pipeline->getDataLoader()) {
    std::string outputDataType;
    double outputScale;
    int32_t outputOffset;
    float outputByteWidth;
    m_encoder->getOutputQuantParam(outputDataType, outputScale, outputOffset, outputByteWidth);

    m_pipeline->getDataLoader()->addTensorData(
        "encoder_hidden_states",  // tensor name for text encoder output
        m_data.data(),
        m_data.size(),
        outputDataType,
        outputScale,
        outputOffset,
        dimensions);
  } else if (isConnected()) {
    std::string outputDataType = "QNN_DATATYPE_FLOAT_32";
    double outputScale         = 1.0;
    int32_t outputOffset       = 0;
    float outputByteWidth      = 4;
    m_encoder->getOutputQuantParam(outputDataType, outputScale, outputOffset, outputByteWidth);
    size_t numElements = m_data.size() / outputByteWidth;

    uint32_t embeddingTokenNum = dimensions.empty() ? 0 : dimensions.at(0);
    m_pipeline->m_accumulator->append(
        m_data.data(), outputDataType, outputScale, outputOffset, numElements, embeddingTokenNum);
  }
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextEncoder::execute(void* userData,
                                       std::shared_ptr<ProfileStat>,
                                       nlohmann::json /*executionConfig*/ /*profileStat*/) {
  std::vector<uint32_t> dimensions;
  m_encoder->getOutputDimensions(dimensions);

  if (m_embeddingOutputCallback) {  // invoke userCallback if set
    m_embeddingOutputCallback(dimensions.data(),
                              dimensions.size(),
                              m_data.size(),
                              reinterpret_cast<void*>(m_data.data()),
                              userData);
  }
  m_data.clear();  // clear encoder buffer after callback invoked
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextEncoder::applyLora(std::string loraAdapterName,
                                         std::string engine,
                                         std::shared_ptr<ProfileStat> /*profileStat*/) {
  return m_encoder->applyLora(loraAdapterName, engine, nullptr);
}

int32_t pipeline::TextEncoder::applyLoraStrength(std::string tensorName,
                                                 std::string engine,
                                                 float alpha) {
  return m_encoder->applyLoraStrength(tensorName, engine, alpha);
}
