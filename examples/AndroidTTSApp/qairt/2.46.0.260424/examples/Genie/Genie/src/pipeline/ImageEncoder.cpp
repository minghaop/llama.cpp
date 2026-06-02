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
#include "ImageEncoder.hpp"
#include "Pipeline.hpp"

using namespace genie;

pipeline::ImageEncoder::ImageEncoder(nlohmann::json config,
                                     std::shared_ptr<ProfileStat> profileStat,
                                     std::shared_ptr<genie::log::Logger> logger,
                                     std::shared_ptr<qualla::Env> env)
    : Node(config) {
  for (auto item : m_config.items()) {
    nlohmann::json embeddingConfig;
    embeddingConfig["embedding"]         = item.value();
    embeddingConfig["embedding"]["type"] = "image-encoder";
    if (embeddingConfig["embedding"].contains("engine")) {
      nlohmann::json& embeddingEngineConfig = embeddingConfig["embedding"]["engine"];
      // Set defaults for Embedding QnnHtp backend
      if (embeddingEngineConfig.contains("backend")) {
        if (embeddingEngineConfig["backend"].contains("QnnHtp")) {
          embeddingEngineConfig["backend"]["QnnHtp"]["pooled-output"]    = false;
          embeddingEngineConfig["backend"]["QnnHtp"]["disable-kv-cache"] = true;
        }
      }
      if (embeddingEngineConfig.contains("type")) {
        m_type = embeddingEngineConfig["type"];
      }
    }
    Embedding::validateEmbeddingConfig(embeddingConfig["embedding"], false);
    m_encoder = std::make_shared<genie::Embedding>(embeddingConfig, profileStat, logger, env);

    if (embeddingConfig["embedding"].contains("engine") &&
        embeddingConfig["embedding"]["engine"].contains("model") &&
        embeddingConfig["embedding"]["engine"]["model"].contains("vision-param")) {
      m_height = embeddingConfig["embedding"]["engine"]["model"]["vision-param"]["height"];
      m_width  = embeddingConfig["embedding"]["engine"]["model"]["vision-param"]["width"];
    }
  }

  const std::unordered_map<std::string, GenieNode_IOName_t> inputNameToIOName{
      {"pixel_values", GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT},
      {"position_ids_sin", GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN},
      {"position_ids_cos", GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS},
      {"full_attention_mask", GENIE_NODE_IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK},
      {"window_attention_mask", GENIE_NODE_IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK},
      {"pretile_embedding", GENIE_NODE_IMAGE_ENCODER_PRETILE_EMBEDDING_INPUT},     // LLama3.2-11B
      {"posttile_embedding", GENIE_NODE_IMAGE_ENCODER_POSTTILE_EMBEDDING_INPUT},   // LLama3.2-11B
      {"gated_pos_embedding", GENIE_NODE_IMAGE_ENCODER_GATED_POS_EMBEDDING_INPUT}  // Llama3.2-11B
  };

  std::unordered_set<std::string> inputNames;
  m_encoder->getInputNames(inputNames);
  for (auto& name : inputNames) {
    if (inputNameToIOName.find(name) == inputNameToIOName.end()) {
      throw Exception(GENIE_STATUS_ERROR_GENERAL,
                      "ImageEncoder meet unsupported input layer of model");
    }

    m_inputIOMap[inputNameToIOName.at(name)] = name;
  }
}

int32_t pipeline::ImageEncoder::setEmbeddingOutputCallback(
    GenieNode_IOName_t nodeIOName, GenieNode_EmbeddingOutputCallback_t callback) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT) {
    throw Exception(
        GENIE_STATUS_ERROR_GENERAL,
        "setEmbeddingOutputCallback can only be set for GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT");
  }
  m_embeddingOutputCallback = callback;
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::ImageEncoder::setImageInputData(GenieNode_IOName_t nodeIOName,
                                                  const void* imageData,
                                                  size_t imageSize,
                                                  std::shared_ptr<ProfileStat> /*profileStat*/) {
  if (m_inputIOMap.find(nodeIOName) == m_inputIOMap.end()) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Unsupported IOName in setImageInputData");
  }
  if (imageData == nullptr) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "setImageInputData get nullptr imageData");
  }
  if (imageSize == 0) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "setImageInputData get imageSize 0");
  }

  const uint8_t* dataPtr = static_cast<const uint8_t*>(imageData);
  std::string name       = m_inputIOMap[nodeIOName];
  m_input[name]          = std::vector<uint8_t>(dataPtr, dataPtr + imageSize);

  int32_t status = GENIE_STATUS_SUCCESS;
  if (m_input.size() == m_inputIOMap.size()) {
    status = m_encoder->encode(m_input, m_data, nullptr);
    m_input.clear();  // clear input buffer after encode
    if (status != GENIE_STATUS_SUCCESS) {
      throw Exception(status, "ImageEncoder::setImageInputData failed");
    }

    // If connected to pipeline with DataLoader, store data
    std::vector<uint32_t> dimensions;
    m_encoder->getOutputDimensions(dimensions);
    if (m_pipeline && m_pipeline->getDataLoader()) {
      std::string outputDataType;
      double outputScale;
      int32_t outputOffset;
      float outputByteWidth;
      m_encoder->getOutputQuantParam(outputDataType, outputScale, outputOffset, outputByteWidth);

      m_pipeline->getDataLoader()->addTensorData(
          "image_embeddings",  // tensor name for image encoder output
          m_data.data(),
          m_data.size(),
          outputDataType,
          outputScale,
          outputOffset,
          dimensions);
    } else if (isConnected()) {
      std::string outputDataType;
      double outputScale;
      int32_t outputOffset;
      float outputByteWidth;
      m_encoder->getOutputQuantParam(outputDataType, outputScale, outputOffset, outputByteWidth);
      size_t numElements = m_data.size() / outputByteWidth;
      if (m_height > 0 && m_width > 0) {
        uint32_t visionPos = m_pipeline->m_accumulator->getTokenNum();
        m_pipeline->m_accumulator->setVisionParam(visionPos, 1, m_height, m_width);
      }
      uint32_t visionEmbedingSize = dimensions.empty() ? 0 : dimensions.back();
      if (visionEmbedingSize == 0) {
        throw Exception(GENIE_STATUS_ERROR_GENERAL, "setImageInputData get embedding size 0");
      }
      uint32_t embeddingTokenNum = numElements / visionEmbedingSize;
      m_pipeline->m_accumulator->append(
          m_data.data(), outputDataType, outputScale, outputOffset, numElements, embeddingTokenNum);
    }
  }
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::ImageEncoder::execute(void* userData,
                                        std::shared_ptr<ProfileStat>,
                                        nlohmann::json /*executionConfig*/) {
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

int32_t pipeline::ImageEncoder::applyLora(std::string loraAdapterName,
                                          std::string engine,
                                          std::shared_ptr<ProfileStat> profileStat) {
  return m_encoder->applyLora(loraAdapterName, engine, profileStat);
}

int32_t pipeline::ImageEncoder::applyLoraStrength(std::string tensorName,
                                                  std::string engine,
                                                  float alpha) {
  return m_encoder->applyLoraStrength(tensorName, engine, alpha);
}

void pipeline::ImageEncoder::registerModelAdaptor(std::shared_ptr<qualla::ModelIOAdaptor> adaptor) {
  m_encoder->registerModelAdaptor(adaptor);
}
