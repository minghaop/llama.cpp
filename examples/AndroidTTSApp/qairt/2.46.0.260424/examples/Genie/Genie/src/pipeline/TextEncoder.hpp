//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#ifndef TEXT_ENCODER_HPP
#define TEXT_ENCODER_HPP
#include <memory>
#include <vector>

#include "GeniePipeline.h"
#include "Node.hpp"

namespace genie {
namespace pipeline {

class Pipeline;

class TextEncoder final : public Node {
 public:
  TextEncoder(nlohmann::json config,
              std::shared_ptr<ProfileStat> profileStat,
              std::shared_ptr<genie::log::Logger> logger = nullptr,
              std::shared_ptr<qualla::Env> env           = nullptr);

  int32_t execute(void* userData,
                  std::shared_ptr<ProfileStat> profileStat,
                  nlohmann::json executionConfig = {});
  int32_t setTextInputData(GenieNode_IOName_t nodeIOName,
                           const char* txt,
                           std::shared_ptr<ProfileStat>);
  int32_t setEmbeddingOutputCallback(GenieNode_IOName_t nodeIOName,
                                     GenieNode_EmbeddingOutputCallback_t callback);

  int32_t applyLora(std::string loraAdapterName,
                    std::string engine,
                    std::shared_ptr<ProfileStat> profileStat);
  int32_t applyLoraStrength(std::string tensorName, std::string engine, float alpha);

 private:
  std::shared_ptr<Embedding> m_encoder;
  std::vector<uint8_t> m_data;
  GenieNode_EmbeddingOutputCallback_t m_embeddingOutputCallback = nullptr;
};

}  // namespace pipeline
}  // namespace genie
#endif  // TEXT_ENCODER_HPP