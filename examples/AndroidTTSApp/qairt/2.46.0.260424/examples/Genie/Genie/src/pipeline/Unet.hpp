//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#ifndef UNET_HPP
#define UNET_HPP
#include <memory>
#include <vector>

#include "GenieNode.h"
#include "GeniePipeline.h"
#include "Node.hpp"

namespace genie {
namespace pipeline {

class Pipeline;

class Unet final : public Node {
 public:
  Unet(nlohmann::json config,
       std::shared_ptr<ProfileStat> profileStat,
       std::shared_ptr<genie::log::Logger> logger = nullptr,
       std::shared_ptr<qualla::Env> env           = nullptr);

  int32_t setEmbeddingInputData(GenieNode_IOName_t nodeIOName,
                                const void* embedding,
                                size_t embeddingSize,
                                std::shared_ptr<ProfileStat> profileStat) override;
  int32_t execute(void* userData,
                  std::shared_ptr<ProfileStat> /*profileStat*/,
                  nlohmann::json executionConfig = {}) override;
  int32_t train() override;

  int32_t saveLora(std::string loraAdapterName,
                   std::string engine,
                   std::shared_ptr<ProfileStat> profileStat) override;

 private:
  void validateTrainingDataConsistency();
  std::shared_ptr<genie::EncoderDecoder> m_encoderDecoder;
  std::unordered_map<GenieNode_IOName_t, std::string> m_inputIOMap;
  std::vector<uint8_t> m_data;
};

}  // namespace pipeline
}  // namespace genie
#endif  // UNET_HPP
