//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#ifndef TEXT_GENERATOR_HPP
#define TEXT_GENERATOR_HPP
#include <memory>
#include <vector>

#include "Dialog.hpp"
#include "GeniePipeline.h"
#include "Node.hpp"

namespace genie {
namespace pipeline {

class Pipeline;

class TextGenerator final : public Node {
 public:
  TextGenerator(nlohmann::json config,
                std::shared_ptr<ProfileStat> profileStat,
                std::shared_ptr<genie::log::Logger> logger = nullptr,
                std::shared_ptr<qualla::Env> env           = nullptr);

  int32_t bindPipeline(Pipeline& pipeline);
  int32_t execute(void* userData,
                  std::shared_ptr<ProfileStat> profileStat,
                  nlohmann::json executionConfig = {});
  int32_t train();
  int32_t save(const std::string&);
  int32_t restore(const std::string&);
  void reset();
  int32_t setPriority(std::string engine, GeniePipeline_Priority_t priority);
  int32_t setOemkey(const std::string& oemKey);

  // Set input data for the node
  int32_t setTextInputData(GenieNode_IOName_t nodeIOName,
                           const char* txt,
                           std::shared_ptr<ProfileStat> profileStat);
  int32_t setEmbeddingInputData(GenieNode_IOName_t nodeIOName,
                                const void* embedding,
                                size_t embeddingSize,
                                std::shared_ptr<ProfileStat> profileStat);

  int32_t setTextOutputCallback(GenieNode_IOName_t nodeIOName,
                                GenieNode_TextOutput_Callback_t callback);

  int32_t applyLora(std::string loraAdapterName,
                    std::string engine,
                    std::shared_ptr<ProfileStat> profileStat);
  int32_t applyLoraStrength(std::string tensorName, std::string engine, float alpha);

  GenieEngine_Handle_t getEngineHandle(const std::string& engineRole,
                                       std::shared_ptr<ProfileStat> profileStat);
  int32_t bindEngine(const std::string& engineRole,
                     std::shared_ptr<Engine> engine,
                     std::shared_ptr<ProfileStat> profileStat);
  GenieSampler_Handle_t getSamplerHandle();
  GenieTokenizer_Handle_t getTokenizerHandle();

  void registerModelAdaptor(std::shared_ptr<qualla::ModelIOAdaptor> adaptor);

  std::shared_ptr<Dialog> m_generator;

 private:
  void validateTrainingDataConsistency();

 private:
  std::string m_queryString = "";
  size_t m_accumulatorSize  = 0;
  GenieNode_TextOutput_Callback_t m_textOutputCallback{nullptr};
  bool m_usingMRope{false};
};

}  // namespace pipeline
}  // namespace genie
#endif  // TEXT_GENERATOR_HPP
