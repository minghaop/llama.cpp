//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#ifndef LM_EXECUTOR_HPP
#define LM_EXECUTOR_HPP
#include <memory>
#include <vector>

#include "GeniePipeline.h"
#include "Node.hpp"

namespace genie {
namespace pipeline {

class Pipeline;

class LmExecutor final : public Node {
 public:
  LmExecutor(nlohmann::json config,
             std::shared_ptr<ProfileStat> profileStat,
             std::shared_ptr<genie::log::Logger> logger = nullptr,
             std::shared_ptr<qualla::Env> env           = nullptr);

  // set input data from the node
  int32_t setInputBufferData(GenieNode_IOName_t nodeIOName,
                             const void* data,
                             size_t dataSize,
                             nlohmann::json dataConfig,
                             std::shared_ptr<ProfileStat> profileStat);

  // get output data from the node
  int32_t getOutputBufferData(GenieNode_IOName_t nodeIOName,
                              nlohmann::json& ioConfig,
                              GenieNode_IOCallback_t callback,
                              const void* userData,
                              std::shared_ptr<ProfileStat>);

  int32_t execute(void* userData,
                  std::shared_ptr<ProfileStat> profileStat,
                  nlohmann::json executionConfig = {});
  int32_t save(const std::string&);
  int32_t restore(const std::string&);
  void reset();

 private:
  std::shared_ptr<Engine> m_executor;
  // state management variables.
  std::size_t m_nPast{0};
  std::size_t m_inputsToProcess{0};
};

}  // namespace pipeline
}  // namespace genie
#endif  // LM_EXECUTOR_HPP
