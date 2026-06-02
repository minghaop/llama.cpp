//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "IOBuffer.hpp"
#include "LoraConfig.hpp"
#include "env.hpp"

namespace qualla {

class EngineState {
 public:
  EngineState(std::shared_ptr<IOBuffer> iobuffer,
              std::shared_ptr<Env> env,
              std::shared_ptr<LoraConfig> loraConfig = nullptr);
  EngineState(std::shared_ptr<Env> env, std::shared_ptr<LoraConfig> loraConfig = nullptr);

  ~EngineState() {}

  // Delete copy and move constructor
  EngineState(const EngineState &)            = delete;
  EngineState &operator=(const EngineState &) = delete;
  EngineState(EngineState &&)                 = delete;
  EngineState &operator=(EngineState &&)      = delete;

  void initialize(std::shared_ptr<IOBuffer> ioBuffer);
  bool isInitialize();

  std::shared_ptr<IOBuffer> getIOBuffer();
  void setIOBuffer(std::shared_ptr<IOBuffer> ioBuffer);
  std::shared_ptr<Env> getEnv();
  std::shared_ptr<LoraConfig> getLoraConfig();

  bool changeIOEvent(IOEVENT event);
  bool update(std::shared_ptr<EngineState> engineState);

 private:
  std::shared_ptr<IOBuffer> m_ioBuffer;
  bool m_isInitialized{false};
  std::shared_ptr<Env> _env;  // Shared between multiple dialogs
  std::shared_ptr<LoraConfig> m_loraConfig;
};

}  // namespace qualla