//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include "qualla/engineState.hpp"

namespace qualla {

EngineState::EngineState(std::shared_ptr<IOBuffer> iobuffer,
                         std::shared_ptr<Env> env,
                         std::shared_ptr<LoraConfig> loraConfig)
    : m_ioBuffer(iobuffer), _env(env), m_loraConfig(loraConfig) {
  m_isInitialized = true;
}

EngineState::EngineState(std::shared_ptr<Env> env, std::shared_ptr<LoraConfig> loraConfig)
    : _env(env), m_loraConfig(loraConfig) {
  m_isInitialized = false;
}

void EngineState::initialize(std::shared_ptr<IOBuffer> ioBuffer) {
  m_ioBuffer      = ioBuffer;
  m_isInitialized = true;
}

std::shared_ptr<IOBuffer> EngineState::getIOBuffer() { return m_ioBuffer; }

void EngineState::setIOBuffer(std::shared_ptr<IOBuffer> ioBuffer) { m_ioBuffer = ioBuffer; }

std::shared_ptr<Env> EngineState::getEnv() { return _env; }

std::shared_ptr<LoraConfig> EngineState::getLoraConfig() { return m_loraConfig; }

bool EngineState::isInitialize() { return m_isInitialized; }

bool EngineState::changeIOEvent(IOEVENT event) {
  if (!m_isInitialized) {
    return false;
  }
  return m_ioBuffer->setEvent(event);
}

bool EngineState::update(std::shared_ptr<EngineState> engineState) {
  m_isInitialized = engineState->isInitialize();
  m_ioBuffer      = engineState->getIOBuffer();
  m_loraConfig    = engineState->getLoraConfig();
  if (true != _env->update(engineState->getEnv())) {
    __ERROR("Failed to update the logger environment");
    return false;
  }
  return true;
}

}  // namespace qualla