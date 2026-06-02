//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <vector>

#include "buffer/IOTensor.hpp"
#include "kvmanager.hpp"
#include "qualla/detail/htp-io.hpp"

namespace qualla {

class TokenEmbedIO : public HtpIO {
 public:
  TokenEmbedIO(uint32_t padToken,
               InputType inputType,
               QnnUtils::Tensor* tensor,
               ExternalBufferType bufferType,
               DataFillPolicy fillPolicy,
               std::shared_ptr<IOTensor> ioTensor,
               std::shared_ptr<Env> env)
      : HtpIO(bufferType, fillPolicy),
        m_dataLength(0),
        _env(env),
        m_tensor(tensor),
        m_dataConfigInitialized(false),
        m_inputType(inputType),
        m_data(nullptr),
        m_startOffset(0),
        m_padToken(padToken),
        m_ioTensor(ioTensor) {}

  ~TokenEmbedIO() {}
  size_t addData(void* data, size_t dataSize, nlohmann::json& dataConfig);

  void populateTensor(const InferenceStep& curStep);

  size_t m_dataLength;

 protected:
  std::shared_ptr<Env> _env;

 private:
  QnnUtils::Tensor* m_tensor;
  std::unique_ptr<DataConfig> m_dataConfig;
  bool m_dataConfigInitialized{false};
  InputType m_inputType{InputType::UNKNOWN};
  void* m_data;
  uint32_t m_startOffset{0};
  uint32_t m_padToken{0};
  std::shared_ptr<IOTensor> m_ioTensor;
};
}  // namespace qualla
