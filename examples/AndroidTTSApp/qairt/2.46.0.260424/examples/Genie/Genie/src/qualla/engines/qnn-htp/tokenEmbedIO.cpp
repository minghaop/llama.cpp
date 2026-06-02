//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "tokenEmbedIO.hpp"

namespace qualla {

size_t TokenEmbedIO::addData(void* data, size_t dataSize, nlohmann::json& dataConfig) {
  // Always perform data Config checks.
  if (dataConfig.empty() && !m_dataConfigInitialized)
    throw std::runtime_error("No data Config passed.");

  if (!dataConfig.empty()) {
    m_dataConfig            = std::make_unique<DataConfig>(dataConfig);
    m_dataConfigInitialized = true;
    __DEBUG("Data Config is updated, config:: {}", dataConfig.dump());
  } else
    __DEBUG("Using previous data Config");

  // Validate data type matches expected input type
  if (m_inputType == InputType::TOKENS) {
    if (m_dataConfig->m_dataType != DataType::UINT_32 &&
        m_dataConfig->m_dataType != DataType::INT32) {
      throw std::runtime_error("Token input requires INT32 or UINT32 data type");
    }
  }

  size_t expectedSize = m_dataConfig->m_dataLength * sizeof(uint32_t);
  if (dataSize != expectedSize) {
    throw std::runtime_error("Data size mismatch. Expected " + std::to_string(expectedSize) +
                             " bytes but got " + std::to_string(dataSize) + " bytes");
  }

  if (m_bufferType == ExternalBufferType::PERSISTENT) {
    // save the shared pointer
    m_data = data;
  } else {
    throw std::runtime_error("Unsupported Buffer Type is supplied.");
  }

  // Reset after every new data set calls
  m_startOffset       = 0;
  return m_dataLength = m_dataConfig->m_dataLength;
}

void TokenEmbedIO::populateTensor(const InferenceStep& curStep) {
  const size_t variant   = static_cast<size_t>(curStep.variant);
  const size_t n_process = static_cast<size_t>(curStep.n_process);
  if (m_startOffset + n_process > m_dataLength) {
    throw std::runtime_error("Encountered overflow data access");
  }
  // Fill buffers
  // This is not working for other types of AR-ContextLength graphs
  if (m_inputType == InputType::TOKENS) {
    if (!m_ioTensor) {
      throw std::runtime_error("IO Tensors are not existing");
    }
    uint32_t* input_id_buffer =
        reinterpret_cast<uint32_t*>(m_ioTensor->getBuffer(m_tensor->tensor));

    // I guess this is not needed if we gurantee to fill as many tokens as variant length ?
    if (n_process < variant)
      std::fill_n(input_id_buffer, variant, static_cast<uint32_t>(m_padToken));

    uint32_t* tokens = reinterpret_cast<uint32_t*>(m_data);
    std::memcpy(input_id_buffer, &tokens[m_startOffset], n_process * sizeof(uint32_t));
  } else {
    throw std::runtime_error("Unsupported Input Type");
  }
  m_startOffset += n_process;
}
}  // namespace qualla