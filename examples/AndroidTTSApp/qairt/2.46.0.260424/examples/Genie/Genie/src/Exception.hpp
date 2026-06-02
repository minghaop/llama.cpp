//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <exception>
#include <stdexcept>
#include <string>

#include "GenieCommon.h"

namespace genie {

class Exception : public std::runtime_error {
 public:
  Exception(Genie_Status_t status, std::string what) : std::runtime_error(what), m_status(status) {}

  Genie_Status_t status() const { return m_status; }

 private:
  Genie_Status_t m_status;
};

class ContextLimitException : public Exception {
 public:
  ContextLimitException(std::string what) : Exception(GENIE_STATUS_WARNING_CONTEXT_EXCEEDED, what) {}

};

}  // namespace genie
