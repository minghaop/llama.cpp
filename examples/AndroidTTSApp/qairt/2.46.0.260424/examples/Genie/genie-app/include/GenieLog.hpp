//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef GENIE_LOG_HPP
#define GENIE_LOG_HPP

/**
 * @file
 * @brief Defines the Log C++ class which exposes all functionality of the GenieLog.h C API.
 */

#include <iostream>
#include <utility>

#include "GenieCommon.hpp"
#include "GenieLog.h"

namespace genie {
/**
 * @class Log
 * @brief The Log C++ class represents the GenieLog_Handle_t and it's related functionality.
 */
class Log final {
 public:
  /**
   * @brief Constructs a `Log` instance with the given configuration.
   * @param[in] callback Callback function which is called when new log messages are generated.
   *                     Can be NULL which indicates that the default system logger will be used.
   * @param[in] logLevel Maximum level of messages which will be generated.
   * @throws Exception If there is failure in creating the Log.
   */
  explicit Log(const GenieLog_Level_t logLevel, const GenieLog_Callback_t callback = nullptr) {
    const Genie_Status_t status = GenieLog_create(nullptr, callback, logLevel, &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw Exception(status, "Failed to create the log handle", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Deleted copy constructor to prevent copying of Log objects
   */
  Log(const Log&) = delete;

  /**
   * @brief Deleted copy assignment operator to prevent copying of Log objects.
   */
  Log& operator=(const Log&) = delete;

  /**
   * @brief Move constructor for Log objects.
   * @note Calls the move assignment operator.
   * @param[in/out] other The source Log object to move from.
   */
  Log(Log&& other) noexcept : m_handle(nullptr) { *this = std::move(other); }

  /**
   * @brief Move assignment operator for Log objects.
   * @param[in/out] other The source Log object to move from.
   */
  Log& operator=(Log&& other) {
    std::swap(m_handle, other.m_handle);
    return *this;
  }

  /**
   * @brief Returns the handle associated with this object
   */
  const GenieLog_Handle_t operator()() const { return m_handle; }

  /**
   * @brief Destructor for the Log class.
   */
  ~Log() {
    const Genie_Status_t status = GenieLog_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the log handle." << std::endl;
    }
  }

 private:
  GenieLog_Handle_t m_handle = nullptr;
};

}  // namespace genie

#endif  // GENIE_LOG_HPP
