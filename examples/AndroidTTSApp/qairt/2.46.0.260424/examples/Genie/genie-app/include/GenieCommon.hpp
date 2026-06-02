//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef GENIE_COMMON_HPP
#define GENIE_COMMON_HPP

/**
 * @file
 * @brief Defines the common classes and functionality of the Genie C++ API.
 */

#include <stdexcept>
#include <string>

#include "GenieCommon.h"

namespace genie {
/**
 * @brief Retrieves the major version of the Genie library.
 * @return The Genie library major version.
 */
inline uint32_t getMajorVersion() { return Genie_getApiMajorVersion(); }

/**
 * @brief Retrieves the minor version of the Genie library.
 * @return The Genie library minor version.
 */
inline uint32_t getMinorVersion() { return Genie_getApiMinorVersion(); }

/**
 * @brief Retrieves the patch version of the Genie library.
 * @return The Genie library patch version.
 */
inline uint32_t getPatchVersion() { return Genie_getApiPatchVersion(); }

/**
 * @brief Retrieves full version of the Genie library.
 * @return The Genie library version represented as a std::string.
 */
inline std::string getVersion() {
  return std::to_string(getMajorVersion()) + "." + std::to_string(getMinorVersion()) + "." +
         std::to_string(getPatchVersion());
}

/**
 * @class Exception
 * @brief Exception is a custom exception class which extends std::runtime_error.
 */
class Exception final : public std::runtime_error {
 public:
  /**
   * @brief Exception class constructor which includes file name and line number where the
   *        exception was thrown which allows for a more detailed exception messages.
   *
   * @param[in] message A message describing the error condition.
   * @param[in] filename The name of the file where the exception was thrown.
   * @param[in] line The line number where the exception was thrown.
   */
  explicit Exception(std::string message, std::string filename, int32_t line)
      : std::runtime_error(filename + "[" + std::to_string(line) + "]: " + message + ".") {}

  /**
   * @brief Exception class constructor which includes the Genie C API status, file name and line
   *        number where the exception was thrown which allows for a more detailed exception
   *        messages.
   *
   * @param[in] status The Genie C API status code associated with the exception.
   * @param[in] message A message describing the error condition.
   * @param[in] filename The name of the file where the exception was thrown.
   * @param[in] line The line number where the exception was thrown.
   */
  explicit Exception(Genie_Status_t status, std::string message, std::string filename, int32_t line)
      : std::runtime_error(filename + "[" + std::to_string(line) + "]: " + message + " (" +
                           std::to_string(status) + ").") {}
};

}  // namespace genie

#endif  // GENIE_COMMON_HPP
