//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef GENIE_DLC_HPP
#define GENIE_DLC_HPP

/**
 * @file
 * @brief Defines the Dlc and Dlc::Config C++ classes which expose all functionality of the
 *        GenieDlc.h C API.
 */

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "GenieCommon.hpp"
#include "GenieDlc.h"

namespace genie {
/**
 * @class Dlc
 * @brief The Dlc C++ class represents the GenieDlc_Handle_t and it's related functionality.
 */
class Dlc final {
 public:
  /**
   * @class Dlc::Config
   * @brief Represents configuration for the `Dlc` class
   */
  class Config final {
   public:
    /**
     * @brief Constructs a Config object from a DLC file path.
     * @param[in] dlcPath The path of a DLC file.
     * @throws Exception if the DLC file cannot be loaded or is invalid.
     */
    explicit Config(std::string dlcPath) : m_handle(nullptr) {
      const Genie_Status_t status = GenieDlcConfig_create(dlcPath.c_str(), nullptr, &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw Exception(status, "Failed to create the DLC config from file", __FILE__, __LINE__);
      }
    }

    /**
     * @brief Deleted copy constructor to prevent copying of Config objects
     */
    Config(const Config&) = delete;

    /**
     * @brief Deleted copy assignment operator to prevent copying of Config objects.
     */
    Config& operator=(const Config&) = delete;

    /**
     * @brief Move constructor for Config objects.
     * @note Calls the move assignment operator.
     * @param[in/out] other The source Config object to move from.
     */
    Config(Config&& other) noexcept : m_handle(nullptr) { *this = std::move(other); }

    /**
     * @brief Move assignment operator for Config objects.
     * @param[in/out] other The source Config object to move from.
     */
    Config& operator=(Config&& other) {
      std::swap(m_handle, other.m_handle);
      return *this;
    }

    /**
     * @brief Destructor for the Config class.
     */
    ~Config() {
      const Genie_Status_t status = GenieDlcConfig_free(m_handle);
      if (GENIE_STATUS_SUCCESS != status) {
        std::cerr << "Failed to free the DLC config." << std::endl;
      }
    }

    /**
     * @brief Returns the handle associated with this object
     */
    const GenieDlcConfig_Handle_t operator()() const { return m_handle; }

   private:
    GenieDlcConfig_Handle_t m_handle = nullptr;
  };

  /**
   * @brief Constructs a `Dlc` instance with the given configuration.
   * @param[in] config An lvalue or rvalue to the Config configuration object.
   * @throws Exception If there is failure in creating the Dlc.
   */
  template <typename T>
  explicit Dlc(T&& config) {
    const Genie_Status_t status = GenieDlc_create(std::forward<T>(config)(), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw Exception(status, "Failed to create the DLC", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Deleted copy constructor to prevent copying of Dlc objects
   */
  Dlc(const Dlc&) = delete;

  /**
   * @brief Deleted copy assignment operator to prevent copying of Dlc objects.
   */
  Dlc& operator=(const Dlc&) = delete;

  /**
   * @brief Move constructor for Dlc objects.
   * @note Calls the move assignment operator.
   * @param[in/out] other The source Dlc object to move from.
   */
  Dlc(Dlc&& other) noexcept : m_handle(nullptr) { *this = std::move(other); }

  /**
   * @brief Move assignment operator for Dlc objects.
   * @param[in/out] other The source Dlc object to move from.
   */
  Dlc& operator=(Dlc&& other) {
    std::swap(m_handle, other.m_handle);
    return *this;
  }

  /**
   * @brief Destructor for the Dlc class.
   */
  ~Dlc() {
    const Genie_Status_t status = GenieDlc_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the DLC." << std::endl;
    }
  }

  /**
   * @brief Gets all supported use cases from the DLC.
   * @return A string containing the JSON data with use cases information.
   * @throws Exception if use cases retrieval fails.
   */
  inline std::string getUseCases() {
    const Genie_AllocCallback_t callback = [](size_t size, const char** data) {
      *data = new char[size];
    };

    const char* useCasesPtr     = nullptr;
    const Genie_Status_t status = GenieDlc_getUseCases(m_handle, callback, &useCasesPtr);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to get use cases", __FILE__, __LINE__);
    }
    if (!useCasesPtr) {
      throw Exception("Unexpected null Use cases JSON pointer", __FILE__, __LINE__);
    }

    std::string useCasesStr(useCasesPtr);
    delete[] useCasesPtr;
    return useCasesStr;
  }

  /**
   * @brief Returns the handle associated with this object
   */
  const GenieDlc_Handle_t operator()() const { return m_handle; }

 private:
  GenieDlc_Handle_t m_handle = nullptr;
};

}  // namespace genie

#endif  // GENIE_DLC_HPP