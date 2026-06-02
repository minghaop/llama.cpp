//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef GENIE_PROFILE_HPP
#define GENIE_PROFILE_HPP

/**
 * @file
 * @brief Defines the Profile and Profile::Config C++ classes which expose all functionality of the
 *        GenieProfile.h C API.
 */

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "GenieCommon.hpp"
#include "GenieProfile.h"

namespace genie {
/**
 * @class Profile
 * @brief The Profile C++ class represents the GenieProfile_Handle_t and it's related functionality.
 */
class Profile final {
 public:
  /**
   * @class Profile::Config
   * @brief Represents configuration for the `Profile` class
   */
  class Config final {
   public:
    /**
     * @brief Constructs an empty Config object.
     */
    explicit Config() : m_handle(nullptr) {}

    /**
     * @brief Constructs a Config object.
     * @param[in] json A string containing the JSON configuration.
     * @throws Exception if the JSON string does not match the schema.
     */
    explicit Config(std::string json) : m_handle(nullptr) {
      const Genie_Status_t status = GenieProfileConfig_createFromJson(json.c_str(), &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw Exception(status, "Failed to create the profile config", __FILE__, __LINE__);
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
      if (m_handle) {
        const Genie_Status_t status = GenieProfileConfig_free(m_handle);
        if (GENIE_STATUS_SUCCESS != status) {
          std::cerr << "Failed to free the profile config." << std::endl;
        }
      }
    }

    /**
     * @brief Returns the handle associated with this object
     */
    const GenieProfileConfig_Handle_t operator()() const { return m_handle; }

   private:
    GenieProfileConfig_Handle_t m_handle = nullptr;
  };

  /**
   * @brief Constructs a `Profile` instance with the given configuration.
   * @param[in] config An lvalue or rvalue to the Config configuration object.
   * @throws Exception If there is failure in creating the Profile.
   */
  template <typename T>
  explicit Profile(T&& config) {
    const Genie_Status_t status = GenieProfile_create(std::forward<T>(config)(), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw Exception(status, "Failed to create the profile handle", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Deleted copy constructor to prevent copying of Profile objects
   */
  Profile(const Profile&) = delete;

  /**
   * @brief Deleted copy assignment operator to prevent copying of Profile objects.
   */
  Profile& operator=(const Profile&) = delete;

  /**
   * @brief Move constructor for Profile objects.
   * @note Calls the move assignment operator.
   * @param[in/out] other The source Profile object to move from.
   */
  Profile(Profile&& other) noexcept : m_handle(nullptr) { *this = std::move(other); }

  /**
   * @brief Move assignment operator for Profile objects.
   * @param[in/out] other The source Profile object to move from.
   */
  Profile& operator=(Profile&& other) {
    std::swap(m_handle, other.m_handle);
    return *this;
  }

  /**
   * @brief Destructor for the Profile class.
   */

  ~Profile() {
    const Genie_Status_t status = GenieProfile_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the profile handle." << std::endl;
    }
  }

  /**
   * @brief Retrieves the JSON data associated with this Profile object.
   * @return A string containing the JSON data for the profile.
   * @throws Exception if the JSON data retrieval fails.
   */
  inline std::string getJsonData() {
    const Genie_AllocCallback_t callback = [](size_t size, const char** data) {
      *data = new char[size];
    };
    const char* jsonPtr         = nullptr;
    const Genie_Status_t status = GenieProfile_getJsonData(m_handle, callback, &jsonPtr);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to get the profile data", __FILE__, __LINE__);
    }
    if (!jsonPtr) {
      throw Exception(status, "Unexpected null JSON pointer", __FILE__, __LINE__);
    }
    std::string jsonStr(jsonPtr);
    delete[] jsonPtr;
    return jsonStr;
  }

 private:
  friend class Dialog;
  friend class Embedding;
  friend class Pipeline;
  friend class Node;
  /**
   * @brief Returns the handle associated with this object
   */
  const GenieProfile_Handle_t operator()() const { return m_handle; }

  GenieProfile_Handle_t m_handle = nullptr;
};

}  // namespace genie

#endif  // GENIE_PROFILE_HPP
