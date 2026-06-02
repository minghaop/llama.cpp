//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef GENIE_ACCURACY_HPP
#define GENIE_ACCURACY_HPP

/**
 * @file
 * @brief Defines the Accuracy and Accuracy::Config C++ classes which expose all functionality of
 *        the GenieAccuracy.h C API.
 */

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "GenieAccuracy.h"
#include "GenieCommon.hpp"
#include "GenieDialog.hpp"
#include "GenieNode.hpp"

namespace genie {
/**
 * @class Accuracy
 * @brief The Accuracy C++ class represents the GenieAccuracy_Handle_t and it's related
 *        functionality.
 */
class Accuracy final {
 public:
  /**
   * @class Accuracy::Config
   * @brief Represents configuration for the `Accuracy` class
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
      const Genie_Status_t status = GenieAccuracyConfig_createFromJson(json.c_str(), &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw Exception(status, "Failed to create the accuracy config", __FILE__, __LINE__);
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
        const Genie_Status_t status = GenieAccuracyConfig_free(m_handle);
        if (GENIE_STATUS_SUCCESS != status) {
          std::cerr << "Failed to free the accuracy config." << std::endl;
        }
      }
    }

    /**
     * @brief Returns the handle associated with this object
     */
    const GenieAccuracyConfig_Handle_t operator()() const { return m_handle; }

   private:
    GenieAccuracyConfig_Handle_t m_handle = nullptr;
  };

  /**
   * @brief Constructs an `Accuracy` instance with the given configuration and dialog.
   * @param[in] config An lvalue or rvalue to the Config configuration object.
   * @param[in] dialog A pointer to the Dialog object.
   * @throws Exception If there is failure in creating the Accuracy.
   */
  template <typename T>
  explicit Accuracy(T&& config, std::shared_ptr<Dialog> dialog) {
    const Genie_Status_t status =
        GenieAccuracy_createFromDialog(std::forward<T>(config)(), (*dialog)(), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw Exception(status, "Failed to create the accuracy handle", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Constructs an `Accuracy` instance with the given configuration and node.
   * @param[in] config An lvalue or rvalue to the Config configuration object.
   * @param[in] node A pointer to the Node object.
   * @throws Exception If there is failure in creating the Accuracy.
   */
  template <typename T>
  explicit Accuracy(T&& config, std::shared_ptr<Node> node) {
    const Genie_Status_t status =
        GenieAccuracy_createFromNode(std::forward<T>(config)(), (*node)(), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw Exception(status, "Failed to create the accuracy handle", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Deleted copy constructor to prevent copying of Accuracy objects
   */
  Accuracy(const Accuracy&) = delete;

  /**
   * @brief Deleted copy assignment operator to prevent copying of Accuracy objects.
   */
  Accuracy& operator=(const Accuracy&) = delete;

  /**
   * @brief Move constructor for Accuracy objects.
   * @note Calls the move assignment operator.
   * @param[in/out] other The source Accuracy object to move from.
   */
  Accuracy(Accuracy&& other) noexcept : m_handle(nullptr) { *this = std::move(other); }

  /**
   * @brief Move assignment operator for Accuracy objects.
   * @param[in/out] other The source Accuracy object to move from.
   */
  Accuracy& operator=(Accuracy&& other) {
    std::swap(m_handle, other.m_handle);
    return *this;
  }

  /**
   * @brief Destructor for the Accuracy class.
   */
  ~Accuracy() {
    const Genie_Status_t status = GenieAccuracy_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the accuracy handle." << std::endl;
    }
  }

  /**
   * @brief Retrieves the JSON data associated with this Accuracy object.
   * @return A string containing the JSON data for the profile.
   * @throws Exception if the JSON data retrieval fails.
   */
  inline std::string getJsonData() {
    const Genie_AllocCallback_t callback([](size_t size, const char** data) {
      *data = reinterpret_cast<char*>(malloc(size));
      if (*data == nullptr) {
        throw std::runtime_error("Cannot allocate memory for JSON data");
      }
    });
    const char* jsonPtr         = nullptr;
    const Genie_Status_t status = GenieAccuracy_compute(m_handle, callback, &jsonPtr);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to get the accuracy data", __FILE__, __LINE__);
    }
    if (!jsonPtr) {
      throw Exception(status, "Unexpected null JSON pointer", __FILE__, __LINE__);
    }
    std::string jsonStr(jsonPtr);
    delete[] jsonPtr;
    return jsonStr;
  }

 private:
  /**
   * @brief Returns the handle associated with this object
   */
  const GenieAccuracy_Handle_t operator()() const { return m_handle; }

  GenieAccuracy_Handle_t m_handle = nullptr;
};

}  // namespace genie

#endif  // GENIE_ACCURACY_HPP
