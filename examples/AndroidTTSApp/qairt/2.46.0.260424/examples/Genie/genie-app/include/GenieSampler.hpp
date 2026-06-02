//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef GENIE_SAMPLER_HPP
#define GENIE_SAMPLER_HPP

/**
 * @file
 * @brief Defines the Sampler and Sampler::Config C++ classes which expose all functionality of the
 *        GenieSampler.h C API.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

#include "GenieCommon.hpp"
#include "GenieSampler.h"

namespace genie {
/**
 * @class Sampler
 * @brief The Sampler C++ class represents the GenieSampler_Handle_t and its related functionality.
 */
class Sampler final {
 public:
  /**
   * @class Sampler::Config
   * @brief Represents configuration for the `Sampler` class
   */
  class Config final {
   public:
    /**
     * @brief Constructs a Config object.
     * @param[in] json A string containing the JSON configuration.
     * @throws Exception if the JSON string does not match the schema.
     */
    explicit Config(std::string json) : m_handle(nullptr) {
      const Genie_Status_t status = GenieSamplerConfig_createFromJson(json.c_str(), &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw Exception(status, "Failed to create the sampler config", __FILE__, __LINE__);
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
      const Genie_Status_t status = GenieSamplerConfig_free(m_handle);
      if (GENIE_STATUS_SUCCESS != status) {
        std::cerr << "Failed to free the sampler config." << std::endl;
      }
    }

    /**
     * @brief Sets a parameter for the sampler configuration.
     * @param[in] key A string indicating the sampler parameter to update. If empty, the value must
     *                have the entire sampler configuration string.
     * @param[in] value The value of the parameter to set.
     * @throws Exception if setting the parameter fails.
     */
    inline void setParam(const std::string& key, const std::string& value) {
      const Genie_Status_t status =
          GenieSamplerConfig_setParam(m_handle, key.c_str(), value.c_str());
      if (GENIE_STATUS_SUCCESS != status) {
        throw Exception(status, "Failed to set the parameter", __FILE__, __LINE__);
      }
    }

    /**
     * @brief Returns the handle associated with this object
     */
    GenieSamplerConfig_Handle_t operator()() const { return m_handle; }

   private:
    GenieSamplerConfig_Handle_t m_handle = nullptr;
  };

  /**
   * @brief Constructs a `Sampler` instance with the given configuration.
   * @param[in] config An lvalue or rvalue to the Config configuration object.
   * @throws Exception If there is failure in creating the Sampler.
   */
  template <typename T>
  explicit Sampler(T&& config) : m_handle(nullptr) {
    const Genie_Status_t status = GenieSampler_create(std::forward<T>(config)(), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw Exception(status, "Failed to create the sampler", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Destructor for the Sampler class.
   */
  ~Sampler() {
    if (m_handle) {
      const Genie_Status_t status = GenieSampler_free(m_handle);
      if (GENIE_STATUS_SUCCESS != status) {
        std::cerr << "Failed to free the sampler." << std::endl;
      }
    }
  }

  /**
   * @brief Deleted copy constructor to prevent copying of Sampler objects
   */
  Sampler(const Sampler&) = delete;

  /**
   * @brief Deleted copy assignment operator to prevent copying of Sampler objects.
   */
  Sampler& operator=(const Sampler&) = delete;

  /**
   * @brief Move constructor for Sampler objects.
   * @note Calls the move assignment operator.
   * @param[in/out] other The source Sampler object to move from.
   */
  Sampler(Sampler&& other) noexcept : m_handle(nullptr) { *this = std::move(other); }

  /**
   * @brief Move assignment operator for Sampler objects.
   * @param[in/out] other The source Sampler object to move from.
   */
  Sampler& operator=(Sampler&& other) {
    std::swap(m_handle, other.m_handle);
    return *this;
  }

  /**
   * @brief Applied a sampler configuration.
   * @param[in] config The sampler configuration.
   * @throws Exception if applying the config fails.
   */
  template <typename T>
  void applyConfig(T&& config) {
    const Genie_Status_t status = GenieSampler_applyConfig(m_handle, std::forward<T>(config)());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to apply the sampler configuration", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Registers a callback function for the sampler. Note: This API will soon be deprecated in
   * favor of registerUserDataCallback
   * @param[in] name The name of the callback function.
   * @param[in] callback The callback function to register.
   * @throws Exception if the callback registration fails.
   */
  inline static void registerCallback(const std::string& name,
                                      GenieSampler_ProcessCallback_t callback) {
    const Genie_Status_t status = GenieSampler_registerCallback(name.c_str(), callback);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to register sampler callback", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Registers a callback function for the sampler.
   * @param[in] name The name of the callback function.
   * @param[in] callback The callback function to register.
   * @param[in] userData Void pointer to add user data pertaining to callback function.
   * @throws Exception if the callback registration fails.
   */
  inline static void registerUserDataCallback(const std::string& name,
                                              GenieSampler_UserDataCallback_t callback,
                                              const void* userData) {
    const Genie_Status_t status =
        GenieSampler_registerUserDataCallback(name.c_str(), callback, userData);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to register sampler callback", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Sample tokens from a logits buffer.
   * @param[in] data Pointer to logits buffer (raw bytes).
   * @param[in] dataSize Size of logits buffer in bytes.
   * @param[in] dataConfig JSON string describing the logits tensor.
   * @param[in] callback Callback invoked with sampled tokens.
   * @param[in] userData Opaque pointer forwarded to callback.
   * @throws Exception if sampling fails.
   */
  inline void sampleData(const void* data,
                         size_t dataSize,
                         const std::string& dataConfig,
                         GenieSampler_Callback_t callback,
                         const void* userData) {
    const Genie_Status_t status =
        GenieSampler_sampleData(m_handle, data, dataSize, dataConfig.c_str(), callback, userData);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to sample data", __FILE__, __LINE__);
    }
  }

 private:
  friend class Dialog;
  friend class Node;

  /**
   * @brief Directly constructs a `Sampler` instance.
   * @note This instance is created via Dialog::getSampler or Node::getSampler which returns the
   *       sampler instance.
   * @param[in] handle The sampler handle obtained from the dialog or node.
   */
  explicit Sampler(GenieSampler_Handle_t handle) : m_handle(handle) {}

  GenieSampler_Handle_t m_handle = nullptr;
};

}  // namespace genie

#endif  // GENIE_SAMPLER_HPP
