//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef GENIE_EMBEDDING_HPP
#define GENIE_EMBEDDING_HPP

/**
 * @file
 * @brief Defines the Embedding and Embedding::Config C++ classes which expose all functionality of
 * the GenieEmbedding.h C API.
 */

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "GenieCommon.hpp"
#include "GenieDlc.hpp"
#include "GenieEmbedding.h"
#include "GenieLog.hpp"
#include "GenieProfile.hpp"

namespace genie {
/**
 * @class Embedding
 * @brief The Embedding C++ class represents the GenieEmbedding_Handle_t and it's related
 * functionality.
 */
class Embedding final {
 public:
  /**
   * @class Embedding::Config
   * @brief Represents configuration for the `Embedding` class
   */
  class Config final {
   public:
    /**
     * @brief Constructs a Config object.
     * @param[in] json A string containing the JSON configuration.
     * @throws Exception if the JSON string does not match the schema.
     */
    explicit Config(std::string json) : m_handle(nullptr) {
      const Genie_Status_t status = GenieEmbeddingConfig_createFromJson(json.c_str(), &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw Exception(status, "Failed to create the embedding config", __FILE__, __LINE__);
      }
    }

    /**
     * @brief Constructs a Config object.
     * @param[in] dlc The DLC object.
     * @param[in] useCaseNameStr The name of the genie use case supported by the DLC.
     * @param[in] configStr The configuration string to overwrite the config corresponding to
     * useCaseName.
     * @throws Exception if the JSON string does not match the schema.
     */
    explicit Config(std::shared_ptr<Dlc> dlc,
                    std::string useCaseNameStr,
                    std::string configStr = "")
        : m_handle(nullptr) {
      if (!dlc) {
        throw Exception("Dlc object is null", __FILE__, __LINE__);
      }
      const char* configStrPtr    = configStr.empty() ? nullptr : configStr.c_str();
      const Genie_Status_t status = GenieEmbeddingConfig_createFromDlc(
          (*dlc)(), useCaseNameStr.c_str(), configStrPtr, &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw Exception(
            status, "Failed to create the embedding config from DLC", __FILE__, __LINE__);
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
      const Genie_Status_t status = GenieEmbeddingConfig_free(m_handle);
      if (GENIE_STATUS_SUCCESS != status) {
        std::cerr << "Failed to free the embedding config." << std::endl;
      }
    }

    /**
     * @brief Binds profiler to config object
     * @param[in] profile A shared pointer to a Profile object
     * @throws Exception if there is a failure in binding profile handle with the config
     */
    inline void bindProfile(std::shared_ptr<Profile> profile) {
      if (!profile) {
        throw Exception("Profile object is null", __FILE__, __LINE__);
      }
      const Genie_Status_t status = GenieEmbeddingConfig_bindProfiler(m_handle, (*profile)());
      if (GENIE_STATUS_SUCCESS != status) {
        throw Exception(status,
                        "Failed to bind the profile handle with the embedding config",
                        __FILE__,
                        __LINE__);
      }
    }

    /**
     * @brief Binds logger to config object
     * @param[in] logger A shared pointer to a Log object
     * @throws Exception if there is a failure in binding log handle with the config
     */
    inline void bindLogger(std::shared_ptr<Log> logger) {
      if (!logger) {
        throw Exception("Log object is null", __FILE__, __LINE__);
      }
      const Genie_Status_t status = GenieEmbeddingConfig_bindLogger(m_handle, (*logger)());
      if (GENIE_STATUS_SUCCESS != status) {
        throw Exception(
            status, "Failed to bind the logger handle with the dialog config", __FILE__, __LINE__);
      }
    }

    /**
     * @brief Returns the handle associated with this object
     */
    GenieEmbeddingConfig_Handle_t operator()() const { return m_handle; }

   private:
    GenieEmbeddingConfig_Handle_t m_handle = nullptr;
  };

  /**
   * @brief Constructs a `Embedding` instance with the given configuration.
   * @param[in] config An lvalue or rvalue to the Config configuration object.
   * @throws Exception If there is failure in creating the Embedding.
   */
  template <typename T>
  explicit Embedding(T&& config) {
    const Genie_Status_t status = GenieEmbedding_create(std::forward<T>(config)(), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw Exception(status, "Failed to create the embedding", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Deleted copy constructor to prevent copying of Embedding objects
   */
  Embedding(const Embedding&) = delete;

  /**
   * @brief Deleted copy assignment operator to prevent copying of Embedding objects.
   */
  Embedding& operator=(const Embedding&) = delete;

  /**
   * @brief Move constructor for Embedding objects.
   * @note Calls the move assignment operator.
   * @param[in/out] other The source Embedding object to move from.
   */
  Embedding(Embedding&& other) noexcept : m_handle(nullptr) { *this = std::move(other); }

  /**
   * @brief Move assignment operator for Embedding objects.
   * @param[in/out] other The source Embedding object to move from.
   */
  Embedding& operator=(Embedding&& other) {
    std::swap(m_handle, other.m_handle);
    return *this;
  }

  /**
   * @brief Destructor for the Embedding class.
   */
  ~Embedding() {
    const Genie_Status_t status = GenieEmbedding_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free embedding." << std::endl;
    }
  }

  /**
   * @brief A function to generate embedding for prompted text.
   * @param[in] prompt A std::string prompt to be sent to the model.
   * @param[in] callback Callback function to handle generated embeddings. Cannot be NULL.
   * @param[in] userData User defined field provided in the query responses. Can be NULL.
   * @throws Exception If there is failure in generation.
   */
  inline void generate(const std::string& prompt,
                       const GenieEmbedding_GenerateCallback_t callback,
                       const void* userData) {
    const Genie_Status_t status =
        GenieEmbedding_generate(m_handle, prompt.c_str(), callback, userData);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception("Failed to generate embedding", __FILE__, __LINE__);
    }
  }

  /**
   * @brief sets Performance policy on the QNN backend.
   * @param[in] perfProfile The performance profile.
   * @throws Exception if setting performance policy failed.
   */
  inline void setPerformancePolicy(const Genie_PerformancePolicy_t perfProfile) {
    const Genie_Status_t status = GenieEmbedding_setPerformancePolicy(m_handle, perfProfile);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set Performance policy", __FILE__, __LINE__);
    }
  }

  /**
   * @brief gets Performance policy from the QNN backend.
   * @param[in] perfProfile The performance profile.
   * @throws Exception if getting performance policy failed.
   */
  inline Genie_PerformancePolicy_t getPerformancePolicy() {
    Genie_PerformancePolicy_t perfProfile;
    const Genie_Status_t status = GenieEmbedding_getPerformancePolicy(m_handle, &perfProfile);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to get Performance policy", __FILE__, __LINE__);
    }
    return perfProfile;
  }

 private:
  GenieEmbedding_Handle_t m_handle = nullptr;
};

}  // namespace genie

#endif  // GENIE_EMBEDDING_HPP
