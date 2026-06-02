//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef GENIE_PIPELINE_HPP
#define GENIE_PIPELINE_HPP

/**
 * @file
 * @brief Defines the Pipeline C++ class which exposes all functionality of the GeniePipeline.h C
 * API.
 */

#include "GenieCommon.hpp"
#include "GenieNode.hpp"
#include "GeniePipeline.h"

namespace genie {
/**
 * @class Pipeline
 * @brief The Pipeline C++ class represents the GeniePipeline_Handle_t and it's related
 * functionality.
 */
class Pipeline final {
 public:
  /**
   * @class Pipeline::Config
   * @brief Represents configuration for the `Pipeline` class
   */
  class Config final {
   public:
    /**
     * @brief Constructs a Config object.
     * @param[in] json A string containing the JSON configuration.
     * @throws Exception if the JSON string does not match the schema.
     */
    explicit Config(std::string json) : m_handle(nullptr) {
      const Genie_Status_t status = GeniePipelineConfig_createFromJson(json.c_str(), &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw Exception(status, "Failed to create the pipeline config", __FILE__, __LINE__);
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
      const Genie_Status_t status = GeniePipelineConfig_free(m_handle);
      if (GENIE_STATUS_SUCCESS != status) {
        std::cerr << "Failed to free the pipeline config." << std::endl;
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
      const Genie_Status_t status = GeniePipelineConfig_bindProfiler(m_handle, (*profile)());
      if (GENIE_STATUS_SUCCESS != status) {
        throw Exception(status,
                        "Failed to bind the profile handle with the pipeline config",
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
      const Genie_Status_t status = GeniePipelineConfig_bindLogger(m_handle, (*logger)());
      if (GENIE_STATUS_SUCCESS != status) {
        throw Exception(status,
                        "Failed to bind the logger handle with the pipeline config",
                        __FILE__,
                        __LINE__);
      }
    }

    /**
     * @brief Returns the handle associated with this object
     */
    const GeniePipelineConfig_Handle_t operator()() const { return m_handle; }

   private:
    GeniePipelineConfig_Handle_t m_handle = nullptr;
  };

  /**
   * @brief Constructs a `Pipeline` instance with the given configuration.
   * @param[in] config An lvalue or rvalue to the Config configuration object.
   * @throws Exception If there is failure in creating the Pipeline.
   */
  template <typename T>
  explicit Pipeline(T&& config) {
    const Genie_Status_t status = GeniePipeline_create(std::forward<T>(config)(), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw Exception(status, "Failed to create the pipeline", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Deleted copy constructor to prevent copying of Pipeline objects
   */
  Pipeline(const Pipeline&) = delete;

  /**
   * @brief Deleted copy assignment operator to prevent copying of Pipeline objects.
   */
  Pipeline& operator=(const Pipeline&) = delete;

  /**
   * @brief Move constructor for Pipeline objects.
   * @note Calls the move assignment operator.
   * @param[in/out] other The source Pipeline object to move from.
   */
  Pipeline(Pipeline&& other) noexcept : m_handle(nullptr) { *this = std::move(other); }

  /**
   * @brief Move assignment operator for Pipeline objects.
   * @param[in/out] other The source Pipeline object to move from.
   */
  Pipeline& operator=(Pipeline&& other) {
    std::swap(m_handle, other.m_handle);
    return *this;
  }

  /**
   * @brief Destructor for the Pipeline class.
   */
  ~Pipeline() {
    const Genie_Status_t status = GeniePipeline_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the pipeline." << std::endl;
    }
  }

  inline void execute(void* userData) {
    const Genie_Status_t status = GeniePipeline_execute(m_handle, userData);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to execute", __FILE__, __LINE__);
    }
  }

  inline void connect(std::shared_ptr<Node> producerNode,
                      GenieNode_IOName_t producerIoName,
                      std::shared_ptr<Node> consumerNode,
                      GenieNode_IOName_t consumerIoName) {
    const Genie_Status_t status = GeniePipeline_connect(
        m_handle, (*producerNode)(), producerIoName, (*consumerNode)(), consumerIoName);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to connect", __FILE__, __LINE__);
    }
  }
  /**
   * @brief Resets pipeline state
   * @throws Exception If reset fails
   */
  inline void reset() {
    const Genie_Status_t status = GeniePipeline_reset(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to reset", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Saves the state of pipeline to filesystem
   * @param[in] path Path where pipeline state will be saved
   * @throws Exception If save fails
   */
  template <typename T>
  inline void save(T&& path) {
    const Genie_Status_t status = GeniePipeline_save(m_handle, std::forward<T>(path).c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to save", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Restores state of a pipeline pipeline from a file
   * @param[in] path Path where pipeline state will be restored from.
   * @throws Exception If restore fails
   */
  template <typename T>
  inline void restore(T&& path) {
    const Genie_Status_t status = GeniePipeline_restore(m_handle, std::forward<T>(path).c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to restore", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Sets the pipeline priority.
   * @param[in] engineRole The engine role to apply the priority.
   * @param[in] priority The priority.
   * @throws Exception if setting the priority failed.
   */
  inline void setPriority(const std::string engineRole, const GeniePipeline_Priority_t priority) {
    const Genie_Status_t status = GeniePipeline_setPriority(m_handle, engineRole.c_str(), priority);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set the priority", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Provides an OEM key to the QNN backend.
   * @param[in] oemKey The OEM key.
   * @throws Exception if setting the OEM key failed.
   */
  inline void setOemKey(const std::string oemKey) {
    const Genie_Status_t status = GeniePipeline_setOemKey(m_handle, oemKey.c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set the OEM key", __FILE__, __LINE__);
    }
  }

  inline void addNode(std::shared_ptr<Node> node) {
    const Genie_Status_t status = GeniePipeline_addNode(m_handle, (*node)());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to add node", __FILE__, __LINE__);
    }
  }

 private:
  GeniePipeline_Handle_t m_handle = nullptr;
};

}  // namespace genie

#endif  // GENIE_PIPELINE_HPP
