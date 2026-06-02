//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef GENIE_NODE_HPP
#define GENIE_NODE_HPP

#include "GenieCommon.hpp"
#include "GenieDlc.hpp"
#include "GenieLog.hpp"
#include "GenieNode.h"
#include "GenieProfile.hpp"
#include "GenieSampler.hpp"

namespace genie {

/**
 * @class Node
 * @brief The Node C++ class represents the GenieNode_Handle_t and it's related
 * functionality.
 */
class Node final {
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
      const Genie_Status_t status = GenieNodeConfig_createFromJson(json.c_str(), &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw Exception(status, "Failed to create the node config", __FILE__, __LINE__);
      }
    }

    /**
     * @brief Constructs a Config object.
     * @param[in] dlcHandle The DLC handle.
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
      const char* configStrPtr = configStr.empty() ? nullptr : configStr.c_str();
      const Genie_Status_t status =
          GenieNodeConfig_createFromDlc((*dlc)(), useCaseNameStr.c_str(), configStrPtr, &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw Exception(status, "Failed to create the node config from DLC", __FILE__, __LINE__);
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
      const Genie_Status_t status = GenieNodeConfig_free(m_handle);
      if (GENIE_STATUS_SUCCESS != status) {
        std::cerr << "Failed to free the node config." << std::endl;
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
      const Genie_Status_t status = GenieNodeConfig_bindProfiler(m_handle, (*profile)());
      if (GENIE_STATUS_SUCCESS != status) {
        throw Exception(
            status, "Failed to bind the profile handle with the node config", __FILE__, __LINE__);
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
      const Genie_Status_t status = GenieNodeConfig_bindLogger(m_handle, (*logger)());
      if (GENIE_STATUS_SUCCESS != status) {
        throw Exception(
            status, "Failed to bind the logger handle with the node config", __FILE__, __LINE__);
      }
    }

    /**
     * @brief Returns the handle associated with this object
     */
    const GenieNodeConfig_Handle_t operator()() const { return m_handle; }

   private:
    GenieNodeConfig_Handle_t m_handle = nullptr;
  };
  /**
   * @brief Constructor for Node class.
   * @param[in] handle The GenieNode_Handle_t handle.
   */
  template <typename T>
  explicit Node(T&& config) {
    const Genie_Status_t status = GenieNode_create(std::forward<T>(config)(), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw Exception(status, "Failed to create the Genie Node", __FILE__, __LINE__);
    }
  }

  ~Node() {
    const Genie_Status_t status = GenieNode_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the Genie Node." << std::endl;
    }
  }

  void setTextCallback(GenieNode_IOName_t nodeIOName, GenieNode_TextOutput_Callback_t callback) {
    const Genie_Status_t status = GenieNode_setTextCallback(m_handle, nodeIOName, callback);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set the text output callback", __FILE__, __LINE__);
    }
  }

  void setEmbeddingCallback(GenieNode_IOName_t nodeIOName,
                            GenieNode_EmbeddingOutputCallback_t callback) {
    const Genie_Status_t status = GenieNode_setEmbeddingCallback(m_handle, nodeIOName, callback);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set the embedding output callback", __FILE__, __LINE__);
    }
  }

  void setData(GenieNode_IOName_t nodeIOName, std::string text, const char* dataConfig = nullptr) {
    const Genie_Status_t status =
        GenieNode_setData(m_handle, nodeIOName, (void*)(text.c_str()), text.size(), dataConfig);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set the text input data", __FILE__, __LINE__);
    }
  }

  void train() {
    const Genie_Status_t status = GenieNode_train(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set the text input data", __FILE__, __LINE__);
    }
  }

  void saveLora(const std::string& engine, const std::string& loraAdapterName) {
    const Genie_Status_t status =
        GenieNode_saveLora(m_handle, engine.c_str(), loraAdapterName.c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to save the LoRA adapter", __FILE__, __LINE__);
    }
  }

  void setData(GenieNode_IOName_t nodeIOName,
               const void* data,
               const size_t dataSize,
               const char* dataConfig = nullptr) {
    const Genie_Status_t status =
        GenieNode_setData(m_handle, nodeIOName, (void*)data, dataSize, dataConfig);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set the input data", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Applies LoRA Adapter.
   * @param[in] engine The Engine to which the LoRA Adapter weights are intended to be applied
   * @param[in] loraAdapterName The name of the LoRA Adapter weights
   * @note When switching LoRA adapters, it is recommended to call Node::reset().
   * @throws Exception if applying the LoRA adapter failed.
   */
  inline void applyLora(const std::string& engine, const std::string& loraAdapterName) {
    const Genie_Status_t status =
        GenieNode_applyLora(m_handle, engine.c_str(), loraAdapterName.c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to apply the LoRA adapter", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Sets the LoRA alpha strength.
   * @param[in] engine The Engine to which LoRA strength is intended to be applied (i.e. "primary").
   * @param[in] tensorName LoRA alpha tensor name.
   * @param[in] alpha Applies the value to LoRA alpha tensor.
   * @throws Exception if applying the LoRA adapter failed.
   *
   */
  inline void setLoraStrength(const std::string& engine,
                              const std::string& tensorName,
                              const float& alpha) {
    const Genie_Status_t status =
        GenieNode_setLoraStrength(m_handle, engine.c_str(), tensorName.c_str(), alpha);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set the LoRA alpha strength", __FILE__, __LINE__);
    }
  }

  void execute(std::string executionConfig, void* userData = nullptr) {
    const Genie_Status_t status = GenieNode_execute(m_handle, executionConfig.c_str(), userData);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to execute the Genie Node", __FILE__, __LINE__);
    }
  }

  void getData(GenieNode_IOName_t nodeIOName,
               std::string ioConfig,
               GenieNode_IOCallback_t ioCallback,
               void* userData = nullptr) {
    const Genie_Status_t status =
        GenieNode_getData(m_handle, nodeIOName, ioConfig.c_str(), ioCallback, userData);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to get the node output data", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Resets node state
   * @throws Exception If reset fails
   */
  inline void reset() {
    const Genie_Status_t status = GenieNode_reset(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to reset", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Gets the sampler associated with this node
   * @return Sampler object
   * @throws Exception if getting the sampler fails
   */
  inline Sampler getSampler() {
    GenieSampler_Handle_t samplerHandle = nullptr;
    const Genie_Status_t status         = GenieNode_getSampler(m_handle, &samplerHandle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to get sampler from node", __FILE__, __LINE__);
    }
    return Sampler(samplerHandle);
  }

  const GenieNode_Handle_t operator()() const { return m_handle; }

 private:
  GenieNode_Handle_t m_handle = nullptr;
};
}  // namespace genie
#endif  // GENIE_NODE_HPP
