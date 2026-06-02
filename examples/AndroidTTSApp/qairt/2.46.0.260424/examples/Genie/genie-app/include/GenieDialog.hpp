//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef GENIE_DIALOG_HPP
#define GENIE_DIALOG_HPP

/**
 * @file
 * @brief Defines the Dialog and Dialog::Config C++ classes which expose all functionality of the
 *        GenieDialog.h C API.
 */

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "GenieCommon.hpp"
#include "GenieDialog.h"
#include "GenieDlc.hpp"
#include "GenieLog.hpp"
#include "GenieProfile.hpp"
#include "GenieSampler.hpp"
#include "GenieTokenizer.hpp"

namespace genie {
/**
 * @class Dialog
 * @brief The Dialog C++ class represents the GenieDialog_Handle_t and it's related functionality.
 */
class Dialog final {
 public:
  /**
   * @class Dialog::Config
   * @brief Represents configuration for the `Dialog` class
   */
  class Config final {
   public:
    /**
     * @brief Constructs a Config object.
     * @param[in] configStr A string containing the JSON configuration.
     * @throws Exception if the JSON string does not match the schema.
     */
    explicit Config(std::string configStr) : m_handle(nullptr) {
      const Genie_Status_t status = GenieDialogConfig_createFromJson(configStr.c_str(), &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw Exception(status, "Failed to create the dialog config from JSON", __FILE__, __LINE__);
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
      const Genie_Status_t status = GenieDialogConfig_createFromDlc(
          (*dlc)(), useCaseNameStr.c_str(), configStrPtr, &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw Exception(status, "Failed to create the dialog config from DLC", __FILE__, __LINE__);
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
      const Genie_Status_t status = GenieDialogConfig_free(m_handle);
      if (GENIE_STATUS_SUCCESS != status) {
        std::cerr << "Failed to free the dialog config." << std::endl;
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
      const Genie_Status_t status = GenieDialogConfig_bindProfiler(m_handle, (*profile)());
      if (GENIE_STATUS_SUCCESS != status) {
        throw Exception(
            status, "Failed to bind the profile handle with the dialog config", __FILE__, __LINE__);
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
      const Genie_Status_t status = GenieDialogConfig_bindLogger(m_handle, (*logger)());
      if (GENIE_STATUS_SUCCESS != status) {
        throw Exception(
            status, "Failed to bind the logger handle with the dialog config", __FILE__, __LINE__);
      }
    }

    /**
     * @brief Returns the handle associated with this object
     */
    const GenieDialogConfig_Handle_t operator()() const { return m_handle; }

   private:
    GenieDialogConfig_Handle_t m_handle = nullptr;
  };

  /**
   * @brief Constructs a `Dialog` instance with the given configuration.
   * @param[in] config An lvalue or rvalue to the Config configuration object.
   * @throws Exception If there is failure in creating the Dialog.
   */
  template <typename T>
  explicit Dialog(T&& config) {
    const Genie_Status_t status = GenieDialog_create(std::forward<T>(config)(), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw Exception(status, "Failed to create the dialog", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Deleted copy constructor to prevent copying of Dialog objects
   */
  Dialog(const Dialog&) = delete;

  /**
   * @brief Deleted copy assignment operator to prevent copying of Dialog objects.
   */
  Dialog& operator=(const Dialog&) = delete;

  /**
   * @brief Move constructor for Dialog objects.
   * @note Calls the move assignment operator.
   * @param[in/out] other The source Dialog object to move from.
   */
  Dialog(Dialog&& other) noexcept : m_handle(nullptr) { *this = std::move(other); }

  /**
   * @brief Move assignment operator for Dialog objects.
   * @param[in/out] other The source Dialog object to move from.
   */
  Dialog& operator=(Dialog&& other) {
    std::swap(m_handle, other.m_handle);
    return *this;
  }

  /**
   * @brief Destructor for the Dialog class.
   */
  ~Dialog() {
    const Genie_Status_t status = GenieDialog_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the dialog." << std::endl;
    }
  }

  /**
   * @brief Sends a query (prompt) to the model and retrieves its response
   * @param[in] prompt The input string to be sent to the model.
   * @param[in] callback A client-defined callback function to handle query responses.
   * @param[in] userData The userData field provided to query.
   * @throws Exception If query fails
   */
  inline void query(const std::string prompt,
                    GenieDialog_SentenceCode_t scode,
                    GenieDialog_QueryCallback_t callback,
                    const void* userData) {
    const Genie_Status_t status =
        GenieDialog_query(m_handle, prompt.c_str(), scode, callback, userData);
    if (GENIE_STATUS_SUCCESS != status && GENIE_STATUS_WARNING_ABORTED != status &&
        GENIE_STATUS_WARNING_PAUSED != status) {
      throw Exception(status, "Failed to query", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Sends batch queries (prompts) to the model and retrieves their responses
   * @param[in] prompts The batch query strings to be sent to the model.
   * @param[in] callback A client-defined callback function to handle query responses.
   * @param[in] userData The userData field provided to query.
   * @throws Exception If query fails
   */
  inline void batchQuery(const std::vector<const char*>& prompts,
                         GenieDialog_SentenceCode_t scode,
                         GenieDialog_QueryCallback_t callback,
                         const void** userData) {
    const Genie_Status_t status =
        GenieDialog_batchQuery(m_handle,
                               const_cast<const char**>(prompts.data()),
                               prompts.size(),
                               scode,
                               callback,
                               userData);

    if (GENIE_STATUS_SUCCESS != status && GENIE_STATUS_WARNING_ABORTED != status &&
        GENIE_STATUS_WARNING_PAUSED != status) {
      throw Exception(status, "Failed to batch query", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Executes an embedding query with the provided embeddings.
   * @param[in] embeddings A pointer to the array of embeddings to be queried.
   * @param[in] embeddingsSize The size of the embeddings array
   * @param[in] t2eCallback A callback function to handle the results of the query.
   * @param[in] queryCallback A client-defined callback function to handle query responses.
   * @param[in] userData The userData field provided to query.
   * @throws Exception If query fails
   */
  inline void embeddingQuery(const void* embeddings,
                             const uint32_t embeddingsSize,
                             const GenieDialog_TokenToEmbeddingCallback_t t2eCallback,
                             const GenieDialog_QueryCallback_t queryCallback,
                             const void* userData) {
    const Genie_Status_t status =
        GenieDialog_embeddingQuery(m_handle,
                                   embeddings,
                                   embeddingsSize,
                                   GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE,
                                   t2eCallback,
                                   queryCallback,
                                   userData);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to query with embedding", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Executes an embedding query with the provided embeddings and outputs token ids.
   * @param[in] embeddings A pointer to the array of embeddings to be queried.
   * @param[in] embeddingsSize The size of the embeddings array
   * @param[in] t2eCallback A callback function to handle the results of the query.
   * @param[in] tokenQueryCallback A client-defined callback function to handle token responses.
   * @param[in] userData The userData field provided to query.
   * @throws Exception If query fails
   */
  inline void embeddingTokenQuery(const void* embeddings,
                                  const uint32_t embeddingsSize,
                                  const GenieDialog_TokenToEmbeddingCallback_t t2eCallback,
                                  const GenieDialog_TokenQueryCallback_t tokenQueryCallback,
                                  const void* userData) {
    const Genie_Status_t status =
        GenieDialog_embeddingTokenQuery(m_handle,
                                        embeddings,
                                        embeddingsSize,
                                        GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE,
                                        t2eCallback,
                                        tokenQueryCallback,
                                        userData);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to query with embedding", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Executes token to token query.
   *
   * @param[in] tokens The input tokens array.
   * @param[in] numTokens The size of input tokens array.
   * @param[in] sentenceCode The sentence code indicating the contents of the queryStr.
   * @param[in] callback Callback function to handle token to token query responses. Cannot be NULL.
   * @param[in] userData User defined field provided in the query responses. Can be NULL.
   * @throws Exception If token query fails
   */
  inline void tokenQuery(const uint32_t* tokens,
                         const uint32_t numTokens,
                         const GenieDialog_SentenceCode_t sentenceCode,
                         const GenieDialog_TokenQueryCallback_t callback,
                         const void* userData) {
    const Genie_Status_t status =
        GenieDialog_tokenQuery(m_handle, tokens, numTokens, sentenceCode, callback, userData);
    if (GENIE_STATUS_SUCCESS != status)
      throw Exception(status, "Token Query failed", __FILE__, __LINE__);
  }

  /**
   * @brief Resets dialog state
   * @throws Exception If reset fails
   */
  inline void reset() {
    const Genie_Status_t status = GenieDialog_reset(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to reset", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Saves the state of dialog to filesystem
   * @param[in] path Path where dialog state will be saved
   * @throws Exception If save fails
   */
  template <typename T>
  inline void save(T&& path) {
    const Genie_Status_t status = GenieDialog_save(m_handle, std::forward<T>(path).c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to save", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Restores state of a dialog dialog from a file
   * @param[in] path Path where dialog state will be restored from.
   * @throws Exception If restore fails
   */
  template <typename T>
  inline void restore(T&& path) {
    const Genie_Status_t status = GenieDialog_restore(m_handle, std::forward<T>(path).c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to restore", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Applies LoRA Adapter.
   * @param[in] engine The Engine to which the LoRA Adapter weights are intended to be applied
   * @param[in] loraAdapterName The name of the LoRA Adapter weights
   * @note When switching LoRA adapters, it is recommended to call Dialog::reset().
   * @throws Exception if applying the LoRA adapter failed.
   */
  inline void applyLora(const std::string& engine, const std::string& loraAdapterName) {
    const Genie_Status_t status =
        GenieDialog_applyLora(m_handle, engine.c_str(), loraAdapterName.c_str());
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
        GenieDialog_setLoraStrength(m_handle, engine.c_str(), tensorName.c_str(), alpha);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set the LoRA alpha strength", __FILE__, __LINE__);
    }
  }

 /**
   * @brief Releases memory associated with a specified LoRA adapter from the given engine.
   * @param[in] engine The Engine to which the LoRA Adapter weights are intended to be released
   * @param[in] loraAdapterName The name of the LoRA adapter whose memory should be released.
   * @throws Exception if releasing the LoRA adapter failed.
   */
  inline void releaseLoraMemory(const std::string& engine, const std::string& loraAdapterName) {
    const Genie_Status_t status =
        GenieDialog_releaseLoraMemory(m_handle, engine.c_str(), loraAdapterName.c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to release the LoRA adapter memory", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Signals the dialog instance with an action.
   * @param[in] action Action to perform.
   * @throws Exception if signalling the action failed.
   */
  inline void signal(const GenieDialog_Action_t action) {
    const Genie_Status_t status = GenieDialog_signal(m_handle, action);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to signal action", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Obtains the dialogs internal sampler instance.
   * @return The `Sampler` object associated with this Dialog instance.
   * @throws Exception if constructing the Sampler failed.
   */
  inline Sampler getSampler() {
    GenieSampler_Handle_t samplerHandle = nullptr;
    const Genie_Status_t status         = GenieDialog_getSampler(m_handle, &samplerHandle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception("Failed to get sampler", __FILE__, __LINE__);
    }
    return Sampler(samplerHandle);
  }

  /**
   * @brief Obtains the dialogs internal tokenizer instance.
   * @return The `Tokenizer` object associated with this Dialog instance.
   * @throws Exception if constructing the Tokenizer failed.
   */
  inline Tokenizer getTokenizer() {
    GenieTokenizer_Handle_t tokenizerHandle = nullptr;
    const Genie_Status_t status             = GenieDialog_getTokenizer(m_handle, &tokenizerHandle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception("Failed to get tokenizer", __FILE__, __LINE__);
    }
    return Tokenizer(tokenizerHandle);
  }

  /**
   * @brief Sets the stop sequence string.
   * @param[in] newStopSequences The stop sequence string.
   * @throws Exception if setting the stop sequence failed.
   */
  inline void setStopSequence(const std::string newStopSequences) {
    const Genie_Status_t status = GenieDialog_setStopSequence(m_handle, newStopSequences.c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set the stop sequence", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Sets the dialog priority.
   * @param[in] engineRole The engine role to apply the priority.
   * @param[in] priority The priority.
   * @throws Exception if setting the priority failed.
   */
  inline void setPriority(const std::string engineRole, const GenieDialog_Priority_t priority) {
    const Genie_Status_t status = GenieDialog_setPriority(m_handle, engineRole.c_str(), priority);
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
    const Genie_Status_t status = GenieDialog_setOemKey(m_handle, oemKey.c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to set the OEM key", __FILE__, __LINE__);
    }
  }

  /**
   * @brief sets Performance policy on the QNN backend.
   * @param[in] perfProfile The performance profile.
   * @throws Exception if setting performance policy failed.
   */
  inline void setPerformancePolicy(const Genie_PerformancePolicy_t perfProfile) {
    const Genie_Status_t status = GenieDialog_setPerformancePolicy(m_handle, perfProfile);
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
    const Genie_Status_t status = GenieDialog_getPerformancePolicy(m_handle, &perfProfile);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to get Performance policy", __FILE__, __LINE__);
    }
    return perfProfile;
  }

  /**
   * @brief sets the maximum number of generated tokens for a dialog.
   * @param[in] maxNumTokens The maximum number of generated tokens.
   * @throws Exception if setting the maximum number of generated tokens failed.
   */
  inline void setMaxNumTokens(const uint32_t maxNumTokens) {
    const Genie_Status_t status = GenieDialog_setMaxNumTokens(m_handle, maxNumTokens);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(
          status, "Failed to set the maximum number of generated tokens", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Gets a parameter or state variable from the dialog.
   * @param[in] key An enum key indicating which value is to be returned.
   * @param[in] callback A callback function for allocating string types. Must not be NULL when
   * dataType can be GENIE_DATATYPE_STRING.
   * @return A pair containing the data type and value.
   * @throws Exception if the operation fails.
   * @note For string types, the caller is responsible for managing the memory allocated by the
   * callback.
   */
  inline std::pair<Genie_DataType_t, Genie_Value_t> getValue(const GenieDialog_Param_t key,
                                                             const Genie_AllocCallback_t callback) {
    Genie_DataType_t dataType;
    Genie_Value_t value;

    const Genie_Status_t status = GenieDialog_getValue(m_handle, key, callback, &dataType, &value);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to get dialog parameter", __FILE__, __LINE__);
    }

    return std::make_pair(dataType, value);
  }

  /**
   * @brief Returns the handle associated with this object
   */
  const GenieDialog_Handle_t operator()() const { return m_handle; }

 private:
  GenieDialog_Handle_t m_handle = nullptr;
};

}  // namespace genie

#endif  // GENIE_DIALOG_HPP
