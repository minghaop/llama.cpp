//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef GENIE_TOKENIZER_HPP
#define GENIE_TOKENIZER_HPP

/**
 * @file
 * @brief Defines the Tokenizer C++ class which exposes all functionality of the
 *        GenieTokenizer.h C API.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

#include "GenieCommon.hpp"
#include "GenieTokenizer.h"

namespace genie {
/**
 * @class Tokenizer
 * @brief The Tokenizer C++ class represents the GenieTokenizer_Handle_t and its related
 * functionality.
 */
class Tokenizer final {
 public:
  /**
   * @brief Destructor for the Tokenizer class.
   * @note Since the Tokenizer can only be created via a Dialog::getTokenizer, the Dialog will
   *       implicitly destruct the internal tokenizer resources.
   */
  ~Tokenizer() {}

  /**
   * @brief Deleted copy constructor to prevent copying of Tokenizer objects
   */
  Tokenizer(const Tokenizer&) = delete;

  /**
   * @brief Deleted copy assignment operator to prevent copying of Tokenizer objects.
   */
  Tokenizer& operator=(const Tokenizer&) = delete;

  /**
   * @brief Move constructor for Tokenizer objects.
   * @note Calls the move assignment operator.
   * @param[in/out] other The source Tokenizer object to move from.
   */
  Tokenizer(Tokenizer&& other) noexcept : m_handle(nullptr) { *this = std::move(other); }

  /**
   * @brief Move assignment operator for Tokenizer objects.
   * @param[in/out] other The source Sampler object to move from.
   */
  Tokenizer& operator=(Tokenizer&& other) {
    std::swap(m_handle, other.m_handle);
    return *this;
  }

  /**
   * @brief Encodes input text into token ids.
   * @param[in] inputString Null-terminated Input string.
   * @param[out] tokenIds The encoded token ids.
   * @param[out] numTokenIds The number of encoded token ids.
   * @throws Exception if the encode text fails.
   */
  inline void encode(const std::string inputString,
                     const int32_t** tokenIds,
                     uint32_t* numTokenIds) {
    const Genie_AllocCallback_t callback = [](size_t size, const char** data) {
      *data = new char[size];
    };
    const Genie_Status_t status =
        GenieTokenizer_encode(m_handle, inputString.c_str(), callback, tokenIds, numTokenIds);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to encode input text into token ids.", __FILE__, __LINE__);
    }
  }

  /**
   * @brief Decodes input token ids into text.
   * @param[in] tokenIds Input token ids.
   * @param[out] numTokenIds The number of input token ids.
   * @return A string containing the decoded text of the input token ids.
   * @throws Exception if the decode token ids fails.
   */
  inline std::string decode(const int32_t* tokenIds, const uint32_t numTokenIds) {
    const Genie_AllocCallback_t callback = [](size_t size, const char** data) {
      *data = new char[size];
    };
    const char* strPtr = nullptr;
    const Genie_Status_t status =
        GenieTokenizer_decode(m_handle, tokenIds, numTokenIds, callback, &strPtr);
    if (GENIE_STATUS_SUCCESS != status) {
      throw Exception(status, "Failed to decode input token ids into text.", __FILE__, __LINE__);
    }
    if (!strPtr) {
      throw Exception(status, "Unexpected null string pointer", __FILE__, __LINE__);
    }
    std::string outStr(strPtr);
    delete strPtr;
    return outStr;
  }

 private:
  friend class Dialog;

  /**
   * @brief Directly constructs a `Tokenizer` instance.
   * @note This instance is created via Dialog::getTokenizer which returns the dialogs tokenizer
   *       instance.
   * @param[in] handle The tokenizer handle obtained from the dialog.
   */
  explicit Tokenizer(GenieTokenizer_Handle_t handle) : m_handle(handle) {}

  GenieTokenizer_Handle_t m_handle = nullptr;
};

}  // namespace genie

#endif  // GENIE_SAMPLER_HPP
