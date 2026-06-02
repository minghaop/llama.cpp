//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "qualla/detail/sentence.hpp"
#include "qualla/tokenizer.hpp"
#include <fstream>
#include <vector>
#include <memory>
#include <string>

namespace qualla {

typedef std::function<bool(const std::string&, Sentence::Code)> QueryCbFunction;
typedef std::function<bool(const std::vector<std::string>&, const std::vector<bool>&, Sentence::Code)> BatchQueryCbFunction;
typedef std::function<bool(const int32_t*, const uint32_t, Sentence::Code)> TokenCbFunction;

typedef enum {
  QUALLA_CALLBACK_TYPE_TEXT      = 1,
  QUALLA_CALLBACK_TYPE_TOKEN     = 2,
  QUALLA_CALLBACK_TYPE_BATCH_TEXT= 3,
  QUALLA_CALLBACK_TYPE_UNDEFINED = 0x7fffffff
} QuallaCallBackType;

class DialogCallback {
 public:
  DialogCallback() {
    m_basicQueryCb = std::make_shared<QueryCbFunction>();
    m_basicBatchQueryCb = std::make_shared<BatchQueryCbFunction>();
    m_basicTokenCb = std::make_shared<TokenCbFunction>();
  }

  DialogCallback(QuallaCallBackType cbType) {
    if (cbType == QUALLA_CALLBACK_TYPE_TEXT) {
      m_basicQueryCb.reset(new QueryCbFunction);
      m_callBackType = QUALLA_CALLBACK_TYPE_TEXT;
    } else if (cbType == QUALLA_CALLBACK_TYPE_TOKEN) {
      m_basicTokenCb.reset(new TokenCbFunction);
      m_callBackType = QUALLA_CALLBACK_TYPE_TOKEN;
    } else if (cbType == QUALLA_CALLBACK_TYPE_BATCH_TEXT) {
      m_basicBatchQueryCb.reset(new BatchQueryCbFunction);
      m_callBackType = QUALLA_CALLBACK_TYPE_BATCH_TEXT;
    }
  }

  bool callBack(const int32_t* tokens,
                const uint32_t sizeOfTokens,
                Sentence::Code scode,
                Tokenizer& _tokenizer) {
    if (m_callBackType == QUALLA_CALLBACK_TYPE_TEXT) {
      if (tokens) {
        std::vector<int32_t> tokenVec(tokens, tokens + sizeOfTokens);
        std::string outString = _tokenizer.decode(tokenVec);
        return (*m_basicQueryCb)(outString, scode);
      }
      return (*m_basicQueryCb)("", scode);
    } else if (m_callBackType == QUALLA_CALLBACK_TYPE_BATCH_TEXT) {
      if (tokens) {
        std::vector<std::string> responseStrings;
        if (scode == Sentence::BEGIN) {
            responseStrings.clear();
        }
        if (responseStrings.size() < sizeOfTokens) {
            responseStrings.resize(sizeOfTokens);
        }

        for (uint32_t i = 0; i < sizeOfTokens; ++i) {
          if (i < m_batchFinished.size() && m_batchFinished[i]) {
              responseStrings[i] = "";
              continue;
          }
          std::vector<int32_t> singleTokenVec = { tokens[i] };
          std::string batchText = _tokenizer.decode(singleTokenVec);
          responseStrings[i] = batchText;
        }

        if (m_basicBatchQueryCb && *m_basicBatchQueryCb) {
            return (*m_basicBatchQueryCb)(responseStrings, m_batchFinished, scode);
        }
        return true;
      }

      if (m_basicBatchQueryCb && *m_basicBatchQueryCb) {
          std::vector<std::string> emptyStrings;
          return (*m_basicBatchQueryCb)(emptyStrings, m_batchFinished, scode);

      }
      return true;
    } else if (m_callBackType == QUALLA_CALLBACK_TYPE_TOKEN) {
      return (*m_basicTokenCb)(tokens, sizeOfTokens, scode);
    } else {
      return false;
    }
  }

  void setCallBackType(QuallaCallBackType CbType) { m_callBackType = CbType; }

  QuallaCallBackType getCallBackType() { return m_callBackType; }

  std::shared_ptr<QueryCbFunction> getQueryCbFunc() { return m_basicQueryCb; }

  std::shared_ptr<BatchQueryCbFunction> getBatchQueryCbFunc() { return m_basicBatchQueryCb; }

  std::shared_ptr<TokenCbFunction> getTokenCbFunc() { return m_basicTokenCb; }

  void setBatchFinished(const std::vector<bool>& batchFinished) { m_batchFinished = batchFinished; }


 private:
  QuallaCallBackType m_callBackType;
  std::shared_ptr<TokenCbFunction> m_basicTokenCb;
  std::shared_ptr<QueryCbFunction> m_basicQueryCb;
  std::shared_ptr<BatchQueryCbFunction> m_basicBatchQueryCb;
  std::vector<bool> m_batchFinished;
};

}  // namespace qualla
