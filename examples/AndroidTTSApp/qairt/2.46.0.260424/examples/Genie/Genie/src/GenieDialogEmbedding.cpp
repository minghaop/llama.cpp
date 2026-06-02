//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "nlohmann/json.hpp"
#include "Dialog.hpp"
#include "Exception.hpp"
#include "GenieDialog.h"
#include "Macro.hpp"
#include "Util/HandleManager.hpp"

using namespace genie;

GENIE_API
Genie_Status_t GenieDialog_embeddingQuery(const GenieDialog_Handle_t dialogHandle,
                                          const void* embeddings,
                                          const uint32_t embeddingsSize,
                                          const GenieDialog_SentenceCode_t sentenceCode,
                                          const GenieDialog_TokenToEmbeddingCallback_t t2eCallback,
                                          const GenieDialog_QueryCallback_t callback,
                                          const void* userData) {
  Genie_Status_t status;
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    std::shared_ptr<ProfileStat> profileStat;
    const std::unordered_set<std::shared_ptr<Profiler>> profiler = dialog->getProfiler();
    if (!profiler.empty())
      profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY,
                                                  startTime,
                                                  dialog->getName(),
                                                  GENIE_PROFILE_COMPONENTTYPE_DIALOG);

    if (sentenceCode != GENIE_DIALOG_SENTENCE_RESUME) {
      GENIE_ENSURE(embeddings, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    } else {
      if (embeddings != nullptr && embeddingsSize != 0) {
        return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
      }
    }
    GENIE_ENSURE(callback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    switch (sentenceCode) {
      case GENIE_DIALOG_SENTENCE_COMPLETE:
      case GENIE_DIALOG_SENTENCE_BEGIN:
      case GENIE_DIALOG_SENTENCE_CONTINUE:
      case GENIE_DIALOG_SENTENCE_END:
      case GENIE_DIALOG_SENTENCE_ABORT:
      case GENIE_DIALOG_SENTENCE_REWIND:
      case GENIE_DIALOG_SENTENCE_RESUME:
        // Do nothing
        break;
      default:
        return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
    }

    status = dialog->embeddingQuery(
        embeddings, embeddingsSize, sentenceCode, t2eCallback, callback, userData, profileStat);

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) profileStat->setDuration(stopTime - startTime);
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const ContextLimitException& e) {
    callback("", GENIE_DIALOG_SENTENCE_END, userData);
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const Exception& e) {
    callback("", GENIE_DIALOG_SENTENCE_ABORT, userData);
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const std::exception& e) {
    callback("", GENIE_DIALOG_SENTENCE_ABORT, userData);
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return status;
}

GENIE_API
Genie_Status_t GenieDialog_embeddingTokenQuery(
    const GenieDialog_Handle_t dialogHandle,
    const void* embeddings,
    const uint32_t embeddingsSize,
    const GenieDialog_SentenceCode_t sentenceCode,
    const GenieDialog_TokenToEmbeddingCallback_t t2eCallback,
    const GenieDialog_TokenQueryCallback_t callback,
    const void* userData) {
  Genie_Status_t status;
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    std::shared_ptr<ProfileStat> profileStat;
    const std::unordered_set<std::shared_ptr<Profiler>> profiler = dialog->getProfiler();
    if (!profiler.empty())
      profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY,
                                                  startTime,
                                                  dialog->getName(),
                                                  GENIE_PROFILE_COMPONENTTYPE_DIALOG);

    if (sentenceCode != GENIE_DIALOG_SENTENCE_RESUME) {
      GENIE_ENSURE(embeddings, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    } else {
      if (embeddings != nullptr && embeddingsSize != 0) {
        return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
      }
    }
    GENIE_ENSURE(callback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    switch (sentenceCode) {
      case GENIE_DIALOG_SENTENCE_COMPLETE:
      case GENIE_DIALOG_SENTENCE_BEGIN:
      case GENIE_DIALOG_SENTENCE_CONTINUE:
      case GENIE_DIALOG_SENTENCE_END:
      case GENIE_DIALOG_SENTENCE_ABORT:
      case GENIE_DIALOG_SENTENCE_REWIND:
      case GENIE_DIALOG_SENTENCE_RESUME:
        // Do nothing
        break;
      default:
        return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
    }

    status = dialog->embeddingQuery(
        embeddings, embeddingsSize, sentenceCode, t2eCallback, callback, userData, profileStat);

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) profileStat->setDuration(stopTime - startTime);
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const ContextLimitException& e) {
    callback(nullptr, 0, GENIE_DIALOG_SENTENCE_END, userData);
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const Exception& e) {
    callback(nullptr, 0, GENIE_DIALOG_SENTENCE_ABORT, userData);
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const std::exception& e) {
    callback(nullptr, 0, GENIE_DIALOG_SENTENCE_ABORT, userData);
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return status;
}
