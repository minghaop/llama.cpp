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
Genie_Status_t GenieDialogConfig_createFromJson(const char* str,
                                                GenieDialogConfig_Handle_t* configHandle) {
  try {
    GENIE_ENSURE(str, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto config = std::make_shared<Dialog::Config>(str);
    GENIE_ENSURE(config, GENIE_STATUS_ERROR_MEM_ALLOC);
    *configHandle = genie::Dialog::Config::add(config);
  } catch (const nlohmann::json::parse_error& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_JSON_FORMAT;
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialogConfig_bindProfiler(const GenieDialogConfig_Handle_t configHandle,
                                              const GenieProfile_Handle_t profileHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(profileHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto configObj = genie::Dialog::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto profiler = Profiler::get(profileHandle);
    GENIE_ENSURE(profiler, GENIE_STATUS_ERROR_INVALID_HANDLE);
    configObj->bindProfiler(profiler);
  } catch (const nlohmann::json::parse_error& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_JSON_FORMAT;
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialogConfig_bindLogger(const GenieDialogConfig_Handle_t configHandle,
                                            const GenieLog_Handle_t logHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(logHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto configObj = genie::Dialog::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto logger = genie::log::Logger::getLogger(logHandle);
    GENIE_ENSURE(logger, GENIE_STATUS_ERROR_INVALID_HANDLE);
    configObj->bindLogger(logger);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialogConfig_createFromDlc(GenieDlc_Handle_t dlcHandle,
                                               const char* useCaseName,
                                               const char* configStr,
                                               GenieDialogConfig_Handle_t* configHandle) {
  try {
    GENIE_ENSURE(dlcHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(useCaseName, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    // configStr can be nullptr and will be check in the Dialog::Config ctor
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    auto dlc = genie::Dlc::get(dlcHandle);
    GENIE_ENSURE(dlc, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto config = std::make_shared<Dialog::Config>(dlc, useCaseName, configStr);
    GENIE_ENSURE(config, GENIE_STATUS_ERROR_MEM_ALLOC);
    *configHandle = genie::Dialog::Config::add(config);
  } catch (const nlohmann::json::parse_error& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_JSON_FORMAT;
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialogConfig_free(const GenieDialogConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    // Check if the dialog actually exists
    auto configObj = genie::Dialog::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    configObj->unbindProfiler();
    configObj->unbindLogger();
    genie::Dialog::Config::remove(configHandle);
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_create(const GenieDialogConfig_Handle_t configHandle,
                                  GenieDialog_Handle_t* dialogHandle) {
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    std::shared_ptr<ProfileStat> profileStat;
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    // Get config object
    auto configObj = genie::Dialog::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    if (!configObj->getProfiler().empty())
      profileStat = std::make_shared<ProfileStat>(
          GENIE_PROFILE_EVENTTYPE_DIALOG_CREATE, startTime, "", GENIE_PROFILE_COMPONENTTYPE_DIALOG);

    std::shared_ptr<genie::log::Logger> logger;
    if (!configObj->getLogger().empty())
      logger = *configObj->getLogger().begin();  // currently uses one logger with one dialog
    else
      logger = nullptr;
    // Create dialog
    auto dialog = std::make_shared<genie::Dialog>(configObj, profileStat, logger);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_MEM_ALLOC);

    // Create Handle
    *dialogHandle = genie::Dialog::add(dialog);

    dialog->bindProfiler(configObj->getProfiler());

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) {
      profileStat->setComponentId(dialog->getName().c_str());
      profileStat->setDuration(stopTime - startTime);
    }
    const std::unordered_set<std::shared_ptr<Profiler>> profiler = dialog->getProfiler();
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  // Return SUCCESS
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_query(const GenieDialog_Handle_t dialogHandle,
                                 const char* queryStr,
                                 const GenieDialog_SentenceCode_t sentenceCode,
                                 const GenieDialog_QueryCallback_t callback,
                                 const void* userData) {
  int32_t status;

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
      GENIE_ENSURE(queryStr, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    } else {
      if (queryStr != nullptr && queryStr[0] != '\0') {
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
    status = dialog->query(queryStr, sentenceCode, callback, userData, profileStat);

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
Genie_Status_t GenieDialog_batchQuery(const GenieDialog_Handle_t dialogHandle,
                                      const char** queryStrs,
                                      size_t queryCount,
                                      const GenieDialog_SentenceCode_t sentenceCode,
                                      const GenieDialog_QueryCallback_t callback,
                                      const void** userData) {
  int32_t status;
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
      GENIE_ENSURE(queryStrs, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
      GENIE_ENSURE(queryCount > 0, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    } else {
      if (queryStrs != nullptr && queryCount > 0) {
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
        break;
      default:
        return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
    }
    std::vector<const char*> queriesVec;
    if (queryStrs && queryCount > 0) {
      queriesVec.reserve(queryCount);
      for (size_t i = 0; i < queryCount; ++i) {
        if (queryStrs[i] == nullptr) {
            return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
        }
        queriesVec.push_back(queryStrs[i]);
      }
    }

    status = dialog->query(queriesVec, sentenceCode, callback, userData, profileStat);

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) profileStat->setDuration(stopTime - startTime);
    for (auto it : profiler) it->addProfileStat(profileStat);

  } catch (const ContextLimitException& e) {
     callback("", GENIE_DIALOG_SENTENCE_END, userData);
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const Exception& e) {
     callback("", GENIE_DIALOG_SENTENCE_END, userData);
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const std::exception& e) {
     callback("", GENIE_DIALOG_SENTENCE_END, userData);
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return status;
}

GENIE_API
Genie_Status_t GenieDialog_save(const GenieDialog_Handle_t dialogHandle, const char* path) {
  int32_t status;

  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(path, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    status = dialog->save(path);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return status;
}

GENIE_API
Genie_Status_t GenieDialog_restore(const GenieDialog_Handle_t dialogHandle, const char* path) {
  int32_t status;

  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(path, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    status = dialog->restore(path);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return status;
}

GENIE_API
Genie_Status_t GenieDialog_reset(const GenieDialog_Handle_t dialogHandle) {
  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    dialog->reset();
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_applyLora(const GenieDialog_Handle_t dialogHandle,
                                     const char* engine,
                                     const char* loraAdapterName) {
  int32_t status;
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(engine, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    std::string eng(engine);
    GENIE_ENSURE(loraAdapterName, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    std::string loraName(loraAdapterName);
    std::shared_ptr<ProfileStat> profileStat;
    const std::unordered_set<std::shared_ptr<Profiler>> profiler = dialog->getProfiler();
    if (!profiler.empty())
      profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_DIALOG_APPLY_LORA,
                                                  startTime,
                                                  dialog->getName(),
                                                  GENIE_PROFILE_COMPONENTTYPE_DIALOG);
    status = dialog->applyLora(loraName, eng, profileStat);

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) profileStat->setDuration(stopTime - startTime);
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return status;
}

GENIE_API
Genie_Status_t GenieDialog_setLoraStrength(const GenieDialog_Handle_t dialogHandle,
                                           const char* engine,
                                           const char* tensorName,
                                           const float alpha) {
  int32_t status;
  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(engine, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    std::string eng(engine);
    GENIE_ENSURE(tensorName, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    std::string alphaTensorName(tensorName);
    GENIE_ENSURE_NOT_EMPTY(alphaTensorName, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    status = dialog->applyLoraStrength(tensorName, eng, alpha);
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return status;
}

GENIE_API
Genie_Status_t GenieDialog_releaseLoraMemory(const GenieDialog_Handle_t dialogHandle,
                                             const char *engine,
                                             const char *loraAdapterName) {
    int32_t status;
    try {
      GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
      GENIE_ENSURE(engine, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
      GENIE_ENSURE(loraAdapterName, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
      auto dialog = genie::Dialog::get(dialogHandle);
      GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
      std::string eng(engine);
      std::string loraName(loraAdapterName);

      status = dialog->releaseLoraMemory(loraName, eng);
    } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      return GENIE_STATUS_ERROR_GENERAL;
    }

    return status;
}

GENIE_API
Genie_Status_t GenieDialog_tokenQuery(const GenieDialog_Handle_t dialogHandle,
                                      const uint32_t* inputTokens,
                                      const uint32_t numTokens,
                                      const GenieDialog_SentenceCode_t sentenceCode,
                                      const GenieDialog_TokenQueryCallback_t callback,
                                      const void* userData) {
  bool status;
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
      GENIE_ENSURE(inputTokens, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    } else {
      if (inputTokens != nullptr && numTokens != 0) {
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

    status =
        dialog->tokenQuery(inputTokens, numTokens, sentenceCode, callback, userData, profileStat);

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

GENIE_API
Genie_Status_t GenieDialog_getSampler(const GenieDialog_Handle_t dialogHandle,
                                      GenieSampler_Handle_t* dialogSamplerHandle) {
  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(dialogSamplerHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    *dialogSamplerHandle = genie::Dialog::getSamplerHandle(dialog);
    GENIE_ENSURE(*dialogSamplerHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GET_HANDLE_FAILED;
  }

  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_getTokenizer(const GenieDialog_Handle_t dialogHandle,
                                        GenieTokenizer_Handle_t* tokenizerHandle) {
  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(tokenizerHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    *tokenizerHandle = genie::Dialog::getTokenizerHandle(dialog);
    GENIE_ENSURE(*tokenizerHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GET_HANDLE_FAILED;
  }

  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_setStopSequence(const GenieDialog_Handle_t dialogHandle,
                                           const char* newStopSequences) {
  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    newStopSequences ? dialog->setStopSequence(newStopSequences) : dialog->setStopSequence("{}");
  } catch (const nlohmann::json::parse_error& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_JSON_FORMAT;
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_setPerformancePolicy(const GenieDialog_Handle_t dialogHandle,
                                                const Genie_PerformancePolicy_t perfProfile) {
  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    dialog->setPerformancePolicy(perfProfile);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_getPerformancePolicy(const GenieDialog_Handle_t dialogHandle,
                                                Genie_PerformancePolicy_t* perfProfile) {
  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(perfProfile, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    *perfProfile = dialog->getPerformancePolicy();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GET_HANDLE_FAILED;
  }

  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_getValue(const GenieDialog_Handle_t dialogHandle,
                                    const GenieDialog_Param_t key,
                                    const Genie_AllocCallback_t callback,
                                    Genie_DataType_t* dataType,
                                    Genie_Value_t* value) {
  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(dataType, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(value, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    switch (key) {
      case GENIE_DIALOG_PARAM_CONTEXT_OCCUPANCY:
        (void)callback;
        *dataType          = GENIE_DATATYPE_UINT_32;
        value->uint32Value = dialog->getContextOccupancy();
        break;
      case GENIE_DIALOG_PARAM_APPLIED_LORA_ADAPTER:
        GENIE_ENSURE(callback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
        {
          std::string adapterName = dialog->getAppliedLoraAdapter("primary");
          *dataType = GENIE_DATATYPE_STRING;
          size_t len = adapterName.length() + 1;
          const char* tempPtr = nullptr;
          callback(len, &tempPtr);
          value->stringValue = tempPtr;
          if (value->stringValue) {
            char* mutablePtr = const_cast<char*>(value->stringValue);
            std::strncpy(mutablePtr, adapterName.c_str(), len);
          }
        }
        break;
      default:
        throw Exception(GENIE_STATUS_ERROR_INVALID_ARGUMENT,
                        "Invalid parameter key in GenieDialog_getValue");
    }
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_free(const GenieDialog_Handle_t dialogHandle) {
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    std::shared_ptr<ProfileStat> profileStat;
    std::unordered_set<std::shared_ptr<Profiler>> profiler;
    {
      // Check if the dialog actually exists
      auto dialog = genie::Dialog::get(dialogHandle);
      GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
      profiler = dialog->getProfiler();
      if (!profiler.empty())
        profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_DIALOG_FREE,
                                                    startTime,
                                                    dialog->getName(),
                                                    GENIE_PROFILE_COMPONENTTYPE_DIALOG);
      dialog->unbindProfiler();
    }
    genie::Dialog::remove(dialogHandle);

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) profileStat->setDuration(stopTime - startTime);
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_signal(const GenieDialog_Handle_t dialogHandle,
                                  const GenieDialog_Action_t action) {
  int32_t status;
  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);

    switch (action) {
      case GENIE_DIALOG_ACTION_ABORT:
        // Do nothing
        break;
      case GENIE_DIALOG_ACTION_PAUSE:
        // Do nothing
        break;
      default:
        return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
    }

    status = dialog->signalAction(action);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return status;
}

GENIE_API
Genie_Status_t GenieDialog_setPriority(const GenieDialog_Handle_t dialogHandle,
                                       const char* engineRole,
                                       const GenieDialog_Priority_t priority) {
  int32_t status;
  try {
    switch (priority) {
      case GENIE_DIALOG_PRIORITY_LOW:
      case GENIE_DIALOG_PRIORITY_NORMAL:
      case GENIE_DIALOG_PRIORITY_NORMAL_HIGH:
      case GENIE_DIALOG_PRIORITY_HIGH:
        // Do nothing
        break;
      default:
        return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
    }
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(engineRole, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    status = dialog->setPriority(engineRole, priority);
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return status;
}

GENIE_API
Genie_Status_t GenieDialog_setOemKey(const GenieDialog_Handle_t dialogHandle, const char* oemKey) {
  int32_t status;
  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(oemKey, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    status = dialog->setOemkey(oemKey);
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return status;
}

GENIE_API
Genie_Status_t GenieDialog_getEngine(const GenieDialog_Handle_t dialogHandle,
                                     const char* engineRole,
                                     GenieEngine_Handle_t* dialogEngineHandle) {
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(dialogEngineHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    std::shared_ptr<ProfileStat> profileStat;
    const std::unordered_set<std::shared_ptr<Profiler>> profiler = dialog->getProfiler();
    if (!profiler.empty())
      profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_DIALOG_GETENGINE,
                                                  startTime,
                                                  dialog->getName(),
                                                  GENIE_PROFILE_COMPONENTTYPE_DIALOG);
    GENIE_ENSURE(engineRole, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    *dialogEngineHandle = dialog->getEngineHandle(engineRole, profileStat);
    GENIE_ENSURE(*dialogEngineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) profileStat->setDuration(stopTime - startTime);
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GET_HANDLE_FAILED;
  }

  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_bindEngine(const GenieDialog_Handle_t dialogHandle,
                                      const char* engineRole,
                                      const GenieEngine_Handle_t engineHandle) {
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(engineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto engine = genie::Engine::get(engineHandle);
    GENIE_ENSURE(engine, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(engineRole, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    std::shared_ptr<ProfileStat> profileStat;
    const std::unordered_set<std::shared_ptr<Profiler>> profiler = dialog->getProfiler();
    if (!profiler.empty())
      profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_DIALOG_BINDENGINE,
                                                  startTime,
                                                  dialog->getName(),
                                                  GENIE_PROFILE_COMPONENTTYPE_DIALOG);
    dialog->bindEngine(engineRole, engine, profileStat);
    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) profileStat->setDuration(stopTime - startTime);
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDialog_setMaxNumTokens(const GenieDialog_Handle_t dialogHandle,
                                           const uint32_t maxNumTokens) {
  try {
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);
    dialog->setMaxNumTokens(maxNumTokens);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}
