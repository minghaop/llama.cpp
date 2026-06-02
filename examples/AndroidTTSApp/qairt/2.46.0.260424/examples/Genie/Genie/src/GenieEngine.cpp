//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <iostream>

#include "nlohmann/json.hpp"
#include "Engine.hpp"
#include "Exception.hpp"
#include "GenieEngine.h"
#include "Macro.hpp"
#include "Util/HandleManager.hpp"

using namespace genie;

GENIE_API
Genie_Status_t GenieEngineConfig_createFromJson(const char* str,
                                                GenieEngineConfig_Handle_t* configHandle) {
  try {
    GENIE_ENSURE(str, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto config = std::make_shared<Engine::EngineConfig>(str);
    GENIE_ENSURE(config, GENIE_STATUS_ERROR_MEM_ALLOC);
    *configHandle = Engine::EngineConfig::add(config);
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
Genie_Status_t GenieEngineConfig_bindProfiler(const GenieEngineConfig_Handle_t configHandle,
                                              const GenieProfile_Handle_t profileHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(profileHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto configObj = genie::Engine::EngineConfig::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto profiler = Profiler::get(profileHandle);
    GENIE_ENSURE(profiler, GENIE_STATUS_ERROR_INVALID_HANDLE);
    configObj->bindProfiler(profiler);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieEngineConfig_bindLogger(const GenieEngineConfig_Handle_t configHandle,
                                            const GenieLog_Handle_t logHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(logHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto configObj = genie::Engine::EngineConfig::get(configHandle);
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
Genie_Status_t GenieEngineConfig_free(const GenieEngineConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    {
      // Check if the engine config actually exists
      auto configObj = Engine::EngineConfig::get(configHandle);
      GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
      configObj->unbindProfiler();
      configObj->unbindLogger();
    }
    Engine::EngineConfig::remove(configHandle);
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieEngine_create(const GenieEngineConfig_Handle_t configHandle,
                                  GenieEngine_Handle_t* engineHandle) {
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    std::shared_ptr<ProfileStat> profileStat;
    GENIE_ENSURE(engineHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    // Get config object
    auto configObj = Engine::EngineConfig::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    if (!configObj->getProfiler().empty())
      profileStat = std::make_shared<ProfileStat>(
          GENIE_PROFILE_EVENTTYPE_ENGINE_CREATE, startTime, "", GENIE_PROFILE_COMPONENTTYPE_ENGINE);

    std::shared_ptr<genie::log::Logger> logger;
    if (!configObj->getLogger().empty())
      logger = *configObj->getLogger().begin();  // currently uses one logger with one engine
    else
      logger = nullptr;
    // Create engine
    auto engine = std::make_shared<genie::Engine>(configObj, profileStat, logger);
    GENIE_ENSURE(engine, GENIE_STATUS_ERROR_MEM_ALLOC);
    // Create Handle
    *engineHandle = genie::Engine::add(engine);

    engine->bindProfiler(configObj->getProfiler());

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) {
      profileStat->setComponentId(engine->getName().c_str());
      profileStat->setDuration(stopTime - startTime);
    }
    const std::unordered_set<std::shared_ptr<Profiler>> profiler = engine->getProfiler();
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  // Return SUCCESS
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieEngine_free(const GenieEngine_Handle_t engineHandle) {
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    GENIE_ENSURE(engineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    std::shared_ptr<ProfileStat> profileStat;
    std::unordered_set<std::shared_ptr<Profiler>> profiler;
    {
      // Check if the engine actually exists
      auto engine = genie::Engine::get(engineHandle);
      GENIE_ENSURE(engine, GENIE_STATUS_ERROR_INVALID_HANDLE);
      profiler = engine->getProfiler();
      if (!profiler.empty())
        profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_ENGINE_FREE,
                                                    startTime,
                                                    engine->getName(),
                                                    GENIE_PROFILE_COMPONENTTYPE_ENGINE);
      engine->unbindProfiler();
    }
    genie::Engine::remove(engineHandle);
    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) {
      profileStat->setDuration(stopTime - startTime);
    }
    for (auto it : profiler) {
      it->addProfileStat(profileStat);
    }
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}