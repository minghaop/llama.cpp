//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <iostream>

#include "nlohmann/json.hpp"
#include "Exception.hpp"
#include "GenieSampler.h"
#include "Macro.hpp"
#include "Sampler.hpp"
#include "Util/HandleManager.hpp"

using namespace genie;
GENIE_API
Genie_Status_t GenieSamplerConfig_createFromJson(const char* str,
                                                 GenieSamplerConfig_Handle_t* configHandle) {
  try {
    GENIE_ENSURE(str, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto config = std::make_shared<Sampler::SamplerConfig>(str);
    GENIE_ENSURE(config, GENIE_STATUS_ERROR_MEM_ALLOC);
    *configHandle = Sampler::SamplerConfig::add(config);
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
Genie_Status_t GenieSamplerConfig_setParam(const GenieSamplerConfig_Handle_t configHandle,
                                           const char* keyStr,
                                           const char* valueStr) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto samplerConfig = Sampler::SamplerConfig::get(configHandle);
    GENIE_ENSURE(samplerConfig, GENIE_STATUS_ERROR_INVALID_HANDLE);
    samplerConfig->setParam(keyStr, valueStr);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_SET_PARAMS_FAILED;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieSamplerConfig_free(const GenieSamplerConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    {
      // Check if the sampler actually exists
      auto configObj = Sampler::SamplerConfig::get(configHandle);
      GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    }
    Sampler::SamplerConfig::remove(configHandle);
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieSampler_create(const GenieSamplerConfig_Handle_t configHandle,
                                   GenieSampler_Handle_t* samplerHandle) {
  try {
    GENIE_ENSURE(samplerHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    // Get config object
    auto configObj = Sampler::SamplerConfig::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    // Create sampler
    auto sampler = std::make_shared<Sampler>(configObj);
    GENIE_ENSURE(sampler, GENIE_STATUS_ERROR_MEM_ALLOC);

    // Create Handle
    *samplerHandle = Sampler::add(sampler);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  // Return SUCCESS
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieSampler_sampleData(const GenieSampler_Handle_t samplerHandle,
                                       const void* data,
                                       const size_t dataSize,
                                       const char* dataConfig,
                                       GenieSampler_Callback_t callback,
                                       const void* userData) {
  try {
    GENIE_ENSURE(samplerHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(data, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(dataSize > 0, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(dataConfig, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(callback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    auto sampler = Sampler::get(samplerHandle);
    GENIE_ENSURE(sampler, GENIE_STATUS_ERROR_INVALID_HANDLE);

    sampler->sampleData(data, dataSize, dataConfig, callback, userData);
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
Genie_Status_t GenieSampler_free(const GenieSampler_Handle_t samplerHandle) {
  try {
    GENIE_ENSURE(samplerHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    Sampler::remove(samplerHandle);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieSampler_applyConfig(const GenieSampler_Handle_t samplerHandle,
                                        const GenieSamplerConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(samplerHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);

    auto sampler = Sampler::get(samplerHandle);
    GENIE_ENSURE(sampler, GENIE_STATUS_ERROR_INVALID_HANDLE);

    auto samplerConfig = Sampler::SamplerConfig::get(configHandle);
    GENIE_ENSURE(samplerConfig, GENIE_STATUS_ERROR_INVALID_HANDLE);

    sampler->applyConfig(samplerConfig->getSamplerJson());

  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_APPLY_CONFIG_FAILED;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieSampler_registerCallback(const char* name,
                                             GenieSampler_ProcessCallback_t samplerCallback) {
  try {
    GENIE_ENSURE(samplerCallback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    Sampler::registerCallback(name, samplerCallback);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieSampler_registerUserDataCallback(
    const char* name, GenieSampler_UserDataCallback_t samplerCallback, const void* userData) {
  try {
    GENIE_ENSURE(samplerCallback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    Sampler::registerUserDataCallback(name, samplerCallback, userData);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}
