//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <iostream>

#include "Dlc.hpp"
#include "Exception.hpp"
#include "GenieDlc.h"
#include "Macro.hpp"
#include "Util/HandleManager.hpp"
#include "nlohmann/json.hpp"

using namespace genie;

GENIE_API
Genie_Status_t GenieDlcConfig_create(const char* dlcSource,
                                     const char* /*jsonStr*/,
                                     GenieDlcConfig_Handle_t* configHandle) {
  try {
    GENIE_ENSURE(dlcSource, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto config = std::make_shared<Dlc::Config>(dlcSource);
    GENIE_ENSURE(config, GENIE_STATUS_ERROR_MEM_ALLOC);
    *configHandle = genie::Dlc::Config::add(config);
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
Genie_Status_t GenieDlcConfig_free(const GenieDlcConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    // Check if the DLC config actually exists
    auto configObj = genie::Dlc::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    genie::Dlc::Config::remove(configHandle);
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDlc_create(const GenieDlcConfig_Handle_t configHandle,
                               GenieDlc_Handle_t* dlcHandle) {
  try {
    GENIE_ENSURE(dlcHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    auto configObj = genie::Dlc::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);

    auto dlc = std::make_shared<genie::Dlc>(configObj);
    GENIE_ENSURE(dlc, GENIE_STATUS_ERROR_MEM_ALLOC);
    *dlcHandle = genie::Dlc::add(dlc);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  // Return SUCCESS
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieDlc_getUseCases(const GenieDlc_Handle_t dlcHandle,
                                    Genie_AllocCallback_t callback,
                                    const char** useCases) {
  try {
    GENIE_ENSURE(dlcHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(callback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(useCases, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto dlc = genie::Dlc::get(dlcHandle);
    GENIE_ENSURE(dlc, GENIE_STATUS_ERROR_INVALID_HANDLE);
    const uint32_t jsonSize = dlc->serializeUseCases();
    callback(jsonSize, useCases);
    dlc->getUseCases(useCases);
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
Genie_Status_t GenieDlc_free(const GenieDlc_Handle_t dlcHandle) {
  try {
    GENIE_ENSURE(dlcHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    // Check if the DLC actually exists
    auto dlc = genie::Dlc::get(dlcHandle);
    GENIE_ENSURE(dlc, GENIE_STATUS_ERROR_INVALID_HANDLE);
    genie::Dlc::remove(dlcHandle);
  } catch (const std::exception&) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}