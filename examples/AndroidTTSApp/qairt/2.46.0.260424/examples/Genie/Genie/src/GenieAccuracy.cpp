//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "Accuracy.hpp"
#include "Dialog.hpp"
#include "Exception.hpp"
#include "GenieAccuracy.h"
#include "Macro.hpp"
#include "pipeline/Node.hpp"
#include "pipeline/TextGenerator.hpp"

using namespace genie;

#ifdef __cplusplus
extern "C" {
#endif

GENIE_API
Genie_Status_t GenieAccuracyConfig_createFromJson(const char* str,
                                                  GenieAccuracyConfig_Handle_t* configHandle) {
  try {
    GENIE_ENSURE(str, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto config = std::make_shared<Accuracy::Config>(str);
    GENIE_ENSURE(config, GENIE_STATUS_ERROR_MEM_ALLOC);
    *configHandle = genie::Accuracy::Config::add(config);
  } catch (const nlohmann::json::parse_error& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_JSON_FORMAT;
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieAccuracyConfig_free(const GenieAccuracyConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    // Check if the dialog actually exists
    auto configObj = genie::Accuracy::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    genie::Accuracy::Config::remove(configHandle);
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieAccuracy_createFromDialog(const GenieAccuracyConfig_Handle_t configHandle,
                                              const GenieDialog_Handle_t dialogHandle,
                                              GenieAccuracy_Handle_t* accuracyHandle) {
  try {
    // Config Handle must be NULL per the current API definition.
    auto configObj = genie::Accuracy::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(dialogHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(accuracyHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);

    // validate Dialog object
    auto dialog = genie::Dialog::get(dialogHandle);
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    // Make accuracy object and bind dialog handle to it
    auto accuracyObj = std::make_shared<genie::Accuracy>(configObj);
    GENIE_ENSURE(accuracyObj, GENIE_STATUS_ERROR_MEM_ALLOC);
    *accuracyHandle = Accuracy::add(accuracyObj);
    accuracyObj->bindDialog(dialog);
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieAccuracy_createFromNode(const GenieAccuracyConfig_Handle_t configHandle,
                                            const GenieNode_Handle_t nodeHandle,
                                            GenieAccuracy_Handle_t* accuracyHandle) {
  try {
    // Config Handle must be NULL per the current API definition.
    auto configObj = genie::Accuracy::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(accuracyHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);

    auto nodePtr =
        std::dynamic_pointer_cast<genie::pipeline::TextGenerator>(pipeline::Node::get(nodeHandle));
    std::shared_ptr<Dialog> dialog;
    if (nodePtr == nullptr) {
      return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
    } else {
      dialog = nodePtr->m_generator;
    }
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    // Make accuracy object and bind dialog handle to it
    auto accuracyObj = std::make_shared<genie::Accuracy>(configObj);
    GENIE_ENSURE(accuracyObj, GENIE_STATUS_ERROR_MEM_ALLOC);
    *accuracyHandle = Accuracy::add(accuracyObj);
    accuracyObj->bindDialog(dialog);
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieAccuracy_compute(const GenieAccuracy_Handle_t accuracyHandle,
                                     Genie_AllocCallback_t callback,
                                     const char** jsonData) {
  try {
    GENIE_ENSURE(accuracyHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(callback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(jsonData, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto accuracyObj = genie::Accuracy::get(accuracyHandle);
    auto dialog      = accuracyObj->m_dialog;
    GENIE_ENSURE(dialog, GENIE_STATUS_ERROR_INVALID_HANDLE);

    Genie_Status_t status = accuracyObj->computeFromText();
    if (GENIE_STATUS_SUCCESS != status) {
      return status;
    }

    const uint32_t jsonSize = accuracyObj->serialize();
    callback(jsonSize, jsonData);
    accuracyObj->getJsonData(jsonData);
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieAccuracy_free(const GenieAccuracy_Handle_t accuracyHandle) {
  try {
    GENIE_ENSURE(accuracyHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto accuracyObj = genie::Accuracy::get(accuracyHandle);
    GENIE_ENSURE(accuracyObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    accuracyObj->unbindDialog();
    genie::Accuracy::remove(accuracyHandle);
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  }
  return GENIE_STATUS_SUCCESS;
}
#ifdef __cplusplus
}
#endif