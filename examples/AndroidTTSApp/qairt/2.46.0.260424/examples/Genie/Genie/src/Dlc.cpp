//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>

#include "Dlc.hpp"
#include "Exception.hpp"
#include "Macro.hpp"
#include "PAL/DynamicLoading.hpp"
#include "ResourceManager.hpp"
#include "Util.hpp"

using namespace genie;

//=============================================================================
// Dlc::Config functions
//=============================================================================

qnn::util::HandleManager<Dlc::Config>& Dlc::Config::getManager() {
  static qnn::util::HandleManager<Dlc::Config> s_manager;
  return s_manager;
}

GenieDlcConfig_Handle_t Dlc::Config::add(std::shared_ptr<Dlc::Config> config) {
  return reinterpret_cast<GenieDlcConfig_Handle_t>(getManager().add(config));
}

std::shared_ptr<Dlc::Config> Dlc::Config::get(GenieDlcConfig_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Dlc::Config::remove(GenieDlcConfig_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

Dlc::Config::Config(const char* dlcPath) : m_dlcPath(dlcPath) {
  if (!dlcPath) {
    throw Exception(GENIE_STATUS_ERROR_INVALID_ARGUMENT, "DLC path cannot be null");
  }

  std::ifstream file(m_dlcPath);
  if (!file.is_open()) {
    throw Exception(GENIE_STATUS_ERROR_INVALID_ARGUMENT, "Cannot open DLC file: " + m_dlcPath);
  }
  file.close();
}

const std::string& Dlc::Config::getDlcPath() const { return m_dlcPath; }

//=============================================================================
// Dlc functions
//=============================================================================

qnn::util::HandleManager<Dlc>& Dlc::getManager() {
  static qnn::util::HandleManager<Dlc> s_manager;
  return s_manager;
}

GenieDlc_Handle_t Dlc::add(std::shared_ptr<Dlc> dlc) {
  return reinterpret_cast<GenieDlc_Handle_t>(getManager().add(dlc));
}

std::shared_ptr<Dlc> Dlc::get(GenieDlc_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Dlc::remove(GenieDlc_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

Dlc::Dlc(const std::shared_ptr<Config>& config) {
  if (!config) {
    throw Exception(GENIE_STATUS_ERROR_INVALID_ARGUMENT, "DLC config cannot be null");
  } else {
    m_path = config->getDlcPath();
  }
  auto resourceManager = getResourceManager();
  if (!resourceManager) {
    throw Exception(GENIE_STATUS_ERROR_MEM_ALLOC, "Resource manager allocation failed");
  }
}

void Dlc::getRecordBuffer(const std::string& recordName,
                          std::shared_ptr<const uint8_t[]>& recordBuffer,
                          uint64_t* recordBufferSize) {
  QnnSystemDlc_RecordHandle_t recordHandle = nullptr;
  auto qnnError = m_resourceManager->getQnnSystemInterface().systemDlcGetRecordByName(
      m_resourceManager->getDlcHandle(), recordName.c_str(), &recordHandle);
  if (QNN_SUCCESS != QNN_GET_ERROR_CODE(qnnError) || !recordHandle) {
    throw genie::Exception(GENIE_STATUS_ERROR_GENERAL,
                           "Issue with retrieving the handle for record " + recordName);
  }

  const uint8_t* recordBuf = nullptr;
  uint64_t bufferSize{0};
  qnnError = m_resourceManager->getQnnSystemInterface().systemDlcReadRecordDataMemoryMapped(
      recordHandle, &recordBuf, &bufferSize);
  if (QNN_SUCCESS != QNN_GET_ERROR_CODE(qnnError)) {
    throw genie::Exception(GENIE_STATUS_ERROR_GENERAL,
                           "Failed to read record data from record handle.");
  }

  std::shared_ptr<uint8_t[]> sanitizedBuf;
  *recordBufferSize = util::sanitizeJsonBuffer(recordBuf, bufferSize, sanitizedBuf);
  recordBuffer      = std::shared_ptr<const uint8_t[]>(sanitizedBuf, sanitizedBuf.get());

  if (QNN_SUCCESS != m_resourceManager->getQnnSystemInterface().systemDlcFreeRecord(recordHandle)) {
    *recordBufferSize = 0u;
    throw genie::Exception(GENIE_STATUS_ERROR_GENERAL,
                           "QnnSystemDlc_freeRecord failed for record " + recordName);
  }
}

void Dlc::getGenieMetadataBuffer(std::shared_ptr<const char[]>& recordBuffer,
                                 uint64_t* recordBufferSize) {
  uint32_t numMetadataRecord{0};
  QnnSystemDlc_RecordHandle_t* metadataRecordHandle = nullptr;
  auto qnnError = m_resourceManager->getQnnSystemInterface().systemDlcGetRecordsByType(
      m_resourceManager->getDlcHandle(),
      QNN_SYSTEM_DLC_RECORD_TYPE_GENAI_METADATA,
      false,
      &metadataRecordHandle,
      &numMetadataRecord);
  if (QNN_SUCCESS != QNN_GET_ERROR_CODE(qnnError) || !metadataRecordHandle) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL,
                    "Failed to retrieve GenAI metadata for dlc " + m_path);
  }

  if (numMetadataRecord == 0) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "No GenAI metadata found in this dlc");
  }

  const uint8_t* metadataRecordBuffer = nullptr;
  uint64_t metadataBufferSize{0};
  // There should only be one GenAI metadata record.
  qnnError = m_resourceManager->getQnnSystemInterface().systemDlcReadRecordDataMemoryMapped(
      metadataRecordHandle[0], &metadataRecordBuffer, &metadataBufferSize);
  if (QNN_SUCCESS != QNN_GET_ERROR_CODE(qnnError)) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL,
                    "Failed to read GenAI metadata record data from record handle.");
  }

  std::shared_ptr<uint8_t[]> sanitizedBuf;
  *recordBufferSize =
      util::sanitizeJsonBuffer(metadataRecordBuffer, metadataBufferSize, sanitizedBuf);
  recordBuffer = std::shared_ptr<const char[]>(sanitizedBuf,
                                               reinterpret_cast<const char*>(sanitizedBuf.get()));

  if (QNN_SUCCESS !=
      m_resourceManager->getQnnSystemInterface().systemDlcFreeRecord(metadataRecordHandle[0])) {
    *recordBufferSize = 0u;
    throw genie::Exception(GENIE_STATUS_ERROR_GENERAL,
                           "QnnSystemDlc_freeRecord failed for GenAI metadata JSON");
  }
}

uint32_t Dlc::serializeUseCases() {
  const std::unique_lock<std::mutex> lock(m_serializeMutex);

  std::shared_ptr<const char[]> metadataBuffer;
  uint64_t metadataSize{};
  getGenieMetadataBuffer(metadataBuffer, &metadataSize);

  nlohmann::json metadataJson;
  util::getJsonFromStr(metadataBuffer.get(), metadataJson);

  m_useCases = metadataJson["use-cases"].dump(2);
  return static_cast<uint32_t>(m_useCases.length() + 1);
}

void Dlc::getUseCases(const char** useCases) {
  const std::unique_lock<std::mutex> lock(m_serializeMutex);
  std::memcpy(static_cast<void*>(const_cast<char*>(*useCases)),
              static_cast<void*>(const_cast<char*>(m_useCases.c_str())),
              m_useCases.length());
  (const_cast<char*>(*useCases))[m_useCases.length()] = '\0';
}

const std::string& Dlc::getPath() const { return m_path; }

std::shared_ptr<ResourceManager> Dlc::getResourceManager() {
  std::lock_guard<std::mutex> lock(m_dlcMutex);
  if (!m_resourceManager) {
    m_resourceManager = std::make_shared<ResourceManager>(m_path);
  }
  return m_resourceManager;
}
