//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "GenieDlc.h"
#include "ResourceManager.hpp"
#include "System/QnnSystemInterface.h"
#include "Util/HandleManager.hpp"
#include "nlohmann/json.hpp"

// DLC Version values
#define DLC_VERSION_MAJOR 0
#define DLC_VERSION_MINOR 1
#define DLC_VERSION_PATCH 0

namespace genie {

class Dlc final {
 public:
  /**
   * @brief Structure representing a DLC use case.
   */
  struct UseCase {
    /**
     * @brief An enum which defines the use case configuration types.
     */
    enum class Type : int32_t {
      Dialog    = 0,
      Embedding = 10,
      Node      = 20,
      Pipeline  = 30,
      // Unused, present to ensure 32 bits if not using int32_t base.
      Undefined = 0x7FFFFFFF
    };

    // Use case name
    const char* name;
    // Type of the config
    Type type;
    // DLC record name of the config
    const char* recordName;
  };

  class Config {
   public:
    static GenieDlcConfig_Handle_t add(std::shared_ptr<Config> config);
    static std::shared_ptr<Config> get(GenieDlcConfig_Handle_t handle);
    static void remove(GenieDlcConfig_Handle_t handle);
    Config(const char* dlcPath);
    const std::string& getDlcPath() const;

   private:
    static qnn::util::HandleManager<Config>& getManager();
    std::string m_dlcPath;
  };

  Dlc(const std::shared_ptr<Config>& config);
  ~Dlc() = default;

  Dlc(const Dlc&)            = delete;
  Dlc& operator=(const Dlc&) = delete;
  Dlc(Dlc&&)                 = delete;
  Dlc& operator=(Dlc&&)      = delete;

  static GenieDlc_Handle_t add(std::shared_ptr<Dlc> dlc);
  static std::shared_ptr<Dlc> get(GenieDlc_Handle_t handle);
  static void remove(GenieDlc_Handle_t handle);

  void getGenieMetadataBuffer(std::shared_ptr<const char[]>& recordBuffer,
                              uint64_t* recordBufferSize);
  void getRecordBuffer(const std::string& recordName,
                       std::shared_ptr<const uint8_t[]>& recordBuffer,
                       uint64_t* recordBufferSize);
  uint32_t serializeUseCases();
  void getUseCases(const char** useCases);
  const std::string& getPath() const;

  std::shared_ptr<genie::ResourceManager> getResourceManager();

 private:
  static qnn::util::HandleManager<Dlc>& getManager();
  mutable std::mutex m_serializeMutex;
  nlohmann::ordered_json m_jsonUseCases;
  std::string m_useCases;
  std::string m_path;
  mutable std::mutex m_dlcMutex;
  typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
      const QnnSystemInterface_t*** providerList, uint32_t* numProviders);
  std::shared_ptr<ResourceManager> m_resourceManager;
};

}  // namespace genie