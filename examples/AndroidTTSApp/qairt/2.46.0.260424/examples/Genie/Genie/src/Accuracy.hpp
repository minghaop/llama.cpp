//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once

#include "Dialog.hpp"
#include "GenieAccuracy.h"
#include "Util/HandleManager.hpp"
#include "nlohmann/json.hpp"
#include "qualla/dialog.hpp"
#include "qualla/encoder.hpp"

namespace genie {

class Accuracy {
 public:
  class Config {
   public:
    static GenieAccuracyConfig_Handle_t add(std::shared_ptr<Config> config);
    static std::shared_ptr<Config> get(GenieAccuracyConfig_Handle_t handle);
    static void remove(GenieAccuracyConfig_Handle_t handle);
    Config(const char *configStr);
    nlohmann::json &getJson();

   private:
    static qnn::util::HandleManager<Config> &getManager();
    nlohmann::json m_config;
  };
  Accuracy(std::shared_ptr<Config> config);
  static GenieAccuracy_Handle_t add(std::shared_ptr<Accuracy> accuracyObj);
  static std::shared_ptr<Accuracy> get(GenieAccuracy_Handle_t handle);

  void bindDialog(std::shared_ptr<Dialog> &dialog);
  void unbindDialog();

  static void remove(GenieAccuracy_Handle_t handle);

  int32_t compute(const int32_t *queryTokens, uint32_t queryTokensSize);
  int32_t computeEmbeddings(const std::vector<int32_t> &tokens);
  int32_t computeFromText();

  uint32_t serialize();
  void getJsonData(const char **jsonData);

  std::shared_ptr<Dialog> m_dialog;

 private:
  static qnn::util::HandleManager<Accuracy> &getManager();
  nlohmann::ordered_json m_jsonData;
  std::string m_data;
  std::string m_dataset;
  float m_ppl;
  std::string m_engineRole          = "primary";
  bool m_configContainsConvertField = false;
};
}  // namespace genie
