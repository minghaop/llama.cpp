//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "qualla/dialog.hpp"

namespace qualla {

class KvShareDialog : public Dialog {
 public:
  static constexpr const char* TYPE = "kv-share";

  KvShareDialog(std::shared_ptr<Env> env, const std::string& name, const nlohmann::json& conf);

  virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;

  virtual bool process(std::vector<int32_t>& tokens, DialogCallback callback) override;

  virtual void reset() override;

  // file-io implementation
  bool convertKV(const std::filesystem::path& cache_dir, Engine& s_engine);

  size_t convertKV(Engine& p_engine, Engine& s_engine);

  virtual const char* getTraceNamespace() const override { return "Dialog::KV-Share"; };

  void completeInit() override;

 private:
  bool _enable_in_memory_kv_share;  // use buffer
};

}  // namespace qualla
