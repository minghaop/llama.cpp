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

class MultiStreamDialog : public Dialog {
 public:
  static constexpr const char* TYPE = "multistream";

  MultiStreamDialog(std::shared_ptr<Env> env, const std::string& name, const nlohmann::json& conf);

  virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;

  virtual bool process(std::vector<uint8_t>& embedding_vectors,
                       Dialog::T2ECallback t2eCallback,
                       Dialog::Callback callback) override;

  virtual bool process(std::vector<int32_t>& /*tokens*/, DialogCallback /*callback*/) override {
    return false;
  }

  virtual const char* getTraceNamespace() const override { return "Dialog::Multistream"; };

 protected:
  uint32_t _vocab{0};
  uint32_t _n_streams{0};
  uint32_t _prompt_len{0};
  float _p_threshold{0.0f};

 private:
  bool processFollowOnGeneration(std::vector<std::vector<int32_t>>& streams,
                                 Tensor& logits,
                                 Dialog::Callback callback);
};

}  // namespace qualla
